"""Trajectory simulation for regime-switching RL."""

from collections.abc import Callable

import torch

from .config import Config
from .environment import RegimeSwitchingEnv, RegimeSwitchingEnv2Regime
from .network import ValueNetwork
from .policy import compute_policy, compute_policy_2regime


def simulate_trajectory(
    value_net: ValueNetwork,
    config: Config,
    dynamics_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ],
    running_reward_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    device: torch.device,
    init_state_fn: Callable[[int, torch.device], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Simulate a batch of trajectories using current policy.

    Args:
        value_net: Value function network
        config: Problem configuration
        dynamics_fn: Function (t, x, i) -> (drift, diffusion)
        running_reward_fn: Function (t, x, i) -> reward
        device: Torch device
        init_state_fn: Optional function (batch_size, device) -> initial_state
            If None, uses N(x0_mean, x0_std) for 1D state.

    Returns:
        Dictionary containing:
        - times: (batch, K+1) time points
        - states: (batch, K+1) or (batch, K+1, state_dim) state values
        - regimes: (batch, K+1) regime indices
        - rewards: (batch, K) running rewards
        - entropies: (batch, K) entropy values
        - switch_costs: (batch, K) switching costs incurred
    """
    batch_size = config.batch_size
    K = config.K
    num_regimes = config.num_regimes

    # Use optimized 2-regime path if applicable
    if num_regimes == 2:
        return _simulate_trajectory_2regime(
            value_net, config, dynamics_fn, running_reward_fn, device, init_state_fn
        )

    # General m-regime path
    env = RegimeSwitchingEnv(config, dynamics_fn, running_reward_fn, device)

    # Initialize time
    t = torch.zeros(batch_size, device=device)

    # Initialize state
    if init_state_fn is not None:
        x = init_state_fn(batch_size, device)
    else:
        # Default: 1D Gaussian
        x = torch.randn(batch_size, device=device) * config.x0_std + config.x0_mean

    # Initialize regime uniformly
    i = torch.randint(0, num_regimes, (batch_size,), device=device)

    # Storage
    times = [t.clone()]
    states = [x.clone()]
    regimes = [i.clone()]
    rewards = []
    entropies = []
    switch_costs = []

    for _ in range(K):
        # Compute policy (full intensity row)
        with torch.no_grad():
            pi_row, entropy = compute_policy(value_net, t, x, i, config)

        # Take environment step
        next_x, next_i, reward, cost = env.step(t, x, i, pi_row)

        # Store
        rewards.append(reward)
        entropies.append(entropy)
        switch_costs.append(cost)

        # Update state
        t = t + config.dt
        x = next_x
        i = next_i

        times.append(t.clone())
        states.append(x.clone())
        regimes.append(i.clone())

    # Stack results
    # For multi-dim states, states list has shape [(batch, state_dim), ...]
    # Need to handle both 1D and multi-dim cases
    if states[0].dim() == 1:
        states_tensor = torch.stack(states, dim=1)  # (batch, K+1)
    else:
        states_tensor = torch.stack(states, dim=1)  # (batch, K+1, state_dim)

    return {
        "times": torch.stack(times, dim=1),
        "states": states_tensor,
        "regimes": torch.stack(regimes, dim=1),
        "rewards": torch.stack(rewards, dim=1),
        "entropies": torch.stack(entropies, dim=1),
        "switch_costs": torch.stack(switch_costs, dim=1),
    }


def _simulate_trajectory_2regime(
    value_net: ValueNetwork,
    config: Config,
    dynamics_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ],
    running_reward_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    device: torch.device,
    init_state_fn: Callable[[int, torch.device], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Optimized trajectory simulation for 2-regime case.

    Uses scalar switching intensity for efficiency and backward compatibility.
    """
    env = RegimeSwitchingEnv2Regime(config, dynamics_fn, running_reward_fn, device)
    batch_size = config.batch_size
    K = config.K

    # Initialize
    t = torch.zeros(batch_size, device=device)

    if init_state_fn is not None:
        x = init_state_fn(batch_size, device)
    else:
        x = torch.randn(batch_size, device=device) * config.x0_std + config.x0_mean

    i = torch.randint(0, 2, (batch_size,), device=device)

    # Storage
    times = [t.clone()]
    states = [x.clone()]
    regimes = [i.clone()]
    rewards = []
    entropies = []
    switch_costs = []

    for _ in range(K):
        # Compute policy (scalar switching intensity)
        with torch.no_grad():
            pi_switch, entropy = compute_policy_2regime(value_net, t, x, i, config)

        # Take environment step
        next_x, next_i, reward, cost = env.step(t, x, i, pi_switch)

        # Store
        rewards.append(reward)
        entropies.append(entropy)
        switch_costs.append(cost)

        # Update state
        t = t + config.dt
        x = next_x
        i = next_i

        times.append(t.clone())
        states.append(x.clone())
        regimes.append(i.clone())

    return {
        "times": torch.stack(times, dim=1),
        "states": torch.stack(states, dim=1),
        "regimes": torch.stack(regimes, dim=1),
        "rewards": torch.stack(rewards, dim=1),
        "entropies": torch.stack(entropies, dim=1),
        "switch_costs": torch.stack(switch_costs, dim=1),
    }
