"""Trajectory simulation for regime-switching RL."""

from collections.abc import Callable

import torch

from .config import Config
from .environment import RegimeSwitchingEnv
from .network import ValueNetwork
from .policy import compute_policy


def simulate_trajectory(
    value_net: ValueNetwork,
    config: Config,
    running_reward: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Simulate a batch of trajectories using current policy.

    Args:
        value_net: Value function network
        config: Problem configuration
        running_reward: Function f(x) that computes running reward
        device: Torch device

    Returns:
        Dictionary containing:
        - times: (batch, K+1) time points
        - states: (batch, K+1) state values
        - regimes: (batch, K+1) regime indices
        - rewards: (batch, K) running rewards
        - entropies: (batch, K) entropy values
        - switch_costs: (batch, K) switching costs incurred
    """
    env = RegimeSwitchingEnv(config, running_reward, device)
    batch_size = config.batch_size
    K = config.K

    # Initialize
    t = torch.zeros(batch_size, device=device)
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
        # Compute policy
        with torch.no_grad():
            pi_switch, entropy = compute_policy(value_net, t, x, i, config)

        # Take environment step
        next_x, next_i, reward, switched = env.step(t, x, i, pi_switch)

        # Compute switching cost incurred
        cost = torch.zeros(batch_size, device=device)
        switched_01 = switched & (i == 0)
        switched_10 = switched & (i == 1)
        cost = torch.where(switched_01, torch.tensor(config.g01, device=device), cost)
        cost = torch.where(switched_10, torch.tensor(config.g10, device=device), cost)

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
