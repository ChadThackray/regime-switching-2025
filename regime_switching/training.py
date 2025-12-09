"""Training functions for regime-switching RL."""

from collections.abc import Callable

import torch

from .config import Config
from .network import ValueNetwork
from .policy import compute_policy, compute_policy_2regime
from .simulation import simulate_trajectory


def compute_martingale_loss(
    value_net: ValueNetwork,
    trajectory: dict[str, torch.Tensor],
    config: Config,
) -> torch.Tensor:
    """
    Compute the score-weighted martingale loss (Algorithm 1, eqs. 5.9-5.11).

    Delta_k is computed without tracking gradients, then multiplied by the current
    value estimate v(t_k, X_k, I_k) with gradients on. The loss drives
    E[Delta_k * v(t_k, X_k, I_k)] -> 0.

    Supports arbitrary state dimension and number of regimes.

    Args:
        value_net: Value function network
        trajectory: Dictionary from simulate_trajectory
        config: Problem configuration

    Returns:
        Scalar loss tensor
    """
    times = trajectory["times"]
    states = trajectory["states"]
    regimes = trajectory["regimes"]
    rewards = trajectory["rewards"]
    switch_costs = trajectory["switch_costs"]

    batch_size, num_steps = times.shape[0], times.shape[1] - 1
    dt = config.dt
    num_regimes = config.num_regimes

    # Compute Delta_k with no gradients (detach).
    with torch.no_grad():
        # Handle both 1D and multi-dim states
        if states.dim() == 2:
            # 1D state: (batch, K+1)
            t_flat = times.reshape(-1)
            x_flat = states.reshape(-1)
        else:
            # Multi-dim state: (batch, K+1, state_dim)
            t_flat = times.reshape(-1)
            x_flat = states.reshape(-1, states.shape[-1])

        i_flat = regimes.reshape(-1)

        all_values = value_net(t_flat, x_flat)
        v_flat = all_values.gather(1, i_flat.unsqueeze(1)).squeeze(1)
        v_detached = v_flat.reshape(batch_size, num_steps + 1)

        # Compute entropies at each step
        entropies = []
        for k in range(num_steps):
            t_k = times[:, k]
            if states.dim() == 2:
                x_k = states[:, k]
            else:
                x_k = states[:, k, :]
            i_k = regimes[:, k]

            # Use appropriate policy function based on num_regimes
            if num_regimes == 2:
                _, entropy = compute_policy_2regime(value_net, t_k, x_k, i_k, config)
            else:
                _, entropy = compute_policy(value_net, t_k, x_k, i_k, config)
            entropies.append(entropy)
        entropies = torch.stack(entropies, dim=1)

        delta_detached = (
            v_detached[:, 1:]
            - v_detached[:, :-1]
            + (rewards + config.temperature * entropies) * dt
            - switch_costs
        ).detach()

    # Compute v(t_k, X_k, I_k) with gradients.
    t_curr = times[:, :-1].reshape(-1)
    if states.dim() == 2:
        x_curr = states[:, :-1].reshape(-1)
    else:
        x_curr = states[:, :-1].reshape(-1, states.shape[-1])
    i_curr = regimes[:, :-1].reshape(-1)

    v_curr_all = value_net(t_curr, x_curr)
    v_curr = v_curr_all.gather(1, i_curr.unsqueeze(1)).squeeze(1)
    v_curr = v_curr.reshape(batch_size, num_steps)

    # Score-weighted orthogonality loss (no square).
    # Gradient descent on -E[v * Delta] replicates the ascent update in eq. (5.9).
    loss = -(v_curr * delta_detached).mean()

    return loss


def train(
    config: Config,
    terminal_reward: Callable[[torch.Tensor], torch.Tensor],
    dynamics_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ],
    running_reward_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    device: torch.device,
    init_state_fn: Callable[[int, torch.device], torch.Tensor] | None = None,
) -> tuple[ValueNetwork, list[float]]:
    """
    Train the value function using Algorithm 1.

    Args:
        config: Problem configuration
        terminal_reward: Function h(x) for terminal reward
        dynamics_fn: Function (t, x, i) -> (drift, diffusion)
        running_reward_fn: Function (t, x, i) -> reward
        device: Torch device
        init_state_fn: Optional function (batch_size, device) -> initial_state

    Returns:
        Trained value network and list of losses
    """
    value_net = ValueNetwork(
        terminal_reward,
        config.hidden_dim,
        config.T,
        config.state_dim,
        config.num_regimes,
    ).to(device)
    optimizer = torch.optim.Adam(value_net.parameters(), lr=config.learning_rate)

    losses = []

    for episode in range(config.num_episodes):
        # Simulate trajectories
        value_net.eval()
        with torch.no_grad():
            trajectory = simulate_trajectory(
                value_net, config, dynamics_fn, running_reward_fn, device, init_state_fn
            )

        # Compute loss and update
        value_net.train()
        trajectory = {k: v.detach() for k, v in trajectory.items()}

        optimizer.zero_grad()
        loss = compute_martingale_loss(value_net, trajectory, config)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1}/{config.num_episodes}, Loss: {loss.item():.6f}"
            )

    return value_net, losses
