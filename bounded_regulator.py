"""
Continuous-time RL for Optimal Regime Switching: Bounded Regulator Problem

Implementation of the algorithm from arXiv:2512.04697v1 (Section 6.1)

Problem Setup:
- 2 regimes with drift μ₀ = -2, μ₁ = 2
- Volatility σ = 0.5
- Running reward f(x) = 2e^(-2x²) - 0.1
- Terminal reward h(x) = 2e^(-2x²)
- Switching costs g₀₁ = g₁₀ = 0.5
- Temperature λ = 0.2
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Config:
    """Problem and training configuration."""
    # Problem parameters
    mu: tuple[float, float] = (-2.0, 2.0)  # Drift for each regime
    sigma: float = 0.5  # Volatility
    g01: float = 0.5  # Switching cost 0 -> 1
    g10: float = 0.5  # Switching cost 1 -> 0
    temperature: float = 0.2  # Lambda (exploration parameter)

    # Time discretization
    T: float = 1.0  # Time horizon
    K: int = 100  # Number of time steps

    # Training parameters
    batch_size: int = 64  # Number of trajectories per episode
    num_episodes: int = 1000  # Total training episodes
    learning_rate: float = 1e-3
    hidden_dim: int = 128

    # Initial state distribution
    x0_mean: float = 0.0
    x0_std: float = 2.0

    @property
    def dt(self) -> float:
        return self.T / self.K

    def switching_cost(self, i: int, j: int) -> float:
        """Get switching cost from regime i to regime j."""
        if i == j:
            return 0.0
        elif i == 0 and j == 1:
            return self.g01
        else:  # i == 1 and j == 0
            return self.g10


def running_reward(x: torch.Tensor) -> torch.Tensor:
    """Running reward f(x) = 2e^(-2x²) - 0.1"""
    return 2.0 * torch.exp(-2.0 * x**2) - 0.1


def terminal_reward(x: torch.Tensor) -> torch.Tensor:
    """Terminal reward h(x) = 2e^(-2x²)"""
    return 2.0 * torch.exp(-2.0 * x**2)


class ValueNetwork(nn.Module):
    """
    Neural network to approximate the value function v(t, x, i).

    Uses a structural decomposition to enforce terminal condition:
    v(t, x, i) = h(x) + (T - t) * phi(t, x, i)

    This ensures v(T, x, i) = h(x) exactly.
    """

    def __init__(self, hidden_dim: int = 128, T: float = 1.0):
        super().__init__()
        self.T = T

        # Network outputs the "correction" phi for both regimes
        # Input: (t, x) -> 2 features
        # Output: (phi_0, phi_1) -> 2 values
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute value function for both regimes.

        v(t, x, i) = h(x) + (T - t) * phi(t, x, i)

        Args:
            t: Time tensor of shape (batch,)
            x: State tensor of shape (batch,)

        Returns:
            Values tensor of shape (batch, 2) - one column per regime
        """
        inputs = torch.stack([t, x], dim=-1)
        phi = self.net(inputs)  # (batch, 2)

        # Terminal reward (same for both regimes)
        h = terminal_reward(x).unsqueeze(-1).expand_as(phi)  # (batch, 2)

        # Time-to-maturity weighting ensures v(T, x, i) = h(x)
        time_factor = (self.T - t).unsqueeze(-1).expand_as(phi)  # (batch, 2)

        return h + time_factor * phi

    def get_value(self, t: torch.Tensor, x: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Get value for specific regime indices."""
        all_values = self.forward(t, x)  # (batch, 2)
        return all_values.gather(1, i.unsqueeze(1)).squeeze(1)


class RegimeSwitchingEnv:
    """
    Environment for simulating regime-switching diffusion.

    State dynamics: dX_t = μ_i dt + σ dW_t
    Regime transitions governed by CTMC with controlled intensity.
    """

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.mu = torch.tensor(config.mu, device=device)

    def step(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        i: torch.Tensor,
        pi_switch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate one time step.

        Args:
            t: Current time (batch,)
            x: Current state (batch,)
            i: Current regime (batch,) - values in {0, 1}
            pi_switch: Switching intensity to the OTHER regime (batch,)

        Returns:
            next_x: Next state (batch,)
            next_i: Next regime (batch,)
            reward: Running reward for this step (batch,)
            switched: Whether a switch occurred (batch,) - bool
        """
        batch_size = x.shape[0]
        dt = self.config.dt
        sigma = self.config.sigma

        # Get drift based on current regime
        drift = self.mu[i]  # (batch,)

        # Simulate SDE: X_{t+dt} = X_t + μ_i * dt + σ * dW
        dW = torch.randn(batch_size, device=self.device) * np.sqrt(dt)
        next_x = x + drift * dt + sigma * dW

        # Simulate regime switching via CTMC
        # Probability of switching in interval dt: 1 - exp(-π * dt) ≈ π * dt
        switch_prob = 1.0 - torch.exp(-pi_switch * dt)
        switch_random = torch.rand(batch_size, device=self.device)
        switched = switch_random < switch_prob

        # If switched, flip regime (0 -> 1 or 1 -> 0)
        next_i = torch.where(switched, 1 - i, i)

        # Running reward
        reward = running_reward(x)

        return next_x, next_i, reward, switched


def compute_policy(
    value_net: ValueNetwork,
    t: torch.Tensor,
    x: torch.Tensor,
    i: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute switching intensity from value function (eq 3.5 in paper).

    π*_ij(t,x) = exp((V_j(t,x) - g_ij - V_i(t,x)) / λ)

    Args:
        value_net: Value function network
        t: Time tensor (batch,)
        x: State tensor (batch,)
        i: Current regime indices (batch,)
        config: Problem configuration

    Returns:
        pi_switch: Switching intensity to the other regime (batch,)
        entropy: Entropy term R(π, i) for the objective (batch,)
    """
    values = value_net(t, x)  # (batch, 2)
    v_0 = values[:, 0]
    v_1 = values[:, 1]

    # Compute switching intensities
    # pi_01 = exp((v_1 - g_01 - v_0) / lambda)
    pi_01 = torch.exp((v_1 - config.g01 - v_0) / config.temperature)
    # pi_10 = exp((v_0 - g_10 - v_1) / lambda)
    pi_10 = torch.exp((v_0 - config.g10 - v_1) / config.temperature)

    # Select the appropriate switching intensity based on current regime
    pi_switch = torch.where(i == 0, pi_01, pi_10)

    # Clamp to prevent numerical blow-up in simulation
    pi_switch = pi_switch.clamp(min=1e-10, max=20.0)

    # Compute entropy term: R(π, i) = Σ_{j≠i} (π_ij - π_ij * log(π_ij))
    entropy = pi_switch - pi_switch * torch.log(pi_switch)

    return pi_switch, entropy


def simulate_trajectory(
    value_net: ValueNetwork,
    config: Config,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Simulate a batch of trajectories using current policy.
    """
    env = RegimeSwitchingEnv(config, device)
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

    for k in range(K):
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
        'times': torch.stack(times, dim=1),
        'states': torch.stack(states, dim=1),
        'regimes': torch.stack(regimes, dim=1),
        'rewards': torch.stack(rewards, dim=1),
        'entropies': torch.stack(entropies, dim=1),
        'switch_costs': torch.stack(switch_costs, dim=1),
    }


def compute_martingale_loss(
    value_net: ValueNetwork,
    trajectory: dict[str, torch.Tensor],
    config: Config,
) -> torch.Tensor:
    """
    Compute the score-weighted martingale loss (Algorithm 1, eqs. 5.9–5.11).

    Δ_k is computed without tracking gradients, then multiplied by the current
    value estimate v(t_k, X_k, I_k) with gradients on. The loss drives
    E[Δ_k * v(t_k, X_k, I_k)] → 0.
    """
    times = trajectory['times']
    states = trajectory['states']
    regimes = trajectory['regimes']
    rewards = trajectory['rewards']
    switch_costs = trajectory['switch_costs']

    batch_size, num_steps = times.shape[0], times.shape[1] - 1
    dt = config.dt

    # Compute Δ_k with no gradients (detach).
    with torch.no_grad():
        t_flat = times.reshape(-1)
        x_flat = states.reshape(-1)
        i_flat = regimes.reshape(-1)

        all_values = value_net(t_flat, x_flat)
        v_flat = all_values.gather(1, i_flat.unsqueeze(1)).squeeze(1)
        v_detached = v_flat.reshape(batch_size, num_steps + 1)

        entropies = []
        for k in range(num_steps):
            t_k = times[:, k]
            x_k = states[:, k]
            i_k = regimes[:, k]
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
    x_curr = states[:, :-1].reshape(-1)
    i_curr = regimes[:, :-1].reshape(-1)

    v_curr_all = value_net(t_curr, x_curr)
    v_curr = v_curr_all.gather(1, i_curr.unsqueeze(1)).squeeze(1)
    v_curr = v_curr.reshape(batch_size, num_steps)

    # Score-weighted orthogonality loss (no square).
    # Gradient descent on -E[v * Δ] replicates the ascent update in eq. (5.9).
    loss = -(v_curr * delta_detached).mean()

    return loss


def train(config: Config, device: torch.device) -> tuple[ValueNetwork, list[float]]:
    """Train the value function using Algorithm 1."""
    value_net = ValueNetwork(config.hidden_dim, config.T).to(device)
    optimizer = torch.optim.Adam(value_net.parameters(), lr=config.learning_rate)

    losses = []

    for episode in range(config.num_episodes):
        # Simulate trajectories
        value_net.eval()
        with torch.no_grad():
            trajectory = simulate_trajectory(value_net, config, device)

        # Compute loss and update
        value_net.train()
        trajectory = {k: v.detach() for k, v in trajectory.items()}

        optimizer.zero_grad()
        loss = compute_martingale_loss(value_net, trajectory, config)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{config.num_episodes}, Loss: {loss.item():.6f}")

    return value_net, losses


def plot_results(
    value_net: ValueNetwork,
    losses: list[float],
    config: Config,
    device: torch.device,
):
    """Generate visualization plots matching Figure 1 in the paper."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training loss
    ax1 = axes[0]
    episodes = np.arange(1, len(losses) + 1)
    ax1.semilogy(episodes, losses, alpha=0.3, label='Raw Loss')

    window = 50
    if len(losses) >= window:
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax1.semilogy(episodes[window-1:], moving_avg, label='Moving Avg', color='orange')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Value functions and switching probabilities at t=0.5
    ax2 = axes[1]

    value_net.eval()
    x_range = torch.linspace(-2.5, 2.5, 200, device=device)
    t_fixed = torch.full_like(x_range, 0.5)

    with torch.no_grad():
        values = value_net(t_fixed, x_range)
        v_0 = values[:, 0].cpu().numpy()
        v_1 = values[:, 1].cpu().numpy()

        # Switching intensities
        pi_01 = torch.exp((values[:, 1] - config.g01 - values[:, 0]) / config.temperature)
        pi_10 = torch.exp((values[:, 0] - config.g10 - values[:, 1]) / config.temperature)

        # Convert to switch probability over one time step: 1 - exp(-π Δt)
        p_01 = (1.0 - torch.exp(-pi_01 * config.dt)).cpu().numpy()
        p_10 = (1.0 - torch.exp(-pi_10 * config.dt)).cpu().numpy()

    x_np = x_range.cpu().numpy()

    # Value functions on left y-axis
    ax2.plot(x_np, v_0, 'b', linestyle='dashed', label='V₀ (Regime 0: Push Left)', linewidth=3)
    ax2.plot(x_np, v_1, color='red', linestyle='dashed', label='V₁ (Regime 1: Push Right)', linewidth=3)
    ax2.set_xlabel('State x')
    ax2.set_ylabel('Value Function')
    ax2.set_title(f'Values at t=0.5 (λ={config.temperature})')

    # Switching probabilities on right y-axis
    ax2_right = ax2.twinx()
    ax2_right.plot(x_np, p_01, 'g', linestyle='solid', label='Switch 0→1', linewidth=3)
    ax2_right.plot(x_np, p_10, 'orange', linestyle='solid', label='Switch 1→0', linewidth=3)
    ax2_right.set_ylabel('Switch Probability')
    ax2_right.set_ylim(0, 1)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bounded_regulator_results.png', dpi=150)
    plt.show()
    print("Results saved to 'bounded_regulator_results.png'")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = Config()
    print(f"Problem parameters:")
    print(f"  Drifts: μ₀={config.mu[0]}, μ₁={config.mu[1]}")
    print(f"  Volatility: σ={config.sigma}")
    print(f"  Switching costs: g₀₁={config.g01}, g₁₀={config.g10}")
    print(f"  Temperature: λ={config.temperature}")
    print(f"  Time horizon: T={config.T}, steps={config.K}, dt={config.dt}")
    print(f"Training parameters:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Episodes: {config.num_episodes}")
    print(f"  Learning rate: {config.learning_rate}")
    print()

    print("Starting training...")
    value_net, losses = train(config, device)
    print("Training complete!")

    print("\nGenerating plots...")
    plot_results(value_net, losses, config, device)


if __name__ == "__main__":
    main()
