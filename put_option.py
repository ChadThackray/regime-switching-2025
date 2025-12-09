"""
Continuous-time RL for Optimal Regime Switching: Put Option Selection Problem

Implementation of the algorithm from arXiv:2512.04697v1 (Section 6.2)

Problem Setup:
- 3 regimes: Put option on Stock A (0), Put option on Stock B (1), Bank (2)
- 2D state: (S^A, S^B) - two stock prices
- GBM dynamics: dS = mu*S dt + sigma*S dW
- Stock A: (mu_A, sigma_A) = (0.1, 0.2)
- Stock B: (mu_B, sigma_B) = (0.05, 0.1)
- Risk-free rate: r = 0.05
- Strike price: S_K = 1
- Running reward: put payoffs or bank interest
- Terminal reward: h = 0
- Temperature lambda = 0.1
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from regime_switching import Config, ValueNetwork, train

# Fixed seed for reproducibility
SEED = 42

# Problem-specific parameters
MU_A = 0.1  # Drift for Stock A
SIGMA_A = 0.2  # Volatility for Stock A
MU_B = 0.05  # Drift for Stock B
SIGMA_B = 0.1  # Volatility for Stock B
R = 0.05  # Risk-free rate
S_K = 1.0  # Strike price

# Initial stock prices
S0_A = 1.0
S0_B = 1.0
S0_STD = 0.3  # Std for initial price distribution

# Switching cost matrix
SWITCHING_COSTS = [
    [0.0, 0.02, 0.01],  # From Put A
    [0.02, 0.0, 0.01],  # From Put B
    [0.02, 0.02, 0.0],  # From Bank
]


def dynamics_fn(
    t: torch.Tensor, x: torch.Tensor, i: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    State dynamics for the put option problem (Geometric Brownian Motion).

    dS^A_t = mu^A * S^A_t dt + sigma^A * S^A_t dW_t
    dS^B_t = mu^B * S^B_t dt + sigma^B * S^B_t dW_t

    Note: The dynamics are independent of the regime i in this problem.

    Args:
        t: Time (batch,) - unused
        x: State (batch, 2) where x[:, 0] = S^A, x[:, 1] = S^B
        i: Regime index (batch,) - unused for dynamics

    Returns:
        drift: (batch, 2)
        diffusion: (batch, 2)
    """
    s_a = x[:, 0]  # (batch,)
    s_b = x[:, 1]  # (batch,)

    # Drift: mu * S
    drift = torch.stack([MU_A * s_a, MU_B * s_b], dim=1)  # (batch, 2)

    # Diffusion: sigma * S
    diffusion = torch.stack([SIGMA_A * s_a, SIGMA_B * s_b], dim=1)  # (batch, 2)

    return drift, diffusion


def running_reward_fn(
    t: torch.Tensor, x: torch.Tensor, i: torch.Tensor
) -> torch.Tensor:
    """
    Running reward for the put option problem.

    f(s^A, s^B, i) = (S_K - s^A)^+  if i = 0 (Put A)
                   = (S_K - s^B)^+  if i = 1 (Put B)
                   = r * S_K        if i = 2 (Bank)

    Args:
        t: Time (batch,) - unused
        x: State (batch, 2) where x[:, 0] = S^A, x[:, 1] = S^B
        i: Regime index (batch,)

    Returns:
        reward: (batch,)
    """
    s_a = x[:, 0]  # (batch,)
    s_b = x[:, 1]  # (batch,)

    # Put payoffs and bank interest
    put_a = torch.clamp(S_K - s_a, min=0.0)  # (batch,)
    put_b = torch.clamp(S_K - s_b, min=0.0)  # (batch,)
    bank = torch.full_like(s_a, R * S_K)  # (batch,)

    # Select reward based on regime
    reward = torch.where(i == 0, put_a, torch.where(i == 1, put_b, bank))

    return reward


def terminal_reward(x: torch.Tensor) -> torch.Tensor:
    """
    Terminal reward h = 0.

    Args:
        x: State (batch, 2)

    Returns:
        reward: (batch,) all zeros
    """
    return torch.zeros(x.shape[0], device=x.device)


def init_state_fn(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Initialize stock prices with log-normal distribution.

    Args:
        batch_size: Number of samples
        device: Torch device

    Returns:
        x: (batch_size, 2) initial stock prices
    """
    # Sample from log-normal to ensure positive prices
    log_s_a = torch.randn(batch_size, device=device) * S0_STD + np.log(S0_A)
    log_s_b = torch.randn(batch_size, device=device) * S0_STD + np.log(S0_B)

    s_a = torch.exp(log_s_a)
    s_b = torch.exp(log_s_b)

    return torch.stack([s_a, s_b], dim=1)  # (batch_size, 2)


def plot_results(
    value_net: ValueNetwork,
    losses: list[float],
    config: Config,
    device: torch.device,
) -> None:
    """Generate visualization plots matching Figure 4 and Figure 5 in the paper."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training loss (Figure 4)
    ax1 = axes[0]
    episodes = np.arange(1, len(losses) + 1)
    ax1.semilogy(episodes, losses, alpha=0.3, label="Raw Loss")

    window = 50
    if len(losses) >= window:
        moving_avg = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax1.semilogy(
            episodes[window - 1 :],
            moving_avg,
            label="Smoothed Loss (MA-50)",
            color="orange",
        )

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("MSE Loss (Log Scale)")
    ax1.set_title("Training Convergence: Loss per Episode")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Optimal regime heatmap at t=0.5 (Figure 5)
    ax2 = axes[1]

    value_net.eval()

    # Create grid of stock prices
    n_grid = 100
    s_a_range = torch.linspace(0.4, 1.6, n_grid, device=device)
    s_b_range = torch.linspace(0.4, 1.6, n_grid, device=device)

    # Create meshgrid
    s_a_grid, s_b_grid = torch.meshgrid(s_a_range, s_b_range, indexing="xy")
    s_a_flat = s_a_grid.reshape(-1)
    s_b_flat = s_b_grid.reshape(-1)
    x_flat = torch.stack([s_a_flat, s_b_flat], dim=1)  # (n_grid^2, 2)

    t_fixed = torch.full((n_grid * n_grid,), 0.5, device=device)

    with torch.no_grad():
        values = value_net(t_fixed, x_flat)  # (n_grid^2, 3)
        optimal_regime = values.argmax(dim=1).cpu().numpy()  # (n_grid^2,)

    optimal_regime_grid = optimal_regime.reshape(n_grid, n_grid)

    # Create custom colormap
    from matplotlib.colors import ListedColormap

    colors = ["#1f77b4", "#d62728", "#2ca02c"]  # Blue, Red, Green
    cmap = ListedColormap(colors)

    # Plot heatmap
    # With indexing="xy", optimal_regime_grid[row, col] corresponds to (s_a_range[col], s_b_range[row])
    # imshow with origin="lower" displays row as y-axis and col as x-axis, which is correct
    im = ax2.imshow(
        optimal_regime_grid,
        extent=[0.4, 1.6, 0.4, 1.6],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=2.5,
    )

    # Add strike price reference lines
    ax2.axhline(y=S_K, color="black", linestyle="--", alpha=0.5, label=f"S_K={S_K}")
    ax2.axvline(x=S_K, color="black", linestyle="--", alpha=0.5)

    ax2.set_xlabel("Stock A Price")
    ax2.set_ylabel("Stock B Price")
    ax2.set_title("Optimal Switching Policy (at t=0.5)")

    # Add colorbar with regime labels
    cbar = plt.colorbar(im, ax=ax2, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(
        ["Hold Put A\n(Bear A)", "Hold Put B\n(Bear B)", "Bank\n(Cash)"]
    )

    plt.tight_layout()
    plt.savefig("put_option_results.png", dpi=150)
    plt.show()
    print("Results saved to 'put_option_results.png'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Put Option Selection Problem")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting",
    )
    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Put option problem parameters (from Section 6.2)
    config = Config(
        num_regimes=3,
        state_dim=2,
        switching_costs=SWITCHING_COSTS,
        temperature=0.1,
        T=1.0,
        K=50,
        batch_size=512,
        num_episodes=1000,
        learning_rate=1e-4,
        hidden_dim=128,
    )

    print("Put Option Selection Problem (Section 6.2)")
    print("Problem parameters:")
    print(f"  Stock A: mu={MU_A}, sigma={SIGMA_A}")
    print(f"  Stock B: mu={MU_B}, sigma={SIGMA_B}")
    print(f"  Risk-free rate: r={R}")
    print(f"  Strike price: S_K={S_K}")
    print("  Switching costs:")
    print(f"    Put A -> Put B: {SWITCHING_COSTS[0][1]}")
    print(f"    Put A -> Bank:  {SWITCHING_COSTS[0][2]}")
    print(f"    Put B -> Put A: {SWITCHING_COSTS[1][0]}")
    print(f"    Put B -> Bank:  {SWITCHING_COSTS[1][2]}")
    print(f"    Bank -> Put A:  {SWITCHING_COSTS[2][0]}")
    print(f"    Bank -> Put B:  {SWITCHING_COSTS[2][1]}")
    print(f"  Temperature: lambda={config.temperature}")
    print(f"  Time horizon: T={config.T}, steps={config.K}, dt={config.dt:.4f}")
    print("Training parameters:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Episodes: {config.num_episodes}")
    print(f"  Learning rate: {config.learning_rate}")
    print()

    print("Starting training...")
    value_net, losses = train(
        config, terminal_reward, dynamics_fn, running_reward_fn, device, init_state_fn
    )
    print("Training complete!")

    if not args.no_plot:
        print("\nGenerating plots...")
        plot_results(value_net, losses, config, device)


if __name__ == "__main__":
    main()
