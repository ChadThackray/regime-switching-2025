"""
Continuous-time RL for Optimal Regime Switching: BTC/ETH Pairs Trading

3 regimes:
  0 = Long spread (long BTC, short ETH)
  1 = Short spread (short BTC, long ETH)
  2 = Flat (cash)

State: log(BTC/ETH) spread, modeled as Ornstein-Uhlenbeck process
  dS = θ(μ - S)dt + σdW
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from regime_switching import Config, ValueNetwork, train

# Fixed seed for reproducibility
SEED = 42

# OU parameters (estimated from real data, in daily units)
THETA = 41.6  # Mean-reversion speed per day
MU = 3.367  # Long-term mean of log(BTC/ETH)
SIGMA = 0.015  # Volatility per sqrt(day)

# Funding and cash rates (daily)
R_FUNDING = 0.10 / 365  # ~10% annualized funding cost, per day
R_CASH = 0.05 / 365  # ~5% annualized cash yield, per day

# Trading fees
FEE = 0.0004  # 0.04% per trade

# Switching cost matrix (as fraction of position)
SWITCHING_COSTS = [
    [0.0, 4 * FEE, 2 * FEE],  # from Long: reverse=4x, exit=2x
    [4 * FEE, 0.0, 2 * FEE],  # from Short: reverse=4x, exit=2x
    [2 * FEE, 2 * FEE, 0.0],  # from Flat: enter=2x
]

# Initial spread distribution (normalized: S_norm = (S - MU) / SPREAD_SCALE)
SPREAD_SCALE = 0.01  # Normalize so typical spread is O(1)
S0_MEAN = 0.0  # Normalized mean
S0_STD = 0.5  # Normalized std (0.5 * 0.01 = 0.005 in raw terms)


def dynamics_fn(
    t: torch.Tensor, x: torch.Tensor, i: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    OU dynamics for the normalized spread.

    x = (S - μ) / SPREAD_SCALE, so x=0 is equilibrium.
    dx = -θ * x * dt + (σ/SPREAD_SCALE) * dW

    Args:
        t: Time (batch,) - unused
        x: Normalized spread (batch,)
        i: Regime index (batch,) - unused for dynamics

    Returns:
        drift: (batch,)
        diffusion: (batch,)
    """
    drift = -THETA * x  # Mean-reverts to 0
    diffusion = torch.full_like(x, SIGMA / SPREAD_SCALE)
    return drift, diffusion


def running_reward_fn(
    t: torch.Tensor, x: torch.Tensor, i: torch.Tensor
) -> torch.Tensor:
    """
    Running reward based on position and normalized spread.

    x = (S - μ) / SPREAD_SCALE, so:
    - x < 0: spread below equilibrium, Long profits
    - x > 0: spread above equilibrium, Short profits

    Long spread (i=0):  -θ * SPREAD_SCALE * x - r_funding
    Short spread (i=1): +θ * SPREAD_SCALE * x - r_funding
    Flat (i=2):         r_cash

    Args:
        t: Time (batch,) - unused
        x: Normalized spread (batch,)
        i: Regime index (batch,)

    Returns:
        reward: (batch,)
    """
    # Expected profit rate from mean-reversion
    drift_profit_long = -THETA * SPREAD_SCALE * x  # Positive when x < 0
    drift_profit_short = THETA * SPREAD_SCALE * x  # Positive when x > 0

    reward_long = drift_profit_long - R_FUNDING
    reward_short = drift_profit_short - R_FUNDING
    reward_flat = torch.full_like(x, R_CASH)

    reward = torch.where(
        i == 0,
        reward_long,
        torch.where(i == 1, reward_short, reward_flat),
    )
    return reward


def terminal_reward(x: torch.Tensor) -> torch.Tensor:
    """Terminal reward h = 0."""
    return torch.zeros(x.shape[0], device=x.device)


def plot_results(
    value_net: ValueNetwork,
    losses: list[float],
    config: Config,
    device: torch.device,
) -> None:
    """Generate visualization plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training loss
    ax1 = axes[0]
    episodes = np.arange(1, len(losses) + 1)
    ax1.semilogy(episodes, np.abs(losses), alpha=0.3, label="Raw |Loss|")

    window = 50
    if len(losses) >= window:
        abs_losses = np.abs(losses)
        moving_avg = np.convolve(abs_losses, np.ones(window) / window, mode="valid")
        ax1.semilogy(
            episodes[window - 1 :],
            moving_avg,
            label="Smoothed |Loss| (MA-50)",
            color="orange",
        )

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("|Loss| (Log Scale)")
    ax1.set_title("Training Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Value functions and optimal regime
    ax2 = axes[1]

    value_net.eval()

    # Range of normalized spread values (±2 std)
    spread_range = torch.linspace(-2.0, 2.0, 200, device=device)
    t_mid = torch.full_like(spread_range, config.T / 2)  # Middle of horizon

    with torch.no_grad():
        values = value_net(t_mid, spread_range)
        v_long = values[:, 0].cpu().numpy()
        v_short = values[:, 1].cpu().numpy()
        v_flat = values[:, 2].cpu().numpy()
        optimal_regime = values.argmax(dim=1).cpu().numpy()

    spread_np = spread_range.cpu().numpy()

    # Value functions
    ax2.plot(spread_np, v_long, "b-", label="V(Long)", linewidth=2)
    ax2.plot(spread_np, v_short, "r-", label="V(Short)", linewidth=2)
    ax2.plot(spread_np, v_flat, "g-", label="V(Flat)", linewidth=2)

    # Mark equilibrium
    ax2.axvline(x=0, color="black", linestyle="--", alpha=0.5, label="Equilibrium")

    # Shade optimal regions
    for idx in range(len(spread_np) - 1):
        color = ["blue", "red", "green"][optimal_regime[idx]]
        ax2.axvspan(spread_np[idx], spread_np[idx + 1], alpha=0.1, color=color)

    ax2.set_xlabel("Normalized Spread (S - μ) / scale")
    ax2.set_ylabel("Value Function")
    ax2.set_title(f"Optimal Policy at t={config.T/2:.3f} days")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pairs_trading_results.png", dpi=150)
    plt.show()
    print("Results saved to 'pairs_trading_results.png'")


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC/ETH Pairs Trading")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes",
    )
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # T = 2 hours = 2/24 days
    T = 2.0 / 24.0  # ~0.0833 days
    K = 120  # 1 step per minute

    config = Config(
        num_regimes=3,
        state_dim=1,
        switching_costs=SWITCHING_COSTS,
        temperature=0.1,  # Higher temperature for exploration
        T=T,
        K=K,
        batch_size=256,
        num_episodes=args.episodes,
        learning_rate=1e-3,
        hidden_dim=64,
        x0_mean=S0_MEAN,
        x0_std=S0_STD,
    )

    print("BTC/ETH Pairs Trading with Regime Switching")
    print()
    print("OU Parameters:")
    print(f"  θ (mean-reversion): {THETA:.2f} /day")
    print(f"  μ (equilibrium):    {MU:.4f}")
    print(f"  σ (volatility):     {SIGMA:.4f} /√day")
    print(f"  Half-life:          {np.log(2)/THETA*24*60:.1f} minutes")
    print()
    print("Costs:")
    print(f"  Trading fee:        {FEE*100:.2f}% per trade")
    print(f"  Funding rate:       {R_FUNDING*365*100:.1f}% annualized")
    print(f"  Cash yield:         {R_CASH*365*100:.1f}% annualized")
    print()
    print("Training:")
    print(f"  Horizon T:          {T*24:.1f} hours ({T:.4f} days)")
    print(f"  Steps K:            {K}")
    print(f"  dt:                 {config.dt*24*60:.2f} minutes")
    print(f"  Temperature:        {config.temperature}")
    print(f"  Episodes:           {config.num_episodes}")
    print()

    print("Starting training...")
    value_net, losses = train(
        config, terminal_reward, dynamics_fn, running_reward_fn, device
    )
    print("Training complete!")

    # Show final policy summary
    value_net.eval()
    # Test at normalized spread values: -1, -0.5, 0, 0.5, 1
    test_spreads = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=device)
    test_t = torch.full_like(test_spreads, T / 2)

    with torch.no_grad():
        values = value_net(test_t, test_spreads)
        optimal = values.argmax(dim=1).cpu().numpy()

    regime_names = ["Long", "Short", "Flat"]
    print("\nLearned Policy (at t=T/2):")
    print("Normalized spread: x = (S - μ) / scale")
    print("  x < 0: spread below equilibrium")
    print("  x > 0: spread above equilibrium")
    print("-" * 60)
    print(f"{'x':>8} {'V_Long':>12} {'V_Short':>12} {'V_Flat':>12} {'Optimal':>10}")
    print("-" * 60)
    for idx, s in enumerate(test_spreads.cpu().numpy()):
        v_l = values[idx, 0].item()
        v_s = values[idx, 1].item()
        v_f = values[idx, 2].item()
        print(f"  {s:>6.2f}   {v_l:>12.6f} {v_s:>12.6f} {v_f:>12.6f} {regime_names[optimal[idx]]:>10}")

    if not args.no_plot:
        print("\nGenerating plots...")
        plot_results(value_net, losses, config, device)


if __name__ == "__main__":
    main()
