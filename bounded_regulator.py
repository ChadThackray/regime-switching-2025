"""
Continuous-time RL for Optimal Regime Switching: Bounded Regulator Problem

Implementation of the algorithm from arXiv:2512.04697v1 (Section 6.1)

Problem Setup:
- 2 regimes with drift mu_0 = -2, mu_1 = 2
- Volatility sigma = 0.5
- Running reward f(x) = 2e^(-2x^2) - 0.1
- Terminal reward h(x) = 2e^(-2x^2)
- Switching costs g_01 = g_10 = 0.5
- Temperature lambda = 0.2
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from regime_switching import Config, ValueNetwork, train


def running_reward(x: torch.Tensor) -> torch.Tensor:
    """Running reward f(x) = 2e^(-2x^2) - 0.1"""
    return 2.0 * torch.exp(-2.0 * x**2) - 0.1


def terminal_reward(x: torch.Tensor) -> torch.Tensor:
    """Terminal reward h(x) = 2e^(-2x^2)"""
    return 2.0 * torch.exp(-2.0 * x**2)


def plot_results(
    value_net: ValueNetwork,
    losses: list[float],
    config: Config,
    device: torch.device,
) -> None:
    """Generate visualization plots matching Figure 1 in the paper."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training loss
    ax1 = axes[0]
    episodes = np.arange(1, len(losses) + 1)
    ax1.semilogy(episodes, losses, alpha=0.3, label="Raw Loss")

    window = 50
    if len(losses) >= window:
        moving_avg = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax1.semilogy(episodes[window - 1 :], moving_avg, label="Moving Avg", color="orange")

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
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

        # Convert to switch probability over one time step: 1 - exp(-pi * dt)
        p_01 = (1.0 - torch.exp(-pi_01 * config.dt)).cpu().numpy()
        p_10 = (1.0 - torch.exp(-pi_10 * config.dt)).cpu().numpy()

    x_np = x_range.cpu().numpy()

    # Value functions on left y-axis
    ax2.plot(x_np, v_0, "b", linestyle="dashed", label="V_0 (Regime 0: Push Left)", linewidth=3)
    ax2.plot(x_np, v_1, color="red", linestyle="dashed", label="V_1 (Regime 1: Push Right)", linewidth=3)
    ax2.set_xlabel("State x")
    ax2.set_ylabel("Value Function")
    ax2.set_title(f"Values at t=0.5 (lambda={config.temperature})")

    # Switching probabilities on right y-axis
    ax2_right = ax2.twinx()
    ax2_right.plot(x_np, p_01, "g", linestyle="solid", label="Switch 0->1", linewidth=3)
    ax2_right.plot(x_np, p_10, "orange", linestyle="solid", label="Switch 1->0", linewidth=3)
    ax2_right.set_ylabel("Switch Probability")
    ax2_right.set_ylim(0, 1)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("bounded_regulator_results.png", dpi=150)
    plt.show()
    print("Results saved to 'bounded_regulator_results.png'")


def main() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Bounded regulator problem parameters
    config = Config(
        mu=(-2.0, 2.0),
        sigma=0.5,
        g01=0.5,
        g10=0.5,
        temperature=0.2,
    )

    print("Problem parameters:")
    print(f"  Drifts: mu_0={config.mu[0]}, mu_1={config.mu[1]}")
    print(f"  Volatility: sigma={config.sigma}")
    print(f"  Switching costs: g_01={config.g01}, g_10={config.g10}")
    print(f"  Temperature: lambda={config.temperature}")
    print(f"  Time horizon: T={config.T}, steps={config.K}, dt={config.dt}")
    print("Training parameters:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Episodes: {config.num_episodes}")
    print(f"  Learning rate: {config.learning_rate}")
    print()

    print("Starting training...")
    value_net, losses = train(config, terminal_reward, running_reward, device)
    print("Training complete!")

    print("\nGenerating plots...")
    plot_results(value_net, losses, config, device)


if __name__ == "__main__":
    main()
