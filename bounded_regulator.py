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

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from regime_switching import Config, ValueNetwork, train

# Fixed seed for reproducibility
SEED = 42

# Problem-specific parameters
MU = (-2.0, 2.0)  # Drift for each regime
SIGMA = 0.5  # Volatility
G01 = 0.5  # Switching cost 0 -> 1
G10 = 0.5  # Switching cost 1 -> 0


def dynamics_fn(
    t: torch.Tensor, x: torch.Tensor, i: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    State dynamics for the bounded regulator problem.

    dX_t = mu_i dt + sigma dW_t

    Args:
        t: Time (batch,) - unused
        x: State (batch,)
        i: Regime index (batch,)

    Returns:
        drift: (batch,)
        diffusion: (batch,) - constant sigma
    """
    mu = torch.tensor(MU, device=x.device)
    drift = mu[i]  # (batch,)
    diffusion = torch.full_like(x, SIGMA)  # (batch,)
    return drift, diffusion


def running_reward_fn(
    t: torch.Tensor, x: torch.Tensor, i: torch.Tensor
) -> torch.Tensor:
    """
    Running reward f(x) = 2e^(-2x^2) - 0.1

    Regime-independent in this problem.

    Args:
        t: Time (batch,) - unused
        x: State (batch,)
        i: Regime index (batch,) - unused

    Returns:
        reward: (batch,)
    """
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
        ax1.semilogy(
            episodes[window - 1 :], moving_avg, label="Moving Avg", color="orange"
        )

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
        g01 = config.switching_cost(0, 1)
        g10 = config.switching_cost(1, 0)
        pi_01 = torch.exp((values[:, 1] - g01 - values[:, 0]) / config.temperature)
        pi_10 = torch.exp((values[:, 0] - g10 - values[:, 1]) / config.temperature)

        # Convert to switch probability over one time step: 1 - exp(-pi * dt)
        p_01 = (1.0 - torch.exp(-pi_01 * config.dt)).cpu().numpy()
        p_10 = (1.0 - torch.exp(-pi_10 * config.dt)).cpu().numpy()

    x_np = x_range.cpu().numpy()

    # Value functions on left y-axis
    ax2.plot(
        x_np,
        v_0,
        "b",
        linestyle="dashed",
        label="V_0 (Regime 0: Push Left)",
        linewidth=3,
    )
    ax2.plot(
        x_np,
        v_1,
        color="red",
        linestyle="dashed",
        label="V_1 (Regime 1: Push Right)",
        linewidth=3,
    )
    ax2.set_xlabel("State x")
    ax2.set_ylabel("Value Function")
    ax2.set_title(f"Values at t=0.5 (lambda={config.temperature})")

    # Switching probabilities on right y-axis
    ax2_right = ax2.twinx()
    ax2_right.plot(x_np, p_01, "g", linestyle="solid", label="Switch 0->1", linewidth=3)
    ax2_right.plot(
        x_np, p_10, "orange", linestyle="solid", label="Switch 1->0", linewidth=3
    )
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


def compute_test_values(
    value_net: ValueNetwork, config: Config, device: torch.device
) -> dict[str, np.ndarray]:
    """Compute value functions at test points for regression testing."""
    value_net.eval()
    test_x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)
    test_t = torch.full_like(test_x, 0.5)

    with torch.no_grad():
        values = value_net(test_t, test_x)
        v_0 = values[:, 0].cpu().numpy()
        v_1 = values[:, 1].cpu().numpy()

        # Switching intensities
        g01 = config.switching_cost(0, 1)
        g10 = config.switching_cost(1, 0)
        pi_01 = torch.exp((values[:, 1] - g01 - values[:, 0]) / config.temperature)
        pi_10 = torch.exp((values[:, 0] - g10 - values[:, 1]) / config.temperature)

    return {
        "test_x": test_x.cpu().numpy(),
        "v_0": v_0,
        "v_1": v_1,
        "pi_01": pi_01.cpu().numpy(),
        "pi_10": pi_10.cpu().numpy(),
    }


def save_baseline(losses: list[float], test_values: dict[str, np.ndarray]) -> None:
    """Save baseline outputs for regression testing."""
    baseline_path = Path("baseline_outputs.npz")
    np.savez(
        baseline_path,
        losses=np.array(losses),
        **test_values,
    )
    print(f"Baseline saved to {baseline_path}")


def verify_against_baseline(
    losses: list[float], test_values: dict[str, np.ndarray]
) -> bool:
    """Verify outputs match baseline within tolerance."""
    baseline_path = Path("baseline_outputs.npz")
    if not baseline_path.exists():
        print(f"ERROR: Baseline file {baseline_path} not found!")
        print("Run with --save-baseline first.")
        return False

    baseline = np.load(baseline_path)
    rtol = 1e-5
    atol = 1e-7
    all_match = True

    # Check losses
    if not np.allclose(losses, baseline["losses"], rtol=rtol, atol=atol):
        print("MISMATCH: Losses do not match baseline!")
        max_diff = np.max(np.abs(np.array(losses) - baseline["losses"]))
        print(f"  Max difference: {max_diff}")
        all_match = False
    else:
        print("OK: Losses match baseline")

    # Check test values
    for key in ["v_0", "v_1", "pi_01", "pi_10"]:
        if not np.allclose(test_values[key], baseline[key], rtol=rtol, atol=atol):
            print(f"MISMATCH: {key} does not match baseline!")
            max_diff = np.max(np.abs(test_values[key] - baseline[key]))
            print(f"  Max difference: {max_diff}")
            all_match = False
        else:
            print(f"OK: {key} matches baseline")

    if all_match:
        print("\nREGRESSION TEST PASSED!")
    else:
        print("\nREGRESSION TEST FAILED!")

    return all_match


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded Regulator Problem")
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save outputs as baseline for regression testing",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify outputs match saved baseline",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (useful for automated testing)",
    )
    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Bounded regulator problem parameters
    config = Config(
        num_regimes=2,
        state_dim=1,
        switching_costs=[[0.0, G01], [G10, 0.0]],
        temperature=0.2,
    )

    print("Problem parameters:")
    print(f"  Drifts: mu_0={MU[0]}, mu_1={MU[1]}")
    print(f"  Volatility: sigma={SIGMA}")
    print(f"  Switching costs: g_01={G01}, g_10={G10}")
    print(f"  Temperature: lambda={config.temperature}")
    print(f"  Time horizon: T={config.T}, steps={config.K}, dt={config.dt}")
    print("Training parameters:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Episodes: {config.num_episodes}")
    print(f"  Learning rate: {config.learning_rate}")
    print()

    print("Starting training...")
    value_net, losses = train(
        config, terminal_reward, dynamics_fn, running_reward_fn, device
    )
    print("Training complete!")

    # Compute test values for regression testing
    test_values = compute_test_values(value_net, config, device)

    if args.save_baseline:
        save_baseline(losses, test_values)

    if args.verify:
        verify_against_baseline(losses, test_values)

    if not args.no_plot:
        print("\nGenerating plots...")
        plot_results(value_net, losses, config, device)


if __name__ == "__main__":
    main()
