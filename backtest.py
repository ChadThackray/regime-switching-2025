"""
Walk-Forward Backtesting Framework for BTC/ETH Pairs Trading

Compares RL-learned thresholds vs naive fixed-threshold strategies.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from fetch_data import estimate_ou_parameters, fetch_btc_eth_history
from regime_switching import Config, train


# Constants
SEED = 42
FEE = 0.0004  # 0.04% per trade
SPREAD_SCALE = 0.01  # Normalization scale for spread

# Switching costs (same as pairs_trading.py)
SWITCHING_COSTS = [
    [0.0, 4 * FEE, 2 * FEE],  # from Long
    [4 * FEE, 0.0, 2 * FEE],  # from Short
    [2 * FEE, 2 * FEE, 0.0],  # from Flat
]

# Funding and cash rates (daily)
R_FUNDING = 0.10 / 365  # ~10% annualized
R_CASH = 0.05 / 365  # ~5% annualized


@dataclass
class BacktestConfig:
    """Configuration for walk-forward backtest."""

    lookback_days: int = 3
    trading_period_days: int = 1
    total_days: int = 90
    interval: str = "1h"
    training_episodes: int = 500
    start_date: datetime | None = None  # If None, ends at now


def create_dynamics_fn(theta: float, spread_scale: float):
    """Create dynamics function with given OU parameters."""

    def dynamics_fn(t: torch.Tensor, x: torch.Tensor, i: torch.Tensor):
        drift = -theta * x
        diffusion = torch.full_like(x, 0.015 / spread_scale)  # Approximate sigma
        return drift, diffusion

    return dynamics_fn


def create_running_reward_fn(theta: float, spread_scale: float):
    """Create running reward function with given OU parameters."""

    def running_reward_fn(t: torch.Tensor, x: torch.Tensor, i: torch.Tensor):
        drift_profit_long = -theta * spread_scale * x
        drift_profit_short = theta * spread_scale * x

        reward_long = drift_profit_long - R_FUNDING
        reward_short = drift_profit_short - R_FUNDING
        reward_flat = torch.full_like(x, R_CASH)

        reward = torch.where(
            i == 0,
            reward_long,
            torch.where(i == 1, reward_short, reward_flat),
        )
        return reward

    return running_reward_fn


def terminal_reward(x: torch.Tensor) -> torch.Tensor:
    """Terminal reward h = 0."""
    return torch.zeros(x.shape[0], device=x.device)


def extract_thresholds(
    value_net: torch.nn.Module,
    device: torch.device,
) -> dict:
    """Extract switching thresholds from trained value network.

    Returns normalized thresholds where:
    - x < long_threshold: Go Long
    - x > short_threshold: Go Short
    - Otherwise: Stay Flat
    """
    value_net.eval()

    x_range = torch.linspace(-3.0, 3.0, 1000, device=device)
    t = torch.zeros_like(x_range)

    with torch.no_grad():
        values = value_net(t, x_range)

    # Find where each regime is optimal
    optimal = values.argmax(dim=1)

    # Find thresholds (boundaries between regimes)
    x_np = x_range.cpu().numpy()
    optimal_np = optimal.cpu().numpy()

    # Long threshold: rightmost x where Long (0) is optimal
    long_mask = optimal_np == 0
    if long_mask.any():
        long_threshold = x_np[long_mask].max()
    else:
        long_threshold = -3.0  # Default to never long

    # Short threshold: leftmost x where Short (1) is optimal
    short_mask = optimal_np == 1
    if short_mask.any():
        short_threshold = x_np[short_mask].min()
    else:
        short_threshold = 3.0  # Default to never short

    return {
        "long_threshold": float(long_threshold),
        "short_threshold": float(short_threshold),
    }


def calculate_switching_cost(from_regime: int, to_regime: int) -> float:
    """Calculate cost of switching between regimes."""
    return SWITCHING_COSTS[from_regime][to_regime]


def simulate_period(
    prices: pd.DataFrame,
    thresholds: dict,
    mu: float,
    initial_position: int,
    initial_spread: float | None,
) -> tuple[pd.DataFrame, int, float]:
    """Simulate trading for one period using given thresholds.

    Args:
        prices: DataFrame with close_btc, close_eth, log_ratio
        thresholds: Dict with long_threshold and short_threshold
        mu: Equilibrium spread value
        initial_position: Starting regime (0=Long, 1=Short, 2=Flat)
        initial_spread: Spread value from end of previous period (or None)

    Returns:
        results: DataFrame with per-candle results
        final_position: Ending regime
        final_spread: Spread value at end of period
    """
    results = []
    position = initial_position
    prev_spread = initial_spread

    for row in prices.itertuples():
        spread = row.log_ratio
        assert isinstance(spread, float)
        x = (spread - mu) / SPREAD_SCALE

        # Determine target position based on thresholds
        if x < thresholds["long_threshold"]:
            target = 0  # Long
        elif x > thresholds["short_threshold"]:
            target = 1  # Short
        else:
            target = 2  # Flat

        # Calculate PnL from spread movement
        pnl = 0.0
        if prev_spread is not None:
            spread_change = spread - prev_spread
            if position == 0:  # Long
                pnl = spread_change  # Profit when spread rises
            elif position == 1:  # Short
                pnl = -spread_change  # Profit when spread falls
            # Flat: no PnL from spread movement

        # Calculate switching cost
        switch_cost = calculate_switching_cost(position, target)

        results.append(
            {
                "time": row.open_time,
                "spread": spread,
                "x_normalized": x,
                "position": position,
                "target": target,
                "pnl": pnl,
                "switch_cost": switch_cost,
                "net_pnl": pnl - switch_cost,
            }
        )

        position = target
        prev_spread = spread

    assert prev_spread is not None, "prices DataFrame was empty"
    return pd.DataFrame(results), position, prev_spread


class NaiveStrategy:
    """Fixed z-score threshold strategy."""

    def __init__(self, z_threshold: float = 1.0):
        self.z_threshold = z_threshold

    def get_thresholds(self, spread_std: float) -> dict:
        """Convert z-score threshold to normalized spread thresholds.

        Args:
            spread_std: Standard deviation of spread from lookback window
        """
        # z = (spread - mu) / spread_std
        # x = (spread - mu) / SPREAD_SCALE
        # So x = z * spread_std / SPREAD_SCALE
        return {
            "long_threshold": -self.z_threshold * spread_std / SPREAD_SCALE,
            "short_threshold": self.z_threshold * spread_std / SPREAD_SCALE,
        }


def run_backtest(config: BacktestConfig) -> dict:
    """Run walk-forward backtest comparing RL vs naive strategies."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Calculate end_time from start_date
    if config.start_date is not None:
        end_time = config.start_date + timedelta(days=config.total_days)
    else:
        end_time = None  # Will default to now

    # Fetch historical data
    data = fetch_btc_eth_history(
        days=config.total_days, interval=config.interval, end_time=end_time
    )

    # Calculate candles per day
    candles_per_day = 24 if config.interval == "1h" else 1440

    # Initialize strategies
    naive_1sigma = NaiveStrategy(z_threshold=1.0)
    naive_2sigma = NaiveStrategy(z_threshold=2.0)

    # Results storage
    all_results = {"rl": [], "naive_1sigma": [], "naive_2sigma": []}
    threshold_history = []

    # Initial positions and spreads
    positions = {"rl": 2, "naive_1sigma": 2, "naive_2sigma": 2}
    spreads: dict[str, float | None] = {"rl": None, "naive_1sigma": None, "naive_2sigma": None}

    # Walk-forward loop
    lookback_candles = config.lookback_days * candles_per_day
    trading_candles = config.trading_period_days * candles_per_day

    num_periods = (len(data) - lookback_candles) // trading_candles
    print(f"Running {num_periods} trading periods...")

    for period in range(num_periods):
        start_idx = period * trading_candles
        lookback_end = start_idx + lookback_candles
        trading_end = lookback_end + trading_candles

        if trading_end > len(data):
            break

        lookback_data = data.iloc[start_idx:lookback_end]
        trading_data = data.iloc[lookback_end:trading_end]

        # Estimate OU parameters from lookback window
        spread = lookback_data["log_ratio"].to_numpy()
        dt_hours = 1.0 / 24  # 1 hour in days
        ou_params = estimate_ou_parameters(spread, dt_hours)

        mu = ou_params["mu"]
        theta = ou_params["theta"]
        spread_std = spread.std()  # Direct std for naive strategies

        print(
            f"Period {period + 1}/{num_periods}: "
            f"θ={theta:.2f}, μ={mu:.4f}, std={spread_std:.4f}"
        )

        # Train RL model
        dynamics_fn = create_dynamics_fn(theta, SPREAD_SCALE)
        running_reward_fn = create_running_reward_fn(theta, SPREAD_SCALE)

        T = config.trading_period_days / 24  # Convert to days (horizon in days)
        K = trading_candles

        rl_config = Config(
            num_regimes=3,
            state_dim=1,
            switching_costs=SWITCHING_COSTS,
            temperature=0.1,
            T=T * 24,  # Horizon covers one trading period
            K=K,
            batch_size=256,
            num_episodes=config.training_episodes,
            learning_rate=1e-3,
            hidden_dim=64,
            x0_mean=0.0,
            x0_std=0.5,
        )

        value_net, _ = train(
            rl_config, terminal_reward, dynamics_fn, running_reward_fn, device
        )

        # Extract RL thresholds
        rl_thresholds = extract_thresholds(value_net, device)
        threshold_history.append(
            {
                "period": period,
                "long_threshold": rl_thresholds["long_threshold"],
                "short_threshold": rl_thresholds["short_threshold"],
                "theta": theta,
                "mu": mu,
            }
        )

        # Get naive thresholds (using direct std from lookback)
        naive_1sigma_thresholds = naive_1sigma.get_thresholds(spread_std)
        naive_2sigma_thresholds = naive_2sigma.get_thresholds(spread_std)

        # Simulate trading for each strategy
        rl_results, positions["rl"], spreads["rl"] = simulate_period(
            trading_data, rl_thresholds, mu, positions["rl"], spreads["rl"]
        )
        all_results["rl"].append(rl_results)

        naive_1_results, positions["naive_1sigma"], spreads["naive_1sigma"] = simulate_period(
            trading_data, naive_1sigma_thresholds, mu, positions["naive_1sigma"], spreads["naive_1sigma"]
        )
        all_results["naive_1sigma"].append(naive_1_results)

        naive_2_results, positions["naive_2sigma"], spreads["naive_2sigma"] = simulate_period(
            trading_data, naive_2sigma_thresholds, mu, positions["naive_2sigma"], spreads["naive_2sigma"]
        )
        all_results["naive_2sigma"].append(naive_2_results)

    # Close final positions (add exit fee if not flat)
    for strategy in ["rl", "naive_1sigma", "naive_2sigma"]:
        if positions[strategy] != 2:  # Not flat
            exit_fee = calculate_switching_cost(positions[strategy], 2)
            # Get last timestamp from results
            last_result = all_results[strategy][-1]
            last_time = last_result["time"].iloc[-1] if not last_result.empty else None
            # Append closing transaction
            closing_row = pd.DataFrame([{
                "time": last_time,
                "spread": spreads[strategy],
                "x_normalized": 0.0,  # Not meaningful for closing
                "position": positions[strategy],
                "target": 2,  # Close to flat
                "pnl": 0.0,
                "switch_cost": exit_fee,
                "net_pnl": -exit_fee,
            }])
            all_results[strategy].append(closing_row)

    # Combine results
    combined = {
        "rl": pd.concat(all_results["rl"], ignore_index=True),
        "naive_1sigma": pd.concat(all_results["naive_1sigma"], ignore_index=True),
        "naive_2sigma": pd.concat(all_results["naive_2sigma"], ignore_index=True),
        "thresholds": pd.DataFrame(threshold_history),
    }

    return combined


def calculate_metrics(results: pd.DataFrame) -> dict:
    """Calculate performance metrics from trading results."""
    cumulative_pnl = results["net_pnl"].cumsum()
    returns = results["net_pnl"]

    # Count trades (regime changes)
    trades = (results["position"] != results["target"]).sum()

    # Gross PnL (before fees) and total fees
    gross_pnl = results["pnl"].sum()
    total_fees = results["switch_cost"].sum()
    net_pnl = results["net_pnl"].sum()

    # Sharpe ratio (annualized, assuming hourly data)
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(24 * 365)
    else:
        sharpe = 0.0

    # Max drawdown
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    max_drawdown = drawdown.min()

    return {
        "gross_pnl": gross_pnl,
        "total_fees": total_fees,
        "net_pnl": net_pnl,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "num_trades": trades,
    }


def plot_results(results: dict) -> None:
    """Plot backtest results."""
    fig = plt.figure(figsize=(14, 10))

    # Plot 1: Cumulative PnL comparison (top, full width)
    ax1 = fig.add_subplot(2, 1, 1)
    for name, df in [
        ("RL", results["rl"]),
        ("Naive ±1σ", results["naive_1sigma"]),
        ("Naive ±2σ", results["naive_2sigma"]),
    ]:
        cumulative = df["net_pnl"].cumsum()
        ax1.plot(df["time"], cumulative, label=name)

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Cumulative PnL")
    ax1.set_title("Strategy Comparison: Cumulative PnL")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Spread with RL signals (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    rl_df = results["rl"]
    ax3.plot(rl_df["time"], rl_df["spread"], "gray", alpha=0.5, label="Spread")

    # Color by position
    for pos, color, label in [(0, "blue", "Long"), (1, "red", "Short"), (2, "green", "Flat")]:
        mask = rl_df["position"] == pos
        if mask.any():
            ax3.scatter(
                rl_df.loc[mask, "time"],
                rl_df.loc[mask, "spread"],
                c=color,
                s=1,
                alpha=0.5,
                label=label,
            )

    ax3.set_xlabel("Time")
    ax3.set_ylabel("log(BTC/ETH)")
    ax3.set_title("RL Strategy Positions")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 3: Performance metrics table (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")

    metrics_data = []
    for name, key in [("RL", "rl"), ("Naive ±1σ", "naive_1sigma"), ("Naive ±2σ", "naive_2sigma")]:
        m = calculate_metrics(results[key])
        metrics_data.append([
            name,
            f"{m['gross_pnl']:.4f}",
            f"{m['total_fees']:.4f}",
            f"{m['net_pnl']:.4f}",
            f"{m['num_trades']}",
        ])

    table = ax4.table(
        cellText=metrics_data,
        colLabels=["Strategy", "Gross PnL", "Fees", "Net PnL", "Trades"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title("Performance Metrics", pad=20)

    plt.tight_layout()
    plt.savefig("backtest_results.png", dpi=150)
    plt.show()
    print("Results saved to 'backtest_results.png'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-Forward Backtest")
    parser.add_argument("--days", type=int, default=90, help="Total backtest days")
    parser.add_argument("--lookback", type=int, default=3, help="Lookback days for OU estimation")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes per period")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for backtest (YYYY-MM-DD). If not specified, ends at now.",
    )
    args = parser.parse_args()

    # Parse start date
    start_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")

    config = BacktestConfig(
        lookback_days=args.lookback,
        total_days=args.days,
        training_episodes=args.episodes,
        start_date=start_date,
    )

    print("Walk-Forward Backtest: BTC/ETH Pairs Trading")
    print(f"  Lookback: {config.lookback_days} days")
    print(f"  Trading period: {config.trading_period_days} day")
    print(f"  Total days: {config.total_days}")
    if config.start_date:
        print(f"  Start date: {config.start_date.strftime('%Y-%m-%d')}")
    print(f"  Training episodes: {config.training_episodes}")
    print()

    results = run_backtest(config)

    # Print summary
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    for name, key in [("RL", "rl"), ("Naive ±1σ", "naive_1sigma"), ("Naive ±2σ", "naive_2sigma")]:
        metrics = calculate_metrics(results[key])
        print(f"\n{name}:")
        print(f"  Gross PnL:    {metrics['gross_pnl']:.4f}")
        print(f"  Total Fees:   {metrics['total_fees']:.4f}")
        print(f"  Net PnL:      {metrics['net_pnl']:.4f}")
        print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.4f}")
        print(f"  Num Trades:   {metrics['num_trades']}")

    if not args.no_plot:
        print("\nGenerating plots...")
        plot_results(results)


if __name__ == "__main__":
    main()
