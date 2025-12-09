"""
Fetch BTC/ETH data and estimate OU parameters.
"""

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests


def fetch_binance_klines(
    symbol: str,
    interval: str,
    limit: int = 1000,
    end_time: datetime | None = None,
) -> pd.DataFrame:
    """Fetch klines (candlestick) data from Binance.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        interval: Candle interval (e.g., "1m", "1h", "1d")
        limit: Max number of candles (up to 1000)
        end_time: Optional end time for the query

    Returns:
        DataFrame with open_time and close columns
    """
    url = "https://api.binance.com/api/v3/klines"
    params: dict = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if end_time is not None:
        params["endTime"] = int(end_time.timestamp() * 1000)

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(
        data,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = df["close"].astype(float)
    return df[["open_time", "close"]]


def fetch_historical_klines(
    symbol: str,
    interval: str,
    days: int,
    end_time: datetime | None = None,
) -> pd.DataFrame:
    """Fetch historical klines with pagination to overcome 1000-candle limit.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        interval: Candle interval (e.g., "1h" for hourly)
        days: Number of days of history to fetch
        end_time: Optional end time (defaults to now)

    Returns:
        DataFrame with open_time and close columns, sorted chronologically
    """
    if end_time is None:
        end_time = datetime.now()

    # Calculate candles per day based on interval
    interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    minutes_per_candle = interval_minutes.get(interval, 60)
    candles_per_day = 1440 // minutes_per_candle
    total_candles_needed = days * candles_per_day

    all_data = []
    current_end = end_time

    while len(all_data) < total_candles_needed:
        batch = fetch_binance_klines(symbol, interval, limit=1000, end_time=current_end)
        if batch.empty:
            break
        all_data.append(batch)

        # Move end_time back to before the earliest candle in this batch
        current_end = batch["open_time"].min() - timedelta(milliseconds=1)
        time.sleep(0.1)  # Rate limiting

        # Safety check to prevent infinite loops
        if len(all_data) > 10:
            break

    if not all_data:
        return pd.DataFrame(columns=["open_time", "close"])

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["open_time"])
    combined = combined.sort_values("open_time").reset_index(drop=True)

    # Trim to requested number of days
    return combined.tail(total_candles_needed).reset_index(drop=True)


def fetch_btc_eth_history(
    days: int = 90,
    interval: str = "1h",
) -> pd.DataFrame:
    """Fetch aligned BTC and ETH historical data.

    Args:
        days: Number of days of history
        interval: Candle interval

    Returns:
        DataFrame with open_time, close_btc, close_eth, log_ratio columns
    """
    print(f"Fetching {days} days of {interval} data for BTC and ETH...")

    btc = fetch_historical_klines("BTCUSDT", interval, days)
    eth = fetch_historical_klines("ETHUSDT", interval, days)

    # Merge on time
    df = btc.merge(eth, on="open_time", suffixes=("_btc", "_eth"))
    df["log_ratio"] = np.log(df["close_btc"] / df["close_eth"])

    print(f"Got {len(df)} data points")
    print(f"Time range: {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}")

    return df


def estimate_ou_parameters(spread: np.ndarray, dt: float) -> dict:
    """
    Estimate OU parameters from discrete observations.

    dS = θ(μ - S)dt + σdW

    Discrete form: S_{t+1} = a + b*S_t + ε
    Where: b = 1 - θΔt, a = θμΔt

    Args:
        spread: Array of spread values
        dt: Time step (in whatever units you want θ, σ in)

    Returns:
        Dict with theta, mu, sigma estimates
    """
    s_t = spread[:-1]
    s_next = spread[1:]

    # OLS regression: S_{t+1} = a + b * S_t
    n = len(s_t)
    sum_s = np.sum(s_t)
    sum_s_next = np.sum(s_next)
    sum_ss = np.sum(s_t * s_t)
    sum_s_snext = np.sum(s_t * s_next)

    # b = (n * Σ(S_t * S_{t+1}) - Σ S_t * Σ S_{t+1}) / (n * Σ S_t² - (Σ S_t)²)
    b = (n * sum_s_snext - sum_s * sum_s_next) / (n * sum_ss - sum_s**2)
    a = (sum_s_next - b * sum_s) / n

    # Convert to OU parameters
    theta = (1 - b) / dt
    mu = a / (theta * dt) if theta * dt > 1e-10 else spread.mean()

    # Estimate sigma from residuals
    residuals = s_next - (a + b * s_t)
    sigma = np.std(residuals) / np.sqrt(dt)

    return {
        "theta": theta,
        "mu": mu,
        "sigma": sigma,
        "b": b,
        "a": a,
        "half_life": np.log(2) / theta if theta > 0 else np.inf,
    }


def main():
    print("Fetching BTC/USDT and ETH/USDT minutely data from Binance...")

    # Fetch last 1000 minutes (~16.7 hours)
    btc = fetch_binance_klines("BTCUSDT", "1m", limit=1000)
    time.sleep(0.1)  # Be nice to the API
    eth = fetch_binance_klines("ETHUSDT", "1m", limit=1000)

    # Merge on time
    df = btc.merge(eth, on="open_time", suffixes=("_btc", "_eth"))
    df["log_ratio"] = np.log(df["close_btc"] / df["close_eth"])

    print(f"Got {len(df)} data points")
    print(f"Time range: {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}")
    print()

    # Current spread
    current_spread = df["log_ratio"].iloc[-1]
    current_btc = df["close_btc"].iloc[-1]
    current_eth = df["close_eth"].iloc[-1]
    print(f"Current BTC: ${current_btc:,.2f}")
    print(f"Current ETH: ${current_eth:,.2f}")
    print(f"Current ratio: {current_btc/current_eth:.2f}")
    print(f"Current log ratio (spread): {current_spread:.4f}")
    print()

    # Estimate OU parameters
    # dt = 1 minute, but let's express in "per day" units
    # 1 minute = 1/(60*24) days
    dt_minutes = 1.0
    dt_days = 1.0 / (60 * 24)

    spread = df["log_ratio"].values

    # Estimate with daily units
    params = estimate_ou_parameters(spread, dt_days)

    print("OU Parameter Estimates (daily units):")
    print(f"  θ (mean-reversion speed): {params['theta']:.4f} per day")
    print(f"  μ (long-term mean):       {params['mu']:.4f}")
    print(f"  σ (volatility):           {params['sigma']:.4f} per √day")
    print(f"  Half-life:                {params['half_life']:.2f} days")
    print()

    # Also show in minute units for sanity check
    params_min = estimate_ou_parameters(spread, dt_minutes)
    print("OU Parameter Estimates (minute units):")
    print(f"  θ (mean-reversion speed): {params_min['theta']:.6f} per minute")
    print(f"  μ (long-term mean):       {params_min['mu']:.4f}")
    print(f"  σ (volatility):           {params_min['sigma']:.6f} per √minute")
    print(f"  Half-life:                {params_min['half_life']:.1f} minutes")
    print()

    # Summary stats
    print("Spread statistics:")
    print(f"  Mean:   {spread.mean():.4f}")
    print(f"  Std:    {spread.std():.4f}")
    print(f"  Min:    {spread.min():.4f}")
    print(f"  Max:    {spread.max():.4f}")


if __name__ == "__main__":
    main()
