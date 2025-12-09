"""
Fetch BTC/ETH minutely data and estimate OU parameters.
"""

import time

import numpy as np
import pandas as pd
import requests


def fetch_binance_klines(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch klines (candlestick) data from Binance."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
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
