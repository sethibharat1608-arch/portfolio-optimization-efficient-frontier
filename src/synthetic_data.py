"""
synthetic_data.py
------------------
DEMO-ONLY utility. Generates realistic synthetic daily stock prices via
correlated Geometric Brownian Motion, calibrated to plausible sector-level
drift/volatility, so the full pipeline can be run and demonstrated without
live internet access.

This is NOT part of the "real" project pipeline -- in production/local use,
data_loader.py pulls real prices from yfinance instead. This file exists
purely so the repo's example notebooks/results can be regenerated offline.
"""

import numpy as np
import pandas as pd


def generate_synthetic_prices(tickers, start_date, end_date, benchmark_ticker=None, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    n_assets = len(tickers)

    # Plausible annualized drift/vol per sector archetype (rotates through tickers)
    drift_choices = [0.12, 0.10, 0.14, 0.08, 0.11, 0.09, 0.07, 0.06, 0.05, 0.13]
    vol_choices = [0.28, 0.24, 0.30, 0.20, 0.32, 0.18, 0.16, 0.22, 0.35, 0.19]

    annual_drift = np.array([drift_choices[i % len(drift_choices)] for i in range(n_assets)])
    annual_vol = np.array([vol_choices[i % len(vol_choices)] for i in range(n_assets)])

    daily_drift = annual_drift / 252
    daily_vol = annual_vol / np.sqrt(252)

    # Build a correlation structure (moderate positive correlation, like real equities)
    base_corr = 0.35
    corr_matrix = np.full((n_assets, n_assets), base_corr)
    np.fill_diagonal(corr_matrix, 1.0)
    # add some idiosyncratic cluster correlation
    L = np.linalg.cholesky(corr_matrix)

    uncorrelated_shocks = rng.standard_normal((n_days, n_assets))
    correlated_shocks = uncorrelated_shocks @ L.T

    log_returns = daily_drift + daily_vol * correlated_shocks
    log_prices = np.cumsum(log_returns, axis=0)
    prices = 100 * np.exp(log_prices)

    df = pd.DataFrame(prices, index=dates, columns=tickers)

    if benchmark_ticker:
        # Benchmark ~ average of assets with lower vol, like a diversified index
        bench_drift = 0.09 / 252
        bench_vol = 0.16 / np.sqrt(252)
        bench_shocks = 0.7 * correlated_shocks.mean(axis=1) + 0.3 * rng.standard_normal(n_days)
        bench_log_returns = bench_drift + bench_vol * bench_shocks
        bench_prices = 100 * np.exp(np.cumsum(bench_log_returns))
        df[benchmark_ticker] = bench_prices

    df.index.name = "Date"
    return df
