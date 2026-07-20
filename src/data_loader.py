"""
data_loader.py
---------------
Fetches historical daily adjusted-close prices for a universe of stocks
using yfinance, and computes monthly returns + covariance matrix.

Usage:
    python src/data_loader.py
"""

import os
import yaml
import pandas as pd
import numpy as np
import yfinance as yf

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_config():
    with open(os.path.join(CONFIG_DIR, "tickers.yaml")) as f:
        tickers_cfg = yaml.safe_load(f)
    with open(os.path.join(CONFIG_DIR, "params.yaml")) as f:
        params_cfg = yaml.safe_load(f)
    return tickers_cfg["tickers"], params_cfg


def fetch_prices(tickers, start_date, end_date, benchmark_ticker=None):
    """
    Downloads daily adjusted close prices for the given tickers (and
    optionally a benchmark) from Yahoo Finance via yfinance.
    """
    all_tickers = list(tickers)
    if benchmark_ticker and benchmark_ticker not in all_tickers:
        all_tickers = all_tickers + [benchmark_ticker]

    print(f"Downloading {len(all_tickers)} tickers from {start_date} to {end_date}...")
    raw = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)

    # yfinance returns a MultiIndex column DataFrame when multiple tickers are requested
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = all_tickers

    prices = prices.dropna(how="all").ffill().dropna()
    return prices


def compute_monthly_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Resamples daily prices to month-end and computes simple monthly returns."""
    monthly_prices = daily_prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()
    return monthly_returns


def compute_stats(monthly_returns: pd.DataFrame):
    """Returns annualized mean returns vector and annualized covariance matrix."""
    mean_returns_annual = monthly_returns.mean() * 12
    cov_matrix_annual = monthly_returns.cov() * 12
    return mean_returns_annual, cov_matrix_annual


def main():
    tickers, params = load_config()
    prices = fetch_prices(
        tickers,
        params["start_date"],
        params["end_date"],
        benchmark_ticker=params.get("benchmark_ticker"),
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "stock_prices.csv")
    prices.to_csv(out_path)
    print(f"Saved daily prices to {out_path}")

    monthly_returns = compute_monthly_returns(prices[tickers])
    mean_returns, cov_matrix = compute_stats(monthly_returns)

    print("\nAnnualized mean returns:")
    print(mean_returns.round(4))
    print("\nAnnualized covariance matrix (head):")
    print(cov_matrix.round(4).iloc[:5, :5])


if __name__ == "__main__":
    main()
