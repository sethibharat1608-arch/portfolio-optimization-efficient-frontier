"""
backtest.py
-----------
Out-of-sample backtest of the MSR portfolio with monthly rebalancing.
Computes Sharpe ratio, max drawdown, annualized return, and annualized volatility.
"""

import numpy as np
import pandas as pd


def backtest_fixed_weights(monthly_returns: pd.DataFrame, weights: np.ndarray,
                            backtest_start: str) -> pd.Series:
    """
    Simulates a portfolio that is rebalanced back to fixed target `weights` at the
    start of every month over the out-of-sample period. Because we rebalance every
    period back to the same target weights, monthly portfolio return is simply the
    weighted average of asset returns each month.

    Returns a Series of cumulative growth of $1.
    """
    oos_returns = monthly_returns.loc[monthly_returns.index >= backtest_start]
    port_monthly_returns = oos_returns.values @ weights
    port_monthly_returns = pd.Series(port_monthly_returns, index=oos_returns.index)
    cumulative = (1 + port_monthly_returns).cumprod()
    return cumulative, port_monthly_returns


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """Computes the maximum peak-to-trough drawdown from a cumulative return series."""
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()


def performance_metrics(monthly_returns: pd.Series, risk_free_rate: float) -> dict:
    """Computes annualized return, annualized volatility, Sharpe ratio, and max drawdown."""
    ann_return = (1 + monthly_returns.mean()) ** 12 - 1
    ann_vol = monthly_returns.std() * np.sqrt(12)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan

    cumulative = (1 + monthly_returns).cumprod()
    mdd = max_drawdown(cumulative)

    return {
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": mdd,
    }


def summarize_performance(name: str, monthly_returns: pd.Series, risk_free_rate: float) -> str:
    metrics = performance_metrics(monthly_returns, risk_free_rate)
    lines = [f"--- {name} ---"]
    for k, v in metrics.items():
        if "Ratio" in k:
            lines.append(f"{k}: {v:.2f}")
        else:
            lines.append(f"{k}: {v:.2%}")
    return "\n".join(lines)
