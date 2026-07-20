"""
visualization.py
-----------------
Plots for the efficient frontier (with GMV, MSR, and the Capital Market Line)
and for cumulative backtest returns vs a benchmark.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_efficient_frontier(frontier_returns, frontier_vols, gmv_point, msr_point,
                             risk_free_rate, save_path=None):
    """
    gmv_point: (vol, return) tuple for the GMV portfolio
    msr_point: (vol, return) tuple for the MSR (tangency) portfolio
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Efficient frontier curve
    ax.plot(frontier_vols, frontier_returns, "b-", linewidth=2, label="Efficient Frontier")

    # GMV portfolio
    ax.scatter(*gmv_point[::-1] if False else (gmv_point[0], gmv_point[1]),
               color="green", marker="o", s=120, label="Global Min Variance", zorder=5)

    # MSR portfolio
    ax.scatter(msr_point[0], msr_point[1], color="red", marker="*", s=250,
               label="Max Sharpe Ratio", zorder=5)

    # Capital Market Line: from risk-free rate through the MSR portfolio
    cml_x = np.linspace(0, max(frontier_vols) * 1.2, 50)
    msr_sharpe = (msr_point[1] - risk_free_rate) / msr_point[0]
    cml_y = risk_free_rate + msr_sharpe * cml_x
    ax.plot(cml_x, cml_y, "k--", linewidth=1.5, label="Capital Market Line")

    ax.scatter(0, risk_free_rate, color="black", marker="D", s=80, label="Risk-Free Rate", zorder=5)

    ax.set_xlabel("Annualized Volatility (Risk)")
    ax.set_ylabel("Annualized Expected Return")
    ax.set_title("Efficient Frontier with Capital Market Line")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    return fig


def plot_cumulative_returns(portfolio_cum_returns, benchmark_cum_returns, save_path=None):
    """
    portfolio_cum_returns, benchmark_cum_returns: pandas Series indexed by date,
    representing cumulative growth of $1 invested.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(portfolio_cum_returns.index, portfolio_cum_returns.values,
            label="MSR Portfolio (Monthly Rebalanced)", linewidth=2, color="darkblue")
    ax.plot(benchmark_cum_returns.index, benchmark_cum_returns.values,
            label="Benchmark (S&P 500)", linewidth=2, color="gray", linestyle="--")

    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.set_title("Out-of-Sample Backtest: Cumulative Returns")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    return fig
