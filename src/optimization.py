"""
optimization.py
----------------
Mean-variance portfolio optimization:
  - Global Minimum Variance (GMV) portfolio
  - Maximum Sharpe Ratio (MSR) portfolio
  - Efficient Frontier (a series of minimum-variance portfolios across target returns)

Constraints: weights sum to 1, no short selling (long-only), consistent with the
"intermediate scope" of this project (no leverage, no sector constraints).
"""

import numpy as np
import cvxpy as cp


def gmv_portfolio(cov_matrix: np.ndarray):
    """Solves for the Global Minimum Variance portfolio weights."""
    n = cov_matrix.shape[0]
    w = cp.Variable(n)
    risk = cp.quad_form(w, cov_matrix)
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve()
    return w.value


def min_variance_for_target_return(mean_returns: np.ndarray, cov_matrix: np.ndarray, target_return: float):
    """Solves for minimum-variance weights subject to hitting a target expected return."""
    n = cov_matrix.shape[0]
    w = cp.Variable(n)
    risk = cp.quad_form(w, cov_matrix)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        mean_returns @ w == target_return,
    ]
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve()
    if w.value is None:
        return None
    return w.value


def efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray, n_points: int = 30):
    """
    Traces the efficient frontier by solving min-variance problems across a range
    of target returns spanning the achievable return range.
    Returns (frontier_returns, frontier_vols, frontier_weights).
    """
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_points)
    frontier_returns, frontier_vols, frontier_weights = [], [], []

    for target in target_returns:
        w = min_variance_for_target_return(mean_returns, cov_matrix, target)
        if w is None:
            continue
        port_return = mean_returns @ w
        port_vol = np.sqrt(w @ cov_matrix @ w)
        frontier_returns.append(port_return)
        frontier_vols.append(port_vol)
        frontier_weights.append(w)

    return np.array(frontier_returns), np.array(frontier_vols), frontier_weights


def max_sharpe_portfolio(mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float,
                          n_search_points: int = 200):
    """
    Finds the Maximum Sharpe Ratio (tangency) portfolio.

    cvxpy can't directly maximize a ratio under a QP, so we search over the
    efficient frontier (traced at fine resolution) and pick the point with the
    highest Sharpe ratio. This is a standard, interview-safe approach.
    """
    frontier_returns, frontier_vols, frontier_weights = efficient_frontier(
        mean_returns, cov_matrix, n_points=n_search_points
    )
    sharpe_ratios = (frontier_returns - risk_free_rate) / frontier_vols
    best_idx = np.argmax(sharpe_ratios)
    return frontier_weights[best_idx], frontier_returns[best_idx], frontier_vols[best_idx], sharpe_ratios[best_idx]


def portfolio_performance(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray):
    """Returns (expected annual return, annual volatility) for a given weight vector."""
    ret = mean_returns @ weights
    vol = np.sqrt(weights @ cov_matrix @ weights)
    return ret, vol
