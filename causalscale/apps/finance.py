"""Financial Time-Series Causal Discovery.

Discover causal relationships in financial data (returns, volatility, sectors).
"""

import numpy as np
from typing import Dict, List, Optional
from ..core.lowrank import train_lowrank_gnn


def finance_causal_graph(
    returns: np.ndarray,
    tickers: Optional[List[str]] = None,
    method: str = "lowrank",
    rank: int = 32,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """Discover causal relationships in financial time series.

    Args:
        returns: (n_timesteps, d_assets) return matrix
        tickers: asset ticker symbols
        method: 'lowrank' (scalable) or 'granger' (classical, d <= 30)
        rank: factorization rank for lowrank method
        device: 'cuda' or 'cpu'
        verbose: print progress

    Returns:
        dict with edges, adjacency, summary
    """
    n, d = returns.shape
    if tickers is None:
        tickers = [f"A{i}" for i in range(d)]

    if method == "lowrank":
        result = train_lowrank_gnn(
            returns, rank=rank, device=device, verbose=verbose
        )
        W = result["adjacency"]
    elif method == "granger":
        # Granger causality via VAR + F-test
        from scipy import stats as sp_stats

        W = np.zeros((d, d))
        for i in range(d):
            y = returns[1:, i]
            for j in range(d):
                if i == j:
                    continue
                X = np.column_stack(
                    [returns[:-1, i], returns[:-1, j], np.ones(n - 1)]
                )
                # Restricted model (no j)
                X_r = np.column_stack([returns[:-1, i], np.ones(n - 1)])
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
                resid = y - X @ beta
                resid_r = y - X_r @ beta_r
                rss = (resid**2).sum()
                rss_r = (resid_r**2).sum()
                if rss < rss_r and rss > 0:
                    F = ((rss_r - rss) / 1) / (rss / (n - 3))
                    p = 1 - sp_stats.f.cdf(F, 1, n - 3)
                    if p < 0.05:
                        W[j, i] = beta[1]
    else:
        raise ValueError(f"Unknown method: {method}")

    # Extract edges
    threshold = 0.2
    edges = []
    for i in range(d):
        for j in range(d):
            if i != j and abs(W[i, j]) > threshold:
                edges.append(
                    {
                        "source": tickers[i],
                        "target": tickers[j],
                        "weight": float(W[i, j]),
                    }
                )
    edges.sort(key=lambda x: -abs(x["weight"]))

    # Sector influence (out-degree)
    out_degree = np.sum(np.abs(W) > threshold, axis=0)
    influencers = [
        {"ticker": tickers[int(i)], "influence": int(out_degree[int(i)])}
        for i in np.argsort(-out_degree)[:10]
    ]

    return {
        "edges": edges,
        "adjacency": W,
        "n_edges": len(edges),
        "top_influencers": influencers,
        "method": method,
    }
