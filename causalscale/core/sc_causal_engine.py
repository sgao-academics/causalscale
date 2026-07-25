"""scCausal Engine: Negative binomial likelihood ratio CI testing for single-cell RNA-seq.

Implements a PC-algorithm skeleton discovery with NB-LR conditional independence
test, replacing Fisher's z-test. Operates on raw UMI counts without normalization.
The engine provides a distribution-matched alternative to Gaussian-based testing
for zero-inflated, overdispersed single-cell data.
"""

import numpy as np
import warnings
from scipy import stats as sp_stats

try:
    import statsmodels.api as sm
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


def _nb_lr_test(X, i, j, cond_set, alpha=0.05):
    """Negative Binomial Likelihood Ratio conditional independence test.

    Tests H0: X_i ⟂ X_j | S  using NB GLM regression.
    Returns (is_independent, p_value, test_statistic).
    """
    n = X.shape[0]
    y = X[:, i].astype(np.float64)
    x_j = X[:, j].astype(np.float64).reshape(-1, 1)

    # Build design matrix with conditioning set
    if cond_set:
        X_cond = X[:, list(cond_set)].astype(np.float64)
        X_full = np.column_stack([X_cond, x_j])
        X_reduced = X_cond
        df_diff = 1
    else:
        X_full = x_j
        X_reduced = np.ones((n, 1))
        df_diff = 1

    try:
        # Full model: y ~ X_full with NB
        model_full = sm.GLM(y, sm.add_constant(X_full),
                            family=sm.families.NegativeBinomial())
        result_full = model_full.fit(disp=0, maxiter=100)

        # Reduced model: y ~ X_reduced
        model_reduced = sm.GLM(y, sm.add_constant(X_reduced),
                               family=sm.families.NegativeBinomial())
        result_reduced = model_reduced.fit(disp=0, maxiter=100)

        # Likelihood ratio test
        ll_full = result_full.llf
        ll_reduced = result_reduced.llf
        lr_stat = 2 * (ll_full - ll_reduced)

        if lr_stat < 0 or np.isnan(lr_stat):
            lr_stat = 0.0

        if lr_stat > 0:
            p_val = 1.0 - sp_stats.chi2.cdf(lr_stat, df_diff)
        else:
            p_val = 1.0

        is_indep = p_val > alpha
        return is_indep, p_val, lr_stat
    except Exception:
        # Fallback: edge not independent (conservative)
        return False, 0.0, float('nan')


def _pc_skeleton(X, alpha=0.05, max_cond=1, verbose=True):
    """PC algorithm skeleton discovery with NB-LR CI test.

    Args:
        X: (n, d) raw count matrix.
        alpha: significance level for CI test.
        max_cond: maximum conditioning set size (default 1 for scRNA-seq).
        verbose: print progress.

    Returns:
        adjacency: (d, d) undirected skeleton adjacency (1 = edge, 0 = no edge).
        edges: list of (i, j) tuples.
        p_values: dict of {(i,j): p_value}.
    """
    d = X.shape[1]
    adjacency = np.ones((d, d), dtype=np.int8) - np.eye(d, dtype=np.int8)
    p_values = {}
    edges = [(i, j) for i in range(d) for j in range(i + 1, d)]

    n_tests = 0
    n_rejected = 0

    for depth in range(max_cond + 1):
        if verbose:
            print(f"  scCausal PC: depth={depth}, edges={len(edges)}")

        new_edges = []
        for i, j in edges:
            # Get neighbors for conditioning
            neighbors_i = set(np.where(adjacency[i] == 1)[0]) - {j}
            neighbors_j = set(np.where(adjacency[j] == 1)[0]) - {i}
            candidates = sorted(neighbors_i | neighbors_j)

            if len(candidates) < depth:
                new_edges.append((i, j))
                continue

            # Test all conditioning sets of given depth
            is_dependent = False
            min_p = 1.0
            from itertools import combinations as combs

            for cond in combs(candidates, depth):
                n_tests += 1
                is_indep, p_val, _ = _nb_lr_test(X, i, j, cond, alpha)
                min_p = min(min_p, p_val)
                if is_indep:
                    break
                is_dependent = True

            p_values[(i, j)] = min_p
            p_values[(j, i)] = min_p

            if is_dependent or depth == 0:
                new_edges.append((i, j))
            else:
                adjacency[i, j] = 0
                adjacency[j, i] = 0
                n_rejected += 1

        edges = new_edges
        if len(edges) == 0:
            break

    if verbose:
        print(f"  scCausal: {len(edges)} edges retained, {n_tests} tests, "
              f"{n_rejected} rejected")

    return adjacency, edges, p_values


def fit_sc_causal(X, alpha=0.05, max_cond=1, verbose=True):
    """scCausal skeleton discovery for single-cell RNA-seq data.

    Args:
        X: (n, d) raw UMI count matrix (no normalization needed).
        alpha: significance level for NB-LR CI test.
        max_cond: maximum conditioning set size.
        verbose: print progress.

    Returns:
        adjacency: (d, d) skeleton adjacency.
        metadata: dict with edge counts, validation info.
    """
    if not _HAS_STATSMODELS:
        raise ImportError(
            "scCausal requires statsmodels. Install with: pip install statsmodels")

    X = np.nan_to_num(X, nan=0.0)
    # Raw counts - no normalization, no log transform
    d = X.shape[1]
    n = X.shape[0]

    if verbose:
        print(f"  scCausal: n={n}, d={d}, alpha={alpha}, max_cond={max_cond}")

    adjacency, edges, p_values = _pc_skeleton(X, alpha=alpha, max_cond=max_cond,
                                              verbose=verbose)

    W = adjacency.astype(np.float64)  # undirected for skeleton
    nz = len(edges)

    metadata = {
        "edges": nz,
        "method": "NB-LR PC skeleton",
        "alpha": alpha,
        "max_cond": max_cond,
        "is_dag": False,  # skeleton is undirected
    }

    return W, metadata
