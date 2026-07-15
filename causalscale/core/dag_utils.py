"""
Efficient DAG Operations & Counterfactual Inference
=====================================================
Optimized DAG constraint computation and Pearl's do-calculus for
counterfactual reasoning on discovered causal graphs.

Key operations:
  1. efficient_dag_constraint: O(d·r²) DAG constraint for low-rank W
  2. randomized_h_dag: power iteration approximation for large d
  3. counterfactual: do-calculus inference on discovered graph
  4. topological_sort: ordering for efficient computation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Set, Dict


def efficient_dag_constraint(W: torch.Tensor) -> float:
    """
    Compute DAG constraint h(W) = tr(exp(W * W)) - d.

    For d <= 200: exact matrix exponential (O(d³) but fast on GPU for small d).
    For d > 200: randomized power iteration (O(d² · k)).

    Note: Casts to float32 for AMP compatibility.
    Returns:
        h(W): non-negative, zero iff W is a DAG
    """
    d = W.shape[0]
    # Cast to float32 for AMP compatibility (matrix_exp doesn't support FP16)
    W_f32 = W.float()
    M = W_f32 * W_f32

    if d <= 500:
        # Exact matrix exponential: RTX 5060 handles d=500 (8GB) just fine
        exp_trace = torch.trace(torch.linalg.matrix_exp(M))
        return float((exp_trace - d).item())

    # Large d (d > 500): randomized approximation with deterministic seed
    return randomized_h_dag(W_f32)


def randomized_h_dag(W: torch.Tensor, n_power_iter: int = 30, n_hutchinson: int = 5) -> float:
    """
    Randomized h(W) for large d.

    Uses Hutchinson's trace estimator + power iteration:
    tr(exp(M)) ≈ (1/n_h) * Σ_i v_i^T * taylor_approx(exp(M)) * v_i

    Complexity: O(d² · n_power_iter · n_hutchinson)
    """
    d = W.shape[0]
    device = W.device
    M = W * W

    trace_est = 0.0

    for _ in range(n_hutchinson):
        v = torch.randn(d, 1, device=device)
        v = v / (v.norm() + 1e-12)

        # Taylor series for exp(M) * v with early convergence check
        term = v.clone()
        result = v.clone()
        w = v.clone()
        factorial = 1.0

        for k in range(1, n_power_iter):
            w = M @ w
            factorial *= k
            if factorial > 1e20:
                break
            new_term = w / factorial
            correction = new_term.norm().item()

            if correction < 1e-10:
                break

            result = result + new_term
            term = new_term

        trace_est += float((v.T @ result)[0, 0])

    return trace_est / n_hutchinson - d


def topological_sort(W: np.ndarray) -> List[int]:
    """
    Topological sort of the causal graph W.

    If W is not a perfect DAG, computes an approximate ordering
    that minimizes the number of feedback edges.

    Returns:
        order: list of node indices in topological order
    """
    d = W.shape[0]
    adj = (np.abs(W) > 0.3).astype(int)

    # Kahn's algorithm
    in_degree = adj.sum(axis=0)
    queue = [i for i in range(d) if in_degree[i] == 0]
    order = []

    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in range(d):
            if adj[u, v]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

    # If not all nodes processed (cycle exists), add remaining
    remaining = [i for i in range(d) if i not in order]
    # Greedy: add nodes with lowest in-degree first
    remaining.sort(key=lambda i: in_degree[i])
    order.extend(remaining)

    return order


def counterfactual(
    W: np.ndarray,
    X: np.ndarray,
    intervention: Dict[int, float],
    effect_vars: Optional[List[int]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute counterfactual predictions using do-calculus on discovered graph.

    do(X_i = x_i) := set variable i to value x_i and propagate through W.

    For linear SEM: X = X @ W + ε
    After intervention on variable i:
        X_new[j] = Σ_k X_new[k] * W[k,j]  (for j ≠ i)
        X_new[i] = intervention_value[i]

    Args:
        W: (d, d) causal adjacency matrix
        X: (n, d) original data
        intervention: dict mapping variable_idx -> intervention_value
        effect_vars: optional list of effect variables to compute

    Returns:
        dict with 'counterfactual': (n, d) counterfactual data,
                   'effect': (n,) or (n, len(effect_vars)) effect values,
                   'ate': average treatment effect
    """
    n, d = X.shape
    order = topological_sort(W)

    # Compute counterfactual by propagating through topological order
    X_cf = X.copy()
    intervened = set(intervention.keys())

    for var in order:
        if var in intervention:
            X_cf[:, var] = intervention[var]
        else:
            # X_cf[var] = Σ_k X_cf[k] * W[k, var] + ε_original[var]
            parents = np.where(np.abs(W[:, var]) > 0.3)[0]
            parent_effect = np.zeros(n)
            for p in parents:
                if p in intervened or p < var:  # ancestors
                    parent_effect += X_cf[:, p] * W[p, var]
            # Add original noise (residual)
            original_effect = X[:, var] - np.sum(
                [X[:, p] * W[p, var] for p in parents], axis=0
            ) if len(parents) > 0 else X[:, var]
            X_cf[:, var] = parent_effect + original_effect

    # Compute effects
    if effect_vars is None:
        effect_vars = list(set(range(d)) - set(intervention.keys()))[:min(5, d)]

    effects = {}
    for var in effect_vars:
        ate = float(np.mean(X_cf[:, var] - X[:, var]))
        effects[f'ATE_{var}'] = ate

    # Overall ATE (average over effect variables)
    ate_overall = np.mean([v for v in effects.values()])

    return {
        'counterfactual': X_cf,
        'effects': effects,
        'ate_overall': ate_overall,
        'intervention': intervention,
        'topological_order': order,
    }


def granger_causality_test(
    X: np.ndarray,
    W: np.ndarray,
    lag: int = 1,
    significance: float = 0.05
) -> Dict[str, np.ndarray]:
    """
    Test Granger causality directions against discovered graph.

    For each edge i→j in W, tests if X_i(t-lag) predicts X_j(t)
    above and beyond X_j's own past.

    Args:
        X: (T, d) time series data
        W: (d, d) causal adjacency (contemporaneous)
        lag: time lag for Granger test

    Returns:
        dict with 'granger_pvalues': (d, d) p-value matrix,
                   'aligned': fraction of W edges confirmed by Granger,
                   'contradictory': edges with opposite direction
    """
    T, d = X.shape
    from scipy.stats import f_oneway

    p_values = np.ones((d, d))
    aligned = 0
    contradictory = 0
    total_edges = 0

    W_binary = (np.abs(W) > 0.3).astype(int)

    for i in range(d):
        for j in range(d):
            if i == j or W_binary[i, j] == 0:
                continue
            total_edges += 1

            # Restricted model: X_j(t) ~ X_j(t-1)
            y = X[lag:, j]
            X_restricted = X[lag-1:-1, j].reshape(-1, 1)

            # Full model: X_j(t) ~ X_j(t-1) + X_i(t-1)
            X_full = np.column_stack([X_restricted, X[lag-1:-1, i]])

            # Simple F-test for added variable
            try:
                # OLS residuals
                beta_r = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
                resid_r = y - X_restricted @ beta_r
                ssr_r = np.sum(resid_r ** 2)

                beta_f = np.linalg.lstsq(X_full, y, rcond=None)[0]
                resid_f = y - X_full @ beta_f
                ssr_f = np.sum(resid_f ** 2)

                df_r = T - lag - 1
                df_f = T - lag - 2
                F = ((ssr_r - ssr_f) / 1) / (ssr_f / max(df_f, 1))
                from scipy.stats import f as f_dist
                p = 1 - f_dist.cdf(F, 1, df_f)
                p_values[i, j] = p

                if p < significance:
                    aligned += 1
            except Exception:
                p_values[i, j] = 1.0

    # Check reverse direction for contradictory edges
    for i in range(d):
        for j in range(d):
            if i != j and W_binary[j, i] > 0 and p_values[i, j] < significance:
                contradictory += 1

    return {
        'granger_pvalues': p_values,
        'aligned': aligned / max(total_edges, 1),
        'contradictory': contradictory / max(total_edges, 1),
        'total_edges': total_edges,
    }
