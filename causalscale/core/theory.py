"""
Theory Engine: Convergence Diagnostics & Statistical Inference
================================================================
Provides theoretical guarantees and diagnostics for causal discovery:

1. Convergence Diagnostics: monitors optimization progress, detects issues
2. Identifiability Checks: verifies conditions for unique causal discovery
3. Sample Complexity Estimates: predicts required n for given d and sparsity
4. Edge Significance Tests: statistical tests for individual edges

All diagnostics are computed post-hoc from training data and results.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm, chi2
import warnings


def convergence_diagnostic(
    loss_history: List[float],
    h_history: List[float],
    grad_norm_history: Optional[List[float]] = None,
    rho_history: Optional[List[float]] = None
) -> Dict[str, any]:
    """
    Comprehensive convergence diagnostic.

    Checks:
      1. Loss convergence (monotonic decrease?)
      2. DAG constraint satisfaction (h(W) → 0?)
      3. Gradient norm decay
      4. Augmented Lagrangian rho stability

    Returns:
        dict with 'converged': bool, 'issues': list of warnings,
                   'convergence_rate': estimated rate,
                   'final_h': final DAG constraint value
    """
    issues = []
    n = len(loss_history)

    if n < 2:
        return {
            'converged': False,
            'issues': ['Insufficient history (<2 points)'],
            'convergence_rate': None,
            'final_h': h_history[-1] if h_history else None,
        }

    # 1. Check loss convergence
    loss_initial = loss_history[0]
    loss_final = loss_history[-1]
    loss_reduction = (loss_initial - loss_final) / max(loss_initial, 1e-10)

    if loss_reduction < 0.1:
        issues.append(f'Loss barely decreased ({loss_reduction:.1%})')
    elif loss_reduction < 0.5:
        issues.append(f'Loss decreased moderately ({loss_reduction:.1%})')

    # Check for oscillations (variance in tail)
    tail = loss_history[-max(n // 5, 3):]
    tail_std = np.std(tail)
    tail_mean = np.mean(tail)
    if tail_mean > 0 and tail_std / tail_mean > 0.1:
        issues.append(f'Loss oscillating in tail (CV={tail_std/tail_mean:.2f})')

    # Check for divergence
    if loss_final > loss_initial * 2:
        issues.append('Loss DIVERGED (final > 2x initial)')

    # 2. DAG constraint
    final_h = h_history[-1] if h_history else None
    if final_h is not None:
        if final_h > 1.0:
            issues.append(f'DAG constraint NOT satisfied: h(W)={final_h:.2e}')
        elif final_h > 1e-4:
            issues.append(f'DAG constraint partially satisfied: h(W)={final_h:.2e}')

    # 3. Gradient norm
    if grad_norm_history:
        final_grad = grad_norm_history[-1]
        initial_grad = grad_norm_history[0]
        if final_grad > initial_grad * 0.1 and final_grad > 0.01:
            issues.append(f'Gradient not decaying: |g|={final_grad:.4f}')

    # 4. Estimate convergence rate (linear or sublinear)
    if n >= 4 and loss_history[-1] > 0:
        # Fit: log(loss_t - loss_min) = a + b * log(t)
        losses = np.array(loss_history)
        loss_min = losses.min()
        adjusted = losses - loss_min + 1e-10
        t = np.arange(1, n + 1)
        try:
            b, a = np.polyfit(np.log(t[-n//2:]), np.log(adjusted[-n//2:]), 1)
            rate = -b  # negative of slope = convergence rate
        except Exception:
            rate = None
    else:
        rate = None

    # Overall verdict
    converged = len(issues) == 0 or (len(issues) == 1 and 'partially' in issues[0])

    return {
        'converged': converged,
        'issues': issues,
        'convergence_rate': round(rate, 3) if rate else None,
        'final_h': final_h,
        'loss_reduction': round(loss_reduction, 4),
        'n_epochs': n,
    }


def edge_significance_test(
    W: np.ndarray,
    n_samples: int,
    d_variables: int,
    alpha: float = 0.05,
    bonferroni: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute statistical significance for each edge.

    Under the null hypothesis (no edge), estimated weights are approximately
    normally distributed with zero mean and variance ≈ 1/(n·d).

    Uses asymptotic normality of M-estimators for the NOTEARS loss.

    Args:
        W: (d, d) estimated adjacency matrix
        n_samples: number of observations
        d_variables: number of variables
        alpha: significance level
        bonferroni: apply Bonferroni correction for multiple testing

    Returns:
        dict with 'z_scores': (d,d), 'p_values': (d,d),
                   'significant': (d,d) boolean, 'n_significant': int
    """
    d = W.shape[0]

    # Asymptotic variance estimate (inverse Fisher information for NOTEARS)
    # For linear SEM with Gaussian noise: Var(W_ij) ≈ σ² / (n·d)
    # Conservative estimate
    se = np.sqrt(1.0 / (n_samples * max(d_variables, 1)))
    se = np.maximum(se, 1e-10)

    # Z-scores
    z_scores = W / se

    # Two-sided p-values
    p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

    # Significance threshold
    n_tests = d * (d - 1)  # off-diagonal only
    threshold = alpha / n_tests if bonferroni else alpha

    significant = p_values < threshold
    # Ensure diagonal is not significant
    np.fill_diagonal(significant, False)
    np.fill_diagonal(p_values, 1.0)

    return {
        'z_scores': z_scores,
        'p_values': p_values,
        'significant': significant,
        'n_significant': int(np.sum(significant)),
        'threshold': threshold,
        'bonferroni': bonferroni,
    }


def sample_complexity_estimate(
    d: int,
    expected_sparsity: float = 0.05,
    target_error: float = 0.1,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Estimate required sample size for reliable causal discovery.

    Based on:
    - NOTEARS requires n ≥ d for identifiability
    - Low-rank structure reduces this to n ≥ O(r·log(d))
    - Sparse structure: n ≥ O(s·log(d))
    where s = expected number of edges = sparsity · d²

    Args:
        d: number of variables
        expected_sparsity: fraction of possible edges (default: 0.05)
        target_error: desired edge weight estimation error
        confidence: confidence level

    Returns:
        dict with recommended sample sizes for different scenarios
    """
    z_score = norm.ppf(1 - (1 - confidence) / 2)
    s = expected_sparsity * d * (d - 1)  # expected number of edges

    # 1. Identifiability minimum: n ≥ d
    n_identifiability = d

    # 2. Sparse recovery: n ≥ C · s · log(d)  (from compressive sensing)
    n_sparse = max(d, int(10 * s * np.log(d) / (target_error ** 2)))

    # 3. Edge-level precision: n ≥ z² / (2·ε²) for each edge
    n_precision = max(d, int(z_score ** 2 / (2 * target_error ** 2)))

    # 4. Low-rank recovery: n ≥ C · r · log(d) · max(d/n, 1)
    # approximating rank r ≈ sqrt(s)
    r_approx = max(1, int(np.sqrt(s)))
    n_lowrank = max(d, int(5 * r_approx * np.log(d)))

    # Recommended: maximum of all requirements
    n_recommended = max(n_identifiability, n_sparse, n_precision, n_lowrank)

    return {
        'n_identifiability': n_identifiability,
        'n_sparse_recovery': n_sparse,
        'n_precision': n_precision,
        'n_lowrank': n_lowrank,
        'n_recommended': n_recommended,
        'expected_edges': s,
        'approx_rank': r_approx,
        'd': d,
    }


def structural_identifiability_check(
    W: np.ndarray,
    noise_var: Optional[np.ndarray] = None
) -> Dict[str, bool]:
    """
    Check conditions for structural identifiability of the causal graph.

    For linear SEM with non-Gaussian noise (LiNGAM) or equal variance:
    - No feedback cycles (DAG)
    - Sufficient excitation (well-conditioned)
    - Faithfulness (no exact cancellations)

    Args:
        W: (d, d) adjacency matrix
        noise_var: optional (d,) noise variances

    Returns:
        dict with checklist of identifiability conditions
    """
    d = W.shape[0]

    # 1. DAG check: compute if any cycles exist
    adj = (np.abs(W) > 0.3).astype(int)
    # Check via matrix powers: no cycles iff (I-A)^(-1) exists and is finite
    is_dag = True
    try:
        M = np.eye(d) - adj
        if np.linalg.matrix_rank(M) < d:
            is_dag = False
        # Power method: if spectral radius of adj < 1, it's a DAG
        eigs = np.linalg.eigvals(adj.astype(float))
        if np.max(np.abs(eigs)) > 0.99:
            is_dag = False
    except Exception:
        is_dag = False

    # 2. Well-conditioned: check condition number of I-W
    try:
        cond = np.linalg.cond(np.eye(d) - W)
        well_conditioned = cond < 1000
    except Exception:
        well_conditioned = False

    # 3. Faithfulness heuristic: no near-zero entries in W
    w_flat = np.abs(W[np.triu_indices(d, 1)])
    if len(w_flat) > 0:
        min_edge = w_flat[w_flat > 0].min() if np.any(w_flat > 0) else 0
        faithful = min_edge > 0.01
    else:
        faithful = True

    # 4. Coverage: all variables have at least one connection
    has_edge = np.any(adj, axis=0) | np.any(adj, axis=1)
    full_coverage = np.all(has_edge)

    return {
        'is_dag': is_dag,
        'well_conditioned': well_conditioned,
        'faithful': faithful,
        'full_coverage': full_coverage,
        'all_identified': is_dag and well_conditioned,
    }


def report_edge_quality(
    W: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    p_values: Optional[np.ndarray] = None,
    names: Optional[List[str]] = None
) -> str:
    """
    Generate a human-readable report of top edges with quality metrics.

    Args:
        W: (d, d) adjacency
        confidence: (d, d) confidence scores (optional)
        p_values: (d, d) p-values (optional)
        names: variable names

    Returns:
        Multi-line string report
    """
    d = W.shape[0]
    edges = []

    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            w = W[i, j]
            if abs(w) < 0.1:
                continue
            conf = confidence[i, j] if confidence is not None else None
            pval = p_values[i, j] if p_values is not None else None

            si = names[i] if names else f"V{i}"
            sj = names[j] if names else f"V{j}"
            edges.append((abs(w), w, si, sj, conf, pval))

    edges.sort(key=lambda x: -x[0])

    lines = [
        f"{'Source':>12s} → {'Target':<12s} {'Weight':>8s} {'Confidence':>10s} {'p-value':>10s} {'Significant':>12s}",
        "-" * 75,
    ]

    for _, w, si, sj, conf, pval in edges[:30]:
        conf_str = f"{conf:.3f}" if conf is not None else "N/A"
        pval_str = f"{pval:.2e}" if pval is not None else "N/A"
        sig_str = "✓" if (pval is not None and pval < 0.05) else " "
        lines.append(f"{si:>12s} → {sj:<12s} {w:>+8.4f} {conf_str:>10s} {pval_str:>10s} {sig_str:>12s}")

    lines.append(f"\nTotal edges: {len(edges)}")
    if confidence is not None:
        n_high = int(np.sum(confidence > 0.8))
        lines.append(f"High confidence (≥0.8): {n_high}")

    return "\n".join(lines)
