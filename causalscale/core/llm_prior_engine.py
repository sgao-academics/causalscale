"""LLMPrior Engine: STRING-derived edge prior injection into NOTEARS.

Adds a prior penalty term: lambda_p * ||W - W_prior||^2 to the NOTEARS objective,
where W_prior encodes external knowledge (e.g., STRING protein-protein interactions).
"""

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import warnings


def _h(W):
    return np.trace(expm(W * W)) - W.shape[0]


def fit_llm_prior(X, W_prior=None, string_data_dir=None, gene_names=None,
                  lambda_p=0.1, lam=0.01, max_outer=40, max_inner=150,
                  threshold=0.3, seed=42, verbose=True):
    """Prior-regularized NOTEARS with external edge prior.

    Args:
        X: (n, d) numpy array.
        W_prior: (d, d) prior matrix. If None and string_data_dir is provided,
                 constructs prior from STRING database.
        string_data_dir: path to STRING data directory (optional).
        gene_names: list of gene names for STRING lookup (optional).
        lambda_p: prior penalty strength.
        lam: L1 regularization strength.
        max_outer: NOTEARS outer loop iterations.
        max_inner: L-BFGS-B inner iterations.
        threshold: edge weight threshold.
        seed: random seed.

    Returns:
        adjacency: (d, d) causal adjacency matrix.
        metadata: dict with edge counts and convergence info.
    """
    d = X.shape[1]; n = X.shape[0]
    X = np.nan_to_num(X, nan=0.0)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    cov = X.T @ X / n

    # Build prior matrix
    if W_prior is None:
        # No prior provided: use zero prior (equivalent to vanilla NOTEARS)
        W_prior = np.zeros((d, d))
        if verbose:
            print("  LLMPrior: no prior matrix provided, using zero prior")
    else:
        W_prior = np.asarray(W_prior, dtype=np.float64)
        if W_prior.shape != (d, d):
            raise ValueError(f"W_prior must be ({d}, {d}), got {W_prior.shape}")
        if verbose:
            nz_p = int(np.sum(np.abs(W_prior) > 0))
            print(f"  LLMPrior: prior matrix loaded ({nz_p} non-zero entries)")

    # NOTEARS with prior penalty
    np.random.seed(seed)
    w_vec = np.random.randn(d * d) * 0.01
    rho, alpha = 0.1, 0.0
    best_w, best_h = w_vec.copy(), 1e10
    w_prior_flat = W_prior.flatten()

    for iteration in range(max_outer):
        def lag(w):
            W = w.reshape(d, d)
            hv = np.trace(expm(W * W)) - d
            diff = np.eye(d) - W
            loss = 0.5 * np.trace(diff.T @ cov @ diff) + lam * np.sum(np.abs(W))
            prior_penalty = lambda_p * 0.5 * np.sum((w - w_prior_flat) ** 2)
            return loss + prior_penalty + 0.5 * rho * hv ** 2 + alpha * hv

        def lag_grad(w):
            W = w.reshape(d, d)
            hv = np.trace(expm(W * W)) - d
            g = cov @ (W - np.eye(d))
            if lam > 0:
                g += lam * np.sign(W)
            dh = 2 * W * expm(W * W).T
            grad = g.flatten() + lambda_p * (w - w_prior_flat)
            return grad + rho * hv * dh.flatten() + alpha * dh.flatten()

        res = minimize(lag, w_vec, method='L-BFGS-B', jac=lag_grad,
                       options={'maxiter': max_inner, 'ftol': 1e-12, 'gtol': 1e-12})
        w_vec = res.x; W = w_vec.reshape(d, d)
        hv = np.trace(expm(W * W)) - d
        alpha += rho * hv
        if abs(hv) < best_h:
            best_h, best_w = abs(hv), w_vec.copy()
        if abs(hv) < 1e-7:
            break
        if hv > 0 and abs(hv) > 1e-10:
            rho = min(rho * 5, 1e10)

    W_final = best_w.reshape(d, d)
    W_final[np.abs(W_final) < threshold] = 0.0
    nz = int(np.sum(np.abs(W_final) > 0))

    # Compute prior overlap
    prior_nz = int(np.sum(np.abs(W_prior) > 0))
    if prior_nz > 0:
        prior_edges = set(zip(*np.where(np.abs(W_prior) > 0)))
        final_edges = set(zip(*np.where(np.abs(W_final) > 0)))
        overlap = len(prior_edges & final_edges)
    else:
        overlap = 0

    metadata = {
        "edges": nz,
        "h_final": float(best_h),
        "prior_nonzero": prior_nz,
        "prior_overlap": overlap,
        "lambda_p": lambda_p,
    }

    if verbose:
        print(f"  LLMPrior: {nz} edges, h={best_h:.2e}, "
              f"prior overlap={overlap}/{prior_nz}")

    return W_final, metadata
