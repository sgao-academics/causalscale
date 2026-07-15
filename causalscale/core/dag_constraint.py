"""
DAG Constraints for Differentiable Causal Discovery.

Implements the NOTEARS acyclicity constraint:
    h(W) = tr(e^{W ⊙ W}) - d = 0

where W is a weighted adjacency matrix and h(W)=0 iff W is a DAG.

Reference: Zheng et al. (2018) "DAGs with NO TEARS"
"""

import numpy as np
import scipy.linalg as slin
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════
# PyTorch Autograd Implementation of trace(expm(W ⊙ W))
# ═══════════════════════════════════════════════════════════════════

class TraceExpm(torch.autograd.Function):
    """Differentiable trace of matrix exponential.

    Forward:  E = expm(input), f = trace(E)
    Backward: grad_input = grad_output * E^T
    """

    @staticmethod
    def forward(ctx, inp):
        E = slin.expm(inp.detach().cpu().numpy())
        f = np.trace(E)
        E = torch.from_numpy(E).to(inp.device)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=inp.dtype, device=inp.device)

    @staticmethod
    def backward(ctx, grad_output):
        (E,) = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input


trace_expm = TraceExpm.apply


# ═══════════════════════════════════════════════════════════════════
# DAG Constraint Functions
# ═══════════════════════════════════════════════════════════════════

def dag_constraint(W: torch.Tensor) -> float:
    """NOTEARS DAG constraint: h(W) = tr(e^{W ⊙ W}) - d.

    Args:
        W: (d, d) adjacency matrix

    Returns:
        h(W) value — should approach 0 for a DAG
    """
    M = W * W  # element-wise square
    return float(torch.trace(torch.linalg.matrix_exp(M)) - W.shape[0])


def note_ars_linear_h(W: np.ndarray) -> tuple:
    """Evaluate NOTEARS h(W) and gradient using NumPy/Scipy.

    Args:
        W: (d, d) adjacency matrix

    Returns:
        (h_val, G_h) where h_val = tr(e^{W⊙W}) - d, G_h = E^T * W * 2
    """
    d = W.shape[0]
    E = slin.expm(W * W)
    h = np.trace(E) - d
    G_h = E.T * W * 2
    return h, G_h


def is_dag(W: np.ndarray, threshold: float = 0.3) -> bool:
    """Check if thresholded adjacency is a DAG.

    Args:
        W: (d, d) adjacency matrix
        threshold: edge weight threshold

    Returns:
        True if the thresholded graph is a DAG
    """
    W_thresh = np.abs(W) > threshold
    # A graph is a DAG iff it can be topologically sorted
    # Quick check: no self-loops, check for cycles via powers
    d = W_thresh.shape[0]
    # If W_thresh is a DAG, W_thresh^k should be zero for k >= d
    power = np.eye(d, dtype=bool)
    total = np.zeros((d, d), dtype=bool)
    for _ in range(d):
        power = power @ W_thresh
        total = total | power
    # If there's a cycle, some diagonal will be True at some power
    return not np.any(np.diag(total))


# ═══════════════════════════════════════════════════════════════════
# NOTEARS Linear Solver (NumPy, for d < 200)
# ═══════════════════════════════════════════════════════════════════

def notears_linear(
    X: np.ndarray,
    lambda1: float = 0.1,
    loss_type: str = "l2",
    max_iter: int = 100,
    h_tol: float = 1e-8,
    w_threshold: float = 0.3,
) -> np.ndarray:
    """Solve min_W L(W;X) + lambda1 ||W||_1 s.t. h(W)=0.

    Augmented Lagrangian method for linear NOTEARS.

    Args:
        X: (n, d) sample matrix
        lambda1: L1 penalty
        loss_type: 'l2', 'logistic', or 'poisson'
        max_iter: dual ascent steps
        h_tol: convergence tolerance for DAG constraint
        w_threshold: edge threshold

    Returns:
        W_est: (d, d) estimated DAG adjacency
    """
    from scipy.optimize import minimize
    from scipy.special import expit as sigmoid

    n, d = X.shape

    def _adj(w):
        return (w[: d * d] - w[d * d :]).reshape([d, d])

    def _loss(W):
        M = X @ W
        if loss_type == "l2":
            R = X - M
            loss = 0.5 / n * (R**2).sum()
            G_loss = -1.0 / n * X.T @ R
        elif loss_type == "logistic":
            loss = 1.0 / n * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / n * X.T @ (sigmoid(M) - X)
        elif loss_type == "poisson":
            S = np.exp(M)
            loss = 1.0 / n * (S - X * M).sum()
            G_loss = 1.0 / n * X.T @ (S - X)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        return loss, G_loss

    w_est = np.zeros(2 * d * d)
    rho, alpha, h = 1.0, 0.0, np.inf
    bnds = [
        (0, 0) if i == j else (0, None)
        for _ in range(2)
        for i in range(d)
        for j in range(d)
    ]

    if loss_type == "l2":
        X = X - np.mean(X, axis=0, keepdims=True)

    for _ in range(max_iter):
        while rho < 1e16:

            def _func(w):
                W = _adj(w)
                loss, G_loss = _loss(W)
                hv, G_h = note_ars_linear_h(W)
                obj = loss + 0.5 * rho * hv * hv + alpha * hv + lambda1 * w.sum()
                G_smooth = G_loss + (rho * hv + alpha) * G_h
                g_obj = np.concatenate(
                    (G_smooth + lambda1, -G_smooth + lambda1), axis=None
                )
                return obj, g_obj

            sol = minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = note_ars_linear_h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if abs(h) <= h_tol or rho >= 1e16:
            break

    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est
