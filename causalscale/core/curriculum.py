"""
Curriculum Learning for Causal Discovery.
Progressively trains from easy (small d, strong signal) to hard (target d).
Inspired by Knowledge-Informed Pretrained Model (Xu et al., 2026).
"""
import numpy as np, torch, time
from typing import Optional, Dict, List


def curriculum_notears(
    X: np.ndarray,
    target_d: int,
    curriculum_steps: int = 3,
    device: str = "cuda",
    verbose: bool = True,
) -> np.ndarray:
    """Train NOTEARS with curriculum: start at small d, expand to target.

    At each step:
    1. Select top-k variables by variance (k = d_start * (step+1))
    2. Run NOTEARS on the subset
    3. Use learned W as initialization for next step (padding with zeros)

    Args:
        X: (n, d) data matrix (d >= target_d)
        target_d: final number of variables
        curriculum_steps: number of progressive expansions
        device: 'cuda' or 'cpu'
        verbose: print progress

    Returns:
        W: (target_d, target_d) adjacency matrix
    """
    d_full = X.shape[1]
    d_start = max(10, target_d // (2 ** (curriculum_steps - 1)))

    if d_start >= target_d:
        d_start = target_d

    # Select genes by variance
    variances = X.var(axis=0)
    top_idx = np.argsort(-variances)[:target_d]
    X_target = X[:, top_idx]

    # Curriculum loop
    d_current = d_start
    W_current = None

    for step in range(curriculum_steps):
        d_next = min(target_d, d_start * (2 ** (step + 1) - 1) // (2 - 1))
        d_next = max(d_current + 1, d_next)  # ensure growth
        if d_next > target_d:
            d_next = target_d

        if verbose:
            print(f"  Curriculum step {step+1}/{curriculum_steps}: "
                  f"d={d_current} -> {d_next}")

        # Run NOTEARS on current subset
        X_sub = X_target[:, :d_next]
        W_sub = _run_notears_curriculum(
            X_sub, d_current, W_current, device=device
        )

        W_current = W_sub
        d_current = d_next

        if d_current >= target_d:
            break

    return W_current


def _run_notears_curriculum(
    X: np.ndarray,
    d_prev: int,
    W_init: Optional[np.ndarray],
    device: str = "cuda",
    outer: int = 25,
    inner: int = 200,
    lr: float = 0.002,
) -> np.ndarray:
    """Run NOTEARS with optional warm-start from previous curriculum step.

    If W_init is provided (from smaller d), it's padded with zeros
    to match the current d, giving the optimizer a better starting point.
    """
    n, d = X.shape
    X_t = torch.tensor(X.astype(np.float32), device=device)

    W = torch.zeros(d, d, requires_grad=True, device=device)

    # Warm-start: copy previous W into top-left corner
    if W_init is not None and d_prev < d:
        with torch.no_grad():
            W.data[:d_prev, :d_prev] = torch.tensor(
                W_init[:d_prev, :d_prev].astype(np.float32), device=device
            )

    rho, alpha = 1.0, 0.0
    opt = torch.optim.Adam([W], lr=lr)
    h_prev = float('inf')

    for o in range(outer):
        for _ in range(inner):
            opt.zero_grad()
            M = torch.eye(d, device=device) - W
            sq = (X_t @ M.T).pow(2)
            loss = sq.mean()
            h_val = torch.trace(torch.linalg.matrix_exp(W * W)) - d
            loss = loss + 0.5 * rho * h_val * h_val + alpha * h_val
            loss.backward()
            opt.step()

        with torch.no_grad():
            h_val = torch.trace(torch.linalg.matrix_exp(W * W)) - d
        h_curr = h_val.item()

        if o > 0 and h_curr > 0.25 * h_prev:
            rho *= 10
        else:
            alpha += rho * h_curr
        h_prev = h_curr

        if h_curr < 1e-8:
            break

    return W.detach().cpu().numpy()
