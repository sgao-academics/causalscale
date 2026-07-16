"""
DAGMA: log-determinant acyclicity constraint.
Theoretically tighter than NOTEARS trace-exp, better convergence.
Bello et al., NeurIPS 2022.
"""
import numpy as np, torch, time


def dagma_linear(
    X: np.ndarray,
    lambda1: float = 0.02,
    lambda2: float = 0.005,
    lr: float = 0.002,
    T: int = 4,
    mu_init: float = 1e-3,
    mu_factor: float = 10.0,
    max_iter: int = 60,
    inner_iter: int = 200,
    device: str = "cuda",
    verbose: bool = False,
) -> np.ndarray:
    """
    DAGMA with log-det acyclicity: logdet(sI - W*W) - d*log(s).

    Args:
        X: (n, d) data matrix
        lambda1: L1 sparsity penalty
        lambda2: L2 penalty on W
        lr: learning rate
        T: number of outer loop iterations per mu
        mu_init: initial penalty weight
        mu_factor: how much to multiply mu each outer step
        max_iter: max outer iterations
        inner_iter: inner SGD iterations per outer
        device: 'cuda' or 'cpu'
        verbose: print progress

    Returns:
        W: (d, d) adjacency matrix
    """
    X_t = torch.tensor(X.astype(np.float32), device=device)
    n, d = X.shape

    # Covariance-based loss (DAGMA eq. 4)
    S = (X_t.T @ X_t) / n  # (d, d) covariance

    W = torch.zeros(d, d, requires_grad=True, device=device)

    def logdet_h(W, s=1.0):
        """h(W) = -logdet(sI - W*W) + d*log(s)"""
        M = s * torch.eye(d, device=device) - W * W
        # Use Cholesky for stable log-det
        try:
            L = torch.linalg.cholesky(M)
            return -2.0 * torch.sum(torch.log(torch.diag(L))) + d * np.log(s)
        except:
            # Fallback: use eigenvalues
            return float('inf')

    mu = mu_init
    opt = torch.optim.Adam([W], lr=lr)
    t0 = time.time()

    for outer in range(max_iter):
        # Inner optimization
        for _ in range(inner_iter):
            opt.zero_grad()
            # Reconstruction loss: tr((I-W)^T S (I-W))
            M = torch.eye(d, device=device) - W
            loss_recon = torch.trace(M.T @ S @ M)
            # L1 penalty
            loss_l1 = lambda1 * torch.sum(torch.abs(W))
            # L2 penalty
            loss_l2 = lambda2 * torch.sum(W * W)
            # DAG penalty
            h = logdet_h(W)
            loss = loss_recon + loss_l1 + loss_l2 + mu * h
            loss.backward()
            opt.step()

        # Post-hoc pruning: zero small weights, update mu
        with torch.no_grad():
            W.data[torch.abs(W.data) < 0.01] = 0.0

        h_val = logdet_h(W)
        if h_val <= 1e-8:
            break

        mu *= mu_factor
        mu = min(mu, 1e15)
        # Reset optimizer with smaller lr for next stage
        opt = torch.optim.Adam([W], lr=lr * 0.9)

        if verbose and outer % 10 == 0:
            print(f"  DAGMA outer {outer}: h={h_val:.2e}, mu={mu:.1e}, "
                  f"nonzero={int((torch.abs(W)>0.05).sum().item())}")

    W_np = W.detach().cpu().numpy()
    if verbose:
        n_edges = int(np.sum(np.abs(W_np) > 0.3))
        print(f"  DAGMA done: {n_edges} edges, {time.time()-t0:.1f}s")
    return W_np
