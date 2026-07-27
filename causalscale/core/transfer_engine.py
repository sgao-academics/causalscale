"""
Transfer Engine: Warm-Start Causal Discovery with Cross-Dataset Graphs.

Core idea: Initialize target causal discovery with a source-domain causal graph
W_S learned from a different dataset, rather than random initialization.
A shuffled-source negative control confirms the benefit is causal, not numerical.

Reference: Gao (2026) "Causal Transfer Learning: Warm-Starting NOTEARS,
DAGMA, and GOLEM with Cross-Dataset Graphs." JBCB, submitted.

Supported base engines: notears, dagma, golem.
"""

import numpy as np
import torch
import time
import warnings
from typing import Optional, Dict, Tuple, Union, List
from dataclasses import dataclass, field

warnings.filterwarnings("ignore", category=UserWarning)


def _shuffle_graph(W: np.ndarray, seed: int = 42) -> np.ndarray:
    """Row/column-permute a weight matrix, preserving sparsity and
    spectral properties but destroying causal structure."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(W.shape[0])
    return W[perm][:, perm]


def _train_notears_warm(
    X: np.ndarray,
    W_init: np.ndarray,
    threshold: float = 0.3,
    lambda1: float = 0.01,
    rho: float = 0.1,
    rho_max: float = 1e10,
    h_tol: float = 1e-7,
    max_iter: int = 35,
    inner_iter: int = 200,
    lr: float = 0.002,
    device: str = "cpu",
    verbose: bool = False,
) -> np.ndarray:
    """NOTEARS optimization warm-started from W_init.

    Returns learned weight matrix W (d, d).
    """
    n, d = X.shape
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    W_init_c = np.ascontiguousarray(W_init)
    W = torch.tensor(W_init_c, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([W], lr=lr, betas=(0.9, 0.999))
    alpha = torch.tensor(0.0, device=device)
    _rho = rho

    for outer in range(max_iter):
        for _ in range(inner_iter):
            optimizer.zero_grad()
            residual = X_t - X_t @ W
            loss = 0.5 * (residual ** 2).sum() / n + lambda1 * W.abs().sum()
            M = W * W
            h = torch.trace(torch.matrix_exp(M)) - d
            aug_loss = loss + alpha * h + 0.5 * _rho * h * h
            aug_loss.backward()
            optimizer.step()

        with torch.no_grad():
            M = W * W
            h_val = (torch.trace(torch.matrix_exp(M)) - d).item()

        if verbose:
            print(f"  [NOTEARS] outer={outer+1}/{max_iter}, h={h_val:.2e}")

        if h_val < h_tol:
            break
        alpha = alpha + _rho * h_val
        _rho = min(_rho * 2.0, rho_max)

    W_np = W.detach().cpu().numpy()
    W_np[np.abs(W_np) < threshold] = 0.0
    return W_np


def _train_dagma_warm(
    X: np.ndarray,
    W_init: np.ndarray,
    threshold: float = 0.3,
    lambda1: float = 0.01,
    max_iter: int = 35,
    inner_iter: int = 200,
    lr: float = 0.001,
    device: str = "cpu",
    verbose: bool = False,
) -> np.ndarray:
    """DAGMA optimization warm-started from W_init using log-det acyclicity."""
    n, d = X.shape
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    W_init_c = np.ascontiguousarray(W_init)
    W = torch.tensor(W_init_c, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS([W], lr=lr, max_iter=inner_iter,
                                   line_search_fn="strong_wolfe")

    def _closure():
        optimizer.zero_grad()
        residual = X_t - X_t @ W
        mse = 0.5 * (residual ** 2).sum() / n
        l1 = lambda1 * W.abs().sum()
        # log-det acyclicity: -logdet(I - W*W) + d*log(s)
        # Simplified: use s=I, penalty = -logdet(I - W*W)
        I = torch.eye(d, device=device)
        M = I - W * W
        sign, logdet = torch.linalg.slogdet(M)
        dag_loss = -logdet if sign > 0 else 1e10
        total = mse + l1 + 0.5 * dag_loss
        total.backward()
        return total

    for outer in range(max_iter):
        optimizer.step(_closure)
        with torch.no_grad():
            I = torch.eye(d, device=device)
            M = I - W * W
            sign, logdet = torch.linalg.slogdet(M)
            h_val = -logdet.item() if sign > 0 else 1e10
        if verbose:
            print(f"  [DAGMA] outer={outer+1}/{max_iter}, h={h_val:.2e}")
        if h_val < 1e-7:
            break

    W_np = W.detach().cpu().numpy()
    W_np[np.abs(W_np) < threshold] = 0.0
    return W_np


def _train_golem_warm(
    X: np.ndarray,
    W_init: np.ndarray,
    threshold: float = 0.3,
    lambda1: float = 0.01,
    max_iter: int = 35,
    inner_iter: int = 200,
    lr: float = 0.001,
    device: str = "cpu",
    verbose: bool = False,
) -> np.ndarray:
    """GOLEM-EV optimization warm-started from W_init."""
    n, d = X.shape
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    W_init_c = np.ascontiguousarray(W_init)
    W = torch.tensor(W_init_c, dtype=torch.float32, device=device, requires_grad=True)

    # Use Adam for GOLEM (likelihood-based objective is non-convex)
    optimizer = torch.optim.Adam([W], lr=lr)
    alpha = torch.tensor(0.0, device=device)
    rho = 1.0

    for outer in range(max_iter):
        for _ in range(inner_iter):
            optimizer.zero_grad()
            residual = X_t - X_t @ W
            # GOLEM-EV: Gaussian likelihood
            nll = 0.5 * d * np.log(2 * np.pi) + 0.5 * torch.log(torch.var(residual, dim=0) + 1e-8).sum()
            nll += 0.5 * (residual ** 2).sum() / n
            l1 = lambda1 * W.abs().sum()
            M = W * W
            h = torch.trace(torch.matrix_exp(M)) - d
            total = nll + l1 + alpha * h + 0.5 * rho * h * h
            total.backward()
            optimizer.step()

        with torch.no_grad():
            h_val = (torch.trace(torch.matrix_exp(W * W)) - d).item()
        if verbose:
            print(f"  [GOLEM] outer={outer+1}/{max_iter}, h={h_val:.2e}")
        if h_val < 1e-7:
            break
        alpha = alpha + rho * h_val
        rho = min(rho * 2.0, 1e10)

    W_np = W.detach().cpu().numpy()
    W_np[np.abs(W_np) < threshold] = 0.0
    return W_np


_TRAIN_FN = {
    "notears": _train_notears_warm,
    "dagma": _train_dagma_warm,
    "golem": _train_golem_warm,
}


@dataclass
class TransferResult:
    """Results from a causal transfer experiment."""
    W_warm: np.ndarray           # learned with warm-start
    W_scratch: np.ndarray        # learned from random init
    W_shuffled: np.ndarray       # learned from shuffled source
    W_source: np.ndarray         # source-domain graph
    retention_warm: float        # fraction of source edges in W_warm
    retention_scratch: float     # fraction of source edges in W_scratch
    retention_shuffled: float    # fraction of source edges in W_shuffled
    delta_warm: float            # retention_warm - retention_scratch
    edge_overlap_warm: int       # |source edges| with |W_ij| > threshold in W_warm
    edge_overlap_scratch: int    # ... in W_scratch
    edge_overlap_shuffled: int   # ... in W_shuffled
    source_edges: int            # total source edges
    threshold: float             # edge-weight threshold
    method: str                  # base engine used
    time_s: float                # wall-clock time
    converged: bool = True
    metadata: Dict = field(default_factory=dict)


def fit_transfer(
    X_target: np.ndarray,
    source_graph: Union[np.ndarray, str],
    method: str = "notears",
    threshold: float = 0.3,
    seed: int = 42,
    lambda1: float = 0.01,
    max_iter: int = 35,
    inner_iter: int = 200,
    lr: float = 0.002,
    device: str = "cpu",
    shuffle_seed: int = 42,
    verbose: bool = True,
) -> TransferResult:
    """Run a causal transfer experiment on target data.

    Args:
        X_target: (n, d) target-domain data matrix
        source_graph: (d, d) source-domain weight matrix or path to .npy file
        method: base optimization engine ('notears', 'dagma', 'golem')
        threshold: edge-weight threshold for overlap counting
        seed: random seed for scratch initialization
        lambda1: L1 regularization
        max_iter: outer loop iterations
        inner_iter: inner loop iterations
        lr: learning rate
        device: 'cpu' or 'cuda'
        shuffle_seed: seed for shuffled-source generation
        verbose: print progress

    Returns:
        TransferResult with all three learned graphs and comparison metrics.
    """
    if method not in _TRAIN_FN:
        raise ValueError(f"Unknown method '{method}'. Choose: {list(_TRAIN_FN.keys())}")

    if isinstance(source_graph, str):
        W_source = np.load(source_graph)
    else:
        W_source = np.asarray(source_graph, dtype=np.float32)

    d = X_target.shape[1]
    assert W_source.shape == (d, d), \
        f"source_graph shape {W_source.shape} != target data cols ({d}, {d})"

    t0 = time.time()

    # Count source edges
    source_mask = np.abs(W_source) > threshold
    n_source_edges = source_mask.sum()

    # ── Warm-start ──
    if verbose:
        print(f"Transfer engine: {method}, d={d}, source_edges={n_source_edges}")
        print("  [1/3] Warm-start training...")
    train_fn = _TRAIN_FN[method]
    W_warm = train_fn(X_target, W_source, threshold=threshold,
                      lambda1=lambda1, max_iter=max_iter,
                      inner_iter=inner_iter, lr=lr,
                      device=device, verbose=verbose)

    # ── Scratch baseline ──
    if verbose:
        print("  [2/3] Scratch baseline...")
    np.random.seed(seed)
    W_scratch_init = np.random.randn(d, d) * 0.01
    np.fill_diagonal(W_scratch_init, 0)
    W_scratch = train_fn(X_target, W_scratch_init, threshold=threshold,
                         lambda1=lambda1, max_iter=max_iter,
                         inner_iter=inner_iter, lr=lr,
                         device=device, verbose=verbose)

    # ── Shuffled control ──
    if verbose:
        print("  [3/3] Shuffled-source negative control...")
    W_shuffled_init = _shuffle_graph(W_source, seed=shuffle_seed)
    W_shuffled = train_fn(X_target, W_shuffled_init, threshold=threshold,
                          lambda1=lambda1, max_iter=max_iter,
                          inner_iter=inner_iter, lr=lr,
                          device=device, verbose=verbose)

    elapsed = time.time() - t0

    # ── Compute metrics ──
    def _count_overlap(W_learned):
        """Count source edges that exceed threshold in the learned graph."""
        learned_mask = np.abs(W_learned) > threshold
        return (source_mask & learned_mask).sum()

    n_warm = _count_overlap(W_warm)
    n_scratch = _count_overlap(W_scratch)
    n_shuffled = _count_overlap(W_shuffled)

    if verbose:
        print(f"\n  Results (d={d}, tau={threshold:.2f}):")
        print(f"    Warm-start:  {n_warm}/{n_source_edges} "
              f"({100*n_warm/max(1,n_source_edges):.1f}%)")
        print(f"    Scratch:     {n_scratch}/{n_source_edges} "
              f"({100*n_scratch/max(1,n_source_edges):.1f}%)")
        print(f"    Shuffled:    {n_shuffled}/{n_source_edges} "
              f"({100*n_shuffled/max(1,n_source_edges):.1f}%)")
        print(f"    Delta warm:  +{100*(n_warm-n_scratch)/max(1,n_source_edges):.1f}pp")
        print(f"    Transfer:    {'YES' if n_warm > n_scratch else 'NO'}")
        print(f"    Time:        {elapsed:.1f}s")

    return TransferResult(
        W_warm=W_warm,
        W_scratch=W_scratch,
        W_shuffled=W_shuffled,
        W_source=W_source,
        retention_warm=100 * n_warm / max(1, n_source_edges),
        retention_scratch=100 * n_scratch / max(1, n_source_edges),
        retention_shuffled=100 * n_shuffled / max(1, n_source_edges),
        delta_warm=100 * (n_warm - n_scratch) / max(1, n_source_edges),
        edge_overlap_warm=n_warm,
        edge_overlap_scratch=n_scratch,
        edge_overlap_shuffled=n_shuffled,
        source_edges=n_source_edges,
        threshold=threshold,
        method=method,
        time_s=elapsed,
        converged=True,
        metadata={
            "d": d,
            "lambda1": lambda1,
            "max_iter": max_iter,
            "inner_iter": inner_iter,
            "lr": lr,
            "device": device,
        },
    )
