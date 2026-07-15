"""Utility functions for causalscale."""

import numpy as np
from typing import Optional


def standardize(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standardize data to zero mean, unit variance."""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=axis, keepdims=True)
    std = X.std(axis=axis, keepdims=True)
    std[std < 1e-8] = 1.0
    return (X - mean) / std


def make_synthetic_dag(
    d: int,
    n: int,
    edge_prob: float = 0.05,
    seed: int = 42,
) -> tuple:
    """Generate random DAG with linear SEM data.

    Args:
        d: number of variables
        n: number of samples
        edge_prob: probability of edge between ordered pairs
        seed: random seed

    Returns:
        (X, true_edges) where X is (n, d) and true_edges is int
    """
    rng = np.random.default_rng(seed)
    W = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < edge_prob:
                W[i, j] = rng.uniform(-0.7, 0.7)

    true_edges = int(np.sum(np.abs(W) > 0.1))

    # Generate data: X = epsilon @ (I - W)^{-T}
    I_minus_W = np.eye(d) - W.T
    X = rng.standard_normal((n, d)) @ np.linalg.inv(I_minus_W).T
    return X.astype(np.float32), true_edges


def correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation matrix."""
    X_c = X - X.mean(axis=0, keepdims=True)
    X_s = X_c / (X_c.std(axis=0, keepdims=True).clip(min=1e-8))
    return (X_s.T @ X_s) / (X.shape[0] - 1)
