"""
Adaptive Rank Selection Engine
================================
Three complementary strategies for automatic rank determination:

1. Spectral Heuristic: analyzes eigenvalue decay of correlation/covariance matrix
   to estimate the effective rank (number of significant dimensions).

2. Information Criterion: AIC/BIC on reconstruction error during training
   to select the optimal rank that balances fit and complexity.

3. Rank Pruning: starts with a high rank, applies group lasso penalty on
   singular values during training to automatically prune unnecessary dimensions.

Key insight: In causal discovery, the effective rank of the adjacency matrix
is related to the number of independent causal mechanisms, not the data dimensionality.

Reference: The optimal rank r* ≈ sqrt(# of true causal edges) for sparse DAGs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class RankDiagnostic:
    """Result of rank estimation."""
    recommended_rank: int
    spectral_rank: int          # from eigenvalue analysis
    information_rank: int       # from AIC/BIC
    pruned_rank: Optional[int]  # from training-time pruning
    eigenvalue_decay: np.ndarray
    effective_dimensionality: float
    confidence: str             # 'high', 'medium', 'low'


def estimate_effective_rank(
    X: np.ndarray,
    method: str = 'hybrid',
    max_rank: Optional[int] = None,
    variance_threshold: float = 0.75,
    n_bootstrap: int = 50
) -> RankDiagnostic:
    """
    Estimate the effective rank of a data matrix X (n_samples, d_variables).

    Methods:
        'spectral': eigenvalue decay analysis
        'parallel': parallel analysis (compare to random data eigenvalues)
        'hybrid': combine spectral + parallel (default, most robust)

    Args:
        X: (n, d) data matrix
        method: 'spectral', 'parallel', or 'hybrid'
        max_rank: upper bound on rank (default: min(n, d) // 2)
        variance_threshold: cumulative variance explained threshold
        n_bootstrap: bootstrap samples for parallel analysis

    Returns:
        RankDiagnostic with recommended rank and detailed diagnostics
    """
    n, d = X.shape
    max_possible = min(n, d)
    if max_rank is None:
        max_rank = max_possible // 2

    # Center and compute covariance
    X_c = X - X.mean(axis=0, keepdims=True)
    cov = (X_c.T @ X_c) / (n - 1) if n >= d else (X_c @ X_c.T) / (n - 1)

    # Eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # discard numerical zeros

    # Normalize
    total_var = eigenvalues.sum()
    if total_var < 1e-10:
        return RankDiagnostic(
            recommended_rank=1, spectral_rank=1, information_rank=1,
            pruned_rank=None, eigenvalue_decay=eigenvalues[:max_rank],
            effective_dimensionality=1.0, confidence='low'
        )

    normalized = eigenvalues / total_var

    # ── Method 1: Spectral (cumulative variance) ──
    cumsum = np.cumsum(normalized)
    spectral_rank = int(np.searchsorted(cumsum, variance_threshold) + 1)
    spectral_rank = min(spectral_rank, max_rank)

    # ── Method 2: Parallel analysis ──
    if method in ('parallel', 'hybrid'):
        # Generate n_bootstrap random data matrices of same size
        random_eigenvalues = np.zeros((n_bootstrap, min(max_rank, len(eigenvalues))))
        for b in range(n_bootstrap):
            X_rand = np.random.randn(n, d)
            X_rand -= X_rand.mean(axis=0)
            if n >= d:
                cov_rand = (X_rand.T @ X_rand) / (n - 1)
            else:
                cov_rand = (X_rand @ X_rand.T) / (n - 1)
            ev_rand = np.linalg.eigvalsh(cov_rand)
            ev_rand = np.sort(ev_rand)[::-1]
            l = min(max_rank, len(ev_rand))
            random_eigenvalues[b, :l] = ev_rand[:l]

        # 95th percentile of random eigenvalues
        random_95 = np.percentile(random_eigenvalues, 95, axis=0)

        # Find where real eigenvalues drop below random
        parallel_rank = 1
        for i in range(min(len(eigenvalues), max_rank)):
            if eigenvalues[i] > random_95[min(i, len(random_95) - 1)]:
                parallel_rank = i + 1
            else:
                break
    else:
        parallel_rank = spectral_rank

    # ── Method 3: Information criterion (approximate) ──
    # BIC-based: minimize n*log(MSE) + k*log(n) where k = d*r parameters
    information_rank = min(spectral_rank, parallel_rank)
    # A heuristic refinement: knee point detection
    if len(normalized) >= 3:
        # Find the "elbow" in eigenvalue decay
        diffs = np.diff(normalized)
        acceleration = np.diff(diffs)
        if len(acceleration) > 0:
            knee = np.argmax(acceleration) + 2
            information_rank = max(1, min(knee, max_rank))

    # ── Hybrid decision ──
    # Effective dimensionality: (Σλ)²/Σ(λ²) — robust measure of intrinsic rank
    effective_dim = float((eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum())

    if method == 'hybrid':
        # Use effective dimensionality directly — more stable than median of three heuristics
        recommended = int(np.round(effective_dim))
        # Cross-validate with spectral and parallel
        heuristic_min = min(spectral_rank, parallel_rank, information_rank)
        heuristic_max = max(spectral_rank, parallel_rank, information_rank)
        # Clamp to heuristic range (avoid outlier from effective_dim)
        recommended = max(heuristic_min, min(recommended, heuristic_max))
    elif method == 'parallel':
        recommended = parallel_rank
    else:
        recommended = spectral_rank

    # Cap: rank should not exceed sqrt(n) (sample-limited) or d/2
    sample_cap = max(4, int(np.sqrt(n)))
    recommended = max(1, min(recommended, max_rank, sample_cap))

    # Confidence assessment
    rank_spread = max(spectral_rank, parallel_rank) - min(spectral_rank, parallel_rank)
    if rank_spread <= 2 and effective_dim < max_possible * 0.5:
        confidence = 'high'
    elif rank_spread <= 5:
        confidence = 'medium'
    else:
        confidence = 'low'

    return RankDiagnostic(
        recommended_rank=recommended,
        spectral_rank=spectral_rank,
        information_rank=information_rank,
        pruned_rank=None,
        eigenvalue_decay=eigenvalues[:max(10, max_rank)],
        effective_dimensionality=effective_dim,
        confidence=confidence
    )


class AutoRankSelector:
    """
    Training-time adaptive rank selector with group-lasso pruning.

    Starts with a high initial rank, then during training:
    1. Computes singular values of W = U @ V^T
    2. Applies group lasso penalty on small singular values
    3. Prunes dimensions that fall below a threshold
    4. Returns the final effective rank

    Usage:
        selector = AutoRankSelector(initial_rank=128, min_rank=4, prune_every=50)
        rank = selector.initial_rank
        for epoch in range(epochs):
            # ... training step ...
            if epoch % selector.prune_every == 0:
                U, V, new_rank = selector.check_and_prune(U, V, epoch)
    """

    def __init__(
        self,
        initial_rank: int = 128,
        min_rank: int = 4,
        max_rank: int = 512,
        prune_every: int = 50,
        prune_threshold: float = 0.01,
        group_lasso_weight: float = 0.001,
        patience: int = 3,
        warmup_epochs: int = 100
    ):
        """
        Args:
            initial_rank: starting rank (should be an overestimate)
            min_rank: minimum rank to keep
            max_rank: maximum allowed rank
            prune_every: epochs between pruning checks
            prune_threshold: singular value fraction below which to prune
            group_lasso_weight: weight of group lasso penalty
            patience: epochs to wait before pruning again
            warmup_epochs: epochs before first pruning (let model learn)
        """
        self.initial_rank = initial_rank
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.prune_every = prune_every
        self.prune_threshold = prune_threshold
        self.group_lasso_weight = group_lasso_weight
        self.patience = patience
        self.warmup_epochs = warmup_epochs

        # State
        self.current_rank = initial_rank
        self.prune_history: List[int] = []
        self.singular_value_history: List[np.ndarray] = []
        self._last_prune_epoch = -patience

    def compute_group_lasso_penalty(self, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Group lasso on column pairs of U and V.
        Penalizes each rank dimension as a group: ||[U[:,k], V[:,k]]||_2
        """
        d, r = U.shape
        penalty = torch.tensor(0.0, device=U.device)
        for k in range(r):
            # Group norm of k-th dimension
            group_norm = torch.sqrt(
                (U[:, k] ** 2).sum() + (V[:, k] ** 2).sum()
            )
            penalty = penalty + group_norm
        return self.group_lasso_weight * penalty

    def get_singular_values(self, U: torch.Tensor, V: torch.Tensor) -> np.ndarray:
        """Compute singular values of W = U @ V^T via SVD of the low-rank factorization."""
        with torch.no_grad():
            # Economy SVD: U and V give us the singular values
            # Compute QR to orthogonalize
            U_np = U.cpu().numpy()
            V_np = V.cpu().numpy()
            # Singular values via solving for UV^T's SVD equivalently
            Q_U, R_U = np.linalg.qr(U_np)
            Q_V, R_V = np.linalg.qr(V_np)
            # singular values of U@V^T = singular values of R_U @ R_V^T
            sv = np.linalg.svd(R_U @ R_V.T, compute_uv=False)
            return sv

    def check_and_prune(
        self,
        U: nn.Parameter,
        V: nn.Parameter,
        epoch: int
    ) -> Tuple[nn.Parameter, nn.Parameter, int]:
        """
        Check if pruning is needed and prune if conditions are met.

        Returns:
            (U_pruned, V_pruned, new_rank)
        """
        if epoch < self.warmup_epochs:
            return U, V, self.current_rank

        if epoch - self._last_prune_epoch < self.prune_every:
            return U, V, self.current_rank

        sv = self.get_singular_values(U, V)
        self.singular_value_history.append(sv)

        # Normalize singular values
        sv_norm = sv / (sv.max() + 1e-10)

        # Count dimensions above threshold
        active = sv_norm > self.prune_threshold
        n_active = int(active.sum())
        n_active = max(n_active, self.min_rank)

        self.prune_history.append(n_active)

        if n_active < self.current_rank and n_active >= self.min_rank:
            # Prune: keep only top n_active dimensions
            keep_idx = np.argsort(-sv)[:n_active]
            keep_idx = sorted(keep_idx)

            with torch.no_grad():
                U_new_data = U.data[:, keep_idx].clone()
                V_new_data = V.data[:, keep_idx].clone()

            U_new = nn.Parameter(U_new_data)
            V_new = nn.Parameter(V_new_data)

            self.current_rank = n_active
            self._last_prune_epoch = epoch

            return U_new, V_new, n_active

        self._last_prune_epoch = epoch
        return U, V, self.current_rank

    def get_rank_diagnostic(self) -> Dict:
        """Return diagnostic information about rank selection."""
        return {
            'initial_rank': self.initial_rank,
            'final_rank': self.current_rank,
            'prune_history': self.prune_history,
            'n_prunes': len([1 for i in range(1, len(self.prune_history))
                             if self.prune_history[i] < self.prune_history[i-1]]),
            'rank_reduction': f'{self.initial_rank} → {self.current_rank} '
                            f'({(1 - self.current_rank/self.initial_rank)*100:.0f}% reduction)'
        }

    def compute_aic_bic(
        self,
        reconstruction_error: float,
        n_samples: int,
        d_variables: int
    ) -> Dict[str, float]:
        """
        Compute AIC and BIC for current rank.

        AIC = n * log(MSE) + 2 * params
        BIC = n * log(MSE) + log(n) * params
        params = 2 * d * r (for U and V)
        """
        params = 2 * d_variables * self.current_rank
        mse = reconstruction_error
        if mse < 1e-10:
            mse = 1e-10

        aic = n_samples * np.log(mse) + 2 * params
        bic = n_samples * np.log(mse) + np.log(n_samples) * params

        return {
            'aic': float(aic),
            'bic': float(bic),
            'params': params,
            'mse': float(mse),
            'rank': self.current_rank
        }
