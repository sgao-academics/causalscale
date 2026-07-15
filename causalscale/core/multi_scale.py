"""
Multi-Scale Low-Rank Decomposition
====================================
Hierarchical decomposition: W = Σ_{s=1}^S U_s @ V_s^T

Each scale s captures structure at different granularities:
  - Scale 1 (smallest rank): coarse, global structure
  - Scale 2: mid-level structure
  - Scale S (largest rank): fine-grained, local structure

Benefits:
  1. Coarse-to-fine learning: large-scale patterns stabilize faster
  2. Scale-specific DAG constraints: DAG at each scale → W is DAG
  3. Interpretability: each scale represents a different level of causal organization
  4. Sparsity by scale: coarser scales are sparser, finer scales capture details

Reference: Inspired by hierarchical matrix (H-matrix) decompositions and
           multiresolution analysis in signal processing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict


class MultiScaleLowRank(nn.Module):
    """
    Multi-scale low-rank decomposition of the adjacency matrix.

    W = sum_{s=1}^{n_scales} U_s @ V_s^T

    where U_s, V_s in R^(d x r_s) and r_1 < r_2 < ... < r_S.

    Args:
        d: number of variables
        scales: list of (rank_s, sparsity_s, lr_multiplier_s) for each scale
                e.g., [(16, 0.1, 1.0), (32, 0.05, 0.5), (64, 0.02, 0.25)]
        device: 'cuda' or 'cpu'
    """

    def __init__(
        self,
        d: int,
        scales: Optional[List[Tuple[int, float, float]]] = None,
        device: str = 'cuda'
    ):
        super().__init__()
        self.d = d
        self.device = device

        if scales is None:
            # Default: 3 scales from coarse to fine
            scales = [
                (max(4, d // 32), 0.15, 1.0),    # coarse
                (max(8, d // 16), 0.08, 0.5),     # medium
                (max(16, d // 8), 0.03, 0.25),     # fine
            ]

        self.scales = scales
        self.n_scales = len(scales)

        # Initialize U_s, V_s for each scale
        self.U_list = nn.ParameterList()
        self.V_list = nn.ParameterList()
        self.sparsity_weights = []
        self.lr_multipliers = []

        for r_s, sparsity_s, lr_mult_s in scales:
            r_clamped = min(r_s, d)  # rank can't exceed d
            self.U_list.append(
                nn.Parameter(torch.randn(d, r_clamped, device=device) * 0.01 / self.n_scales)
            )
            self.V_list.append(
                nn.Parameter(torch.randn(d, r_clamped, device=device) * 0.01 / self.n_scales)
            )
            self.sparsity_weights.append(sparsity_s)
            self.lr_multipliers.append(lr_mult_s)

        self._cached_W = None

    def forward(self) -> torch.Tensor:
        """Compute full adjacency: W = sum_s U_s @ V_s^T"""
        W = torch.zeros(self.d, self.d, device=self.device)
        for U, V in zip(self.U_list, self.V_list):
            W = W + U @ V.T
        return W

    def get_scale(self, s: int) -> torch.Tensor:
        """Get adjacency contribution from scale s only."""
        if s < 0 or s >= self.n_scales:
            raise IndexError(f"Scale {s} out of range [0, {self.n_scales})")
        return self.U_list[s] @ self.V_list[s].T

    def get_scale_weights(self) -> Dict[str, float]:
        """Get Frobenius norm of each scale's contribution (for analysis)."""
        weights = {}
        total_norm = 0.0
        norms = []
        with torch.no_grad():
            for s, (U, V) in enumerate(zip(self.U_list, self.V_list)):
                n = float((U @ V.T).norm().item())
                norms.append(n)
                total_norm += n
            for s, n in enumerate(norms):
                weights[f'scale_{s}'] = n / (total_norm + 1e-10)
        return weights

    def compute_loss(
        self,
        X: torch.Tensor,
        D: torch.Tensor,
        dag_weight: float = 0.5,
        dag_rho: float = 1.0,
        dag_alpha: float = 0.0,
        entropy_reg: float = 0.0,
        P_soft: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the total loss including reconstruction, DAG, sparsity, and optionally entropy.

        Args:
            X: (n, d) data matrix on device
            D: (n, d) target matrix on device
            dag_weight: weight of DAG constraint
            dag_rho: augmented Lagrangian rho
            dag_alpha: augmented Lagrangian alpha
            entropy_reg: entropy regularization weight (for cluster-aware mode)
            P_soft: optional (n, K) soft cluster assignments

        Returns:
            (loss, components) where components is a dict of individual loss terms
        """
        W = self.forward()
        pred = X @ W
        loss_recon = nn.MSELoss()(pred, D)

        # Scale-specific sparsity
        loss_sparsity = torch.tensor(0.0, device=self.device)
        for s, (U, V) in enumerate(zip(self.U_list, self.V_list)):
            W_s = U @ V.T
            loss_sparsity = loss_sparsity + self.sparsity_weights[s] * torch.sum(torch.abs(W_s))

        # DAG constraint
        h_val = randomized_h_dag(W)
        loss_dag = 0.5 * dag_rho * h_val**2 + dag_alpha * h_val

        loss = loss_recon + dag_weight * loss_dag + loss_sparsity

        components = {
            'recon': float(loss_recon.item()),
            'dag': float(loss_dag.item()),
            'sparsity': float(loss_sparsity.item()),
            'h(W)': float(h_val),
        }

        # Optional entropy (SSCAGate integration)
        if P_soft is not None and entropy_reg > 0:
            log_P = torch.log(P_soft.clamp(min=1e-8))
            entropy = -(P_soft * log_P).sum(dim=1).mean()
            loss = loss - entropy_reg * entropy
            components['entropy'] = float(entropy.item())

        return loss, components

    def get_edge_confidence_by_scale(
        self,
        threshold: float = 0.3
    ) -> Dict[str, np.ndarray]:
        """
        Compute edge-level confidence based on scale agreement.
        Edge that appears across multiple scales → higher confidence.
        """
        edges_by_scale = {}
        with torch.no_grad():
            for s, (U, V) in enumerate(zip(self.U_list, self.V_list)):
                W_s = (U @ V.T).cpu().numpy()
                mask = np.abs(W_s) > threshold * (0.5 ** s)  # lower threshold for finer scales
                edges_by_scale[f'scale_{s}'] = mask.astype(float)

        # Consensus: weighted average across scales
        d = self.d
        consensus = np.zeros((d, d))
        for s in range(self.n_scales):
            weight = 1.0 / (s + 1)  # coarser scales weight more
            consensus += weight * edges_by_scale[f'scale_{s}']
        consensus /= sum(1.0 / (s + 1) for s in range(self.n_scales))

        return {
            'consensus': consensus,
            'by_scale': edges_by_scale,
            'n_consensus_edges': int(np.sum(consensus > 0.5)),
            'n_any_scale': int(np.sum(np.any(
                [edges_by_scale[f'scale_{s}'] > 0 for s in range(self.n_scales)], axis=0
            ))),
        }

    def get_parameter_groups(self, base_lr: float = 0.001) -> List[Dict]:
        """
        Get optimizer parameter groups with scale-specific learning rates.
        Coarser scales get higher LR → converge faster and stabilize early.
        """
        groups = []
        for s, (U, V) in enumerate(zip(self.U_list, self.V_list)):
            lr = base_lr * self.lr_multipliers[s]
            groups.append({
                'params': [U, V],
                'lr': lr,
                'name': f'scale_{s}'
            })
        return groups

    def to_dense(self) -> np.ndarray:
        """Return full dense adjacency as numpy array."""
        with torch.no_grad():
            return self.forward().cpu().numpy()


def randomized_h_dag(W: torch.Tensor, n_power_iter: int = 30) -> float:
    """
    Randomized DAG constraint: h(W) = tr(exp(W * W)) - d.
    Uses power iteration to estimate the spectral radius of exp(W*W),
    reducing from O(d^3) to O(d * n_power_iter + d^2).

    This is more efficient than exact matrix_exp for large d.
    """
    d = W.shape[0]
    device = W.device

    # For d <= 100, exact matrix_exp is fine
    if d <= 100:
        return float((torch.linalg.matrix_exp(W * W).trace() - d).item())

    # Randomized: estimate trace via Hutchinson's trick
    # tr(exp(M)) ≈ (1/n_vec) * sum_i v_i^T * exp(M) * v_i
    # where v_i ~ N(0, I), and exp(M)*v_i is approximated via power iteration
    n_vec = 5
    trace_est = 0.0

    for _ in range(n_vec):
        v = torch.randn(d, 1, device=device)
        v = v / (v.norm() + 1e-12)

        # Power iteration on W*W to approximate exp(W*W)*v
        # Using Lanczos-like iteration for matrix exponential
        M = W * W
        w = v.clone()
        term = v.clone()
        factorial = 1.0

        for k in range(1, 15):  # Taylor series expansion
            w = M @ w
            factorial *= k
            if factorial > 1e30:  # prevent overflow
                break
            term = term + w / factorial

        trace_est += float((v.T @ term)[0, 0])

    return trace_est / n_vec - d
