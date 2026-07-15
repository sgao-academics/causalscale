"""
Cluster-Aware Gate: Differentiable Cluster-Conditioned Gradient Attenuation.

Core innovation: Clusters with high residual variance receive attenuated
gradient contributions, preventing noisy clusters from dominating optimization.

Formula:
    gate_g = 1 / (1 + exp(-alpha * (sigma_bar - sigma_g)))

where sigma_g is within-cluster residual std.

Reference: Gao et al. (2026) "CAGate: Cluster-Aware Gating for Causal Discovery"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
# Per-Cluster Gradient Gate
# ═══════════════════════════════════════════════════════════════════

def compute_cluster_gates(
    residuals: torch.Tensor,
    group_ids: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, Dict]:
    """Compute per-cluster gradient attenuation coefficients.

    Args:
        residuals: (n_obs,) residual vector
        group_ids: (n_obs,) integer cluster labels
        alpha: sharpness of gating (higher = sharper cutoff)

    Returns:
        gates: (n_obs,) per-observation attenuation weight
        stats: dict with sigma_g, sigma_bar, gates_per_cluster
    """
    n_clusters = int(group_ids.max().item() + 1)
    device = residuals.device

    sigma_g = torch.zeros(n_clusters, device=device)
    for g in range(n_clusters):
        mask = group_ids == g
        if mask.sum() <= 1:
            sigma_g[g] = 0.0
        else:
            sigma_g[g] = residuals[mask].std(unbiased=True)

    sigma_bar = sigma_g[sigma_g > 0].median()

    # Sigmoid gate: clusters with sigma above median -> attenuated
    gates_per_cluster = 1.0 / (
        1.0 + torch.exp(-alpha * (sigma_bar - sigma_g))
    )

    # Map back to per-observation
    gates = torch.zeros_like(residuals)
    for g in range(n_clusters):
        mask = group_ids == g
        gates[mask] = gates_per_cluster[g]

    stats = {
        "sigma_g": sigma_g,
        "sigma_bar": sigma_bar,
        "gates_per_cluster": gates_per_cluster,
    }
    return gates, stats


# ═══════════════════════════════════════════════════════════════════
# ClusterAwareGate — PyTorch Module
# ═══════════════════════════════════════════════════════════════════

class ClusterAwareGate(nn.Module):
    """MSE loss with cluster-conditioned gradient attenuation.

    Forward returns (loss, se, stats) where loss incorporates
    the cluster gate for backpropagation.

    Args:
        gate_alpha: sharpness of gating
    """

    def __init__(self, gate_alpha: float = 1.0):
        super().__init__()
        self.gate_alpha = gate_alpha

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        X: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Compute gated MSE loss.

        Args:
            y_pred: (n,) or (n, 1) predictions
            y_true: (n,) or (n, 1) ground truth
            X: (n, d) design matrix (for SE computation)
            group_ids: (n,) integer cluster labels

        Returns:
            weighted_mse: gated MSE loss for backprop
            se: cluster-robust standard errors
            gate_stats: diagnostic dict
        """
        residuals = (y_true - y_pred).squeeze()
        gates, gate_stats = compute_cluster_gates(
            residuals, group_ids, self.gate_alpha
        )
        weighted_mse = (gates * residuals**2).sum() / gates.sum().clamp(min=1)
        V = _cluster_robust_variance(residuals, X, group_ids)
        se = torch.sqrt(torch.diag(V).clamp(min=1e-10))
        return weighted_mse, se, gate_stats


# ═══════════════════════════════════════════════════════════════════
# Soft Cluster Gate (for learnable cluster assignments)
# ═══════════════════════════════════════════════════════════════════

def soft_cluster_gates(
    residuals: torch.Tensor,
    P_soft: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute per-sample gates from soft cluster assignments.

    Args:
        residuals: (n,) per-sample residuals
        P_soft: (n, K) soft cluster assignment probabilities
        alpha: gate sharpness

    Returns:
        gates: (n,) per-sample gate weights
    """
    n, K = P_soft.shape
    cw = P_soft.sum(dim=0)  # cluster weights
    wm = (P_soft * residuals.unsqueeze(1)).sum(dim=0) / cw.clamp(min=1e-8)
    ds = (residuals.unsqueeze(1) - wm.unsqueeze(0)) ** 2
    wv = (P_soft * ds).sum(dim=0) / cw.clamp(min=1e-8)
    cs = torch.sqrt(wv.clamp(min=1e-8))
    sm = torch.median(cs)
    rg = torch.sigmoid(alpha * (sm / cs.clamp(min=1e-8) - 1))
    return (P_soft * rg.unsqueeze(0)).sum(dim=1)


# ═══════════════════════════════════════════════════════════════════
# Cluster-Robust Sandwich Estimator (foundation)
# ═══════════════════════════════════════════════════════════════════

def _cluster_robust_variance(
    residuals: torch.Tensor,
    X: torch.Tensor,
    group_ids: torch.Tensor,
) -> torch.Tensor:
    """Cluster-robust sandwich variance estimator.

    V_hat = c * (X^T X)^{-1} [sum_g (X_g^T u_g)(u_g^T X_g)] (X^T X)^{-1}
    """
    n_prov = int(group_ids.max().item() + 1)
    n_obs, n_feat = X.shape
    device = X.device

    XtX = X.T @ X
    XtX_inv = torch.linalg.inv(XtX + 1e-6 * torch.eye(n_feat, device=device))

    meat = torch.zeros(n_feat, n_feat, device=device)
    for p in range(n_prov):
        mask = group_ids == p
        if mask.sum() <= 1:
            continue
        X_g = X[mask]
        u_g = residuals[mask]
        meat += (X_g.T @ u_g) @ (u_g.T @ X_g)

    G, K = n_prov, n_feat
    correction = (G / max(G - 1, 1)) * ((n_obs - 1) / max(n_obs - K, 1))
    return correction * XtX_inv @ meat @ XtX_inv
