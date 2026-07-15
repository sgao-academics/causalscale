"""
Uncertainty Quantification Engine
===================================
Three complementary approaches for edge-level confidence:

1. Bootstrap Ensemble: multiple fits on resampled data → edge frequency = confidence
2. MC Dropout Ensemble: multiple forward passes with noise → variance → confidence
3. Stability Selection: subsampling-based edge selection probability

All methods output:
  - adjacency: (d, d) point estimate
  - confidence: (d, d) in [0, 1], where 1 = highest confidence
  - ci_lower, ci_upper: (d, d) 95% confidence interval bounds
  - n_high_confidence: number of edges with confidence > 0.8
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass
import warnings


@dataclass
class UncertaintyResult:
    """Complete uncertainty quantification output."""
    adjacency: np.ndarray           # (d, d) point estimate (mean or median)
    confidence: np.ndarray          # (d, d) edge confidence [0, 1]
    ci_lower: np.ndarray            # (d, d) 2.5th percentile
    ci_upper: np.ndarray            # (d, d) 97.5th percentile
    edge_probability: np.ndarray    # (d, d) probability of edge existing
    n_bootstrap: int
    n_high_confidence_edges: int    # edges with confidence > 0.8
    n_total_edges: int              # total edges in point estimate
    method: str


class BootstrapEnsemble:
    """
    Bootstrap ensemble for uncertainty quantification.

    Performs K bootstrap resamples of the data, fits the model on each,
    and aggregates edge weights across samples.

    Edge confidence = frequency with which |W_ij| > threshold across bootstrap samples.

    Args:
        n_bootstrap: number of bootstrap samples (default: 100)
        sample_fraction: fraction of data to resample each time (default: 0.8)
        threshold: edge weight threshold
        seed: random seed for reproducibility
    """

    def __init__(
        self,
        n_bootstrap: int = 100,
        sample_fraction: float = 0.8,
        threshold: float = 0.3,
        seed: int = 42
    ):
        self.n_bootstrap = n_bootstrap
        self.sample_fraction = sample_fraction
        self.threshold = threshold
        self.seed = seed
        self._W_samples: Optional[List[np.ndarray]] = None

    def fit(
        self,
        X: np.ndarray,
        fit_fn: Callable[[np.ndarray], np.ndarray],
        D: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> UncertaintyResult:
        """
        Run bootstrap ensemble.

        Args:
            X: (n, d) data matrix
            fit_fn: function that takes (X_subset, D_subset) and returns (d, d) adjacency
            D: optional (n, d) target matrix
            verbose: print progress

        Returns:
            UncertaintyResult with aggregated statistics
        """
        n, d = X.shape
        np.random.seed(self.seed)
        rng = np.random.RandomState(self.seed)

        self._W_samples = []
        n_sample = max(int(n * self.sample_fraction), d + 1)

        for b in range(self.n_bootstrap):
            # Resample with replacement
            idx = rng.choice(n, size=n_sample, replace=True)

            X_b = X[idx]
            D_b = D[idx] if D is not None else None

            try:
                if D_b is not None:
                    W_b = fit_fn(X_b, D_b)
                else:
                    W_b = fit_fn(X_b)
                self._W_samples.append(W_b)
            except Exception as e:
                if verbose:
                    print(f"  Bootstrap {b}: failed ({e})")
                continue

            if verbose and (b + 1) % max(1, self.n_bootstrap // 10) == 0:
                print(f"  Bootstrap {b+1}/{self.n_bootstrap}")

        return self._aggregate()

    def fit_parallel(
        self,
        X: np.ndarray,
        fit_fn: Callable[[np.ndarray], np.ndarray],
        D: Optional[np.ndarray] = None,
        n_workers: int = 4,
        verbose: bool = False
    ) -> UncertaintyResult:
        """
        Parallel bootstrap using multiprocessing.
        Falls back to sequential if multiprocessing fails.
        """
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import pickle

            n, d = X.shape
            np.random.seed(self.seed)
            n_sample = max(int(n * self.sample_fraction), d + 1)

            def _bootstrap_task(task_seed):
                rng = np.random.RandomState(task_seed)
                idx = rng.choice(n, size=n_sample, replace=True)
                X_b = X[idx]
                D_b = D[idx] if D is not None else None
                if D_b is not None:
                    return fit_fn(X_b, D_b)
                return fit_fn(X_b)

            seeds = [self.seed * (b + 1) * 137 for b in range(self.n_bootstrap)]

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_bootstrap_task, s): i
                          for i, s in enumerate(seeds)}
                self._W_samples = [None] * len(futures)
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        self._W_samples[i] = future.result(timeout=300)
                    except Exception as e:
                        if verbose:
                            print(f"  Bootstrap {i}: failed ({e})")

            self._W_samples = [w for w in self._W_samples if w is not None]

        except (ImportError, pickle.PicklingError):
            warnings.warn("Parallel bootstrap failed, falling back to sequential")
            return self.fit(X, fit_fn, D, verbose)

        return self._aggregate()

    def _aggregate(self) -> UncertaintyResult:
        """Aggregate bootstrap samples into uncertainty statistics."""
        if not self._W_samples:
            raise RuntimeError("No bootstrap samples available. Call .fit() first.")

        W_stack = np.stack(self._W_samples, axis=0)  # (K, d, d)
        K, d, _ = W_stack.shape

        # Point estimate: mean across bootstrap samples
        adjacency = np.mean(W_stack, axis=0)

        # Edge probability: fraction of samples where |W| > threshold
        edge_masks = np.abs(W_stack) > self.threshold
        edge_probability = np.mean(edge_masks.astype(float), axis=0)

        # Confidence: based on both edge probability and weight stability
        # confidence_ij = prob(|W_ij| > threshold) * (1 - CV(W_ij))
        weight_mean = adjacency
        weight_std = np.std(W_stack, axis=0)
        cv = np.divide(weight_std, np.abs(weight_mean) + 1e-10,
                       out=np.ones_like(weight_mean),
                       where=np.abs(weight_mean) > 1e-10)
        cv = np.clip(cv, 0, 2)  # cap CV at 2

        # Confidence: product of edge probability and inverse CV
        confidence = edge_probability * (1 - cv / 2)

        # CI bounds: percentiles
        ci_lower = np.percentile(W_stack, 2.5, axis=0)
        ci_upper = np.percentile(W_stack, 97.5, axis=0)

        # Thresholded adjacency
        adj_thresh = adjacency * (np.abs(adjacency) > self.threshold)

        n_high = int(np.sum(confidence > 0.8))
        n_total = int(np.sum(np.abs(adjacency) > self.threshold))

        return UncertaintyResult(
            adjacency=adj_thresh,
            confidence=confidence,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            edge_probability=edge_probability,
            n_bootstrap=K,
            n_high_confidence_edges=n_high,
            n_total_edges=n_total,
            method='bootstrap'
        )


class StabilitySelector:
    """
    Stability selection for robust edge discovery.

    Instead of full bootstrap, uses subsampling without replacement.
    Edge is selected if its selection probability exceeds a threshold
    across subsamples.

    Reference: Meinshausen & Buhlmann (2010) JRSSB.
    """

    def __init__(
        self,
        n_subsamples: int = 100,
        sample_fraction: float = 0.5,
        selection_threshold: float = 0.6,
        edge_threshold: float = 0.3,
        seed: int = 42
    ):
        self.n_subsamples = n_subsamples
        self.sample_fraction = sample_fraction
        self.selection_threshold = selection_threshold
        self.edge_threshold = edge_threshold
        self.seed = seed
        self._selection_probs: Optional[np.ndarray] = None

    def select(
        self,
        X: np.ndarray,
        fit_fn: Callable[[np.ndarray], np.ndarray],
        D: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> UncertaintyResult:
        """
        Run stability selection.

        Args:
            X: (n, d) data matrix
            fit_fn: function returning (d, d) adjacency matrix
            D: optional target matrix
            verbose: print progress

        Returns:
            UncertaintyResult
        """
        n, d = X.shape
        np.random.seed(self.seed)
        rng = np.random.RandomState(self.seed)

        n_sample = max(int(n * self.sample_fraction), d + 1)
        selection_counts = np.zeros((d, d))
        W_samples = []
        n_successful = 0

        for b in range(self.n_subsamples):
            idx = rng.choice(n, size=n_sample, replace=False)

            X_b = X[idx]
            D_b = D[idx] if D is not None else None

            try:
                if D_b is not None:
                    W_b = fit_fn(X_b, D_b)
                else:
                    W_b = fit_fn(X_b)
                W_samples.append(W_b)
                selection_counts += (np.abs(W_b) > self.edge_threshold).astype(float)
                n_successful += 1
            except Exception:
                continue

            if verbose and (b + 1) % max(1, self.n_subsamples // 10) == 0:
                print(f"  Stability {b+1}/{self.n_subsamples}")

        if n_successful == 0:
            raise RuntimeError("All subsamples failed")

        # Selection probability
        selection_probs = selection_counts / n_successful
        self._selection_probs = selection_probs

        # Stable edges: selection probability > threshold
        stable_mask = selection_probs >= self.selection_threshold

        # Point estimate: mean of W over subsamples
        W_stack = np.stack(W_samples, axis=0)
        W_mean = np.mean(W_stack, axis=0)
        adjacency = W_mean * stable_mask

        # CI from subsample distribution
        ci_lower = np.percentile(W_stack, 2.5, axis=0) * stable_mask
        ci_upper = np.percentile(W_stack, 97.5, axis=0) * stable_mask

        n_high = int(np.sum(selection_probs >= 0.8))
        n_total = int(np.sum(stable_mask))

        return UncertaintyResult(
            adjacency=adjacency,
            confidence=selection_probs,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            edge_probability=selection_probs,
            n_bootstrap=n_successful,
            n_high_confidence_edges=n_high,
            n_total_edges=n_total,
            method='stability_selection'
        )


class MCDropoutEnsemble:
    """
    Monte Carlo Dropout ensemble for rapid uncertainty estimation.

    Adds Gaussian noise to U and V during multiple forward passes,
    approximating Bayesian inference over edge weights.

    Much faster than bootstrap (no refitting), but less calibrated.
    Good for quick uncertainty visualization.

    Args:
        n_samples: number of MC samples
        noise_scale: standard deviation of additive noise
    """

    def __init__(self, n_samples: int = 50, noise_scale: float = 0.05):
        self.n_samples = n_samples
        self.noise_scale = noise_scale

    def sample(
        self,
        U: torch.Tensor,
        V: torch.Tensor
    ) -> UncertaintyResult:
        """
        Generate MC samples and compute uncertainty.

        Args:
            U: (d, r) tensor (on device)
            V: (d, r) tensor (on device)

        Returns:
            UncertaintyResult
        """
        device = U.device
        d, r = U.shape
        W_samples = np.zeros((self.n_samples, d, d))

        with torch.no_grad():
            for s in range(self.n_samples):
                U_noisy = U + torch.randn_like(U) * self.noise_scale * U.std()
                V_noisy = V + torch.randn_like(V) * self.noise_scale * V.std()
                W_s = (U_noisy @ V_noisy.T).cpu().numpy()
                W_samples[s] = W_s

        return self._aggregate(W_samples)

    def _aggregate(self, W_samples: np.ndarray) -> UncertaintyResult:
        """Aggregate MC samples."""
        K, d, _ = W_samples.shape

        adjacency = np.mean(W_samples, axis=0)
        edge_prob = np.mean((np.abs(W_samples) > 0.3).astype(float), axis=0)

        w_std = np.std(W_samples, axis=0)
        w_mean = np.abs(adjacency)
        cv = np.divide(w_std, w_mean + 1e-10, out=np.zeros_like(w_mean),
                      where=w_mean > 1e-10)
        cv = np.clip(cv, 0, 2)
        confidence = edge_prob * (1 - cv / 2)

        ci_lower = np.percentile(W_samples, 2.5, axis=0)
        ci_upper = np.percentile(W_samples, 97.5, axis=0)

        adj_thresh = adjacency * (np.abs(adjacency) > 0.3)
        n_high = int(np.sum(confidence > 0.8))
        n_total = int(np.sum(np.abs(adjacency) > 0.3))

        return UncertaintyResult(
            adjacency=adj_thresh,
            confidence=confidence,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            edge_probability=edge_prob,
            n_bootstrap=self.n_samples,
            n_high_confidence_edges=n_high,
            n_total_edges=n_total,
            method='mc_dropout'
        )
