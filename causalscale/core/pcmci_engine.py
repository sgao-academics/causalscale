"""
PCMCI Time-Series Engine (v3.2.0).

Integrates Runge et al.'s PCMCI algorithm (Science Advances, 2019) into the
causalscale ecosystem with standard fit() / get_edges() API.

PCMCI addresses the time-series gap: for data with temporal ordering
(e.g., monthly climate measurements, fMRI BOLD signals, financial time
series), it discovers lagged causal relationships using conditional
independence tests and the PC algorithm.

Reference:
  Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., & Sejdinovic, D. (2019).
  Detecting and quantifying causal associations in large nonlinear time
  series datasets. Science Advances, 5(11), eaau4996.
"""

import numpy as np
from typing import List, Tuple
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import data_processing as pp


class PCMCIEngine:
    """PCMCI time-series causal discovery wrapper."""
    
    def __init__(
        self,
        X: np.ndarray,
        tau_max: int = 5,
        pc_alpha: float = 0.05,
        var_names: list = None,
    ):
        T, N = X.shape
        self.d = N
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.var_names = var_names or [f"V{i}" for i in range(N)]
        
        self.dataframe = pp.DataFrame(X)
        self.ci_test = ParCorr(significance="analytic")
        self.pcmci = PCMCI(
            dataframe=self.dataframe,
            cond_ind_test=self.ci_test,
            verbosity=0,
        )
        self._results = None
    
    def fit(self) -> dict:
        """Run PCMCI. Returns dict with p_matrix, val_matrix, graph."""
        self._results = self.pcmci.run_pcmci(
            tau_max=self.tau_max, pc_alpha=self.pc_alpha
        )
        return self._results
    
    def get_edges(self, alpha: float = None) -> List[Tuple[str, str, int, float]]:
        """Get significant causal edges sorted by p-value."""
        if self._results is None:
            self.fit()
        alpha = alpha or self.pc_alpha
        edges = []
        for lag in range(self.tau_max + 1):
            for i in range(self.d):
                for j in range(self.d):
                    if i != j:
                        p = float(self._results["p_matrix"][lag, i, j])
                        if p < alpha:
                            edges.append((self.var_names[i], self.var_names[j], lag, p))
        edges.sort(key=lambda x: x[3])
        return edges
    
    def summary(self) -> str:
        edges = self.get_edges()
        by_lag = {}
        for _, _, lag, _ in edges:
            by_lag[lag] = by_lag.get(lag, 0) + 1
        return (
            f"PCMCI (tau_max={self.tau_max}, alpha={self.pc_alpha}): "
            f"{len(edges)} edges, by lag={dict(sorted(by_lag.items()))}"
        )
