"""
causalscale V3.0.0 — Unified Causal Discovery API
Backed by CausalDiscoveryEngine V2 (dual-patent).

Usage:
    import causalscale as cs
    model = cs.CausalDiscovery(data)
    model.fit()
    network = model.get_network()
    model.plot()
"""

import numpy as np
import torch
import warnings
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

_METHOD_MAP = {
    "lowrank": "lowrank",
    "multi_scale": "multi_scale",
    "cluster_aware": "cluster_aware",
    "gate": "cluster_aware",
    "transformer": "transformer",
    "ct": "transformer",
    "multimodal": "multimodal",
    "mm": "multimodal",
    "full": "full",
    "auto": "auto",
}


def _auto_method(d: int, n: int) -> str:
    if n < 200:
        return "cluster_aware"
    elif d <= 500:
        return "multi_scale"
    elif d > 1000:
        return "lowrank"
    else:
        return "multi_scale"


@dataclass
class CausalNetwork:
    adjacency: np.ndarray
    edges: List[Tuple[str, str, float]] = field(default_factory=list)
    edge_count: int = 0
    is_dag: bool = False
    n_vars: int = 0
    var_names: List[str] = field(default_factory=list)
    time_s: float = 0.0
    params: int = 0
    metadata: Dict = field(default_factory=dict)


class CausalDiscovery:
    """One-line causal discovery engine (V3.0.0).

    Backed by CausalDiscoveryEngine V2 with:
    - Adaptive rank selection (spectral + AIC/BIC + pruning)
    - Multi-scale decomposition: W = sum U_s @ V_s^T
    - Cluster-aware gate (CAGate/SSCAGate)
    - Uncertainty quantification (Bootstrap/Stability/MC Dropout)
    - Counterfactual inference (do-calculus)
    - Mixed precision training (FP16)

    Args:
        data: (n, d) array or CSV/TSV path
        method: 'auto' (default), 'lowrank', 'multi_scale', 'cluster_aware', 'full'
        rank: factorization rank, or 'auto' for adaptive
        device: 'cpu' (default) or 'cuda'
        var_names: optional variable names
        **kwargs: passed to EngineConfig (epochs, lr, threshold, etc.)
    """

    def __init__(
        self,
        data: Union[np.ndarray, str],
        method: str = "auto",
        rank: Union[int, str] = "auto",
        device: str = "cpu",
        var_names: Optional[List[str]] = None,
        **kwargs,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.device = device
        self.rank = rank
        self.var_names = var_names
        self.kwargs = kwargs

        # Load data
        if isinstance(data, str):
            import pandas as pd
            df = pd.read_csv(data, sep=None, engine="python", index_col=0)
            self.var_names = list(df.columns)
            self.X = df.values.astype(np.float32)
        else:
            self.X = np.asarray(data, dtype=np.float32)
            self.var_names = var_names or [f"V{i}" for i in range(self.X.shape[1])]

        # Data sanitization
        self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        col_std = self.X.std(axis=0)
        col_std[col_std < 1e-8] = 1.0
        self.X = (self.X - self.X.mean(axis=0)) / col_std

        self.n, self.d = self.X.shape

        # Validate
        if self.d < 2:
            raise ValueError(f"Need >= 2 variables, got {self.d}")
        if self.n < 10:
            raise ValueError(f"Need >= 10 samples, got {self.n}")

        # Auto method
        if method == "auto":
            method = _auto_method(self.d, self.n)
        self.method = _METHOD_MAP.get(method, method)

        # Build engine config kwargs
        engine_kwargs = {
            k: v for k, v in kwargs.items()
            if k in {"epochs", "lr", "threshold", "use_dag", "use_amp",
                     "compute_uncertainty", "n_bootstrap", "seed"}
        }
        if "verbose" in kwargs:
            engine_kwargs["verbose"] = kwargs["verbose"]

        self._engine = None
        self._result = None
        self._network = None
        self._fitted = False

    def _init_engine(self):
        from .core.engine import CausalDiscoveryEngine
        self._engine = CausalDiscoveryEngine(
            d=self.d,
            rank=self.rank,
            mode=self.method,
            device=self.device,
            **self.kwargs,
        )

    def fit(self, verbose: bool = True) -> "CausalDiscovery":
        import time
        t0 = time.time()

        if verbose:
            print(f"causalscale V3: method={self.method}, d={self.d}, "
                  f"n={self.n}, rank={self.rank}, device={self.device}")

        # ── Transformer mode (CT) ──
        if self.method == "transformer":
            from .core.transformer import CausalTransformer, fit_causal_transformer

            rt = self.rank if isinstance(self.rank, int) else min(64, self.d // 4)
            X_t = torch.tensor(self.X, dtype=torch.float32, device=self.device)
            model = CausalTransformer(d_vars=self.d, d_model=rt,
                                       n_heads=4, n_layers=2,
                                       lambda_dag=0.5, lr=0.001)
            epochs = self.kwargs.get("epochs", 200)
            fit_causal_transformer(model, X_t, n_epochs=epochs,
                                   batch_size=min(128, self.n),
                                   device=self.device, verbose=verbose)
            # Extract adjacency from trained attention
            with torch.no_grad():
                emb = model.encoder(X_t[:min(256, self.n)])
                _, adj = model.layers[0]['attention'](emb, return_adjacency=True)
                W = adj.mean(dim=0).cpu().numpy()
                # Average over batch for stability
                W_avg = np.zeros((self.d, self.d))
                batch_size = min(128, self.n)
                for b in range(0, min(512, self.n), batch_size):
                    end = min(b + batch_size, self.n)
                    emb_b = model.encoder(X_t[b:end])
                    _, adj_b = model.layers[0]['attention'](emb_b, return_adjacency=True)
                    W_avg += adj_b.mean(dim=0).cpu().numpy()
                W = W_avg / max(1, self.n // batch_size + 1)
            # Lower threshold for attention-based adjacency (softer weights)
            thresh = 0.1
            wc = int(np.sum(np.abs(W) > thresh))
            self._network = CausalNetwork(
                adjacency=W, edges=self._extract_edges(W, threshold=thresh),
                edge_count=wc,
                n_vars=self.d, var_names=self.var_names,
                time_s=time.time() - t0,
                metadata={"method": "transformer"}
            )
            self._fitted = True
            if verbose:
                print(f"  Found {self._network.edge_count} edges in {self._network.time_s:.1f}s")
            return self

        # ── Multimodal mode (MM-CDSM) ──
        if self.method == "multimodal":
            extra_data = self.kwargs.get("extra_data", [])
            if not extra_data:
                # No real multi-omics data: fallback to multi_scale
                if verbose:
                    print("  No extra modalities provided. Using multi_scale as fallback.")
                self.method = "multi_scale"
                return self.fit(verbose=verbose)

            X_t = torch.tensor(self.X, dtype=torch.float32, device=self.device)
            X_list = [X_t] + [torch.tensor(e.astype(np.float32), device=self.device)
                               for e in extra_data]
            n_mods = len(X_list)
            mod_names = self.kwargs.get("modality_names", [f"mod{i}" for i in range(n_mods)])
            dims = {n: int(X.shape[1]) for n, X in zip(mod_names, X_list)}
            from .core.multimodal import MultiModalNOTEARS
            mm = MultiModalNOTEARS(dims, lambda_consistency=0.1,
                                   consensus_threshold=0.33,
                                   lr=0.002,
                                   outer=self.kwargs.get("outer", 20),
                                   inner=self.kwargs.get("inner", 100))
            result = mm.fit(X_list, n_seeds=self.kwargs.get("n_seeds", 2),
                            device=self.device, verbose=verbose)
            W_list = result.get("per_modality_W", [])
            if W_list and len(W_list) > 0 and all(len(w.shape) == 2 for w in W_list):
                W_stack = torch.stack([w if isinstance(w, torch.Tensor) else torch.tensor(w)
                                        for w in W_list])
                W = torch.mean(W_stack, dim=0).cpu().numpy()
            else:
                W = np.zeros((self.d, self.d))
            wc = int(np.sum(np.abs(W) > 0.3))
            self._network = CausalNetwork(
                adjacency=W, edges=self._extract_edges(W),
                edge_count=wc, n_vars=self.d, var_names=self.var_names,
                time_s=time.time() - t0,
                metadata={"method": "multimodal", "modalities": n_mods,
                          "consistency": result.get("consistency_score", 0)}
            )
            self._fitted = True
            if verbose:
                print(f"  Found {wc} consensus edges in {self._network.time_s:.1f}s "
                      f"({n_mods} modalities)")
            return self

        # ── V2 Engine modes (lowrank, multi_scale, cluster_aware, full) ──
        if self._engine is None:
            self._init_engine()
        self._result = self._engine.fit(self.X)

        net = CausalNetwork(
            adjacency=self._result.adjacency,
            edges=self._extract_edges(self._result.adjacency),
            edge_count=self._result.edge_count,
            is_dag=bool(self._result.h_history[-1] < 0.01) if self._result.h_history else False,
            n_vars=self.d,
            var_names=self.var_names,
            time_s=self._result.training_time,
            params=getattr(self._result, "final_rank", self.rank),
            metadata={
                "method": self.method,
                "rank": self._result.final_rank,
                "convergence": self._result.convergence,
                "significance": self._result.significance,
            },
        )
        self._network = net
        self._fitted = True

        if verbose:
            print(f"  Found {net.edge_count} edges in {net.time_s:.1f}s")
            if self._result.uncertainty:
                print(f"  High-confidence edges (>=0.8): "
                      f"{self._result.uncertainty.n_high_confidence_edges}")

        return self

    def _extract_edges(self, W, threshold=0.3):
        edges = []
        for i in range(self.d):
            for j in range(self.d):
                if i != j and abs(W[i, j]) > threshold:
                    edges.append((self.var_names[i], self.var_names[j], float(W[i, j])))
        edges.sort(key=lambda x: -abs(x[2]))
        return edges

    def get_network(self, top_k=None) -> CausalNetwork:
        if not self._fitted:
            raise RuntimeError("Not fitted. Call .fit() first.")
        if top_k and self._network:
            self._network.edges = self._network.edges[:top_k]
        return self._network

    def get_adjacency(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        return self._network.adjacency

    def get_edges(self, confidence=0.0, names=None):
        if self._engine is None:
            raise RuntimeError("Not fitted.")
        return self._engine.get_edges(confidence=confidence, names=names or self.var_names)

    def predict(self, X_new):
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        return X_new @ self._network.adjacency

    def counterfactual(self, X, intervention, effect_vars=None):
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        return self._engine.counterfactual(X, intervention, effect_vars)

    def plot(self, save_path=None, top_k=50, **kwargs):
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.DiGraph()
        for src, tgt, w in self._network.edges[:top_k]:
            G.add_edge(src, tgt, weight=abs(w))

        pos = nx.spring_layout(G, k=2, seed=42)
        plt.figure(figsize=(12, 10))
        edge_colors = ["#2166ac" if w > 0 else "#b2182b"
                       for _, _, w in self._network.edges[:top_k]]
        edge_widths = [abs(w) * 3 for _, _, w in self._network.edges[:top_k]]
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color="#f0f0f0",
                               edgecolors="#333333", linewidths=1)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                               width=edge_widths, alpha=0.7,
                               arrowsize=15, connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_labels(G, pos, font_size=7)
        plt.title(f"Causal Network: {self._network.edge_count} edges "
                  f"(d={self.d}, method={self.method})", fontsize=14)
        plt.axis("off")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        plt.show()

    def summary(self) -> str:
        if not self._fitted:
            return "Not fitted."
        net = self._network
        lines = [
            f"CausalDiscovery V3 Summary",
            f"{'='*40}",
            f"Method: {self.method}",
            f"Variables: {self.d}",
            f"Samples: {self.n}",
            f"Rank: {net.params}",
            f"Edges: {net.edge_count}",
            f"Time: {net.time_s:.1f}s",
        ]
        if net.metadata.get("convergence"):
            c = net.metadata["convergence"]
            lines.append(f"Converged: {c.get('converged', 'N/A')}")
        if net.edges:
            lines.append(f"\nTop 10 edges:")
            for src, tgt, w in net.edges[:10]:
                arrow = "->" if w > 0 else "-|"
                lines.append(f"  {src} {arrow} {tgt}: {w:+.3f}")
        return "\n".join(lines)

    def generate_report(self, filepath=None) -> str:
        if self._engine is None:
            raise RuntimeError("Not fitted.")
        return self._engine.generate_report(filepath)

    def __repr__(self):
        status = "fitted" if self._fitted else "not fitted"
        return f"CausalDiscovery(d={self.d}, n={self.n}, method='{self.method}', {status})"
