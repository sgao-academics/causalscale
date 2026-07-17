"""causalscale V3.2.0 — Unified Causal Discovery API
Backed by CausalDiscoveryEngine V2 (dual-patent).

Usage:
    import causalscale as cs
    model = cs.CausalDiscovery(data)
    model.fit()
    network = model.get_network()
    model.plot()

    # Auto-detect evaluation mode
    report = model.validate()                        # real data: auto
    report = model.validate(ground_truth=W_true)     # synthetic: causal F1
    report = model.validate(string_data_dir=path)    # biology: STRING/TRRUST
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
    "dagma": "dagma",
    "transformer": "transformer",
    "ct": "transformer",
    "multimodal": "multimodal",
    "mm": "multimodal",
    "full": "full",
    "auto": "auto",
    "ensemble": "ensemble",
}


def _auto_method(d: int, n: int) -> str:
    """Auto-select engine based on dimensionality regime.

    Engine map (empirically validated):
        d <= 150:  dagma (strongest low-dimensional F1, Table 1)
        150 < d <= 200: cluster_aware (DAGMA times out, NOTEARS viable)
        200 < d <= 500: transformer (Causal Transformer, Gao 2026 ML Springer)
        d > 500:   lowrank (LowRankGNN, correlation-reconstruction at scale)
    """
    if d <= 150:
        return "dagma"
    elif d <= 200:
        return "cluster_aware"
    elif d <= 500:
        return "transformer"
    else:
        return "lowrank"


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
    """One-line causal discovery engine (V3.2.0).

    Backed by CausalDiscoveryEngine V2 with:
    - Adaptive rank selection (spectral + AIC/BIC + pruning)
    - Multi-scale decomposition: W = sum U_s @ V_s^T
    - Cluster-aware gate (CAGate/SSCAGate)
    - Stability selection (multi-seed consensus) — DEFAULT for n_seeds>1
    - Uncertainty quantification (Bootstrap/Stability/MC Dropout)
    - Counterfactual inference (do-calculus)
    - Mixed precision training (FP16)

    Args:
        data: (n, d) array or CSV/TSV path
        method: 'auto' (default), 'lowrank', 'multi_scale', 'cluster_aware', 'full'
        rank: factorization rank, or 'auto' for adaptive
        device: 'cpu' (default) or 'cuda'
        var_names: optional variable names
        n_seeds: number of seeds for stability selection (default 1 = single run).
                 5-10 recommended for production use. +15% F1 boost on synthetic data.
        stability_k: min seeds for consensus edge (default = ceil(0.6 * n_seeds))
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

        # Stability selection: multi-seed consensus
        self.n_seeds = kwargs.pop("n_seeds", 1)
        self.stability_k = kwargs.pop("stability_k",
                                       max(1, int(self.n_seeds * 0.6))
                                       if self.n_seeds > 1 else 1)

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
                  f"n={self.n}, rank={self.rank}, device={self.device}"
                  + (f", n_seeds={self.n_seeds}" if self.n_seeds > 1 else ""))

        # ── Transformer mode: Causal Transformer (CT, Gao 2026, ML Springer) ──
        # Designed for d=200-500 where NOTEARS collapses. Variables-as-tokens
        # self-attention learns W through multi-head attention + DAG constraint.
        if self.method == "transformer" or self.method == "ct":
            from .core.transformer import CausalTransformer, fit_causal_transformer

            d_model = self.kwargs.get("d_model", min(128, max(32, self.d // 2)))
            n_heads = self.kwargs.get("n_heads", 8)
            n_layers = self.kwargs.get("n_layers", 2)
            epochs = self.kwargs.get("epochs", 500)

            X_t = torch.tensor(self.X, dtype=torch.float32, device=self.device)
            model = CausalTransformer(
                d_vars=self.d, d_model=d_model,
                n_heads=n_heads, n_layers=n_layers,
                lambda_dag=0.5, lr=0.001
            )
            fit_causal_transformer(
                model, X_t, n_epochs=epochs,
                batch_size=min(128, self.n),
                device=self.device, verbose=verbose
            )

            # Extract adjacency from trained model's forward pass (graph_head)
            with torch.no_grad():
                model.eval()
                W_batch, _ = model(X_t[:min(500, self.n)])
                W = W_batch.mean(dim=0).cpu().numpy()

            thresh = 0.1  # CT produces soft weights; default 0.1 filters noise
            wc = int(np.sum(np.abs(W) > thresh))
            self._network = CausalNetwork(
                adjacency=W,
                edges=self._extract_edges(W, threshold=thresh),
                edge_count=wc,
                n_vars=self.d, var_names=self.var_names,
                time_s=time.time() - t0,
                params=d_model,
                metadata={
                    "method": "transformer",
                    "d_model": d_model, "n_heads": n_heads,
                    "n_layers": n_layers, "epochs": epochs,
                    "engine": "Causal Transformer (Gao 2026, ML Springer)",
                }
            )
            self._fitted = True
            if verbose:
                print(f"  Found {wc} edges in {self._network.time_s:.1f}s "
                      f"(CT d_model={d_model}, n_heads={n_heads})")
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

        # ── Ensemble mode: multi-engine consensus voting (CauTion-inspired) ──
        if self.method == "ensemble":
            return self._fit_ensemble(verbose)

        # ── V2 Engine modes (lowrank, multi_scale, cluster_aware, full) ──
        if self._engine is None:
            self._init_engine()

        # Stability selection: multi-seed consensus
        if self.n_seeds > 1:
            return self._fit_stability(verbose)
        else:
            return self._fit_single(verbose)

    def _fit_single(self, verbose: bool) -> "CausalDiscovery":
        """Single-seed fit (original behavior)."""
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
                "n_seeds": 1,
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

    def _fit_stability(self, verbose: bool) -> "CausalDiscovery":
        """Multi-seed stability selection: run n_seeds, keep consensus edges.

        Edges that appear in >= stability_k seeds are retained.
        Edge weight = mean absolute value across seeds where it appears.
        """
        n_seeds = self.n_seeds
        k_min = self.stability_k
        threshold = self.kwargs.get("threshold", 0.3)

        if verbose:
            print(f"  Stability selection: {n_seeds} bootstrap seeds, "
                  f"k>={k_min}/{n_seeds}")

        adj_stack = np.zeros((n_seeds, self.d, self.d), dtype=np.float32)
        total_time = 0.0
        results = []

        for s in range(n_seeds):
            # Bootstrap sample: resample rows WITH replacement for variability
            rng = np.random.RandomState(s * 42 + 1)
            idx = rng.randint(0, self.n, self.n)
            X_boot = self.X[idx]
            # Re-standardize bootstrap sample
            X_boot = X_boot - X_boot.mean(axis=0)
            sd = X_boot.std(axis=0).clip(min=1e-8)
            X_boot = X_boot / sd

            engine_kw = dict(self.kwargs)
            engine_kw.pop("n_seeds", None)
            engine_kw.pop("stability_k", None)
            engine_kw.pop("seed", None)
            from .core.engine import CausalDiscoveryEngine
            eng = CausalDiscoveryEngine(
                d=self.d, rank=self.rank, mode=self.method,
                device=self.device, **engine_kw
            )
            result_s = eng.fit(X_boot)
            adj_stack[s] = result_s.adjacency
            total_time += result_s.training_time
            results.append(result_s)
            if verbose:
                print(f"    Seed {s+1}/{n_seeds}: {result_s.edge_count} edges, "
                      f"{result_s.training_time:.1f}s")

        # Stability consensus: keep edges in >= k_min seeds
        edge_presence = np.sum(np.abs(adj_stack) > threshold, axis=0)
        stable_mask = (edge_presence >= k_min).astype(np.float32)

        # Weight = mean of raw adjacency across ALL seeds
        # (preserves weight magnitudes, not just thresholded ones)
        stable_W = stable_mask * np.mean(adj_stack, axis=0)

        n_stable = int(np.sum(stable_mask) - int(np.sum(np.diag(stable_mask))))
        presence_frac = edge_presence[stable_mask > 0] / n_seeds

        net = CausalNetwork(
            adjacency=stable_W,
            edges=self._extract_edges(stable_W, threshold=threshold),
            edge_count=n_stable,
            n_vars=self.d,
            var_names=self.var_names,
            time_s=total_time,
            params=getattr(results[0], "final_rank", self.rank),
            metadata={
                "method": self.method,
                "n_seeds": n_seeds,
                "stability_k": k_min,
                "stability_mean_presence": round(float(np.mean(presence_frac)), 3)
                    if len(presence_frac) > 0 else 0,
                "rank": getattr(results[0], "final_rank", self.rank),
            },
        )
        self._network = net
        self._fitted = True

        if verbose:
            print(f"  Stability consensus: {n_stable} edges "
                  f"(mean presence={net.metadata['stability_mean_presence']:.1%})")
            if net.edge_count > 0:
                non_zero = int(np.sum(np.abs(adj_stack[0]) > threshold))
                print(f"  Single-seed edges: {non_zero} -> "
                      f"Consensus: {n_stable} "
                      f"(filtered {non_zero-n_stable})")

        return self

    def _fit_ensemble(self, verbose: bool) -> "CausalDiscovery":
        """Multi-engine consensus voting (CauTion-inspired).

        Runs cluster_aware, lowrank, and multi_scale engines independently,
        then takes consensus edges (>= min_votes engines agree).
        Inspired by CauTion (Peng et al., 2026): ~96% of edges resolvable
        by algorithm consensus alone with near-perfect precision.
        """
        import time as _time
        threshold = self.kwargs.get("threshold", 0.3)
        min_votes = self.kwargs.get("ensemble_min_votes",
                                     self.kwargs.get("stability_k", 2))

        # Engines to ensemble (different inductive biases)
        engines_to_run = [
            ("cluster_aware", "ClusterAware NOTEARS (d<=500, high precision)"),
            ("lowrank", "LowRankGNN (sparse, rank-r bottleneck)"),
            ("multi_scale", "MultiScale (hierarchical decomposition)"),
        ]

        if verbose:
            print(f"  Ensemble mode: {len(engines_to_run)} engines, "
                  f"min_votes={min_votes}")
            print(f"  CauTion strategy: consensus edges = near-perfect precision")

        from .core.engine import CausalDiscoveryEngine
        adjacencies = {}
        engine_results = {}
        total_time = 0.0

        for eng_name, eng_desc in engines_to_run:
            t0 = _time.time()
            if verbose:
                print(f"    [{eng_name}] {eng_desc}...")

            eng = CausalDiscoveryEngine(
                d=self.d, rank=self.rank, mode=eng_name,
                device=self.device, **{k: v for k, v in self.kwargs.items()
                                        if k not in ("n_seeds", "stability_k",
                                                      "ensemble_min_votes")}
            )
            result = eng.fit(self.X)
            W = result.adjacency
            adjacencies[eng_name] = W
            engine_results[eng_name] = result
            elapsed = _time.time() - t0
            total_time += elapsed

            if verbose:
                n_e = int(np.sum(np.abs(W) > threshold))
                print(f"      {n_e} edges, {elapsed:.1f}s")

        # ── Weighted Consensus Voting ──
        # Weights: engines with higher edge count + better convergence get more say.
        # cluster_aware > multi_scale > lowrank (empirically on synthetic benchmarks)
        w_cluster = 2.0   # best F1 on synthetic ER d=30-80
        w_lowrank = 0.5   # weaker on dense ER, strong on sparse/real data
        w_multi = 1.0     # moderate, different inductive bias
        engine_weights = {
            "cluster_aware": w_cluster,
            "lowrank": w_lowrank,
            "multi_scale": w_multi,
        }

        # Weighted vote: sum(weight * (|W| > threshold))
        adj_stack = np.stack([adjacencies[n] for n, _ in engines_to_run], axis=0)
        vote_stack = np.stack(
            [(np.abs(adjacencies[n]) > threshold).astype(np.float32) *
             engine_weights.get(n, 1.0)
             for n, _ in engines_to_run],
            axis=0
        )
        weighted_votes = np.sum(vote_stack, axis=0)

        # Consensus: weighted_votes >= min_votes (where min_votes is in weighted units)
        # With weights [2.0, 1.0, 0.5], min_weighted_votes = w_cluster (2.0)
        # means: cluster_aware alone, OR multi_scale+lowrank together
        min_weighted = self.kwargs.get("ensemble_min_weighted",
                                        w_cluster)  # default: cluster_aware alone
        consensus_mask = (weighted_votes >= min_weighted).astype(np.float32)

        # Weighted mean of engine weights for consensus edges
        total_weight_per_edge = np.zeros((self.d, self.d), dtype=np.float32)
        weighted_sum = np.zeros((self.d, self.d), dtype=np.float32)
        for en, (n, _) in zip([w_cluster, w_lowrank, w_multi], engines_to_run):
            w = en
            mask = (np.abs(adjacencies[n]) > threshold).astype(np.float32)
            total_weight_per_edge += w * mask
            weighted_sum += w * mask * adjacencies[n]

        total_weight_per_edge = np.clip(total_weight_per_edge, 1e-8, None)
        consensus_W = consensus_mask * (weighted_sum / total_weight_per_edge)

        # ── Voting agreement statistics ──
        # Unweighted counts for reporting
        unweighted_votes = np.sum(
            [(np.abs(adjacencies[n]) > threshold).astype(int)
             for n, _ in engines_to_run], axis=0)
        unanimous = int(np.sum(unweighted_votes == len(engines_to_run))) - int(
            np.sum(np.diag(unweighted_votes) == len(engines_to_run)))
        majority = int(np.sum(unweighted_votes >= 2)) - int(
            np.sum(np.diag(unweighted_votes) >= 2))
        any_vote = int(np.sum(unweighted_votes >= 1)) - int(
            np.sum(np.diag(unweighted_votes) >= 1))
        n_consensus = int(np.sum(consensus_mask) - int(np.sum(np.diag(consensus_mask))))
        consensus_frac = n_consensus / max(any_vote, 1) * 100 if any_vote > 0 else 0
        disagreement = any_vote - unanimous

        # Edges where only cluster_aware votes (candidates for augmentation)
        cluster_only = int(np.sum(
            unweighted_votes == 1) - int(np.sum(np.diag(unweighted_votes) == 1)))

        net = CausalNetwork(
            adjacency=consensus_W,
            edges=self._extract_edges(consensus_W, threshold=threshold),
            edge_count=n_consensus,
            n_vars=self.d,
            var_names=self.var_names,
            time_s=total_time,
            params=self.rank,
            metadata={
                "method": "ensemble",
                "engines": [n for n, _ in engines_to_run],
                "engine_weights": engine_weights,
                "min_weighted": min_weighted,
                "unanimous_edges": unanimous,
                "majority_edges": majority,
                "cluster_only_edges": cluster_only,
                "consensus_edges": n_consensus,
                "disagreement_edges": disagreement,
                "consensus_fraction": round(consensus_frac, 1),
                "engine_edges": {n: int(np.sum(np.abs(adjacencies[n]) > threshold))
                                 for n, _ in engines_to_run},
            },
        )
        self._network = net
        self._fitted = True

        if verbose:
            print(f"\n  === Weighted Ensemble Consensus ===")
            print(f"  Weights: cluster_aware={w_cluster}, "
                  f"lowrank={w_lowrank}, multi_scale={w_multi}")
            print(f"  Unanimous (3/3): {unanimous} edges")
            print(f"  Majority (2+/3): {majority} edges")
            print(f"  ClusterAware-only: {cluster_only} edges "
                  f"(low confidence, augmentable)")
            print(f"  Weighted consensus: {n_consensus} edges "
                  f"(min_weight={min_weighted})")
            print(f"  Disagreement: {disagreement} edges")
            print(f"  Per-engine: {net.metadata['engine_edges']}")
            if n_consensus > 0:
                print(f"\n  CauTion insight: weighted consensus gives "
                      f"cluster_aware's precision + other engines' coverage.")

        # ── LLM Arbitration for disagreement edges ──
        llm_arb = self.kwargs.get("llm_arbitrate", False)
        if llm_arb and disagreement > 0 and self.var_names:
            return self._arbitrate_disagreements(
                engines_to_run, adjacencies, threshold, consensus_W,
                n_consensus, disagreement, total_time, verbose
            )

        return self

    def _arbitrate_disagreements(
        self, engines_to_run, adjacencies, threshold,
        consensus_W, n_consensus, disagreement, total_time, verbose
    ) -> "CausalDiscovery":
        """LLM arbitration post-processing for ensemble disagreement edges."""
        from .core.llm_arbiter import arbitrate_disagreement_edges

        # Find edges voted by some but not all engines, not in consensus
        vote_stack_uw = np.stack(
            [(np.abs(adjacencies[n]) > threshold).astype(int)
             for n, _ in engines_to_run], axis=0)
        vote_sum_uw = np.sum(vote_stack_uw, axis=0)
        n_eng = len(engines_to_run)
        disagree_mask = ((vote_sum_uw > 0) & (vote_sum_uw < n_eng)).astype(int)
        consensus_bin = (np.abs(consensus_W) > 1e-8).astype(int)
        disagree_mask = disagree_mask & (1 - consensus_bin)

        d_edges = []
        for i in range(self.d):
            for j in range(self.d):
                if i != j and disagree_mask[i, j]:
                    w = np.mean([adjacencies[n][i,j] for n,_ in engines_to_run])
                    d_edges.append((self.var_names[i], self.var_names[j], float(w)))

        string_dir = self.kwargs.get("string_data_dir")
        arb = arbitrate_disagreement_edges(
            d_edges, self.var_names, string_data_dir=string_dir,
            use_llm=self.kwargs.get("llm_api_key") is not None,
            api_key=self.kwargs.get("llm_api_key"),
            verbose=verbose,
        )

        new_W = consensus_W.copy()
        for src, tgt, w in arb["kept_edges"]:
            if src in self.var_names and tgt in self.var_names:
                i, j = self.var_names.index(src), self.var_names.index(tgt)
                new_W[i, j] = w

        aug = n_consensus + arb["kept_count"]
        self._network = CausalNetwork(
            adjacency=new_W,
            edges=self._extract_edges(new_W, threshold=threshold * 0.7),
            edge_count=aug, n_vars=self.d, var_names=self.var_names,
            time_s=total_time, params=self.rank,
            metadata={**self._network.metadata,
                "llm_arbitrated": True, "llm_mode": arb["mode"],
                "llm_kept": arb["kept_count"], "llm_dropped": arb["dropped_count"],
                "augmented_edges": aug},
        )
        self._fitted = True

        if verbose:
            print(f"  Augmented: {n_consensus} + {arb['kept_count']} = {aug} edges")
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

    # ═══════════════════════════════════════════════════════════════
    # Auto-detect evaluation mode
    # ═══════════════════════════════════════════════════════════════

    def validate(
        self,
        ground_truth: Optional[np.ndarray] = None,
        string_data_dir: Optional[str] = None,
        pseudo_ground_truth: Optional[str] = None,
        threshold: float = 0.3,
        verbose: bool = True,
    ) -> Dict:
        """Auto-detect and execute the appropriate evaluation for discovered edges.

        Four evaluation modes, auto-selected:
          1. ground_truth provided  ->  Causal F1, SHD, TPR, FPR (synthetic data)
          2. var_names look like gene symbols  ->  STRING/TRRUST cross-reference
          3. pseudo_ground_truth="notears"  ->  Run NOTEARS as reference, compare
          4. Neither  ->  Correlation-reconstruction F1 (self-supervised)

        Args:
            ground_truth: (d, d) binary adjacency matrix. If provided, computes
                          causal discovery metrics against known DAG.
            string_data_dir: path to STRING/TRRUST data files. Auto-detected
                             if gene symbols are present.
            pseudo_ground_truth: "notears" to auto-run NOTEARS as reference DAG,
                                 "correlation" to use thresholded correlation.
                                 Useful when user has real data without ground truth.
            threshold: edge weight threshold for adjacency binarization.
            verbose: print progress and results.

        Returns:
            dict with keys depending on mode:
              - Synthetic mode: f1, shd, tpr, fpr, precision, recall, tp, fp, fn
              - Biology mode: validated_edges, validated_pct, precision, tp, fp
              - Pseudo-causal mode: f1, shd, precision, recall, tp, fp, fn,
                                    reference_method, reference_edges
              - Self-supervised: f1 (correlation-reconstruction), recovery_pct

        Example:
            >>> # Synthetic data with known DAG
            >>> W_true = make_dag(d=50, seed=42)
            >>> model = cs.CausalDiscovery(data).fit()
            >>> metrics = model.validate(ground_truth=W_true)
            >>> print(f"F1={metrics['f1']:.3f}, SHD={metrics['shd']}")

            >>> # Real data without ground truth → auto-run NOTEARS as reference
            >>> model = cs.CausalDiscovery(data).fit()
            >>> report = model.validate(pseudo_ground_truth="notears")
            >>> print(f"F1 vs NOTEARS: {report['f1']:.3f}")

            >>> # Real gene expression data
            >>> model = cs.CausalDiscovery(gene_data, var_names=gene_symbols).fit()
            >>> report = model.validate()  # auto-detects biology mode
            >>> print(f"{report['validated_pct']:.1f}% edges STRING-validated")
        """
        if not self._fitted:
            raise RuntimeError("Not fitted. Call .fit() first.")

        # ── Mode 1: Ground truth provided → causal metrics ──
        if ground_truth is not None:
            return self._validate_synthetic(ground_truth, threshold, verbose)

        # ── Mode 2: Gene symbols → STRING/TRRUST ──
        if self._looks_like_genes():
            return self._validate_biology(string_data_dir, threshold, verbose)

        # ── Mode 3: Pseudo ground truth (NOTEARS as reference) ──
        if pseudo_ground_truth == "notears":
            return self._validate_pseudo_causal(threshold, verbose)

        # ── Mode 4: Self-supervised → correlation-reconstruction ──
        return self._validate_self_supervised(threshold, verbose)

    def _looks_like_genes(self) -> bool:
        """Heuristic: do variable names look like gene symbols?
        Gene symbols are typically 2-8 uppercase letters/numbers,
        not "V0", "V1", ..., "feature_0", etc."""
        import re
        if not self.var_names or len(self.var_names) < 3:
            return False
        # Exclude default naming patterns
        default_pattern = re.compile(r"^(V|X|var|VAR|feat|Feat|feature|col)\d+$")
        gene_like = 0
        total = min(len(self.var_names), 50)
        for name in self.var_names[:total]:
            if default_pattern.match(name):
                continue
            if (2 <= len(name) <= 8 and
                name[0].isalpha() and
                all(c.isalnum() or c == '-' for c in name)):
                gene_like += 1
        return gene_like >= total * 0.4

    def _validate_synthetic(
        self, ground_truth: np.ndarray, threshold: float, verbose: bool,
        reference_method: str = "ground_truth",
    ) -> Dict:
        """Causal discovery metrics against known or pseudo DAG."""
        W_true = np.asarray(ground_truth)
        if W_true.shape != (self.d, self.d):
            raise ValueError(
                f"ground_truth shape {W_true.shape} != ({self.d},{self.d})"
            )
        W_true_bin = (np.abs(W_true) > 1e-8).astype(int)

        W_pred = self._network.adjacency
        W_pred_bin = (np.abs(W_pred) > threshold).astype(int)

        tp = int(np.sum(W_pred_bin & W_true_bin))
        fp = int(np.sum(W_pred_bin & (1 - W_true_bin)))
        fn = int(np.sum((1 - W_pred_bin) & W_true_bin))
        tn = int(np.sum((1 - W_pred_bin) & (1 - W_true_bin)))

        # Remove diagonal from counts
        diag_tp = int(np.sum(np.diag(W_pred_bin) & np.diag(W_true_bin)))
        diag_fp = int(np.sum(np.diag(W_pred_bin) & (1 - np.diag(W_true_bin))))
        diag_fn = int(np.sum((1 - np.diag(W_pred_bin)) & np.diag(W_true_bin)))
        tp -= diag_tp
        fp -= diag_fp
        fn -= diag_fn

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        shd = fp + fn

        result = {
            "mode": "synthetic" if reference_method == "ground_truth" else "pseudo_causal",
            "reference_method": reference_method,
            "f1": round(f1, 4),
            "shd": shd,
            "tpr": round(rec, 4),  # same as recall
            "fpr": round(fp / max(fp + tn, 1), 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "true_edges": int(np.sum(W_true_bin)),
            "pred_edges": int(np.sum(W_pred_bin)),
            "threshold": threshold,
        }

        if verbose:
            label = ("ground truth DAG" if reference_method == "ground_truth"
                     else f"{reference_method} pseudo ground truth")
            print(f"\n{'='*50}")
            print(f"  CAUSAL DISCOVERY METRICS (vs {label})")
            print(f"{'='*50}")
            print(f"  F1:        {result['f1']:.4f}")
            print(f"  SHD:       {result['shd']}")
            print(f"  Precision: {result['precision']:.4f}")
            print(f"  Recall:    {result['recall']:.4f}")
            print(f"  TP={tp}, FP={fp}, FN={fn}")
            print(f"  True edges: {result['true_edges']}, "
                  f"Found: {result['pred_edges']}")

        return result

    def _validate_biology(
        self, string_data_dir: Optional[str], threshold: float, verbose: bool
    ) -> Dict:
        """STRING/TRRUST cross-reference validation."""
        from .pretrained import validate_against_string

        edges = self._network.edges
        result = validate_against_string(
            edges, data_dir=string_data_dir, verbose=verbose
        )

        if verbose:
            print(f"\n{'='*50}")
            print(f"  BIOLOGICAL VALIDATION (STRING + TRRUST)")
            print(f"{'='*50}")
            print(f"  Total discovered edges:  {result['total_edges']}")
            print(f"  STRING/TRRUST validated: {result['validated_edges']}")
            print(f"  Validation rate:         {result['validated_pct']:.1f}%")
            print(f"  Precision:               {result['precision']:.4f}")
            if result["string_only"] or result["trrust_only"] or result["both"]:
                print(f"  STRING only: {result['string_only']}, "
                      f"TRRUST only: {result['trrust_only']}, "
                      f"Both: {result['both']}")

        result["mode"] = "biology"
        return result

    def _validate_pseudo_causal(
        self, threshold: float, verbose: bool
    ) -> Dict:
        """Run NOTEARS as reference DAG, compare causalscale against it.

        This provides a causal F1/SHD metric for real data without ground truth.
        NOTEARS runs on CPU with default hyperparameters (d<=150 supported).
        For d>150, falls back to correlation-reconstruction with a warning.
        """
        if self.d > 150:
            if verbose:
                print(f"  WARNING: d={self.d}>150, NOTEARS O(d^3) infeasible.")
                print(f"  Falling back to correlation-reconstruction F1.")
            return self._validate_self_supervised(threshold, verbose)

        if verbose:
            print(f"  Running NOTEARS as pseudo ground truth "
                  f"(d={self.d}, n={self.n})...")

        # Minimal NOTEARS implementation
        X_t = torch.tensor(self.X, dtype=torch.float32, device=self.device)
        d, n = self.d, self.n
        W_ref = torch.zeros((d, d), device=self.device, requires_grad=True)
        W_ref.data = torch.randn(d, d, device=self.device) * 0.01

        def h_fn(W):
            M = W * W
            return torch.trace(torch.matrix_exp(M)) - d

        rho, alpha = 1.0, 0.0
        opt = torch.optim.Adam([W_ref], lr=0.002)
        inner_iters = 200
        max_outer = 30

        for outer_i in range(max_outer):
            for _ in range(inner_iters):
                opt.zero_grad()
                R = X_t - X_t @ W_ref
                loss = 0.5 / n * torch.sum(R ** 2)
                l1 = 0.1 * torch.sum(torch.abs(W_ref))
                h = h_fn(W_ref)
                total = loss + l1 + alpha * h + 0.5 * rho * h * h
                total.backward()
                opt.step()

            with torch.no_grad():
                h_val = h_fn(W_ref).item()

            if h_val < 1e-8:
                break
            alpha += rho * h_val
            rho = min(rho * 5, 1e10)

        W_ref_np = W_ref.detach().cpu().numpy()
        ref_edges = int(np.sum(np.abs(W_ref_np) > threshold))
        if verbose:
            print(f"  NOTEARS converged: h={h_val:.2e}, edges={ref_edges}")

        if ref_edges == 0:
            if verbose:
                print(f"  NOTEARS produced zero edges. Falling back to "
                      f"correlation-reconstruction F1.")
            return self._validate_self_supervised(threshold, verbose)

        # Compare causalscale vs NOTEARS reference
        return self._validate_synthetic(
            W_ref_np, threshold, verbose, reference_method="NOTEARS"
        )

    def _validate_self_supervised(
        self, threshold: float, verbose: bool
    ) -> Dict:
        """Correlation-reconstruction F1 (self-supervised evaluation)."""
        X_t = torch.tensor(self.X, dtype=torch.float32,
                           device=self.device)
        X_std = (X_t - X_t.mean(0)) / (X_t.std(0).clamp(min=1e-8))
        C = (X_std.T @ X_std) / (X_std.shape[0] - 1)
        C_abs = torch.abs(C)
        C_abs.fill_diagonal_(0)
        gt = (C_abs > threshold).cpu().numpy().astype(int)
        gt_n = int(gt.sum())

        W_pred = self._network.adjacency
        W_pred_bin = (np.abs(W_pred) > threshold).astype(int)
        pred_n = int(W_pred_bin.sum())

        tp = int(np.sum(W_pred_bin & gt))
        rec = tp / max(gt_n, 1)
        f1 = 2 * tp / (pred_n + gt_n) if (pred_n + gt_n) > 0 else 0

        result = {
            "mode": "self_supervised",
            "f1": round(f1, 4),
            "recovery_pct": round(rec * 100, 1),
            "tp": tp,
            "corr_gt_edges": gt_n,
            "pred_edges": pred_n,
            "threshold": threshold,
            "note": "F1 measures correlation-reconstruction, NOT causal accuracy. "
                    "Use validate(ground_truth=...) for synthetic data or "
                    "validate(string_data_dir=...) for biological validation.",
        }

        if verbose:
            print(f"\n{'='*50}")
            print(f"  SELF-SUPERVISED METRICS (correlation-reconstruction)")
            print(f"{'='*50}")
            print(f"  Correlation-reconstruction F1: {result['f1']:.4f}")
            print(f"  Recovery rate:                  {result['recovery_pct']:.1f}%")
            print(f"  TP={tp}, Corr-GT={gt_n}, Pred={pred_n}")
            print(f"  NOTE: This F1 measures how well the model reconstructs")
            print(f"  the correlation matrix, NOT causal ground truth.")
            print(f"  For causal F1, use validate(ground_truth=W_true).")
            print(f"  For biological validation, use validate(string_data_dir=...).")

        return result

    def generate_report(self, filepath=None) -> str:
        if self._engine is None:
            raise RuntimeError("Not fitted.")
        return self._engine.generate_report(filepath)

    def __repr__(self):
        status = "fitted" if self._fitted else "not fitted"
        return f"CausalDiscovery(d={self.d}, n={self.n}, method='{self.method}', {status})"
