"""Core engines: V2 Enterprise + V1 Classic + V3.3 Specialized."""
from .engine import CausalDiscoveryEngine, EngineConfig, EngineResult
from .adaptive_rank import AutoRankSelector, estimate_effective_rank
from .multi_scale import MultiScaleLowRank
from .uncertainty import BootstrapEnsemble, StabilitySelector, MCDropoutEnsemble
from .optimization import MixedPrecisionTrainer, CosineScheduler
from .dag_utils import efficient_dag_constraint, counterfactual, granger_causality_test
from .theory import convergence_diagnostic, sample_complexity_estimate
from .lowrank import LowRankGNN, train_lowrank_gnn
from .dag_constraint import dag_constraint, trace_expm, note_ars_linear_h
from .cluster_gate import ClusterAwareGate, compute_cluster_gates
from .multibatch_engine import fit_multibatch
from .llm_prior_engine import fit_llm_prior
from .bayes_lowrank_engine import fit_bayes_lowrank
from .sc_causal_engine import fit_sc_causal

__all__ = [
    "CausalDiscoveryEngine", "EngineConfig", "EngineResult",
    "AutoRankSelector", "estimate_effective_rank",
    "MultiScaleLowRank",
    "BootstrapEnsemble", "StabilitySelector", "MCDropoutEnsemble",
    "MixedPrecisionTrainer", "CosineScheduler",
    "efficient_dag_constraint", "counterfactual", "granger_causality_test",
    "convergence_diagnostic", "sample_complexity_estimate",
    "LowRankGNN", "train_lowrank_gnn",
    "dag_constraint", "trace_expm", "note_ars_linear_h",
    "ClusterAwareGate", "compute_cluster_gates",
    "fit_multibatch", "fit_llm_prior",
    "fit_bayes_lowrank", "fit_sc_causal",
]
