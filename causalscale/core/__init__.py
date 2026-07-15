"""Core engines: V2 Enterprise + V1 Classic."""
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
]
