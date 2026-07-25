"""causalscale V3.3: Unified Causal Discovery Platform — 11 engines under one API.

Core API:
    import causalscale as cs
    model = cs.CausalDiscovery(data)
    model.fit()
    network = model.get_network()

    # Biological validation
    edges = model.get_edges()
    result = cs.validate_against_string(edges)

Engines (11 methods):
    dagma         — DAGMA (log-det acyclicity, d <= 150)
    cluster_aware — Verified NOTEARS with exact DAG constraint (d <= 200)
    transformer   — Causal Transformer (attention-based, d=200-500)
    lowrank       — LowRankGNN W = U @ V^T (genome-scale)
    multibatch    — Dataset-specific residual adapters
    llm_prior     — STRING-derived edge prior injection
    bayes_lowrank — Bayesian bootstrap + low-rank DAG (uncertainty)
    sc_causal     — NB-LR CI test for single-cell RNA-seq
    multiscale    — Multi-scale low-rank decomposition
    multimodal    — MM-CDSM (multi-omics consensus)
    ensemble      — Weighted multi-engine voting
"""

from .api import CausalDiscovery, CausalNetwork
from .core.lowrank import LowRankGNN, train_lowrank_gnn
from .core.dag_constraint import dag_constraint, trace_expm
from .core.cluster_gate import ClusterAwareGate
from .core.engine import CausalDiscoveryEngine
from .core.transformer import CausalTransformer
from .core.multimodal import MultiModalNOTEARS
from .pretrained import validate_against_string

__version__ = "3.3.0"
__author__ = "Shuaidong Gao (ORCID: 0009-0004-5641-3581)"

__all__ = [
    "CausalDiscovery", "CausalNetwork",
    "CausalDiscoveryEngine", "CausalTransformer", "MultiModalNOTEARS",
    "LowRankGNN", "train_lowrank_gnn",
    "dag_constraint", "trace_expm",
    "ClusterAwareGate",
    "validate_against_string",
]
