"""causalscale V3: Unified Causal Discovery Platform.

Core API:
    import causalscale as cs
    model = cs.CausalDiscovery(data)
    model.fit()
    network = model.get_network()

Engines (6 methods):
    lowrank       — W = U @ V^T (d up to 100M)
    multi_scale   — W = sum U_s @ V_s^T (hierarchical)
    cluster_aware — Joint W + cluster (CAGate/SSCAGate)
    transformer   — Causal Transformer (attention-based)
    multimodal    — MM-CDSM (multi-omics consensus)
    full          — All + uncertainty + counterfactual
"""

from .api import CausalDiscovery, CausalNetwork
from .core.lowrank import LowRankGNN, train_lowrank_gnn
from .core.dag_constraint import dag_constraint, trace_expm
from .core.cluster_gate import ClusterAwareGate
from .core.engine import CausalDiscoveryEngine
from .core.transformer import CausalTransformer
from .core.multimodal import MultiModalNOTEARS

__version__ = "3.0.0"
__author__ = "Shuaidong Gao (ORCID: 0009-0004-5641-3581)"

__all__ = [
    "CausalDiscovery", "CausalNetwork",
    "CausalDiscoveryEngine", "CausalTransformer", "MultiModalNOTEARS",
    "LowRankGNN", "train_lowrank_gnn",
    "dag_constraint", "trace_expm",
    "ClusterAwareGate",
]
