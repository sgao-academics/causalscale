"""Gene Causal Network Visualization and Analysis.

Build and visualize gene regulatory networks from expression data.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from ..core.lowrank import train_lowrank_gnn


def gene_causal_network(
    expression: np.ndarray,
    gene_names: List[str],
    rank: int = 64,
    threshold: float = 0.3,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """Discover causal gene regulatory network from expression data.

    Args:
        expression: (n_samples, d_genes) expression matrix
        gene_names: list of gene symbols
        rank: factorization rank
        threshold: edge weight threshold
        device: 'cuda' or 'cpu'
        verbose: print progress

    Returns:
        dict with edges, adjacency, hub_genes, network_stats
    """
    result = train_lowrank_gnn(
        expression, rank=rank, threshold=threshold, device=device, verbose=verbose
    )
    W = result["adjacency"]
    d = W.shape[0]

    # Extract edges
    edges = []
    for i in range(d):
        for j in range(d):
            if i != j and abs(W[i, j]) > threshold:
                edges.append(
                    {
                        "source": gene_names[i],
                        "target": gene_names[j],
                        "weight": float(W[i, j]),
                    }
                )
    edges.sort(key=lambda x: -abs(x["weight"]))

    # Hub genes (top out-degree)
    out_degree = np.sum(np.abs(W) > threshold, axis=0)
    hub_idx = np.argsort(-out_degree)[:10]
    hub_genes = [
        {"gene": gene_names[int(i)], "out_degree": int(out_degree[int(i)])}
        for i in hub_idx
    ]

    # Network statistics
    stats = {
        "n_genes": d,
        "n_samples": expression.shape[0],
        "n_edges": len(edges),
        "density": len(edges) / (d * (d - 1)),
        "hub_genes": hub_genes,
    }

    if verbose:
        print(f"Gene network: {len(edges)} edges, "
              f"top hub: {hub_genes[0]['gene']} ({hub_genes[0]['out_degree']} targets)")

    return {"edges": edges, "adjacency": W, "stats": stats, "hub_genes": hub_genes}


def find_target_genes(
    network: Dict,
    source_gene: str,
    top_k: int = 20,
) -> List[Dict]:
    """Find causal targets of a given gene.

    Args:
        network: output from gene_causal_network()
        source_gene: query gene symbol
        top_k: max targets to return

    Returns:
        ranked list of {target, weight} dicts
    """
    targets = [
        e for e in network["edges"] if e["source"] == source_gene
    ]
    targets.sort(key=lambda x: -abs(x["weight"]))
    return targets[:top_k]


def find_regulators(
    network: Dict,
    target_gene: str,
    top_k: int = 20,
) -> List[Dict]:
    """Find causal regulators of a given gene.

    Args:
        network: output from gene_causal_network()
        target_gene: query gene symbol
        top_k: max regulators to return

    Returns:
        ranked list of {source, weight} dicts
    """
    regulators = [
        {"source": e["source"], "weight": e["weight"]}
        for e in network["edges"]
        if e["target"] == target_gene
    ]
    regulators.sort(key=lambda x: -abs(x["weight"]))
    return regulators[:top_k]
