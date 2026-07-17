"""
Two-Stage Causal Discovery Pipeline (v3.2.0).

Strategy: When d is too large for DAG-constrained discovery,
use the DAG engines (DAGMA/ClusterAware) as a "seed finder"
on the most promising variable subset, then expand outward
with LowRankGNN.

Stage 1: DAG engine on top-k variables → high-confidence DAG edges
Stage 2: For each discovered causal pair (A->B), expand neighborhood
         to all variables correlated with A or B, run LowRankGNN
         on the expanded subgraph → genome-scale edge discovery

Example (genomics):
  User has d=17,787 genes, wants causal edges for ARID1A-MTOR.
  Stage 1: ClusterAware on top-200 variable genes → finds ARID1A->MTOR
  Stage 2: For ARID1A, expand to all genes with |corr| > 0.3 (~500),
           run LowRankGNN → discovers 200+ edges involving ARID1A

Author: Shuaidong Gao
"""

import numpy as np
import torch
from typing import Optional, List, Dict, Tuple
from .lowrank import train_lowrank_gnn
from .engine import CausalDiscoveryEngine


def find_seed_edges(
    X: np.ndarray,
    gene_names: Optional[List[str]] = None,
    top_k: int = 200,
    method: str = "cluster_aware",
    threshold: float = 0.3,
    seed_genes: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]], List[str]]:
    """Stage 1: Run DAG-constrained engine on top-k variables.

    Selects the top-k variables by variance (or user-specified seeds),
    runs a DAG engine to find high-confidence causal edges.

    Args:
        X: (n, d) data matrix
        gene_names: optional variable names
        top_k: number of top variables to select (by variance)
        method: DAG engine ('cluster_aware', 'dagma')
        threshold: edge weight threshold
        seed_genes: if provided, force-include these genes regardless of variance

    Returns:
        W_dag: (top_k, top_k) adjacency matrix
        edges: list of (i, j, weight) tuples
        selected_names: names of the top_k selected variables
    """
    n, d = X.shape

    # Select top-k by variance
    var = X.var(axis=0)
    top_idx = np.argsort(-var)[:top_k]

    # Force-include seed genes
    if seed_genes and gene_names:
        name_to_idx = {n: i for i, n in enumerate(gene_names)}
        for g in seed_genes:
            if g in name_to_idx:
                idx = name_to_idx[g]
                if idx not in top_idx:
                    top_idx[-1] = idx  # replace lowest-variance
        top_idx = np.unique(top_idx)

    X_sub = X[:, top_idx]
    d_sub = len(top_idx)

    # Run DAG engine
    engine = CausalDiscoveryEngine(
        d=d_sub, rank=min(64, d_sub // 4), mode=method,
        device="cuda" if torch.cuda.is_available() else "cpu",
        epochs=500, threshold=threshold, verbose=False,
    )
    result = engine.fit(X_sub)
    W_dag = result.adjacency

    # Extract edges
    edges = []
    for i in range(d_sub):
        for j in range(d_sub):
            if i != j and abs(W_dag[i, j]) > threshold:
                edges.append((i, j, float(W_dag[i, j])))

    selected_names = [gene_names[top_idx[i]] if gene_names else f"V{top_idx[i]}"
                      for i in range(d_sub)]

    return W_dag, edges, selected_names


def expand_neighborhood(
    X: np.ndarray,
    source_idx: int,
    target_idx: int,
    corr_threshold: float = 0.3,
    max_expand: int = 2000,
    gene_names: Optional[List[str]] = None,
) -> np.ndarray:
    """For a discovered edge (source->target), expand to correlated variables.

    Computes correlation of ALL variables with source and target,
    selects those above threshold, returns the expanded submatrix.
    """
    d = X.shape[1]

    # Compute correlation of all variables with source and target
    X_std = (X - X.mean(0)) / (X.std(0).clip(1e-8) + 1e-8)
    corr_source = np.abs(np.corrcoef(X_std.T[source_idx], X_std.T)[0, 1:])
    corr_target = np.abs(np.corrcoef(X_std.T[target_idx], X_std.T)[0, 1:])

    # Combined score: max correlation with either endpoint
    max_corr = np.maximum(corr_source, corr_target)

    # Select variables above threshold
    candidates = np.where(max_corr > corr_threshold)[0]
    if len(candidates) > max_expand:
        # Keep top max_expand by correlation strength
        candidates = candidates[np.argsort(-max_corr[candidates])[:max_expand]]
    # Ensure source and target are included
    candidates = np.unique(np.append(candidates, [source_idx, target_idx]))

    return X[:, candidates], candidates


def two_stage_discovery(
    X: np.ndarray,
    gene_names: Optional[List[str]] = None,
    stage1_k: int = 200,
    stage1_method: str = "cluster_aware",
    corr_threshold: float = 0.3,
    edge_threshold: float = 0.3,
    max_expand: int = 2000,
    lowrank_rank: int = 64,
    seed_genes: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict:
    """Full two-stage causal discovery pipeline.

    Stage 1: DAG engine on top-k variables → seed edges
    Stage 2: For each seed edge, expand neighborhood → LowRankGNN

    This is the v3.2.0 answer to "how to find causal edges at genome scale":
    use the strong-but-small DAG engines as a lens, then LowRankGNN
    as a wide-angle camera on the expanded neighborhood.

    Args:
        X: (n, d) data matrix
        gene_names: variable names
        stage1_k: top-k for stage 1
        stage1_method: 'cluster_aware' or 'dagma'
        corr_threshold: correlation threshold for neighborhood expansion
        edge_threshold: edge weight threshold
        max_expand: max expanded neighborhood size per seed edge
        lowrank_rank: rank for LowRankGNN in stage 2
        seed_genes: force-include these genes in stage 1
        verbose: print progress

    Returns:
        dict with stage1_edges, stage2_edges, all_edges, stats
    """
    n, d = X.shape
    results = {"d": d, "n": n, "stage1_k": stage1_k}

    # ── Stage 1: DAG seed discovery ──
    if verbose:
        print(f"=== Stage 1: {stage1_method.upper()} on top-{stage1_k} variables ===")
    W_dag, seed_edges, sel_names = find_seed_edges(
        X, gene_names, top_k=stage1_k, method=stage1_method,
        threshold=edge_threshold, seed_genes=seed_genes,
    )
    results["stage1_edges"] = len(seed_edges)
    results["stage1_selected"] = sel_names

    if verbose:
        print(f"  Found {len(seed_edges)} DAG-constrained edges")
        for i, j, w in seed_edges[:5]:
            print(f"    {sel_names[i]} -> {sel_names[j]} ({w:+.3f})")

    # ── Map seed indices back to full d indices ──
    var = X.var(axis=0)
    top_idx = np.argsort(-var)[:stage1_k]
    if seed_genes and gene_names:
        name_to_idx_all = {n: i for i, n in enumerate(gene_names)}
        for g in seed_genes:
            if g in name_to_idx_all:
                idx = name_to_idx_all[g]
                if idx not in top_idx:
                    top_idx = np.append(top_idx, idx)
        top_idx = np.unique(top_idx)

    idx_map = {i: int(top_idx[i]) for i in range(len(top_idx))}
    full_seed_edges = [(idx_map[i], idx_map[j], w) for i, j, w in seed_edges]

    # ── Stage 2: Neighborhood expansion ──
    if verbose:
        print(f"\n=== Stage 2: LowRankGNN on expanded neighborhoods ===")

    all_stage2_edges = []
    seen_pairs = set()

    for si, (src, tgt, w) in enumerate(full_seed_edges):
        if verbose and (si == 0 or si % max(1, len(full_seed_edges) // 5) == 0):
            src_name = gene_names[src] if gene_names else f"V{src}"
            tgt_name = gene_names[tgt] if gene_names else f"V{tgt}"
            print(f"  Expanding: {src_name} -> {tgt_name} (seed {si+1}/{len(full_seed_edges)})")

        X_expand, expand_idx = expand_neighborhood(
            X, src, tgt, corr_threshold=corr_threshold,
            max_expand=max_expand, gene_names=gene_names,
        )

        d_expand = X_expand.shape[1]
        if d_expand < 3:
            continue

        rank = min(lowrank_rank, d_expand // 4)
        res = train_lowrank_gnn(
            X_expand, rank=rank, epochs=min(500, d_expand * 2),
            threshold=edge_threshold, lr=0.01,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=False,
        )

        # Extract edges (mapped back to full d indices)
        W_expand = res["adjacency"]
        for i in range(d_expand):
            for j in range(d_expand):
                if i != j and abs(W_expand[i, j]) > edge_threshold:
                    full_i, full_j = int(expand_idx[i]), int(expand_idx[j])
                    pair = (min(full_i, full_j), max(full_i, full_j))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        all_stage2_edges.append((full_i, full_j, float(W_expand[i, j])))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results["stage2_edges"] = len(all_stage2_edges)
    results["stage2_unique_pairs"] = len(seen_pairs)
    results["total_edges"] = len(seed_edges) + len(all_stage2_edges)

    if verbose:
        print(f"\n=== Results ===")
        print(f"  Stage 1 (DAG): {len(seed_edges)} edges")
        print(f"  Stage 2 (LowRank): {len(all_stage2_edges)} edges")
        print(f"  Total: {results['total_edges']} edges")

    return results
