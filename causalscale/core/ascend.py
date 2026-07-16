"""
ASCEND-inspired Two-Tier Causal Discovery.
Leverages known biological hierarchy: upstream regulators → downstream responses.
Constraints the search space to enforce causal direction from upstream to downstream.
"""
import numpy as np, torch, time
from typing import Optional, Dict, List, Tuple


def two_tier_discovery(
    X: np.ndarray,
    upstream_idx: List[int],
    downstream_idx: List[int],
    device: str = "cuda",
    verbose: bool = True,
) -> Dict:
    """Two-tier causal discovery with ASCEND-inspired ancestral conditioning.

    Tier 1 (upstream → downstream): Run NOTEARS with constrained W.
        Only edges from upstream to downstream are allowed.
    Tier 2 (within-tier): Run standard NOTEARS on each tier separately.

    Args:
        X: (n, d) data matrix
        upstream_idx: indices of upstream variables (e.g., TFs, kinases)
        downstream_idx: indices of downstream variables (e.g., gene expression)
        device: 'cuda' or 'cpu'
        verbose: print progress

    Returns:
        dict: adjacency, edges, tier_info, time
    """
    t0 = time.time()
    d = X.shape[1]
    upstream = np.array(upstream_idx)
    downstream = np.array(downstream_idx)

    if len(upstream) + len(downstream) != d:
        other = list(set(range(d)) - set(upstream) - set(downstream))
        downstream = np.concatenate([downstream, other])

    if verbose:
        print(f"  Two-tier: {len(upstream)} upstream, "
              f"{len(downstream)} downstream")

    X_t = torch.tensor(X.astype(np.float32), device=device)

    # ── Tier 1: upstream → downstream (directed constraint) ──
    # Build constraint mask: 1 for allowed edges (upstream→downstream + within-upstream)
    mask = np.zeros((d, d), dtype=np.float32)
    for ui in upstream:
        for dj in downstream:
            mask[ui, dj] = 1.0  # upstream → downstream ONLY
    for ui in upstream:
        for uj in upstream:
            if ui != uj:
                mask[ui, uj] = 1.0  # within upstream

    mask_t = torch.tensor(mask, device=device)

    W = torch.zeros(d, d, requires_grad=True, device=device)
    rho, alpha = 1.0, 0.0
    opt = torch.optim.Adam([W], lr=0.002)
    h_prev = float('inf')

    if verbose:
        print(f"  Tier 1: constrained upstream->downstream NOTEARS...")

    for o in range(30):
        for _ in range(200):
            opt.zero_grad()
            M = torch.eye(d, device=device) - (W * mask_t)  # apply mask
            sq = (X_t @ M.T).pow(2)
            loss = sq.mean()
            h_val = torch.trace(torch.linalg.matrix_exp(
                (W * mask_t) * (W * mask_t))) - d
            loss = loss + 0.5 * rho * h_val * h_val + alpha * h_val
            loss.backward()
            opt.step()

        with torch.no_grad():
            h_val = torch.trace(torch.linalg.matrix_exp(
                (W * mask_t) * (W * mask_t))) - d
        h_curr = h_val.item()
        if o > 0 and h_curr > 0.25 * h_prev:
            rho *= 10
        else:
            alpha += rho * h_curr
        h_prev = h_curr
        if h_curr < 1e-8:
            break

    W_tier1 = W.detach().cpu().numpy()
    # Apply mask post-hoc
    W_tier1 = W_tier1 * mask

    # ── Tier 2: within-downstream (no constraint needed) ──
    # Skip if downstream is small
    if verbose:
        print(f"  Tier 2: within-tier downstream...")

    # Downstream NOTEARS
    X_down = X[:, downstream]
    W_down = _run_notears_subset(X_down, device=device)

    # Map back to full indices
    W_full = W_tier1.copy()
    for i, di in enumerate(downstream):
        for j, dj in enumerate(downstream):
            if di != dj:
                W_full[di, dj] = W_down[i, j]

    n_edges = int(np.sum(np.abs(W_full) > 0.3))
    elapsed = time.time() - t0

    if verbose:
        n_up_down = int(np.sum(np.abs(W_tier1) > 0.3))
        n_down = int(np.sum(np.abs(W_down) > 0.3))
        print(f"  Tier 1 (up->down): {n_up_down} edges")
        print(f"  Tier 2 (within-down): {n_down} edges")
        print(f"  Total: {n_edges} edges, {elapsed:.1f}s")

    return {
        "adjacency": W_full,
        "edge_count": n_edges,
        "time_s": elapsed,
        "tier1_adjacency": W_tier1,
        "tier2_adjacency": W_down,
        "n_upstream": len(upstream),
        "n_downstream": len(downstream),
        "h_final": float(h_curr) if 'h_curr' in dir() else 0.0,
    }


def _run_notears_subset(X, device="cuda", outer=30, inner=200):
    """Standard NOTEARS on subset."""
    n, d = X.shape
    X_t = torch.tensor(X.astype(np.float32), device=device)
    W = torch.zeros(d, d, requires_grad=True, device=device)
    rho, alpha = 1.0, 0.0
    opt = torch.optim.Adam([W], lr=0.002)
    h_prev = float('inf')

    for o in range(outer):
        for _ in range(inner):
            opt.zero_grad()
            M = torch.eye(d, device=device) - W
            sq = (X_t @ M.T).pow(2)
            loss = sq.mean()
            h_val = torch.trace(torch.linalg.matrix_exp(W * W)) - d
            loss = loss + 0.5 * rho * h_val * h_val + alpha * h_val
            loss.backward()
            opt.step()

        with torch.no_grad():
            h_val = torch.trace(torch.linalg.matrix_exp(W * W)) - d
        h_curr = h_val.item()
        if o > 0 and h_curr > 0.25 * h_prev:
            rho *= 10
        else:
            alpha += rho * h_curr
        h_prev = h_curr
        if h_curr < 1e-8:
            break

    return W.detach().cpu().numpy()


def auto_detect_tiers(
    var_names: List[str],
    string_data_dir: str,
    n_upstream: int = 100,
) -> Tuple[List[int], List[int]]:
    """Auto-detect upstream/downstream tiers using STRING connectivity.

    Upstream = highly connected genes (likely TFs/regulators)
    Downstream = rest (likely effectors/responders)

    Args:
        var_names: gene symbols matching data columns
        string_data_dir: path to STRING/TRRUST data
        n_upstream: how many top-connected genes to use as upstream

    Returns:
        (upstream_idx, downstream_idx) lists of column indices
    """
    import gzip, os

    # Load STRING connectivity
    info_path = os.path.join(string_data_dir, "string_info.txt.gz")
    ppi_path = os.path.join(string_data_dir, "string_ppi_full.txt.gz")

    ensp2sym = {}
    with gzip.open(info_path, "rt", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            p = line.strip().split("\t")
            if len(p) >= 2:
                eid = p[0]; sym = p[1].strip()
                ensp2sym[eid] = sym
                if eid.startswith("9606."):
                    ensp2sym[eid[5:]] = sym

    degree = {v: 0 for v in var_names}
    with gzip.open(ppi_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2:
                s1 = ensp2sym.get(p[0])
                s2 = ensp2sym.get(p[1])
                if s1 in degree:
                    degree[s1] += 1
                if s2 in degree:
                    degree[s2] += 1

    # Top-connected → upstream
    sorted_genes = sorted(degree.items(), key=lambda x: -x[1])
    upstream_genes = set(g for g, _ in sorted_genes[:n_upstream])
    upstream_idx = [i for i, v in enumerate(var_names) if v in upstream_genes]
    downstream_idx = [i for i, v in enumerate(var_names) if v not in upstream_genes]

    return upstream_idx, downstream_idx
