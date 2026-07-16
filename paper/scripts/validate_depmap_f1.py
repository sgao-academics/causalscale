"""
DepMap F1 Validation: causalscale LowRankGNN vs STRING/TRRUST Gold Standard
============================================================================
Verifies the paper claim: "LowRankGNN identifies 28 causal edges (F1>0.97) on DepMap"
using the actual causalscale package (not standalone prototype).

Strategy:
1. Load DepMap CRISPR CERES scores (1208 cell lines x 18531 genes)
2. Run causalscale CausalDiscovery with method="lowrank"
3. Get top edges
4. Compare against STRING+TRRUST curated gold standard
5. GO enrichment analysis
"""
import sys, os, time, json, gzip
import numpy as np
import pandas as pd
from collections import defaultdict

# Add causalscale to path
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")

import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ───────────────────────────────────────────────────────────────────
# 1. LOAD DEPMAP DATA
# ───────────────────────────────────────────────────────────────────
print("\n[1] Loading DepMap CRISPR data...")
t0 = time.time()

DATA = r"D:\NO.1\cancer_application\data\depmap"
# Use CRISPRGeneEffect (CERES scores) - the gold standard for gene dependency
df = pd.read_csv(f"{DATA}/CRISPRGeneEffect.csv", index_col=0)
print(f"  Raw shape: {df.shape}")

# Parse gene names from column headers (format: "GENE (ID)")
gene_names = []
for col in df.columns:
    gene = col.split(" (")[0] if " (" in col else col
    gene_names.append(gene)

# Drop genes with >50% NaN
nan_frac = df.isna().mean()
keep_genes = nan_frac[nan_frac < 0.5].index
df = df[keep_genes]
gene_names_filtered = [gene_names[df.columns.get_loc(c)] for c in df.columns]
print(f"  After NaN filter: {df.shape}")

# Fill remaining NaN with 0
X_raw = df.fillna(0).values.astype(np.float32)
print(f"  Final matrix: {X_raw.shape[0]} cell lines x {X_raw.shape[1]} genes")

# Standardize
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X_raw).astype(np.float32)
d = X.shape[1]
print(f"  d={d}, n={X.shape[0]}, memory={X.nbytes/1e6:.1f} MB")
print(f"  Load time: {time.time()-t0:.1f}s")

# ───────────────────────────────────────────────────────────────────
# 2. RUN CAUSALSCALE LOWRANKGNN
# ───────────────────────────────────────────────────────────────────
print(f"\n[2] Running causalscale LowRankGNN (d={d}, rank=64)...")
t1 = time.time()

from causalscale.core.lowrank import LowRankGNN, train_lowrank_gnn

result = train_lowrank_gnn(
    X,
    rank=64,
    epochs=500,        # more epochs for genome-scale data
    threshold=0.3,
    lr=0.005,
    device="cuda",
    verbose=True
)

train_time = time.time() - t1
W = result["adjacency"]
n_edges = result["gnn_edges"]
print(f"\n  Training complete: {train_time/60:.1f} min")
print(f"  Edges found (|W|>0.3): {n_edges}")
print(f"  Correlation-reconstruction F1: {result['f1']:.4f}")

# Get top edges
edges = []
for i in range(d):
    for j in range(d):
        if i != j and abs(W[i, j]) > 0.3:
            edges.append((i, j, float(W[i, j])))
edges.sort(key=lambda x: -abs(x[2]))
print(f"  Top edges: {len(edges)}")

# ───────────────────────────────────────────────────────────────────
# 3. LOAD STRING + TRRUST GOLD STANDARD
# ───────────────────────────────────────────────────────────────────
print("\n[3] Loading STRING/TRRUST gold standard...")

# Build gene name -> index mapping
gene_to_idx = {g: i for i, g in enumerate(gene_names_filtered)}

# STRING PPI
STRING_PATH = r"D:\NO.1\cancer_application\data\validation\string_ppi_full.txt.gz"
string_pairs = set()
try:
    with gzip.open(STRING_PATH, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                string_pairs.add((parts[0], parts[1]))
    print(f"  STRING pairs: {len(string_pairs)}")
except Exception as e:
    print(f"  STRING load error: {e}")
    string_pairs = set()

# TRRUST
TRRUST_PATH = r"D:\NO.1\cancer_application\data\validation\trrust_human.tsv"
trrust_pairs = set()
try:
    with open(TRRUST_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                trrust_pairs.add((parts[0].upper(), parts[1].upper()))
    print(f"  TRRUST pairs: {len(trrust_pairs)}")
except Exception as e:
    print(f"  TRRUST load error: {e}")
    trrust_pairs = set()

# Merge gold standard
gold_pairs = string_pairs | trrust_pairs
print(f"  Gold standard (UNION): {len(gold_pairs)} pairs")

# Map to indices
gold_idx_pairs = set()
for src, tgt in gold_pairs:
    if src in gene_to_idx and tgt in gene_to_idx:
        gold_idx_pairs.add((gene_to_idx[src], gene_to_idx[tgt]))
print(f"  Gold pairs mapped to causalscale indices: {len(gold_idx_pairs)}")

# ───────────────────────────────────────────────────────────────────
# 4. COMPUTE F1
# ───────────────────────────────────────────────────────────────────
print("\n[4] Computing F1 vs gold standard...")

# Predicted edges (top N and various thresholds)
def compute_f1(pred_edges, gold, top_n=None):
    """Compute F1 for top-N predicted edges vs gold standard."""
    if top_n:
        pred_set = set((i, j) for i, j, _ in pred_edges[:top_n])
    else:
        pred_set = set((i, j) for i, j, _ in pred_edges)
    
    tp = len(pred_set & gold)
    fp = len(pred_set - gold)
    fn = len(gold - pred_set)
    
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    return {"tp": tp, "fp": fp, "fn": fn, "prec": prec, "rec": rec, "f1": f1}

# Try multiple thresholds
for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
    top_edges = [(i, j, float(W[i, j])) for i in range(d) for j in range(d)
                 if i != j and abs(W[i, j]) > thresh]
    top_edges.sort(key=lambda x: -abs(x[2]))
    
    if len(top_edges) > 0:
        stats = compute_f1(top_edges, gold_idx_pairs)
        print(f"  threshold={thresh}: edges={len(top_edges)}, TP={stats['tp']}, "
              f"FP={stats['fp']}, FN={stats['fn']}, F1={stats['f1']:.4f}")

# Also try top-28 (matching paper claim)
top_28 = edges[:28]
stats_28 = compute_f1(top_28, gold_idx_pairs)
print(f"\n  Top-28 edges (paper claim):")
print(f"    TP={stats_28['tp']}, FP={stats_28['fp']}, FN={stats_28['fn']}")
print(f"    Precision={stats_28['prec']:.4f}, Recall={stats_28['rec']:.4f}, F1={stats_28['f1']:.4f}")

# ───────────────────────────────────────────────────────────────────
# 5. GO ENRICHMENT (simplified)
# ───────────────────────────────────────────────────────────────────
print("\n[5] Edge gene pairs (for GO enrichment)...")
top_edge_genes = []
for i, j, w in edges[:50]:
    gn_i = gene_names_filtered[i]
    gn_j = gene_names_filtered[j]
    top_edge_genes.append((gn_i, gn_j, w))
    if len(top_edge_genes) <= 20:
        print(f"  {gn_i} -> {gn_j}: {w:.4f}")

# ───────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ───────────────────────────────────────────────────────────────────
result_json = {
    "engine": "causalscale v3.0.0 LowRankGNN",
    "data": "DepMap 24Q2 CRISPR CERES",
    "shape": [int(X.shape[0]), int(X.shape[1])],
    "rank": 64,
    "epochs": 500,
    "train_time_s": round(train_time, 1),
    "total_edges_found": n_edges,
    "top28_f1": round(stats_28["f1"], 4),
    "top28_precision": round(stats_28["prec"], 4),
    "top28_recall": round(stats_28["rec"], 4),
    "top28_tp": stats_28["tp"],
    "top28_fp": stats_28["fp"],
    "top28_fn": stats_28["fn"],
    "gold_standard_size": len(gold_idx_pairs),
    "top_edges": [(gene_names_filtered[i], gene_names_filtered[j], round(w, 4))
                  for i, j, w in edges[:100]]
}

OUT = r"D:\NO.1\causalscale_kdd2027_experiments"
with open(f"{OUT}/exp13_depmap_causalscale_f1.json", "w") as f:
    json.dump(result_json, f, indent=2)
print(f"\nResults saved to exp13_depmap_causalscale_f1.json")
print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")
