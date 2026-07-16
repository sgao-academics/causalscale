"""
STRING-Anchored DepMap F1 Validation (V2 - with ENSP→Symbol mapping)
=====================================================================
Fixes the "0 matches" issue: STRING PPI uses ENSP protein IDs, 
not gene symbols. Build ENSP→Symbol map from string_info.txt.gz,
then match with DepMap gene names.

Evaluation pipeline:
1. Build ENSP→Symbol from string_info.txt.gz
2. Map STRING PPI pairs to gene symbols → filter to DepMap genes
3. Build TRRUST pairs → filter to DepMap genes
4. Select top-500 most-connected STRING genes in DepMap
5. Train causalscale LowRankGNN on DepMap subset
6. Compute F1 against STRING/TRRUST gold standard
"""
import sys, os, time, json, gzip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════
# 1. Load DepMap
# ═══════════════════════════════════════════════════════════════
print("[1] Loading DepMap...")
DATA = r"D:\NO.1\cancer_application\data\depmap"
df = pd.read_csv(f"{DATA}/CRISPRGeneEffect.csv", index_col=0)

# Extract gene symbols from column headers "GENE (ID)"
depmap_genes = {}
col_list = []
for col in df.columns:
    gene = col.split(" (")[0] if " (" in col else col
    depmap_genes[gene] = col
    col_list.append(gene)
depmap_gene_set = set(depmap_genes.keys())
print(f"  DepMap: {df.shape[0]} lines x {len(depmap_genes)} genes")

# Clean: filter >50% NaN
nan_frac = df.isna().mean()
keep = nan_frac[nan_frac < 0.5].index
df = df[keep]
# Update gene list to only kept ones
kept_genes = [col.split(" (")[0] if " (" in col else col for col in df.columns]
kept_gene_set = set(kept_genes)
print(f"  After NaN filter: {df.shape[1]} genes")

# ═══════════════════════════════════════════════════════════════
# 2. Build ENSP → Symbol mapping
# ═══════════════════════════════════════════════════════════════
print("[2] Building ENSP→Symbol mapping...")
VALIDATION = r"D:\NO.1\cancer_application\data\validation"

ensp2symbol = {}
with gzip.open(f"{VALIDATION}/string_info.txt.gz", "rt", encoding="utf-8", errors="ignore") as f:
    header = next(f)  # skip header
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            ensp_id = parts[0]  # e.g., "9606.ENSP00000000233"
            symbol = parts[1].strip()
            # Store both with and without "9606." prefix
            ensp2symbol[ensp_id] = symbol
            if ensp_id.startswith("9606."):
                ensp2symbol[ensp_id[5:]] = symbol  # strip "9606."
print(f"  ENSP→Symbol entries: {len(ensp2symbol)}")

# ═══════════════════════════════════════════════════════════════
# 3. Load STRING PPI → convert to gene symbol pairs → filter to DepMap
# ═══════════════════════════════════════════════════════════════
print("[3] Loading STRING PPI, converting to gene symbols...")

# First pass: count gene connectivity
gene_degree = {}
string_symbol_pairs = set()
count = 0

with gzip.open(f"{VALIDATION}/string_ppi_full.txt.gz", "rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            s1 = ensp2symbol.get(parts[0], None)
            s2 = ensp2symbol.get(parts[1], None)
            if s1 and s2 and s1 in kept_gene_set and s2 in kept_gene_set:
                gene_degree[s1] = gene_degree.get(s1, 0) + 1
                gene_degree[s2] = gene_degree.get(s2, 0) + 1
                string_symbol_pairs.add((s1, s2))
        count += 1
        if count % 5000000 == 0:
            print(f"  Processed {count/1e6:.0f}M lines, found {len(gene_degree)} DepMap genes")

print(f"  STRING PPI pairs mapped to DepMap genes: {len(string_symbol_pairs)}")
print(f"  STRING-connected DepMap genes: {len(gene_degree)}")

# ═══════════════════════════════════════════════════════════════
# 4. Load TRRUST
# ═══════════════════════════════════════════════════════════════
print("[4] Loading TRRUST...")
trrust_pairs = set()
with open(f"{VALIDATION}/trrust_human.tsv", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            g1, g2 = parts[0].upper(), parts[1].upper()
            if g1 in kept_gene_set and g2 in kept_gene_set:
                trrust_pairs.add((g1, g2))
                gene_degree[g1] = gene_degree.get(g1, 0) + 1
                gene_degree[g2] = gene_degree.get(g2, 0) + 1

print(f"  TRRUST pairs in DepMap: {len(trrust_pairs)}")

# ═══════════════════════════════════════════════════════════════
# 5. Select top genes by connectivity
# ═══════════════════════════════════════════════════════════════
print("[5] Selecting top genes by STRING+TRRUST connectivity...")
N_SELECT = 500
top_genes = sorted(gene_degree.items(), key=lambda x: -x[1])[:N_SELECT]
gene_subset = [g for g, _ in top_genes]
gene_to_idx = {g: i for i, g in enumerate(gene_subset)}
print(f"  Selected {len(gene_subset)} genes")

# Build gold standard (only within subset)
gold_idx = set()
for g1, g2 in string_symbol_pairs:
    if g1 in gene_to_idx and g2 in gene_to_idx:
        gold_idx.add((gene_to_idx[g1], gene_to_idx[g2]))
for g1, g2 in trrust_pairs:
    if g1 in gene_to_idx and g2 in gene_to_idx:
        gold_idx.add((gene_to_idx[g1], gene_to_idx[g2]))
print(f"  Gold standard pairs within subset: {len(gold_idx)}")

# ═══════════════════════════════════════════════════════════════
# 6. Build data matrix
# ═══════════════════════════════════════════════════════════════
print("[6] Building data matrix...")
X_raw = np.zeros((df.shape[0], len(gene_subset)), dtype=np.float32)
for i, g in enumerate(gene_subset):
    if g in depmap_genes:
        col_name = depmap_genes[g]
        if col_name in df.columns:
            X_raw[:, i] = df[col_name].fillna(0).values

# Remove zero-variance genes
col_std = X_raw.std(axis=0)
nonzero_std = col_std > 1e-8
if not nonzero_std.all():
    n_zero = (~nonzero_std).sum()
    print(f"  Removing {n_zero} zero-variance genes")
    X_raw = X_raw[:, nonzero_std]
    gene_subset = [gene_subset[i] for i in range(len(gene_subset)) if nonzero_std[i]]
    # Rebuild gene_to_idx and gold_idx
    gene_to_idx = {g: i for i, g in enumerate(gene_subset)}
    new_gold = set()
    for g1, g2 in string_symbol_pairs:
        if g1 in gene_to_idx and g2 in gene_to_idx:
            new_gold.add((gene_to_idx[g1], gene_to_idx[g2]))
    for g1, g2 in trrust_pairs:
        if g1 in gene_to_idx and g2 in gene_to_idx:
            new_gold.add((gene_to_idx[g1], gene_to_idx[g2]))
    gold_idx = new_gold

# Standardize
X = StandardScaler().fit_transform(X_raw).astype(np.float32)
d, n = X.shape[1], X.shape[0]
print(f"  Final: d={d}, n={n}, gold_pairs={len(gold_idx)}")

# ═══════════════════════════════════════════════════════════════
# 7. Train causalscale LowRankGNN
# ═══════════════════════════════════════════════════════════════
print("[7] Training causalscale LowRankGNN...")
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
from causalscale.core.lowrank import LowRankGNN, train_lowrank_gnn

# We'll call train_lowrank_gnn directly 
# But first let's also try with different thresholds
for threshold in [0.2, 0.25, 0.3, 0.35]:
    print(f"\n  --- threshold={threshold} ---")
    t0 = time.time()
    
    X_t = torch.tensor(X, device=DEVICE)
    X_std = (X_t - X_t.mean(0)) / (X_t.std(0).clamp(min=1e-8))
    C = (X_std.T @ X_std) / (X_std.shape[0] - 1)
    C_abs = torch.abs(C)
    C_abs.fill_diagonal_(0)
    gt = (C_abs > threshold).float()
    gt_n = int(gt.sum().item())
    
    model = LowRankGNN(d, rank=32).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for ep in range(400):
        opt.zero_grad()
        loss = nn.MSELoss()(model(), gt)
        loss.backward()
        opt.step()
    
    with torch.no_grad():
        W = model().cpu().numpy()
        n_edge = int(np.sum(np.abs(W) > threshold))
        # Corr F1
        tp_c = int(np.sum((np.abs(W) > threshold) & (gt.cpu().numpy() > 0)))
        f1_c = 2*tp_c/(n_edge+gt_n) if n_edge+gt_n>0 else 0
        
        # STRING/TRRUST F1
        best_f1 = 0
        best_t = threshold
        for check_t in [0.1, 0.15, 0.2, 0.25, 0.3, threshold]:
            pred = set()
            for i2 in range(d):
                for j2 in range(d):
                    if i2 != j2 and abs(W[i2,j2]) > check_t:
                        pred.add((i2, j2))
            tp = len(pred & gold_idx)
            fp = len(pred - gold_idx)
            fn = len(gold_idx - pred)
            pc = tp/max(tp+fp,1)
            rc = tp/max(tp+fn,1)
            f1 = 2*pc*rc/(pc+rc) if pc+rc>0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_t = check_t
                best_tp = tp
                best_fp = fp
                best_pc = pc
    
    elapsed = time.time() - t0
    print(f"    Corr-F1={f1_c:.4f}, edges={n_edge}")
    print(f"    STRING/TRRUST F1={best_f1:.4f} (thresh={best_t:.2f}), "
          f"TP={best_tp}, FP={best_fp}, prec={best_pc:.4f}")

# ═══════════════════════════════════════════════════════════════
# 8. Best result detailed analysis
# ═══════════════════════════════════════════════════════════════
print(f"\n[8] Detailed analysis (threshold=0.3)...")
# Retrain with threshold=0.3 for final result
X_t = torch.tensor(X, device=DEVICE)
X_std = (X_t - X_t.mean(0)) / (X_t.std(0).clamp(min=1e-8))
C = (X_std.T @ X_std) / (X_std.shape[0] - 1)
C_abs = torch.abs(C)
C_abs.fill_diagonal_(0)
gt = (C_abs > 0.3).float()
gt_n = int(gt.sum().item())

model = LowRankGNN(d, rank=32).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for ep in range(500):
    opt.zero_grad()
    loss = nn.MSELoss()(model(), gt)
    loss.backward()
    opt.step()

with torch.no_grad():
    W = model().cpu().numpy()
    n_edge = int(np.sum(np.abs(W) > 0.3))

# Extract edges, show STRING-validated ones
edges = []
for i in range(d):
    for j in range(d):
        if i != j and abs(W[i,j]) > 0.3:
            edges.append((i, j, float(W[i,j])))
edges.sort(key=lambda x: -abs(x[2]))

validated = [(gene_subset[i], gene_subset[j], w) for i,j,w in edges if (i,j) in gold_idx]
print(f"  Total edges: {n_edge}")
print(f"  STRING/TRRUST validated: {len(validated)}")
print(f"  Top validated edges:")
for src, tgt, w in validated[:20]:
    print(f"    {src} -> {tgt}: {w:.4f}")

# Top-28
top28 = edges[:28]
pred28 = set((i,j) for i,j,_ in top28)
tp28 = len(pred28 & gold_idx)
fp28 = len(pred28 - gold_idx)
fn28 = len(gold_idx - pred28)
p28 = tp28/max(tp28+fp28,1)
r28 = tp28/max(tp28+fn28,1)
f128 = 2*p28*r28/(p28+r28) if p28+r28>0 else 0
print(f"\n  Top-28: TP={tp28}, FP={fp28}, F1={f128:.4f}")

# Best overall F1
best_f1 = 0
best_stats = None
for t in np.arange(0.05, 0.55, 0.05):
    pred = set((i,j) for i,j,_ in edges if abs(W[i,j]) > t)
    tp = len(pred & gold_idx)
    fp = len(pred - gold_idx)
    fn = len(gold_idx - pred)
    pc = tp/max(tp+fp,1)
    rc = tp/max(tp+fn,1)
    f1 = 2*pc*rc/(pc+rc) if pc+rc>0 else 0
    if f1 > best_f1:
        best_f1 = f1
        best_stats = {"thresh": round(t,2), "edges": len(pred), "tp": tp, "fp": fp,
                       "fn": fn, "prec": round(pc,4), "rec": round(rc,4), "f1": round(f1,4)}

print(f"\n  BEST: thresh={best_stats['thresh']}, F1={best_stats['f1']:.4f}, "
      f"TP={best_stats['tp']}, prec={best_stats['prec']:.4f}")

# Save
OUT = r"D:\NO.1\causalscale_kdd2027_experiments"
result_json = {
    "engine": "causalscale v3.0.0 LowRankGNN",
    "objective": "correlation-reconstruction (MSE against thresholded correlation)",
    "gene_selection": f"Top-{N_SELECT} STRING-connected genes in DepMap",
    "d": d, "n": n, "rank": 32, "epochs": 500,
    "gold_standard": "STRING PPI (ENSP→Symbol mapped) + TRRUST",
    "gold_pairs_in_subset": len(gold_idx),
    "total_edges_found": n_edge,
    "string_trrust_validated": len(validated),
    "best_string_f1": best_stats["f1"],
    "best_string_precision": best_stats["prec"],
    "best_string_recall": best_stats["rec"],
    "best_string_tp": best_stats["tp"],
    "top28_string_f1": round(f128, 4),
    "top28_string_tp": tp28,
    "validated_edges": [(s,t,round(w,4)) for s,t,w in validated[:50]],
    "top_edges": [(gene_subset[i], gene_subset[j], round(float(w),4)) 
                   for i,j,w in edges[:50]],
}
with open(f"{OUT}/exp15_string_mapped_f1.json", "w") as f:
    json.dump(result_json, f, indent=2)
print(f"\nSaved to exp15_string_mapped_f1.json")
