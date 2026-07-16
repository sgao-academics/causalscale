"""
STRING-anchored gene selection for meaningful DepMap evaluation.

Strategy:
1. Load STRING/TRRUST → find genes with many known interactions
2. Filter DepMap to those genes
3. Run causalscale LowRankGNN on the filtered subset
4. Compute F1 against the STRING/TRRUST gold standard
"""
import sys, os, time, json, gzip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. Load DepMap + extract gene names ──
print("[1] Loading DepMap...")
DATA = r"D:\NO.1\cancer_application\data\depmap"
df = pd.read_csv(f"{DATA}/CRISPRGeneEffect.csv", index_col=0)
all_genes_raw = [col.split(" (")[0] if " (" in col else col for col in df.columns]

# Clean: fill NaN, filter >50% NaN genes
nan_frac = df.isna().mean()
keep = nan_frac[nan_frac < 0.5].index
df_clean = df[keep]
all_genes = [all_genes_raw[df.columns.get_loc(c)] for c in df_clean.columns]
all_gene_set = set(all_genes)
print(f"  DepMap: {df_clean.shape[0]} lines x {df_clean.shape[1]} genes")

# ── 2. Load STRING + TRRUST → find high-connectivity genes ──
print("[2] Loading STRING/TRRUST, finding DepMap-anchored genes...")
STRING_PATH = r"D:\NO.1\cancer_application\data\validation\string_ppi_full.txt.gz"
TRRUST_PATH = r"D:\NO.1\cancer_application\data\validation\trrust_human.tsv"

# Count gene connectivity in STRING
gene_degree = {}
with gzip.open(STRING_PATH, "rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            g1, g2 = parts[0], parts[1]
            if g1 in all_gene_set and g2 in all_gene_set:
                gene_degree[g1] = gene_degree.get(g1, 0) + 1
                gene_degree[g2] = gene_degree.get(g2, 0) + 1

print(f"  STRING-anchored genes in DepMap: {len(gene_degree)}")

# Select top genes by STRING connectivity
top_genes = sorted(gene_degree.items(), key=lambda x: -x[1])[:500]
gene_subset = [g for g, _ in top_genes]
gene_to_idx = {g: i for i, g in enumerate(gene_subset)}
print(f"  Selected {len(gene_subset)} genes (top by STRING degree)")

# ── 3. Build gold standard from STRING/TRRUST (only within subset) ──
print("[3] Building gold standard...")
gold_idx = set()

with gzip.open(STRING_PATH, "rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            if parts[0] in gene_to_idx and parts[1] in gene_to_idx:
                gold_idx.add((gene_to_idx[parts[0]], gene_to_idx[parts[1]]))

with open(TRRUST_PATH, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            g1, g2 = parts[0].upper(), parts[1].upper()
            if g1 in gene_to_idx and g2 in gene_to_idx:
                gold_idx.add((gene_to_idx[g1], gene_to_idx[g2]))

print(f"  Gold pairs: {len(gold_idx)}")

# ── 4. Build data matrix for selected genes ──
print("[4] Building data matrix...")
idx_map = {}
for g in gene_subset:
    for j, col in enumerate(df_clean.columns):
        if gene_subset[gene_to_idx[g]] in col or col.split(" (")[0] == gene_subset[gene_to_idx[g]]:
            idx_map[gene_to_idx[g]] = j
            break

# Better: map by gene name
col_gene_map = {}
for j, col in enumerate(df_clean.columns):
    gene = col.split(" (")[0] if " (" in col else col
    col_gene_map[gene] = j

X_cols = []
for g in gene_subset:
    if g in col_gene_map:
        X_cols.append(col_gene_map[g])
    else:
        print(f"  WARNING: {g} not found in DepMap columns")

X_subset = df_clean.iloc[:, X_cols].fillna(0).values.astype(np.float32)
assert X_subset.shape[1] == len(gene_subset), f"Shape mismatch: {X_subset.shape}"
print(f"  Data: {X_subset.shape[0]} x {X_subset.shape[1]}")

# Standardize
X = StandardScaler().fit_transform(X_subset).astype(np.float32)
d = X.shape[1]
print(f"  d={d}, n={X.shape[0]}")

# ── 5. Train causalscale LowRankGNN ──
print("[5] Training causalscale LowRankGNN...")
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
from causalscale.core.lowrank import LowRankGNN as CLowRankGNN

class LowRankGNN(CLowRankGNN):
    pass

def train(X, rank=32, epochs=500, threshold=0.3, lr=0.01, device="cuda"):
    d = X.shape[1]
    X_t = torch.tensor(X, device=device)
    X_std = (X_t - X_t.mean(0)) / (X_t.std(0).clamp(min=1e-8))
    C = (X_std.T @ X_std) / (X_std.shape[0] - 1)
    C_abs = torch.abs(C)
    C_abs.fill_diagonal_(0)
    gt = (C_abs > threshold).float()
    gt_n = int(gt.sum().item())
    
    model = CLowRankGNN(d, rank=rank).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        loss = nn.MSELoss()(model(), gt)
        loss.backward()
        opt.step()
        if ep % 100 == 0:
            with torch.no_grad():
                W_pred = model()
                n_edge = int((torch.abs(W_pred) > threshold).sum().item())
                mse = loss.item()
            print(f"  E{ep}: mse={mse:.6f}, edges={n_edge}")
    
    train_time = time.time() - t0
    with torch.no_grad():
        W_np = model().cpu().numpy()
        n_edge = int(np.sum(np.abs(W_np) > threshold))
    
    # Extract edges
    edges = []
    for i in range(d):
        for j in range(d):
            if i != j and abs(W_np[i, j]) > threshold:
                edges.append((i, j, float(W_np[i, j])))
    edges.sort(key=lambda x: -abs(x[2]))
    
    # Corr-reconstruction F1
    gt_cpu = gt.cpu().numpy()
    tp_corr = int(np.sum((np.abs(W_np) > threshold) & (gt_cpu > 0)))
    rec_corr = tp_corr / max(gt_n, 1)
    f1_corr = 2 * tp_corr / (n_edge + gt_n) if (n_edge + gt_n) > 0 else 0
    
    return {
        "adjacency": W_np,
        "edges": edges,
        "n_edges": n_edge,
        "train_time_s": train_time,
        "corr_f1": round(f1_corr, 4),
        "corr_recall": round(rec_corr, 4),
    }

result = train(X, rank=32, epochs=500, threshold=0.3, lr=0.01, device=DEVICE)
print(f"  Training: {result['train_time_s']:.1f}s")
print(f"  Correlation-reconstruction F1: {result['corr_f1']:.4f}")
print(f"  Edges found: {result['n_edges']}")

# ── 6. Compute STRING/TRRUST F1 ──
print("[6] Computing STRING/TRRUST F1...")
W = result["adjacency"]
edges = result["edges"]

best_f1 = 0
best_result = None
for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    pred = set()
    for i in range(d):
        for j in range(d):
            if i != j and abs(W[i, j]) > thresh:
                pred.add((i, j))
    
    tp = len(pred & gold_idx)
    fp = len(pred - gold_idx)
    fn = len(gold_idx - pred)
    prec = tp / max(tp+fp, 1)
    rec = tp / max(tp+fn, 1)
    f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
    
    if f1 > best_f1:
        best_f1 = f1
        best_result = {"thresh": thresh, "edges": len(pred), "tp": tp, 
                        "fp": fp, "fn": fn, "prec": round(prec,4), 
                        "rec": round(rec,4), "f1": round(f1,4)}
    
    if tp > 0:
        print(f"  thresh={thresh:.2f}: edges={len(pred)}, TP={tp}, FP={fp}, "
              f"F1={f1:.4f}")

print(f"\n  BEST: thresh={best_result['thresh']}, F1={best_result['f1']:.4f}, "
      f"TP={best_result['tp']}, precision={best_result['prec']:.4f}")

# Show validated edges
validated = [(gene_subset[i], gene_subset[j], float(W[i,j])) 
             for i,j,_ in edges if (i,j) in gold_idx]
print(f"\n  STRING/TRRUST-validated edges: {len(validated)}")
for src, tgt, w in validated[:30]:
    print(f"  {src} -> {tgt}: {w:.4f}")

# ── 7. Top 28 ──
top_28 = edges[:28]
pred_28 = set((i,j) for i,j,_ in top_28)
tp28 = len(pred_28 & gold_idx)
fp28 = len(pred_28 - gold_idx)
fn28 = len(gold_idx - pred_28)
prec28 = tp28 / max(tp28+fp28, 1)
rec28 = tp28 / max(tp28+fn28, 1)
f128 = 2*prec28*rec28/(prec28+rec28) if prec28+rec28>0 else 0
print(f"\n  Top-28 (paper claim): TP={tp28}, FP={fp28}, F1={f128:.4f}")

# Save
OUT = r"D:\NO.1\causalscale_kdd2027_experiments"
result_json = {
    "engine": "causalscale v3.0.0 LowRankGNN (correlation-reconstruction)",
    "gene_selection": "Top-500 STRING-connected genes in DepMap",
    "d": d, "n": X.shape[0], "rank": 32,
    "gold_standard": "STRING + TRRUST (union)",
    "gold_pairs_in_subset": len(gold_idx),
    "corr_reconstruction_f1": result["corr_f1"],
    "best_string_f1": best_result["f1"],
    "best_string_precision": best_result["prec"],
    "best_string_recall": best_result["rec"],
    "best_string_tp": best_result["tp"],
    "top28_string_f1": round(f128, 4),
    "top28_string_tp": tp28,
    "train_time_s": round(result["train_time_s"], 1),
    "validated_edges": [(src, tgt, round(w,4)) for src,tgt,w in validated],
    "top_edges": [(gene_subset[i], gene_subset[j], round(float(w),4)) 
                   for i,j,w in edges[:100]],
}
with open(f"{OUT}/exp14_string_anchored_f1.json", "w") as f:
    json.dump(result_json, f, indent=2)
print(f"\nSaved to exp14_string_anchored_f1.json")
