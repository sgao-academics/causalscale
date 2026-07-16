"""
Quick test: NOTEARS on DepMap d=100 subset → STRING/TRRUST overlap.
This tells us the ceiling: if NOTEARS also has low overlap, gold standard is the bottleneck.
"""
import sys, os, time, json, gzip
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Load DepMap ──
DATA = r"D:\NO.1\cancer_application\data\depmap"
df = pd.read_csv(f"{DATA}/CRISPRGeneEffect.csv", index_col=0)
gene_names = [col.split(" (")[0] if " (" in col else col for col in df.columns]

nan_frac = df.isna().mean()
df = df[nan_frac[nan_frac < 0.5].index]
X_raw = df.fillna(0).values.astype(np.float32)

# Standardize, select top-d by variance
X = StandardScaler().fit_transform(X_raw).astype(np.float32)
variances = X.var(axis=0)
top_idx = np.argsort(-variances)[:100]
X_top = X[:, top_idx]
genes_top = [gene_names[i] for i in top_idx]
print(f"Selected top-100 genes by variance: {X_top.shape}")

# ── Build gene->index mapping ──
gene_to_idx = {g: i for i, g in enumerate(genes_top)}

# ── Load gold standard ──
STRING_PATH = r"D:\NO.1\cancer_application\data\validation\string_ppi_full.txt.gz"
TRRUST_PATH = r"D:\NO.1\cancer_application\data\validation\trrust_human.tsv"

string_pairs = set()
with gzip.open(STRING_PATH, "rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            string_pairs.add((parts[0], parts[1]))

trrust_pairs = set()
with open(TRRUST_PATH, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            trrust_pairs.add((parts[0].upper(), parts[1].upper()))

gold_pairs = string_pairs | trrust_pairs
gold_idx = set()
for src, tgt in gold_pairs:
    if src in gene_to_idx and tgt in gene_to_idx:
        gold_idx.add((gene_to_idx[src], gene_to_idx[tgt]))
print(f"Gold pairs in subset: {len(gold_idx)}")

# ── NOTEARS implementation ──
def notears_linear(X, lambda1=0.1, loss_type='l2', max_iter=100, h_tol=1e-8, 
                    rho_max=1e16, verbose=True):
    """Minimal NOTEARS implementation."""
    n, d = X.shape
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    
    W = torch.zeros((d, d), device=DEVICE, requires_grad=True)
    # Initialize with small random values
    W.data = torch.randn(d, d, device=DEVICE) * 0.01
    
    def h(W):
        M = W * W
        return torch.trace(torch.matrix_exp(M)) - d
    
    def loss_fn(W):
        R = X_t - X_t @ W
        if loss_type == 'l2':
            return 0.5 / n * torch.sum(R ** 2)
        return 0.5 / n * torch.sum(R ** 2)
    
    rho, alpha = 1.0, 0.0
    opt = torch.optim.Adam([W], lr=0.002)
    
    for outer in range(max_iter):
        for inner in range(200):
            opt.zero_grad()
            loss = loss_fn(W)
            l1 = lambda1 * torch.sum(torch.abs(W))
            h_val = h(W)
            total = loss + l1 + alpha * h_val + 0.5 * rho * h_val * h_val
            total.backward()
            opt.step()
        
        with torch.no_grad():
            h_val = h(W)
        if verbose and outer % 20 == 0:
            print(f"  Outer {outer}: h={h_val.item():.4e}, loss={loss.item():.6f}")
        
        if h_val > 0.25 * h(W).item() if outer > 0 else True:
            alpha += rho * h_val.item()
        if h_val < h_tol or rho > rho_max:
            break
        rho = min(rho * 10, rho_max)
    
    return W.detach().cpu().numpy()

# ── Run NOTEARS ──
print("\nRunning NOTEARS on d=100 DepMap subset...")
t0 = time.time()
W_notears = notears_linear(X_top, lambda1=0.05, max_iter=50, verbose=True)
print(f"  Time: {time.time()-t0:.1f}s")

n_edges = int(np.sum(np.abs(W_notears) > 0.1))
print(f"  Edges (|W|>0.1): {n_edges}")

# NOTEARS on correlation (for comparison)
W_corr = np.corrcoef(X_top.T)
n_corr = int(np.sum((np.abs(W_corr) > 0.5) & (np.eye(100) == 0)))
print(f"  Correlation edges (|r|>0.5): {n_corr}")

# ── Compute F1 ──
print("\nSTRING/TRRUST overlap:")
for name, W, thresh in [
    ("NOTEARS", W_notears, 0.1),
    ("NOTEARS (loose)", W_notears, 0.05),
    ("Correlation", W_corr, 0.5),
    ("Correlation (loose)", W_corr, 0.3),
]:
    pred = set()
    for i in range(100):
        for j in range(100):
            if i != j and abs(W[i, j]) > thresh:
                pred.add((i, j))
    
    tp = len(pred & gold_idx)
    fp = len(pred - gold_idx)
    fn = len(gold_idx - pred)
    prec = tp / max(tp+fp, 1)
    rec = tp / max(tp+fn, 1)
    f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
    print(f"  {name} (thresh>{thresh}): edges={len(pred)}, TP={tp}, FP={fp}, "
          f"prec={prec:.4f}, rec={rec:.4f}, F1={f1:.4f}")

# Show which gold pairs exist in the top-100
print(f"\nGold pairs among top-100 genes:")
for (i, j) in list(gold_idx)[:20]:
    wi = W_notears[i, j]
    ci = W_corr[i, j]
    print(f"  {genes_top[i]} -> {genes_top[j]}: NOTEARS={wi:.4f}, corr={ci:.4f}")
