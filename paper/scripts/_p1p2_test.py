"""
Test P1 (LLM arb + curriculum) and P2 (ASCEND) optimizations.
"""
import sys, numpy as np, time
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
import causalscale as cs

DEV = "cuda" if __import__('torch').cuda.is_available() else "cpu"
np.random.seed(42)

def f1_best(W, W_true):
    bf = 0
    for th in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        Wb = (np.abs(W) > th).astype(int)
        Wt = (np.abs(W_true) > 0).astype(int)
        tp = int(np.sum(Wb & Wt)) - int(np.sum(np.diag(Wb) & np.diag(Wt)))
        fp = int(np.sum(Wb & (1-Wt))) - int(np.sum(np.diag(Wb) & (1-np.diag(Wt))))
        fn = int(np.sum((1-Wb) & Wt)) - int(np.sum((1-np.diag(Wb)) & np.diag(Wt)))
        p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
        f = 2*p*r/(p+r) if p+r>0 else 0
        if f > bf: bf = f
    return bf

# ═══════════════════════════════════════════════════════════════
# SYNTHETIC BENCHMARK
# ═══════════════════════════════════════════════════════════════
print("="*60)
print("  SYNTHETIC ER BENCHMARK (d=50, n=500)")
print("="*60)

d, n = 50, 500
W_true = np.triu(np.random.binomial(1, 0.1, (d, d)), k=1).astype(float)
for i in range(d):
    for j in range(i+1, d):
        if W_true[i,j] > 0:
            W_true[i,j] = np.random.uniform(0.3, 0.8)
true_edges = int(np.sum(W_true > 0))
print(f"True edges: {true_edges}")

X = np.random.randn(n, d)
for j in range(d):
    for p in range(j):
        if W_true[p,j] > 0:
            X[:,j] += W_true[p,j] * X[:,p]
    X[:,j] += 0.3 * np.random.randn(n)
X = (X - X.mean(0)) / X.std(0).clip(min=1e-8)

# Baseline
m = cs.CausalDiscovery(X, method="cluster_aware", device=DEV, verbose=False)
m.fit(verbose=False)
f1_base = f1_best(m._network.adjacency, W_true)
print(f"\n  Baseline (cluster_aware):     F1={f1_base:.4f}")

# ── P1a: Curriculum Learning ──
print("\n  [P1a] Curriculum Learning...")
from causalscale.core.curriculum import curriculum_notears
W_curr = curriculum_notears(X, target_d=d, curriculum_steps=3, device=DEV, verbose=False)
f1_curr = f1_best(W_curr, W_true)
print(f"  Curriculum NOTEARS:           F1={f1_curr:.4f} (gain={f1_curr-f1_base:+.4f})")

# ── P2: ASCEND two-tier (simulated tiers by variance) ──
print("\n  [P2] ASCEND Two-Tier...")
variances = X.var(axis=0)
upstream = list(np.argsort(-variances)[:20])   # top 20 by variance
downstream = list(np.argsort(-variances)[20:])  # rest
from causalscale.core.ascend import two_tier_discovery
ascend_result = two_tier_discovery(X, upstream, downstream, device=DEV, verbose=False)
f1_ascend = f1_best(ascend_result["adjacency"], W_true)
print(f"  ASCEND Two-Tier:              F1={f1_ascend:.4f} (gain={f1_ascend-f1_base:+.4f})")

# ── Summary ──
print(f"\n  {'Method':<30s} {'F1':>8s} {'Gain':>8s}")
print(f"  {'-'*46}")
for name, f1 in [("Baseline cluster_aware", f1_base),
                  ("Curriculum NOTEARS", f1_curr),
                  ("ASCEND Two-Tier", f1_ascend)]:
    print(f"  {name:<30s} {f1:>8.4f} {f1-f1_base:>+8.4f}")

# ═══════════════════════════════════════════════════════════════
# STRING-ANCHORED BIOLOGICAL TEST
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  STRING-ANCHORED BIOLOGICAL VALIDATION")
print(f"{'='*60}")

import pandas as pd
VALID = r"D:\NO.1\cancer_application\data\validation"
DATA = r"D:\NO.1\cancer_application\data\depmap"

# Load DepMap + STRING-anchored genes (from earlier exp)
import gzip
ensp2sym = {}
with gzip.open(f"{VALID}/string_info.txt.gz", "rt", encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        p = line.strip().split("\t")
        if len(p) >= 2:
            eid = p[0]; sym = p[1].strip()
            ensp2sym[eid] = sym
            if eid.startswith("9606."):
                ensp2sym[eid[5:]] = sym

# Quick: load top-200 STRING-connected genes
df = pd.read_csv(f"{DATA}/CRISPRGeneEffect.csv", index_col=0)
depmap_genes = {col.split(" (")[0] if " (" in col else col: col for col in df.columns}
depmap_set = set(depmap_genes.keys())

degree = {}
with gzip.open(f"{VALID}/string_ppi_full.txt.gz", "rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        p = line.strip().split()
        if len(p) >= 2:
            s1 = ensp2sym.get(p[0]); s2 = ensp2sym.get(p[1])
            if s1 in depmap_set and s2 in depmap_set:
                degree[s1] = degree.get(s1, 0) + 1
                degree[s2] = degree.get(s2, 0) + 1

top200 = sorted(degree.items(), key=lambda x: -x[1])[:200]
gene_list = [g for g, _ in top200]
X_bio = np.zeros((df.shape[0], len(gene_list)), dtype=np.float32)
for i, g in enumerate(gene_list):
    if g in depmap_genes:
        col = depmap_genes[g]
        if col in df.columns:
            X_bio[:, i] = df[col].fillna(0).values
X_bio = (X_bio - X_bio.mean(0)) / X_bio.std(0).clip(min=1e-8)

# Build gold standard within these genes
gene2idx = {g: i for i, g in enumerate(gene_list)}
gold_idx = set()
with gzip.open(f"{VALID}/string_ppi_full.txt.gz", "rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        p = line.strip().split()
        if len(p) >= 2:
            s1 = ensp2sym.get(p[0]); s2 = ensp2sym.get(p[1])
            if s1 in gene2idx and s2 in gene2idx:
                gold_idx.add((gene2idx[s1], gene2idx[s2]))

trrust_path = f"{VALID}/trrust_human.tsv"
if __import__('os').path.exists(trrust_path):
    with open(trrust_path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) >= 2:
                g1, g2 = p[0].upper(), p[1].upper()
                if g1 in gene2idx and g2 in gene2idx:
                    gold_idx.add((gene2idx[g1], gene2idx[g2]))

print(f"  d=200 STRING-anchored genes, gold pairs: {len(gold_idx)}")

# Baseline: cluster_aware
m_bio = cs.CausalDiscovery(X_bio, method="cluster_aware", device=DEV, verbose=False)
m_bio.fit(verbose=False)
W_b = m_bio._network.adjacency
pred = set((i,j) for i in range(200) for j in range(200)
           if i!=j and abs(W_b[i,j]) > 0.3)
tp_b = len(pred & gold_idx)
prec_b = tp_b/max(len(pred),1)
print(f"  Baseline: edges={len(pred)}, validated={tp_b}, prec={prec_b:.4f}")

# P2: ASCEND two-tier
upstream_idx, downstream_idx = [], []
for i, g in enumerate(gene_list):
    if degree[g] >= sorted(degree.values(), reverse=True)[40]:  # top 40 connected
        upstream_idx.append(i)
    else:
        downstream_idx.append(i)
print(f"  ASCEND tiers: {len(upstream_idx)} upstream, {len(downstream_idx)} downstream")
ascend_bio = two_tier_discovery(X_bio, upstream_idx, downstream_idx, device=DEV, verbose=False)
W_a = ascend_bio["adjacency"]
pred_a = set((i,j) for i in range(200) for j in range(200)
             if i!=j and abs(W_a[i,j]) > 0.3)
tp_a = len(pred_a & gold_idx)
prec_a = tp_a/max(len(pred_a),1)
print(f"  ASCEND: edges={len(pred_a)}, validated={tp_a}, prec={prec_a:.4f}")

print(f"\n  Bio Precision: baseline={prec_b:.4f} -> ASCEND={prec_a:.4f} "
      f"(gain={prec_a-prec_b:+.4f})")

print("\nDONE")
