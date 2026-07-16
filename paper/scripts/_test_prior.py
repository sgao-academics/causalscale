"""Test prior knowledge constraint: give 50% of true DAG, check F1 improvement."""
import sys, numpy as np, torch
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
from causalscale.core.engine import CausalDiscoveryEngine, EngineConfig

np.random.seed(42)
d, n = 50, 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Generate ER DAG
W_true = np.triu(np.random.binomial(1, 0.1, (d, d)), k=1).astype(float)
for i in range(d):
    for j in range(i+1, d):
        if W_true[i, j] > 0:
            W_true[i, j] = np.random.uniform(0.3, 0.8)
true_edges = int(np.sum(W_true > 0))
print(f"True edges: {true_edges}")

X = np.random.randn(n, d)
for j in range(d):
    for p in range(j):
        if W_true[p, j] > 0:
            X[:, j] += W_true[p, j] * X[:, p]
    X[:, j] += 0.3 * np.random.randn(n)
X = (X - X.mean(0)) / X.std(0).clip(min=1e-8)

def quick_f1(W_pred, W_true, thresh=0.2):
    Wb = (np.abs(W_pred) > thresh).astype(int)
    Wt = (np.abs(W_true) > 0).astype(int)
    tp = int(np.sum(Wb & Wt)) - int(np.sum(np.diag(Wb) & np.diag(Wt)))
    fp = int(np.sum(Wb & (1-Wt))) - int(np.sum(np.diag(Wb) & (1-np.diag(Wt))))
    fn = int(np.sum((1-Wb) & Wt)) - int(np.sum((1-np.diag(Wb)) & np.diag(Wt)))
    p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
    return 2*p*r/(p+r) if p+r>0 else 0, p, r, tp, fp, fn

# ── Baseline (no prior) ──
eng_no = CausalDiscoveryEngine(d=d, mode="cluster_aware", device=DEVICE)
res_no = eng_no.fit(X)
f1_no, p_no, r_no, tp_no, fp_no, fn_no = quick_f1(res_no.adjacency, W_true)
print(f"\n  No prior:    F1={f1_no:.4f}, P={p_no:.4f}, R={r_no:.4f}, "
      f"TP={tp_no}/{true_edges}")

# ── With prior (50% of true edges revealed) ──
rng = np.random.RandomState(123)
prior_mask = np.zeros((d, d))
true_idx = np.where(W_true > 0)
n_reveal = len(true_idx[0]) // 2  # reveal 50%
reveal = rng.choice(len(true_idx[0]), n_reveal, replace=False)
for k in reveal:
    prior_mask[true_idx[0][k], true_idx[1][k]] = 1.0
print(f"  Prior: {int(prior_mask.sum())} known edges (50%)")

for pw in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
    eng_p = CausalDiscoveryEngine(d=d, mode="cluster_aware", device=DEVICE,
                                   prior_matrix=prior_mask, prior_weight=pw)
    res_p = eng_p.fit(X)
    f1_p, pp, rp, tpp, fpp, fnp = quick_f1(res_p.adjacency, W_true)
    gain = f1_p - f1_no
    print(f"  prior_w={pw:.1f}: F1={f1_p:.4f}, P={pp:.4f}, R={rp:.4f}, "
          f"TP={tpp}/{true_edges}, GAIN={gain:+.4f}")
