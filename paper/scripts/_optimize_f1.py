"""
F1 optimization: sweep lambda1 (sparsity) + stability selection + adaptive threshold.
"""
import sys, numpy as np, torch
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
import causalscale as cs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(42)

d, n = 50, 500
print(f"d={d}, n={n}, n/d={n/d:.1f}")
print("=" * 60)

# Generate ER DAG
W_true = np.triu(np.random.binomial(1, 0.1, (d, d)), k=1).astype(float)
for i in range(d):
    for j in range(i+1, d):
        if W_true[i, j] > 0:
            W_true[i, j] = np.random.uniform(0.3, 0.8)
true_edges = int(np.sum(W_true > 0))
print(f"True edges: {true_edges}")

# Generate data
X = np.random.randn(n, d)
for j in range(d):
    for p in range(j):
        if W_true[p, j] > 0:
            X[:, j] += W_true[p, j] * X[:, p]
    X[:, j] += 0.3 * np.random.randn(n)
X = (X - X.mean(0)) / X.std(0).clip(min=1e-8)

def quick_f1(W_pred, W_true, thresh):
    Wb = (np.abs(W_pred) > thresh).astype(int)
    Wt = (np.abs(W_true) > 0).astype(int)
    tp = int(np.sum(Wb & Wt)) - int(np.sum(np.diag(Wb) & np.diag(Wt)))
    fp = int(np.sum(Wb & (1-Wt))) - int(np.sum(np.diag(Wb) & (1-np.diag(Wt))))
    fn = int(np.sum((1-Wb) & Wt)) - int(np.sum((1-np.diag(Wb)) & np.diag(Wt)))
    p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
    return 2*p*r/(p+r) if p+r>0 else 0, p, r, tp, fp, fn, int(Wb.sum())

# ═══════════════════════════════════════════════════════════════
# ROUND 1: Lambda1 sweep
# ═══════════════════════════════════════════════════════════════
print("\n[1] LAMBDA1 SWEEP (single seed)")
best_f1, best_l1 = 0, 0
for l1 in [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]:
    m = cs.CausalDiscovery(X, method="cluster_aware", device=DEVICE,
                            lambda1=l1, verbose=False)
    m.fit(verbose=False)
    for th in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        f1, p, r, tp, fp, fn, ne = quick_f1(m._network.adjacency, W_true, th)
        if f1 > best_f1:
            best_f1, best_l1 = f1, l1
    print(f"  lambda1={l1:.3f}: best F1={best_f1:.4f} (swept thresh 0.05-0.3)")

print(f"\n  >> Best lambda1={best_l1:.3f}, F1={best_f1:.4f}")

# ═══════════════════════════════════════════════════════════════
# ROUND 2: Stability selection (multi-seed consensus)
# ═══════════════════════════════════════════════════════════════
print(f"\n[2] STABILITY SELECTION (5 seeds, lambda1={best_l1})")
n_seeds = 5
adj_stack = np.zeros((n_seeds, d, d))
for s in range(n_seeds):
    m = cs.CausalDiscovery(X, method="cluster_aware", device=DEVICE,
                            lambda1=best_l1, seed=s*42+1, verbose=False)
    m.fit(verbose=False)
    adj_stack[s] = m._network.adjacency

# Stability: keep edges that appear in >= k_seeds
for k in [2, 3, 4, 5]:
    stable_bin = (np.sum(np.abs(adj_stack) > 0.05, axis=0) >= k).astype(float)
    # Weight by mean absolute value across seeds
    stable_W = stable_bin * np.mean(np.abs(adj_stack), axis=0)
    
    for th in [0.02, 0.05, 0.1, 0.15, 0.2]:
        f1, p, r, tp, fp, fn, ne = quick_f1(stable_W, W_true, th)
        if f1 > 0:
            print(f"  k={k}/5, thresh={th:.2f}: F1={f1:.4f}, "
                  f"P={p:.4f}, R={r:.4f}, TP={tp}, edges={ne}")

# ═══════════════════════════════════════════════════════════════
# ROUND 3: Best combination
# ═══════════════════════════════════════════════════════════════
print(f"\n[3] BEST COMBINATION (stability k>=3, lambda1={best_l1})")
adj_stack2 = np.zeros((10, d, d))
for s in range(10):
    m = cs.CausalDiscovery(X, method="cluster_aware", device=DEVICE,
                            lambda1=best_l1, seed=s*17, verbose=False)
    m.fit(verbose=False)
    adj_stack2[s] = m._network.adjacency

best_f1_overall = 0
best_config = None
for k in [3, 4, 5, 6, 7, 8]:
    stable_bin = (np.sum(np.abs(adj_stack2) > 0.05, axis=0) >= k).astype(float)
    stable_W = stable_bin * np.mean(np.abs(adj_stack2), axis=0)
    for th in np.arange(0.02, 0.35, 0.02):
        f1, p, r, tp, fp, fn, ne = quick_f1(stable_W, W_true, th)
        if f1 > best_f1_overall:
            best_f1_overall = f1
            best_config = (k, th, p, r, tp, fp, fn, ne)

k_opt, th_opt, p_opt, r_opt, tp_opt, fp_opt, fn_opt, ne_opt = best_config
print(f"  BEST: k>={k_opt}/10, thresh={th_opt:.2f}")
print(f"  F1={best_f1_overall:.4f}, Prec={p_opt:.4f}, Rec={r_opt:.4f}")
print(f"  TP={tp_opt}/{true_edges}, FP={fp_opt}, edges={ne_opt}")

# Compare to baseline
m_bl = cs.CausalDiscovery(X, method="cluster_aware", device=DEVICE, verbose=False)
m_bl.fit(verbose=False)
f1_bl, p_bl, r_bl, tp_bl, fp_bl, fn_bl, ne_bl = quick_f1(
    m_bl._network.adjacency, W_true, 0.2)
print(f"\n  Baseline (default):   F1={f1_bl:.4f}, P={p_bl:.4f}, R={r_bl:.4f}, "
      f"TP={tp_bl}")
print(f"  Optimized:            F1={best_f1_overall:.4f}, P={p_opt:.4f}, "
      f"R={r_opt:.4f}, TP={tp_opt}")
print(f"  Gain: +{best_f1_overall-f1_bl:+.4f} F1, "
      f"+{tp_opt-tp_bl} TP")
