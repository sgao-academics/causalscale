"""
CAGate strategy: KMeans cluster → per-cluster NOTEARS → union edges.
This is like ensemble at the data level, not the model level.
"""
import sys, numpy as np, torch, time
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
from causalscale.core._notears import run_cagate

np.random.seed(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def f1_best(W, W_true):
    bf = 0; bp = 0; br = 0
    for th in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        Wb = (np.abs(W) > th).astype(int)
        Wt = (np.abs(W_true) > 0).astype(int)
        tp = int(np.sum(Wb & Wt)) - int(np.sum(np.diag(Wb) & np.diag(Wt)))
        fp = int(np.sum(Wb & (1-Wt))) - int(np.sum(np.diag(Wb) & (1-np.diag(Wt))))
        fn = int(np.sum((1-Wb) & Wt)) - int(np.sum((1-np.diag(Wb)) & np.diag(Wt)))
        p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
        f = 2*p*r/(p+r) if p+r>0 else 0
        if f > bf: bf, bp, br = f, p, r
    return bf, bp, br

for d, n in [(50, 500), (80, 500), (100, 500)]:
    print(f"\n{'='*55}")
    print(f"  d={d}, n={n}")
    
    W_true = np.triu(np.random.binomial(1, 0.1, (d, d)), k=1).astype(float)
    for i in range(d):
        for j in range(i+1, d):
            if W_true[i,j] > 0:
                W_true[i,j] = np.random.uniform(0.3, 0.8)
    true_edges = int(np.sum(W_true > 0))
    
    X = np.random.randn(n, d)
    for j in range(d):
        for p in range(j):
            if W_true[p,j] > 0:
                X[:,j] += W_true[p,j] * X[:,p]
        X[:,j] += 0.3 * np.random.randn(n)
    X = (X - X.mean(0)) / X.std(0).clip(min=1e-8)
    
    # Baseline: standard NOTEARS
    from causalscale.core._notears import run_notears
    t0 = time.time()
    W_n, ec_n, h_n, _ = run_notears(X, device=DEV, outer=40, inner=250)
    f_n, p_n, r_n = f1_best(W_n, W_true)
    t_n = time.time() - t0
    print(f"  Standard NOTEARS:     F1={f_n:.4f}, P={p_n:.4f}, R={r_n:.4f}, "
          f"edges={ec_n}, t={t_n:.1f}s")
    
    # CAGate with different K
    from causalscale.core._notears import run_cagate
    for K in [4, 6, 8, 10, 12]:
        t0 = time.time()
        W_c, ec_c, t_c = run_cagate(X, device=DEV, K=K, n_seeds=2, outer=30, inner=200)
        f_c, p_c, r_c = f1_best(W_c, W_true)
        gain = f_c - f_n
        tag = "🔥" if gain > 0.05 else ("✅" if gain > 0 else "  ")
        print(f"  CAGate K={K:2d}:         F1={f_c:.4f}, P={p_c:.4f}, R={r_c:.4f}, "
              f"edges={ec_c}, t={t_c:.1f}s, gain={gain:+.4f} {tag}")
