"""DAGMA vs NOTEARS head-to-head."""
import sys, numpy as np, torch, time
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
from causalscale.core.dagma_solver import dagma_linear
from causalscale.core._notears import run_notears as notears

DEV = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(42)

def best_f1(W, W_true):
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

for d, n in [(50, 500), (80, 500)]:
    print(f"\n{'='*50}")
    print(f"  d={d}, n={n}")
    W_true = np.triu(np.random.binomial(1, 0.1, (d, d)), k=1).astype(float)
    for i in range(d):
        for j in range(i+1, d):
            if W_true[i,j] > 0:
                W_true[i,j] = np.random.uniform(0.3, 0.8)
    true_e = int(np.sum(W_true > 0))
    print(f"  True edges: {true_e}")

    X = np.random.randn(n, d)
    for j in range(d):
        for p in range(j):
            if W_true[p,j] > 0:
                X[:,j] += W_true[p,j] * X[:,p]
        X[:,j] += 0.3 * np.random.randn(n)
    X = (X - X.mean(0)) / X.std(0).clip(min=1e-8)

    # NOTEARS
    t0 = time.time()
    W_n, _, _, _ = notears(X, device=DEV, outer=40, inner=250)
    f1_n = best_f1(W_n, W_true)
    t_n = time.time() - t0

    # DAGMA
    t0 = time.time()
    W_d = dagma_linear(X, device=DEV, verbose=False)
    f1_d = best_f1(W_d, W_true)
    t_d = time.time() - t0

    print(f"  NOTEARS: F1={f1_n:.4f}, {t_n:.1f}s")
    print(f"  DAGMA:   F1={f1_d:.4f}, {t_d:.1f}s, gain={f1_d-f1_n:+.4f}")
    print(f"  Winner: {'DAGMA' if f1_d>f1_n else 'NOTEARS'}")
