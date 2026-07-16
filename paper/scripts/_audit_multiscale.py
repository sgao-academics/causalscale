"""
Audit: Is multi_scale engine correctly configured?
"""
import sys, numpy as np, torch, time
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
from causalscale.core.engine import CausalDiscoveryEngine
from causalscale.core.multi_scale import MultiScaleLowRank

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

for d, n in [(50, 500), (100, 500)]:
    print(f"\n{'='*55}")
    print(f"  AUDIT: multi_scale d={d}, n={n}")
    
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
    
    # Baseline: cluster_aware
    from causalscale.core._notears import run_notears
    W_n, _, _, _ = run_notears(X, device=DEV, outer=40, inner=250)
    f1_n = best_f1(W_n, W_true)
    print(f"  cluster_aware (NOTEARS): F1={f1_n:.4f}")
    
    # Audit multi_scale: check what scale_ranks and sparsities are used
    # Default: n_scales=3, auto rank
    for n_scales in [1, 2, 3, 4]:
        base_r = max(4, 16 // 4)  # approximate auto-rank
        scale_ranks = [min(d, base_r * (2**s)) for s in range(n_scales)]
        scale_sparsities = [0.15/(2**s) for s in range(n_scales)]
        scale_weights = [1.0/(1.5**i) for i in range(n_scales)]
        
        eng = CausalDiscoveryEngine(d=d, mode="multi_scale", device=DEV,
                                     n_scales=n_scales,
                                     scale_ranks=scale_ranks,
                                     scale_sparsities=scale_sparsities,
                                     verbose=False)
        try:
            result = eng.fit(X)
            f1_ms = best_f1(result.adjacency, W_true)
            nz = int(np.sum(np.abs(result.adjacency) > 0.05))
            print(f"  multi_scale (n_scales={n_scales}, ranks={scale_ranks}): "
                  f"F1={f1_ms:.4f}, nonzero={nz}")
        except Exception as e:
            print(f"  multi_scale (n_scales={n_scales}): FAILED - {str(e)[:80]}")

    # Try with fixed rank
    for r in [4, 8, 16, 32]:
        eng = CausalDiscoveryEngine(d=d, mode="multi_scale", device=DEV,
                                     rank=r, n_scales=2,
                                     scale_ranks=[r, r*2],
                                     scale_sparsities=[0.1, 0.05],
                                     verbose=False)
        try:
            result = eng.fit(X)
            f1_ms = best_f1(result.adjacency, W_true)
            nz = int(np.sum(np.abs(result.adjacency) > 0.05))
            print(f"  multi_scale (rank={r}, 2 scales): F1={f1_ms:.4f}, nonzero={nz}")
        except Exception as e:
            print(f"  multi_scale (rank={r}): FAILED - {str(e)[:80]}")
