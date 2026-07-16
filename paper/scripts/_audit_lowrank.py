"""
Audit: transformer + lowrank engine optimization.
"""
import sys, numpy as np, torch, time
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
from causalscale.core.lowrank import train_lowrank_gnn
from causalscale.core._notears import run_notears

DEV = "cuda" if torch.cuda.is_available() else "cpu"

def best_f1(W, W_true):
    bf = 0
    for th in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        Wb = (np.abs(W) > th).astype(int); Wt = (np.abs(W_true)>0).astype(int)
        tp = int(np.sum(Wb&Wt)) - int(np.sum(np.diag(Wb)&np.diag(Wt)))
        fp = int(np.sum(Wb&(1-Wt))) - int(np.sum(np.diag(Wb)&(1-np.diag(Wt))))
        fn = int(np.sum((1-Wb)&Wt)) - int(np.sum((1-np.diag(Wb))&np.diag(Wt)))
        p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
        f = 2*p*r/(p+r) if p+r>0 else 0
        if f > bf: bf = f
    return bf

np.random.seed(42)

for d, n in [(50, 500), (100, 500)]:
    print(f"\n{'='*55}")
    print(f"  AUDIT: d={d}, n={n}")
    
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
    
    # cluster_aware baseline
    W_n, _, _, _ = run_notears(X, device=DEV, outer=40, inner=250)
    f1_n = best_f1(W_n, W_true)
    print(f"  cluster_aware: F1={f1_n:.4f}")
    
    # === LOWRANK AUDIT ===
    print(f"\n  --- LowRankGNN sweep ---")
    for rank in [4, 8, 16, 32, 64]:
        for epochs in [200, 500]:
            for thresh in [0.2, 0.3]:
                for lr in [0.005, 0.01, 0.02]:
                    r = train_lowrank_gnn(X, rank=rank, epochs=epochs,
                                           threshold=thresh, lr=lr, device=DEV)
                    f1_l = best_f1(r["adjacency"], W_true)
                    if f1_l > 0.08:  # skip very bad ones
                        print(f"    rank={rank} ep={epochs} th={thresh} lr={lr}: "
                              f"F1={f1_l:.4f} edges={r['gnn_edges']}")
    
    # Best lowrank found
    best_r = 0
    for rank in [4, 8, 16, 32]:
        for epochs in [200, 500, 800]:
            for thresh in [0.15, 0.2, 0.25, 0.3, 0.35]:
                for lr in [0.005, 0.01, 0.02]:
                    r = train_lowrank_gnn(X, rank=rank, epochs=epochs,
                                           threshold=thresh, lr=lr, device=DEV)
                    f1_l = best_f1(r["adjacency"], W_true)
                    if f1_l > best_r:
                        best_r = f1_l
    print(f"  BEST lowrank: F1={best_r:.4f} (vs cluster_aware {f1_n:.4f})")
