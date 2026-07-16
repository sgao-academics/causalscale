"""Test ensemble consensus voting vs single-engine baselines."""
import sys, numpy as np, time
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
import causalscale as cs

def quick_f1(W_pred, W_true, thresh):
    Wb = (np.abs(W_pred) > thresh).astype(int)
    Wt = (np.abs(W_true) > 0).astype(int)
    tp = int(np.sum(Wb & Wt)) - int(np.sum(np.diag(Wb) & np.diag(Wt)))
    fp = int(np.sum(Wb & (1-Wt))) - int(np.sum(np.diag(Wb) & (1-np.diag(Wt))))
    fn = int(np.sum((1-Wb) & Wt)) - int(np.sum((1-np.diag(Wb)) & np.diag(Wt)))
    p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
    return 2*p*r/(p+r) if p+r>0 else 0, p, r, tp, fp, fn

for d, n in [(30, 500), (50, 500), (80, 500)]:
    print(f"\n{'='*60}")
    print(f"  d={d}, n={n}")
    print(f"{'='*60}")

    np.random.seed(42)
    W_true = np.triu(np.random.binomial(1, 0.1, (d, d)), k=1).astype(float)
    for i in range(d):
        for j in range(i+1, d):
            if W_true[i, j] > 0:
                W_true[i, j] = np.random.uniform(0.3, 0.8)
    true_edges = int(np.sum(W_true > 0))
    print(f"  True edges: {true_edges}")

    X = np.random.randn(n, d)
    for j in range(d):
        for p in range(j):
            if W_true[p, j] > 0:
                X[:, j] += W_true[p, j] * X[:, p]
        X[:, j] += 0.3 * np.random.randn(n)
    X = (X - X.mean(0)) / X.std(0).clip(min=1e-8)

    # ── ENSEMBLE ──
    t0 = time.time()
    m_ens = cs.CausalDiscovery(X, method="ensemble", device="cuda",
                                ensemble_min_votes=2, verbose=False)
    m_ens.fit(verbose=True)
    t_ens = time.time() - t0

    # ── Single engines (for comparison) ──
    results = {}
    for eng in ["cluster_aware", "lowrank", "multi_scale"]:
        m = cs.CausalDiscovery(X, method=eng, device="cuda", verbose=False)
        m.fit(verbose=False)
        # Sweep threshold for best F1
        best_f1_s = 0
        for th in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            f1s, ps, rs, tps, fps, fns = quick_f1(m._network.adjacency, W_true, th)
            if f1s > best_f1_s:
                best_f1_s = f1s
        results[eng] = best_f1_s

    # Ensemble F1 (sweep threshold)
    best_f1_e = 0
    best_th = 0
    for th in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        f1e, pe, re, tpe, fpe, fne = quick_f1(m_ens._network.adjacency, W_true, th)
        if f1e > best_f1_e:
            best_f1_e = f1e
            best_th = th

    # Best engine F1
    best_single = max(results.values())
    best_single_name = max(results, key=results.get)

    print(f"\n  Single-engine best: {best_single_name} F1={best_single:.4f}")
    for eng, f1 in results.items():
        print(f"    {eng}: {f1:.4f}")
    print(f"  Ensemble (min_votes=2): F1={best_f1_e:.4f} (thresh={best_th})")
    gain = best_f1_e - best_single
    print(f"  GAIN: {gain:+.4f} ({'+' if gain>0 else ''}{gain/best_single*100:+.1f}%)")
    print(f"  Unanimous edges: {m_ens._network.metadata['unanimous_edges']}")
    print(f"  Consensus edges: {m_ens._network.metadata['consensus_edges']}")
    print(f"  Time: {t_ens:.1f}s")
