"""
FINAL COMPREHENSIVE BENCHMARK: causalscale v3.1 vs NOTEARS.
Tests: single engine, stability, ensemble across d=30-100.
"""
import sys, numpy as np, time, json
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

def best_f1(W, W_true):
    best = 0
    for th in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        f1, p, r, tp, fp, fn = quick_f1(W, W_true, th)
        if f1 > best: best = f1
    return best

results = {}
for d, n in [(30, 500), (50, 500), (80, 500), (100, 500)]:
    print(f"\n{'='*60}")
    print(f"  d={d}, n={n}")
    
    np.random.seed(42)
    W_true = np.triu(np.random.binomial(1, 0.1, (d, d)), k=1).astype(float)
    for i in range(d):
        for j in range(i+1, d):
            if W_true[i, j] > 0:
                W_true[i, j] = np.random.uniform(0.3, 0.8)
    true_edges = int(np.sum(W_true > 0))
    
    X = np.random.randn(n, d)
    for j in range(d):
        for p in range(j):
            if W_true[p, j] > 0:
                X[:, j] += W_true[p, j] * X[:, p]
        X[:, j] += 0.3 * np.random.randn(n)
    X = (X - X.mean(0)) / X.std(0).clip(min=1e-8)
    
    row = {"true_edges": true_edges}
    
    # causalscale cluster_aware
    m = cs.CausalDiscovery(X, method="cluster_aware", device="cuda", verbose=False)
    m.fit(verbose=False)
    row["cs_single"] = round(best_f1(m._network.adjacency, W_true), 4)
    
    # causalscale stability 5-seed
    m = cs.CausalDiscovery(X, method="cluster_aware", device="cuda",
                            n_seeds=5, verbose=False)
    m.fit(verbose=False)
    row["cs_stability5"] = round(best_f1(m._network.adjacency, W_true), 4)
    
    # causalscale ensemble
    m = cs.CausalDiscovery(X, method="ensemble", device="cuda", verbose=False)
    m.fit(verbose=False)
    row["cs_ensemble"] = round(best_f1(m._network.adjacency, W_true), 4)
    
    # NOTEARS baseline
    import torch
    X_t = torch.tensor(X, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    W = torch.zeros((d, d), device=X_t.device, requires_grad=True)
    W.data = torch.randn(d, d, device=X_t.device) * 0.01
    def h_fn(W): return torch.trace(torch.matrix_exp(W*W)) - d
    rho, alpha = 1.0, 0.0
    opt = torch.optim.Adam([W], lr=0.002)
    for outer in range(30):
        for _ in range(200):
            opt.zero_grad()
            R = X_t - X_t @ W
            loss = 0.5/n*torch.sum(R**2) + 0.1*torch.sum(torch.abs(W))
            h = h_fn(W)
            total = loss + alpha*h + 0.5*rho*h*h
            total.backward()
            opt.step()
        with torch.no_grad():
            h_val = h_fn(W).item()
        if h_val < 1e-8: break
        alpha += rho * h_val
        rho = min(rho*5, 1e10)
    row["notears"] = round(best_f1(W.detach().cpu().numpy(), W_true), 4)
    
    results[f"d{d}"] = row
    print(f"  True: {true_edges} | cs_single={row['cs_single']} | "
          f"cs_stab5={row['cs_stability5']} | cs_ens={row['cs_ensemble']} | "
          f"NOTEARS={row['notears']}")
    print(f"  vs NOTEARS: single={row['cs_single']-row['notears']:+.4f} | "
          f"stability={row['cs_stability5']-row['notears']:+.4f} | "
          f"ensemble={row['cs_ensemble']-row['notears']:+.4f}")

# Table
print(f"\n{'='*70}")
print(f"  COMPREHENSIVE BENCHMARK: causalscale v3.1 vs NOTEARS")
print(f"{'='*70}")
print(f"  {'d':>5s} {'True':>5s} {'cs_single':>10s} {'cs_stab5':>10s} "
      f"{'cs_ens':>10s} {'NOTEARS':>10s} {'best_gain':>10s}")
print(f"  {'-'*65}")
for d, r in results.items():
    best_cs = max(r["cs_single"], r["cs_stability5"], r["cs_ensemble"])
    gain = best_cs - r["notears"]
    print(f"  {d:>5s} {r['true_edges']:>5d} {r['cs_single']:>10.4f} "
          f"{r['cs_stability5']:>10.4f} {r['cs_ensemble']:>10.4f} "
          f"{r['notears']:>10.4f} {gain:>+10.4f}")

with open(r"D:\NO.1\causalscale_kdd2027_experiments\exp16_final_bench.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to exp16_final_bench.json")
