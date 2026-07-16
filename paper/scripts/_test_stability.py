"""Test stability selection: n_seeds=5 vs n_seeds=1."""
import sys, numpy as np, time
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
import causalscale as cs

np.random.seed(42)
d, n = 50, 500
print(f"d={d}, n={n}")

# Generate ER DAG + data
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

# ── Single seed ──
print("\n--- Single seed ---")
t0 = time.time()
m1 = cs.CausalDiscovery(X, method="cluster_aware", device="cuda", verbose=False)
m1.fit(verbose=True)
r1 = m1.validate(ground_truth=W_true, threshold=0.2, verbose=False)
print(f"  F1={r1['f1']:.4f}, Prec={r1['precision']:.4f}, Rec={r1['recall']:.4f}, "
      f"TP={r1['tp']}, time={time.time()-t0:.1f}s")

# ── Stability 5 seeds ──
print("\n--- Stability 5 seeds ---")
t0 = time.time()
m5 = cs.CausalDiscovery(X, method="cluster_aware", device="cuda",
                         n_seeds=5, verbose=False)
m5.fit(verbose=True)
r5 = m5.validate(ground_truth=W_true, threshold=0.05, verbose=False)
print(f"  F1={r5['f1']:.4f}, Prec={r5['precision']:.4f}, Rec={r5['recall']:.4f}, "
      f"TP={r5['tp']}, time={time.time()-t0:.1f}s")

# ── Stability 10 seeds ──
print("\n--- Stability 10 seeds ---")
t0 = time.time()
m10 = cs.CausalDiscovery(X, method="cluster_aware", device="cuda",
                          n_seeds=10, verbose=False)
m10.fit(verbose=True)
r10 = m10.validate(ground_truth=W_true, threshold=0.05, verbose=False)
print(f"  F1={r10['f1']:.4f}, Prec={r10['precision']:.4f}, Rec={r10['recall']:.4f}, "
      f"TP={r10['tp']}, time={time.time()-t0:.1f}s")

print(f"\n  Summary: 1seed={r1['f1']:.3f} -> 5seed={r5['f1']:.3f} "
      f"-> 10seed={r10['f1']:.3f}")
