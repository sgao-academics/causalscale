"""Test pseudo_causal mode: NOTEARS as reference DAG."""
import sys, numpy as np, time
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
import causalscale as cs

np.random.seed(42)
d, n = 20, 300

# Generate data from a known DAG
W_true = np.triu(np.random.binomial(1, 0.15, (d, d)), k=1).astype(float)
for i in range(d):
    for j in range(i+1, d):
        if W_true[i, j] > 0:
            W_true[i, j] = np.random.uniform(0.3, 0.8)

X = np.random.randn(n, d)
for j in range(d):
    for p in range(j):
        if W_true[p, j] > 0:
            X[:, j] += W_true[p, j] * X[:, p]
    X[:, j] += 0.3 * np.random.randn(n)

true_edges = int(np.sum(W_true > 0))
print(f"True DAG: d={d}, n={n}, edges={true_edges}")

# Run causalscale
t0 = time.time()
model = cs.CausalDiscovery(X, method="cluster_aware", device="cpu", verbose=False)
model.fit(verbose=False)

# Mode: pseudo_ground_truth -> NOTEARS as reference
print("\n--- pseudo_ground_truth='notears' ---")
report = model.validate(pseudo_ground_truth="notears", threshold=0.2, verbose=True)
print(f"  Mode: {report['mode']}")
print(f"  Reference: {report.get('reference_method', 'N/A')}")
print(f"  F1 (vs NOTEARS): {report['f1']:.4f}")

# Mode: actual ground truth (for comparison)
print("\n--- ground_truth=W_true (for comparison) ---")
report2 = model.validate(ground_truth=W_true, threshold=0.2, verbose=False)
print(f"  F1 (vs true DAG): {report2['f1']:.4f}, SHD: {report2['shd']}")

# Mode: self-supervised (correlation)
report3 = model.validate(verbose=False)
print(f"  Correlation-reconstruction F1: {report3['f1']:.4f}")

print(f"\nTotal: {time.time()-t0:.1f}s")
print("ALL PASS")
