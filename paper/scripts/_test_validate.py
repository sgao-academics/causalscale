"""Test causalscale V3.1 validate() auto-detect evaluation."""
import sys, numpy as np
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
import causalscale as cs

print(f"causalscale v{cs.__version__}")

# ── Mode 1: Synthetic ground truth ──
print("\n=== MODE 1: Synthetic data with ground truth ===")
np.random.seed(42)
n, d = 200, 10
# Simple ground truth DAG (upper triangular)
W_true = np.triu(np.random.binomial(1, 0.3, (d, d)), k=1).astype(float)
W_true[W_true > 0] = np.random.uniform(0.3, 0.8, int(W_true.sum()))
X = np.random.randn(n, d)
for i in range(d):
    for j in range(i):
        if W_true[j, i] > 0:
            X[:, i] += W_true[j, i] * X[:, j] + 0.1 * np.random.randn(n)

model = cs.CausalDiscovery(X, method="lowrank", rank=4, device="cpu", verbose=False)
model.fit(verbose=False)
report = model.validate(ground_truth=W_true, threshold=0.2, verbose=True)
assert report["mode"] == "synthetic"
print(f"  Result: F1={report['f1']:.4f}, SHD={report['shd']}")

# ── Mode 2: Gene symbols ──
print("\n=== MODE 2: Gene symbols (auto-detect biology) ===")
gene_data = np.random.randn(100, 5)
gene_names = ["TP53", "MDM2", "BRCA1", "EGFR", "KRAS"]
model2 = cs.CausalDiscovery(gene_data, var_names=gene_names,
                             method="lowrank", rank=2, device="cpu", verbose=False)
model2.fit(verbose=False)
report2 = model2.validate(verbose=False)
print(f"  Mode: {report2['mode']}")
assert report2["mode"] == "biology", f"Expected biology, got {report2['mode']}"

# ── Mode 3: Default variable names ──
print("\n=== MODE 3: Default names (self-supervised) ===")
plain_data = np.random.randn(100, 8)
model3 = cs.CausalDiscovery(plain_data, method="lowrank", rank=4,
                             device="cpu", verbose=False)
model3.fit(verbose=False)
report3 = model3.validate(verbose=False)
print(f"  Mode: {report3['mode']}")
assert report3["mode"] == "self_supervised"

print("\nALL 3 MODES PASSED")
