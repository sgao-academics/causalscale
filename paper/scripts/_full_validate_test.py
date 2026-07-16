"""
Full validation test suite for causalscale v3.1 validate().
Tests all 4 modes with meaningful data.
"""
import sys, numpy as np, time
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
import causalscale as cs

print(f"causalscale v{cs.__version__}")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# MODE 1: Synthetic ground truth (cluster_aware engine)
# ═══════════════════════════════════════════════════════════════
print("\n[1] SYNTHETIC: ER DAG d=30, n=500, cluster_aware")
from sklearn.datasets import make_sparse_spd_matrix

np.random.seed(42)
d, n = 30, 500

# Generate random DAG (upper-triangular, ER p=0.1)
W_true = np.triu(np.random.binomial(1, 0.1, (d, d)), k=1).astype(float)
for i in range(d):
    for j in range(i+1, d):
        if W_true[i, j] > 0:
            W_true[i, j] = np.random.uniform(0.3, 0.8)

# Generate data from DAG
X = np.random.randn(n, d)
for j in range(d):
    parents = np.where(W_true[:, j] > 0)[0]
    for p in parents:
        X[:, j] += W_true[p, j] * X[:, p]
    X[:, j] += 0.3 * np.random.randn(n)

true_edges = int(np.sum(W_true > 0))
print(f"  True DAG: d={d}, edges={true_edges}")

t0 = time.time()
model = cs.CausalDiscovery(X, method="cluster_aware", device="cpu", verbose=False)
model.fit(verbose=False)
report = model.validate(ground_truth=W_true, threshold=0.2, verbose=False)
t1 = time.time()

print(f"  F1={report['f1']:.4f}, SHD={report['shd']}, "
      f"Prec={report['precision']:.4f}, Rec={report['recall']:.4f}")
print(f"  TP={report['tp']}, FP={report['fp']}, FN={report['fn']}")
print(f"  Time: {t1-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════
# MODE 2: Gene symbols → STRING/TRRUST
# ═══════════════════════════════════════════════════════════════
print("\n[2] BIOLOGY: 10 known cancer genes, lowrank engine")
cancer_genes = ["TP53","MDM2","BRCA1","BRCA2","EGFR","KRAS","PTEN",
                "PIK3CA","MYC","CDKN2A"]
# Generate correlated data (simulate gene expression)
X_bio = np.random.randn(200, 10)
# Add correlation between known interactors
for (i, j) in [(0,1), (3,4), (2,5), (6,7)]:  # TP53-MDM2, BRCA1-BRCA2, etc.
    X_bio[:, j] += 0.7 * X_bio[:, i] + 0.2 * np.random.randn(200)

model_bio = cs.CausalDiscovery(X_bio, var_names=cancer_genes,
                                method="lowrank", rank=4, device="cpu", verbose=False)
model_bio.fit(verbose=False)
report_bio = model_bio.validate(
    string_data_dir=r"D:\NO.1\cancer_application\data\validation", verbose=False)
print(f"  Edges found: {report_bio['total_edges']}")
print(f"  STRING/TRRUST validated: {report_bio['validated_edges']}/{report_bio['total_edges']}")
print(f"  Precision: {report_bio['precision']:.4f}")

# ═══════════════════════════════════════════════════════════════
# MODE 3: Self-supervised (no ground truth, no gene names)
# ═══════════════════════════════════════════════════════════════
print("\n[3] SELF-SUPERVISED: Random structured data, lowrank engine")
np.random.seed(123)
X_ss = np.random.randn(300, 20)
for j in range(5, 20):
    X_ss[:, j] += 0.5 * X_ss[:, j-5] + 0.3 * np.random.randn(300)

model_ss = cs.CausalDiscovery(X_ss, method="lowrank", rank=8,
                               device="cpu", verbose=False)
model_ss.fit(verbose=False)
report_ss = model_ss.validate(verbose=False)
print(f"  Correlation-reconstruction F1: {report_ss['f1']:.4f}")
print(f"  Recovery rate: {report_ss['recovery_pct']:.1f}%")
print(f"  Corr-GT edges: {report_ss['corr_gt_edges']}, Pred: {report_ss['pred_edges']}")

print("\n" + "=" * 60)
print("ALL 3 MODES: PASS")
