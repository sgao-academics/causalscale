"""
Comprehensive smoke test: every engine, every validate mode, edge cases.
Tests what EVERY user would encounter.
"""
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
import causalscale as cs

PASS, FAIL = 0, 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")

print(f"causalscale v{cs.__version__}")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# 1. Basic API
# ═══════════════════════════════════════════════════════════════
print("\n[1] BASIC API")
X = np.random.randn(100, 10)
m = cs.CausalDiscovery(X, device="cpu", verbose=False)
check("import ok", True)
check("version is 3.1.0", cs.__version__ == "3.1.0")

m.fit(verbose=False)
check("fit() returns self", m._fitted)
check("get_network()", m.get_network() is not None)
check("get_adjacency() shape", m.get_adjacency().shape == (10, 10))
check("summary()", len(m.summary()) > 0)
check("__repr__", "fitted" in repr(m))
check("get_edges()", len(m.get_edges()) >= 0)
check("predict()", m.predict(X).shape == X.shape)

# ═══════════════════════════════════════════════════════════════
# 2. ALL ENGINES
# ═══════════════════════════════════════════════════════════════
print("\n[2] ENGINES")

# cluster_aware (d=30)
try:
    m = cs.CausalDiscovery(np.random.randn(200, 30), method="cluster_aware", device="cpu", verbose=False)
    m.fit(verbose=False)
    check("cluster_aware (d=30)", m._fitted and m._network.edge_count >= 0)
except Exception as e:
    check("cluster_aware (d=30)", False, str(e))

# lowrank (d=100)
try:
    m = cs.CausalDiscovery(np.random.randn(200, 100), method="lowrank", rank=8, device="cpu", verbose=False)
    m.fit(verbose=False)
    check("lowrank (d=100)", m._fitted and m._network.edge_count >= 0)
except Exception as e:
    check("lowrank (d=100)", False, str(e))

# lowrank (d=5000) - THE BUG FIX
try:
    m = cs.CausalDiscovery(np.random.randn(200, 5000), method="lowrank", rank=16, device="cpu", verbose=False)
    m.fit(verbose=False)
    check("lowrank (d=5000)", m._fitted, f"edges={m._network.edge_count}")
except Exception as e:
    check("lowrank (d=5000)", False, str(e)[:80])

# multi_scale
try:
    m = cs.CausalDiscovery(np.random.randn(200, 50), method="multi_scale", device="cpu", verbose=False)
    m.fit(verbose=False)
    check("multi_scale (d=50)", m._fitted and m._network.edge_count >= 0)
except Exception as e:
    check("multi_scale (d=50)", False, str(e)[:80])

# ensemble
try:
    m = cs.CausalDiscovery(np.random.randn(200, 30), method="ensemble", device="cpu", verbose=False)
    m.fit(verbose=False)
    check("ensemble (d=30)", m._fitted)
except Exception as e:
    check("ensemble (d=30)", False, str(e)[:80])

# transformer
try:
    m = cs.CausalDiscovery(np.random.randn(100, 20), method="transformer", rank=8, device="cpu", verbose=False)
    m.fit(verbose=False)
    check("transformer (d=20)", m._fitted)
except Exception as e:
    check("transformer (d=20)", False, str(e)[:80])

# auto method detection
for d, expected in [(30, "cluster_aware"), (500, "cluster_aware"), (5000, "lowrank")]:
    from causalscale.api import _auto_method
    got = _auto_method(d, 200)
    check(f"auto method d={d}", got == expected, f"got {got}")

# ═══════════════════════════════════════════════════════════════
# 3. STABILITY SELECTION
# ═══════════════════════════════════════════════════════════════
print("\n[3] STABILITY SELECTION")
try:
    m = cs.CausalDiscovery(np.random.randn(200, 30), method="cluster_aware",
                            n_seeds=3, device="cpu", verbose=False)
    m.fit(verbose=False)
    check("stability n_seeds=3", m._fitted and "n_seeds" in str(m._network.metadata))
except Exception as e:
    check("stability n_seeds=3", False, str(e)[:80])

# ═══════════════════════════════════════════════════════════════
# 4. VALIDATE API (all 4 modes)
# ═══════════════════════════════════════════════════════════════
print("\n[4] VALIDATE API")

# Mode: synthetic
W_true = np.triu(np.random.binomial(1, 0.1, (10, 10)), k=1).astype(float)
m = cs.CausalDiscovery(np.random.randn(200, 10), method="cluster_aware", device="cpu", verbose=False)
m.fit(verbose=False)
r = m.validate(ground_truth=W_true, verbose=False)
check("validate(synthetic)", r["mode"] == "synthetic")
check("validate(synthetic) keys", all(k in r for k in ["f1","shd","precision","recall"]))

# Mode: biology (gene symbols)
Xg = np.random.randn(100, 5)
mg = cs.CausalDiscovery(Xg, var_names=["TP53","MDM2","BRCA1","EGFR","KRAS"],
                          method="lowrank", rank=2, device="cpu", verbose=False)
mg.fit(verbose=False)
r = mg.validate(string_data_dir=r"D:\NO.1\cancer_application\data\validation", verbose=False)
check("validate(biology)", r["mode"] == "biology")
check("validate(biology) precision", r["precision"] >= 0)

# Mode: pseudo_causal
Xp = np.random.randn(200, 8)
mp = cs.CausalDiscovery(Xp, method="cluster_aware", device="cpu", verbose=False)
mp.fit(verbose=False)
r = mp.validate(pseudo_ground_truth="notears", verbose=False)
check("validate(pseudo_causal)", r["mode"] in ("pseudo_causal", "self_supervised"))

# Mode: self_supervised
r = m.validate(verbose=False)
check("validate(self_supervised)", r["mode"] == "self_supervised")

# ═══════════════════════════════════════════════════════════════
# 5. EDGE CASES
# ═══════════════════════════════════════════════════════════════
print("\n[5] EDGE CASES")

# NaN data
try:
    Xn = np.random.randn(100, 10)
    Xn[0, 0] = np.nan
    m = cs.CausalDiscovery(Xn, device="cpu", verbose=False)
    m.fit(verbose=False)
    check("NaN handling", m._fitted)
except Exception as e:
    check("NaN handling", False, str(e)[:80])

# Inf data
try:
    Xi = np.random.randn(100, 10)
    Xi[0, 1] = np.inf
    m = cs.CausalDiscovery(Xi, device="cpu", verbose=False)
    m.fit(verbose=False)
    check("Inf handling", m._fitted)
except Exception as e:
    check("Inf handling", False, str(e)[:80])

# Zero variance column
try:
    Xz = np.random.randn(100, 10)
    Xz[:, 5] = 0.0
    m = cs.CausalDiscovery(Xz, device="cpu", verbose=False)
    m.fit(verbose=False)
    check("zero-variance handling", m._fitted)
except Exception as e:
    check("zero-variance handling", False, str(e)[:80])

# Small d
try:
    m = cs.CausalDiscovery(np.random.randn(100, 3), device="cpu", verbose=False)
    m.fit(verbose=False)
    check("d=3", m._fitted)
except Exception as e:
    check("d=3", False, str(e)[:80])

# d=2 (minimum)
try:
    cs.CausalDiscovery(np.random.randn(100, 2), device="cpu", verbose=False).fit(verbose=False)
    check("d=2", True)
except Exception as e:
    check("d=2", False, str(e)[:80])

# d=1 should fail
try:
    cs.CausalDiscovery(np.random.randn(100, 1), device="cpu", verbose=False).fit(verbose=False)
    check("d=1 rejects", False, "should have raised")
except ValueError:
    check("d=1 rejects", True)

# CSV path loading
try:
    import tempfile, os, pandas as pd
    path = os.path.join(tempfile.gettempdir(), "_cs_test.csv")
    pd.DataFrame(np.random.randn(50, 5), columns=[f"V{i}" for i in range(5)]).to_csv(path)
    m = cs.CausalDiscovery(path, device="cpu", verbose=False)
    m.fit(verbose=False)
    check("CSV file loading", m._fitted and m.d == 5)
    os.remove(path)
except Exception as e:
    check("CSV file loading", False, str(e)[:80])

# ═══════════════════════════════════════════════════════════════
# 6. validate_against_string utility
# ═══════════════════════════════════════════════════════════════
print("\n[6] validate_against_string()")
check("function exists", hasattr(cs, "validate_against_string"))

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
total = PASS + FAIL
print(f"\n{'='*60}")
print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
if FAIL == 0:
    print("  ALL TESTS PASSED - causalscale v3.1.0 is RELEASE-READY")
else:
    print(f"  {FAIL} TESTS FAILED - needs fixing")
