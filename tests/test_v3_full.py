"""causalscale V3.0.0 Full Smoke Test: All 6 engines."""
import sys, time, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, r'C:\Users\高帅东\Desktop\causalscale')
import numpy as np
import causalscale as cs

np.random.seed(42)
d, n = 20, 300
W_true = np.zeros((d, d))
for i in range(d):
    for j in range(i + 1, min(d, i + 3)):
        W_true[i, j] = np.random.uniform(-0.7, 0.7)
X = np.random.randn(n, d) @ np.linalg.inv(np.eye(d) - W_true.T)
X = X.astype(np.float32)

tests = [
    ("lowrank", dict(rank=8, epochs=200)),
    ("multi_scale", dict(rank="auto", epochs=200)),
    ("cluster_aware", dict(rank=8, epochs=200, n_clusters=5)),
    ("transformer", dict(rank=16, epochs=50)),
    ("multimodal", dict(rank="auto", outer=15, inner=80, n_seeds=2)),
    ("auto", dict(rank="auto", epochs=200)),
]

passed = 0
total_time = 0
for name, kwargs in tests:
    try:
        print(f"[{name}] ", end="", flush=True)
        t0 = time.time()
        m = cs.CausalDiscovery(X, method=name, **kwargs)
        m.fit(verbose=False)
        net = m.get_network()
        elapsed = time.time() - t0
        total_time += elapsed
        status = "PASS" if net.edge_count > 0 else "FAIL(0 edges)"
        if net.edge_count > 0:
            passed += 1
        print(f"{net.edge_count:3d} edges in {elapsed:.1f}s [{status}]", flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)

print(f"\n{'='*50}")
print(f"PASSED: {passed}/{len(tests)} modes in {total_time:.0f}s")
print(f"causalscale V{cs.__version__}")
