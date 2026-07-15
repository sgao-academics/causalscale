"""V3 Smoke Test: 4 modes, 1 synthetic DAG."""
import sys, os, time, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, r'C:\Users\高帅东\Desktop\causalscale')
import numpy as np
import causalscale as cs

print(f"causalscale V{cs.__version__}")
print("=" * 50)

# Generate synthetic DAG
np.random.seed(42)
d, n = 20, 300
W_true = np.zeros((d, d))
for i in range(d):
    for j in range(i + 1, min(d, i + 3)):
        W_true[i, j] = np.random.uniform(-0.7, 0.7)
X = np.random.randn(n, d) @ np.linalg.inv(np.eye(d) - W_true.T)
X = X.astype(np.float32)

# Test 1: lowrank
print("\n[1/4] lowrank mode...", flush=True)
t0 = time.time()
m = cs.CausalDiscovery(X, method="lowrank", rank=8, epochs=200, verbose=False)
m.fit(verbose=False)
net = m.get_network()
print(f"  Edges: {net.edge_count}, Time: {net.time_s:.1f}s", flush=True)
assert net.edge_count > 0, "No edges in lowrank mode"

# Test 2: multi_scale
print("\n[2/4] multi_scale mode...", flush=True)
t0 = time.time()
m = cs.CausalDiscovery(X, method="multi_scale", rank="auto", epochs=200, verbose=False)
m.fit(verbose=False)
net = m.get_network()
print(f"  Edges: {net.edge_count}, Time: {net.time_s:.1f}s", flush=True)
assert net.edge_count > 0, "No edges in multi_scale mode"

# Test 3: cluster_aware
print("\n[3/4] cluster_aware mode...", flush=True)
t0 = time.time()
m = cs.CausalDiscovery(X, method="cluster_aware", rank=8, epochs=200, n_clusters=5, verbose=False)
m.fit(verbose=False)
net = m.get_network()
print(f"  Edges: {net.edge_count}, Time: {net.time_s:.1f}s", flush=True)
assert net.edge_count > 0, "No edges in cluster_aware mode"

# Test 4: auto
print("\n[4/4] auto mode...", flush=True)
t0 = time.time()
m = cs.CausalDiscovery(X, method="auto", rank="auto", epochs=200, verbose=False)
m.fit(verbose=False)
net = m.get_network()
print(f"  Edges: {net.edge_count}, Method: {net.metadata['method']}, Time: {net.time_s:.1f}s", flush=True)
assert net.edge_count > 0, "No edges in auto mode"

# Test summary + plot
print("\n" + m.summary())

print("\n" + "=" * 50)
print("ALL 4 MODES PASSED: causalscale V3.0.0")
print("=" * 50)
