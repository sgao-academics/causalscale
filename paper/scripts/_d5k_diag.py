"""Diagnose: why does d=5000 fail? Trace the actual code path."""
import sys, numpy as np, torch, time
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
import causalscale as cs

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEV}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# Generate synthetic data at d=5000
d, n = 5000, 200
print(f"\n=== Testing d={d}, n={n} ===")
np.random.seed(42)
X = np.random.randn(n, d).astype(np.float32)
# Add some structure
for j in range(100, d, 100):
    X[:, j] += 0.5 * X[:, j-100] + 0.2 * np.random.randn(n)

print(f"Data: {X.shape}, memory: {X.nbytes/1e6:.1f}MB")

# Test 1: Check what auto method selects
from causalscale.api import _auto_method
method = _auto_method(d, n)
print(f"  Auto method: {method}")
assert method == "lowrank", f"Expected lowrank, got {method}"

# Test 2: Direct LowRankGNN through causalscale API
print("\n--- Test: cs.CausalDiscovery(method='lowrank') ---")
t0 = time.time()
try:
    model = cs.CausalDiscovery(X, method="lowrank", rank=32, device=DEV,
                                epochs=100, verbose=False)
    model.fit(verbose=True)
    print(f"  DONE: {model._network.edge_count} edges, {time.time()-t0:.1f}s")
    print(f"  Method used: {model.method}")
except Exception as e:
    print(f"  FAILED: {e}")
    print(f"  Elapsed: {time.time()-t0:.1f}s")

# Test 3: Direct engine call (bypass API to see actual path)
print("\n--- Test: Direct CausalDiscoveryEngine ---")
from causalscale.core.engine import CausalDiscoveryEngine
t0 = time.time()
try:
    eng = CausalDiscoveryEngine(d=d, rank=32, mode="lowrank", device=DEV,
                                 epochs=100, verbose=True)
    result = eng.fit(X)
    print(f"  DONE: {result.edge_count} edges, adjacency shape: {result.adjacency.shape}")
    print(f"  Time: {time.time()-t0:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")

print("\n--- Test: Direct LowRankGNN module ---")
from causalscale.core.lowrank import LowRankGNN, train_lowrank_gnn
t0 = time.time()
try:
    result = train_lowrank_gnn(X, rank=32, epochs=100, device=DEV, verbose=True)
    print(f"  DONE: {result['gnn_edges']} edges, F1={result['f1']}, {time.time()-t0:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
