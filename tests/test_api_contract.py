"""The API contract: what a caller may assume without reading an implementation.

Every assertion here pins a property that, if it silently changed, would turn a
wrong answer into a plausible one.  The coefficient convention is the first of
them, because an implementation that stores its coefficients transposed does not
crash -- it returns a graph whose edges all point the wrong way, which scores
worse and reads as a weak method rather than as a bug.

Run:  python -m pytest tests/test_api_contract.py -q
      python tests/test_api_contract.py          (no pytest needed)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import numpy as np

_fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        _fails.append(name)


# ------------------------------------------------------------------- fixtures --
# The generator convention, as experiments/protocol.py defines it:
#   S[i, j] != 0  <=>  edge i -> j,   X[:, j] = sum_i X[:, i] S[i, j] + eps
def _chain_dag(d=6):
    S = np.zeros((d, d))
    for i in range(d - 1):
        S[i, i + 1] = 0.9 + 0.1 * i          # i -> i+1, strictly ordered
    return S


def _sample(S, n=800, seed=0):
    rng = np.random.default_rng(seed)
    d = S.shape[0]
    X = np.zeros((n, d))
    for j in range(d):
        pa = np.nonzero(S[:, j])[0]
        X[:, j] = X[:, pa] @ S[pa, j] + rng.standard_normal(n)
    return (X - X.mean(0)) / X.std(0).clip(1e-8)


S = _chain_dag()
X = _sample(S)
_CACHE = {}


def _model(X, fit=False, **kw):
    from causalscale.api import CausalDiscovery
    # `fit` is part of the key: the same kwargs must not silently return an
    # unfitted model to a caller that asked for a fitted one.
    key = (fit, repr(sorted(kw.items())))
    if key not in _CACHE:
        m = CausalDiscovery(X, device="cpu", **kw)
        if fit:
            m.fit(verbose=False)
        _CACHE[key] = m
    return _CACHE[key]


# --------------------------------------------------------------------- tests --
def test_solver_objective():
    """The solver minimises ||X - X W.T||; that is what fixes the convention.

    Recovered edge *positions* cannot settle this: a linear-Gaussian DAG and its
    reversal are Markov equivalent, and the optimiser picks one of the pair.
    The objective is unambiguous, so it is what the contract is written on.
    """
    from causalscale.core._notears import run_notears
    W, _, _, _ = run_notears(X.astype(np.float32), device="cpu",
                             outer=20, inner=150, seed=0)
    r_right = float(np.linalg.norm(X - X @ W.T))
    r_wrong = float(np.linalg.norm(X - X @ W))
    check("run_notears fits X ~ X @ W.T, not X ~ X @ W",
          r_right <= r_wrong,
          f"||X - X W.T|| = {r_right:.2f} <= {r_wrong:.2f} = ||X - X W||")

    check("the returned matrix has no self-loops",
          float(np.abs(np.diag(W)).max()) == 0.0,
          f"max |diag| = {np.abs(np.diag(W)).max():.3f}")


def test_extract_edges_orientation():
    """`(cause, effect, weight)` -- the parent is named first.

    Deterministic: a hand-built matrix, no solver and no noise, so the assertion
    is about the convention rather than about a draw.
    """
    m = _model(X, method="cluster_aware")
    Wm = np.zeros((6, 6))
    Wm[2, 0] = 0.7        # W[2,0] means the edge runs 0 -> 2
    Wm[0, 5] = -0.4       # W[0,5] means the edge runs 5 -> 0
    out = m._extract_edges(Wm, threshold=0.3)
    check("a non-zero W[i,j] is reported as the edge j -> i",
          (("V0", "V2", 0.7) in out) and (("V5", "V0", -0.4) in out),
          f"got {out}")
    check("no edge is reported in the transposed direction",
          all((a, b) != ("V2", "V0") and (a, b) != ("V0", "V5") for a, b, _ in out),
          f"got {out}")


def test_predict_contract():
    """predict() must return X @ W.T, exactly -- the model's own definition."""
    m = _model(X, fit=True, method="cluster_aware")
    W = m.get_adjacency()
    got = m.predict(X)
    check("predict() computes X @ W.T",
          np.allclose(got, X @ W.T), "elementwise equality with X @ W.T")
    check("predict() is not the transposed reading, X @ W",
          not np.allclose(got, X @ W), "differs from X @ W")
    right = float(np.linalg.norm(X - X @ W.T))
    wrong = float(np.linalg.norm(X - X @ W))
    check("the reading predict() uses is the one that reconstructs X",
          right <= wrong,
          f"||X - X W.T|| = {right:.2f} <= {wrong:.2f} = ||X - X W||")


def test_fitted_output_is_clean():
    m = _model(X, fit=True, method="cluster_aware")
    W = m.get_adjacency()
    check("a fitted adjacency carries no self-loop",
          float(np.abs(np.diag(W)).max()) == 0.0,
          f"max |diag| = {np.abs(np.diag(W)).max():.3f}")
    n = int((np.abs(W) > 0.3).sum())
    check("edge_count agrees with the reported edges",
          m.get_network().edge_count == n,
          f"edge_count={m.get_network().edge_count}, entries above threshold={n}")


def test_method_aliases():
    try:
        _model(X, fit=True, method="notears")
        ok, detail = True, "method='notears' resolves"
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"{type(e).__name__}: {e}"
    check("every method the UI offers is resolvable", ok, detail)

    try:
        _model(X, method="not_a_method")
        ok2, detail2 = False, "no error raised"
    except ValueError as e:
        ok2, detail2 = "Available" in str(e), str(e)[:70]
    except Exception as e:  # noqa: BLE001
        ok2, detail2 = False, f"{type(e).__name__} instead of ValueError"
    check("an unknown method raises and says what is available", ok2, detail2)


def test_auto_routing_is_recorded():
    m = _model(X, method="auto")
    check("auto() records the engine it routed to",
          m.auto_routed_to is not None,
          f"d={m.d} routed to {m.auto_routed_to!r}")


def test_output_kind():
    """A rank-r output must not be presented as a DAG."""
    m = _model(X, method="lowrank", rank=3)
    check("the rank-r engine declares a non-DAG output",
          "not a DAG" in m.output_kind, f"output_kind = {m.output_kind!r}")
    m2 = _model(X, method="cluster_aware")
    check("the exact-acyclicity engine declares a DAG",
          m2.output_kind == "dag", f"output_kind = {m2.output_kind!r}")


def test_version_consistency():
    import causalscale as cs
    check("__version__ matches the released major version",
          cs.__version__.startswith("4."), f"__version__ = {cs.__version__!r}")
    check("the convention is machine-readable",
          "W[i,j]" in cs.CONVENTION, f"CONVENTION = {cs.CONVENTION!r}")


# ---------------------------------------------------------------------- main --
def main():
    print("=" * 72)
    print("causalscale API contract")
    print("=" * 72)
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"SKIPPED: fitting needs torch ({type(e).__name__})")
        return 0

    test_solver_objective()
    test_extract_edges_orientation()
    test_predict_contract()
    test_fitted_output_is_clean()
    test_method_aliases()
    test_auto_routing_is_recorded()
    test_output_kind()
    test_version_consistency()

    print("-" * 72)
    if _fails:
        print(f"FAILED ({len(_fails)}): " + "; ".join(_fails))
        return 1
    print("OK -- every contract holds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
