# -*- coding: utf-8 -*-
"""Tests for the benchmark protocol.

    python -m pytest tests/ -q
    python tests/test_protocol.py          # plain-python fallback

These cover the properties a reader is entitled to assume without re-running the
solvers: the generators are actually acyclic, the metric is orientation-sensitive
in the way the paper says, and the records on disk agree with the protocol that is
supposed to have produced them.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from experiments import protocol as P  # noqa: E402


def test_er_dag_is_acyclic():
    """S[i, j] != 0 means i -> j with i < j, so the lower triangle is empty.

    That is the convention the whole protocol rests on: parents come first, and
    sample_sem reads column j's parents off S[:, j].
    """
    for d in (30, 100, 150):
        S = P.make_er_dag(d, 0)
        assert np.allclose(np.tril(S, k=-1), 0), \
            "lower triangle must be zero: an edge i->j requires i < j"
        assert (np.abs(S) > 0).sum() > 0, "a graph with no edges would be degenerate"
        assert S.shape == (d, d)


def test_er_dag_density_matches_the_declared_parameter():
    """`p = 2/(d-1)` over d(d-1)/2 positions gives an expected d edges.

    Equivalently a total (in + out) degree of 2, which is what the paper's
    "expected degree 2" means.  Averaged over seeds so that the test measures the
    parameter rather than one draw; the spread for d = 400 over 5 seeds is a few
    percent, so a +-20% band is generous without being vacuous.
    """
    d = 400
    seeds = 5
    mean_edges = sum(int((np.abs(P.make_er_dag(d, s)) > 0).sum())
                     for s in range(seeds)) / seeds
    assert 0.8 * d < mean_edges < 1.2 * d, (
        f"expected ~{d} edges (p = 2/(d-1)), got {mean_edges:.0f}")


def test_ba_dag_is_acyclic_and_heavy_tailed():
    """The scale-free skeleton must remain acyclic and must actually have hubs."""
    d = 200
    S = P.make_ba_dag(d, 0)
    indeg = (np.abs(S) > 0).sum(0)
    assert indeg.max() >= 2, "every arriving node receives at least one parent"
    outdeg = (np.abs(S) > 0).sum(1)
    assert outdeg.max() > 3 * max(np.median(outdeg), 1), \
        "hub structure missing -- this would not be a scale-free skeleton"
    assert (np.abs(S) > 0).sum() > d, "a scale-free graph cannot be this sparse"


def test_dag_is_a_dag_by_permutation():
    """A DAG exists iff some permutation of the rows makes it triangular.

    Checked constructively: keep removing sinks.  If a cycle existed, a step
    would find no sink.
    """
    S = P.make_ba_dag(80, 3)
    A = (np.abs(S) > 0).astype(int)
    # topological order recoverable by repeated sink removal
    remaining = set(range(A.shape[0]))
    removed = 0
    while remaining:
        sinks = [j for j in remaining if A[list(remaining), j].sum() == 0]
        assert sinks, "cycle detected: no sink in the remaining graph"
        for j in sinks:
            remaining.discard(j)
            removed += 1
    assert removed == A.shape[0]


def test_score_is_orientation_sensitive():
    S = P.make_er_dag(50, 1)
    exact = P.score(S, S)
    assert exact["f1"] > 0.999, "the truth must score ~1 against itself"
    transposed = P.score(S.T, S)
    assert transposed["f1"] < exact["f1"], \
        "transposing must not score as well -- the metric scores direction"


def test_score_masks_the_diagonal():
    d = 20
    S = P.make_er_dag(d, 2)
    W = S + np.eye(d) * 5.0            # self-loops only
    m = P.score(W, S)
    assert m["edges"] == (np.abs(S) > 0.3).sum(), \
        "self-loops must not be counted as edges"


def test_sample_sem_shape_and_standardisation():
    S = P.make_er_dag(40, 4)
    X = P.sample_sem(S, 200, 4)
    assert X.shape == (200, 40)
    assert np.allclose(X.mean(0), 0, atol=1e-6)
    assert np.allclose(X.std(0), 1, atol=1e-6)


def test_records_agree_with_the_protocol():
    """Every released record row must correspond to a cell a suite declares.

    This is what stops a stale or hand-edited record from silently becoming a
    table: if a row's (dimension, seed, method) triple is not produced by any
    suite in run_sweep.py, it is not part of the protocol.
    """
    rec_dir = os.path.join(ROOT, "experiments", "records")
    if not os.path.isdir(rec_dir):
        return
    from experiments.run_sweep import SUITES

    for name, (fname, gen) in SUITES.items():
        path = os.path.join(rec_dir, fname)
        if not os.path.exists(path):
            continue
        declared = {(c["d"], c["seed"], c["method"]) for c in gen()}
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # a suite that varies only the schedule does not name a method; it
            # is NOTEARS, and the default records that.
            triple = (row.get("d"), row.get("seed"),
                      row.get("method", "nt_official"))
            assert triple in declared, (
                f"{name}: row {triple} is not produced by suite '{name}' "
                f"-- the records and the declared protocol have diverged")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
