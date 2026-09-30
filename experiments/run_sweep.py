# -*- coding: utf-8 -*-
"""Run one suite of the protocol and append per-seed records.

    python -m experiments.run_sweep --list
    python -m experiments.run_sweep --suite boundary
    python -m experiments.run_sweep --suite lamsweep --seeds 3

Every suite is resumable: a row is written and flushed after each cell, and the
(row keys) already present in the record file are skipped on the next launch.  A
run interrupted at any point therefore loses at most the cell in progress.

Nothing here is creative -- the solvers live in protocol.py and are the ones the
paper describes.  A suite is only a list of cells.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from experiments import protocol as P  # noqa: E402

RECORDS = os.path.join(HERE, "records")
os.makedirs(RECORDS, exist_ok=True)

KEY_FIELDS = ("topo", "d", "seed", "sem", "method", "q", "lam1", "rho0", "factor", "outer")


def _cells_boundary():
    for d in (30, 50, 80, 100, 150):
        for seed in range(10):
            for m in ("nt_official", "nt_tuned", "dagma", "cs_shipped"):
                yield dict(d=d, seed=seed, method=m)


def _cells_attribution():
    for d, seed, lam1, fac, out in itertools.product(
            (100, 150, 200), range(5), (0.01, 0.1), (2.0, 10.0), (10, 100)):
        yield dict(d=d, seed=seed, method="nt_official", q=0.0, **{
            "lam1": lam1, "rho0": 1.0, "factor": fac, "outer": out})


def _cells_lamsweep():
    for d in (100, 150, 200):
        for seed in range(5):
            for lam1 in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
                yield dict(d=d, seed=seed, method="nt_official",
                           lam1=lam1, rho0=1.0, factor=10.0, outer=100)
    # the configuration the submission under review describes for its baseline
    for seed in range(5):
        for lam1 in (0.1, 0.01):
            yield dict(d=150, seed=seed, method="nt_official",
                       lam1=lam1, rho0=0.01, factor=2.0, outer=10)
    # rho0 varied on its own
    for seed in range(5):
        for rho0 in (0.01, 0.1, 1.0):
            yield dict(d=150, seed=seed, method="nt_official",
                       lam1=0.1, rho0=rho0, factor=10.0, outer=100)


def _cells_topology():
    for topo in ("er", "ba"):
        for seed in range(5):
            for sem in ("linear-gaussian", "nonlinear-tanh"):
                for m in ("nt_official", "cs_shipped"):
                    yield dict(topo=topo, d=100, seed=seed, sem=sem, method=m)


def _cells_mechanism():
    for sem in ("linear-gaussian", "linear-exponential", "nonlinear-tanh"):
        for d in (30, 100):
            for seed in range(10):
                for m in ("nt_official", "dagma", "cs_shipped"):
                    yield dict(d=d, seed=seed, sem=sem, method=m)


def _cells_confounder():
    for q in (0.0, 0.1, 0.2):
        for d in (50, 100):
            for seed in range(10):
                for m in ("nt_official", "cs_shipped"):
                    yield dict(d=d, seed=seed, sem="linear-gaussian", q=q, method=m)


SUITES = {
    "boundary":    ("boundary_er.jsonl", _cells_boundary),
    "attribution": ("attribution_factorial.jsonl", _cells_attribution),
    "lamsweep":    ("penalty_sweep.jsonl", _cells_lamsweep),
    "topology":    ("topology.jsonl", _cells_topology),
    "mechanism":   ("mechanism.jsonl", _cells_mechanism),
    "confounder":  ("confounder.jsonl", _cells_confounder),
}


def key_of(row):
    return tuple(row.get(k) for k in KEY_FIELDS)


def load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    done.add(key_of(json.loads(line)))
                except Exception:
                    pass
    return done


def run_suite(name, seeds=None, dims=None, methods=None, dry=False):
    fname, gen = SUITES[name]
    path = os.path.join(RECORDS, fname)
    cells = list(gen())
    if seeds is not None:
        cells = [c for c in cells if c["seed"] < seeds]
    if dims is not None:
        cells = [c for c in cells if c["d"] in dims]
    if methods is not None:
        cells = [c for c in cells if c["method"] in methods]
    done = load_done(path)
    todo = [c for c in cells if key_of({**{"q": 0.0, "lam1": 0.1, "rho0": 1.0,
                                          "factor": 10.0, "outer": 100,
                                          "topo": "er", "sem": "linear-gaussian"},
                                       **c}) not in done]

    print(f"[{name}] {len(cells)} cells, {len(cells) - len(todo)} already done, "
          f"{len(todo)} to run -> {os.path.relpath(path, ROOT)}", flush=True)
    if dry:
        return 0

    t_start = time.time()
    for i, cell in enumerate(todo, 1):
        row = P.one_run(**cell)
        if cell.get("lam1") is not None or cell.get("rho0") is not None:
            for k in ("lam1", "rho0", "factor", "outer"):
                if cell.get(k) is not None:
                    row[k] = cell[k]
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        eta = ((time.time() - t_start) / i) * (len(todo) - i)
        if row.get("status") == "ok":
            extra = " ".join(f"{k}={cell[k]}" for k in ("lam1", "rho0", "factor", "outer")
                             if cell.get(k) is not None)
            print(f"  [{i}/{len(todo)}] d={row['d']:>4} s={row['seed']} {row['sem']:<17} "
                  f"{row['method']:<12} {extra:<34} F1={row['f1']:.3f} "
                  f"edges={row['edges']:<5} ({row['fit_time_s']}s, eta {eta/60:.1f}m)",
                  flush=True)
        else:
            print(f"  [{i}/{len(todo)}] d={cell.get('d')} s={cell.get('seed')} "
                  f"{cell.get('method')} ERROR {row.get('error', '?')[:90]}", flush=True)
    print(f"[{name}] done in {(time.time() - t_start) / 60:.1f} min", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Run one protocol suite.")
    ap.add_argument("--suite", choices=sorted(SUITES))
    ap.add_argument("--list", action="store_true", help="list suites and exit")
    ap.add_argument("--seeds", type=int, default=None, help="cap the seed index")
    ap.add_argument("--dims", default=None, help="comma-separated subset of d")
    ap.add_argument("--methods", default=None, help="comma-separated subset of methods")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.list or not args.suite:
        print("Available suites:")
        for k, (f, g) in SUITES.items():
            print(f"  {k:<12} -> records/{f:<32} ({len(list(g()))} cells)")
        return 0
    dims = [int(x) for x in args.dims.split(",")] if args.dims else None
    methods = args.methods.split(",") if args.methods else None
    return run_suite(args.suite, seeds=args.seeds, dims=dims, methods=methods,
                     dry=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
