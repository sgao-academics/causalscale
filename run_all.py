#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""causalscale replication package -- one entry point.

    python run_all.py --verify      check the package and execute the paper's claims
    python run_all.py --tables      rebuild every table from the released records
    python run_all.py --figures     rebuild every figure from the released records
    python run_all.py --audit       run only the claim-by-claim assertions
    python run_all.py --list        list the experiment suites
    python run_all.py --suite NAME  re-run one suite (needs torch / dagma / causalscale)
    python run_all.py --reproduce   re-run every suite (hours, GPU recommended)

Design note
-----------
``--verify``, ``--tables``, ``--figures`` and ``--audit`` read *only* the released
per-seed records under ``experiments/records/``.  They import numpy at most, never
torch, never an external solver, and never this package's own engines.  That is
deliberate: the inexpensive checks must not be able to fail because of a missing
optional dependency, a GPU that is not there, or a solver that changed its API.

Only ``--suite`` and ``--reproduce`` touch a solver.  They are the expensive path
and are never required to read the results.
"""
from __future__ import annotations

import os
import subprocess
import sys

PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PKG_ROOT)

SEP = "=" * 72
RECORDS_DIR = os.path.join(PKG_ROOT, "experiments", "records")

# (file, cells the suite declares, what it feeds)
EXPECTED_RECORDS = [
    ("boundary_er.jsonl", 200, "tab:main, fig:main a"),
    ("attribution_factorial.jsonl", 120, "tab:factorial, fig:main c"),
    ("penalty_sweep.jsonl", 115, "tab:penalty, fig:main b c"),
    ("mechanism.jsonl", 180, "fig:robustness a"),
    ("topology.jsonl", 40, "fig:robustness a (text)"),
    ("confounder.jsonl", 120, "fig:robustness b"),
]


def _py():
    return sys.executable


def _run_module(mod, *args):
    cmd = [_py(), "-m", mod] + list(args)
    return subprocess.call(cmd, cwd=PKG_ROOT)


def run_verify():
    print(SEP)
    print("MODE: --verify  (package integrity + the paper's claims)")
    print(SEP)
    ok = True

    print("\n[1] Package layout")
    for rel in ("experiments/protocol.py", "experiments/run_sweep.py",
                "experiments/records_to_tables.py", "experiments/records_to_figures.py",
                "experiments/audit_paper_numbers.py", "PROTOCOL.md", "REPRODUCTION.md"):
        path = os.path.join(PKG_ROOT, rel)
        present = os.path.isfile(path)
        ok &= present
        print(f"  {'OK   ' if present else 'MISS '} {rel}")

    print("\n[2] Released per-seed records")
    print("    (coverage is reported, not judged: an incomplete set still reads)")
    for name, cells, used_by in EXPECTED_RECORDS:
        path = os.path.join(RECORDS_DIR, name)
        if not os.path.isfile(path):
            print(f"  absent  {name:<32} ({used_by})")
            continue
        n = sum(1 for line in open(path, encoding="utf-8") if line.strip())
        if n == 0:
            print(f"  EMPTY   {name:<32} ({used_by})")
            ok = False
            continue
        mark = "full   " if n >= cells else f"{n / cells:4.0%}  "
        print(f"  {mark} {name:<32} {n:>5} / {cells} rows  ({used_by})")

    print("\n[3] Optional solver dependencies (needed only to re-run, not to read)")
    for mod in ("numpy", "matplotlib", "torch", "dagma", "causalscale"):
        try:
            __import__(mod)
            print(f"  present  {mod}")
        except Exception as e:  # noqa: BLE001
            print(f"  absent   {mod}   ({type(e).__name__}) -- fine unless you re-run")

    print("\n[4] Executing the paper's claims against the records")
    rc = _run_module("experiments.audit_paper_numbers")
    ok &= (rc == 0)

    print(f"\n{SEP}")
    print("VERIFICATION " + ("COMPLETE -- all claims hold." if ok
                             else "FINISHED WITH PROBLEMS (see above)."))
    print(SEP)
    return 0 if ok else 1


def main():
    argv = set(sys.argv[1:])
    print("causalscale replication package")
    print("author: Shuaidong Gao  (ORCID 0009-0004-5641-3581)")
    print(f"python: {sys.version.split()[0]}\n")

    if "--verify" in argv:
        return run_verify()
    if "--tables" in argv:
        return _run_module("experiments.records_to_tables")
    if "--figures" in argv:
        # No --only: the paper prints three figures and all three are rebuilt, so
        # the package can never ship a figure the paper no longer shows, or show
        # one the package cannot rebuild.
        return _run_module("experiments.records_to_figures",
                           "--out", os.path.join(PKG_ROOT, "figures"))
    if "--audit" in argv:
        return _run_module("experiments.audit_paper_numbers")
    if "--list" in argv:
        return _run_module("experiments.run_sweep", "--list")

    if "--suite" in argv:
        i = sys.argv.index("--suite")
        name = sys.argv[i + 1]
        return _run_module("experiments.run_sweep", "--suite", name,
                           *[a for a in sys.argv[i + 2:]])

    if "--reproduce" in argv:
        print(SEP)
        print("MODE: --reproduce  (re-runs every suite; hours on a GPU)")
        print(SEP)
        rc = 0
        for suite in ("boundary", "attribution", "lamsweep", "mechanism",
                      "topology", "confounder"):
            print(f"\n---- suite: {suite} ----")
            rc |= _run_module("experiments.run_sweep", "--suite", suite)
        print("\nRe-run complete. Rebuild the paper with:")
        print("  python run_all.py --tables")
        print("  python run_all.py --figures")
        return rc

    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
