# Verification

One command, its expected output, and the exit code that separates a real pass
from a silent one. Written down because "there is a verification script" and "the
verification script says the package is sound" are different claims, and only the
second one is checkable by a reader.

---

## 1. The package

```bash
pip install numpy matplotlib
python run_all.py --verify
echo $?                      # 0 on success
```

Expected output, on a machine where the optional solvers happen to be installed:

```
causalscale replication package
author: Shuaidong Gao  (ORCID 0009-0004-5641-3581)
python: 3.14.3

========================================================================
MODE: --verify  (package integrity + the paper's claims)
========================================================================

[1] Package layout
  OK    experiments/protocol.py
  OK    experiments/run_sweep.py
  OK    experiments/records_to_tables.py
  OK    experiments/records_to_figures.py
  OK    experiments/audit_paper_numbers.py
  OK    PROTOCOL.md
  OK    REPRODUCTION.md

[2] Released per-seed records
    (coverage is reported, not judged: an incomplete set still reads)
  full    boundary_er.jsonl                  200 / 200 rows  (tab:main, fig:main a)
  full    attribution_factorial.jsonl        120 / 120 rows  (tab:factorial, fig:main c)
  full    penalty_sweep.jsonl                115 / 115 rows  (tab:penalty, fig:main b c)
  full    mechanism.jsonl                    180 / 180 rows  (fig:robustness a)
  full    topology.jsonl                      40 /  40 rows  (fig:robustness a (text))
  full    confounder.jsonl                   120 / 120 rows  (fig:robustness b)

[3] Optional solver dependencies (needed only to re-run, not to read)
  present  numpy
  present  matplotlib
  ...                 <- environment-dependent; "absent" here is not a failure

[4] Executing the paper's claims against the records
[PASS] the published NOTEARS schedule does not return an empty graph at d=150
        F1 = 0.628, edges = 111 (an empty graph would be F1 = 0, edges = 0)
...                   <- 28 checks, one block each, all [PASS]

28/28 checks passed

========================================================================
VERIFICATION COMPLETE -- all claims hold.
========================================================================
```

**What must be invariant across machines:** the `[1]` layout lines, the six
`full` record lines, the final `28/28 checks passed`, and `VERIFYING COMPLETE`
with exit code 0. Section `[3]` is informational by design — a missing `torch`
must not, and does not, change the verdict.

**What to do if a check fails.** Each `[PASS]`/`[FAIL]` block names the sentence
in the paper the predicate belongs to and prints the measured value beside it, so
a failure says which claim stopped holding rather than only that something
broke. `python run_all.py --audit` runs the same 28 predicates without the
package-integrity passes.

## 2. The toolkit's own promises

The other half of the package is the toolkit measured in §4.4, and it has its own
contract. This one is a different command because it needs `torch` to fit
anything:

```bash
python tests/test_api_contract.py        # or: python -m pytest tests/test_api_contract.py -q
```

Expected tail:

```
  PASS  run_notears fits X ~ X @ W.T, not X ~ X @ W  -- ||X - X W.T|| = 43.62 <= 46.16 = ||X - X W||
  PASS  the returned matrix has no self-loops  -- max |diag| = 0.000
  PASS  a non-zero W[i,j] is reported as the edge j -> i  -- got [('V0', 'V2', 0.7), ('V5', 'V0', -0.4)]
  ...
------------------------------------------------------------------------
OK -- every contract holds.
```

If `torch` is absent the script prints `SKIPPED` and exits 0: the contract is
about the engines, and there is nothing to assert without them.

## 3. The protocol's own properties

```bash
python -m pytest tests/ -q
```

or, without pytest:

```bash
python tests/test_protocol.py
```

These are the properties a reader is entitled to assume without re-running a
solver: the generators produce acyclic graphs, the metric scores direction,
self-loops are masked, and every released row corresponds to a cell some suite
declares.

---

## Summary

| Command | Needs | Expected |
|:--|:--|:--|
| `python run_all.py --verify` | numpy | `28/28 checks passed`, exit 0 |
| `python run_all.py --audit` | numpy | `28/28 checks passed`, exit 0 |
| `python run_all.py --list` | numpy | the six suites and their cell counts |
| `python tests/test_api_contract.py` | torch | `OK -- every contract holds`, exit 0 |
| `python tests/test_protocol.py` | numpy | all assertions hold, exit 0 |
| `python run_all.py --reproduce` | torch, dagma, GPU | re-runs every suite (hours) |
