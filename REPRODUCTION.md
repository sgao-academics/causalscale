# Reproduction, step by step

Two ways to use this package. The first takes seconds and needs no GPU. The second
re-runs the experiments and takes hours.

---

## A. Rebuild the paper from the released records (no GPU, no solver)

```bash
pip install numpy matplotlib

python run_all.py --verify
```

`--verify` performs four passes: it checks the package layout, counts the rows in
each record file, lists which optional dependencies are installed, and then
executes the paper's claims. Its [3] section is purely informational — a missing
`torch` is not a failure.

```bash
python run_all.py --tables
```

Prints every regenerated table to stdout in the same form it appears in the paper,
each rebuilt from the per-seed records, keyed by LaTeX label (`tab:main`,
`tab:factorial`, `tab:penalty`) so an edit to the paper's layout cannot silently
change which table a number belongs to. Add `--tex DIR` to also write the LaTeX:

```bash
python -m experiments.records_to_tables --tex tables/
```

```bash
python run_all.py --figures
```

Rebuilds all three printed figures (each `.pdf` and `.png`):

* `figures/fig_rev_main` -- **a**, structure F1 against dimension for the
  published NOTEARS schedule, the official DAGMA baseline and the shipped engine;
  **b**, the penalty curve, F1 and returned-edge count against `lambda_1`;
  **c**, per-seed score against the final value of the acyclicity term `h(W)`.
* `figures/fig_robustness` -- **a**, the head-to-head per mechanism family;
  **b**, the F1 gap against confounder strength.
* `figures/fig_cohort` -- the ARID1A--MTOR signed weight and sample size across
  the 33 cohorts.

The superseded figures of the earlier submission are in `figures/legacy/`; the
generators that produced them are in `legacy/gen_figures/` rather than
`scripts/`, so that nothing in `scripts/` looks like a live path to a figure the
paper no longer prints.

```bash
python run_all.py --audit
```

This is the strongest check in the package. Each claim the paper makes about a
number is written as a predicate over the records, so the command reports not just
whether the numbers can be regenerated but whether the *sentences* they support are
still true. Expected output ends with `N/N checks passed`.

Unit tests for the protocol itself:

```bash
python -m pytest tests/ -q      # or: python tests/test_protocol.py
```

They cover the properties a reader is entitled to assume without re-running a
solver: the generators are acyclic, the metric scores direction, self-loops are
masked, and every released row corresponds to a cell some suite declares.

---

## B. Re-run the experiments

```bash
pip install -r requirements.txt

python run_all.py --list
```

```
boundary     -> records/boundary_er.jsonl           200 cells
attribution  -> records/attribution_factorial.jsonl 120 cells
lamsweep     -> records/penalty_sweep.jsonl         115 cells
topology     -> records/topology.jsonl               40 cells
mechanism    -> records/mechanism.jsonl             180 cells
confounder   -> records/confounder.jsonl            120 cells
```

Each suite is **resumable and restartable**: a row is written and flushed after
every cell, and cells already present in the record file are skipped on the next
launch. A run interrupted at any point loses at most the cell in progress — which
matters, because the full set is several hours on one GPU.

```bash
python run_all.py --suite boundary                 # one suite
python run_all.py --suite lamsweep --seeds 3       # a quick subset
python run_all.py --reproduce                      # everything
```

When a suite finishes, rebuild the artefacts:

```bash
python run_all.py --tables
python run_all.py --figures
python run_all.py --audit
```

Timing, on the hardware described in `PROTOCOL.md §6`: `boundary` is the slowest
per cell because `cs_shipped` runs a full augmented-Lagrangian solve on the GPU;
`dagma` is the fastest. `attribution` and `lamsweep` skip the toolkit and finish
much sooner. The suites are independent, so they can be run in any order or in
parallel on separate devices by pointing `CUDA_VISIBLE_DEVICES` at each.

---

## Expected values

The audit prints the measured value next to every claim, so this table is a
convenience rather than the contract; the contract is the audit itself.

| Claim | Expected |
|:--|:--|
| NOTEARS (published cfg) at `d = 150` | F1 well above zero, graph not empty |
| NOTEARS vs shipped toolkit, every `d` | NOTEARS higher |
| NOTEARS vs DAGMA, every `d` | NOTEARS higher |
| main effect of λ₁ in the factorial | more than an order of magnitude larger than the other two |
| main effects of growth factor and iteration budget | below seed-to-seed spread |
| the penalty sweep | F1 low at both ends, high in the middle |

Numbers will shift slightly if the suites are re-run with a different seed set.
What should not shift is the sign of every comparison and the size ordering of the
main effects — that is what the audit asserts.
