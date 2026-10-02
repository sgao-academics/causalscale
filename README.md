---
language: en
license: mit
library_name: causalscale
tags:
- causal-discovery
- benchmark
- reproducibility
- structure-learning
- dag
- pytorch
---

# causalscale 4.0.0 — replication package

For the KDD 2027 Datasets & Benchmarks submission

> **Protocol, Not Physics: Re-measuring the Scalability Limit of Differentiable
> Causal Discovery**

Author: Shuaidong Gao (ORCID [0009-0004-5641-3581](https://orcid.org/0009-0004-5641-3581))

This is the **revision**. It replaces the earlier release, in which the paper was
organised around a twelve-engine toolkit. That framing is gone: the paper now asks
a single question — *is the widely reported `d ≈ 150` scalability limit a property
of the methods or of the protocol used to measure them?* — and the toolkit is one
of the objects being measured, not the contribution.

The version number is 4.0 rather than 3.5 because the defects fixed here are the
kind that made the previous release untrustworthy, not additions to it. They are
listed under **What changed in 4.0** below, each with the check that now pins it.

---

## Quick start

The three checks below need **numpy and matplotlib only**. They do not import
PyTorch, do not import this package's engines, and do not touch a GPU. They read
the released per-seed records and rebuild everything the paper reports.

```bash
pip install numpy matplotlib

python run_all.py --verify      # package integrity + execute the paper's claims
python run_all.py --tables      # every table, rebuilt from the records
python run_all.py --figures     # all three printed figures, from the records
python run_all.py --audit       # only the claim-by-claim assertions
```

`--verify` is designed so that it cannot fail for an environmental reason: if the
records are present, it reports on the science; optional dependencies are listed
but never required. `VERIFY.md` gives the exact expected output, so a reader can
tell a real pass from a silent one.

Re-running the experiments is a separate, expensive path:

```bash
pip install -e ".[solvers]"              # torch, dagma, scipy, scikit-learn
python run_all.py --list                 # list the six suites
python run_all.py --suite boundary       # one suite, resumable
python run_all.py --reproduce            # all suites (hours; GPU recommended)
```

---

## What changed in 4.0

Five defects, each of which could produce a wrong answer that looked like a
plausible one. All five were found by reading the implementation against its own
objective, and each is now covered by an assertion.

| # | Defect in the earlier release | Fixed by | Pinned by |
|:--|:--|:--|:--|
| 1 | `get_network().edges` named `W[i, j]` as the edge `i -> j`, while the solver minimises `X - X @ W.T` — so `W[i, j]` is the edge `j -> i` and **every reported edge was reversed** | `api.CausalDiscovery._extract_edges` reports the parent first | `tests/test_api_contract.py::test_extract_edges_orientation` |
| 2 | `predict()` returned `X @ W`, propagating the graph the wrong way | `predict()` returns `X @ W.T` | `tests/...::test_predict_contract` |
| 3 | The returned adjacency carried a self-coefficient of up to 0.8 on the diagonal, and the reported edge count included it | the diagonal is zeroed at the boundary and the count is taken over off-diagonal entries | `tests/...::test_fitted_output_is_clean` |
| 4 | `method="notears"` — offered in the web UI — raised `ValueError` | registered as an alias; an unknown method now lists what is available | `tests/...::test_method_aliases` |
| 5 | `method="auto"` silently routed to DAGMA at `d <= 150`, so "causalscale" numbers at low dimension were DAGMA numbers | the routing decision is kept in `metadata["auto_routed_to"]` and documented | `tests/...::test_auto_routing_is_recorded` |

Two further changes are about honesty rather than arithmetic, and they are the
reason the package now states what it is producing:

* **A rank-`r` output is not called a DAG.** `output_kind` returns
  `"co-variation network (rank-r; not a DAG)"` for `lowrank`, `multi_scale` and
  `full`. Those engines optimise a correlation-reconstruction objective, not
  `h(W) = 0`; reading their output as a directed causal graph is the substantive
  error the paper reports against this package in §4.4.
* **The claims of the earlier README are gone.** Every number that release
  advertised and this one cannot reproduce has been removed rather than
  reworded. `docs/ENGINES_LEGACY.md` keeps the old engine descriptions, marked
  **unreplicated**.

Two conventions are now written down instead of assumed — the coefficient
orientation above, and what each output is. `CONVENTIONS.md` is one page and is
the thing to read before reading any matrix this package returns.

---

## API surface, read rather than asserted

An earlier description of this package named five entry points. Two of those
names exist; three do not, and the real names are different. They are listed here
rather than described, so the mismatch cannot recur silently:

| Named in the earlier description | Actual entry point |
|:--|:--|
| `causalscale.list_models()` | `causalscale.pretrained.list_models()` |
| `causalscale.two_stage_discovery()` | `causalscale.core.two_stage.two_stage_discovery()` |
| `ascend_discover()` | `causalscale.core.ascend.two_tier_discovery()` |
| `stability_select()` | `causalscale.core.uncertainty.StabilitySelector` |
| `run_pcmci()` | `causalscale.core.pcmci_engine.PCMCIEngine` |

---

## Paper → package cross-reference

Every number in the paper is rebuilt from `experiments/records/*.jsonl`. The
records are one JSON object per `(method, dimension, seed, condition)`; nothing is
aggregated before it is written.

Tables are identified by their LaTeX label first, because the printed number
depends on how the paper is assembled: the protocol card is `tab:card`, so
`tab:main` prints as Table 2, not Table 1.

| LaTeX label | Printed as | What it shows | Rebuilt by | Records |
|:--|:--|:--|:--|:--|
| `tab:main` | Table 2 | structure F1 vs dimension, four configurations, one protocol | `records_to_tables.table_main` | `boundary_er.jsonl` |
| `tab:factorial` | Table 3 | the 2×2×2 schedule factorial, as main effects | `records_to_tables.table_factorial` | `attribution_factorial.jsonl` |
| `tab:penalty` | Table 4 | the penalty sweep: F1 and edge count vs `lambda_1` | `records_to_tables.table_penalty` | `penalty_sweep.jsonl` |
| `fig:main` a/b/c | Figure 1 | the boundary, the penalty curve, score vs `h(W)` | `records_to_figures` | `boundary_er.jsonl`, `penalty_sweep.jsonl`, `attribution_factorial.jsonl` |
| `fig:robustness` a/b | Figure 2 | mechanism families; gap vs confounder strength | `records_to_figures` | `mechanism.jsonl`, `confounder.jsonl` |
| `fig:cohort` | Figure 3 | the ARID1A–MTOR sign across 33 cohorts | `records_to_figures` | `results/pan_cancer_ckpt.json` |
| §3 | — | the protocol itself | `experiments/protocol.py` | — |
| §5 | — | identifiability boundary and matched null | see `PROTOCOL.md §5` | `results/exp15_string_mapped_f1.json` |

The claims themselves are executable:

```bash
python run_all.py --audit
```

This does not merely re-derive the numbers; it states them as predicates over the
records and fails if a sentence the paper relies on stops holding. A
reproduction script shows that the numbers can be regenerated. An audit shows
what would be false if they were not.

---

## Layout

| Path | Contents |
|:--|:--|
| `experiments/protocol.py` | The measurement: generators, scoring, solvers, tuning budget. The single source of truth. |
| `experiments/run_sweep.py` | The six suites, each a list of cells. Resumable, one row per cell. |
| `experiments/records/` | Released per-seed records, one JSON row per cell. |
| `experiments/records_to_tables.py` | Records → LaTeX and console tables. |
| `experiments/records_to_figures.py` | Records → the three printed figures. |
| `experiments/audit_paper_numbers.py` | The paper's claims, as assertions. |
| `tests/` | `test_protocol.py` (the measurement) and `test_api_contract.py` (the toolkit's promises). |
| `causalscale/` | The toolkit measured in §4.4, kept for inspection. |
| `results/` | Result files from the earlier submission, plus the four added with the revision: `exp8_notears_vs_cs.json` and its recomputed `exp8_paired_summary.json` (the earlier submission's own paired run, six dimensions x five paired seeds), `bio_baseline.json` (the genome-scale matched null and same-edge-count correlation baseline), and `paired_stats_single_protocol.json` (the 10-seed single-protocol paired statistics behind the revised Table 2). |
| `figures/` | The three printed figures. `figures/legacy/` holds the superseded set. |
| `legacy/gen_figures/` | The generators for the superseded figures. Not a live path. |
| `docs/ENGINES_LEGACY.md` | Engine documentation carried over, marked unreplicated. |
| `CONVENTIONS.md` | Coefficient orientation, scoring, and what each output is. |
| `VERIFY.md` | The expected output of `run_all.py --verify`. |

---

## What is and is not claimed

* **Claimed.** Under one fixed protocol, a plain NOTEARS solver does not return an
  empty graph at `d = 150`; the collapse that is normally reported is dominated by
  the sparsity penalty, a scalar that was never swept, and not by the two causes
  usually blamed (an exhausted iteration budget, an unsatisfied constraint).
* **Not claimed.** That NOTEARS scales to genome-wide resolution, or that this
  package establishes a new state of the art. The obstruction past `d ≈ 500` is
  stated as identifiability, not compute.
* **Not re-measured.** The other engines in `docs/ENGINES_LEGACY.md` are carried
  over with their original, self-reported numbers. They were not independently
  re-run and should not be read as evidence.

All experiments were run by one author on a single 8 GB GPU. Timings and
out-of-memory thresholds are hardware-specific; the F1 comparisons are not, and
every record states the seed that produced it.

---

## Citation

```bibtex
@inproceedings{gao2027causalscale,
  title={Protocol, Not Physics: Re-measuring the Scalability Limit of
         Differentiable Causal Discovery},
  author={Gao, Shuaidong},
  booktitle={Proceedings of the 33rd ACM SIGKDD Conference on Knowledge
             Discovery and Data Mining (KDD), Datasets and Benchmarks Track},
  year={2027}
}
```

## License

MIT. See `LICENSE`.
