# Conventions

One page, because the arithmetic in this package is only readable if four
conventions are fixed, and three of the four are conventions we had to *choose*
rather than inherit.

---

## 1. Which way an edge points

Two conventions are in use in this package, and they are transposes of each
other. Both are stated in code, both are asserted in tests, and neither is left
to the reader.

**Generator / protocol convention** (`experiments/protocol.py`):

```
S[i, j] != 0   <=>   edge i -> j
X[:, j] = sum_i X[:, i] * S[i, j] + eps
```

So in the protocol, `S[i, j]` is named by the **parent** first, and data are
generated so that column `j` is a function of its parents.

**Engine convention** (every matrix `causalscale` returns):

```
X ~= X @ W.T
X[:, i] ~= sum_j X[:, j] * W[i, j]
```

So `W[i, j]` is the coefficient of variable `j` in the equation for variable
`i`, meaning `W[i, j] != 0` **is the edge `j -> i`**: in the engine's matrix the
entry is named by the **child** first. Equivalently, `W = S.T`.

Three consequences that are easy to get wrong and are therefore stated here:

* `causalscale.CONVENTION` is the machine-readable form of the second sentence,
  so a caller can assert on it instead of remembering it.
* The residual the solver actually minimises is `X - X @ W.T`, not `X - X @ W`.
  `tests/test_api_contract.py::test_solver_objective` pins this, and
  `test_predict_contract` requires `predict(X) == X @ W.T` exactly.
* `get_network().edges` and `get_edges()` report `(cause, effect, weight)`, i.e.
  the parent first -- the *generator* order, applied to the engine's matrix.

**Why this matters, and why it is written down.** An implementation that stores
its coefficients transposed does not crash. It returns a graph whose edges all
point the wrong way, which scores worse and reads as a weak method rather than
as a bug. Recovered edge *positions* cannot settle the question either: a
linear-Gaussian DAG and its reversal are Markov equivalent, so the optimiser may
return either. The objective is the only unambiguous evidence, which is why the
test is written on the objective.

**A known naming inconsistency, not hidden.** `pan_cancer_scan.py` reads
`adj[ARID1A_idx, MTOR_idx]` and writes it under the key `arid1a_to_mtor` -- named
by lookup order, not by edge direction, so under the convention above the two
keys are the transpose of the edge directions. The shipped
`results/pan_cancer_ckpt.json` was produced by that code and is released
unchanged; the paper therefore reports the case study as "eight cohorts one way,
thirteen the other" rather than committing the direction. The one-line fix is
deferred to the next release so that the released numbers and the released code
stay in step.

---

## 2. How a score is computed

```
F1(|W| > tau,  |S| > 0),  diagonal masked, full d x d matrix
```

No triangular shortcut. This is deliberate: with the full matrix, an
implementation that stores its coefficients transposed scores *worse*, not
*differently*. `tau = 0.3` throughout, unless a table says otherwise; the
`tau`-free `oracle` F1 (best over a threshold sweep) is recorded alongside as a
diagnostic and is never used as a headline number.

`f1_flipped` -- the score of `W.T` -- is recorded for every cell. It is the
control for convention errors: for a genuine recovery the forward score should
exceed the flipped one, and the records let a reader check that rather than trust
it.

## 3. What the numbers are

Every released row is one `(method, dimension, seed, condition)` cell. Nothing is
averaged before it is written. Means and standard deviations in the tables are
computed by `experiments/records_to_tables.py` from those rows, so a table can
always be traced back to the cells behind it.

* Tuning budget: `lambda_1` is chosen on held-out DAGs (seeds 1000--1002), never
  on the test DAG, and every method receives the same budget. The selected value
  is cached per `(method, d)` and is deterministic.
* Threshold: `tau = 0.3` is applied to `|W|`; a row also carries the number of
  entries above it, so a reported F1 can always be paired with the graph size
  that produced it.
* Seeds: the seed is in every row. Re-running a suite with a different seed set
  moves the numbers slightly; the audit asserts the signs and the orderings, not
  the decimals.

## 4. What each output *is*

Not every engine returns a DAG, and the package says which is which rather than
leaving it to the method name.

| Method | Output | Read it as |
|:--|:--|:--|
| `dagma`, `cluster_aware`, `transformer` | a DAG under an exact acyclicity constraint | directed edges |
| `lowrank`, `multi_scale`, `full` | a rank-`r` co-variation network | associations; direction is **not** identified |

`CausalDiscovery.output_kind` returns the string, and every network carries it in
`metadata["output_kind"]`. The rank-`r` engines optimise a
correlation-reconstruction objective, not `h(W) = 0`; calling their output a
causal DAG is the substantive error the accompanying paper reports against this
package in Section 4.4.

## 5. What is *not* claimed

The engine descriptions in `docs/ENGINES_LEGACY.md` are carried over from the
earlier release. Their numbers are self-reported, were not independently re-run
for this package, and are marked **unreplicated**. The paper measures the
toolkit as shipped; it does not endorse it.
