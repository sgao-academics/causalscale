# -*- coding: utf-8 -*-
"""Rebuild every table in the paper from the released per-seed records.

    python -m experiments.records_to_tables            # print to stdout
    python -m experiments.records_to_tables --tex DIR  # also write LaTeX

Requires numpy only.  No solver is run and no GPU is touched: the records are the
artefact, and this script is the claim that they determine the tables.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS = os.path.join(HERE, "records")

METHOD_LABEL = {
    "nt_official": "NOTEARS (published)",
    "nt_tuned": "NOTEARS (held-out tuned)",
    "nt_original_cfg": "NOTEARS (review's cfg)",
    "dagma": "DAGMA",
    "cs_shipped": "Toolkit (shipped)",
}
ORDER = ["nt_official", "nt_tuned", "dagma", "cs_shipped"]


def load(name):
    path = os.path.join(RECORDS, name)
    rows = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return [r for r in rows
            if r.get("status", "ok") != "error" and "error" not in r]


def ms(v):
    if not v:
        return None, None
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def fmt(mean, sd, nd=3):
    if mean is None:
        return "--"
    return f"{mean:.{nd}f}±{sd:.{nd}f}"


# ----------------------------------------------------------------- tab:main ----
# Titles name the LaTeX label rather than a printed table number: the printed
# number depends on how the paper is assembled (tab:card comes first).
def table_main():
    rows = load("boundary_er.jsonl")
    agg = collections.defaultdict(list)
    for r in rows:
        if r.get("sem", "linear-gaussian") != "linear-gaussian" or r.get("q", 0):
            continue
        agg[(r["d"], r["method"])].append(r["f1"])
    dims = sorted({k[0] for k in agg})
    present = [m for m in ORDER if any((d, m) in agg for d in dims)]

    # The published schedule and its held-out-tuned variant are separate runs.  When
    # the tuning selects the published penalty at every dimension the two columns are
    # identical, which reads as a mistake and also overflows the column.  Detect it,
    # merge the columns, and say so; keep them apart if the tuning ever moves.
    merged_note = ""
    if "nt_official" in present and "nt_tuned" in present:
        paired = [d for d in dims
                  if (d, "nt_official") in agg and (d, "nt_tuned") in agg]
        identical = (len(paired) == len(dims) and
                     all(sorted(agg[(d, "nt_official")]) ==
                         sorted(agg[(d, "nt_tuned")]) for d in paired))
        if identical:
            present = [m for m in present if m != "nt_tuned"]
            merged_note = (r" A held-out sweep over five penalties selects the "
                           r"published value at every dimension, so a single "
                           r"NOTEARS column is shown; in the records the two runs "
                           r"are identical.")

    head = ["d"] + [METHOD_LABEL[m] for m in present]
    body = []
    for d in dims:
        row = [str(d)]
        for m in present:
            row.append(fmt(*ms(agg.get((d, m)) or [])))
        body.append(row)
    tex = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{Structure $F_1$ (mean$\pm$SD) under the protocol of "
           r"\S\ref{sec:protocol}, linear-Gaussian Erd\H{o}s--R\'{e}nyi DAGs, $n=2d$, "
           r"$\tau=0.3$, 10 seeds per cell. Every method, the baseline included, "
           r"selects its penalty on held-out DAGs disjoint from the test DAGs."
           + merged_note + "}",
           r"\label{tab:main}", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
           r"\begin{tabular}{@{}l" + "c" * len(present) + r"@{}}", r"\toprule",
           # typeset header: the dimension symbol is math; method names are plain
           " & ".join([r"$d$"] + [METHOD_LABEL[m] for m in present]) + r" \\",
           r"\midrule"]
    tex += [" & ".join(r) + r" \\" for r in body]
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "tab:main  single-protocol boundary  (printed as Table 2)", head, body, tex


# ------------------------------------------------------------ tab:factorial ----
def table_factorial():
    """Main effects of the 2x2x2 factorial, not the 24 individual cells.

    The claim is *which* component matters, so the table reports the effect of each
    component read against the seed-to-seed spread.  The grid itself stays in
    records/attribution_factorial.jsonl, and the consistency check the text quotes
    (the penalty ordering holding in every cell the other two factors define) is
    asserted in audit_paper_numbers.py rather than printed here.
    """
    rows = load("attribution_factorial.jsonl")
    if not rows:
        return None

    def level_means(field, vals):
        out = {}
        for v in vals:
            vv = [r["f1"] for r in rows if r.get(field) == v]
            if vv:
                out[v] = st.mean(vv)
        return out

    # pooled seed-to-seed SD: the yardstick every effect is read against
    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[(r["d"], r.get("lam1"), r.get("factor"),
                 r.get("outer"))].append(r["f1"])
    sds = [st.stdev(v) for v in by_cell.values() if len(v) > 1]
    seed_sd = st.mean(sds) if sds else 0.0

    components = [
        (r"$\lambda_1$ (sparsity)", "lam1", [0.01, 0.1]),
        ("growth factor", "factor", [2.0, 10.0]),
        ("outer iterations", "outer", [10, 100]),
    ]
    body = []
    for label, field, vals in components:
        d = level_means(field, vals)
        levels = sorted(d)
        spread = (max(d.values()) - min(d.values())) if len(d) > 1 else 0.0
        ratio = spread / seed_sd if seed_sd else float("nan")
        body.append([label,
                     ", ".join(f"{k:g}" for k in levels),
                     " / ".join(f"{d[k]:.3f}" for k in levels),
                     f"{spread:.3f}",
                     f"{ratio:.1f}"])
    body.append([r"\emph{seed-to-seed SD}", "", "", f"{seed_sd:.3f}", "1.0"])

    head = ["component", "levels", "F1 per level", "spread", "/ SD"]
    tex = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{Attribution of the reported collapse. $2\times2\times2$ factorial "
           r"over the three components that are normally set together, $d\in\{100,150,"
           r"200\}$, 5 seeds per cell. Each row gives the component's two levels, the "
           r"pooled mean $F_1$ at each, the spread between them, and that spread divided "
           r"by the pooled seed-to-seed standard deviation. The penalty is the only "
           r"component whose effect is larger than the noise it is measured against.}",
           r"\label{tab:factorial}", r"\small", r"\setlength{\tabcolsep}{5pt}",
           r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
           r"component & levels & $F_1$ per level & spread & $\div$ seed SD \\",
           r"\midrule"]
    tex += [" & ".join(r) + r" \\" for r in body]
    tex += [r"\midrule", r"\multicolumn{5}{@{}l}{\footnotesize All 24 cells: "
            r"\texttt{records/attribution\_factorial.jsonl}.}\\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "tab:factorial  attribution (main effects)  (printed as Table 3)", head, body, tex


# -------------------------------------------------------------- tab:penalty ----
def table_penalty():
    rows = load("penalty_sweep.jsonl")
    g = collections.defaultdict(list)
    for r in rows:
        # the published schedule; the rho sweep shares d=150/lambda1=0.1 with it at
        # rho0=1.0 and would otherwise double-count that cell
        if (r.get("rho0"), r.get("factor"), r.get("outer")) == (1.0, 10.0, 100) \
                and r.get("tag", "lam") != "rho":
            g[(r["d"], r.get("lam1"))].append(r)
    dims = sorted({k[0] for k in g})
    lams = sorted({k[1] for k in g})
    head = ["lambda1"] + [f"d={d}" for d in dims]
    body = []
    for lam in lams:
        row = [f"{lam}"]
        for d in dims:
            v = g.get((d, lam))
            if not v:
                row.append("--")
                continue
            m, s = ms([x["f1"] for x in v])
            eg = st.mean([x["edges"] for x in v])
            row.append(f"{m:.3f}" + (" [0]" if eg < 0.5 else f" [{eg:.0f}]"))
        body.append(row)
    tex = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{The penalty sweep. $F_1$ at the published schedule with the "
           r"sparsity penalty varied over two orders of magnitude; brackets give the "
           r"mean number of returned edges. Emptiness and saturation are the two ends "
           r"of this curve, not properties of $d$.}",
           r"\label{tab:penalty}", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{@{}l" + "c" * len(dims) + r"@{}}", r"\toprule",
           # the LaTeX header is typeset, so it must use math mode; `head` above
           # stays plain text because it is also printed to stdout
           " & ".join([r"$\lambda_1$"] + [f"$d={d}$" for d in dims]) + r" \\",
           r"\midrule"]
    tex += [" & ".join(r) + r" \\" for r in body]
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "tab:penalty  penalty sweep, F1 [edges]  (printed as Table 4)", head, body, tex


# --------------------------------------------------------------- tab:generators --
# The generator-robustness numbers are no longer emitted as a table.  They are
# carried by Figure 2, which prints the per-cell means on the page and is drawn
# from the same records, so a second carrier would be a second source of truth
# for the same numbers.  The suites stay in the audit, which asserts them.


def render(title, head, body):
    widths = [max(len(str(head[i])), *(len(str(r[i])) for r in body)) if body
              else len(str(head[i])) for i in range(len(head))]
    line = "  ".join(str(head[i]).ljust(widths[i]) for i in range(len(head)))
    out = [f"\n=== {title} ===", line, "  ".join("-" * w for w in widths)]
    for r in body:
        out.append("  ".join(str(r[i]).ljust(widths[i]) for i in range(len(head))))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=None, help="directory to write .tex files into")
    args = ap.parse_args()

    builders = [
        ("tab_main.tex", table_main),
        ("tab_factorial.tex", table_factorial),
        ("tab_penalty.tex", table_penalty),
    ]
    ok = 0
    for fname, fn in builders:
        got = fn()
        if got is None:
            print(f"\n=== {fname}: no records yet ===")
            continue
        title, head, body, tex = got
        print(render(title, head, body))
        ok += 1
        if args.tex:
            os.makedirs(args.tex, exist_ok=True)
            with open(os.path.join(args.tex, fname), "w", encoding="utf-8") as f:
                f.write("\n".join(tex) + "\n")
    if args.tex:
        print(f"\nwrote {ok} LaTeX files to {args.tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
