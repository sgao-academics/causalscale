# -*- coding: utf-8 -*-
"""Rebuild the paper's figures from the released per-seed records.

    python -m experiments.records_to_figures --out figures/

Three figures are produced, all of them from files this package ships:

    fig_rev_main     the boundary re-measured | the penalty curve | the score
                     does not track the acyclicity constraint      (full width)
    fig_robustness   the reversal under non-linear mechanisms | the gap closes
                     as confounders are injected                   (column)
    fig_cohort       the ARID1A--MTOR sign across 33 TCGA cohorts and the
                     sample size behind each sign                  (column)

Every panel is drawn from a released file: ``records/*.jsonl`` for the three
benchmark figures and ``results/pan_cancer_ckpt.json`` for the case study.  The
figures are authored at the width they are printed at, so the point sizes in
this file are the point sizes on the page.

Requires numpy and matplotlib.  No GPU, no solver, no torch.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS = os.path.join(HERE, "records")
RESULTS = os.path.join(os.path.dirname(HERE), "results")

# One palette for the whole paper, keyed by the name of the thing being drawn
# rather than by colour, so a series cannot be recoloured by accident.
INK = "#3F3A5B"
GREY = "#7A7A85"
RULE = "#B0B9CB"
GRID = "#E4E6EB"
COL = {
    "nt_official": "#4D6EAF",   # the published NOTEARS schedule
    "dagma": "#D0A0B6",         # the external baseline
    "cs_shipped": "#8FB578",    # the shipped engine
    "accent": "#9A5A9F",        # held-out tuned / violated constraint
}
LBL = {
    "nt_official": "NOTEARS (published cfg)",
    "dagma": "DAGMA (official)",
    "cs_shipped": "toolkit (shipped engine)",
}

MM_COLUMN = 3.33      # \columnwidth of the ACM sigconf two-column layout
MM_FULL = 7.0         # \textwidth of the same layout


def style(base=8.0, legend=None):
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#5B5B66", "axes.labelcolor": "#2B2B33",
        "xtick.color": "#5B5B66", "ytick.color": "#5B5B66",
        "font.size": base, "axes.labelsize": base,
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "legend.fontsize": legend if legend is not None else base - 1.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.titlesize": base, "figure.dpi": 200,
    })


def band(ax):
    """Panel letter, placed in axes coordinates so it cannot be clipped."""
    return ax


def load(name):
    path = os.path.join(RECORDS, name)
    rows = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
    return [r for r in rows
            if r.get("status", "ok") != "error" and "error" not in r]


def m(rows, key, **sel):
    v = [r[key] for r in rows if all(r.get(k) == val for k, val in sel.items())]
    return st.mean(v) if v else float("nan")


def sd(rows, key, **sel):
    v = [r[key] for r in rows if all(r.get(k) == val for k, val in sel.items())]
    return st.stdev(v) if len(v) > 1 else 0.0


# --------------------------------------------------------------------------- #
# figure 1: the boundary, the penalty, and the constraint that does not decide  #
# --------------------------------------------------------------------------- #
def fig_main(out):
    style(8.0, legend=6.8)
    boundary = [r for r in load("boundary_er.jsonl")
                if r.get("sem", "linear-gaussian") == "linear-gaussian"
                and not r.get("q", 0)]
    sweep = load("penalty_sweep.jsonl")
    fact = load("attribution_factorial.jsonl")
    orig = [r for r in sweep if r.get("tag") == "orig"]

    fig, axes = plt.subplots(1, 3, figsize=(MM_FULL, 2.15))

    # ---------------------------------------------------------- (a) boundary --
    ax = axes[0]
    for meth in ("nt_official", "dagma", "cs_shipped"):
        ds = sorted({r["d"] for r in boundary if r["method"] == meth})
        mu = [m(boundary, "f1", method=meth, d=d) for d in ds]
        er = [sd(boundary, "f1", method=meth, d=d) for d in ds]
        ax.errorbar(ds, mu, yerr=er, marker="o", ms=3.0, lw=1.3, capsize=2,
                    color=COL[meth], label=LBL[meth],
                    markerfacecolor="white", markeredgewidth=1.0)
    ax.set_xlabel("dimension $d$")
    ax.set_ylabel(r"structure $F_1$")
    ax.set_ylim(0, 0.85)
    ax.set_title("a  the boundary, re-measured", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4,
              borderpad=0.1, labelspacing=0.35)
    ax.axvline(150, color=RULE, lw=0.9, ls=(0, (3, 2)))
    ax.text(147, 0.055, "reported\nzero-edge point", fontsize=5.6, color=GREY,
            ha="right")

    # ----------------------------------------------------- (b) penalty curve --
    ax = axes[1]
    pub = [r for r in sweep
           if (r.get("rho0"), r.get("factor"), r.get("outer")) == (1.0, 10.0, 100.0)]
    dims = sorted({r["d"] for r in pub})
    lams = sorted({r["lam1"] for r in pub})
    ramp = ["#7A9AC3", "#4D6EAF", INK]
    for k, d in enumerate(dims):
        mu = [m(pub, "f1", d=d, lam1=lam) for lam in lams]
        ax.plot(lams, mu, marker="o", ms=3.0, lw=1.3, color=ramp[k % len(ramp)],
                label=f"$d={d}$", markerfacecolor="white", markeredgewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"sparsity penalty $\lambda_1$")
    ax.set_ylabel(r"structure $F_1$")
    ax.set_ylim(0, 0.85)
    ax.set_title("b  the penalty curve", loc="left", color=INK)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="none",
              facecolor="white", loc="lower left", handlelength=1.1,
              borderpad=0.25, labelspacing=0.3)
    eax = ax.twinx()
    eax.grid(False)
    eax.spines["top"].set_visible(False)
    eax.spines["right"].set_visible(True)
    eax.spines["right"].set_color(RULE)
    eax.tick_params(colors=GREY, labelsize=5.8)
    eax.plot(lams, [m(pub, "edges", lam1=lam) for lam in lams],
             ls=(0, (2, 2)), lw=0.9, color="#9AA0AE", zorder=0)
    eax.set_ylabel("returned edges", color=GREY, fontsize=6.5)

    # ------------------------------------------- (c) the constraint does not --
    ax = axes[2]
    groups = [
        ("exactly 0\n(%d runs)" % sum(1 for r in fact if r["h_final"] == 0.0),
         [r["f1"] for r in fact if r["h_final"] == 0.0],
         ["#8FB578"] * 1),
        ("$\\leq 0.09$\n(%d runs)" % sum(1 for r in fact if 0 < r["h_final"] <= 0.1),
         [r["f1"] for r in fact if 0 < r["h_final"] <= 0.1], ["#4D6EAF"] * 1),
        ("$\\approx 1.2$\n(%d runs)" % len(orig), [r["f1"] for r in orig],
         ["#9A5A9F"] * 1),
    ]
    rng = np.random.default_rng(7)
    for k, (lab, vals, col) in enumerate(groups):
        jit = rng.uniform(-0.16, 0.16, len(vals))
        ax.scatter(k + jit, vals, s=7, color=col[0], alpha=0.75,
                   edgecolors="white", linewidths=0.25, zorder=3)
        ax.plot([k - 0.30, k + 0.30], [st.mean(vals)] * 2, color=INK, lw=1.1,
                zorder=4)
        ax.text(k, 0.02, f"{st.mean(vals):.2f}", ha="center", fontsize=5.8,
                color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=6.0)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("final $h(W)$")
    ax.set_ylabel(r"structure $F_1$")
    ax.set_title("c  constraint vs score", loc="left", color=INK)
    ax.text(0.03, 0.97, "bar = group mean", transform=ax.transAxes, fontsize=5.4,
            color=GREY, va="top")

    fig.tight_layout(pad=0.35)
    pdf = os.path.join(out, "fig_rev_main.pdf")
    fig.savefig(pdf)
    fig.savefig(pdf.replace(".pdf", ".png"))
    plt.close(fig)
    print("wrote", pdf)
    print("  (a) cells:", len({(r["d"], r["method"]) for r in boundary}),
          "| (b) cells:", len({(r["d"], r["lam1"]) for r in pub}),
          "| (c) rows:", len(fact), "+", len(orig))
    return pdf


# --------------------------------------------------------------------------- #
# figure 2: what moves the comparison                                        #
# --------------------------------------------------------------------------- #
def fig_robustness(out):
    style(6.6, legend=6.0)
    mech = load("mechanism.jsonl")
    conf = load("confounder.jsonl")

    fam_order = ["linear-gaussian", "linear-exponential", "nonlinear-tanh"]
    fam_lab = {"linear-gaussian": "lin-Gauss",
               "linear-exponential": "lin-exp",
               "nonlinear-tanh": "$\\tanh$"}
    ds = sorted({r["d"] for r in mech})
    meths = ("nt_official", "dagma", "cs_shipped")
    mcol = {"nt_official": "#4D6EAF", "dagma": "#D0A0B6", "cs_shipped": "#8FB578"}
    mshape = {"nt_official": "o", "dagma": "^", "cs_shipped": "s"}

    fig, axes = plt.subplots(2, 1, figsize=(MM_COLUMN, 3.05))
    ax = axes[0]
    rows = [(fam, d) for d in ds for fam in fam_order]
    # Who leads is carried by the colour of the row label rather than by a text
    # annotation per row, which collided with the connecting lines.
    AHEAD = INK
    BEHIND = "#2E6B34"
    labels, label_cols = [], []
    for i, (fam, d) in enumerate(rows):
        vals = {me: m(mech, "f1", sem=fam, d=d, method=me) for me in meths}
        lead_nt = vals["nt_official"] > vals["cs_shipped"]
        ax.plot([min(vals.values()), max(vals.values())], [i, i],
                color=AHEAD if lead_nt else BEHIND, lw=0.9, zorder=1)
        for me in meths:
            ax.scatter(vals[me], i, s=13, marker=mshape[me], color=mcol[me],
                       edgecolors="white", linewidths=0.4, zorder=3)
        ax.text(0.625, i, f"{vals['nt_official']:.2f}", fontsize=5.2, va="center",
                color=mcol["nt_official"])
        ax.text(0.015, i, f"{vals['cs_shipped']:.2f}", fontsize=5.2, va="center",
                color=mcol["cs_shipped"])
        labels.append(fam_lab[fam])
        label_cols.append(AHEAD if lead_nt else BEHIND)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=5.4)
    for tick, col in zip(ax.get_yticklabels(), label_cols):
        tick.set_color(col)
    ax.set_ylim(len(rows) - 0.5, -0.75)
    ax.set_xlim(-0.02, 0.68)
    ax.axhline(len(ds) - 0.5, color=INK, lw=0.8)
    for j, d in enumerate(ds):
        ax.text(0.665, (j + 0.5) * len(fam_order) - 0.5, f"$d={d}$", fontsize=5.4,
                va="center", color=GREY)
    ax.set_xlabel(r"structure $F_1$")
    ax.set_title("a  the ordering reverses only for $\\tanh$", loc="left", color=INK)

    ax = axes[1]
    cq = sorted({float(r["q"]) for r in conf})
    cd = sorted({r["d"] for r in conf})
    for k, d in enumerate(cd):
        gaps, errs = [], []
        for q in cq:
            a = [r["f1"] for r in conf if r["d"] == d and float(r["q"]) == q
                 and r["method"] == "nt_official"]
            b = [r["f1"] for r in conf if r["d"] == d and float(r["q"]) == q
                 and r["method"] == "cs_shipped"]
            gaps.append(st.mean(a) - st.mean(b))
            errs.append(0.5 * (st.stdev(a) + st.stdev(b)))
        ax.errorbar(cq, gaps, yerr=errs, marker="o", ms=3.2, lw=1.3, capsize=2.2,
                    color=["#4D6EAF", "#8FB578"][k % 2], label=f"$d={d}$",
                    markerfacecolor="white", markeredgewidth=1.0)
        for j, (q, g) in enumerate(zip(cq, gaps)):
            if not k and j not in (0, len(cq) - 1):
                continue          # label the d=50 line at its two ends only
            # Inside the axes on purpose: a label centred on q=0 is crossed by
            # the spine and its leading "+" reads as a minus sign.
            off = (15, 6) if not k else (13, -8)
            ax.annotate(f"{g:+.3f}", (q, g), textcoords="offset points",
                        xytext=off, ha="center", va="center", fontsize=5.2,
                        color=INK)
    ax.axhline(0, color=RULE, lw=0.9)
    ax.set_xticks(cq)
    ax.set_xticklabels([f"{q:g}" for q in cq])
    ax.set_xlabel("confounder strength $q$")
    ax.set_ylabel(r"$F_1$ gap (NOTEARS $-$ toolkit)")
    ax.set_title("b  the gap closes as confounders are added", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)
    ax.set_ylim(-0.26, 0.34)

    fig.tight_layout(pad=0.3)
    pdf = os.path.join(out, "fig_robustness.pdf")
    fig.savefig(pdf)
    fig.savefig(pdf.replace(".pdf", ".png"))
    plt.close(fig)
    print("wrote", pdf)
    print("  (a) mechanism rows:", len(mech), "| (b) confounder rows:", len(conf))
    return pdf


# --------------------------------------------------------------------------- #
# figure 3: the case study, per cohort                                        #
# --------------------------------------------------------------------------- #
def fig_cohort(out):
    style(6.6, legend=6.0)
    path = os.path.join(RESULTS, "pan_cancer_ckpt.json")
    if not os.path.exists(path):
        # The panel reads the shipped case-study checkpoint, not a record file.
        # Fail loudly about the missing input instead of writing a blank figure.
        print(f"SKIPPED fig_cohort: {path} not found")
        return None
    ckpt = json.load(open(path, encoding="utf-8"))

    up = sorted(c for c, v in ckpt.items() if abs(v.get("arid1a_to_mtor", 0)) > 0.3)
    dn = sorted(c for c, v in ckpt.items() if abs(v.get("mtor_to_arid1a", 0)) > 0.3)
    no = sorted(c for c, v in ckpt.items()
                if abs(v.get("arid1a_to_mtor", 0)) <= 0.3
                and abs(v.get("mtor_to_arid1a", 0)) <= 0.3)
    order = up + dn + no
    signed = [ckpt[c].get("arid1a_to_mtor", 0) - ckpt[c].get("mtor_to_arid1a", 0)
              for c in order]
    ns = [ckpt[c].get("n", 0) for c in order]

    fig, axes = plt.subplots(2, 1, figsize=(MM_COLUMN, 2.45),
                            gridspec_kw={"height_ratios": [2.4, 1.0]})
    ax = axes[0]
    cols = ["#4D6EAF"] * len(up) + ["#9A5A9F"] * len(dn) + ["#B0B9CB"] * len(no)
    ax.bar(range(len(order)), signed, color=cols, width=0.72,
           edgecolor="white", linewidth=0.3)
    ax.axhline(0, color=INK, lw=0.8)
    for x in (len(up) - 0.5, len(up) + len(dn) - 0.5):
        ax.axvline(x, color=INK, lw=0.7, ls=(0, (2, 2)))
    for x, (lab, n_) in enumerate([(f"ARID1A$\\rightarrow$MTOR ({len(up)})",
                                    len(up)),
                                   (f"MTOR$\\rightarrow$ARID1A ({len(dn)})",
                                    len(dn)),
                                   (f"no edge ({len(no)})", len(no))]):
        centre = [len(up) / 2 - 0.5, len(up) + len(dn) / 2 - 0.5,
                  len(up) + len(dn) + len(no) / 2 - 0.5][x]
        ax.text(centre, 0.62, lab, ha="center", fontsize=5.6, color=INK)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(-0.62, 0.72)
    ax.set_ylabel("signed weight $W$")
    ax.set_xticks([])
    ax.set_title("a  sign and size of the ARID1A--MTOR edge, 33 cohorts",
                 loc="left", color=INK)
    ax.text(0.005, 0.02, "positive = ARID1A$\\rightarrow$MTOR",
            transform=ax.transAxes, fontsize=5.2, color=GREY)

    ax = axes[1]
    ax.bar(range(len(order)), ns, color="#D8DCE6", width=0.72,
           edgecolor="white", linewidth=0.3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=4.6)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylabel("$n$")
    ax.set_ylim(0, max(ns) * 1.15)
    ax.set_title("b  sample size of each cohort", loc="left", color=INK)
    med = [st.median([ckpt[c]["n"] for c in g]) for g in (up, dn, no)]
    for centre, mv in zip([len(up) / 2 - 0.5, len(up) + len(dn) / 2 - 0.5,
                           len(up) + len(dn) + len(no) / 2 - 0.5], med):
        ax.hlines(mv, centre - 3, centre + 3, color=COL["accent"], lw=1.2)
    ax.text(0.99, 0.92, "median " + " / ".join(f"{v:.0f}" for v in med),
            transform=ax.transAxes, ha="right", va="top", fontsize=5.2,
            color=COL["accent"])

    fig.tight_layout(pad=0.3)
    pdf = os.path.join(out, "fig_cohort.pdf")
    fig.savefig(pdf)
    fig.savefig(pdf.replace(".pdf", ".png"))
    plt.close(fig)
    print("wrote", pdf)
    print(f"  cohorts {len(order)}: up {len(up)} down {len(dn)} none {len(no)}"
          f" | median n {med}")
    return pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    ap.add_argument("--only", "--name", dest="only", default=None,
                    help="one of fig_rev_main / fig_robustness / fig_cohort")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    todo = {"fig_rev_main": fig_main, "fig_robustness": fig_robustness,
            "fig_cohort": fig_cohort}
    if args.only:
        todo = {args.only: todo[args.only]}
    for fn in todo.values():
        fn(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
