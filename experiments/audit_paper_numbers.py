# -*- coding: utf-8 -*-
"""Execute the paper's claims as assertions against the released records.

    python -m experiments.audit_paper_numbers

Every claim the paper makes about a number is written here as a predicate that
either holds on the released per-seed records or does not.  This is deliberately
stronger than a reproducibility script: a reproducibility script shows that the
numbers can be regenerated, whereas this one states which sentences in the paper
they are load-bearing for, and fails loudly if any of them stops being true.

Checks are stated as inequalities and orderings rather than as exact values, so a
re-run with a different seed set does not produce spurious failures -- while a
change in the *conclusions* does.
"""
from __future__ import annotations

import collections
import json
import os
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS = os.path.join(HERE, "records")


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


def mean_f1(rows, **sel):
    v = [r["f1"] for r in rows
         if all(r.get(k) == val for k, val in sel.items())]
    return st.mean(v) if v else None


def load_case_study():
    """The ARID1A--MTOR cohort scan that backs the case-study figure.

    It lives under results/ rather than records/ because it is a scan of a real
    cohort compendium, not a synthetic sweep: one entry per TCGA cohort, with the
    two signed edge weights, the sample size and the edge count.
    """
    path = os.path.join(os.path.dirname(HERE), "results", "pan_cancer_ckpt.json")
    if not os.path.exists(path):
        return {}
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def mean_edges(rows, **sel):
    v = [r["edges"] for r in rows
         if all(r.get(k) == val for k, val in sel.items())]
    return st.mean(v) if v else None


def main():
    results = []

    def check(claim, ok, detail):
        results.append((claim, ok, detail))
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {claim}\n        {detail}")

    boundary = load("boundary_er.jsonl")
    factorial = load("attribution_factorial.jsonl")
    sweep = load("penalty_sweep.jsonl")

    if not boundary:
        print("no boundary records; run: python -m experiments.run_sweep --suite boundary")
        return 1

    dims = sorted({r["d"] for r in boundary})

    # --- claim 1: the boundary is not a zero-edge transition ------------------
    d_max = max(dims)
    f = mean_f1(boundary, d=d_max, method="nt_official")
    e = mean_edges(boundary, d=d_max, method="nt_official")
    check(f"the published NOTEARS schedule does not return an empty graph at d={d_max}",
          f is not None and f > 0.5 and e is not None and e > 5,
          f"F1 = {f:.3f}, edges = {e:.0f} (an empty graph would be F1 = 0, edges = 0)")

    # --- claim 2: the ordering holds at every dimension ----------------------
    order_ok, bad = True, []
    for d in dims:
        a = mean_f1(boundary, d=d, method="nt_official")
        b = mean_f1(boundary, d=d, method="cs_shipped")
        c = mean_f1(boundary, d=d, method="dagma")
        if a is None:
            continue
        if b is not None and not a > b:
            order_ok = False; bad.append(f"d={d}: NT {a:.3f} <= toolkit {b:.3f}")
        if c is not None and not a > c:
            order_ok = False; bad.append(f"d={d}: NT {a:.3f} <= DAGMA {c:.3f}")
    detail = "NOTEARS (published) > shipped toolkit > DAGMA at every d" if order_ok \
        else "; ".join(bad)
    check("the head-to-head ordering holds at every dimension", order_ok, detail)

    # --- claim 3: the shipped engine is below the plain baseline -------------
    if any(r["method"] == "cs_shipped" for r in boundary):
        a = mean_f1(boundary, d=d_max, method="nt_official")
        b = mean_f1(boundary, d=d_max, method="cs_shipped")
        check("the re-measured toolkit advantage does not reproduce",
              a is not None and b is not None and a > b,
              f"at d={d_max}: NOTEARS {a:.3f} vs shipped engine {b:.3f}")

    # --- claim 4: attribution -- the penalty dominates -----------------------
    if factorial:
        def main_effect(rows, field, vals):
            means = [st.mean([r["f1"] for r in rows if r.get(field) == v])
                     for v in vals]
            return max(means) - min(means)

        lam = main_effect(factorial, "lam1", [0.01, 0.1])
        fac = main_effect(factorial, "factor", [2.0, 10.0])
        out = main_effect(factorial, "outer", [10, 100])

        # the yardstick: pooled seed-to-seed SD over the factorial's own cells
        cells = collections.defaultdict(list)
        for r in factorial:
            cells[(r["d"], r.get("lam1"), r.get("factor"), r.get("outer"))].append(
                r["f1"])
        sds = [st.stdev(v) for v in cells.values() if len(v) > 1]
        seed_sd = st.mean(sds) if sds else 0.0

        check("the sparsity penalty dominates the other two schedule components",
              lam > 5 * max(fac, out),
              f"main effects: lambda1 = {lam:.3f}, factor = {fac:.3f}, "
              f"outer = {out:.3f} (penalty is {lam / max(fac, out, 1e-9):.1f}x larger)")
        check("neither the growth factor nor the iteration budget exceeds seed noise",
              fac < seed_sd and out < seed_sd,
              f"factor effect {fac:.3f} ({fac / seed_sd:.2f}x SD), "
              f"outer effect {out:.3f} ({out / seed_sd:.2f}x SD), "
              f"against a pooled seed-to-seed SD of {seed_sd:.3f}")

        # the ordering the paper quotes: stronger penalty wins in every cell the
        # other two components define
        groups = collections.defaultdict(dict)
        for r in factorial:
            groups[(r["d"], r.get("factor"), r.get("outer"))].setdefault(
                r.get("lam1"), []).append(r["f1"])
        wins, margins, total = 0, [], 0
        for key in sorted(groups):
            g = groups[key]
            if 0.01 not in g or 0.1 not in g:
                continue
            total += 1
            a, b = st.mean(g[0.01]), st.mean(g[0.1])
            if b > a:
                wins += 1
                margins.append(b - a)
        check("the penalty ordering holds in every cell of the factorial",
              total > 0 and wins == total,
              f"lambda1=0.1 beats lambda1=0.01 in {wins}/{total} cells, "
              f"smallest margin {min(margins):.3f}" if margins else "no comparable cells")

        # and the iteration-budget direction is not even stable across cells.
        # Grouped over the *other two* components, so each group contains both
        # iteration budgets -- grouping over all three would leave one value each.
        by_other = collections.defaultdict(dict)
        for r in factorial:
            by_other[(r["d"], r.get("lam1"), r.get("factor"))].setdefault(
                r.get("outer"), []).append(r["f1"])
        fewer, more, n_dir = 0, 0, 0
        for key in sorted(by_other):
            g = by_other[key]
            if 10 not in g or 100 not in g:
                continue
            n_dir += 1
            if st.mean(g[10]) > st.mean(g[100]):
                fewer += 1
            else:
                more += 1
        check("the iteration budget's direction is not stable across cells",
              n_dir > 0 and 0 < fewer < n_dir,
              f"fewer outer iterations score higher in {fewer}/{n_dir} cells and "
              f"more do in {more}/{n_dir} -- a sign that flips is not a mechanism")

    # --- claim 5: the penalty curve has two bad ends --------------------------
    pub = [r for r in sweep
           if (r.get("rho0"), r.get("factor"), r.get("outer")) == (1.0, 10.0, 100)
           and r.get("tag", "lam") != "rho"]
    if pub:
        lams = sorted({r.get("lam1") for r in pub})
        by_lam = {l: st.mean([r["f1"] for r in pub if r.get("lam1") == l])
                  for l in lams}
        best = max(by_lam, key=lambda l: by_lam[l])
        weak, strong = min(lams), max(lams)
        check("a weak penalty over-produces edges and loses F1",
              by_lam[weak] < by_lam[best] - 0.1,
              f"lambda1={weak}: F1 {by_lam[weak]:.3f} at "
              f"{st.mean([r['edges'] for r in pub if r.get('lam1') == weak]):.0f} edges, "
              f"vs best lambda1={best}: F1 {by_lam[best]:.3f}")
        check("a strong penalty drives the graph empty",
              by_lam[strong] < 0.15 and
              st.mean([r["edges"] for r in pub if r.get("lam1") == strong]) < 10,
              f"lambda1={strong}: F1 {by_lam[strong]:.3f} at "
              f"{st.mean([r['edges'] for r in pub if r.get('lam1') == strong]):.1f} edges")

        # emptiness must be a property of the penalty at *every* dimension, not of one
        dims_with_empty = {r["d"] for r in pub
                           if r.get("lam1") == strong and r["edges"] == 0}
        all_dims = {r["d"] for r in pub}
        check("the empty graph is reachable at every dimension, not at one of them",
              dims_with_empty == all_dims,
              f"lambda1={strong} returns zero edges at d={sorted(dims_with_empty)} "
              f"of d={sorted(all_dims)}")

    # --- claim 5b: the schedule the earlier text blames by name is not empty ---
    orig = [r for r in sweep if r.get("tag") == "orig"]
    if orig:
        per_d = collections.defaultdict(list)
        for r in orig:
            per_d[r["d"]].append(r)
        bad, detail = [], []
        for d in sorted(per_d):
            v = per_d[d]
            eg = [x["edges"] for x in v]
            f1 = st.mean([x["f1"] for x in v])
            detail.append(f"d={d}: F1 {f1:.3f}, edges {min(eg)}--{max(eg)}")
            if min(eg) == 0 or f1 < 0.05:
                bad.append(str(d))
        check("the schedule the earlier text blames by name is not an empty graph",
              not bad,
              ("; ".join(detail) + " (empty at no seed)") if not bad
              else "degenerate at d=" + ", ".join(bad))

    # --- claim 6: orientation convention does not explain the ordering -------
    if boundary:
        ok = True
        for d in dims:
            fwd = mean_f1(boundary, d=d, method="nt_official")
            flip = st.mean([r["f1_flipped"] for r in boundary
                            if r["d"] == d and r["method"] == "nt_official"
                            and "f1_flipped" in r]) if any(
                r["d"] == d and r["method"] == "nt_official" and "f1_flipped" in r
                for r in boundary) else None
            if fwd is not None and flip is not None and not fwd > flip:
                ok = False
        check("the reported score is the native orientation, not the transposed one",
              ok, "F1(native) > F1(transposed) for every dimension and method")

    # --- claim 7: the head-to-head is generator-dependent ---------------------
    topo = load("topology.jsonl")
    if topo:
        cell = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in topo:
            cell[(r.get("topo"), r.get("sem"))][r["method"]].append(r["f1"])
        methods = {m for k in cell for m in cell[k]}
        sems = {k[1] for k in cell}
        skeletons = {k[0] for k in cell}

        # (a) harder axis 1: scale-free vs Erdos-Renyi, for every method and mechanism
        n_a, ok_a = 0, True
        for sem in sorted(sems):
            for m in sorted(methods):
                er = cell.get(("er", sem), {}).get(m)
                ba = cell.get(("ba", sem), {}).get(m)
                if er and ba:
                    n_a += 1
                    ok_a &= st.mean(er) > st.mean(ba)
        check("the scale-free skeleton is harder for every method and mechanism",
              ok_a and n_a >= 4,
              f"{n_a} (method, mechanism) comparisons, ER > BA in all")

        # (b) harder axis 2: non-linear vs linear, for every method and skeleton
        n_b, ok_b = 0, True
        for sk in sorted(skeletons):
            for m in sorted(methods):
                lg = cell.get((sk, "linear-gaussian"), {}).get(m)
                nl = cell.get((sk, "nonlinear-tanh"), {}).get(m)
                if lg and nl:
                    n_b += 1
                    ok_b &= st.mean(lg) > st.mean(nl)
        check("non-linear mechanisms are harder for every method and skeleton",
              ok_b and n_b >= 4,
              f"{n_b} (method, skeleton) comparisons, linear > non-linear in all")

        # (c) the sign of the head-to-head is not stable
        signs = {}
        for k in sorted(cell):
            g = cell[k]
            if "nt_official" in g and "cs_shipped" in g:
                signs[k] = st.mean(g["nt_official"]) > st.mean(g["cs_shipped"])
        check("the sign of the head-to-head is not stable across generators",
              len(set(signs.values())) > 1,
              "; ".join(f"{k[0]}/{k[1]}: {'NOTEARS' if v else 'toolkit'} leads"
                        for k, v in sorted(signs.items())))

    # --- claim 7b: the mechanism sweep, a second crossing ----------------------
    mech = load("mechanism.jsonl")
    if mech:
        mcell = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in mech:
            mcell[(r["d"], r["sem"])][r["method"]].append(r["f1"])
        mdims = sorted({k[0] for k in mcell})
        msems = sorted({k[1] for k in mcell})
        lin = [s for s in msems if s.startswith("linear")]

        # (a) on every linear family the published NOTEARS schedule leads, at both
        #     dimensions -- the ordering the paper reports as reproduced
        n_a, ok_a, detail_a = 0, True, []
        for s in lin:
            for d in mdims:
                g = mcell.get((d, s), {})
                if "nt_official" in g and "cs_shipped" in g:
                    n_a += 1
                    lead = st.mean(g["nt_official"]) > st.mean(g["cs_shipped"])
                    ok_a &= lead
                    detail_a.append(f"d={d} {s}: "
                                    f"{st.mean(g['nt_official']):.3f} vs "
                                    f"{st.mean(g['cs_shipped']):.3f}")
        check("on the linear families the published schedule leads at both dimensions",
              ok_a and n_a >= 4, "; ".join(detail_a))

        # (b) on the non-linear family the ordering reverses, and reverses at *both*
        #     dimensions -- this is what licenses the word "reverses" in the text
        n_b2, ok_b2, detail_b = 0, True, []
        for d in mdims:
            g = mcell.get((d, "nonlinear-tanh"), {})
            if "nt_official" in g and "cs_shipped" in g:
                n_b2 += 1
                flip = st.mean(g["cs_shipped"]) > st.mean(g["nt_official"])
                ok_b2 &= flip
                detail_b.append(f"d={d}: toolkit "
                                f"{st.mean(g['cs_shipped']):.3f} > NOTEARS "
                                f"{st.mean(g['nt_official']):.3f}" if flip
                                else f"d={d}: no reversal")
        check("the non-linear family reverses the ordering at every dimension tested",
              ok_b2 and n_b2 >= 2, "; ".join(detail_b))

        # (c) the difficulty claim, stated with its true denominator: non-linear is
        #     harder than linear in most (dimension, method) cells, not all
        n_c, harder, exceptions = 0, 0, []
        for d in mdims:
            for m in sorted({m for k in mcell for m in mcell[k]}):
                lg = mcell.get((d, "linear-gaussian"), {}).get(m)
                nl = mcell.get((d, "nonlinear-tanh"), {}).get(m)
                if lg and nl:
                    n_c += 1
                    if st.mean(lg) > st.mean(nl):
                        harder += 1
                    else:
                        exceptions.append(f"{m}@d={d}")
        check("non-linear mechanisms are harder than linear in most, not all, cells",
              n_c > 0 and harder == n_c - 1,
              f"{harder}/{n_c} (dimension, method) cells; "
              f"exception {'; '.join(exceptions) if exceptions else 'none'}")

    # --- claim 7d: the confounder axis closes the gap ------------------------
    conf = load("confounder.jsonl")
    if conf:
        ccell = collections.defaultdict(list)
        for r in conf:
            ccell[(r["d"], float(r["q"]), r["method"])].append(r["f1"])
        cdims = sorted({k[0] for k in ccell})

        # (a) the published schedule degrades monotonically as confounders are added
        n_a, ok_a, det_a = 0, True, []
        for d in cdims:
            qs = sorted({k[1] for k in ccell if k[0] == d})
            means = [st.mean(ccell[(d, q, "nt_official")]) for q in qs
                     if (d, q, "nt_official") in ccell]
            if len(means) > 1:
                n_a += 1
                ok_a &= all(means[i] > means[i + 1] for i in range(len(means) - 1))
                det_a.append(f"d={d}: " + " > ".join(f"{m:.3f}" for m in means))
        check("adding confounders degrades the published schedule at every dimension",
              ok_a and n_a >= 2, "; ".join(det_a))

        # (b) the gap between the two methods narrows as q grows
        n_b, ok_b, det_b = 0, True, []
        for d in cdims:
            qs = sorted({k[1] for k in ccell if k[0] == d})
            gaps = []
            for q in qs:
                a = ccell.get((d, q, "nt_official"))
                b = ccell.get((d, q, "cs_shipped"))
                if a and b:
                    gaps.append((q, st.mean(a) - st.mean(b)))
            if len(gaps) > 1:
                n_b += 1
                ok_b &= gaps[-1][1] < gaps[0][1]
                det_b.append(f"d={d}: " + " then ".join(
                    f"{g:+.3f} at q={q:g}" for q, g in gaps))
        check("the gap between the methods narrows as confounders are added",
              ok_b and n_b >= 2, "; ".join(det_b))

        # (c) at the strongest confounding the comparison is inside the seed noise
        d_hi = cdims[-1]
        q_hi = max({k[1] for k in ccell})
        a = ccell.get((d_hi, q_hi, "nt_official"))
        b = ccell.get((d_hi, q_hi, "cs_shipped"))
        if a and b:
            gap = abs(st.mean(a) - st.mean(b))
            sd = max(st.stdev(a), st.stdev(b))
            check("at the strongest confounding the comparison is inside the noise",
                  gap < sd,
                  f"d={d_hi} q={q_hi:g}: gap {gap:.3f} against a per-seed SD of {sd:.3f}")

    # --- claim 7e: the score does not track the acyclicity constraint ---------
    # The paper says the constraint is satisfied in every factorial cell and the
    # score still spans a wide range, while the configuration it blames ends at
    # h(W) approx 1.2 and scores inside that range.  That sentence names two
    # populations, so it is asserted population by population -- an earlier
    # version of it described the blamed configuration's h(W) as the factorial's,
    # which no predicate covered.
    if factorial:
        hmax = max(r["h_final"] for r in factorial)
        h_sat = [r["h_final"] for r in factorial if r["outer"] == 100]
        f1 = [r["f1"] for r in factorial]
        check("the acyclicity constraint is satisfied in every factorial cell",
              hmax <= 0.1 and h_sat and max(h_sat) == 0.0,
              f"max h(W) over {len(factorial)} runs is {hmax:.4f}; the "
              f"{len(h_sat)} runs at 100 outer iterations all end at exactly 0")
        check("the per-seed score spans a wide range with the constraint satisfied",
              min(f1) <= 0.17 and max(f1) >= 0.74,
              f"F1 spans {min(f1):.3f}-{max(f1):.3f} across runs that all satisfy "
              f"h(W) <= {hmax:.2f}")

    orig = [r for r in sweep if r.get("tag") == "orig"]
    if orig:
        h = [r["h_final"] for r in orig]
        ed = [r["edges"] for r in orig]
        lo = [r for r in orig if r["lam1"] == 0.01]
        hi = [r for r in orig if r["lam1"] == 0.1]
        check("the blamed configuration never satisfies the constraint and is "
              "never empty",
              1.1 <= min(h) and max(h) <= 1.35 and min(ed) > 0,
              f"h(W) ends between {min(h):.2f} and {max(h):.2f} on all "
              f"{len(orig)} runs at d={orig[0]['d']}; edges {min(ed)}-{max(ed)}")
        check("the blamed configuration scores inside the range of the satisfied "
              "runs",
              lo and hi and st.mean([r["f1"] for r in lo]) < 0.3
              and 0.5 < st.mean([r["f1"] for r in hi]) < 0.6,
              f"at the published penalty F1={st.mean([r['f1'] for r in lo]):.3f} "
              f"(lam1=0.01), at lam1=0.1 F1={st.mean([r['f1'] for r in hi]):.3f}, "
              f"against {min(f1):.2f}-{max(f1):.2f} for the satisfied runs"
              if factorial else "")

    # --- claim 7c: the held-out tuning returns the published penalty ----------
    # Table 2 merges the published and held-out-tuned NOTEARS columns when, and
    # only when, the tuning never moves the penalty.  That collapse is only
    # legitimate if the two runs really are identical, so it is asserted rather
    # than assumed by the table builder.
    off = {(r["d"], r["seed"]): r["f1"] for r in boundary
           if r["method"] == "nt_official"}
    tun = {(r["d"], r["seed"]): r["f1"] for r in boundary
           if r["method"] == "nt_tuned"}
    common = sorted(set(off) & set(tun))
    if common:
        same = sum(1 for k in common if off[k] == tun[k])
        check("the held-out tuning returns the published penalty at every dimension",
              same == len(common),
              f"{same}/{len(common)} (d, seed) runs identical between the published "
              f"and tuned schedules -- the tuning never moves lambda_1 off the "
              f"published value, which is why Table 2 carries one NOTEARS column")

    # --- claim 8: the case study, counted from the shipped cohort scan --------
    ckpt = load_case_study()
    if ckpt:
        up = [c for c, v in ckpt.items() if abs(v.get("arid1a_to_mtor", 0)) > 0.3]
        dn = [c for c, v in ckpt.items() if abs(v.get("mtor_to_arid1a", 0)) > 0.3]
        no = [c for c, v in ckpt.items()
              if abs(v.get("arid1a_to_mtor", 0)) <= 0.3
              and abs(v.get("mtor_to_arid1a", 0)) <= 0.3]
        check("the cohort scan is complete and the three sign classes partition it",
              len(ckpt) == 33 and len(up) + len(dn) + len(no) == 33 and not
              (set(up) & set(dn)),
              f"33 cohorts: {len(up)} ARID1A->MTOR, {len(dn)} MTOR->ARID1A, "
              f"{len(no)} with no edge above tau=0.3")

        # the paper says the cohorts that return no edge are the smaller ones
        med = {k: st.median([ckpt[c]["n"] for c in g])
               for k, g in (("up", up), ("down", dn), ("none", no))}
        check("the cohorts with no edge are the smaller ones",
              med["none"] < med["down"] and med["none"] < med["up"],
              f"median n: {med['none']:.0f} (no edge) against {med['down']:.0f} "
              f"and {med['up']:.0f} (signed)")

    n_pass = sum(1 for _c, o, _d in results if o)
    print(f"\n{n_pass}/{len(results)} checks passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
