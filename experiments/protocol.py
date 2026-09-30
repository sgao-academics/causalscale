# -*- coding: utf-8 -*-
"""The frozen benchmark protocol.

Every number in the paper comes from this file.  It is deliberately small: one
generator family, one scoring rule, one tuning budget, and the two solvers that
are compared.  Nothing here depends on the `causalscale` package -- that package
is the *object* being measured, not part of the measurement.

Conventions
-----------
* A DAG is a matrix ``S`` with ``S[i, j] != 0`` meaning i -> j.
* Data are generated as ``X = X @ S + eps``, so ``X[:, j]`` is a function of its
  parents plus noise, and columns are standardised.
* Structure F1 compares ``|W| > tau`` against ``|S| > 0`` with the diagonal
  masked out, over the full d x d matrix (no triangular shortcut), so an
  implementation that stores its coefficients transposed scores worse, not
  differently.

Imports are lazy: ``numpy`` is enough to read the records and rebuild every
table and figure.  ``torch`` and ``scipy`` are only needed to *re-run* the
solvers.
"""
from __future__ import annotations

import numpy as np

TAU = 0.3


# --------------------------------------------------------------------- data ----
def make_er_dag(d: int, seed: int, p: float | None = None) -> np.ndarray:
    """Erdos-Renyi DAG with expected degree 2, oriented by a random order.

    ``p = 2/(d-1)`` reproduces the density used by the submission under review.
    """
    rng = np.random.default_rng(seed)
    if p is None:
        p = 2.0 / (d - 1)
    mask = np.triu(rng.random((d, d)) < p, k=1)
    mag = rng.uniform(0.5, 1.0, size=(d, d))
    sign = np.where(rng.random((d, d)) < 0.5, -1.0, 1.0)
    return (mask * mag * sign).astype(np.float64)


def make_ba_dag(d: int, seed: int, m: int = 2) -> np.ndarray:
    """Scale-free (Barabasi-Albert) DAG.

    Each arriving node receives ``m`` incoming edges drawn with probability
    proportional to the current degree of existing nodes, so early nodes become
    hubs and the in-degree distribution is heavy-tailed.  Edges are oriented
    along the arrival order, which makes the graph acyclic by construction.
    """
    rng = np.random.default_rng(seed + 4242)
    order = rng.permutation(d)
    W = np.zeros((d, d))
    deg = np.zeros(d)
    for t in range(d):
        node = int(order[t])
        if t == 0:
            deg[node] = 1.0
            continue
        cand = order[:t]
        w = deg[cand] + 1.0
        w = w / w.sum()
        k = min(m, t)
        for c in rng.choice(cand, size=k, replace=False, p=w):
            W[int(c), node] = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0)
            deg[int(c)] += 1.0
        deg[node] = 1.0
    return W


def sample_sem(S: np.ndarray, n: int, seed: int, sem: str = "linear-gaussian") -> np.ndarray:
    """Draw n samples from the structural equation implied by S."""
    rng = np.random.default_rng(seed + 777_000)
    d = S.shape[0]
    X = np.zeros((n, d))
    for j in range(d):
        pa = np.nonzero(S[:, j])[0]
        if sem in ("linear-gaussian", "linear-exponential"):
            X[:, j] = X[:, pa] @ S[pa, j]
        else:
            a = 2.0 if sem == "nonlinear-tanh" else 1.0
            for i in pa:
                X[:, j] += S[i, j] * np.tanh(a * X[:, i])
        eps = (rng.standard_normal(n) if sem != "linear-exponential"
               else rng.exponential(1.0, n) - 1.0)
        X[:, j] += eps
    return ((X - X.mean(0)) / X.std(0).clip(1e-8)).astype(np.float64)


def inject_confounder(S, X, seed, q: float = 0.0, strength: float = 1.0):
    """Add one latent common cause affecting a fraction q of the edges' children."""
    if q <= 0:
        return X, 0
    rng = np.random.default_rng(seed + 555_000)
    d = S.shape[0]
    src, dst = np.nonzero(S)
    k = int(round(q * len(src)))
    idx = rng.choice(len(src), size=k, replace=False)
    coef = np.zeros(d)
    for t in idx:
        coef[dst[t]] += strength * rng.uniform(0.5, 1.0)
    U = rng.standard_normal(X.shape[0])
    X = X + np.outer(U, coef)
    return (X - X.mean(0)) / X.std(0).clip(1e-8), k


# ------------------------------------------------------------------ metrics ----
def score(W_est, S, tau: float = TAU) -> dict:
    """Structure F1 of an estimate against a ground-truth DAG."""
    P = np.abs(W_est) > tau
    T = np.abs(S) > 0
    np.fill_diagonal(P, False)
    np.fill_diagonal(T, False)
    tp = int((P & T).sum())
    fp = int((P & ~T).sum())
    fn = int((~P & T).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(f1=f1, prec=prec, rec=rec, shd=fp + fn,
                edges=tp + fp, true_edges=tp + fn)


def score_best(W_est, S, tau_grid=None):
    """Best achievable F1 over a threshold sweep (reported alongside tau=0.3)."""
    if tau_grid is None:
        tau_grid = np.arange(0.05, 0.65, 0.05)
    best, bt = 0.0, None
    for tau in tau_grid:
        v = score(W_est, S, tau)["f1"]
        if v > best:
            best, bt = v, round(float(tau), 2)
    return best, bt


# ------------------------------------------------------------------ solvers ----
def notears_official(X, lam1: float = 0.1, max_iter: int = 100, h_tol: float = 1e-8,
                     rho_max: float = 1e16, w_threshold: float = TAU,
                     lr: float = 1e-4):
    """Faithful re-implementation of Zheng et al. (2018) `notears_linear`.

    scipy L-BFGS-B on the augmented objective, rho0 = 1.0, x10 growth, early stop
    at h(W) < 1e-8.  Requires scipy only.
    """
    import time
    from scipy.linalg import expm
    from scipy.optimize import minimize

    n, d = X.shape
    state = {"alpha": 0.0, "rho": 1.0}

    def _h(W):
        return np.trace(expm(W * W)) - d

    def _h_grad(W):
        E = expm(W * W)
        return W * (E + E.T)

    def _func(w):
        W = w.reshape(d, d)
        R = X - X @ W
        loss = 0.5 / n * (R ** 2).sum() + lam1 * np.abs(W).sum()
        G = -1.0 / n * (X.T @ R) + lam1 * np.sign(W)
        h = _h(W)
        a, r = state["alpha"], state["rho"]
        return loss + a * h + 0.5 * r * h * h, (G + (a + r * h) * _h_grad(W)).flatten()

    rho, alpha, h, W = 1.0, 0.0, np.inf, np.zeros((d, d))
    t0 = time.time()
    while rho < rho_max:
        state["alpha"], state["rho"] = alpha, rho
        sol = minimize(_func, W.flatten(), method="L-BFGS-B", jac=True,
                       options=dict(maxiter=max_iter, maxfun=max_iter))
        W = sol.x.reshape(d, d)
        h = _h(W)
        if h <= h_tol or rho >= rho_max:
            break
        alpha += rho * h
        rho *= 10
        if time.time() - t0 > 900:
            break
    return W, time.time() - t0, float(h)


def notears_adam(X, lam1: float = 0.1, rho0: float = 1.0, factor: float = 10.0,
                 outer: int = 100, inner: int = 250, lr: float = 2e-3,
                 seed: int = 0, device: str | None = None):
    """The objective of Eq. (1), optimised with Adam.

    ``0.5/n ||X - XW||^2 + lam1*||W||_1`` subject to
    ``h(W) = tr(exp(W*W)) - d = 0``, with the augmented-Lagrangian schedule
    exposed as ``rho0``/``factor`` and the stopping rule as ``outer``/``inner``.
    Those four numbers are the ones a reported "collapse" conflates.

    Note: the released `notears` package additionally symmetrises W and forces
    non-negative weights.  That is a quirk of that release, not of the published
    method, so it is not reproduced here.
    """
    import time
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
    n, d = Xt.shape
    W = torch.zeros(d, d, requires_grad=True, device=device)
    alpha, rho, h_prev = 0.0, rho0, float("inf")
    opt = torch.optim.Adam([W], lr=lr)
    t0 = time.time()
    h_c = float("nan")
    for _ in range(outer):
        for _ in range(inner):
            opt.zero_grad()
            R = Xt - Xt @ W
            h = torch.trace(torch.linalg.matrix_exp(W * W)) - d
            loss = (0.5 / n * (R ** 2).sum() + lam1 * W.abs().sum()
                    + alpha * h + 0.5 * rho * h * h)
            loss.backward()
            opt.step()
        with torch.no_grad():
            h_c = (torch.trace(torch.linalg.matrix_exp(W * W)) - d).item()
        if h_c > 0.25 * h_prev:
            rho = min(rho * factor, 1e16)
        else:
            alpha += rho * h_c
        h_prev = h_c
        if h_c < 1e-8:
            break
    return W.detach().cpu().numpy(), time.time() - t0, float(h_c)


def run_dagma(X, seed: int = 0, lam1: float = 0.02, **kw):
    """Official DAGMA release, used unchanged."""
    import time
    from dagma.linear import DagmaLinear
    t0 = time.time()
    mdl = DagmaLinear(loss_type="l2")
    try:
        W = mdl.fit(X, lambda1=lam1, s=1.0, verbose=False)
    except TypeError:
        W = mdl.fit(X, lambda1=lam1, s=1.0)
    return np.asarray(W), time.time() - t0, None


# --------------------------------------------------------------- baselines ----
_LAM_CACHE: dict[int, float] = {}
LAM_GRID = (0.003, 0.01, 0.03, 0.1, 0.3)


def tune_lam(d: int, name: str = "notears") -> float:
    """Pick the L1 penalty on held-out DAGs (seeds 1000-1002), never on the test DAG.

    This is the tuning budget every method -- baseline included -- receives.  It
    is cached per (name, d) because the selection is deterministic.
    """
    key = (name, d)
    if key in _LAM_CACHE:
        return _LAM_CACHE[key]
    acc = {lam: [] for lam in LAM_GRID}
    for vseed in (1000, 1001, 1002):
        Sv = make_er_dag(d, vseed)
        Xv = sample_sem(Sv, 2 * d, vseed)
        for lam in LAM_GRID:
            Wv, _, _ = notears_adam(Xv, lam1=lam, outer=60, seed=0)
            acc[lam].append(score(Wv, Sv)["f1"])
    best = max(LAM_GRID, key=lambda l: float(np.mean(acc[l])))
    _LAM_CACHE[key] = best
    print(f"    [tune] d={d} -> lambda1={best} "
          f"(held-out F1 {[round(float(np.mean(acc[l])), 3) for l in LAM_GRID]})",
          flush=True)
    return best


def run_nt_official(X, seed: int = 0, **kw):
    """NOTEARS under the schedule its own paper publishes: lam1=0.1, x10, 100 outer."""
    return notears_adam(X, lam1=0.1, rho0=1.0, factor=10.0, outer=100, seed=seed)


def run_nt_tuned(X, seed: int = 0, **kw):
    """NOTEARS with its penalty chosen on held-out DAGs instead of inherited."""
    lam = tune_lam(X.shape[1])
    return notears_adam(X, lam1=lam, rho0=1.0, factor=10.0, outer=100, seed=seed)


def run_nt_original_cfg(X, seed: int = 0, **kw):
    """The configuration the submission under review describes for its baseline:
    rho0 = 0.01, factor = 2, outer = 10.  Recorded so the claim can be checked."""
    return notears_adam(X, lam1=0.1, rho0=0.01, factor=2.0, outer=10, seed=seed)


def run_cs_shipped(X, seed: int = 0, **kw):
    """The engine shipped by the toolkit under audit, called through its own API.

    Imported lazily so that reading the records never requires the package.
    """
    import time
    import causalscale as cs
    t0 = time.time()
    m = cs.CausalDiscovery(X, method="cluster_aware", device=kw.get("device", "cuda"),
                           verbose=False)
    m.fit(verbose=False)
    return m._network.adjacency, time.time() - t0, None


METHODS = {
    "nt_official": run_nt_official,
    "nt_tuned": run_nt_tuned,
    "nt_original_cfg": run_nt_original_cfg,
    "dagma": run_dagma,
    "cs_shipped": run_cs_shipped,
}

# What each method needs installed to run.  Record-reading needs nothing.
REQUIRES = {"nt_official": "torch", "nt_tuned": "torch",
            "nt_original_cfg": "torch", "dagma": "dagma", "cs_shipped": "causalscale"}


def one_run(d, seed, sem="linear-gaussian", method="nt_official", q=0.0,
            n_mult=2.0, tau=TAU, topo="er"):
    """One (method, d, seed, condition) cell.  Returns a JSON-serialisable row."""
    import time
    gen = make_er_dag if topo == "er" else make_ba_dag
    S = gen(d, seed)
    n = int(round(n_mult * d))
    X = sample_sem(S, n, seed, sem=sem)
    n_conf = 0
    if q > 0:
        X, n_conf = inject_confounder(S, X, seed, q=q)
    t0 = time.time()
    try:
        W_est, t_fit, extra = METHODS[method](X, seed, S_true=S)
    except Exception as e:  # noqa: BLE001
        return dict(topo=topo, d=d, seed=seed, sem=sem, method=method, q=q, n=n,
                    error=f"{type(e).__name__}: {e}", status="error",
                    wall_s=round(time.time() - t0, 2))
    a = score(W_est, S, tau)
    b = score(W_est.T, S, tau)
    o, otau = score_best(W_est, S)
    return dict(topo=topo, d=d, seed=seed, sem=sem, method=method, q=q, n=n, tau=tau,
                f1=a["f1"], prec=a["prec"], rec=a["rec"], shd=a["shd"],
                edges=a["edges"], true_edges=a["true_edges"],
                f1_flipped=b["f1"], f1_oracle=o, tau_oracle=otau,
                fit_time_s=round(float(t_fit), 2), wall_s=round(time.time() - t0, 2),
                n_conf_edges=n_conf,
                h_final=(float(extra) if isinstance(extra, (int, float)) else None),
                status="ok")
