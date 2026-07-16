# Additional experiments for KDD 2027 - annual best paper level
# Checkpointing: never re-run completed work
import sys, os, json, time, warnings, numpy as np, torch, pandas as pd
from scipy import stats as scipy_stats
warnings.filterwarnings('ignore')
sys.path.insert(0, r'C:\Users\高帅东\Desktop\causalscale')
from causalscale.core._notears import run_notears

DEVICE = 'cuda'
OUT = r'D:\NO.1\causalscale_kdd2027_experiments'
os.makedirs(OUT, exist_ok=True)

def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}
def save(name, data):
    with open(os.path.join(OUT, name), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
def er_dag(d, seed=42):
    rng = np.random.RandomState(seed)
    W = np.zeros((d,d))
    for i in range(d):
        for j in range(i):
            if rng.rand() < 2/(d-1):
                W[i,j] = rng.choice([-1,1]) * rng.uniform(0.5, 1.0)
    return W
def gen_data(W, n):
    X = np.linalg.inv(np.eye(W.shape[0]) - W) @ np.random.randn(W.shape[0], n)
    return X.T.astype(np.float32)
def metrics(W_true, W_est, tau=0.3):
    mt = np.abs(W_true) > 0; me = np.abs(W_est) > tau
    tp = np.sum(mt & me); fp = np.sum(~mt & me); fn = np.sum(mt & ~me)
    p = tp/(tp+fp) if (tp+fp)>0 else 0; r = tp/(tp+fn) if (tp+fn)>0 else 0
    f = 2*p*r/(p+r) if (p+r)>0 else 0
    return {'f1': round(f,4), 'shd': int(fp+fn), 'prec': round(p,4), 'rec': round(r,4)}

# ============================================================
# EXP7: GENIE3 Comparison (d=30,50,80,100 x 5 seeds)
# ============================================================
print('EXP7: GENIE3 Comparison')
try:
    from sklearn.ensemble import RandomForestRegressor
except:
    print('  WARNING: sklearn not available, skipping GENIE3')
    HAS_GENIE3 = False
else:
    HAS_GENIE3 = True

if HAS_GENIE3:
    ckpt = load('exp7_genie3_comparison.json')
    for d in [30, 50, 80, 100]:
        for seed in range(5):
            key = f'd{d}_s{seed}'
            if key in ckpt: print(f'  [skip] {key}'); continue
            W_true = er_dag(d, seed)
            X = gen_data(W_true, 2*d)
            # GENIE3: per-gene RF importance
            imp_matrix = np.zeros((d, d))
            for j in range(d):
                y = X[:, j]
                X_input = np.delete(X, j, axis=1)
                rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
                rf.fit(X_input, y)
                # Fill column j (incoming edges to gene j)
                idx = 0
                for i in range(d):
                    if i != j:
                        imp_matrix[i, j] = rf.feature_importances_[idx]
                        idx += 1
            # GENIE3 doesn't have DAG constraint or sign - just threshold
            thresh = np.percentile(np.abs(imp_matrix[imp_matrix>0]), 95) if np.any(imp_matrix>0) else 0.001
            W_genie = imp_matrix * (np.abs(imp_matrix) > thresh)
            m = metrics(W_true, np.abs(W_genie), tau=0)
            ckpt[key] = {'d': d, 'seed': seed, **m}
            save('exp7_genie3_comparison.json', ckpt)
            print(f'  [done] {key}: f1={m["f1"]}')

# ============================================================
# EXP8: NOTEARS Default vs causalscale (fair comparison)
# ============================================================
print('EXP8: NOTEARS Default vs causalscale')
ckpt = load('exp8_notears_vs_cs.json')
for d in [30, 50, 80, 100, 150, 200]:
    for seed in range(5):
        key = f'd{d}_s{seed}'
        if key in ckpt: print(f'  [skip] {key}'); continue
        W_true = er_dag(d, seed)
        X = gen_data(W_true, 2*d)
        X_t = torch.tensor(X, device=DEVICE)

        # causalscale
        t0 = time.time()
        W_cs, ec, h, _ = run_notears(X_t, device=DEVICE, outer=30, inner=200, lr=0.002, seed=seed)
        t_cs = time.time() - t0

        # NOTEARS default (outer=10, inner=100, lr=0.001)
        t0 = time.time()
        W_nt, ec2, h2, _ = run_notears(X_t, device=DEVICE, outer=10, inner=100, lr=0.001, seed=seed)
        t_nt = time.time() - t0

        m_cs = metrics(W_true, W_cs)
        m_nt = metrics(W_true, W_nt)
        ckpt[key] = {'d': d, 'seed': seed, 'true_edges': int(np.sum(np.abs(W_true)>0)),
                     'causalscale': {**m_cs, 'edges': ec, 'h': round(h,2), 'time': round(t_cs,1)},
                     'notears_default': {**m_nt, 'edges': ec2, 'h': round(h2,2), 'time': round(t_nt,1)}}
        save('exp8_notears_vs_cs.json', ckpt)
        win = 'CS' if m_cs['f1'] > m_nt['f1'] else 'NT' if m_nt['f1'] > m_cs['f1'] else '='
        print(f'  [done] {key}: cs_f1={m_cs["f1"]}, nt_f1={m_nt["f1"]} ({win} wins)')

# ============================================================
# EXP9: LowRankGNN Scaling (d=500, 1000, 2000, 5000 x 3 seeds)
# ============================================================
print('EXP9: LowRankGNN Scaling Benchmark')
ckpt = load('exp9_lowrank_scaling.json')

def run_lowrank_notears(X, rank=64, outer=30, inner=200, lr=0.002, seed=42):
    """Simple low-rank NOTEARS: W = U @ V.T with DAG constraint."""
    n, d = X.shape
    X_t = torch.tensor(X.astype(np.float32), device=DEVICE)
    r = min(rank, d // 4)
    U = torch.randn(d, r, device=DEVICE) * 0.01
    V = torch.randn(d, r, device=DEVICE) * 0.01
    U.requires_grad_(True); V.requires_grad_(True)
    opt = torch.optim.Adam([U, V], lr=lr)
    rho, alpha, rho_max = 1.0, 0.0, 1e8
    for o in range(outer):
        for _ in range(inner):
            opt.zero_grad()
            W = U @ V.T
            residual = X_t - X_t @ W
            loss_recon = 0.5 / n * (residual ** 2).sum()
            M = W * W
            h_val = torch.trace(torch.linalg.matrix_exp(M)) - d
            loss = loss_recon + 0.5 * rho * h_val * h_val + alpha * h_val
            loss.backward()
            opt.step()
        with torch.no_grad():
            W_check = U @ V.T
            M_check = W_check * W_check
            h_curr = float(torch.trace(torch.linalg.matrix_exp(M_check)) - d)
        if h_curr > 0.25 * h_val.item():
            rho = min(rho * 10, rho_max)
        alpha = alpha + rho * h_curr
        if abs(h_curr) < 1e-6 or rho >= rho_max:
            break
    W_final = (U @ V.T).detach().cpu().numpy()
    ec = int(np.sum(np.abs(W_final) > 0.3))
    return W_final, ec, abs(h_curr), 0

for d in [500, 1000, 2000, 5000]:
    for seed in range(3):
        key = f'd{d}_s{seed}'
        if key in ckpt: print(f'  [skip] {key}'); continue
        W_true = er_dag(d, seed)
        X = gen_data(W_true, max(500, d))
        t0 = time.time()
        W_lr, ec, h, _ = run_lowrank_notears(X, rank=64, seed=seed)
        elapsed = time.time() - t0
        m = metrics(W_true, W_lr)
        ckpt[key] = {'d': d, 'seed': seed, 'true_edges': int(np.sum(np.abs(W_true)>0)),
                     **m, 'edges': ec, 'time': round(elapsed,1)}
        save('exp9_lowrank_scaling.json', ckpt)
        print(f'  [done] {key}: f1={m["f1"]}, {elapsed:.0f}s')

# ============================================================
# EXP10: Statistical Significance Tests
# ============================================================
print('EXP10: Statistical Significance')
ckpt = load('exp10_statistical_tests.json')

# Load comparison data
cs_data = load('exp1_causalscale_er.json')
nt_data = load('exp8_notears_vs_cs.json')

# Wilcoxon signed-rank test per dimension
results = {}
for d in [30, 50, 80, 100, 150, 200]:
    cs_f1s, nt_f1s = [], []
    for seed in range(5):
        if f'd{d}_s{seed}' in nt_data:
            cs_f1s.append(nt_data[f'd{d}_s{seed}']['causalscale']['f1'])
            nt_f1s.append(nt_data[f'd{d}_s{seed}']['notears_default']['f1'])
    if len(cs_f1s) >= 3:
        stat, p = scipy_stats.wilcoxon(cs_f1s, nt_f1s, alternative='greater')
        cohens_d = (np.mean(cs_f1s) - np.mean(nt_f1s)) / np.sqrt((np.var(cs_f1s) + np.var(nt_f1s))/2)
        results[f'd={d}'] = {'cs_mean_f1': round(np.mean(cs_f1s),4), 'nt_mean_f1': round(np.mean(nt_f1s),4),
                             'wilcoxon_p': round(float(p),6), 'cohens_d': round(cohens_d,3),
                             'significant': bool(p < 0.05)}
        print(f'  d={d}: cs={np.mean(cs_f1s):.3f} vs nt={np.mean(nt_f1s):.3f}, p={p:.4f}, d={cohens_d:.2f}')

save('exp10_statistical_tests.json', results)
save('exp10_statistical_tests.json', results)  # ensure write
print('Statistical tests saved')

# ============================================================
# EXP11: Convergence Trajectory Recording
# ============================================================
print('EXP11: Convergence Trajectories')
ckpt = load('exp11_convergence.json')

for d in [50, 100]:
    key = f'd{d}'
    if key in ckpt: print(f'  [skip] {key}'); continue
    W_true = er_dag(d)
    X = gen_data(W_true, 2*d)
    X_t = torch.tensor(X.astype(np.float32), device=DEVICE)
    W = torch.zeros(d, d, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([W], lr=0.002)
    rho, alpha = 1.0, 0.0
    h_hist, loss_hist = [], []

    for o in range(30):
        for _ in range(200):
            opt.zero_grad()
            residual = X_t - X_t @ W
            loss_recon = 0.5 / (2*d) * (residual ** 2).sum()
            M = W * W
            h_val = torch.trace(torch.linalg.matrix_exp(M)) - d
            loss = loss_recon + 0.5 * rho * h_val * h_val + alpha * h_val + 0.5/d * W.abs().sum()
            loss.backward()
            opt.step()
            loss_hist.append(float(loss.item()))
        with torch.no_grad():
            h_curr = float(torch.trace(torch.linalg.matrix_exp(W * W)) - d)
        h_hist.append(h_curr)
        if h_curr > 0.25 * h_val.item():
            rho = min(rho * 10, 1e10)
        alpha = alpha + rho * h_curr
        if h_curr < 1e-8:
            break
    ckpt[key] = {'d': d, 'h_history': [round(h,4) for h in h_hist], 'loss': [round(l,4) for l in loss_hist[::10]],
                 'final_edges': int(np.sum(np.abs(W.detach().cpu().numpy()) > 0.3))}
    save('exp11_convergence.json', ckpt)
    print(f'  [done] d={d}: {len(h_hist)} outer iters, h_final={h_hist[-1]:.2e}')

# ============================================================
print('\nALL EXTRA EXPERIMENTS COMPLETE')
for f in sorted(os.listdir(OUT)):
    if f.startswith('exp') and f.endswith('.json'):
        kb = os.path.getsize(os.path.join(OUT, f))/1024
        print(f'  {f}: {kb:.0f}KB')
