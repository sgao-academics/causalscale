# Run EXP9 (fixed) + EXP10 + EXP11 sequentially
import subprocess, sys, os

# EXP9
print('=== EXP9: LowRankGNN (fixed) ===')
exp9_script = r'C:\Users\高帅东\Desktop\causalscale\paper\scripts\run_exp9_fixed.py'
r = subprocess.run([sys.executable, '-u', exp9_script], capture_output=False)
if r.returncode != 0:
    print(f'EXP9 FAILED: {r.returncode}')
    sys.exit(1)

# EXP10
print('\n=== EXP10: Statistical Tests ===')
import json, numpy as np
from scipy import stats as scipy_stats

OUT = r'D:\NO.1\causalscale_kdd2027_experiments'
nt_data = json.load(open(os.path.join(OUT, 'exp8_notears_vs_cs.json'), encoding='utf-8'))
results = {}
for d in [30, 50, 80, 100, 150, 200]:
    cs_f1s, nt_f1s = [], []
    for seed in range(5):
        k = f'd{d}_s{seed}'
        if k in nt_data:
            cs_f1s.append(nt_data[k]['causalscale']['f1'])
            nt_f1s.append(nt_data[k]['notears_default']['f1'])
    if len(cs_f1s) >= 3:
        stat, p = scipy_stats.wilcoxon(cs_f1s, nt_f1s, alternative='greater')
        d_cohen = (np.mean(cs_f1s)-np.mean(nt_f1s))/np.sqrt((np.var(cs_f1s)+np.var(nt_f1s))/2)
        results[f'd={d}'] = {'cs_f1': round(np.mean(cs_f1s),4), 'nt_f1': round(np.mean(nt_f1s),4),
                            'wilcoxon_p': round(float(p),6), 'cohens_d': round(d_cohen,3), 'sig': bool(p<0.05)}
        print(f'  d={d}: cs={np.mean(cs_f1s):.3f} vs nt={np.mean(nt_f1s):.3f}, p={p:.4f}, d={d_cohen:.2f}')
with open(os.path.join(OUT, 'exp10_statistical_tests.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

# GENIE3 comparison
genie = json.load(open(os.path.join(OUT, 'exp7_genie3_comparison.json'), encoding='utf-8'))
print('\n  GENIE3 comparison:')
for d in [30,50,80,100]:
    f1s = [v['f1'] for k,v in genie.items() if v['d']==d]
    if f1s: print(f'    d={d}: F1={np.mean(f1s):.4f}+-{np.std(f1s):.4f}')

# EXP11
print('\n=== EXP11: Convergence Trajectories ===')
import torch, warnings
warnings.filterwarnings('ignore')

import sys; sys.path.insert(0, r'C:\Users\高帅东\Desktop\causalscale')
from causalscale.core._notears import run_notears
from run_extra_experiments import er_dag, gen_data

ckpt_name = 'exp11_convergence.json'
ckpt = json.load(open(os.path.join(OUT, ckpt_name), encoding='utf-8')) if os.path.exists(os.path.join(OUT, ckpt_name)) else {}

DEVICE = 'cuda'
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
    n = 2 * d
    for o in range(30):
        for _ in range(200):
            opt.zero_grad()
            residual = X_t - X_t @ W
            loss_recon = 0.5 / n * (residual ** 2).sum()
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
    ckpt[key] = {'d': d, 'h_history': [round(h,4) for h in h_hist],
                 'loss': [round(l,4) for l in loss_hist[::10]],
                 'final_edges': int(np.sum(np.abs(W.detach().cpu().numpy()) > 0.3))}
    with open(os.path.join(OUT, ckpt_name), 'w', encoding='utf-8') as f:
        json.dump(ckpt, f, indent=2)
    print(f'  [done] d={d}: {len(h_hist)} outer, h_final={h_hist[-1]:.2e}')

print('\nALL DONE')
for f in sorted(os.listdir(OUT)):
    if f.startswith('exp') and f.endswith('.json'):
        kb = os.path.getsize(os.path.join(OUT, f))/1024
        print(f'  {f}: {kb:.0f}KB')
