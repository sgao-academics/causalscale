# EXP12: Sample size scaling - n=5d vs n=2d
import sys, os, json, time, warnings, numpy as np, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, r'C:\Users\高帅东\Desktop\causalscale')
from causalscale.core._notears import run_notears

DEVICE = 'cuda'
OUT = r'D:\NO.1\causalscale_kdd2027_experiments'

def er_dag(d, seed=42):
    rng = np.random.RandomState(seed); W = np.zeros((d,d))
    for i in range(d):
        for j in range(i):
            if rng.rand() < 2/(d-1): W[i,j] = rng.choice([-1,1]) * rng.uniform(0.5, 1.0)
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

ckpt_name = 'exp12_n5d_scaling.json'
ckpt = json.load(open(os.path.join(OUT, ckpt_name), encoding='utf-8')) if os.path.exists(os.path.join(OUT, ckpt_name)) else {}
os.makedirs(OUT, exist_ok=True)

print('EXP12: Sample Size Scaling (n=2d vs n=5d)')
for d in [50, 80, 100, 150]:
    for seed in range(5):
        key = f'd{d}_s{seed}'
        if key in ckpt: print(f'  [skip] {key}'); continue
        W_true = er_dag(d, seed)
        X_2d = gen_data(W_true, 2*d)
        X_5d = gen_data(W_true, 5*d)

        t0 = time.time()
        W_2d, _, _, _ = run_notears(torch.tensor(X_2d, device=DEVICE), device=DEVICE, outer=30, inner=200, seed=seed)
        t_2d = time.time() - t0

        t0 = time.time()
        W_5d, _, _, _ = run_notears(torch.tensor(X_5d, device=DEVICE), device=DEVICE, outer=30, inner=200, seed=seed)
        t_5d = time.time() - t0

        m2 = metrics(W_true, W_2d); m5 = metrics(W_true, W_5d)
        ckpt[key] = {'d': d, 'seed': seed, 'n2d_f1': m2['f1'], 'n2d_shd': m2['shd'],
                     'n5d_f1': m5['f1'], 'n5d_shd': m5['shd'], 'delta_f1': round(m5['f1']-m2['f1'],4),
                     't_2d': round(t_2d,1), 't_5d': round(t_5d,1)}
        with open(os.path.join(OUT, ckpt_name), 'w', encoding='utf-8') as f:
            json.dump(ckpt, f, indent=2)
        print(f'  [done] {key}: n=2d F1={m2["f1"]}, n=5d F1={m5["f1"]}, delta={m5["f1"]-m2["f1"]:+.4f}')

# Summary
print('\n=== Summary ===')
for d in [50,80,100,150]:
    d2 = [v['n2d_f1'] for k,v in ckpt.items() if v['d']==d]
    d5 = [v['n5d_f1'] for k,v in ckpt.items() if v['d']==d]
    if d2:
        print(f'  d={d}: n=2d F1={np.mean(d2):.4f}+-{np.std(d2):.4f}, n=5d F1={np.mean(d5):.4f}+-{np.std(d5):.4f}, gain={np.mean(d5)-np.mean(d2):+.4f}')
print('DONE')
