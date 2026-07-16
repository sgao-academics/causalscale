# EXP9: LowRankGNN Scaling Benchmark (FIXED)
import sys, os, json, time, warnings, numpy as np, torch
warnings.filterwarnings('ignore')
DEVICE = 'cuda'; OUT = r'D:\NO.1\causalscale_kdd2027_experiments'

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

ckpt_name = 'exp9_lowrank_scaling.json'
ckpt = json.load(open(os.path.join(OUT, ckpt_name), encoding='utf-8')) if os.path.exists(os.path.join(OUT, ckpt_name)) else {}

# Clear bad entries (F1=0)
bad = [k for k,v in ckpt.items() if v.get('f1',0) == 0]
for k in bad: del ckpt[k]
if bad: print(f'Cleared {len(bad)} broken entries: {bad}')

for d in [500, 1000, 2000, 5000]:
    for seed in range(3):
        key = f'd{d}_s{seed}'
        if key in ckpt: print(f'[skip] {key}'); continue

        W_true = er_dag(d, seed); n = max(500, d)
        X = gen_data(W_true, n)
        X_t = torch.tensor(X.astype(np.float32), device=DEVICE)
        r = min(64, d // 4)

        # FIX1: Larger init so W has meaningful values
        U = torch.randn(d, r, device=DEVICE) * 0.1
        V = torch.randn(d, r, device=DEVICE) * 0.1
        U.requires_grad_(True); V.requires_grad_(True)
        opt = torch.optim.Adam([U, V], lr=0.002)
        rho, alpha, rho_max = 1.0, 0.0, 1e12  # FIX2: rho_max to 1e12
        l1_weight = 0.5 / d  # FIX3: adaptive L1

        t0 = time.time()
        outer_done = 0
        for o in range(30):
            for _ in range(200):
                opt.zero_grad()
                W = U @ V.T
                residual = X_t - X_t @ W
                loss_recon = 0.5 / n * (residual ** 2).sum()
                M = W * W
                h_val = torch.trace(torch.linalg.matrix_exp(M)) - d
                # L1 on W directly
                l1 = l1_weight * W.abs().sum()
                loss = loss_recon + 0.5 * rho * h_val * h_val + alpha * h_val + l1
                loss.backward()
                torch.nn.utils.clip_grad_norm_([U, V], 10.0)
                opt.step()

            with torch.no_grad():
                W_check = U @ V.T
                M_check = W_check * W_check
                h_curr = float(torch.trace(torch.linalg.matrix_exp(M_check)) - d)

            outer_done = o + 1
            if h_curr > 0.25 * h_val.item():
                rho = min(rho * 10, rho_max)
            alpha = alpha + rho * h_curr

            # FIX4: Only early-stop when really converged
            if abs(h_curr) < 1e-8:
                break

        elapsed = time.time() - t0
        W_final = (U @ V.T).detach().cpu().numpy()
        m = metrics(W_true, W_final)

        ckpt[key] = {'d': d, 'seed': seed, 'true_edges': int(np.sum(np.abs(W_true)>0)),
                     **m, 'edges': int(np.sum(np.abs(W_final)>0.3)),
                     'time': round(elapsed,1), 'outer_iters': outer_done}
        with open(os.path.join(OUT, ckpt_name), 'w', encoding='utf-8') as f:
            json.dump(ckpt, f, indent=2)
        print(f'[done] {key}: f1={m["f1"]}, edges={int(np.sum(np.abs(W_final)>0.3))}, {elapsed:.0f}s ({outer_done} outer)')

print('EXP9 DONE')
