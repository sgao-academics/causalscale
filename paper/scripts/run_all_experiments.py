# Full experiment matrix for KDD 2027 - causalscale paper
# Checkpointing: never re-run completed experiments
import sys, os, json, time, warnings, numpy as np, torch, pandas as pd
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
def sf_dag(d, seed=42):
    rng = np.random.RandomState(seed)
    W, deg = np.zeros((d,d)), np.ones(d)
    for i in range(1,d):
        probs = deg[:i]/deg[:i].sum()
        n_e = max(1, rng.poisson(2))
        for p in rng.choice(i, size=min(n_e,i), replace=False, p=probs):
            W[i,p] = rng.choice([-1,1]) * rng.uniform(0.5, 1.0)
    return W
def gen_data(W, n):
    X = np.linalg.inv(np.eye(W.shape[0]) - W) @ np.random.randn(W.shape[0], n)
    return X.T.astype(np.float32)
def metrics(W_true, W_est, tau=0.3):
    mt = np.abs(W_true) > 0; me = np.abs(W_est) > tau
    tp = np.sum(mt & me); fp = np.sum(~mt & me); fn = np.sum(mt & ~me)
    p = tp/(tp+fp) if (tp+fp)>0 else 0; r = tp/(tp+fn) if (tp+fn)>0 else 0
    f = 2*p*r/(p+r) if (p+r)>0 else 0
    return {'f1': round(f,4), 'shd': int(fp+fn), 'prec': round(p,4), 'rec': round(r,4), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}

# ============================================================
# EXP1: causalscale on Synthetic ER (6 dims x 5 seeds)
# ============================================================
print('EXP1: causalscale Synthetic ER Benchmark')
ckpt = load('exp1_causalscale_er.json')
for d in [30,50,80,100,150,200]:
    for seed in range(5):
        key = f'd{d}_s{seed}'
        if key in ckpt: print(f'  [skip] {key}'); continue
        W_true = er_dag(d, seed); X = gen_data(W_true, 2*d)
        t0 = time.time()
        W, ec, h, _ = run_notears(X, device=DEVICE, outer=30, inner=200, seed=seed)
        elapsed = time.time() - t0
        ckpt[key] = {'d':d, 'seed':seed, 'true_edges': int(np.sum(np.abs(W_true)>0)), **metrics(W_true, W), 'h': round(h,2), 'cs_edges': ec, 'time': round(elapsed,1)}
        save('exp1_causalscale_er.json', ckpt)
        print(f'  [done] {key}: f1={ckpt[key]["f1"]}, {elapsed:.0f}s')

# ============================================================
# EXP2: causalscale on Scale-Free (2 dims x 3 seeds)
# ============================================================
print('EXP2: causalscale Scale-Free Topology')
ckpt = load('exp2_causalscale_sf.json')
for d in [50, 100]:
    for seed in range(3):
        key = f'd{d}_s{seed}'
        if key in ckpt: print(f'  [skip] {key}'); continue
        W_true = sf_dag(d, seed); X = gen_data(W_true, 2*d)
        t0 = time.time()
        W, ec, h, _ = run_notears(X, device=DEVICE, outer=30, inner=200, seed=seed)
        ckpt[key] = {'d':d, 'seed':seed, 'true_edges': int(np.sum(np.abs(W_true)>0)), **metrics(W_true, W), 'time': round(time.time()-t0,1)}
        save('exp2_causalscale_sf.json', ckpt)
        print(f'  [done] {key}: f1={ckpt[key]["f1"]}')

# ============================================================
# EXP3: Tau Sensitivity (d=50,100 x 3 seeds x 8 taus)
# ============================================================
print('EXP3: Tau Sensitivity')
ckpt = load('exp3_tau_sensitivity.json')
for d in [50, 100]:
    for seed in range(3):
        base_key = f'd{d}_s{seed}'
        # Check if all taus are done
        all_done = all(f'{base_key}_t{tau:.2f}' in ckpt for tau in [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5])
        if all_done: print(f'  [skip] {base_key} (all taus)'); continue
        W_true = er_dag(d, seed); X = gen_data(W_true, 2*d)
        W, ec, h, _ = run_notears(X, device=DEVICE, outer=30, inner=200, seed=seed)
        for tau in [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5]:
            tk = f'{base_key}_t{tau:.2f}'
            if tk in ckpt: continue
            ckpt[tk] = {'d':d, 'seed':seed, 'tau':tau, **metrics(W_true, W, tau=tau)}
            save('exp3_tau_sensitivity.json', ckpt)
        print(f'  [done] {base_key}: all 8 taus')

# ============================================================
# EXP4: Multi-Seed Pan-Cancer (33 cancers x 3 seeds)
# ============================================================
print('EXP4: Multi-Seed Pan-Cancer')
ckpt = load('exp4_pancancer_3seed.json')
TSD = r'D:\NO.1\cancer_application\data'
for fname in sorted(os.listdir(TSD)):
    if not fname.startswith('TCGA_') or not fname.endswith('.tsv') or 'HiSeqV2' not in fname: continue
    cancer = fname.replace('TCGA_','').replace('_HiSeqV2.tsv','')
    # Check if already have 3 seeds
    if cancer in ckpt and len(ckpt[cancer].get('seeds',{})) >= 3:
        print(f'  [skip] {cancer}'); continue
    df = pd.read_csv(os.path.join(TSD, fname), sep='\t', index_col=0, nrows=None)
    # Fix orientation: ensure rows=samples, columns=genes
    if df.index.name != 'sample' or not str(df.index[0]).startswith('TCGA-'):
        df = df.T  # genes were rows, flip to samples as rows
    # Drop any non-gene columns that may have been read
    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64']).columns
    df = df[numeric_cols]
    targets = ['ARID1A','MTOR']
    present = [g for g in targets if g in df.columns]
    var_genes = df.var().nlargest(100).index.tolist()
    selected = list(dict.fromkeys(present + [g for g in var_genes if g not in targets][:100]))
    X = df[selected].values.astype(np.float32)
    seeds_data = ckpt.get(cancer, {}).get('seeds', {})
    for seed in range(3):
        sk = str(seed)
        if sk in seeds_data: continue
        t0 = time.time()
        W, ec, h, _ = run_notears(torch.tensor(X, device=DEVICE), device=DEVICE, outer=30, inner=200, seed=seed)
        ai = selected.index('ARID1A') if 'ARID1A' in selected else -1
        mi = selected.index('MTOR') if 'MTOR' in selected else -1
        seeds_data[sk] = {'edges': ec, 'a2m': round(float(W[ai,mi]),4) if ai>=0 and mi>=0 else 0,
                          'm2a': round(float(W[mi,ai]),4) if ai>=0 and mi>=0 else 0,
                          'time': round(time.time()-t0,1), 'h': round(h,2)}
    ckpt[cancer] = {'n': X.shape[0], 'd': X.shape[1], 'seeds': seeds_data}
    save('exp4_pancancer_3seed.json', ckpt)
    edges = [v['edges'] for v in seeds_data.values()]
    print(f'  [done] {cancer} (n={X.shape[0]}): edges={edges}')

# ============================================================
# EXP5: CPU vs GPU Speedup (d=50,80,100)
# ============================================================
print('EXP5: CPU vs GPU Speedup')
ckpt = load('exp5_cpu_vs_gpu.json')
for d in [50, 80, 100]:
    key = f'd{d}'
    if key in ckpt: print(f'  [skip] {key}'); continue
    W_true = er_dag(d); X = gen_data(W_true, 2*d)
    X_gpu = torch.tensor(X, device='cuda')
    # GPU (warmup first)
    t0 = time.time(); run_notears(X_gpu, device='cuda', outer=30, inner=200, seed=0)
    gpu_t = time.time() - t0
    # CPU
    t0 = time.time(); run_notears(X_gpu.cpu(), device='cpu', outer=30, inner=200, seed=0)
    cpu_t = time.time() - t0
    ckpt[key] = {'d': d, 'gpu_s': round(gpu_t,1), 'cpu_s': round(cpu_t,1), 'speedup': round(cpu_t/gpu_t,1)}
    save('exp5_cpu_vs_gpu.json', ckpt)
    print(f'  [done] d={d}: GPU={gpu_t:.1f}s, CPU={cpu_t:.1f}s, {cpu_t/gpu_t:.1f}x')

# ============================================================
# EXP6: Full Metrics Report
# ============================================================
print('\nEXP6: Summary Report')
print('='*60)
for name in sorted(os.listdir(OUT)):
    if not name.endswith('.json'): continue
    d = load(name)
    kb = os.path.getsize(os.path.join(OUT, name))/1024
    n = len(d) if isinstance(d, dict) else 'list'
    print(f'  {name}: {kb:.0f}KB, {n} entries')
print('='*60)
print('DONE. All results in', OUT)
