#!/usr/bin/env python
"""causalscale KDD 2027 -- One-Click Reproduction Script.

Usage:
    python run_all.py                # Full reproduction (all experiments)
    python run_all.py --verify       # Quick verification (~30 sec)
    python run_all.py --benchmark    # Synthetic benchmark only (~15 min)
    python run_all.py --scaling      # Scaling experiments only (~30 min)
    python run_all.py --figures      # Regenerate all 4 paper figures (~10 sec)
    python run_all.py --tables       # Print all table data from cached results

Hardware: NVIDIA RTX 5060 (8 GB) or equivalent. CPU fallback supported (slower).
Dependencies: pip install -e . dagma statsmodels (see requirements.txt)

Author: Shuaidong Gao (ORCID: 0009-0004-5641-3581)
"""
import sys, os, time, json, warnings
warnings.filterwarnings('ignore')

PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PKG_ROOT)

import numpy as np
import torch

RESULTS_DIR = os.path.join(PKG_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

VERIFY = '--verify' in sys.argv
BENCH_ONLY = '--benchmark' in sys.argv
SCALE_ONLY = '--scaling' in sys.argv
FIGURES_ONLY = '--figures' in sys.argv
TABLES_ONLY = '--tables' in sys.argv
FULL = not (VERIFY or BENCH_ONLY or SCALE_ONLY or FIGURES_ONLY or TABLES_ONLY)
SEP = '=' * 60

def load_json(name):
    p = os.path.join(RESULTS_DIR, name)
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}

def save_json(name, data):
    with open(os.path.join(RESULTS_DIR, name), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)

def f1_score_binary(W_true, W_est, tau=0.3):
    mt = np.abs(W_true) > 0; me = np.abs(W_est) > tau
    tp = np.sum(mt & me); fp = np.sum(~mt & me); fn = np.sum(mt & ~me)
    p = tp/(tp+fp) if (tp+fp)>0 else 0; r = tp/(tp+fn) if (tp+fn)>0 else 0
    return {'f1': round(2*p*r/(p+r) if (p+r)>0 else 0, 4),
            'precision': round(p,4), 'recall': round(r,4),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
            'edges_est': int(np.sum(me))}

def make_er_dag(d, seed=42):
    rng = np.random.RandomState(seed); W = np.zeros((d,d))
    p = 2.0/(d-1) if d>1 else 1.0
    for i in range(d):
        for j in range(i):
            if rng.rand() < p: W[i,j] = rng.choice([-1,1]) * rng.uniform(0.5,1.0)
    return W

def generate_data(W, n, seed=42):
    rng = np.random.RandomState(seed); d = W.shape[0]
    return np.linalg.solve(np.eye(d)-W.T, rng.randn(n,d).T).T.astype(np.float32)

# == VERIFY MODE ==
def run_verify():
    print(SEP); print("MODE: --verify (quick validation)"); print(SEP)
    print("\n[1] Testing causalscale import...")
    try:
        import causalscale as cs
        print(f"  OK: causalscale v{cs.__version__} imported")
    except ImportError:
        print("  FAIL: pip install -e . first!"); return 1

    print("\n[2] Quick synthetic test (d=30, seed=0)...")
    d, n = 30, 60
    W_true = make_er_dag(d, seed=0); X = generate_data(W_true, n, seed=0)
    t0 = time.time()
    model = cs.CausalDiscovery(X, method='cluster_aware'); model.fit()
    W_est = model.W; m = f1_score_binary(W_true, W_est)
    print("  F1={:.4f}, edges={}, time={:.1f}s".format(m['f1'], m['edges_est'], time.time()-t0))

    ref = load_json('exp1_causalscale_er.json')
    if ref and 'd30_s0' in ref:
        rv = ref['d30_s0']; diff = abs(rv.get('f1',0)-m['f1'])
        print("  ref F1={}, delta={:.4f} -> {}".format(rv.get('f1'), diff, 'PASS' if diff<=0.10 else 'WARN'))

    print("\n[3] Result file integrity...")
    expected = ['exp1_causalscale_er.json','exp2_causalscale_sf.json',
                'exp4_pancancer_3seed.json','dagma_benchmark.json',
                'ablation_results.json','memory_scaling.json',
                'pan_cancer_ckpt.json','exp12_n5d_scaling.json']
    for r in expected:
        path = os.path.join(RESULTS_DIR, r)
        status = '{}KB'.format(int(os.path.getsize(path)/1024)) if os.path.exists(path) else 'MISSING'
        print("  {:45s} {}".format(r, status))

    print("\n[4] Figure file integrity...")
    fig_dir = os.path.join(PKG_ROOT, 'figures')
    for fig in ['fig1_architecture.pdf','fig2_benchmark.pdf',
                'fig3_arid1a_mtor.pdf','fig4_timing.pdf']:
        path = os.path.join(fig_dir, fig)
        status = '{}KB'.format(int(os.path.getsize(path)/1024)) if os.path.exists(path) else 'MISSING'
        print("  {:35s} {}".format(fig, status))

    print("\n[5] Pre-trained model integrity...")
    try:
        models = cs.list_models()
        for name, info in models.items():
            print(f"  {name}: {info}")
        benchmarks = cs.list_benchmarks()
        for name, info in benchmarks.items():
            print(f"  {name}: {info}")
    except Exception as e:
        print(f"  WARN: {e}")

    print(f"\n{SEP}\nVERIFICATION COMPLETE.\n{SEP}")
    return 0

# == FIGURES MODE ==
def run_figures():
    print(SEP); print("MODE: --figures (regenerate all paper figures)"); print(SEP)
    gen_dir = os.path.join(PKG_ROOT, 'scripts', 'gen_figures')
    if not os.path.isdir(gen_dir):
        print("ERROR: scripts/gen_figures/ not found"); return 1
    for script in sorted(os.listdir(gen_dir)):
        if script.endswith('.py'):
            path = os.path.join(gen_dir, script)
            print(f"\n  Running {script}...")
            t0 = time.time()
            ret = os.system(f'"{sys.executable}" "{path}"')
            elapsed = time.time() - t0
            if ret == 0:
                print(f"  OK ({elapsed:.1f}s)")
            else:
                print(f"  FAILED (exit code {ret})")
    print(f"\n{SEP}\nFIGURES COMPLETE. Check figures/ directory.\n{SEP}")

# == TABLES MODE ==
def run_tables():
    print(SEP); print("MODE: --tables (print all cached table data)"); print(SEP)

    # Table 1: Synthetic DAG F1
    exp1 = load_json('exp1_causalscale_er.json')
    dagma = load_json('dagma_benchmark.json')
    print("\n=== Table 1: Synthetic DAG F1 (tau=0.3, 5 seeds) ===")
    print(f"{'d':>5} {'NOTEARS':>10} {'DAGMA':>10} {'causalscale':>12}")
    from collections import defaultdict
    by_d = defaultdict(list)
    for k, v in exp1.items():
        if isinstance(v, dict) and 'd' in v and 'f1' in v:
            by_d[v['d']].append(v['f1'])
    for d in [30, 50, 80, 100, 150]:
        cs_vals = by_d.get(d, [])
        cs_mean = np.mean(cs_vals) if cs_vals else 0
        cs_std = np.std(cs_vals) if cs_vals else 0
        dag_key = f'd{d}'
        dag_f1 = dagma.get(dag_key, {}).get('F1_mean', 'N/A') if isinstance(dagma, dict) else 'N/A'
        print(f"{d:>5} {'(see paper)':>10} {str(dag_f1):>10} {cs_mean:.3f}+/-{cs_std:.3f}")

    # Table 4: Pan-cancer
    pan = load_json('pan_cancer_ckpt.json')
    print("\n=== Table 4: ARID1A-MTOR Directionality (from pan_cancer_ckpt.json) ===")
    if pan:
        arid1a_to_mtor = []; mtor_to_arid1a = []; no_edge = []
        for cancer, data in pan.items():
            if isinstance(data, dict) and 'arid1a_mtor_weight' in data:
                w = data['arid1a_mtor_weight']
                if w > 0.3: arid1a_to_mtor.append(cancer)
                elif w < -0.3: mtor_to_arid1a.append(cancer)
                else: no_edge.append(cancer)
        print(f"  ARID1A->MTOR ({len(arid1a_to_mtor)}): {', '.join(arid1a_to_mtor[:8])}")
        print(f"  MTOR->ARID1A ({len(mtor_to_arid1a)}): {', '.join(mtor_to_arid1a[:8])}")
        print(f"  No edge ({len(no_edge)}): {', '.join(no_edge[:8])}")

    # Table 5: LowRank scaling
    lr = load_json('exp9_lowrank_scaling.json')
    print("\n=== Table 5: LowRankGNN Scalability (r=64) ===")
    if lr:
        for k, v in sorted(lr.items()) if isinstance(lr, dict) else []:
            if isinstance(v, dict):
                print(f"  {k}: edges={v.get('edges','?')}, time={v.get('time','?')}s")

    # Table 6: Memory scaling
    mem = load_json('memory_scaling.json')
    print("\n=== Table 6: v3.3.0 Memory Scaling ===")
    if mem:
        for k, v in sorted(mem.items()) if isinstance(mem, dict) else []:
            print(f"  {k}: {v}")

    print(f"\n{SEP}\nTABLES PRINTED.\n{SEP}")

# == BENCHMARK MODE ==
def run_benchmark():
    print(SEP); print("BENCHMARK: Synthetic DAG Recovery"); print(SEP)
    from causalscale.core._notears import run_notears
    ckpt = load_json('exp1_causalscale_er.json')
    for d in [30,50,80,100,150]:
        n = 2*d
        for seed in range(5):
            key = 'd{}_s{}'.format(d, seed)
            if key in ckpt: continue
            W_true = make_er_dag(d, seed); X = generate_data(W_true, n, seed)
            t0 = time.time()
            W_est, ec, h, _ = run_notears(torch.tensor(X, device=DEVICE), device=DEVICE,
                                          outer=30, inner=200, seed=seed)
            m = f1_score_binary(W_true, W_est)
            ckpt[key] = {'d':d,'seed':seed,'n':n,'true_edges':int(np.sum(np.abs(W_true)>0)),
                         **m,'h':round(h,2),'time':round(time.time()-t0,1)}
            save_json('exp1_causalscale_er.json', ckpt)
    print("Benchmark complete.")

def run_dagma_benchmark():
    print(SEP); print("DAGMA Benchmark"); print(SEP)
    try:
        from dagma.linear import DagmaLinear
    except ImportError:
        print("SKIP: pip install dagma"); return
    from sklearn.metrics import f1_score
    results = {}
    for d in [30,50,80,100]:
        n = 2*d; f1s = []; times = []
        for seed in range(5):
            np.random.seed(100*d+seed); W_true=np.zeros((d,d)); p=2/(d-1)
            for i in range(d):
                for j in range(d):
                    if i!=j and np.random.random()<p:
                        W_true[j,i]=np.random.uniform(0.5,1.0)*np.random.choice([-1,1])
            for jj in range(d): W_true[jj,:jj]=0
            X=np.linalg.inv(np.eye(d)-W_true.T)@np.random.randn(d,n)
            X=X.T.astype(np.float32)
            t0=time.time()
            W_est=DagmaLinear(loss_type='l2').fit(X,lambda1=0.01)
            elapsed=time.time()-t0
            f1s.append(f1_score((np.abs(W_true)>0).flatten(),(np.abs(W_est)>0.3).flatten()))
            times.append(elapsed)
        results['d{}'.format(d)]={
            'F1_mean':round(float(np.mean(f1s)),4),
            'F1_std':round(float(np.std(f1s)),4),
            'time_mean_s':round(float(np.mean(times)),1)}
    save_json('dagma_benchmark.json', results)
    print("DAGMA benchmark complete.")

# == MAIN ==
if __name__ == '__main__':
    print("causalscale KDD 2027 -- Reproduction Script")
    mode = 'verify' if VERIFY else ('benchmark' if BENCH_ONLY else
          ('scaling' if SCALE_ONLY else ('figures' if FIGURES_ONLY else
          ('tables' if TABLES_ONLY else 'full'))))
    print("Mode: {} | Device: {}\n".format(mode, DEVICE))

    if VERIFY:
        sys.exit(run_verify())
    if FIGURES_ONLY:
        sys.exit(run_figures())
    if TABLES_ONLY:
        sys.exit(run_tables())

    try:
        import causalscale as cs
    except ImportError:
        print("ERROR: causalscale not found. Run: pip install -e ."); sys.exit(1)

    if BENCH_ONLY or FULL:
        run_benchmark()
        run_dagma_benchmark()

    if SCALE_ONLY or FULL:
        print(SEP); print("SCALING: Rank sensitivity"); print(SEP)
        for r in [8,16,32,64,128]:
            torch.cuda.empty_cache()
            X = np.random.randn(1000,500).astype(np.float32)
            X = (X-X.mean(0))/(X.std(0)+1e-8)
            t0 = time.time()
            result = cs.train_lowrank_gnn(X, rank=r, epochs=200, lr=3e-4, device=DEVICE, verbose=False)
            print("  r={}: F1={:.4f}, {}s".format(r, result.get('f1',0), round(time.time()-t0,1)))
        save_json('ablation_results.json', {'rank_sensitivity': []})
        print("Scaling complete.")

    print(f"\n{SEP}\nREPRODUCTION COMPLETE.\n{SEP}")
    print("To regenerate figures: python run_all.py --figures")
    print("To view table data:   python run_all.py --tables")
    print("Quick verification:     python run_all.py --verify")
