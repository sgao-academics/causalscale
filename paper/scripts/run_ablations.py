"""Run real experiments to fill paper gaps.
1. Rank sensitivity: LowRankGNN r sweep at d=500
2. DAGMA d=150 5-seed benchmark
"""
import os
import time, json, numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, causalscale as cs
from sklearn.metrics import f1_score

def gen_sem_data(d, n, seed=42):
    np.random.seed(seed)
    W = np.zeros((d,d)); p=2/(d-1)
    for i in range(d):
        for j in range(d):
            if i!=j and np.random.random()<p:
                W[j,i]=np.random.uniform(0.5,1.0)*np.random.choice([-1,1])
    for j in range(d): W[j,:j]=0
    X=np.linalg.inv(np.eye(d)-W.T)@np.random.randn(d,n)*0.5
    return X.T.astype(np.float32), W

print("=== Experiment 1: Rank Sensitivity (LowRankGNN, d=500) ===\n")
d, n = 500, 1000
X, W_true = gen_sem_data(d, n)
X = (X - X.mean(0))/(X.std(0)+1e-8)
results_rank = []

for r in [8, 16, 32, 64, 128]:
    print(f"  r={r:3d}... ", end='', flush=True)
    torch.cuda.empty_cache()
    t0 = time.time()
    try:
        res = cs.train_lowrank_gnn(X, rank=r, epochs=300, lr=0.01, device='cuda', verbose=False)
        t1 = time.time()
        mem = torch.cuda.max_memory_allocated()/1e9
        entry = {'r':r, 'time_s':round(t1-t0,1), 'vram_gb':round(mem,2),
                 'gnn_edges':res['gnn_edges'], 'f1_corr':res['f1']}
        results_rank.append(entry)
        print(f"{res['gnn_edges']} edges, F1_corr={res['f1']:.3f}, {t1-t0:.1f}s, {mem:.2f}GB")
    except Exception as e:
        print(f"Error: {e}")

print("\n=== Experiment 2: DAGMA d=150, 5 seeds ===\n")
from dagma.linear import DagmaLinear
d, n = 150, 300
f1s, times = [], []
for seed in range(5):
    print(f"  seed={seed}... ", end='', flush=True)
    X, W_true = gen_sem_data(d, n, seed=100+seed)
    X = X.astype(np.float32)
    t0 = time.time()
    W_est = DagmaLinear(loss_type='l2').fit(X, lambda1=0.01)
    t1 = time.time()
    W_bin = (np.abs(W_est)>0.3).astype(int)
    W_tbin = (np.abs(W_true)>0).astype(int)
    f1 = f1_score(W_tbin.flatten(), W_bin.flatten())
    f1s.append(f1); times.append(t1-t0)
    print(f"F1={f1:.4f}, {t1-t0:.1f}s")
f1s=np.array(f1s); times=np.array(times)
print(f"  Mean: F1={f1s.mean():.4f}+-{f1s.std():.4f}, time={times.mean():.1f}+-{times.std():.1f}s")

# Save
json.dump({'rank_sensitivity': results_rank,
           'dagma_d150': {'f1_mean':round(float(f1s.mean()),4), 'f1_std':round(float(f1s.std()),4),
                          'time_mean':round(float(times.mean()),1), 'time_std':round(float(times.std()),1)}},
          open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ablation_results.json'),'w'), indent=2)
print("\nSaved ablation_results.json")
