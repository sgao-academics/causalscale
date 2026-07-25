"""Run DAGMA benchmark on ER DAGs d=30-100, 5 seeds."""
import os
import numpy as np, json, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.metrics import f1_score
from dagma.linear import DagmaLinear

results = {}
for d in [30, 50, 80, 100]:
    n = 2*d; f1s=[]; edges=[]; times=[]
    for seed in range(5):
        np.random.seed(100*d + seed)
        W_true=np.zeros((d,d)); p=2/(d-1)
        for i in range(d):
            for j in range(d):
                if i!=j and np.random.random()<p:
                    W_true[j,i]=np.random.uniform(0.5,1.0)*np.random.choice([-1,1])
        for jj in range(d): W_true[jj,:jj]=0
        X=np.linalg.inv(np.eye(d)-W_true.T)@np.random.randn(d,n)
        X=X.T.astype(np.float32)
        t0=time.time()
        W_est=DagmaLinear(loss_type='l2').fit(X,lambda1=0.01)
        t1=time.time()
        times.append(t1-t0)
        W_bin=(np.abs(W_est)>0.3).astype(int)
        W_true_bin=(np.abs(W_true)>0).astype(int)
        f1=f1_score(W_true_bin.flatten(),W_bin.flatten())
        f1s.append(f1); edges.append(int(np.sum(W_bin)))
    f1s=np.array(f1s); edges=np.array(edges)
    results[f'd{d}']={
        'F1_mean':round(float(f1s.mean()),4),
        'F1_std':round(float(f1s.std()),4),
        'edges_mean':int(edges.mean()),
        'time_mean_s':round(float(np.mean(times)),1)
    }
    print(f'd={d}: F1={f1s.mean():.4f}+-{f1s.std():.4f}, edges={edges.mean():.0f}, time={np.mean(times):.1f}s')

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dagma_benchmark.json')
json.dump(results,open(out,'w'),indent=2)
print(f'Saved {out}')
