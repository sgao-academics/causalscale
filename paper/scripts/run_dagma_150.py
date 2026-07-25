"""Quick DAGMA d=150 test - does it collapse?"""
import numpy as np, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.metrics import f1_score
from dagma.linear import DagmaLinear

d, n = 150, 300
np.random.seed(42)
W_true=np.zeros((d,d)); p=2/(d-1)
for i in range(d):
    for j in range(d):
        if i!=j and np.random.random()<p:
            W_true[j,i]=np.random.uniform(0.5,1.0)*np.random.choice([-1,1])
for jj in range(d): W_true[jj,:jj]=0
X=np.linalg.inv(np.eye(d)-W_true.T)@np.random.randn(d,n); X=X.T.astype(np.float32)

t0=time.time()
W_est=DagmaLinear(loss_type='l2').fit(X,lambda1=0.01)
t1=time.time()
W_bin=(np.abs(W_est)>0.3).astype(int)
W_true_bin=(np.abs(W_true)>0).astype(int)
f1=f1_score(W_true_bin.flatten(),W_bin.flatten())
n_edges=int(np.sum(W_bin))
print(f'd=150: F1={f1:.4f}, edges={n_edges}, time={t1-t0:.1f}s')
