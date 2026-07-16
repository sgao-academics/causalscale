"""
Tune internal NOTEARS solver: L1 penalty, random init, more iters.
"""
import sys, numpy as np, torch, time
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")

np.random.seed(42)
d, n = 50, 500
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Generate data
W_true = np.triu(np.random.binomial(1, 0.1, (d, d)), k=1).astype(float)
for i in range(d):
    for j in range(i+1, d):
        if W_true[i, j] > 0:
            W_true[i, j] = np.random.uniform(0.3, 0.8)
true_edges = int(np.sum(W_true > 0))

X = np.random.randn(n, d)
for j in range(d):
    for p in range(j):
        if W_true[p, j] > 0:
            X[:, j] += W_true[p, j] * X[:, p]
    X[:, j] += 0.3 * np.random.randn(n)
X = (X - X.mean(0)) / X.std(0).clip(min=1e-8)

def f1(W_pred, th=0.2):
    Wb = (np.abs(W_pred) > th).astype(int)
    Wt = (np.abs(W_true) > 0).astype(int)
    tp = int(np.sum(Wb & Wt)) - int(np.sum(np.diag(Wb) & np.diag(Wt)))
    fp = int(np.sum(Wb & (1-Wt))) - int(np.sum(np.diag(Wb) & (1-np.diag(Wt))))
    fn = int(np.sum((1-Wb) & Wt)) - int(np.sum((1-np.diag(Wb)) & np.diag(Wt)))
    p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
    return 2*p*r/(p+r) if p+r>0 else 0, p, r, tp

def best_f1(W):
    bf = 0
    for th in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        f,_,_,_ = f1(W, th)
        if f > bf: bf = f
    return bf

def notears_tuned(X_t, l1=0.0, init='zero', outer=30, inner=200, lr=0.002):
    n2, d2 = X_t.shape
    W = torch.zeros(d2, d2, requires_grad=True, device=DEV)
    if init == 'rand':
        W.data = torch.randn(d2, d2, device=DEV) * 0.01
    elif init == 'eye':
        W.data = torch.eye(d2, device=DEV) * 0.1

    rho, alpha = 1.0, 0.0
    opt = torch.optim.Adam([W], lr=lr)
    h_prev = float('inf')

    for o in range(outer):
        for _ in range(inner):
            opt.zero_grad()
            M = torch.eye(d2, device=DEV) - W
            sq = (X_t @ M.T).pow(2)
            loss = sq.mean()
            if l1 > 0:
                loss = loss + l1 * torch.sum(torch.abs(W))
            h_val = torch.trace(torch.linalg.matrix_exp(W * W)) - d2
            loss = loss + 0.5 * rho * h_val * h_val + alpha * h_val
            loss.backward()
            opt.step()

        with torch.no_grad():
            h_val = torch.trace(torch.linalg.matrix_exp(W * W)) - d2
        h_curr = h_val.item()
        if o > 0 and h_curr > 0.25 * h_prev:
            rho *= 10
        else:
            alpha += rho * h_curr
        h_prev = h_curr
        if h_curr < 1e-8:
            break

    return W.detach().cpu().numpy()

X_t = torch.tensor(X, dtype=torch.float32, device=DEV)
print(f"True edges: {true_edges}")

# Baseline (current defaults)
W_base = notears_tuned(X_t, l1=0, init='zero', outer=30, inner=200)
print(f"\n  Baseline (zero init, no L1, 30x200):     F1={best_f1(W_base):.4f}")

# L1 penalty
for l1 in [0.01, 0.03, 0.05, 0.1, 0.15]:
    W = notears_tuned(X_t, l1=l1, init='zero', outer=30, inner=200)
    f = best_f1(W)
    print(f"  +L1={l1:.2f} (zero init, 30x200):         F1={f:.4f}")

# Random init
W = notears_tuned(X_t, l1=0, init='rand', outer=30, inner=200)
print(f"  Random init (no L1, 30x200):             F1={best_f1(W):.4f}")
W = notears_tuned(X_t, l1=0.05, init='rand', outer=30, inner=200)
print(f"  Random init + L1=0.05 (30x200):          F1={best_f1(W):.4f}")

# More iterations
W = notears_tuned(X_t, l1=0, init='zero', outer=50, inner=300)
print(f"  More iters (zero, no L1, 50x300):        F1={best_f1(W):.4f}")
W = notears_tuned(X_t, l1=0.05, init='zero', outer=50, inner=300)
print(f"  More iters + L1=0.05 (zero, 50x300):     F1={best_f1(W):.4f}")

# Random init + more iters + L1
W = notears_tuned(X_t, l1=0.05, init='rand', outer=50, inner=300)
print(f"  Rand init + L1=0.05 + 50x300:            F1={best_f1(W):.4f}")

# Best combination sweep
print(f"\n  Searching best config...")
best_f, best_cfg = 0, None
for init in ['rand', 'zero']:
    for l1 in [0, 0.02, 0.05, 0.08, 0.1]:
        for (o,i) in [(30,200),(40,250),(50,300)]:
            W = notears_tuned(X_t, l1=l1, init=init, outer=o, inner=i)
            f = best_f1(W)
            if f > best_f:
                best_f = f
                best_cfg = (init, l1, o, i)

print(f"  BEST: init={best_cfg[0]}, L1={best_cfg[1]}, "
      f"{best_cfg[2]}x{best_cfg[3]}, F1={best_f:.4f}")
