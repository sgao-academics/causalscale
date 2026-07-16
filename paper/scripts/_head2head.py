"""
Head-to-head: causalscale cluster_aware vs raw NOTEARS on same synthetic DAGs.
Answers: is F1=0.44-0.58 actually low, or the ceiling for unsupervised causal discovery?
"""
import sys, numpy as np, time, torch
sys.path.insert(0, r"C:\Users\高帅东\Desktop\causalscale")
import causalscale as cs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

for d, n in [(20, 300), (30, 500), (50, 500)]:
    print(f"\n{'='*60}")
    print(f"  d={d}, n={n}, n/d={n/d:.1f}")
    print(f"{'='*60}")

    # Generate ER DAG
    np.random.seed(42)
    W_true = np.triu(np.random.binomial(1, 0.1, (d, d)), k=1).astype(float)
    for i in range(d):
        for j in range(i+1, d):
            if W_true[i, j] > 0:
                W_true[i, j] = np.random.uniform(0.3, 0.8)
    true_edges = int(np.sum(W_true > 0))

    # Generate data from DAG
    X = np.random.randn(n, d)
    for j in range(d):
        for p in range(j):
            if W_true[p, j] > 0:
                X[:, j] += W_true[p, j] * X[:, p]
        X[:, j] += 0.3 * np.random.randn(n)

    # Standardize
    X = (X - X.mean(0)) / X.std(0).clip(min=1e-8)
    print(f"  True edges: {true_edges}")

    # --- causalscale cluster_aware ---
    t0 = time.time()
    m = cs.CausalDiscovery(X, method="cluster_aware", device=DEVICE, verbose=False)
    m.fit(verbose=False)
    r_cs = m.validate(ground_truth=W_true, threshold=0.2, verbose=False)
    t_cs = time.time() - t0

    # --- Raw NOTEARS ---
    t0 = time.time()
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    W = torch.zeros((d, d), device=DEVICE, requires_grad=True)
    W.data = torch.randn(d, d, device=DEVICE) * 0.01

    def h_fn(W):
        return torch.trace(torch.matrix_exp(W * W)) - d

    rho, alpha = 1.0, 0.0
    opt = torch.optim.Adam([W], lr=0.002)
    for outer in range(30):
        for _ in range(200):
            opt.zero_grad()
            R = X_t - X_t @ W
            loss = 0.5/n*torch.sum(R**2) + 0.1*torch.sum(torch.abs(W))
            h = h_fn(W)
            total = loss + alpha*h + 0.5*rho*h*h
            total.backward()
            opt.step()
        with torch.no_grad():
            h_val = h_fn(W).item()
        if h_val < 1e-8:
            break
        alpha += rho * h_val
        rho = min(rho*5, 1e10)

    W_np = W.detach().cpu().numpy()
    W_bin = (np.abs(W_np) > 0.2).astype(int)
    W_true_bin = (W_true > 0).astype(int)
    tp = int(np.sum(W_bin & W_true_bin))
    fp = int(np.sum(W_bin & (1-W_true_bin)))
    fn = int(np.sum((1-W_bin) & W_true_bin))
    # remove diag
    tp -= int(np.sum(np.diag(W_bin) & np.diag(W_true_bin)))
    fp -= int(np.sum(np.diag(W_bin) & (1-np.diag(W_true_bin))))
    fn -= int(np.sum((1-np.diag(W_bin)) & np.diag(W_true_bin)))
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    f1_n = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
    t_nt = time.time() - t0
    nt_edges = int(W_bin.sum())

    print(f"  causalscale: F1={r_cs['f1']:.4f}, SHD={r_cs['shd']}, "
          f"Prec={r_cs['precision']:.4f}, Rec={r_cs['recall']:.4f}, "
          f"TP={r_cs['tp']}/{true_edges}, time={t_cs:.1f}s")
    print(f"  NOTEARS:     F1={f1_n:.4f}, SHD={fp+fn}, "
          f"Prec={prec:.4f}, Rec={rec:.4f}, "
          f"TP={tp}/{true_edges}, edges={nt_edges}, time={t_nt:.1f}s")
    print(f"  Delta F1: {r_cs['f1']-f1_n:+.4f} "
          f"({'causalscale wins' if r_cs['f1']>f1_n else 'NOTEARS wins'})")
