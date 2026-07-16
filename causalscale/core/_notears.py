"""Verified NOTEARS + CAGate backend - mirrors _ksweep_worker_v2"""
import torch, numpy as np, time
from sklearn.cluster import KMeans

def run_notears(X, device='cuda', lr=0.002, outer=40, inner=250, seed=42):
    """Standard NOTEARS with augmented Lagrangian.
    
    Args:
        X: (n, d) numpy array or torch tensor
    Returns:
        W: (d, d) adjacency matrix, edge_count, h_final, time_s
    """
    if isinstance(X, np.ndarray):
        X = torch.tensor(X.astype(np.float32))
    
    n, d = X.shape
    X = X.to(device)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    W = torch.zeros(d, d, requires_grad=True, device=device)
    rho, alpha = 1.0, 0.0
    opt = torch.optim.Adam([W], lr=lr)
    h_prev = float('inf')
    
    t0 = time.time()
    for o in range(outer):
        for _ in range(inner):
            opt.zero_grad()
            M = torch.eye(d, device=device) - W
            sq = (X @ M.T).pow(2)
            loss = sq.mean()
            h_val = torch.trace(torch.linalg.matrix_exp(W * W)) - d
            loss = loss + 0.5 * rho * h_val * h_val + alpha * h_val
            loss.backward()
            opt.step()
        
        with torch.no_grad():
            h_val = torch.trace(torch.linalg.matrix_exp(W * W)) - d
        h_curr = h_val.item()
        
        if o > 0 and h_curr > 0.25 * h_prev:
            rho *= 10
        else:
            alpha += rho * h_curr
        h_prev = h_curr
        
        if h_curr < 1e-8:
            break
    
    W_np = W.detach().cpu().numpy()
    edge_count = int(np.sum(np.abs(W_np) > 0.3))
    
    return W_np, edge_count, h_curr, time.time() - t0


def run_cagate(X, device='cuda', lr=0.002, outer=30, inner=200,
               K=8, n_seeds=3, seed=42):
    """CAGate: K-means split → per-cluster NOTEARS → union of edges.
    
    Args:
        X: (n, d) numpy array
        K: number of clusters for K-means
        n_seeds: seeds to run per cluster
    Returns:
        W: (d, d) union adjacency, edge_count, time_s
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    
    n, d = X.shape
    t0 = time.time()
    
    # K-means clustering
    km = KMeans(n_clusters=min(K, n // 10), random_state=seed, n_init='auto')
    labels = km.fit_predict(X)
    
    W_union = np.zeros((d, d))
    W_count = np.zeros((d, d))  # how many clusters found each edge
    for k in range(km.n_clusters):
        mask = labels == k
        X_k = X[mask]
        if X_k.shape[0] < 10:
            continue
        
        # Stack multiple seeds for stability
        Ws = []
        for s in range(n_seeds):
            W, ec, h, _ = run_notears(
                X_k, device=device, lr=lr, outer=outer, inner=inner,
                seed=seed + s * 1000 + k
            )
            Ws.append(W)
        
        # Per-cluster consensus: edge exists if majority of seeds find it
        for j in range(d):
            for i in range(d):
                if i != j:
                    vals = [w[i, j] for w in Ws if abs(w[i, j]) > 0.3]
                    if len(vals) > n_seeds // 2:
                        W_union[i, j] += np.mean(vals)
                        W_count[i, j] += 1
    
    # Cross-cluster consensus: edge must appear in >= half the clusters
    min_clusters = max(2, km.n_clusters // 3)
    for i in range(d):
        for j in range(d):
            if W_count[i, j] < min_clusters:
                W_union[i, j] = 0.0
            elif W_count[i, j] > 0:
                W_union[i, j] /= W_count[i, j]  # average across clusters
    
    edge_count = int(np.sum(np.abs(W_union) > 0.3))
    return W_union, edge_count, time.time() - t0
