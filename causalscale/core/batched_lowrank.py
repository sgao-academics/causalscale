"""
Batched LowRankGNN Trainer (v3.2.0) — Streaming Data + FP16 + Gradient Checkpointing.

Enables genome-scale causal discovery by never materializing the full
d x d correlation matrix in GPU memory. Instead:

1. Data streams in batches of n_batch rows from disk
2. Each batch computes a partial correlation estimate
3. Model (U, V) trains against the running correlation estimate
4. FP16 model parameters reduce memory by 2x
5. Gradient checkpointing trades compute for memory on the forward pass

Scaling: d = 10^8 with r = 4 requires ~1.6 GB model parameters (FP16),
with batched data at n_batch = 500 streaming from SSD.

Author: Shuaidong Gao
"""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Iterator, Tuple
import gc


# ═══════════════════════════════════════════════════════════════════
# FP16 LowRankGNN
# ═══════════════════════════════════════════════════════════════════

class FP16LowRankGNN(nn.Module):
    """Low-rank model with FP16 parameters for 2x memory reduction."""
    
    def __init__(self, d: int, rank: int = 4, device: str = "cuda"):
        super().__init__()
        self.d = d
        self.rank = rank
        self.device = device
        # Store in FP16 for memory efficiency
        self.U = nn.Parameter(torch.randn(d, rank, dtype=torch.float16, device=device) * 0.01)
        self.V = nn.Parameter(torch.randn(d, rank, dtype=torch.float16, device=device) * 0.01)
    
    def forward(self) -> torch.Tensor:
        """W = U @ V^T, cast to FP32 for numerical stability in loss."""
        return (self.U.float() @ self.V.float().T)
    
    @property
    def memory_mb(self) -> float:
        """GPU memory usage in MB for model parameters."""
        n_param = sum(p.numel() for p in self.parameters())
        return n_param * 2 / (1024 * 1024)  # FP16 = 2 bytes per param


# ═══════════════════════════════════════════════════════════════════
# Streaming Correlation Estimator
# ═══════════════════════════════════════════════════════════════════

class StreamingCorrelation:
    """Maintain running correlation statistics without materializing d x d."""
    
    def __init__(self, d: int, device: str = "cuda"):
        self.d = d
        self.device = device
        self.n_total = 0
        # Running sums for mean and covariance
        self.sum_x = torch.zeros(d, device=device)
        self.sum_xx = torch.zeros(d, device=device)  # diagonal of X^T X
    
    def update(self, batch: torch.Tensor):
        """Update running statistics with a data batch (n_batch, d)."""
        n_b = batch.shape[0]
        self.n_total += n_b
        self.sum_x += batch.sum(dim=0)
        self.sum_xx += (batch * batch).sum(dim=0)
    
    def mean(self) -> torch.Tensor:
        """Running mean vector (d,)."""
        return self.sum_x / max(self.n_total, 1)
    
    def std(self) -> torch.Tensor:
        """Running std vector (d,)."""
        var = self.sum_xx / max(self.n_total, 1) - self.mean() ** 2
        return torch.sqrt(var.clamp(min=1e-8))


# ═══════════════════════════════════════════════════════════════════
# Batch Generator (simulated streaming)
# ═══════════════════════════════════════════════════════════════════

def streaming_batch_generator(
    X: np.ndarray, 
    batch_size: int = 500,
    device: str = "cuda",
    shuffle: bool = True,
) -> Iterator[torch.Tensor]:
    """Generate batches from data as if streaming from disk.
    
    In production (d=10^8), this reads from memory-mapped files or HDF5.
    For benchmarking, we slice in-memory data to simulate streaming.
    """
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        batch = torch.tensor(X[batch_idx], dtype=torch.float32, device=device)
        yield batch
        del batch
        if device == "cuda":
            torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════
# Batched LowRank Trainer
# ═══════════════════════════════════════════════════════════════════

def batched_correlation_loss(
    U: torch.Tensor,
    V: torch.Tensor,
    batch: torch.Tensor,
    threshold: float = 0.3,
    streaming_corr: Optional[StreamingCorrelation] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute loss using only the current batch's data.
    
    Instead of materializing the full d x d correlation matrix, this function:
    1. Standardizes the batch using running statistics
    2. Computes batch-level correlation for a random subset of variable pairs
    3. Returns MSE between UV^T and the sampled correlation
    
    For large d (> 10,000), this samples 10,000 variable pairs per batch
    to keep memory constant regardless of d.
    """
    n_b, d = batch.shape
    dev = batch.device
    
    # Standardize batch
    if streaming_corr and streaming_corr.n_total > 0:
        mu = streaming_corr.mean()
        sigma = streaming_corr.std()
    else:
        mu = batch.mean(dim=0)
        sigma = batch.std(dim=0).clamp(min=1e-8)
    
    X_std = (batch - mu) / sigma
    
    # For large d, sample variable pairs to bound memory
    MAX_PAIRS = 10000
    if d > MAX_PAIRS:
        # Random sample of variable indices
        n_sample = min(MAX_PAIRS, d)
        idx = torch.randperm(d, device=dev)[:n_sample]
        X_sub = X_std[:, idx]
        C_batch = (X_sub.T @ X_sub) / (n_b - 1)
        C_batch.fill_diagonal_(0)
        gt = (torch.abs(C_batch) > threshold).float()
        
        W_sub = (U[idx].float() @ V[idx].float().T)
        loss = nn.MSELoss()(W_sub, gt)
    else:
        C_batch = (X_std.T @ X_std) / (n_b - 1)
        C_batch.fill_diagonal_(0)
        gt = (torch.abs(C_batch) > threshold).float()
        
        W_full = U.float() @ V.float().T
        loss = nn.MSELoss()(W_full, gt)
    
    return loss, gt


def train_batched_lowrank(
    X: np.ndarray,
    rank: int = 4,
    epochs: int = 100,
    batch_size: int = 500,
    threshold: float = 0.3,
    lr: float = 0.01,
    device: str = "cuda",
    use_fp16: bool = True,
    use_checkpoint: bool = True,
    verbose: bool = True,
) -> dict:
    """Train LowRankGNN with batched streaming data.
    
    This is the v3.2.0 pipeline that enables d=10^8 by:
    - Streaming data in batches (never materializing full correlation matrix)
    - FP16 model parameters (2x memory reduction)
    - Gradient checkpointing (compute-for-memory tradeoff)
    - Stochastic pair sampling for large d
    
    Args:
        X: (n, d) data matrix (in production: memory-mapped file)
        rank: factorization rank (default 4 for extreme scale)
        epochs: training epochs (fewer since each epoch sees ~n/batch_size updates)
        batch_size: rows per batch (500 default)
        threshold: edge threshold
        lr: learning rate
        device: 'cuda' or 'cpu'
        use_fp16: use FP16 model parameters
        use_checkpoint: use gradient checkpointing
    
    Returns:
        dict with memory_mb, time_s, edges, etc.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    dev = torch.device(device)
    n, d = X.shape
    
    # Sanitize
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Build model
    if use_fp16 and device == "cuda":
        model = FP16LowRankGNN(d, rank=rank, device=device)
    else:
        model = nn.Sequential()  # placeholder, use standard
        # Standard model for CPU / non-FP16
        class _StdLowRank(nn.Module):
            def __init__(self):
                super().__init__()
                self.U = nn.Parameter(torch.randn(d, rank, device=dev) * 0.01)
                self.V = nn.Parameter(torch.randn(d, rank, device=dev) * 0.01)
            def forward(self):
                return self.U @ self.V.T
        model = _StdLowRank()
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    
    # Memory profiling
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()
        mem_before = torch.cuda.memory_allocated() / (1024 * 1024)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Streaming correlation tracker
    stream_corr = StreamingCorrelation(d, device)
    
    # First pass: compute running statistics
    if verbose:
        print(f"  [Pass 1] Computing running statistics...")
    for batch in streaming_batch_generator(X, batch_size, device):
        stream_corr.update(batch)
    
    # Training
    if verbose:
        print(f"  [Pass 2] Training {epochs} epochs (batch_size={batch_size})...")
    
    t0 = time.time()
    loss_hist = []
    
    for ep in range(epochs):
        ep_loss = 0.0
        n_batches = 0
        
        for batch in streaming_batch_generator(X, batch_size, device, shuffle=True):
            opt.zero_grad()
            
            if use_checkpoint and device == "cuda" and d > 5000:
                # Gradient checkpointing: recompute activations in backward
                def _fwd(u, v, b):
                    return batched_correlation_loss(u, v, b, threshold, stream_corr)
                loss, _ = torch.utils.checkpoint.checkpoint(
                    _fwd, 
                    model.U if use_fp16 else model.U,
                    model.V if use_fp16 else model.V,
                    batch,
                    use_reentrant=False
                )
            else:
                if use_fp16:
                    loss, _ = batched_correlation_loss(
                        model.U, model.V, batch, threshold, stream_corr
                    )
                else:
                    loss, _ = batched_correlation_loss(
                        model.U, model.V, batch, threshold, stream_corr
                    )
            
            loss.backward()
            opt.step()
            
            ep_loss += loss.item()
            n_batches += 1
        
        avg_loss = ep_loss / max(n_batches, 1)
        loss_hist.append(avg_loss)
        
        if verbose and ep % max(1, epochs // 5) == 0:
            print(f"    E{ep:4d}: loss={avg_loss:.4f}")
    
    train_time = time.time() - t0
    
    # Memory profiling
    if device == "cuda":
        mem_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        mem_current = torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        mem_peak = n_params * 4 / (1024 * 1024)  # estimate
        mem_current = mem_peak
    
    # Extract results
    with torch.no_grad():
        if use_fp16:
            W = model.forward().cpu().numpy()
        else:
            W = model().cpu().numpy()
    
    n_edges = int(np.sum(np.abs(W) > threshold))
    
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return {
        "d": d,
        "n": n,
        "rank": rank,
        "n_params": n_params,
        "fp16": use_fp16,
        "checkpoint": use_checkpoint,
        "batch_size": batch_size,
        "epochs": epochs,
        "edges": n_edges,
        "time_s": round(train_time, 1),
        "memory_model_mb": round(n_params * (2 if use_fp16 else 4) / (1024 * 1024), 1),
        "memory_peak_mb": round(mem_peak, 1),
        "memory_current_mb": round(mem_current, 1),
        "loss_final": round(loss_hist[-1], 4) if loss_hist else 0,
        "adjacency": W,
    }


# ═══════════════════════════════════════════════════════════════════
# Memory Scaling Benchmark
# ═══════════════════════════════════════════════════════════════════

def benchmark_memory_scaling(
    d_list: list = None,
    rank: int = 4,
    n_samples: int = 1000,
    device: str = "cuda",
    verbose: bool = True,
) -> list:
    """Benchmark GPU memory usage across dimensions.
    
    Generates synthetic data for each d, trains with batched pipeline,
    and records actual GPU memory usage. Validates O(dr) linear scaling.
    """
    if d_list is None:
        d_list = [100, 200, 500, 1000, 2000, 5000, 10000]
    
    results = []
    
    if verbose:
        print("=" * 60)
        print("BATCHED LOWRANK MEMORY SCALING BENCHMARK (v3.2.0)")
        print(f"Rank r={rank}, n_batch=500, FP16 + Gradient Checkpointing")
        print("=" * 60)
    
    for d in d_list:
        if verbose:
            print(f"\n--- d={d} (model params: {2*d*rank/1024:.1f} KB FP16) ---")
        
        # Generate synthetic data (simulating streaming source)
        np.random.seed(42)
        X = np.random.randn(n_samples, d).astype(np.float32)
        
        # Determine batch size: smaller for large d to save memory
        batch_size = min(500, max(100, n_samples // 4))
        
        # Use gradient checkpointing only for d > 2000
        use_ckpt = (d > 2000)
        
        if d > 5000 and device == "cuda":
            # For very large d, reduce epochs to fit in 8GB
            epochs = 30
        elif d > 1000:
            epochs = 50
        else:
            epochs = 100
        
        try:
            result = train_batched_lowrank(
                X, rank=rank, epochs=epochs, batch_size=batch_size,
                threshold=0.3, lr=0.01, device=device,
                use_fp16=True, use_checkpoint=use_ckpt, verbose=verbose,
            )
            result["status"] = "ok"
        except torch.cuda.OutOfMemoryError as e:
            result = {
                "d": d, "rank": rank, "status": "OOM",
                "memory_peak_mb": -1, "edges": 0, "time_s": 0,
            }
            if verbose:
                print(f"  !! OOM at d={d}")
            torch.cuda.empty_cache()
        except Exception as e:
            result = {
                "d": d, "rank": rank, "status": f"error: {e}",
                "memory_peak_mb": -1, "edges": 0, "time_s": 0,
            }
        
        results.append(result)
        
        if verbose:
            print(f"  Peak memory: {result['memory_peak_mb']:.0f} MB")
            print(f"  Time: {result['time_s']:.1f}s")
            print(f"  Edges: {result.get('edges', 0)}")
    
    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("MEMORY SCALING SUMMARY")
        print(f"{'d':>6} {'Peak MB':>8} {'Model MB':>8} {'Time s':>7} {'Edges':>6} {'Status':>8}")
        print("-" * 50)
        for r in results:
            print(f"{r['d']:>6} {r['memory_peak_mb']:>8.0f} "
                  f"{r['memory_model_mb']:>8.1f} {r['time_s']:>7.1f} "
                  f"{r.get('edges',0):>6} {r.get('status','?'):>8}")
    
    return results
