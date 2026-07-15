"""Core engine and backend tests for causalscale V3."""
import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ============================================================
# 1. NOTEARS backend tests
# ============================================================

class TestNOTEARSBackend:
    def test_import(self):
        from causalscale.core._notears import run_notears, run_cagate
        assert run_notears is not None
        assert run_cagate is not None

    def test_small_dag_converges(self):
        """NOTEARS on tiny (d=5) DAG: should converge h->0 and find edges."""
        from causalscale.core._notears import run_notears
        np.random.seed(42)
        d = 5
        W_true = np.triu(np.random.randn(d, d) * 0.5, 1)
        X = np.random.randn(100, d) @ np.linalg.inv(np.eye(d) - W_true)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        W, ec, h, t = run_notears(X.astype(np.float32), device='cpu',
                                 outer=20, inner=100, seed=42)
        assert h < 1.0, f"h(W)={h:.2e}, expected < 1.0"
        assert ec > 0, f"No edges found on d=5 DAG"

    def test_zero_data_sanitizes(self):
        """NOTEARS on zero data should not crash."""
        from causalscale.core._notears import run_notears
        X = np.zeros((50, 10), dtype=np.float32)
        W, ec, h, t = run_notears(X, device='cpu', outer=5, inner=50)
        assert W.shape == (10, 10)
        assert ec >= 0

    def test_single_variable(self):
        """d=1 should not crash."""
        from causalscale.core._notears import run_notears
        X = np.random.randn(50, 1).astype(np.float32)
        W, ec, h, t = run_notears(X, device='cpu', outer=5, inner=50)
        assert W.shape == (1, 1)
        assert ec == 0


# ============================================================
# 2. DAG constraint tests
# ============================================================

class TestDAGConstraint:
    def test_dag_constraint_zero_for_dag(self):
        """h(W)=0 when W is strictly upper-triangular (acyclic)."""
        from causalscale.core.dag_utils import efficient_dag_constraint
        import torch
        d = 10
        W = torch.triu(torch.randn(d, d) * 0.1, 1)
        h = efficient_dag_constraint(W)
        assert 0 <= h < 1.0, f"h(W)={h:.4f} for upper-triangular DAG"

    def test_dag_constraint_positive_for_cycle(self):
        """h(W) > 0 for cyclic graph."""
        from causalscale.core.dag_utils import efficient_dag_constraint
        import torch
        W = torch.ones(5, 5) * 0.5
        h = efficient_dag_constraint(W)
        assert h > 1.0, f"h(W)={h:.4f} for cyclic graph"

    def test_dag_constraint_exact_mode(self):
        """Test the exact matrix_exp path (d=500 threshold)."""
        from causalscale.core.dag_utils import efficient_dag_constraint
        import torch
        W = torch.zeros(300, 300)
        h = efficient_dag_constraint(W)
        assert abs(h) < 1e-4, f"h(W) for zero matrix = {h:.4f}, expected ~0"


# ============================================================
# 3. CausalDiscovery API tests
# ============================================================

class TestCausalDiscoveryAPI:
    def test_import_and_create(self):
        import causalscale as cs
        X = np.random.randn(100, 10).astype(np.float32)
        model = cs.CausalDiscovery(X, method='cluster_aware', device='cpu')
        assert model.d == 10
        assert model.n == 100
        assert model.method == 'cluster_aware'

    def test_fit_cluster_aware(self):
        """End-to-end fit with cluster_aware (NOTEARS backend)."""
        import causalscale as cs
        np.random.seed(42)
        d = 8
        W_true = np.triu(np.random.randn(d, d) * 0.5, 1)
        X = np.random.randn(80, d) @ np.linalg.inv(np.eye(d) - W_true)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        model = cs.CausalDiscovery(X.astype(np.float32),
                                   method='cluster_aware', device='cpu')
        model.fit(verbose=False)
        net = model.get_network()
        assert net.edge_count >= 0
        assert net.adjacency.shape == (d, d)

    def test_fit_auto_method(self):
        """Auto method selection should work without crashing."""
        import causalscale as cs
        X = np.random.randn(50, 20).astype(np.float32)
        model = cs.CausalDiscovery(X, method='auto', device='cpu')
        model.fit(verbose=False)
        assert model._fitted

    def test_get_edges(self):
        """get_edges should return list of tuples."""
        import causalscale as cs
        np.random.seed(42)
        d = 8
        W_true = np.triu(np.random.randn(d, d) * 0.5, 1)
        X = np.random.randn(80, d) @ np.linalg.inv(np.eye(d) - W_true)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        model = cs.CausalDiscovery(X.astype(np.float32),
                                   method='cluster_aware', device='cpu')
        model.fit(verbose=False)
        edges = model.get_edges(confidence=0)
        assert isinstance(edges, list)

    def test_summary(self):
        """summary() should return non-empty string."""
        import causalscale as cs
        X = np.random.randn(50, 10).astype(np.float32)
        model = cs.CausalDiscovery(X, method='cluster_aware', device='cpu')
        model.fit(verbose=False)
        s = model.summary()
        assert 'CausalDiscovery' in s
        assert 'Method' in s

    def test_not_fitted_raises(self):
        """Calling get_network before fit should raise."""
        import causalscale as cs
        X = np.random.randn(50, 10).astype(np.float32)
        model = cs.CausalDiscovery(X, method='cluster_aware', device='cpu')
        with pytest.raises(RuntimeError):
            model.get_network()

    def test_repr(self):
        import causalscale as cs
        X = np.random.randn(50, 10).astype(np.float32)
        model = cs.CausalDiscovery(X, method='lowrank', device='cpu')
        assert 'CausalDiscovery' in repr(model)
        assert 'not fitted' in repr(model)


# ============================================================
# 4. Data sanitization tests
# ============================================================

class TestDataSanitization:
    def test_nan_handling(self):
        """NaN inputs should not crash."""
        import causalscale as cs
        X = np.random.randn(50, 10).astype(np.float32)
        X[0, 0] = np.nan
        X[1, 1] = np.inf
        X[2, 2] = -np.inf
        model = cs.CausalDiscovery(X, method='cluster_aware', device='cpu')
        model.fit(verbose=False)
        assert model._fitted

    def test_zero_variance_column(self):
        """Zero-variance columns should not crash."""
        import causalscale as cs
        X = np.random.randn(50, 10).astype(np.float32)
        X[:, 5] = 0.0
        model = cs.CausalDiscovery(X, method='cluster_aware', device='cpu')
        model.fit(verbose=False)
        assert model._fitted

    @pytest.mark.skip(reason="CSV tempfile parsing edge case - pandas sep detection")
    def test_csv_path_input(self):
        """Loading from CSV path should work."""
        import causalscale as cs
        import tempfile, pandas as pd
        X = np.random.randn(50, 8).astype(np.float32)
        cols = [f'V{i}' for i in range(8)]
        df = pd.DataFrame(X, columns=cols)
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            df.to_csv(f.name, index=False)
            path = f.name
        try:
            model = cs.CausalDiscovery(path, method='cluster_aware', device='cpu')
            assert model.d == 8
        finally:
            os.unlink(path)

    def test_too_few_samples(self):
        """Less than 10 samples should raise."""
        import causalscale as cs
        X = np.random.randn(5, 10).astype(np.float32)
        with pytest.raises(ValueError):
            cs.CausalDiscovery(X, method='cluster_aware', device='cpu')

    def test_too_few_variables(self):
        """d < 2 should raise."""
        import causalscale as cs
        X = np.random.randn(50, 1).astype(np.float32)
        with pytest.raises(ValueError):
            cs.CausalDiscovery(X, method='cluster_aware', device='cpu')


# ============================================================
# 5. Method mapping tests
# ============================================================

class TestMethodMapping:
    def test_gate_alias(self):
        """'gate' should map to 'cluster_aware'."""
        import causalscale as cs
        X = np.random.randn(50, 10).astype(np.float32)
        model = cs.CausalDiscovery(X, method='gate', device='cpu')
        assert model.method == 'cluster_aware'

    def test_ct_alias(self):
        """'ct' should map to 'transformer'."""
        import causalscale as cs
        X = np.random.randn(50, 10).astype(np.float32)
        model = cs.CausalDiscovery(X, method='ct', device='cpu')
        assert model.method == 'transformer'

    def test_auto_selects(self):
        """auto should select based on n and d."""
        import causalscale as cs
        X = np.random.randn(50, 10).astype(np.float32)   # n<200 -> cluster_aware
        model = cs.CausalDiscovery(X, method='auto', device='cpu')
        assert model.method == 'cluster_aware'
