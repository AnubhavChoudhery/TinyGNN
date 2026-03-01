"""
TinyGNN — Real-Data GNN Integration Tests  (Phase 11)
tests/test_gnn_integration.py

Tests GCN, SAGE, and GAT end-to-end on actual Cora and Reddit datasets
loaded from CSV via GraphLoader.  Validates:

  ✓ GraphLoader.load() — Cora & Reddit CSV ingestion
  ✓ gcn_norm pre-processing
  ✓ GCNLayer   forward on real graph  (shape, no-NaN, no-Inf)
  ✓ SAGELayer  forward — Mean & Max aggregation on real graph
  ✓ GATLayer   forward on real graph (with self-loops)
  ✓ 2-layer GCN model end-to-end (Cora-scale inference)
  ✓ 2-layer SAGE model end-to-end
  ✓ 2-layer GAT  model end-to-end
  ✓ Tensor.from_edge_index — PyG-style CSR construction
  ✓ Tensor.from_scipy_csr  — SciPy interop
  ✓ Output distribution sanity (finite, bounded range)
  ✓ Reddit (large graph) — GCN shape + no-NaN  (auto-skipped if missing)

Run:
    pytest tests/test_gnn_integration.py -v
    pytest tests/test_gnn_integration.py -v -k reddit    # large graph only
"""

import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

import tinygnn

# ── Dataset paths ─────────────────────────────────────────────────────────────
CORA_EDGES    = os.path.join(ROOT, "datasets", "cora",   "edges.csv")
CORA_FEATS    = os.path.join(ROOT, "datasets", "cora",   "node_features.csv")
REDDIT_EDGES  = os.path.join(ROOT, "datasets", "reddit", "edges.csv")
REDDIT_FEATS  = os.path.join(ROOT, "datasets", "reddit", "node_features.csv")

CORA_AVAIL   = os.path.exists(CORA_EDGES)   and os.path.exists(CORA_FEATS)
REDDIT_AVAIL = os.path.exists(REDDIT_EDGES) and os.path.exists(REDDIT_FEATS)

requires_cora   = pytest.mark.skipif(not CORA_AVAIL,   reason="Cora dataset not found")
requires_reddit = pytest.mark.skipif(not REDDIT_AVAIL, reason="Reddit dataset not found")


# ── Reproducible random weights ───────────────────────────────────────────────
rng = np.random.default_rng(42)

def rand_tensor(rows, cols, scale=0.05):
    arr = rng.standard_normal((rows, cols)).astype(np.float32) * scale
    return tinygnn.Tensor.from_numpy(arr)

def is_finite_nd(arr):
    return bool(np.all(np.isfinite(arr)))


# ============================================================================
#  Fixture: load Cora once per session
# ============================================================================

@pytest.fixture(scope="session")
def cora():
    """Load Cora graph data once for the whole test session."""
    g = tinygnn.GraphLoader.load(CORA_EDGES, CORA_FEATS)
    return g


@pytest.fixture(scope="session")
def cora_norm(cora):
    """Pre-compute GCN-normalised adjacency once."""
    return tinygnn.gcn_norm(cora.adjacency)


@pytest.fixture(scope="session")
def cora_sl(cora):
    """Adjacency with self-loops (for GAT/SAGE)."""
    return tinygnn.add_self_loops(cora.adjacency)


# ============================================================================
#  1. GraphLoader — dataset ingestion
# ============================================================================

class TestGraphLoader:
    @requires_cora
    def test_cora_shape(self, cora):
        assert cora.num_nodes == 2708
        assert cora.num_features == 1433
        assert cora.num_edges == 5429
        assert cora.adjacency.rows == 2708
        assert cora.adjacency.cols == 2708
        assert cora.node_features.rows == 2708
        assert cora.node_features.cols == 1433

    @requires_cora
    def test_cora_adjacency_is_sparse(self, cora):
        assert cora.adjacency.format == tinygnn.StorageFormat.SparseCSR
        assert cora.adjacency.nnz == cora.num_edges

    @requires_cora
    def test_cora_features_finite(self, cora):
        feat_np = cora.node_features.to_numpy()
        assert is_finite_nd(feat_np), "Cora features contain NaN/Inf"

    @requires_cora
    def test_cora_features_range(self, cora):
        feat_np = cora.node_features.to_numpy()
        # Cora bag-of-words features should be in [0, 1]
        assert feat_np.min() >= 0.0, "Feature min < 0"
        assert feat_np.max() <= 1.0 + 1e-5, "Feature max > 1"

    @requires_reddit
    def test_reddit_shape(self):
        g = tinygnn.GraphLoader.load(REDDIT_EDGES, REDDIT_FEATS)
        assert g.num_nodes == 232965
        assert g.num_features == 602
        assert g.num_edges > 100_000_000   # 114M+ edges
        assert g.adjacency.rows == g.num_nodes
        assert g.node_features.rows == g.num_nodes


# ============================================================================
#  2. GCN Layer — real graph inference
# ============================================================================

class TestGCNLayerReal:
    @requires_cora
    def test_single_layer_shape(self, cora, cora_norm):
        F_in, F_out = cora.num_features, 64
        layer = tinygnn.GCNLayer(F_in, F_out, True, tinygnn.Activation.RELU)
        layer.set_weight(rand_tensor(F_in, F_out))
        layer.set_bias(rand_tensor(1, F_out))

        out = layer.forward(cora_norm, cora.node_features)
        arr = out.to_numpy()
        assert arr.shape == (cora.num_nodes, F_out)

    @requires_cora
    def test_single_layer_finite(self, cora, cora_norm):
        F_in, F_out = cora.num_features, 64
        layer = tinygnn.GCNLayer(F_in, F_out, True, tinygnn.Activation.RELU)
        layer.set_weight(rand_tensor(F_in, F_out))
        layer.set_bias(rand_tensor(1, F_out))

        arr = layer.forward(cora_norm, cora.node_features).to_numpy()
        assert is_finite_nd(arr), "GCN output contains NaN/Inf"

    @requires_cora
    def test_single_layer_relu_nonneg(self, cora, cora_norm):
        F_in, F_out = cora.num_features, 16
        layer = tinygnn.GCNLayer(F_in, F_out, True, tinygnn.Activation.RELU)
        layer.set_weight(rand_tensor(F_in, F_out))
        layer.set_bias(rand_tensor(1, F_out))

        arr = layer.forward(cora_norm, cora.node_features).to_numpy()
        assert np.all(arr >= 0), "ReLU output has negative values"

    @requires_cora
    def test_two_layer_gcn_cora(self, cora, cora_norm):
        """Full 2-layer GCN: 1433 → 64 → 7  (Cora classification dims)."""
        l1 = tinygnn.GCNLayer(1433, 64, True, tinygnn.Activation.RELU)
        l1.set_weight(rand_tensor(1433, 64))
        l1.set_bias(rand_tensor(1, 64))

        l2 = tinygnn.GCNLayer(64, 7, True, tinygnn.Activation.NONE)
        l2.set_weight(rand_tensor(64, 7))
        l2.set_bias(rand_tensor(1, 7))

        h1 = l1.forward(cora_norm, cora.node_features)
        h2 = l2.forward(cora_norm, h1)

        arr = h2.to_numpy()
        assert arr.shape == (2708, 7)
        assert is_finite_nd(arr)


# ============================================================================
#  3. SAGE Layer — real graph inference
# ============================================================================

class TestSAGELayerReal:
    @requires_cora
    def test_mean_agg_shape(self, cora):
        F_in, F_out = cora.num_features, 64
        layer = tinygnn.SAGELayer(F_in, F_out,
                                  tinygnn.SAGELayer.Aggregator.Mean,
                                  True, tinygnn.Activation.RELU)
        layer.set_weight_neigh(rand_tensor(F_in, F_out))
        layer.set_weight_self(rand_tensor(F_in, F_out))
        layer.set_bias(rand_tensor(1, F_out))

        out = layer.forward(cora.adjacency, cora.node_features).to_numpy()
        assert out.shape == (cora.num_nodes, F_out)

    @requires_cora
    def test_mean_agg_finite(self, cora):
        F_in, F_out = cora.num_features, 64
        layer = tinygnn.SAGELayer(F_in, F_out,
                                  tinygnn.SAGELayer.Aggregator.Mean,
                                  True, tinygnn.Activation.RELU)
        layer.set_weight_neigh(rand_tensor(F_in, F_out))
        layer.set_weight_self(rand_tensor(F_in, F_out))
        layer.set_bias(rand_tensor(1, F_out))

        out = layer.forward(cora.adjacency, cora.node_features).to_numpy()
        assert is_finite_nd(out), "SAGE Mean output contains NaN/Inf"

    @requires_cora
    def test_max_agg_shape(self, cora):
        F_in, F_out = cora.num_features, 64
        layer = tinygnn.SAGELayer(F_in, F_out,
                                  tinygnn.SAGELayer.Aggregator.Max,
                                  True, tinygnn.Activation.RELU)
        layer.set_weight_neigh(rand_tensor(F_in, F_out))
        layer.set_weight_self(rand_tensor(F_in, F_out))
        layer.set_bias(rand_tensor(1, F_out))

        out = layer.forward(cora.adjacency, cora.node_features).to_numpy()
        assert out.shape == (cora.num_nodes, F_out)

    @requires_cora
    def test_mean_vs_max_differ(self, cora):
        """Mean and Max aggregation should produce different outputs."""
        F_in, F_out = 1433, 16

        wn = rand_tensor(F_in, F_out)
        ws = rand_tensor(F_in, F_out)
        b  = rand_tensor(1, F_out)

        def run(agg):
            layer = tinygnn.SAGELayer(F_in, F_out, agg, True, tinygnn.Activation.NONE)
            layer.set_weight_neigh(wn)
            layer.set_weight_self(ws)
            layer.set_bias(b)
            return layer.forward(cora.adjacency, cora.node_features).to_numpy()

        mean_out = run(tinygnn.SAGELayer.Aggregator.Mean)
        max_out  = run(tinygnn.SAGELayer.Aggregator.Max)
        assert not np.allclose(mean_out, max_out, atol=1e-4), \
            "Mean and Max agg produced identical output"

    @requires_cora
    def test_two_layer_sage_cora(self, cora):
        """Full 2-layer SAGE: 1433 → 128 → 7."""
        l1 = tinygnn.SAGELayer(1433, 128,
                               tinygnn.SAGELayer.Aggregator.Mean,
                               True, tinygnn.Activation.RELU)
        l1.set_weight_neigh(rand_tensor(1433, 128))
        l1.set_weight_self(rand_tensor(1433, 128))
        l1.set_bias(rand_tensor(1, 128))

        l2 = tinygnn.SAGELayer(128, 7,
                               tinygnn.SAGELayer.Aggregator.Mean,
                               True, tinygnn.Activation.NONE)
        l2.set_weight_neigh(rand_tensor(128, 7))
        l2.set_weight_self(rand_tensor(128, 7))
        l2.set_bias(rand_tensor(1, 7))

        h1 = l1.forward(cora.adjacency, cora.node_features)
        h2 = l2.forward(cora.adjacency, h1)

        arr = h2.to_numpy()
        assert arr.shape == (2708, 7)
        assert is_finite_nd(arr)


# ============================================================================
#  4. GAT Layer — real graph inference
# ============================================================================

class TestGATLayerReal:
    @requires_cora
    def test_single_layer_shape(self, cora, cora_sl):
        F_in, F_out = cora.num_features, 8
        layer = tinygnn.GATLayer(F_in, F_out, 0.2, True, tinygnn.Activation.NONE)
        layer.set_weight(rand_tensor(F_in, F_out))
        layer.set_attn_left(rand_tensor(1, F_out))
        layer.set_attn_right(rand_tensor(1, F_out))
        layer.set_bias(rand_tensor(1, F_out))

        out = layer.forward(cora_sl, cora.node_features).to_numpy()
        assert out.shape == (cora.num_nodes, F_out)

    @requires_cora
    def test_single_layer_finite(self, cora, cora_sl):
        F_in, F_out = cora.num_features, 8
        layer = tinygnn.GATLayer(F_in, F_out, 0.2, True, tinygnn.Activation.NONE)
        layer.set_weight(rand_tensor(F_in, F_out))
        layer.set_attn_left(rand_tensor(1, F_out))
        layer.set_attn_right(rand_tensor(1, F_out))
        layer.set_bias(rand_tensor(1, F_out))

        out = layer.forward(cora_sl, cora.node_features).to_numpy()
        assert is_finite_nd(out), "GAT output contains NaN/Inf"

    @requires_cora
    def test_negative_slope_effect(self, cora, cora_sl):
        """Different negative slopes should produce different outputs."""
        F_in, F_out = 1433, 8

        def run(slope):
            layer = tinygnn.GATLayer(F_in, F_out, slope, True, tinygnn.Activation.NONE)
            layer.set_weight(rand_tensor(F_in, F_out))
            layer.set_attn_left(rand_tensor(1, F_out))
            layer.set_attn_right(rand_tensor(1, F_out))
            layer.set_bias(rand_tensor(1, F_out))
            return layer.forward(cora_sl, cora.node_features).to_numpy()

        out_02 = run(0.2)
        out_05 = run(0.5)
        assert not np.allclose(out_02, out_05, atol=1e-4)

    @requires_cora
    def test_two_layer_gat_cora(self, cora, cora_sl):
        """Full 2-layer GAT: 1433 → 8 features → 7 classes."""
        l1 = tinygnn.GATLayer(1433, 8, 0.2, True, tinygnn.Activation.RELU)
        l1.set_weight(rand_tensor(1433, 8))
        l1.set_attn_left(rand_tensor(1, 8))
        l1.set_attn_right(rand_tensor(1, 8))
        l1.set_bias(rand_tensor(1, 8))

        l2 = tinygnn.GATLayer(8, 7, 0.2, True, tinygnn.Activation.NONE)
        l2.set_weight(rand_tensor(8, 7))
        l2.set_attn_left(rand_tensor(1, 7))
        l2.set_attn_right(rand_tensor(1, 7))
        l2.set_bias(rand_tensor(1, 7))

        h1 = l1.forward(cora_sl, cora.node_features)
        h2 = l2.forward(cora_sl, h1)

        arr = h2.to_numpy()
        assert arr.shape == (2708, 7)
        assert is_finite_nd(arr)


# ============================================================================
#  5. Ops on real feature matrix
# ============================================================================

class TestOpsReal:
    @requires_cora
    def test_spmm_shape(self, cora, cora_norm):
        """SpMM: A_norm (2708×2708 sparse) × H (2708×1433 dense) → 2708×1433."""
        out = tinygnn.spmm(cora_norm, cora.node_features)
        arr = out.to_numpy()
        assert arr.shape == (cora.num_nodes, cora.num_features)

    @requires_cora
    def test_spmm_finite(self, cora, cora_norm):
        out = tinygnn.spmm(cora_norm, cora.node_features)
        assert is_finite_nd(out.to_numpy())

    @requires_cora
    def test_matmul_feature_projection(self, cora):
        """matmul: H (2708×1433) × W (1433×64) → 2708×64."""
        W = rand_tensor(cora.num_features, 64)
        out = tinygnn.matmul(cora.node_features, W)
        arr = out.to_numpy()
        assert arr.shape == (cora.num_nodes, 64)
        assert is_finite_nd(arr)

    @requires_cora
    def test_relu_on_spmm(self, cora, cora_norm):
        out = tinygnn.spmm(cora_norm, cora.node_features)
        tinygnn.relu_inplace(out)
        arr = out.to_numpy()
        assert np.all(arr >= 0)

    @requires_cora
    def test_add_bias_broadcast(self, cora):
        W = rand_tensor(cora.num_features, 32)
        out = tinygnn.matmul(cora.node_features, W)
        b = rand_tensor(1, 32)
        tinygnn.add_bias(out, b)
        arr = out.to_numpy()
        assert arr.shape == (cora.num_nodes, 32)
        assert is_finite_nd(arr)

    @requires_cora
    def test_edge_softmax_on_cora(self, cora, cora_sl):
        """edge_softmax: each row's attention weights sum to 1."""
        alpha = tinygnn.edge_softmax(cora_sl)
        rp = alpha.row_ptr_numpy()
        vals = alpha.values_numpy()
        for i in range(cora.num_nodes):
            row_start = rp[i]
            row_end   = rp[i + 1]
            if row_end > row_start:
                row_sum = float(np.sum(vals[row_start:row_end]))
                assert abs(row_sum - 1.0) < 1e-5, \
                    f"Row {i} softmax sum = {row_sum:.6f}"


# ============================================================================
#  6. Graph utility correctness
# ============================================================================

class TestGraphUtilsReal:
    @requires_cora
    def test_gcn_norm_values_bounded(self, cora, cora_norm):
        """GCN-norm values should be in (0, 1] for a connected graph."""
        vals = cora_norm.values_numpy()
        assert np.all(vals > 0), "GCN norm has non-positive values"
        assert np.all(vals <= 1.0 + 1e-6), "GCN norm has values > 1"

    @requires_cora
    def test_add_self_loops_increases_nnz(self, cora):
        A = cora.adjacency
        A_sl = tinygnn.add_self_loops(A)
        # Each of 2708 nodes gains at most 1 self-loop
        assert A_sl.nnz >= A.nnz
        assert A_sl.nnz <= A.nnz + cora.num_nodes

    @requires_cora
    def test_from_edge_index(self, cora):
        """Build CSR from raw edge_index array and validate nnz matches."""
        rp = cora.adjacency.row_ptr_numpy()
        ci = cora.adjacency.col_ind_numpy()

        # Reconstruct a simple edge_index from CSR
        srcs, dsts = [], []
        for src in range(cora.num_nodes):
            for idx in range(int(rp[src]), int(rp[src + 1])):
                srcs.append(src)
                dsts.append(int(ci[idx]))

        edge_index = np.array([srcs, dsts], dtype=np.int32)
        A_rebuilt = tinygnn.Tensor.from_edge_index(edge_index, cora.num_nodes)
        assert A_rebuilt.nnz == cora.adjacency.nnz

    @requires_cora
    def test_from_scipy_csr(self, cora):
        """Convert cora adjacency to scipy and back, check nnz."""
        pytest.importorskip("scipy", reason="scipy not installed")
        import scipy.sparse as sp

        rp  = cora.adjacency.row_ptr_numpy().astype(np.int32)
        ci  = cora.adjacency.col_ind_numpy().astype(np.int32)
        val = cora.adjacency.values_numpy()
        N = cora.num_nodes

        scipy_csr = sp.csr_matrix((val, ci, rp), shape=(N, N))
        A_back = tinygnn.Tensor.from_scipy_csr(scipy_csr)
        assert A_back.nnz == cora.adjacency.nnz
        assert A_back.rows == N


# ============================================================================
#  7. Reddit — large graph smoke tests
# ============================================================================

class TestReddit:
    @requires_reddit
    def test_gcn_forward_large(self):
        """GCN on Reddit (232K nodes, 602 features): shape + finite."""
        g = tinygnn.GraphLoader.load(REDDIT_EDGES, REDDIT_FEATS)
        A_norm = tinygnn.gcn_norm(g.adjacency)

        F_in, F_out = g.num_features, 32
        layer = tinygnn.GCNLayer(F_in, F_out, True, tinygnn.Activation.RELU)
        layer.set_weight(rand_tensor(F_in, F_out))
        layer.set_bias(rand_tensor(1, F_out))

        out = layer.forward(A_norm, g.node_features).to_numpy()
        assert out.shape == (g.num_nodes, F_out)
        assert is_finite_nd(out), "Reddit GCN output has NaN/Inf"

    @requires_reddit
    def test_sage_forward_large(self):
        """SAGE on Reddit: shape + finite."""
        g = tinygnn.GraphLoader.load(REDDIT_EDGES, REDDIT_FEATS)

        F_in, F_out = g.num_features, 32
        layer = tinygnn.SAGELayer(F_in, F_out,
                                  tinygnn.SAGELayer.Aggregator.Mean,
                                  True, tinygnn.Activation.RELU)
        layer.set_weight_neigh(rand_tensor(F_in, F_out))
        layer.set_weight_self(rand_tensor(F_in, F_out))
        layer.set_bias(rand_tensor(1, F_out))

        out = layer.forward(g.adjacency, g.node_features).to_numpy()
        assert out.shape == (g.num_nodes, F_out)
        assert is_finite_nd(out), "Reddit SAGE output has NaN/Inf"

    @requires_reddit
    def test_gat_forward_large(self):
        """GAT on Reddit: shape + finite."""
        g = tinygnn.GraphLoader.load(REDDIT_EDGES, REDDIT_FEATS)
        A_sl = tinygnn.add_self_loops(g.adjacency)

        F_in, F_out = g.num_features, 8
        layer = tinygnn.GATLayer(F_in, F_out, 0.2, True, tinygnn.Activation.NONE)
        layer.set_weight(rand_tensor(F_in, F_out))
        layer.set_attn_left(rand_tensor(1, F_out))
        layer.set_attn_right(rand_tensor(1, F_out))
        layer.set_bias(rand_tensor(1, F_out))

        out = layer.forward(A_sl, g.node_features).to_numpy()
        assert out.shape == (g.num_nodes, F_out)
        assert is_finite_nd(out), "Reddit GAT output has NaN/Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
