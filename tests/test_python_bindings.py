"""
TinyGNN — Comprehensive Python Binding Tests  (Phase 7)
tests/test_python_bindings.py

Tests cover:
  ✓ Tensor creation (Dense, SparseCSR, from_numpy, from_scipy_csr, from_edge_index)
  ✓ Tensor observers (rows, cols, numel, nnz, format, memory_footprint_bytes)
  ✓ Tensor NumPy round-trip (2D, 1D)
  ✓ Tensor CSR accessors (row_ptr_numpy, col_ind_numpy, values_numpy)
  ✓ Ops: matmul, spmm
  ✓ Activations: relu, leaky_relu, elu, sigmoid, tanh, gelu, softmax, log_softmax
  ✓ add_bias
  ✓ Graph utilities: add_self_loops, gcn_norm, edge_softmax, sage_max_aggregate
  ✓ GCNLayer: creation, weight setting, forward pass
  ✓ SAGELayer: creation, weight setting, forward pass (Mean + Max)
  ✓ GATLayer: creation, weight setting, forward pass
  ✓ Model: build, load weights, forward
  ✓ Error handling (wrong shapes, formats)

Run:
    python -m pytest tests/test_python_bindings.py -v
"""

import sys
import os
import math
import struct
import tempfile

import numpy as np
import pytest

# Ensure the extension + package can be found
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

import _tinygnn_core as tg


# ============================================================================
#  Helper utilities
# ============================================================================

def make_dense(rows, cols, data):
    """Convenience wrapper for creating a Dense Tensor from a flat list."""
    return tg.Tensor.dense_from_data(rows, cols, data)


def make_triangle_csr():
    """3-node triangle: edges 0→1, 1→0, 1→2, 2→1, 0→2, 2→0."""
    rp = [0, 2, 4, 6]
    ci = [1, 2, 0, 2, 0, 1]
    vals = [1.0] * 6
    return tg.Tensor.sparse_csr(3, 3, rp, ci, vals)


# ============================================================================
#  Tensor — Dense creation & observers
# ============================================================================

class TestTensorDense:
    def test_zero_filled(self):
        t = tg.Tensor.dense(3, 4)
        assert t.rows == 3
        assert t.cols == 4
        assert t.numel == 12
        assert t.format == tg.StorageFormat.Dense
        for r in range(3):
            for c in range(4):
                assert t.at(r, c) == 0.0

    def test_from_data(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        t = make_dense(2, 3, data)
        assert t.at(0, 0) == 1.0
        assert t.at(0, 2) == 3.0
        assert t.at(1, 0) == 4.0
        assert t.at(1, 2) == 6.0

    def test_from_numpy_2d(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        t = tg.Tensor.from_numpy(arr)
        assert t.rows == 2 and t.cols == 2
        assert t.at(0, 1) == 2.0
        assert t.at(1, 0) == 3.0

    def test_from_numpy_f64_auto_cast(self):
        """float64 arrays should be auto-cast to float32."""
        arr = np.array([[1.5, 2.5]], dtype=np.float64)
        t = tg.Tensor.from_numpy(arr)
        assert abs(t.at(0, 0) - 1.5) < 1e-6

    def test_from_numpy_1d(self):
        arr = np.array([10, 20, 30], dtype=np.float32)
        t = tg.Tensor.from_numpy_1d(arr)
        assert t.rows == 1 and t.cols == 3
        assert t.at(0, 1) == 20.0

    def test_to_numpy_roundtrip(self):
        arr = np.random.randn(5, 7).astype(np.float32)
        t = tg.Tensor.from_numpy(arr)
        out = t.to_numpy()
        np.testing.assert_allclose(arr, out, atol=1e-7)

    def test_memory_footprint(self):
        t = tg.Tensor.dense(10, 10)
        assert t.memory_footprint_bytes == 10 * 10 * 4  # float32

    def test_repr(self):
        t = tg.Tensor.dense(3, 4)
        r = t.repr()
        assert "3x4" in r
        assert "Dense" in r

    def test_dunder_repr(self):
        t = tg.Tensor.dense(2, 5)
        assert "2x5" in repr(t)


# ============================================================================
#  Tensor — SparseCSR creation & observers
# ============================================================================

class TestTensorCSR:
    def test_create(self):
        rp = [0, 2, 3]
        ci = [0, 1, 1]
        vals = [1.0, 2.0, 3.0]
        t = tg.Tensor.sparse_csr(2, 2, rp, ci, vals)
        assert t.format == tg.StorageFormat.SparseCSR
        assert t.rows == 2 and t.cols == 2
        assert t.nnz == 3

    def test_csr_numpy_accessors(self):
        rp = [0, 1, 3]
        ci = [0, 0, 1]
        vals = [5.0, 6.0, 7.0]
        t = tg.Tensor.sparse_csr(2, 2, rp, ci, vals)
        np.testing.assert_array_equal(t.row_ptr_numpy(), [0, 1, 3])
        np.testing.assert_array_equal(t.col_ind_numpy(), [0, 0, 1])
        np.testing.assert_allclose(t.values_numpy(), [5.0, 6.0, 7.0])

    def test_from_scipy_csr(self):
        scipy_sparse = pytest.importorskip("scipy.sparse")
        row = np.array([0, 0, 1, 2])
        col = np.array([0, 2, 1, 0])
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        mat = scipy_sparse.csr_matrix((data, (row, col)), shape=(3, 3))

        t = tg.Tensor.from_scipy_csr(mat)
        assert t.format == tg.StorageFormat.SparseCSR
        assert t.rows == 3 and t.cols == 3
        assert t.nnz == 4
        np.testing.assert_allclose(t.values_numpy(), [1.0, 2.0, 3.0, 4.0])

    def test_from_edge_index(self):
        # Triangle graph: 0→1, 1→0, 1→2, 2→1, 0→2, 2→0
        edge_index = np.array([[0, 1, 1, 2, 0, 2],
                               [1, 0, 2, 1, 2, 0]], dtype=np.int32)
        t = tg.Tensor.from_edge_index(edge_index, 3)
        assert t.format == tg.StorageFormat.SparseCSR
        assert t.rows == 3 and t.cols == 3
        assert t.nnz == 6

    def test_to_numpy_raises_for_sparse(self):
        t = make_triangle_csr()
        with pytest.raises(ValueError, match="Dense"):
            t.to_numpy()


# ============================================================================
#  Ops — matmul & spmm
# ============================================================================

class TestOps:
    def test_matmul_identity(self):
        I = tg.Tensor.from_numpy(np.eye(3, dtype=np.float32))
        X = tg.Tensor.from_numpy(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
        Y = tg.matmul(I, X)
        np.testing.assert_allclose(Y.to_numpy(), X.to_numpy())

    def test_matmul_values(self):
        A = tg.Tensor.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))
        B = tg.Tensor.from_numpy(np.array([[5, 6], [7, 8]], dtype=np.float32))
        C = tg.matmul(A, B)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_allclose(C.to_numpy(), expected)

    def test_spmm(self):
        # Identity-like CSR × Dense
        rp = [0, 1, 2, 3]
        ci = [0, 1, 2]
        vals = [1.0, 1.0, 1.0]
        A = tg.Tensor.sparse_csr(3, 3, rp, ci, vals)
        B = tg.Tensor.from_numpy(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
        C = tg.spmm(A, B)
        np.testing.assert_allclose(C.to_numpy(), B.to_numpy())

    def test_spmm_aggregation(self):
        """Row 0 sums rows 0+1 of B, Row 1 sums rows 1+2."""
        rp = [0, 2, 4]
        ci = [0, 1, 1, 2]
        vals = [1.0, 1.0, 1.0, 1.0]
        A = tg.Tensor.sparse_csr(2, 3, rp, ci, vals)
        B = tg.Tensor.from_numpy(np.array([[1, 0], [0, 1], [2, 2]], dtype=np.float32))
        C = tg.spmm(A, B)
        np.testing.assert_allclose(C.to_numpy(), [[1, 1], [2, 3]])


# ============================================================================
#  Activations
# ============================================================================

class TestActivations:
    def _make(self, data):
        arr = np.array(data, dtype=np.float32).reshape(1, -1)
        return tg.Tensor.from_numpy(arr)

    def test_relu(self):
        t = self._make([-2, -1, 0, 1, 2])
        tg.relu_inplace(t)
        np.testing.assert_allclose(t.to_numpy().flatten(), [0, 0, 0, 1, 2])

    def test_leaky_relu(self):
        t = self._make([-10, 0, 10])
        tg.leaky_relu_inplace(t, 0.1)
        np.testing.assert_allclose(t.to_numpy().flatten(), [-1.0, 0.0, 10.0])

    def test_elu(self):
        t = self._make([0, 1, -1])
        tg.elu_inplace(t, 1.0)
        out = t.to_numpy().flatten()
        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(1.0)
        assert out[2] == pytest.approx(math.exp(-1) - 1, abs=1e-5)

    def test_sigmoid(self):
        t = self._make([0])
        tg.sigmoid_inplace(t)
        assert t.to_numpy().flatten()[0] == pytest.approx(0.5)

    def test_tanh(self):
        t = self._make([0])
        tg.tanh_inplace(t)
        assert t.to_numpy().flatten()[0] == pytest.approx(0.0)

    def test_gelu(self):
        t = self._make([0])
        tg.gelu_inplace(t)
        assert t.to_numpy().flatten()[0] == pytest.approx(0.0, abs=1e-4)

    def test_softmax(self):
        t = self._make([1, 2, 3])
        tg.softmax_inplace(t)
        out = t.to_numpy().flatten()
        assert sum(out) == pytest.approx(1.0, abs=1e-5)
        assert out[2] > out[1] > out[0]

    def test_log_softmax(self):
        t = self._make([1, 2, 3])
        tg.log_softmax_inplace(t)
        out = t.to_numpy().flatten()
        # Verify exp of log_softmax sums to 1
        assert sum(np.exp(out)) == pytest.approx(1.0, abs=1e-5)

    def test_add_bias(self):
        X = tg.Tensor.from_numpy(np.ones((3, 2), dtype=np.float32))
        b = tg.Tensor.from_numpy(np.array([[10.0, 20.0]], dtype=np.float32))
        tg.add_bias(X, b)  # in-place
        expected = np.array([[11, 21], [11, 21], [11, 21]], dtype=np.float32)
        np.testing.assert_allclose(X.to_numpy(), expected)


# ============================================================================
#  Graph utilities
# ============================================================================

class TestGraphUtils:
    def test_add_self_loops(self):
        # Simple 2-node: 0→1
        rp = [0, 1, 1]
        ci = [1]
        vals = [1.0]
        A = tg.Tensor.sparse_csr(2, 2, rp, ci, vals)
        A_hat = tg.add_self_loops(A)
        # Should have 0→0, 0→1, 1→1
        assert A_hat.nnz == 3

    def test_gcn_norm(self):
        A = make_triangle_csr()
        A_norm = tg.gcn_norm(A)
        assert A_norm.format == tg.StorageFormat.SparseCSR
        # Each row should sum to ~1 for a regular graph with self-loops
        # GCN norm includes self-loops: D^{-1/2} (A+I) D^{-1/2}
        rp = A_norm.row_ptr_numpy()
        vals = A_norm.values_numpy()
        for i in range(3):
            row_sum = sum(vals[rp[i]:rp[i+1]])
            assert row_sum == pytest.approx(1.0, abs=1e-5)

    def test_edge_softmax(self):
        A = make_triangle_csr()
        S = tg.edge_softmax(A)
        # Each row's values should sum to 1.0
        rp = S.row_ptr_numpy()
        vals = S.values_numpy()
        for i in range(3):
            row_sum = sum(vals[rp[i]:rp[i+1]])
            assert row_sum == pytest.approx(1.0, abs=1e-5)

    def test_sage_max_aggregate(self):
        A = make_triangle_csr()
        H = tg.Tensor.from_numpy(np.array([
            [1, 0],
            [0, 5],
            [3, 2],
        ], dtype=np.float32))
        agg = tg.sage_max_aggregate(A, H)
        out = agg.to_numpy()
        # Node 0's neighbors: {1, 2} → max([0,5],[3,2]) = [3,5]
        np.testing.assert_allclose(out[0], [3, 5])
        # Node 1's neighbors: {0, 2} → max([1,0],[3,2]) = [3,2]
        np.testing.assert_allclose(out[1], [3, 2])
        # Node 2's neighbors: {0, 1} → max([1,0],[0,5]) = [1,5]
        np.testing.assert_allclose(out[2], [1, 5])


# ============================================================================
#  GCNLayer
# ============================================================================

class TestGCNLayer:
    def test_construction(self):
        layer = tg.GCNLayer(16, 32, True, tg.Activation.RELU)
        assert layer.in_features == 16
        assert layer.out_features == 32
        assert layer.has_bias is True
        assert layer.activation == tg.Activation.RELU

    def test_forward_identity_weights(self):
        """With identity weight and zero bias, output ≈ A_norm @ H."""
        A = make_triangle_csr()
        A_norm = tg.gcn_norm(A)
        H = tg.Tensor.from_numpy(np.array([
            [1, 0],
            [0, 1],
            [1, 1],
        ], dtype=np.float32))

        layer = tg.GCNLayer(2, 2, True, tg.Activation.NONE)
        W = tg.Tensor.from_numpy(np.eye(2, dtype=np.float32))
        b = tg.Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32))
        layer.set_weight(W)
        layer.set_bias(b)

        out = layer.forward(A_norm, H)
        result = out.to_numpy()

        # With identity weight and zero bias:
        # out = A_norm @ (H @ I) + 0 = A_norm @ H
        # Verify shape at minimum
        assert result.shape == (3, 2)

    def test_forward_with_relu(self):
        A = make_triangle_csr()
        A_norm = tg.gcn_norm(A)
        H = tg.Tensor.from_numpy(np.ones((3, 2), dtype=np.float32))

        layer = tg.GCNLayer(2, 2, True, tg.Activation.RELU)
        # Negative weights → should produce negative pre-activation → ReLU clips
        W = tg.Tensor.from_numpy(-np.eye(2, dtype=np.float32))
        b = tg.Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32))
        layer.set_weight(W)
        layer.set_bias(b)

        out = layer.forward(A_norm, H)
        result = out.to_numpy()
        # ReLU: all values should be >= 0
        assert np.all(result >= 0)


# ============================================================================
#  SAGELayer
# ============================================================================

class TestSAGELayer:
    def test_construction_mean(self):
        layer = tg.SAGELayer(8, 4, tg.SAGELayer.Aggregator.Mean)
        assert layer.in_features == 8
        assert layer.out_features == 4
        assert layer.aggregator == tg.SAGELayer.Aggregator.Mean

    def test_construction_max(self):
        layer = tg.SAGELayer(8, 4, tg.SAGELayer.Aggregator.Max)
        assert layer.aggregator == tg.SAGELayer.Aggregator.Max

    def test_forward_mean(self):
        A = make_triangle_csr()
        H = tg.Tensor.from_numpy(np.ones((3, 2), dtype=np.float32))

        layer = tg.SAGELayer(2, 2, tg.SAGELayer.Aggregator.Mean, True, tg.Activation.NONE)
        layer.set_weight_neigh(tg.Tensor.from_numpy(np.eye(2, dtype=np.float32)))
        layer.set_weight_self(tg.Tensor.from_numpy(np.eye(2, dtype=np.float32)))
        layer.set_bias(tg.Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32)))

        out = layer.forward(A, H)
        result = out.to_numpy()
        assert result.shape == (3, 2)
        # With identity weights, zero bias, all-ones input:
        # mean_agg = mean of neighbors = 1, self = 1 → out = 1 + 1 = 2
        np.testing.assert_allclose(result, 2.0, atol=1e-5)

    def test_forward_max(self):
        A = make_triangle_csr()
        H = tg.Tensor.from_numpy(np.array([
            [1, 0],
            [0, 3],
            [2, 1],
        ], dtype=np.float32))

        layer = tg.SAGELayer(2, 2, tg.SAGELayer.Aggregator.Max, True, tg.Activation.NONE)
        layer.set_weight_neigh(tg.Tensor.from_numpy(np.eye(2, dtype=np.float32)))
        layer.set_weight_self(tg.Tensor.from_numpy(np.eye(2, dtype=np.float32)))
        layer.set_bias(tg.Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32)))

        out = layer.forward(A, H)
        result = out.to_numpy()
        assert result.shape == (3, 2)
        # Node 0: max_agg([0,3],[2,1])=[2,3], self=[1,0] → [3,3]
        np.testing.assert_allclose(result[0], [3, 3], atol=1e-5)


# ============================================================================
#  GATLayer
# ============================================================================

class TestGATLayer:
    def test_construction(self):
        layer = tg.GATLayer(16, 8, 0.2, True, tg.Activation.NONE)
        assert layer.in_features == 16
        assert layer.out_features == 8
        assert layer.negative_slope == pytest.approx(0.2)
        assert layer.has_bias is True

    def test_forward(self):
        """Basic forward pass: check shape and non-NaN."""
        A = make_triangle_csr()
        A_sl = tg.add_self_loops(A)
        H = tg.Tensor.from_numpy(np.random.randn(3, 4).astype(np.float32))

        layer = tg.GATLayer(4, 2, 0.2, True, tg.Activation.NONE)
        layer.set_weight(tg.Tensor.from_numpy(
            np.random.randn(4, 2).astype(np.float32) * 0.1))
        layer.set_attn_left(tg.Tensor.from_numpy(
            np.random.randn(1, 2).astype(np.float32) * 0.1))
        layer.set_attn_right(tg.Tensor.from_numpy(
            np.random.randn(1, 2).astype(np.float32) * 0.1))
        layer.set_bias(tg.Tensor.from_numpy(np.zeros((1, 2), dtype=np.float32)))

        out = layer.forward(A_sl, H)
        result = out.to_numpy()
        assert result.shape == (3, 2)
        assert not np.any(np.isnan(result))


# ============================================================================
#  Model
# ============================================================================

class TestModel:
    def _write_weight_file(self, path, tensors, accuracy=0.85):
        """Write a TGNN binary weight file."""
        with open(path, "wb") as f:
            f.write(b"TGNN")
            f.write(struct.pack("<I", 1))  # version
            f.write(struct.pack("<f", accuracy))
            f.write(struct.pack("<I", len(tensors)))
            for name, arr in tensors.items():
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                rows, cols = arr.shape
                f.write(struct.pack("<I", rows))
                f.write(struct.pack("<I", cols))
                f.write(arr.astype(np.float32).tobytes())

    def test_gcn_model(self):
        """Build a 2-layer GCN model, write weights, load, and forward."""
        model = tg.Model()
        model.add_gcn_layer(4, 3, True, tg.Activation.RELU)
        model.add_gcn_layer(3, 2, True, tg.Activation.NONE)
        assert model.num_layers == 2

        # Create weight file
        np.random.seed(42)
        tensors = {
            "layer0.weight": np.random.randn(4, 3).astype(np.float32) * 0.1,
            "layer0.bias": np.zeros((1, 3), dtype=np.float32),
            "layer1.weight": np.random.randn(3, 2).astype(np.float32) * 0.1,
            "layer1.bias": np.zeros((1, 2), dtype=np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            wpath = f.name
        try:
            self._write_weight_file(wpath, tensors)
            model.load_weights(wpath)

            # Forward pass
            A = make_triangle_csr()
            H = tg.Tensor.from_numpy(np.random.randn(3, 4).astype(np.float32))
            out = model.forward(A, H)
            result = out.to_numpy()
            assert result.shape == (3, 2)
            assert not np.any(np.isnan(result))
        finally:
            os.unlink(wpath)

    def test_sage_model(self):
        """Build a 2-layer SAGE model, write weights, load, and forward."""
        model = tg.Model()
        model.add_sage_layer(4, 3)
        model.add_sage_layer(3, 2, tg.SAGELayer.Aggregator.Mean, True, tg.Activation.NONE)
        assert model.num_layers == 2

        np.random.seed(43)
        tensors = {
            "layer0.weight_neigh": np.random.randn(4, 3).astype(np.float32) * 0.1,
            "layer0.weight_self": np.random.randn(4, 3).astype(np.float32) * 0.1,
            "layer0.bias": np.zeros((1, 3), dtype=np.float32),
            "layer1.weight_neigh": np.random.randn(3, 2).astype(np.float32) * 0.1,
            "layer1.weight_self": np.random.randn(3, 2).astype(np.float32) * 0.1,
            "layer1.bias": np.zeros((1, 2), dtype=np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            wpath = f.name
        try:
            self._write_weight_file(wpath, tensors)
            model.load_weights(wpath)

            A = make_triangle_csr()
            H = tg.Tensor.from_numpy(np.random.randn(3, 4).astype(np.float32))
            out = model.forward(A, H)
            result = out.to_numpy()
            assert result.shape == (3, 2)
            assert not np.any(np.isnan(result))
        finally:
            os.unlink(wpath)

    def test_load_weight_file(self):
        """Test load_weight_file standalone function."""
        np.random.seed(44)
        tensors = {
            "foo": np.random.randn(2, 3).astype(np.float32),
            "bar": np.random.randn(1, 5).astype(np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            wpath = f.name
        try:
            self._write_weight_file(wpath, tensors, accuracy=0.92)
            wf = tg.load_weight_file(wpath)
            assert wf.test_accuracy == pytest.approx(0.92, abs=1e-5)
            assert "foo" in wf.tensors
            assert "bar" in wf.tensors
            foo = wf.tensors["foo"]
            assert foo.rows == 2 and foo.cols == 3
        finally:
            os.unlink(wpath)


# ============================================================================
#  Enum tests
# ============================================================================

class TestEnums:
    def test_storage_format(self):
        assert tg.StorageFormat.Dense != tg.StorageFormat.SparseCSR

    def test_activation(self):
        assert tg.Activation.NONE != tg.Activation.RELU

    def test_sage_aggregator(self):
        assert tg.SAGELayer.Aggregator.Mean != tg.SAGELayer.Aggregator.Max


# ============================================================================
#  Error handling
# ============================================================================

class TestErrors:
    def test_from_numpy_wrong_ndim(self):
        arr = np.ones((2, 3, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            tg.Tensor.from_numpy(arr)

    def test_to_numpy_sparse_raises(self):
        t = make_triangle_csr()
        with pytest.raises(ValueError):
            t.to_numpy()

    def test_edge_index_wrong_shape(self):
        arr = np.ones((3, 4), dtype=np.int32)
        with pytest.raises(ValueError):
            tg.Tensor.from_edge_index(arr, 5)


# ============================================================================
#  Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
