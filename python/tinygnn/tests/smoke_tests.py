"""
tinygnn.tests.smoke_tests — self-contained post-install validation tests.

Each test is a plain function returning True/False (or raising on failure).
No pytest / unittest dependency is required so users can run this without
any test framework being installed.

Tests cover:
  C1  Tensor creation (Dense, SparseCSR)
  C2  Tensor numpy round-trip
  C3  matmul correctness
  C4  spmm correctness
  C5  relu_inplace
  C6  add_self_loops
  C7  gcn_norm (sum to 1 check)
  C8  GCNLayer forward pass (shape + no NaN)
  C9  SAGELayer (Mean) forward pass
  C10 SAGELayer (Max) forward pass
  C11 GATLayer forward pass
  C12 2-layer GCN Model
"""

from __future__ import annotations

import math
import os
import struct
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Import the C++ extension.  Whether installed via pip or dev-install, the
# extension lands either as:
#   _tinygnn_core  (flat: python -c "import _tinygnn_core")
# The tinygnn package does "from _tinygnn_core import …" in __init__.py.
# We import via the package so we get both the Python helpers and the C++.
# ---------------------------------------------------------------------------
try:
    import tinygnn as _tg_pkg  # noqa: F401 – ensure the package path is on sys.path
    import _tinygnn_core as tg
except ImportError:
    # Maybe we're running from the source tree without `pip install -e .`
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
    for _p in [_ROOT, os.path.join(_ROOT, "python")]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import _tinygnn_core as tg


# ============================================================================
#  Helpers
# ============================================================================

def _dense(rows, cols, data):
    return tg.Tensor.dense_from_data(rows, cols, data)


def _triangle_csr():
    """3-node triangle (undirected): edges 0-1, 1-2, 0-2."""
    return tg.Tensor.sparse_csr(3, 3,
                                [0, 2, 4, 6],
                                [1, 2, 0, 2, 0, 1],
                                [1.0] * 6)


def _allclose(a: np.ndarray, b, atol: float = 1e-4) -> bool:
    b_arr = np.asarray(b, dtype=np.float32)
    return bool(np.allclose(a.astype(np.float32), b_arr, atol=atol))


# ============================================================================
#  Test registry
# ============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    error: Optional[str] = None


@dataclass
class _Registry:
    tests: List[Callable] = field(default_factory=list)

    def register(self, fn: Callable) -> Callable:
        self.tests.append(fn)
        return fn


_registry = _Registry()
register = _registry.register


# ============================================================================
#  Tests
# ============================================================================

@register
def C1_tensor_creation():
    """Dense + SparseCSR construction."""
    t = tg.Tensor.dense(4, 3)
    assert t.rows == 4 and t.cols == 3 and t.numel == 12
    s = _triangle_csr()
    assert s.nnz == 6 and s.rows == 3 and s.cols == 3
    return True


@register
def C2_numpy_roundtrip():
    """from_numpy ↔ to_numpy round-trip."""
    arr = np.random.randn(5, 7).astype(np.float32)
    out = tg.Tensor.from_numpy(arr).to_numpy()
    assert _allclose(out, arr)
    return True


@register
def C3_matmul():
    """2×2 matrix multiply correctness."""
    A = _dense(2, 2, [1, 2, 3, 4])
    B = _dense(2, 2, [5, 6, 7, 8])
    C = tg.matmul(A, B)
    out = C.to_numpy()
    assert _allclose(out, [[19, 22], [43, 50]])
    return True


@register
def C4_spmm():
    """SpMM: 2×3 sparse × 3×2 dense."""
    A = tg.Tensor.sparse_csr(2, 3, [0, 2, 4], [0, 1, 1, 2], [1.0, 1.0, 1.0, 1.0])
    B = tg.Tensor.from_numpy(np.array([[1, 0], [0, 1], [2, 2]], dtype=np.float32))
    C = tg.spmm(A, B)
    assert _allclose(C.to_numpy(), [[1, 1], [2, 3]])
    return True


@register
def C5_relu():
    """relu_inplace clips negatives to zero."""
    arr = np.array([[-3, -1, 0, 1, 4]], dtype=np.float32)
    t = tg.Tensor.from_numpy(arr)
    tg.relu_inplace(t)
    assert _allclose(t.to_numpy(), [[0, 0, 0, 1, 4]])
    return True


@register
def C6_add_self_loops():
    """add_self_loops: nnz increases by N."""
    A = _triangle_csr()   # 6 edges, no self-loops
    A2 = tg.add_self_loops(A)
    assert A2.nnz == 6 + 3
    return True


@register
def C7_gcn_norm():
    """gcn_norm: each row of A_norm @ ones ≈ degree / sqrt(deg * deg)."""
    A = _triangle_csr()
    A_sl = tg.add_self_loops(A)
    A_norm = tg.gcn_norm(A_sl)
    ones = tg.Tensor.from_numpy(np.ones((3, 1), dtype=np.float32))
    # Row sums of normalised adjacency; for undirected k-regular graph should be < 1
    result = tg.spmm(A_norm, ones).to_numpy().flatten()
    assert all(r > 0 and r <= 1.0 + 1e-5 for r in result)
    return True


@register
def C8_gcn_layer():
    """GCNLayer: shape correct, no NaN/Inf."""
    A  = tg.gcn_norm(tg.add_self_loops(_triangle_csr()))
    H  = tg.Tensor.from_numpy(np.random.randn(3, 4).astype(np.float32))
    ly = tg.GCNLayer(4, 8, True, tg.Activation.RELU)
    ly.set_weight(tg.Tensor.from_numpy(np.random.randn(4, 8).astype(np.float32) * 0.1))
    ly.set_bias(tg.Tensor.from_numpy(np.zeros((1, 8), dtype=np.float32)))
    out = ly.forward(A, H).to_numpy()
    assert out.shape == (3, 8)
    assert not np.any(np.isnan(out) | np.isinf(out))
    return True


@register
def C9_sage_mean():
    """SAGELayer (Mean): shape correct, no NaN."""
    A  = _triangle_csr()
    H  = tg.Tensor.from_numpy(np.random.randn(3, 4).astype(np.float32))
    ly = tg.SAGELayer(4, 8, tg.SAGELayer.Aggregator.Mean, True, tg.Activation.RELU)
    ly.set_weight_neigh(tg.Tensor.from_numpy(np.random.randn(4, 8).astype(np.float32) * 0.1))
    ly.set_weight_self(tg.Tensor.from_numpy(np.random.randn(4, 8).astype(np.float32) * 0.1))
    ly.set_bias(tg.Tensor.from_numpy(np.zeros((1, 8), dtype=np.float32)))
    out = ly.forward(A, H).to_numpy()
    assert out.shape == (3, 8)
    assert not np.any(np.isnan(out) | np.isinf(out))
    return True


@register
def C10_sage_max():
    """SAGELayer (Max): shape correct, no NaN."""
    A  = _triangle_csr()
    H  = tg.Tensor.from_numpy(np.random.randn(3, 4).astype(np.float32))
    ly = tg.SAGELayer(4, 8, tg.SAGELayer.Aggregator.Max, True, tg.Activation.RELU)
    ly.set_weight_neigh(tg.Tensor.from_numpy(np.random.randn(4, 8).astype(np.float32) * 0.1))
    ly.set_weight_self(tg.Tensor.from_numpy(np.random.randn(4, 8).astype(np.float32) * 0.1))
    ly.set_bias(tg.Tensor.from_numpy(np.zeros((1, 8), dtype=np.float32)))
    out = ly.forward(A, H).to_numpy()
    assert out.shape == (3, 8)
    assert not np.any(np.isnan(out) | np.isinf(out))
    return True


@register
def C11_gat_layer():
    """GATLayer: shape correct, no NaN."""
    A  = tg.add_self_loops(_triangle_csr())
    H  = tg.Tensor.from_numpy(np.random.randn(3, 4).astype(np.float32))
    ly = tg.GATLayer(4, 8, 0.2, True, tg.Activation.RELU)
    ly.set_weight(tg.Tensor.from_numpy(np.random.randn(4, 8).astype(np.float32) * 0.1))
    ly.set_attn_left(tg.Tensor.from_numpy(np.random.randn(1, 8).astype(np.float32) * 0.1))
    ly.set_attn_right(tg.Tensor.from_numpy(np.random.randn(1, 8).astype(np.float32) * 0.1))
    ly.set_bias(tg.Tensor.from_numpy(np.zeros((1, 8), dtype=np.float32)))
    out = ly.forward(A, H).to_numpy()
    assert out.shape == (3, 8)
    assert not np.any(np.isnan(out) | np.isinf(out))
    return True


@register
def C12_two_layer_model():
    """2-layer GCN Model: build, load weights, forward, shape OK."""
    model = tg.Model()
    model.add_gcn_layer(4, 8, True, tg.Activation.RELU)
    model.add_gcn_layer(8, 3, True, tg.Activation.NONE)
    assert model.num_layers == 2

    rng = np.random.default_rng(0)
    tensors = {
        "layer0.weight": rng.standard_normal((4, 8)).astype(np.float32) * 0.1,
        "layer0.bias":   np.zeros((1, 8), dtype=np.float32),
        "layer1.weight": rng.standard_normal((8, 3)).astype(np.float32) * 0.1,
        "layer1.bias":   np.zeros((1, 3), dtype=np.float32),
    }

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        wpath = f.name

    try:
        # Write TGNN weight file
        with open(wpath, "wb") as fh:
            fh.write(b"TGNN")
            fh.write(struct.pack("<I", 1))
            fh.write(struct.pack("<f", 0.0))
            fh.write(struct.pack("<I", len(tensors)))
            for name, arr in tensors.items():
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                nb = name.encode("utf-8")
                fh.write(struct.pack("<I", len(nb)))
                fh.write(nb)
                r, c = arr.shape
                fh.write(struct.pack("<II", r, c))
                fh.write(arr.astype(np.float32).tobytes())

        model.load_weights(wpath)
        A = tg.gcn_norm(tg.add_self_loops(_triangle_csr()))
        H = tg.Tensor.from_numpy(np.random.randn(3, 4).astype(np.float32))
        out = model.forward(A, H).to_numpy()
        assert out.shape == (3, 3)
        assert not np.any(np.isnan(out) | np.isinf(out))
    finally:
        os.unlink(wpath)

    return True


# ============================================================================
#  Runner
# ============================================================================

def run_all(verbose: bool = False) -> List[TestResult]:
    results: List[TestResult] = []
    n_pass = 0
    n_fail = 0

    print(f"\n  TinyGNN — post-install smoke tests  ({len(_registry.tests)} tests)\n")

    for fn in _registry.tests:
        name = fn.__name__
        doc  = (fn.__doc__ or "").strip().split("\n")[0]
        try:
            ok  = fn()
            ok  = bool(ok)
        except Exception:
            ok  = False
            err = traceback.format_exc()
            results.append(TestResult(name, False, err))
            if verbose:
                print(f"  FAIL  {name}: {doc}")
                print(err, end="")
            else:
                print(f"  FAIL  {name}: {doc}")
            n_fail += 1
            continue

        results.append(TestResult(name, ok))
        status = "PASS" if ok else "FAIL"
        if verbose or not ok:
            print(f"  {status}  {name}: {doc}")
        else:
            print(f"  {status}  {name}: {doc}")

        if ok:
            n_pass += 1
        else:
            n_fail += 1

    print()
    print(f"  Results: {n_pass} passed, {n_fail} failed / {len(_registry.tests)} total")

    if n_fail == 0:
        print("  All smoke tests PASSED. TinyGNN installation is healthy.\n")
    else:
        print("  Some tests FAILED. Check the error messages above.\n")

    return results


def _main():
    import argparse
    parser = argparse.ArgumentParser(description="TinyGNN post-install smoke tests")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    results = run_all(verbose=args.verbose)
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    _main()
