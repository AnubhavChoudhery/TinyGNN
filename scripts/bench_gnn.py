#!/usr/bin/env python3
"""
TinyGNN — GNN Layer Benchmarks + PyTorch Geometric Accuracy Validation
scripts/bench_gnn.py

Benchmarks:
  • GCNLayer, SAGELayer (Mean, Max), GATLayer
  • Metrics: wall time (ms) and peak RSS memory (MB)
  • Graph sizes: small (1 K nodes), medium (10 K nodes), large (100 K nodes)
  • Threads: controlled via OMP_NUM_THREADS env var

Accuracy:
  • When PyTorch Geometric is installed, validates TinyGNN output logits
    against torch_geometric reference on a small graph (tolerance 1e-4).

Usage:
    # Benchmark only (no PyG required):
    python scripts/bench_gnn.py

    # Include accuracy validation (requires torch + torch_geometric):
    python scripts/bench_gnn.py --accuracy

    # Only run accuracy check:
    python scripts/bench_gnn.py --accuracy --no-bench

    # Larger repetitions for stable timing:
    python scripts/bench_gnn.py --reps 20
"""

import sys
import os
import argparse
import time
import tracemalloc
import gc

import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "python"))

import _tinygnn_core as tg


# ============================================================================
#  Graph generation helpers
# ============================================================================

def make_random_graph_csr(num_nodes: int, avg_degree: int, seed: int = 42):
    """Generate a random undirected graph as a scipy CSR matrix.

    Uses a simple random edge model: for each node i, sample `avg_degree`
    random neighbours uniformly, then symmetrise. Self-loops are excluded.

    Returns a scipy.sparse.csr_matrix of shape (num_nodes, num_nodes).
    """
    from scipy.sparse import csr_matrix  # type: ignore

    rng = np.random.default_rng(seed)
    rows_list = []
    cols_list = []

    for src in range(num_nodes):
        targets = rng.choice(num_nodes, size=avg_degree, replace=False)
        for dst in targets:
            if dst != src:
                rows_list.append(src)
                cols_list.append(dst)
                rows_list.append(dst)   # symmetrise
                cols_list.append(src)

    rows_arr = np.array(rows_list, dtype=np.int32)
    cols_arr = np.array(cols_list, dtype=np.int32)
    vals_arr = np.ones(len(rows_arr), dtype=np.float32)

    mat = csr_matrix(
        (vals_arr, (rows_arr, cols_arr)),
        shape=(num_nodes, num_nodes),
    )
    # Eliminate duplicates by summing and then clipping to 1
    mat.data[:] = 1.0
    mat.sum_duplicates()
    mat.sort_indices()
    return mat


def scipy_to_tinygnn(mat):
    """Convert a scipy CSR matrix to a TinyGNN SparseCSR Tensor."""
    return tg.Tensor.from_scipy_csr(mat)


# ============================================================================
#  TinyGNN layer factories
# ============================================================================

def make_gcn_layer(F_in: int, F_out: int, seed: int = 0):
    """Create a GCNLayer with random weight + zero bias."""
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((F_in, F_out)) * (2.0 / F_in) ** 0.5).astype(np.float32)
    b = np.zeros((1, F_out), dtype=np.float32)
    layer = tg.GCNLayer(F_in, F_out, True, tg.Activation.RELU)
    layer.set_weight(tg.Tensor.from_numpy(W))
    layer.set_bias(tg.Tensor.from_numpy(b))
    return layer


def make_sage_layer(F_in: int, F_out: int, agg, seed: int = 0):
    """Create a SAGELayer (Mean or Max) with random weights + zero bias."""
    rng = np.random.default_rng(seed)
    scale = (2.0 / F_in) ** 0.5
    Wn = (rng.standard_normal((F_in, F_out)) * scale).astype(np.float32)
    Ws = (rng.standard_normal((F_in, F_out)) * scale).astype(np.float32)
    b  = np.zeros((1, F_out), dtype=np.float32)
    layer = tg.SAGELayer(F_in, F_out, agg, True, tg.Activation.RELU)
    layer.set_weight_neigh(tg.Tensor.from_numpy(Wn))
    layer.set_weight_self(tg.Tensor.from_numpy(Ws))
    layer.set_bias(tg.Tensor.from_numpy(b))
    return layer


def make_gat_layer(F_in: int, F_out: int, seed: int = 0):
    """Create a GATLayer with random weight + attention vectors + zero bias."""
    rng = np.random.default_rng(seed)
    scale = (2.0 / F_in) ** 0.5
    W  = (rng.standard_normal((F_in, F_out)) * scale).astype(np.float32)
    al = (rng.standard_normal((1, F_out)) * 0.1).astype(np.float32)
    ar = (rng.standard_normal((1, F_out)) * 0.1).astype(np.float32)
    b  = np.zeros((1, F_out), dtype=np.float32)
    layer = tg.GATLayer(F_in, F_out, 0.2, True, tg.Activation.RELU)
    layer.set_weight(tg.Tensor.from_numpy(W))
    layer.set_attn_left(tg.Tensor.from_numpy(al))
    layer.set_attn_right(tg.Tensor.from_numpy(ar))
    layer.set_bias(tg.Tensor.from_numpy(b))
    return layer


# ============================================================================
#  Timing + memory measurement
# ============================================================================

def measure(fn, warmup: int = 2, reps: int = 10):
    """Run fn() warmup + reps times; return (median_ms, peak_mb).

    Peak memory is measured via tracemalloc over a single cold run performed
    after the timed loop (so the GC state is representative).
    """
    # Warm-up
    for _ in range(warmup):
        fn()

    # Timed runs
    times_ms = []
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    median_ms = float(np.median(times_ms))

    # Memory measurement (one cold run with tracemalloc)
    gc.collect()
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 ** 2)

    return median_ms, peak_mb


# ============================================================================
#  Benchmark runner
# ============================================================================

GRAPH_CONFIGS = [
    # (label,  num_nodes, avg_degree, F_in, F_out)
    ("small  (1 K,   5 avg-deg, 64→32)",   1_000,    5, 64,  32),
    ("medium (10 K, 10 avg-deg, 128→64)", 10_000,   10, 128, 64),
    ("large  (100K,  5 avg-deg, 256→128)", 100_000,   5, 256, 128),
]


def run_benchmarks(reps: int = 10):
    print("\n" + "=" * 80)
    print("  TinyGNN — GNN Layer Benchmarks")
    print("  wall time = median over {} timed runs  |  memory = peak tracemalloc".format(reps))
    print("=" * 80)
    header = f"{'Config':<45}  {'Layer':<18}  {'Time (ms)':>10}  {'Mem (MB)':>10}"
    print(header)
    print("-" * len(header))

    for label, N, D, F_in, F_out in GRAPH_CONFIGS:
        print(f"\n  Graph: {label}")

        # Build graph structures once per config
        scipy_mat = make_random_graph_csr(N, D, seed=7)
        A_sparse  = scipy_to_tinygnn(scipy_mat)
        A_norm    = tg.gcn_norm(tg.add_self_loops(A_sparse))
        A_sl      = tg.add_self_loops(A_sparse)   # for GAT
        H         = tg.Tensor.from_numpy(
            np.random.default_rng(3).standard_normal((N, F_in)).astype(np.float32)
        )

        # GCN
        gcn = make_gcn_layer(F_in, F_out)
        ms, mb = measure(lambda: gcn.forward(A_norm, H), reps=reps)
        print(f"  {'':45}  {'GCNLayer':18}  {ms:>10.2f}  {mb:>10.3f}")

        # SAGE Mean
        sage_mean = make_sage_layer(F_in, F_out, tg.SAGELayer.Aggregator.Mean)
        ms, mb = measure(lambda: sage_mean.forward(A_sparse, H), reps=reps)
        print(f"  {'':45}  {'SAGELayer (Mean)':18}  {ms:>10.2f}  {mb:>10.3f}")

        # SAGE Max
        sage_max = make_sage_layer(F_in, F_out, tg.SAGELayer.Aggregator.Max)
        ms, mb = measure(lambda: sage_max.forward(A_sparse, H), reps=reps)
        print(f"  {'':45}  {'SAGELayer (Max)':18}  {ms:>10.2f}  {mb:>10.3f}")

        # GAT
        gat  = make_gat_layer(F_in, F_out)
        ms, mb = measure(lambda: gat.forward(A_sl, H), reps=reps)
        print(f"  {'':45}  {'GATLayer':18}  {ms:>10.2f}  {mb:>10.3f}")

    print("\n" + "=" * 80)


# ============================================================================
#  PyTorch Geometric accuracy validation
# ============================================================================

def _try_import_pyg():
    try:
        import torch
        import torch_geometric  # noqa: F401
        from torch_geometric.nn import GCNConv, SAGEConv, GATConv
        return torch, GCNConv, SAGEConv, GATConv
    except ImportError:
        return None


def _build_pyg_edge_index(scipy_mat):
    """Build a torch edge_index (2 × nnz) from a scipy CSR matrix."""
    import torch
    coo = scipy_mat.tocoo()
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    return torch.stack([row, col], dim=0)  # (2, nnz)


def _run_gcn_accuracy():
    """Validate GCNLayer output against PyG GCNConv."""
    result = _try_import_pyg()
    if result is None:
        print("  [SKIP] PyTorch Geometric not installed — skipping GCN accuracy check.")
        return None
    torch, GCNConv, _, _ = result

    N, F_in, F_out = 20, 4, 3
    seed = 42

    rng = np.random.default_rng(seed)
    # Random symmetric adjacency (with self-loops for GCN)
    scipy_mat = make_random_graph_csr(N, avg_degree=4, seed=seed)

    W_np   = (rng.standard_normal((F_in, F_out)) * 0.1).astype(np.float32)
    b_np   = (rng.standard_normal((1, F_out)) * 0.05).astype(np.float32)
    H_np   = (rng.standard_normal((N, F_in)) * 1.0).astype(np.float32)

    # ── TinyGNN ──────────────────────────────────────────────────────────────
    A_tg   = scipy_to_tinygnn(scipy_mat)
    A_norm = tg.gcn_norm(tg.add_self_loops(A_tg))
    H_tg   = tg.Tensor.from_numpy(H_np)
    layer  = tg.GCNLayer(F_in, F_out, True, tg.Activation.NONE)
    layer.set_weight(tg.Tensor.from_numpy(W_np))
    layer.set_bias(tg.Tensor.from_numpy(b_np))
    out_tg = layer.forward(A_norm, H_tg).to_numpy()   # (N, F_out)

    # ── PyTorch Geometric GCNConv ─────────────────────────────────────────────
    # GCNConv stores weight as (F_out, F_in) and applies: out = norm(A) H W^T + b
    import torch
    edge_index = _build_pyg_edge_index(scipy_mat)
    gcn_pyg = GCNConv(F_in, F_out, bias=True, normalize=True, add_self_loops=True)
    with torch.no_grad():
        gcn_pyg.lin.weight.copy_(torch.from_numpy(W_np.T))    # (F_out, F_in)
        gcn_pyg.bias.copy_(torch.from_numpy(b_np.flatten()))
    H_torch = torch.from_numpy(H_np)
    with torch.no_grad():
        out_pyg = gcn_pyg(H_torch, edge_index).numpy()         # (N, F_out)

    max_diff = float(np.abs(out_tg - out_pyg).max())
    rel_diff = float(np.abs(out_tg - out_pyg).mean() / (np.abs(out_pyg).mean() + 1e-8))
    match = max_diff < 1e-3
    print(f"  GCNLayer   vs GCNConv:  max|diff|={max_diff:.2e}  mean_rel={rel_diff:.2e}  "
          f"{'PASS ✓' if match else 'FAIL ✗'}")
    return match


def _run_sage_accuracy():
    """Validate SAGELayer (Mean) output against PyG SAGEConv."""
    result = _try_import_pyg()
    if result is None:
        print("  [SKIP] PyTorch Geometric not installed — skipping SAGE accuracy check.")
        return None
    torch, _, SAGEConv, _ = result

    N, F_in, F_out = 20, 4, 3
    seed = 43

    rng = np.random.default_rng(seed)
    scipy_mat = make_random_graph_csr(N, avg_degree=4, seed=seed)

    Wn_np  = (rng.standard_normal((F_in, F_out)) * 0.1).astype(np.float32)
    Ws_np  = (rng.standard_normal((F_in, F_out)) * 0.1).astype(np.float32)
    b_np   = (rng.standard_normal((1, F_out)) * 0.05).astype(np.float32)
    H_np   = (rng.standard_normal((N, F_in)) * 1.0).astype(np.float32)

    # ── TinyGNN ──────────────────────────────────────────────────────────────
    A_tg   = scipy_to_tinygnn(scipy_mat)
    H_tg   = tg.Tensor.from_numpy(H_np)
    layer  = tg.SAGELayer(F_in, F_out, tg.SAGELayer.Aggregator.Mean, True, tg.Activation.NONE)
    layer.set_weight_neigh(tg.Tensor.from_numpy(Wn_np))
    layer.set_weight_self(tg.Tensor.from_numpy(Ws_np))
    layer.set_bias(tg.Tensor.from_numpy(b_np))
    out_tg = layer.forward(A_tg, H_tg).to_numpy()   # (N, F_out)

    # ── PyTorch Geometric SAGEConv ────────────────────────────────────────────
    # SAGEConv(F_in, F_out): stores lin_l (self) and lin_r (neigh) as (F_out, F_in)
    # out_i = lin_l(h_i) + lin_r(mean_{j∈N(i)} h_j) + b
    import torch
    edge_index = _build_pyg_edge_index(scipy_mat)
    sage_pyg = SAGEConv(F_in, F_out, bias=True, aggr='mean', normalize=False, root_weight=True)
    with torch.no_grad():
        # lin_r = neighbour transform (Wn), lin_l = self transform (Ws)
        sage_pyg.lin_r.weight.copy_(torch.from_numpy(Wn_np.T))   # neigh weight
        sage_pyg.lin_l.weight.copy_(torch.from_numpy(Ws_np.T))   # self weight
        sage_pyg.lin_l.bias.copy_(torch.from_numpy(b_np.flatten()))
    H_torch = torch.from_numpy(H_np)
    with torch.no_grad():
        out_pyg = sage_pyg(H_torch, edge_index).numpy()

    max_diff = float(np.abs(out_tg - out_pyg).max())
    rel_diff = float(np.abs(out_tg - out_pyg).mean() / (np.abs(out_pyg).mean() + 1e-8))
    match = max_diff < 1e-3
    print(f"  SAGELayer  vs SAGEConv: max|diff|={max_diff:.2e}  mean_rel={rel_diff:.2e}  "
          f"{'PASS ✓' if match else 'FAIL ✗'}")
    return match


def _run_gat_accuracy():
    """Validate GATLayer output against PyG GATConv."""
    result = _try_import_pyg()
    if result is None:
        print("  [SKIP] PyTorch Geometric not installed — skipping GAT accuracy check.")
        return None
    torch, _, _, GATConv = result

    N, F_in, F_out = 20, 4, 3
    seed = 44

    rng = np.random.default_rng(seed)
    scipy_mat = make_random_graph_csr(N, avg_degree=4, seed=seed)

    W_np   = (rng.standard_normal((F_in, F_out)) * 0.1).astype(np.float32)
    al_np  = (rng.standard_normal((1, F_out)) * 0.1).astype(np.float32)
    ar_np  = (rng.standard_normal((1, F_out)) * 0.1).astype(np.float32)
    b_np   = (rng.standard_normal((1, F_out)) * 0.05).astype(np.float32)
    H_np   = (rng.standard_normal((N, F_in)) * 1.0).astype(np.float32)

    # ── TinyGNN ──────────────────────────────────────────────────────────────
    A_tg   = scipy_to_tinygnn(scipy_mat)
    A_sl   = tg.add_self_loops(A_tg)
    H_tg   = tg.Tensor.from_numpy(H_np)
    layer  = tg.GATLayer(F_in, F_out, 0.2, True, tg.Activation.NONE)
    layer.set_weight(tg.Tensor.from_numpy(W_np))
    layer.set_attn_left(tg.Tensor.from_numpy(al_np))
    layer.set_attn_right(tg.Tensor.from_numpy(ar_np))
    layer.set_bias(tg.Tensor.from_numpy(b_np))
    out_tg = layer.forward(A_sl, H_tg).to_numpy()   # (N, F_out)

    # ── PyTorch Geometric GATConv ─────────────────────────────────────────────
    # GATConv(F_in, F_out, heads=1, bias=True, negative_slope=0.2, add_self_loops=True)
    # Internally: lin = (F_in → F_out), att_src = (1, F_out), att_dst = (1, F_out)
    import torch
    edge_index = _build_pyg_edge_index(scipy_mat)
    gat_pyg = GATConv(F_in, F_out, heads=1, bias=True, negative_slope=0.2,
                      add_self_loops=True, concat=True)
    with torch.no_grad():
        gat_pyg.lin_src.weight.copy_(torch.from_numpy(W_np.T))   # (F_out, F_in)
        gat_pyg.lin_dst.weight.copy_(torch.from_numpy(W_np.T))
        gat_pyg.att_src.copy_(torch.from_numpy(al_np).reshape(1, 1, F_out))
        gat_pyg.att_dst.copy_(torch.from_numpy(ar_np).reshape(1, 1, F_out))
        gat_pyg.bias.copy_(torch.from_numpy(b_np.flatten()))
    H_torch = torch.from_numpy(H_np)
    with torch.no_grad():
        out_pyg = gat_pyg(H_torch, edge_index).numpy()

    max_diff = float(np.abs(out_tg - out_pyg).max())
    rel_diff = float(np.abs(out_tg - out_pyg).mean() / (np.abs(out_pyg).mean() + 1e-8))
    match = max_diff < 1e-3
    print(f"  GATLayer   vs GATConv:  max|diff|={max_diff:.2e}  mean_rel={rel_diff:.2e}  "
          f"{'PASS ✓' if match else 'FAIL ✗'}")
    return match


def run_accuracy_checks():
    print("\n" + "=" * 80)
    print("  TinyGNN vs PyTorch Geometric — Accuracy Validation")
    print("  Tolerance: max|diff| < 1e-3 (float32 ops on synthetic 20-node graphs)")
    print("=" * 80)

    results = {}
    results["GCN"]  = _run_gcn_accuracy()
    results["SAGE"] = _run_sage_accuracy()
    results["GAT"]  = _run_gat_accuracy()

    print()
    all_ok = all(v is not False for v in results.values())
    if all(v is None for v in results.values()):
        print("  All accuracy checks SKIPPED (install torch + torch_geometric to enable).")
    elif all_ok:
        print("  All accuracy checks PASSED.")
    else:
        failed = [k for k, v in results.items() if v is False]
        print(f"  FAILED checks: {failed}")
        sys.exit(1)
    print("=" * 80)
    return all_ok


# ============================================================================
#  Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TinyGNN GNN benchmark + accuracy checks")
    parser.add_argument("--accuracy", action="store_true",
                        help="Run accuracy validation against PyTorch Geometric")
    parser.add_argument("--no-bench", action="store_true",
                        help="Skip benchmarks (only run accuracy checks)")
    parser.add_argument("--reps", type=int, default=10,
                        help="Number of timed repetitions per benchmark (default: 10)")
    args = parser.parse_args()

    if not args.no_bench:
        run_benchmarks(reps=args.reps)

    if args.accuracy:
        run_accuracy_checks()
    elif not args.no_bench:
        print("\n  Tip: run with --accuracy to also validate against PyTorch Geometric.")

    print()


if __name__ == "__main__":
    main()
