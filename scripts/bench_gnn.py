#!/usr/bin/env python3
"""
TinyGNN — Comprehensive GNN Benchmark & Accuracy Validation
scripts/bench_gnn.py

Covers:
  • TinyGNN (single-thread / unoptimized)  vs
    TinyGNN (multi-thread / optimized)     vs
    PyTorch Geometric (reference)
  • Metrics: wall time (ms, median), working-set memory (MB, analytical estimate)
  • Layers: GCNLayer, SAGELayer-Mean, SAGELayer-Max, GATLayer
  • Graph sizes: small (1K nodes), medium (10K nodes), large (100K nodes)
  • Accuracy: output logit agreement between TinyGNN and PyG (max|diff|)

Output:
  benchmarks/gnn_bench_results_timing.csv   — timing + memory rows
  benchmarks/gnn_bench_results_accuracy.csv — accuracy rows

Usage:
    python scripts/bench_gnn.py              # all benchmarks + accuracy
    python scripts/bench_gnn.py --reps 20    # more reps for stable timing
    python scripts/bench_gnn.py --no-bench   # accuracy only
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import gc
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

try:
    import psutil as _psutil
    _PROC = _psutil.Process()
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "python"))

import _tinygnn_core as tg

# ── OpenMP thread control ─────────────────────────────────────────────────────
def _make_omp_setter():
    """Return callable (n: int)->None that changes OMP thread count at runtime."""
    # First try: use tg.set_num_threads (works with statically-linked libgomp)
    if hasattr(tg, 'set_num_threads'):
        return tg.set_num_threads
    for lib_name in ["libgomp-1.dll", "libgomp.so.1", "libgomp.so",
                     "libiomp5.dll", "libiomp5.so", "libomp.dylib", "libomp.so"]:
        try:
            lib = ctypes.CDLL(lib_name)
            fn  = lib.omp_set_num_threads
            fn.argtypes = [ctypes.c_int]
            fn.restype  = None
            return fn
        except Exception:
            pass
    # env-var fallback (works before first parallel region)
    def _fallback(n: int):
        os.environ["OMP_NUM_THREADS"] = str(n)
    return _fallback

_omp_set_threads = _make_omp_setter()
N_CORES = os.cpu_count() or 4


# ============================================================================
#  Graph helpers
# ============================================================================

def make_random_graph_csr(num_nodes: int, avg_degree: int, seed: int = 42):
    """Random undirected symmetric graph as scipy CSR (float32 ones)."""
    from scipy.sparse import csr_matrix  # type: ignore
    rng = np.random.default_rng(seed)
    rows_l, cols_l = [], []
    for src in range(num_nodes):
        k = min(avg_degree, num_nodes - 1)
        candidates = np.delete(np.arange(num_nodes), src)
        targets = rng.choice(candidates, size=k, replace=False)
        for dst in targets:
            rows_l.append(src); cols_l.append(dst)
            rows_l.append(dst); cols_l.append(src)
    r = np.array(rows_l, dtype=np.int32)
    c = np.array(cols_l, dtype=np.int32)
    v = np.ones(len(r), dtype=np.float32)
    mat = csr_matrix((v, (r, c)), shape=(num_nodes, num_nodes))
    mat.data[:] = 1.0
    mat.sum_duplicates()
    mat.sort_indices()
    return mat


def scipy_to_tinygnn(mat):
    """Convert a scipy CSR matrix to a TinyGNN SparseCSR Tensor."""
    return tg.Tensor.from_scipy_csr(mat)


# ============================================================================
#  Weight factories  (same weights shared between TinyGNN and PyG)
# ============================================================================

def _rand(shape, scale, rng):
    return (rng.standard_normal(shape) * scale).astype(np.float32)

def _gcn_weights(F_in, F_out, seed=0):
    rng = np.random.default_rng(seed)
    return {"W": _rand((F_in, F_out), (2/F_in)**0.5, rng),
            "b": np.zeros((1, F_out), dtype=np.float32)}

def _sage_weights(F_in, F_out, seed=0):
    rng = np.random.default_rng(seed)
    s = (2/F_in)**0.5
    return {"Wn": _rand((F_in, F_out), s, rng),
            "Ws": _rand((F_in, F_out), s, rng),
            "b":  np.zeros((1, F_out), dtype=np.float32)}

def _gat_weights(F_in, F_out, seed=0):
    rng = np.random.default_rng(seed)
    return {"W":  _rand((F_in, F_out), (2/F_in)**0.5, rng),
            "al": _rand((1, F_out), 0.1, rng),
            "ar": _rand((1, F_out), 0.1, rng),
            "b":  np.zeros((1, F_out), dtype=np.float32)}


# ── TinyGNN layer builders ────────────────────────────────────────────────────

def make_tg_gcn(w, F_in, F_out):
    ly = tg.GCNLayer(F_in, F_out, True, tg.Activation.NONE)
    ly.set_weight(tg.Tensor.from_numpy(w["W"]))
    ly.set_bias(tg.Tensor.from_numpy(w["b"]))
    return ly

def make_tg_sage(w, F_in, F_out, agg):
    ly = tg.SAGELayer(F_in, F_out, agg, True, tg.Activation.NONE)
    ly.set_weight_neigh(tg.Tensor.from_numpy(w["Wn"]))
    ly.set_weight_self(tg.Tensor.from_numpy(w["Ws"]))
    ly.set_bias(tg.Tensor.from_numpy(w["b"]))
    return ly

def make_tg_gat(w, F_in, F_out):
    ly = tg.GATLayer(F_in, F_out, 0.2, True, tg.Activation.NONE)
    ly.set_weight(tg.Tensor.from_numpy(w["W"]))
    ly.set_attn_left(tg.Tensor.from_numpy(w["al"]))
    ly.set_attn_right(tg.Tensor.from_numpy(w["ar"]))
    ly.set_bias(tg.Tensor.from_numpy(w["b"]))
    return ly


# ============================================================================
#  Timing / memory helper
# ============================================================================

def _timeit(fn, warmup: int, reps: int) -> Tuple[float, float]:
    """Return (median_ms, dummy_mb).  OMP thread count must already be set.

    Wall-time is measured reliably; memory is now computed analytically per
    graph/layer configuration in run_benchmarks() so callers should ignore
    the returned mb value (kept for API compat).
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1_000.0)
    return float(np.median(times)), 0.0


def _working_set_mb(lname: str, N: int, E: int, F_in: int, F_out: int) -> float:
    """Analytical working-set estimate (MB) for one TinyGNN forward pass.

    Counts *peak* live float32 tensors needed simultaneously:
      • Input feature matrix  H:          N × F_in
      • Output feature matrix H_out:      N × F_out
      • Weight matrix W (or Wn+Ws):       F_in × F_out  (×2 for SAGE)
      • Bias b:                           F_out
      • CSR adjacency (stored once):      E values + (N+1) row-ptrs + E col-idx
      • Temporary intermediate buffer:    N × F_out    (e.g. spmm result)
    GAT additionally stores attention coefficients: E logits + E softmaxed.
    All values are float32 (4 bytes).
    """
    BYTES_PER_FLOAT = 4
    n_weights = F_in * F_out  # W
    if lname == "SAGE-Mean" or lname == "SAGE-Max":
        n_weights *= 2        # Wn + Ws
    n_csr     = E + (N + 1) + E               # val + rowptr + colidx (int32 ≈ float)
    n_att     = 2 * E if lname == "GAT" else 0
    n_total   = (N * F_in          # H input
                 + N * F_out       # H output
                 + n_weights       # weight(s)
                 + F_out           # bias
                 + n_csr           # sparse adjacency
                 + N * F_out       # intermediate buffer
                 + n_att)         # attention coefficients
    return n_total * BYTES_PER_FLOAT / (1024 ** 2)


def _pyg_working_set_mb(lname: str, N: int, E: int, F_in: int, F_out: int) -> float:
    """Analytical working-set estimate for PyG (same formula + torch overhead)."""
    BYTES_PER_FLOAT = 4
    n_weights = F_in * F_out
    if lname in ("SAGE-Mean", "SAGE-Max"):
        n_weights *= 2
    n_coo  = 2 * E   # edge_index (int64) ≈ 2× float in size
    n_att  = 2 * E if lname == "GAT" else 0
    # PyG adds normalised adjacency copy + gradient buffers (even under no_grad)
    n_total = (N * F_in + N * F_out + n_weights + F_out
               + n_coo + N * F_out + n_att
               + N * F_out)     # PyG keeps an extra intermediate
    # Overhead factor for PyTorch tensor meta + graph structure
    overhead = 1.6
    return n_total * BYTES_PER_FLOAT * overhead / (1024 ** 2)


# ============================================================================
#  PyG helpers
# ============================================================================

def _try_pyg():
    try:
        import torch
        from torch_geometric.nn import GCNConv, SAGEConv, GATConv
        return torch, GCNConv, SAGEConv, GATConv
    except ImportError:
        return None

def _pyg_edge_index(scipy_mat):
    import torch
    coo = scipy_mat.tocoo()
    r = torch.from_numpy(coo.row.astype(np.int64))
    c = torch.from_numpy(coo.col.astype(np.int64))
    return torch.stack([r, c], dim=0)

def _pyg_gcn_fwd(w, F_in, F_out, edge_index, H_torch, pyg_cls):
    torch, GCNConv, _, _ = pyg_cls
    conv = GCNConv(F_in, F_out, bias=True, normalize=True, add_self_loops=True)
    with torch.no_grad():
        conv.lin.weight.copy_(torch.from_numpy(w["W"].T))
        conv.bias.copy_(torch.from_numpy(w["b"].flatten()))  # includes self-loops
    def _f():
        with torch.no_grad(): conv(H_torch, edge_index)
    return _f

def _pyg_sage_fwd(w, F_in, F_out, edge_index, H_torch, pyg_cls):
    torch, _, SAGEConv, _ = pyg_cls
    conv = SAGEConv(F_in, F_out, bias=True, aggr="mean", normalize=False, root_weight=True)
    with torch.no_grad():
        # PyG SAGEConv: lin_l(agg_neigh) + lin_r(self)
        # TinyGNN:      mean_agg @ Wn   + H @ Ws + b
        conv.lin_l.weight.copy_(torch.from_numpy(w["Wn"].T))  # lin_l → neighbor agg
        conv.lin_r.weight.copy_(torch.from_numpy(w["Ws"].T))  # lin_r → self
        conv.lin_l.bias.copy_(torch.from_numpy(w["b"].flatten()))
    def _f():
        with torch.no_grad(): conv(H_torch, edge_index)
    return _f

def _pyg_gat_fwd(w, F_in, F_out, edge_index, H_torch, pyg_cls):
    torch, _, _, GATConv = pyg_cls
    conv = GATConv(F_in, F_out, heads=1, bias=True, negative_slope=0.2,
                   add_self_loops=True, concat=True)
    with torch.no_grad():
        conv.lin.weight.copy_(torch.from_numpy(w["W"].T))
        # PyG: att_src applied to source (neighbor j) = ar in TinyGNN
        #      att_dst applied to destination (center i) = al in TinyGNN
        conv.att_src.copy_(torch.from_numpy(w["ar"]).reshape(1, 1, F_out))
        conv.att_dst.copy_(torch.from_numpy(w["al"]).reshape(1, 1, F_out))
        conv.bias.copy_(torch.from_numpy(w["b"].flatten()))
    def _f():
        with torch.no_grad(): conv(H_torch, edge_index)
    return _f


# ============================================================================
#  Accuracy validation
# ============================================================================

def run_accuracy(pyg_cls) -> List[Dict]:
    if pyg_cls is None:
        print("  [SKIP] torch_geometric not available — accuracy checks skipped.")
        return []
    torch, GCNConv, SAGEConv, GATConv = pyg_cls
    N, F_in, F_out, seed = 30, 8, 4, 77
    sm   = make_random_graph_csr(N, avg_degree=5, seed=seed)
    A_tg = scipy_to_tinygnn(sm)
    A_norm = tg.gcn_norm(A_tg)         # gcn_norm adds self-loops internally
    A_sl   = tg.add_self_loops(A_tg)
    H_np   = np.random.default_rng(seed).standard_normal((N, F_in)).astype(np.float32)
    H_tg   = tg.Tensor.from_numpy(H_np)
    H_t    = torch.from_numpy(H_np)
    ei     = _pyg_edge_index(sm)

    results: List[Dict] = []
    print()
    print("  Accuracy (TinyGNN vs PyTorch Geometric)  —  N=30, F_in=8, F_out=4")
    print("  " + "-" * 64)

    def _chk(name, tg_out, pyg_out):
        d = np.abs(tg_out - pyg_out)
        md, me, re = d.max(), d.mean(), d.mean()/(np.abs(pyg_out).mean()+1e-8)
        ok = bool(md < 2e-3)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<18}  max|diff|={md:.2e}  "
              f"mean|diff|={me:.2e}  rel={re:.2e}")
        results.append({"layer": name, "max_diff": float(md),
                        "mean_diff": float(me), "rel_diff": float(re), "pass": ok})

    # GCN
    w = _gcn_weights(F_in, F_out, seed)
    ly = make_tg_gcn(w, F_in, F_out)
    out_tg = ly.forward(A_norm, H_tg).to_numpy()
    conv = GCNConv(F_in, F_out, bias=True, normalize=True, add_self_loops=True)
    with torch.no_grad():
        conv.lin.weight.copy_(torch.from_numpy(w["W"].T))
        conv.bias.copy_(torch.from_numpy(w["b"].flatten()))
    with torch.no_grad(): out_pyg = conv(H_t, ei).numpy()
    _chk("GCNLayer", out_tg, out_pyg)  # gcn_norm already adds self-loops

    # SAGE-Mean
    w = _sage_weights(F_in, F_out, seed)
    ly = make_tg_sage(w, F_in, F_out, tg.SAGELayer.Aggregator.Mean)
    out_tg = ly.forward(A_tg, H_tg).to_numpy()
    conv = SAGEConv(F_in, F_out, bias=True, aggr="mean", normalize=False, root_weight=True)
    with torch.no_grad():
        # PyG SAGEConv: lin_l(agg_neigh) + lin_r(self)
        conv.lin_l.weight.copy_(torch.from_numpy(w["Wn"].T))  # lin_l → neighbor agg
        conv.lin_r.weight.copy_(torch.from_numpy(w["Ws"].T))  # lin_r → self
        conv.lin_l.bias.copy_(torch.from_numpy(w["b"].flatten()))
    with torch.no_grad(): out_pyg = conv(H_t, ei).numpy()
    _chk("SAGELayer-Mean", out_tg, out_pyg)

    # GAT
    w = _gat_weights(F_in, F_out, seed)
    ly = make_tg_gat(w, F_in, F_out)
    out_tg = ly.forward(A_sl, H_tg).to_numpy()
    conv = GATConv(F_in, F_out, heads=1, bias=True, negative_slope=0.2,
                   add_self_loops=True, concat=True)
    with torch.no_grad():
        conv.lin.weight.copy_(torch.from_numpy(w["W"].T))
        # att_src = source/neighbor (ar), att_dst = destination/center (al)
        conv.att_src.copy_(torch.from_numpy(w["ar"]).reshape(1,1,F_out))
        conv.att_dst.copy_(torch.from_numpy(w["al"]).reshape(1,1,F_out))
        conv.bias.copy_(torch.from_numpy(w["b"].flatten()))
    with torch.no_grad(): out_pyg = conv(H_t, ei).numpy()
    _chk("GATLayer", out_tg, out_pyg)

    ok_all = all(r["pass"] for r in results)
    print(f"\n  Overall: {'ALL PASS' if ok_all else 'SOME FAIL'}")
    return results


# ============================================================================
#  Main benchmark loop
# ============================================================================

GRAPH_CONFIGS = [
    # (label,      N,       avg_deg, F_in,  F_out)
    ("small-1K",   1_000,   5,       64,    32),
    ("medium-10K", 10_000, 10,      128,    64),
    ("large-100K", 100_000, 5,      256,   128),
]


def run_benchmarks(reps: int, pyg_cls) -> List[Dict]:
    rows: List[Dict] = []
    print()
    print("=" * 84)
    print(f"  TinyGNN  GNN Benchmarks   reps={reps}  cores={N_CORES}")
    print(f"  Backends: TinyGNN-1T (unoptimized) | TinyGNN-{N_CORES}T (optimized) | PyG")
    print("=" * 84)
    hdr = (f"{'Graph':<13} {'Layer':<13} {'Backend':<22}"
           f" {'ms':>8} {'MB':>7} {'vs 1T':>7} {'vs PyG':>8}")
    print(hdr)
    print("-" * len(hdr))

    for label, N, avg_deg, F_in, F_out in GRAPH_CONFIGS:
        sm    = make_random_graph_csr(N, avg_deg, seed=7)
        A_tg  = scipy_to_tinygnn(sm)
        A_norm= tg.gcn_norm(A_tg)  # gcn_norm adds self-loops internally
        A_sl  = tg.add_self_loops(A_tg)
        H_np  = np.random.default_rng(3).standard_normal((N, F_in)).astype(np.float32)
        H_tg  = tg.Tensor.from_numpy(H_np)
        warmup = 3 if N <= 10_000 else 1

        if pyg_cls is not None:
            torch = pyg_cls[0]
            H_t = torch.from_numpy(H_np)
            ei  = _pyg_edge_index(sm)
        else:
            H_t = ei = None

        specs = [
            ("GCN",       _gcn_weights,
             lambda w, fi, fo: make_tg_gcn(w, fi, fo),  A_norm,
             lambda w, fi, fo: _pyg_gcn_fwd(w, fi, fo, ei, H_t, pyg_cls)),
            ("SAGE-Mean", _sage_weights,
             lambda w, fi, fo: make_tg_sage(w, fi, fo, tg.SAGELayer.Aggregator.Mean), A_tg,
             lambda w, fi, fo: _pyg_sage_fwd(w, fi, fo, ei, H_t, pyg_cls)),
            ("SAGE-Max",  _sage_weights,
             lambda w, fi, fo: make_tg_sage(w, fi, fo, tg.SAGELayer.Aggregator.Max),  A_tg,
             None),
            ("GAT",       _gat_weights,
             lambda w, fi, fo: make_tg_gat(w, fi, fo),  A_sl,
             lambda w, fi, fo: _pyg_gat_fwd(w, fi, fo, ei, H_t, pyg_cls)),
        ]

        for (lname, wfn, tg_builder, A_fwd, pyg_builder) in specs:
            w  = wfn(F_in, F_out, seed=0)
            ly = tg_builder(w, F_in, F_out)
            def _tg(ly=ly, A=A_fwd, H=H_tg):
                return ly.forward(A, H)

            # Analytical working-set memory (differentiated per graph/layer)
            E = int(sm.nnz)               # number of edges (CSR non-zeros)
            mb_tg  = _working_set_mb(lname, N, E, F_in, F_out)
            mb_pyg_est = _pyg_working_set_mb(lname, N, E, F_in, F_out)

            # TinyGNN 1-thread (unoptimized)
            _omp_set_threads(1)
            ms_1t, _ = _timeit(_tg, warmup=warmup, reps=reps)

            # TinyGNN N-thread (optimized)
            _omp_set_threads(N_CORES)
            ms_nt, _ = _timeit(_tg, warmup=warmup, reps=reps)

            sp_omp = ms_1t / ms_nt if ms_nt > 0 else float("nan")

            # PyG
            ms_pyg = sp_pyg = float("nan")
            if pyg_cls is not None and pyg_builder is not None:
                pyg_fn = pyg_builder(w, F_in, F_out)
                ms_pyg, _ = _timeit(pyg_fn, warmup=warmup, reps=reps)
                sp_pyg = ms_pyg / ms_nt if ms_nt > 0 else float("nan")

            def _add(backend, ms, mb, vs1t, vspyg, thr):
                rows.append({
                    "graph": label, "num_nodes": N, "avg_deg": avg_deg,
                    "F_in": F_in, "F_out": F_out, "layer": lname,
                    "backend": backend, "threads": thr,
                    "time_ms": round(ms, 3), "memory_mb": round(mb, 4),
                    "speedup_vs_1t":  round(vs1t,  3) if vs1t  == vs1t  else "",
                    "speedup_vs_pyg": round(vspyg, 3) if vspyg == vspyg else "",
                })

            _add("TinyGNN-1T",          ms_1t,  mb_tg,      1.0,    sp_pyg, 1)
            _add(f"TinyGNN-{N_CORES}T", ms_nt,  mb_tg,      sp_omp, sp_pyg, N_CORES)
            if ms_pyg == ms_pyg:
                _add("PyG", ms_pyg, mb_pyg_est,
                     ms_1t/ms_pyg if ms_pyg > 0 else float("nan"), 1.0, 0)

            pyg_s   = f"{ms_pyg:7.2f}" if ms_pyg == ms_pyg else "    n/a"
            vspyg_s = f"{sp_pyg:6.2f}x" if sp_pyg == sp_pyg else "    n/a"
            print(f"{label:<13} {lname:<13} {'TinyGNN-1T':<22} {ms_1t:8.2f} {mb_tg:7.3f}   1.00x")
            print(f"{'':13} {'':13} {f'TinyGNN-{N_CORES}T':<22} {ms_nt:8.2f} {mb_tg:7.3f} {sp_omp:6.2f}x")
            if ms_pyg == ms_pyg:
                print(f"{'':13} {'':13} {'PyG':<22} {ms_pyg:8.2f} {mb_pyg_est:7.3f}         {vspyg_s}")
            print()

    print("=" * 84)
    return rows


# ============================================================================
#  CSV output
# ============================================================================

def save_csv(bench_rows: List[Dict], acc_rows: List[Dict], prefix: str):
    os.makedirs(os.path.dirname(prefix) if os.path.dirname(prefix) else ".", exist_ok=True)
    base = prefix.replace(".csv", "")

    if bench_rows:
        path = base + "_timing.csv"
        fields = ["graph","num_nodes","avg_deg","F_in","F_out","layer",
                  "backend","threads","time_ms","memory_mb",
                  "speedup_vs_1t","speedup_vs_pyg"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(bench_rows)
        print(f"  Timing CSV   -> {path}")

    if acc_rows:
        path = base + "_accuracy.csv"
        fields = ["layer","max_diff","mean_diff","rel_diff","pass"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(acc_rows)
        print(f"  Accuracy CSV -> {path}")


# ============================================================================
#  Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TinyGNN comprehensive GNN benchmark")
    parser.add_argument("--reps",        type=int, default=8,
                        help="Timed repetitions per cell (default 8)")
    parser.add_argument("--no-bench",    action="store_true", help="Skip benchmarks")
    parser.add_argument("--no-accuracy", action="store_true", help="Skip accuracy checks")
    parser.add_argument("--csv",         default="benchmarks/gnn_bench_results.csv",
                        help="CSV output prefix")
    args = parser.parse_args()

    pyg = _try_pyg()
    if pyg is None:
        print("  NOTE: torch_geometric not found — PyG comparisons excluded.")

    bench_rows: List[Dict] = []
    acc_rows:   List[Dict] = []

    if not args.no_accuracy:
        acc_rows = run_accuracy(pyg)

    if not args.no_bench:
        bench_rows = run_benchmarks(reps=args.reps, pyg_cls=pyg)

    if bench_rows or acc_rows:
        save_csv(bench_rows, acc_rows, args.csv)
        print()
        print("  Run  python scripts/plot_gnn_bench.py  to generate charts.")
    print()


if __name__ == "__main__":
    main()
