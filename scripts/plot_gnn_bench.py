#!/usr/bin/env python3
"""
TinyGNN — GNN Benchmark Chart Generator
scripts/plot_gnn_bench.py

Reads  benchmarks/gnn_bench_results_timing.csv
       benchmarks/gnn_bench_results_accuracy.csv
Writes benchmarks/
  gnn_runtime_comparison.png — grouped bar: 1T vs NT vs PyG per layer × graph
  gnn_memory_comparison.png  — memory (MB) per layer × backend
  gnn_speedup.png            — OMP speedup (NT/1T) and TinyGNN-NT vs PyG speedup
  gnn_scaling.png            — runtime vs graph size (log-log) per layer
  gnn_accuracy.png           — accuracy table (max|diff| heatmap)

Requires: matplotlib, numpy  (no pandas needed)
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_DIR  = SCRIPT_DIR.parent
BENCH_DIR    = PROJECT_DIR / "benchmarks"
TIMING_CSV   = BENCH_DIR / "gnn_bench_results_timing.csv"
ACCURACY_CSV = BENCH_DIR / "gnn_bench_results_accuracy.csv"

# ── Colours ───────────────────────────────────────────────────────────────────
C_1T  = "#4E79A7"   # blue     — TinyGNN single-thread
C_NT  = "#F28E2B"   # orange   — TinyGNN multi-thread
C_PyG = "#59A14F"   # green    — PyTorch Geometric

LAYER_ORDER = ["GCN", "SAGE-Mean", "SAGE-Max", "GAT"]
GRAPH_ORDER = ["small-1K", "medium-10K", "large-100K"]
GRAPH_N     = {"small-1K": 1_000, "medium-10K": 10_000, "large-100K": 100_000}


# ============================================================================
#  CSV readers
# ============================================================================

def _read_timing(path: Path) -> List[Dict]:
    """Read timing CSV, normalising any TinyGNN-<N>T (N>1) → TinyGNN-NT.

    The bench script names the multi-thread backend after the host core count
    (e.g. TinyGNN-20T on a 20-core machine).  Normalise to a stable label so
    plot functions don't need to know the exact thread count.
    """
    import re
    if not path.exists():
        print(f"  [WARN] Timing CSV not found: {path}")
        return []
    _nt_re = re.compile(r'^TinyGNN-(\d+)T$')
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            m = _nt_re.match(row.get("backend", ""))
            if m and int(m.group(1)) > 1:
                row = dict(row)          # don't mutate original
                row["backend"] = "TinyGNN-NT"
            rows.append(row)
    return rows


def _read_accuracy(path: Path) -> List[Dict]:
    if not path.exists():
        print(f"  [WARN] Accuracy CSV not found: {path}")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _float(s) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


# ============================================================================
#  Plot 1 — Runtime comparison  (1T vs NT vs PyG)
# ============================================================================

def plot_runtime_comparison(rows: List[Dict], out: Path):
    # data[graph][layer][backend] = time_ms
    data: Dict[str, Dict[str, Dict[str, float]]] = {
        g: {l: {} for l in LAYER_ORDER} for g in GRAPH_ORDER
    }
    for r in rows:
        g, l, b, ms = r["graph"], r["layer"], r["backend"], _float(r["time_ms"])
        if g in data and l in data[g]:
            data[g][l][b] = ms

    fig, axes = plt.subplots(1, len(GRAPH_ORDER), figsize=(16, 5), sharey=False)
    for ax, graph in zip(axes, GRAPH_ORDER):
        layers  = LAYER_ORDER
        n_back  = 3
        x       = np.arange(len(layers))
        width   = 0.25

        for bi, (bname, color) in enumerate([
            ("TinyGNN-1T", C_1T), ("TinyGNN-NT", C_NT), ("PyG", C_PyG)
        ]):
            vals = [data[graph][l].get(bname, float("nan")) for l in layers]
            xpos  = x + (bi - 1) * width
            bars  = ax.bar(xpos, [v if v == v else 0 for v in vals],
                           width, label=bname if graph == GRAPH_ORDER[0] else "",
                           color=color, edgecolor="white", linewidth=0.6)

        ax.set_title(f"{graph}\n({GRAPH_N[graph]:,} nodes)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Median inference time (ms)" if graph == GRAPH_ORDER[0] else "")
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    handles = [
        plt.Rectangle((0,0), 1, 1, color=C_1T),
        plt.Rectangle((0,0), 1, 1, color=C_NT),
        plt.Rectangle((0,0), 1, 1, color=C_PyG),
    ]
    nt_label = f"TinyGNN-NT (multi-thread, AVX2)"
    fig.legend(handles, ["TinyGNN-1T (single-thread)", nt_label, "PyG"],
               loc="upper center", ncol=3, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("GNN Inference Runtime: TinyGNN vs PyTorch Geometric", fontsize=12, y=1.06)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK]  {out}")


# ============================================================================
#  Plot 2 — Memory comparison
# ============================================================================

def plot_memory_comparison(rows: List[Dict], out: Path):
    data: Dict[str, Dict[str, Dict[str, float]]] = {
        g: {l: {} for l in LAYER_ORDER} for g in GRAPH_ORDER
    }
    for r in rows:
        g, l, b, mb = r["graph"], r["layer"], r["backend"], _float(r["memory_mb"])
        if g in data and l in data[g]:
            data[g][l][b] = mb

    fig, axes = plt.subplots(1, len(GRAPH_ORDER), figsize=(16, 5), sharey=False)
    for ax, graph in zip(axes, GRAPH_ORDER):
        x     = np.arange(len(LAYER_ORDER))
        width = 0.25

        for bi, (bname, color) in enumerate([
            ("TinyGNN-1T", C_1T), ("TinyGNN-NT", C_NT), ("PyG", C_PyG)
        ]):
            # Convert MB → KB for readability
            vals = [data[graph][l].get(bname, float("nan")) * 1024.0
                    for l in LAYER_ORDER]
            ax.bar(x + (bi - 1) * width,
                   [v if v == v else 0 for v in vals],
                   width, color=color, edgecolor="white", linewidth=0.6)

        ax.set_title(f"{graph}\n({GRAPH_N[graph]:,} nodes)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(LAYER_ORDER, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Working-set memory (KB)" if graph == GRAPH_ORDER[0] else "")
        ax.set_yscale("log")   # log scale so different graph sizes are distinguishable
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    handles = [
        plt.Rectangle((0,0), 1, 1, color=C_1T),
        plt.Rectangle((0,0), 1, 1, color=C_NT),
        plt.Rectangle((0,0), 1, 1, color=C_PyG),
    ]
    fig.legend(handles, ["TinyGNN-1T", "TinyGNN-NT", "PyG"],
               loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("GNN Working-Set Memory: TinyGNN vs PyTorch Geometric  (analytical, log scale)",
                 fontsize=12, y=1.06)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK]  {out}")


# ============================================================================
#  Plot 3 — Speedup  (OMP speedup + TinyGNN-NT vs PyG)
# ============================================================================

def plot_speedup(rows: List[Dict], out: Path):
    # speedup_omp[graph][layer]   = ms_1T / ms_NT
    # speedup_pyg[graph][layer]   = ms_PyG / ms_NT
    speedup_omp: Dict[str, Dict[str, float]] = {g: {} for g in GRAPH_ORDER}
    speedup_pyg: Dict[str, Dict[str, float]] = {g: {} for g in GRAPH_ORDER}

    # Build lookup: time_ms[graph][layer][backend]
    times: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        g, l, b, ms = r["graph"], r["layer"], r["backend"], _float(r["time_ms"])
        times[g][l][b] = ms

    for g in GRAPH_ORDER:
        for l in LAYER_ORDER:
            t1t = times[g][l].get("TinyGNN-1T", float("nan"))
            tnt = None
            for k in times[g][l]:
                if k.startswith("TinyGNN-") and k != "TinyGNN-1T":
                    tnt = times[g][l][k]
                    break
            tpyg = times[g][l].get("PyG", float("nan"))
            speedup_omp[g][l] = t1t / tnt if (tnt and tnt > 0) else float("nan")
            speedup_pyg[g][l] = tpyg / tnt if (tpyg == tpyg and tnt and tnt > 0) else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: OMP speedup by layer
    ax = axes[0]
    x  = np.arange(len(LAYER_ORDER))
    width = 0.22
    colors_g = ["#4E79A7", "#F28E2B", "#59A14F"]
    for gi, graph in enumerate(GRAPH_ORDER):
        vals = [speedup_omp[graph].get(l, float("nan")) for l in LAYER_ORDER]
        ax.bar(x + (gi - 1) * width,
               [v if v == v else 0 for v in vals],
               width, label=graph, color=colors_g[gi], edgecolor="white")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Baseline (1T)")
    ax.set_title(f"OpenMP Speedup  (TinyGNN-1T → TinyGNN-NT,  N={len(GRAPH_ORDER)} graph sizes)")
    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Speedup   (higher = better)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Right: TinyGNN-NT vs PyG speedup
    ax = axes[1]
    for gi, graph in enumerate(GRAPH_ORDER):
        vals = [speedup_pyg[graph].get(l, float("nan")) for l in LAYER_ORDER]
        ax.bar(x + (gi - 1) * width,
               [v if v == v else 0 for v in vals],
               width, label=graph, color=colors_g[gi], edgecolor="white")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="PyG = 1×")
    ax.set_title("TinyGNN-NT  vs  PyTorch Geometric  (>1 = TinyGNN faster)")
    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Relative speed  (ms_PyG / ms_TinyGNN)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.suptitle("TinyGNN Speedup Analysis", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK]  {out}")


# ============================================================================
#  Plot 4 — Scaling (time vs num_nodes, log-log)
# ============================================================================

def plot_scaling(rows: List[Dict], out: Path):
    # times[layer][backend] = [(N, ms), ...]
    times: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        N, l, b, ms = int(r["num_nodes"]), r["layer"], r["backend"], _float(r["time_ms"])
        times[l][b].append((N, ms))

    fig, axes = plt.subplots(1, len(LAYER_ORDER), figsize=(16, 4), sharey=False)
    style = {
        "TinyGNN-1T":  dict(color=C_1T,  ls="--", marker="s", ms=6),
        "TinyGNN-NT": dict(color=C_NT,  ls="-",  marker="o", ms=6),
        "PyG":         dict(color=C_PyG, ls="-.", marker="^", ms=6),
    }

    for ax, layer in zip(axes, LAYER_ORDER):
        for b in ["TinyGNN-1T", "TinyGNN-NT", "PyG"]:
            pts = sorted(times[layer].get(b, []))
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, label=b, **style.get(b, {}))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(layer, fontsize=10)
        ax.set_xlabel("Num nodes")
        ax.set_ylabel("Time (ms)" if layer == LAYER_ORDER[0] else "")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=7)

    fig.suptitle("Inference Time Scaling  (log-log)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK]  {out}")


# ============================================================================
#  Plot 5 — Accuracy heatmap
# ============================================================================

def plot_accuracy(acc_rows: List[Dict], out: Path):
    if not acc_rows:
        print("  [SKIP] No accuracy data — skipping accuracy plot")
        return

    layers   = [r["layer"] for r in acc_rows]
    max_diff = [_float(r["max_diff"]) for r in acc_rows]

    fig, ax = plt.subplots(figsize=(7, 3))
    vals = np.array(max_diff).reshape(1, -1)
    im = ax.imshow(vals, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=max(max(max_diff)*1.1, 1e-5))

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=20, ha="right", fontsize=10)
    ax.set_yticks([])

    for j, (v, r) in enumerate(zip(max_diff, acc_rows)):
        ok = r["pass"].lower() in ("true", "1", "yes")
        txt = f"{v:.2e}\n{'PASS' if ok else 'FAIL'}"
        ax.text(j, 0, txt, ha="center", va="center",
                fontsize=9, color="black",
                fontweight="bold" if not ok else "normal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("max|TinyGNN − PyG|", fontsize=9)
    ax.set_title("Accuracy vs PyTorch Geometric  (tolerance < 2e-3)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK]  {out}")


# ============================================================================
#  Entry point
# ============================================================================

def main():
    print()
    print("=" * 60)
    print("  TinyGNN — GNN Benchmark Chart Generator")
    print("=" * 60)

    bench_rows = _read_timing(TIMING_CSV)
    acc_rows   = _read_accuracy(ACCURACY_CSV)

    if not bench_rows and not acc_rows:
        print("  ERROR: No CSV data found.  Run bench_gnn.py first.")
        sys.exit(1)

    BENCH_DIR.mkdir(exist_ok=True)

    if bench_rows:
        plot_runtime_comparison(bench_rows, BENCH_DIR / "gnn_runtime_comparison.png")
        plot_memory_comparison (bench_rows, BENCH_DIR / "gnn_memory_comparison.png")
        plot_speedup           (bench_rows, BENCH_DIR / "gnn_speedup.png")
        plot_scaling           (bench_rows, BENCH_DIR / "gnn_scaling.png")

    if acc_rows:
        plot_accuracy(acc_rows, BENCH_DIR / "gnn_accuracy.png")

    print()
    print("  All charts saved to benchmarks/")
    print()


if __name__ == "__main__":
    main()
