#!/usr/bin/env python3
"""
TinyGNN Phase 11 — Dataset Benchmark Chart Generator

Reads CSV from bench_datasets and produces two publication-quality PNG charts:
  1. benchmarks/fusion_speedup.png   — Grouped bar chart: fused vs unfused runtime
  2. benchmarks/dataset_overview.png — All-layers comparison across datasets

Usage:
    python scripts/plot_dataset_bench.py                               # defaults
    python scripts/plot_dataset_bench.py --csv build/bench/dataset_results.csv
"""

import argparse
import csv
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Plot dataset benchmark charts")
    parser.add_argument("--csv", default="build/bench/dataset_results.csv",
                        help="Path to benchmark CSV")
    parser.add_argument("--output-dir", default="benchmarks",
                        help="Directory for output PNGs")
    args = parser.parse_args()

    # ── Read CSV ─────────────────────────────────────────────────────────
    if not os.path.exists(args.csv):
        print(f"Error: CSV not found at {args.csv}")
        print("Run the benchmark first:")
        print("  ./build/bench/bench_datasets --csv build/bench/dataset_results.csv")
        sys.exit(1)

    rows = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["time_us"] = float(row["time_us"])
            row["speedup"] = float(row["speedup"])
            row["num_nodes"] = int(row["num_nodes"])
            row["num_edges"] = int(row["num_edges"])
            row["num_features"] = int(row["num_features"])
            row["F_out"] = int(row["F_out"])
            rows.append(row)

    if not rows:
        print("No data rows found in CSV.")
        sys.exit(1)

    datasets = sorted(set(r["dataset"] for r in rows))
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available — printing ASCII summary:\n")
        _ascii_summary(rows, datasets)
        return

    # ==================================================================
    #  Chart 1: Fusion Speedup — fused vs unfused grouped bars
    # ==================================================================
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5),
                             sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    colors = {
        "unfused": "#EF5350",   # red
        "fused":   "#42A5F5",   # blue
    }

    for ax, ds_name in zip(axes, datasets):
        ds_rows = [r for r in rows if r["dataset"] == ds_name
                   and r["layer"] in ("GAT", "SAGE") and r["variant"] in ("fused", "unfused")]

        layers = sorted(set(r["layer"] for r in ds_rows))
        n_layers = len(layers)
        bar_width = 0.35
        x_pos = list(range(n_layers))

        for i, layer in enumerate(layers):
            unfused_row = [r for r in ds_rows if r["layer"] == layer and r["variant"] == "unfused"]
            fused_row = [r for r in ds_rows if r["layer"] == layer and r["variant"] == "fused"]

            unfused_us = unfused_row[0]["time_us"] if unfused_row else 0
            fused_us = fused_row[0]["time_us"] if fused_row else 0

            # Convert to ms for readability
            unfused_ms = unfused_us / 1000.0
            fused_ms = fused_us / 1000.0

            b1 = ax.bar(i - bar_width / 2, unfused_ms, bar_width,
                        color=colors["unfused"], edgecolor="white", linewidth=0.5)
            b2 = ax.bar(i + bar_width / 2, fused_ms, bar_width,
                        color=colors["fused"], edgecolor="white", linewidth=0.5)

            # Annotate with runtime
            ax.annotate(f"{unfused_ms:.1f}ms",
                        xy=(i - bar_width / 2, unfused_ms),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=9, color=colors["unfused"],
                        fontweight="bold")
            ax.annotate(f"{fused_ms:.1f}ms",
                        xy=(i + bar_width / 2, fused_ms),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=9, color=colors["fused"],
                        fontweight="bold")

            # Speedup annotation
            if fused_row:
                speedup = fused_row[0]["speedup"]
                ax.annotate(f"{speedup:.2f}×",
                            xy=(i, max(unfused_ms, fused_ms)),
                            xytext=(0, 18), textcoords="offset points",
                            ha="center", fontsize=11, fontweight="bold",
                            color="#2E7D32",
                            arrowprops=dict(arrowstyle="-", color="#999",
                                            lw=0.8))

        # Get dataset info
        ds_info = [r for r in rows if r["dataset"] == ds_name]
        n_nodes = ds_info[0]["num_nodes"]
        n_edges = ds_info[0]["num_edges"]

        ax.set_xticks(x_pos)
        ax.set_xticklabels(layers, fontsize=12, fontweight="bold")
        ax.set_ylabel("Forward Pass Time (ms)", fontsize=11)
        ax.set_title(f"{ds_name.capitalize()}\n({n_nodes:,} nodes, {n_edges:,} edges)",
                     fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    # Legend
    legend_patches = [
        mpatches.Patch(color=colors["unfused"], label="Unfused (Phase 6)"),
        mpatches.Patch(color=colors["fused"], label="Fused (Phase 10)"),
    ]
    fig.legend(handles=legend_patches, loc="upper center",
               ncol=2, fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("TinyGNN — Operator Fusion: Fused vs Unfused on Real Datasets",
                 fontsize=15, fontweight="bold", y=1.08)
    plt.tight_layout()

    out1 = os.path.join(args.output_dir, "fusion_speedup.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {out1}")
    plt.close()

    # ==================================================================
    #  Chart 2: All-layers overview (GCN + fused GAT + fused SAGE)
    # ==================================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 5))

    layer_colors = {
        "GCN":  "#4CAF50",
        "GAT":  "#2196F3",
        "SAGE": "#FF9800",
    }

    n_datasets = len(datasets)
    n_layers = 3  # GCN, GAT, SAGE
    bar_width = 0.22
    x_base = list(range(n_datasets))

    for j, layer in enumerate(["GCN", "GAT", "SAGE"]):
        times_ms = []
        for ds_name in datasets:
            # Use fused variant for GAT/SAGE, normal for GCN
            variant = "fused"
            match = [r for r in rows if r["dataset"] == ds_name
                     and r["layer"] == layer and r["variant"] == variant]
            if match:
                times_ms.append(match[0]["time_us"] / 1000.0)
            else:
                times_ms.append(0)

        positions = [x + (j - 1) * bar_width for x in x_base]
        bars = ax2.bar(positions, times_ms, bar_width,
                       color=layer_colors[layer], edgecolor="white",
                       linewidth=0.5, label=layer)

        for pos, val in zip(positions, times_ms):
            if val > 0:
                ax2.annotate(f"{val:.1f}ms",
                             xy=(pos, val),
                             xytext=(0, 5), textcoords="offset points",
                             ha="center", fontsize=9,
                             color=layer_colors[layer], fontweight="bold")

    # X-axis labels with dataset info
    xlabels = []
    for ds_name in datasets:
        ds_info = [r for r in rows if r["dataset"] == ds_name]
        if ds_info:
            xlabels.append(f"{ds_name.capitalize()}\n({ds_info[0]['num_nodes']:,} nodes)")
        else:
            xlabels.append(ds_name.capitalize())

    ax2.set_xticks(x_base)
    ax2.set_xticklabels(xlabels, fontsize=12)
    ax2.set_ylabel("Forward Pass Time (ms)", fontsize=12)
    ax2.set_title("TinyGNN — Layer Performance on Real Datasets (Fused Kernels)",
                  fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11, loc="upper left")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    out2 = os.path.join(args.output_dir, "dataset_overview.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {out2}")
    plt.close()


def _ascii_summary(rows, datasets):
    """Fallback ASCII table when matplotlib is unavailable."""
    print(f"{'Dataset':<10} {'Layer':<6} {'Variant':<10} {'Time (ms)':>12} {'Speedup':>10}")
    print("-" * 52)
    for ds in datasets:
        for r in sorted([r for r in rows if r["dataset"] == ds],
                        key=lambda x: (x["layer"], x["variant"])):
            ms = r["time_us"] / 1000.0
            print(f"{r['dataset']:<10} {r['layer']:<6} {r['variant']:<10} "
                  f"{ms:>10.2f}ms {r['speedup']:>8.2f}×")


if __name__ == "__main__":
    main()
