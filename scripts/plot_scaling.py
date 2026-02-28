#!/usr/bin/env python3
"""
TinyGNN Phase 8 — Thread-Scaling Chart Generator

Reads CSV from bench_parallel and produces a speedup chart.
Outputs PNG to benchmarks/thread_scaling.png.

Usage:
    python scripts/plot_scaling.py                          # default input
    python scripts/plot_scaling.py --csv build/bench/results.csv
"""

import argparse
import csv
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Plot thread-scaling chart")
    parser.add_argument("--csv", default="build/bench/results.csv",
                        help="Path to benchmark CSV")
    parser.add_argument("--output", default="benchmarks/thread_scaling.png",
                        help="Output PNG path")
    parser.add_argument("--config", default="cora",
                        help="Config to plot (cora, medium, large)")
    args = parser.parse_args()

    # ── Read CSV ─────────────────────────────────────────────────────────
    if not os.path.exists(args.csv):
        print(f"Error: CSV not found at {args.csv}")
        print("Run the benchmark first: build/bench/bench_parallel --csv build/bench/results.csv")
        sys.exit(1)

    rows = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["config"].strip() == args.config:
                rows.append(row)

    if not rows:
        print(f"No data for config '{args.config}' in {args.csv}")
        sys.exit(1)

    # ── Parse data ───────────────────────────────────────────────────────
    kernels = sorted(set(r["kernel"] for r in rows))
    # kernel -> {threads: speedup}
    data = {}
    for k in kernels:
        k_rows = [r for r in rows if r["kernel"] == k]
        threads = [int(r["threads"]) for r in k_rows]
        speedups = [float(r["speedup"]) for r in k_rows]
        data[k] = dict(zip(threads, speedups))

    thread_list = sorted(set(int(r["threads"]) for r in rows))

    # ── Try matplotlib ───────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        colors = {"spmm": "#2196F3", "matmul": "#4CAF50", "softmax": "#FF9800"}
        markers = {"spmm": "o", "matmul": "s", "softmax": "^"}
        labels = {"spmm": "SpMM (CSR × Dense)",
                  "matmul": "GEMM (Dense × Dense)",
                  "softmax": "Row-wise Softmax"}

        for k in kernels:
            ts = sorted(data[k].keys())
            sp = [data[k][t] for t in ts]
            ax.plot(ts, sp,
                    marker=markers.get(k, "o"),
                    color=colors.get(k, "#666"),
                    linewidth=2.2, markersize=8,
                    label=labels.get(k, k))

        # Ideal linear scaling reference
        ax.plot(thread_list, thread_list,
                linestyle="--", color="#999", linewidth=1.2,
                label="Ideal (linear)")

        ax.set_xlabel("Threads", fontsize=13)
        ax.set_ylabel("Speedup (×)", fontsize=13)
        ax.set_title(f"TinyGNN Thread-Scaling — {args.config.capitalize()} "
                     f"({rows[0]['nodes']} nodes × {rows[0]['features']} features)",
                     fontsize=14, fontweight="bold")
        ax.set_xticks(thread_list)
        ax.set_xlim(0.5, max(thread_list) + 0.5)
        ax.set_ylim(0, max(thread_list) + 1)
        ax.legend(fontsize=11, loc="upper left")
        ax.grid(True, alpha=0.3)

        # Annotate speedup values
        for k in kernels:
            ts = sorted(data[k].keys())
            for t in ts:
                sp = data[k][t]
                ax.annotate(f"{sp:.1f}×",
                           (t, sp),
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha="center", fontsize=8,
                           color=colors.get(k, "#666"))

        plt.tight_layout()
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Chart saved to: {args.output}")

    except ImportError:
        # Fallback: ASCII table
        print("\nmatplotlib not available — printing ASCII table:\n")
        print(f"Thread-Scaling Results ({args.config}):")
        print(f"{'Threads':>8}", end="")
        for k in kernels:
            print(f"  {k:>12}", end="")
        print()
        print("-" * (8 + 14 * len(kernels)))
        for t in thread_list:
            print(f"{t:>8}", end="")
            for k in kernels:
                sp = data[k].get(t, 0.0)
                print(f"  {sp:>10.2f}×", end="")
            print()


if __name__ == "__main__":
    main()
