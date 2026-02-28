#!/usr/bin/env bash
# ============================================================================
#  TinyGNN — Valgrind Massif Memory Profiling  (Phase 9)
#  scripts/run_massif_phase9.sh
#
#  Compares peak heap usage of fused vs. unfused GAT/SAGE forward passes.
#
#  Usage:
#    bash scripts/run_massif_phase9.sh
# ============================================================================
set -euo pipefail

SRC="src/tensor.cpp src/graph_loader.cpp src/ops.cpp src/layers.cpp src/model.cpp"
CXX="g++ -std=c++17 -O2 -g -Iinclude"
OUTDIR="build/massif"
mkdir -p "$OUTDIR"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  TinyGNN — Massif Memory Profiling  (Phase 9)           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo

# ── Build ────────────────────────────────────────────────────────────────────
echo "Building profiling binaries (no OpenMP for Massif compatibility)..."
$CXX -o "$OUTDIR/massif_gat_unfused" benchmarks/massif_gat_unfused.cpp $SRC
$CXX -o "$OUTDIR/massif_gat_fused"   benchmarks/massif_gat_fused.cpp   $SRC
echo "  ✓ Build complete"
echo

# ── Profile GAT Unfused ─────────────────────────────────────────────────────
echo "Running Massif: GAT Unfused..."
valgrind --tool=massif --pages-as-heap=no --stacks=no \
    --massif-out-file="$OUTDIR/massif_unfused.out" \
    "$OUTDIR/massif_gat_unfused" 2>&1 | grep -v "^=="

PEAK_UNFUSED=$(grep 'mem_heap_B' "$OUTDIR/massif_unfused.out" \
    | sort -t= -k2 -n | tail -1 | cut -d= -f2)
echo "  Peak heap (unfused): $PEAK_UNFUSED bytes ($(echo "scale=2; $PEAK_UNFUSED / 1048576" | bc) MB)"
echo

# ── Profile GAT Fused ───────────────────────────────────────────────────────
echo "Running Massif: GAT Fused..."
valgrind --tool=massif --pages-as-heap=no --stacks=no \
    --massif-out-file="$OUTDIR/massif_fused.out" \
    "$OUTDIR/massif_gat_fused" 2>&1 | grep -v "^=="

PEAK_FUSED=$(grep 'mem_heap_B' "$OUTDIR/massif_fused.out" \
    | sort -t= -k2 -n | tail -1 | cut -d= -f2)
echo "  Peak heap (fused):   $PEAK_FUSED bytes ($(echo "scale=2; $PEAK_FUSED / 1048576" | bc) MB)"
echo

# ── Summary ──────────────────────────────────────────────────────────────────
SAVED=$(( PEAK_UNFUSED - PEAK_FUSED ))
RATIO=$(echo "scale=2; $PEAK_UNFUSED / $PEAK_FUSED" | bc)

echo "──── Summary ────────────────────────────────────────────"
echo "  Unfused peak heap: $PEAK_UNFUSED bytes"
echo "  Fused peak heap:   $PEAK_FUSED bytes"
echo "  Saved:             $SAVED bytes (${RATIO}× reduction)"
echo

# ── Generate ms_print reports ────────────────────────────────────────────────
ms_print "$OUTDIR/massif_unfused.out" > "$OUTDIR/massif_unfused.txt"
ms_print "$OUTDIR/massif_fused.out"   > "$OUTDIR/massif_fused.txt"
echo "  Reports: $OUTDIR/massif_unfused.txt, $OUTDIR/massif_fused.txt"
echo "  ✓ Profiling complete."
