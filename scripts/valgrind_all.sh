#!/usr/bin/env bash
# scripts/valgrind_all.sh
# ─────────────────────────────────────────────────────────────────────────────
# Runs the TinyGNN test suite under three Valgrind tools:
#   1. Memcheck  — memory leaks, invalid reads/writes, use-after-free
#   2. Helgrind  — threading errors (data races, lock-order violations)
#   3. Callgrind — performance profiling (instruction count, cache sim)
#
# Must be run from the project ROOT directory inside WSL Ubuntu-24.04:
#   cd /mnt/c/Users/anubh/OneDrive/Desktop/TinyGNN
#   bash scripts/valgrind_all.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJ_ROOT/build/valgrind"
BIN="$BUILD_DIR/test_tensor_vg"
BIN_GRAPH="$BUILD_DIR/test_graph_loader_vg"
BIN_MATMUL="$BUILD_DIR/test_matmul_vg"
BIN_SPMM="$BUILD_DIR/test_spmm_vg"
LOG_DIR="$PROJ_ROOT/build/valgrind/logs"
SRC_INCLUDE="-I $PROJ_ROOT/include"

mkdir -p "$BUILD_DIR" "$LOG_DIR"

# ── Step 1: Debug build (no -static; valgrind needs dynamic symbols) ──────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  [1/4] Building debug binaries for Valgrind"
echo "══════════════════════════════════════════════════════════"

g++ -std=c++17 -g -O0 -fno-inline -fno-omit-frame-pointer \
    $SRC_INCLUDE \
    "$PROJ_ROOT/src/tensor.cpp" \
    "$PROJ_ROOT/tests/test_tensor.cpp" \
    -o "$BIN"

g++ -std=c++17 -g -O0 -fno-inline -fno-omit-frame-pointer \
    $SRC_INCLUDE \
    "$PROJ_ROOT/src/tensor.cpp" \
    "$PROJ_ROOT/src/graph_loader.cpp" \
    "$PROJ_ROOT/tests/test_graph_loader.cpp" \
    -o "$BIN_GRAPH"

g++ -std=c++17 -g -O0 -fno-inline -fno-omit-frame-pointer \
    $SRC_INCLUDE \
    "$PROJ_ROOT/src/tensor.cpp" \
    "$PROJ_ROOT/src/ops.cpp" \
    "$PROJ_ROOT/tests/test_matmul.cpp" \
    -o "$BIN_MATMUL"

g++ -std=c++17 -g -O0 -fno-inline -fno-omit-frame-pointer \
    $SRC_INCLUDE \
    "$PROJ_ROOT/src/tensor.cpp" \
    "$PROJ_ROOT/src/ops.cpp" \
    "$PROJ_ROOT/tests/test_spmm.cpp" \
    -o "$BIN_SPMM"

echo "  Binary (tensor):       $BIN"
echo "  Binary (graph_loader): $BIN_GRAPH"
echo "  Binary (matmul):       $BIN_MATMUL"
echo "  Size (tensor):       $(du -sh "$BIN" | cut -f1)"
echo "  Size (graph_loader): $(du -sh "$BIN_GRAPH" | cut -f1)"
echo "  Size (matmul):       $(du -sh "$BIN_MATMUL" | cut -f1)"
echo "  Binary (spmm):         $BIN_SPMM"
echo "  Size (spmm):         $(du -sh "$BIN_SPMM" | cut -f1)"

# ── Step 2: Verify clean run first ────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  [2/4] Baseline test (no Valgrind)"
echo "══════════════════════════════════════════════════════════"
echo "  -- test_tensor --"
"$BIN" 2>&1 | grep -E "Total|Passed|Failed|FAIL"
echo "  -- test_graph_loader --"
"$BIN_GRAPH" 2>&1 | grep -E "Total|Passed|Failed|FAIL"
echo "  -- test_matmul --"
"$BIN_MATMUL" 2>&1 | grep -E "Total|Passed|Failed|FAIL"
echo "  -- test_spmm --"
"$BIN_SPMM" 2>&1 | grep -E "Total|Passed|Failed|FAIL"

# ── Step 3: Memcheck ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  [3/4] Valgrind: Memcheck (memory leaks + invalid access)"
echo "══════════════════════════════════════════════════════════"

MEMCHECK_LOG="$LOG_DIR/memcheck_tensor.log"

echo "  -- test_tensor --"
valgrind \
    --tool=memcheck \
    --leak-check=full \
    --show-leak-kinds=all \
    --track-origins=yes \
    --errors-for-leak-kinds=all \
    --error-exitcode=1 \
    --log-file="$MEMCHECK_LOG" \
    "$BIN" > /dev/null 2>&1 \
    && echo "  [PASS] Memcheck (tensor): no errors" \
    || { echo "  [FAIL] Memcheck (tensor) detected issues — see $MEMCHECK_LOG"; cat "$MEMCHECK_LOG" | grep -A3 "ERROR SUMMARY\|definitely lost\|Invalid"; }

echo ""
cat "$MEMCHECK_LOG" | grep -E "ERROR SUMMARY|LEAK SUMMARY|definitely lost|indirectly lost|possibly lost|still reachable|suppressed" || true

MEMCHECK_LOG_GRAPH="$LOG_DIR/memcheck_graph.log"

echo ""
echo "  -- test_graph_loader --"
valgrind \
    --tool=memcheck \
    --leak-check=full \
    --show-leak-kinds=all \
    --track-origins=yes \
    --errors-for-leak-kinds=all \
    --error-exitcode=1 \
    --log-file="$MEMCHECK_LOG_GRAPH" \
    "$BIN_GRAPH" > /dev/null 2>&1 \
    && echo "  [PASS] Memcheck (graph_loader): no errors" \
    || { echo "  [FAIL] Memcheck (graph_loader) detected issues — see $MEMCHECK_LOG_GRAPH"; cat "$MEMCHECK_LOG_GRAPH" | grep -A3 "ERROR SUMMARY\|definitely lost\|Invalid"; }

echo ""
cat "$MEMCHECK_LOG_GRAPH" | grep -E "ERROR SUMMARY|LEAK SUMMARY|definitely lost|indirectly lost|possibly lost|still reachable|suppressed" || true
MEMCHECK_LOG_MATMUL="$LOG_DIR/memcheck_matmul.log"

echo ""
echo "  -- test_matmul --"
valgrind \
    --tool=memcheck \
    --leak-check=full \
    --show-leak-kinds=all \
    --track-origins=yes \
    --errors-for-leak-kinds=all \
    --error-exitcode=1 \
    --log-file="$MEMCHECK_LOG_MATMUL" \
    "$BIN_MATMUL" > /dev/null 2>&1 \
    && echo "  [PASS] Memcheck (matmul): no errors" \
    || { echo "  [FAIL] Memcheck (matmul) detected issues — see $MEMCHECK_LOG_MATMUL"; cat "$MEMCHECK_LOG_MATMUL" | grep -A3 "ERROR SUMMARY\|definitely lost\|Invalid"; }

echo ""
cat "$MEMCHECK_LOG_MATMUL" | grep -E "ERROR SUMMARY|LEAK SUMMARY|definitely lost|indirectly lost|possibly lost|still reachable|suppressed" || true
MEMCHECK_LOG_SPMM="$LOG_DIR/memcheck_spmm.log"

echo ""
echo "  -- test_spmm --"
valgrind \
    --tool=memcheck \
    --leak-check=full \
    --show-leak-kinds=all \
    --track-origins=yes \
    --errors-for-leak-kinds=all \
    --error-exitcode=1 \
    --log-file="$MEMCHECK_LOG_SPMM" \
    "$BIN_SPMM" > /dev/null 2>&1 \
    && echo "  [PASS] Memcheck (spmm): no errors" \
    || { echo "  [FAIL] Memcheck (spmm) detected issues — see $MEMCHECK_LOG_SPMM"; cat "$MEMCHECK_LOG_SPMM" | grep -A3 "ERROR SUMMARY\|definitely lost\|Invalid"; }

echo ""
cat "$MEMCHECK_LOG_SPMM" | grep -E "ERROR SUMMARY|LEAK SUMMARY|definitely lost|indirectly lost|possibly lost|still reachable|suppressed" || true
# ── Step 4: Helgrind ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  [4/4] Valgrind: Helgrind (threading / data races)"
echo "══════════════════════════════════════════════════════════"

HELGRIND_LOG="$LOG_DIR/helgrind_tensor.log"

echo "  -- test_tensor --"
valgrind \
    --tool=helgrind \
    --error-exitcode=1 \
    --log-file="$HELGRIND_LOG" \
    "$BIN" > /dev/null 2>&1 \
    && echo "  [PASS] Helgrind (tensor): no threading errors" \
    || { echo "  [FAIL] Helgrind (tensor) detected threading issues — see $HELGRIND_LOG"; cat "$HELGRIND_LOG" | grep -A3 "ERROR SUMMARY"; }

echo ""
cat "$HELGRIND_LOG" | grep "ERROR SUMMARY" || true

HELGRIND_LOG_GRAPH="$LOG_DIR/helgrind_graph.log"

echo ""
echo "  -- test_graph_loader --"
valgrind \
    --tool=helgrind \
    --error-exitcode=1 \
    --log-file="$HELGRIND_LOG_GRAPH" \
    "$BIN_GRAPH" > /dev/null 2>&1 \
    && echo "  [PASS] Helgrind (graph_loader): no threading errors" \
    || { echo "  [FAIL] Helgrind (graph_loader) detected threading issues — see $HELGRIND_LOG_GRAPH"; cat "$HELGRIND_LOG_GRAPH" | grep -A3 "ERROR SUMMARY"; }

echo ""
cat "$HELGRIND_LOG_GRAPH" | grep "ERROR SUMMARY" || true

HELGRIND_LOG_MATMUL="$LOG_DIR/helgrind_matmul.log"

echo ""
echo "  -- test_matmul --"
valgrind \
    --tool=helgrind \
    --error-exitcode=1 \
    --log-file="$HELGRIND_LOG_MATMUL" \
    "$BIN_MATMUL" > /dev/null 2>&1 \
    && echo "  [PASS] Helgrind (matmul): no threading errors" \
    || { echo "  [FAIL] Helgrind (matmul) detected threading issues — see $HELGRIND_LOG_MATMUL"; cat "$HELGRIND_LOG_MATMUL" | grep -A3 "ERROR SUMMARY"; }

echo ""
cat "$HELGRIND_LOG_MATMUL" | grep "ERROR SUMMARY" || true

HELGRIND_LOG_SPMM="$LOG_DIR/helgrind_spmm.log"

echo ""
echo "  -- test_spmm --"
valgrind \
    --tool=helgrind \
    --error-exitcode=1 \
    --log-file="$HELGRIND_LOG_SPMM" \
    "$BIN_SPMM" > /dev/null 2>&1 \
    && echo "  [PASS] Helgrind (spmm): no threading errors" \
    || { echo "  [FAIL] Helgrind (spmm) detected threading issues — see $HELGRIND_LOG_SPMM"; cat "$HELGRIND_LOG_SPMM" | grep -A3 "ERROR SUMMARY"; }

echo ""
cat "$HELGRIND_LOG_SPMM" | grep "ERROR SUMMARY" || true

# ── Step 5: Callgrind (perf profiling) ────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  [BONUS] Valgrind: Callgrind (performance profile)"
echo "══════════════════════════════════════════════════════════"

CALLGRIND_OUT="$LOG_DIR/callgrind.out"

valgrind \
    --tool=callgrind \
    --callgrind-out-file="$CALLGRIND_OUT" \
    --cache-sim=yes \
    --branch-sim=yes \
    "$BIN" > /dev/null 2>&1

echo "  Callgrind output: $CALLGRIND_OUT"
echo "  Top functions by instruction count:"
callgrind_annotate --auto=no "$CALLGRIND_OUT" 2>/dev/null \
    | grep -E "^\s+[0-9,]" \
    | head -15 \
    || echo "  (install kcachegrind or callgrind_annotate to view)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  All Valgrind logs saved to: $LOG_DIR"
echo "  Files:"
ls -lh "$LOG_DIR/"
echo "══════════════════════════════════════════════════════════"
echo ""
