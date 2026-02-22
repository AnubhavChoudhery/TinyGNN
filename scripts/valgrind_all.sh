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
LOG_DIR="$PROJ_ROOT/build/valgrind/logs"
SRC_INCLUDE="-I $PROJ_ROOT/include"

mkdir -p "$BUILD_DIR" "$LOG_DIR"

# ── Step 1: Debug build (no -static; valgrind needs dynamic symbols) ──────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  [1/4] Building debug binary for Valgrind"
echo "══════════════════════════════════════════════════════════"

g++ -std=c++17 -g -O0 -fno-inline -fno-omit-frame-pointer \
    $SRC_INCLUDE \
    "$PROJ_ROOT/src/tensor.cpp" \
    "$PROJ_ROOT/tests/test_tensor.cpp" \
    -o "$BIN"

echo "  Binary: $BIN"
echo "  Size  : $(du -sh "$BIN" | cut -f1)"

# ── Step 2: Verify clean run first ────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  [2/4] Baseline test (no Valgrind)"
echo "══════════════════════════════════════════════════════════"
"$BIN" 2>&1 | grep -E "Total|Passed|Failed|FAIL"

# ── Step 3: Memcheck ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  [3/4] Valgrind: Memcheck (memory leaks + invalid access)"
echo "══════════════════════════════════════════════════════════"

MEMCHECK_LOG="$LOG_DIR/memcheck.log"

valgrind \
    --tool=memcheck \
    --leak-check=full \
    --show-leak-kinds=all \
    --track-origins=yes \
    --errors-for-leak-kinds=all \
    --error-exitcode=1 \
    --log-file="$MEMCHECK_LOG" \
    "$BIN" > /dev/null 2>&1 \
    && echo "  [PASS] Memcheck: no errors" \
    || { echo "  [FAIL] Memcheck detected issues — see $MEMCHECK_LOG"; cat "$MEMCHECK_LOG" | grep -A3 "ERROR SUMMARY\|definitely lost\|Invalid"; }

echo ""
cat "$MEMCHECK_LOG" | grep -E "ERROR SUMMARY|LEAK SUMMARY|definitely lost|indirectly lost|possibly lost|still reachable|suppressed" || true

# ── Step 4: Helgrind ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  [4/4] Valgrind: Helgrind (threading / data races)"
echo "══════════════════════════════════════════════════════════"

HELGRIND_LOG="$LOG_DIR/helgrind.log"

valgrind \
    --tool=helgrind \
    --error-exitcode=1 \
    --log-file="$HELGRIND_LOG" \
    "$BIN" > /dev/null 2>&1 \
    && echo "  [PASS] Helgrind: no threading errors" \
    || { echo "  [FAIL] Helgrind detected threading issues — see $HELGRIND_LOG"; cat "$HELGRIND_LOG" | grep -A3 "ERROR SUMMARY"; }

echo ""
cat "$HELGRIND_LOG" | grep "ERROR SUMMARY" || true

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
