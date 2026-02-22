#!/usr/bin/env bash
# scripts/run_valgrind_phase2.sh
# Runs Valgrind memcheck on both tensor and graph_loader test binaries
set -euo pipefail

PROJ_ROOT="/mnt/c/Users/Jai Ansh Bindra/TinyGNN"
BUILD_DIR="$PROJ_ROOT/build/valgrind"
LOG_DIR="$BUILD_DIR/logs"

mkdir -p "$BUILD_DIR" "$LOG_DIR"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  [1] Building debug binaries"
echo "══════════════════════════════════════════════════════════"

BIN_T="$BUILD_DIR/test_tensor_vg"
BIN_G="$BUILD_DIR/test_graph_loader_vg"

g++ -std=c++17 -g -O0 -fno-inline -fno-omit-frame-pointer \
    -I"$PROJ_ROOT/include" \
    "$PROJ_ROOT/src/tensor.cpp" \
    "$PROJ_ROOT/tests/test_tensor.cpp" \
    -o "$BIN_T"

g++ -std=c++17 -g -O0 -fno-inline -fno-omit-frame-pointer \
    -I"$PROJ_ROOT/include" \
    "$PROJ_ROOT/src/tensor.cpp" \
    "$PROJ_ROOT/src/graph_loader.cpp" \
    "$PROJ_ROOT/tests/test_graph_loader.cpp" \
    -o "$BIN_G"

echo "  Built: $BIN_T"
echo "  Built: $BIN_G"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  [2] Baseline (no Valgrind)"
echo "══════════════════════════════════════════════════════════"
echo "  -- tensor --"
"$BIN_T" 2>&1 | grep -E "Total|Passed|Failed"
echo "  -- graph_loader --"
"$BIN_G" 2>&1 | grep -E "Total|Passed|Failed"

FAIL=0

run_memcheck() {
    local name="$1"
    local bin="$2"
    local log="$LOG_DIR/memcheck_${name}.log"

    echo ""
    echo "  Memcheck: $name"
    valgrind \
        --tool=memcheck \
        --leak-check=full \
        --show-leak-kinds=all \
        --track-origins=yes \
        --errors-for-leak-kinds=all \
        --error-exitcode=1 \
        --log-file="$log" \
        "$bin" > /dev/null 2>&1 \
        && echo "  [PASS] Memcheck ($name): no errors" \
        || { echo "  [FAIL] Memcheck ($name) — see $log"; ((FAIL++)) || true; }

    grep -E "ERROR SUMMARY|definitely lost|indirectly lost|possibly lost" "$log" || true
}

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  [3] Valgrind Memcheck"
echo "══════════════════════════════════════════════════════════"

run_memcheck "tensor" "$BIN_T"
run_memcheck "graph_loader" "$BIN_G"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Valgrind Results: failures=$FAIL"
echo "══════════════════════════════════════════════════════════"

[[ $FAIL -eq 0 ]] && exit 0 || exit 1
