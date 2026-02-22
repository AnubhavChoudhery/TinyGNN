#!/usr/bin/env bash
# scripts/run_sanitizers_phase2.sh
# Runs sanitizer tests for both tensor and graph_loader
set -euo pipefail

PROJ_ROOT="/mnt/c/Users/Jai Ansh Bindra/TinyGNN"
BUILD_DIR="$PROJ_ROOT/build/sanitizers"
INC="-I${PROJ_ROOT}/include"
CXXFLAGS="-std=c++17 -g -O1 -fno-omit-frame-pointer"

mkdir -p "$BUILD_DIR"

PASS=0
FAIL=0

run_one() {
    local name="$1"
    local san_flags="$2"
    shift 2
    local srcs=("$@")
    local bin="$BUILD_DIR/test_${name}"

    echo ""
    echo "── Build: $name ──"
    g++ $CXXFLAGS $san_flags "$INC" "${srcs[@]}" -o "$bin" 2>&1

    local exit_code=0
    ASAN_OPTIONS="detect_leaks=1:halt_on_error=1" \
    UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1" \
    "$bin" > /tmp/san_out_$$.txt 2>&1 || exit_code=$?

    local summary
    summary=$(grep -E "Total|Passed|Failed" /tmp/san_out_$$.txt || true)

    if [[ $exit_code -eq 0 ]]; then
        echo "  [PASS] $name — $summary"
        ((PASS++)) || true
    else
        echo "  [FAIL] $name — exit code $exit_code"
        tail -20 /tmp/san_out_$$.txt
        ((FAIL++)) || true
    fi
    rm -f /tmp/san_out_$$.txt
}

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  TinyGNN — Sanitizer Suite (Phase 1 + Phase 2)"
echo "══════════════════════════════════════════════════════════"

T_SRCS=("$PROJ_ROOT/src/tensor.cpp" "$PROJ_ROOT/tests/test_tensor.cpp")
G_SRCS=("$PROJ_ROOT/src/tensor.cpp" "$PROJ_ROOT/src/graph_loader.cpp" "$PROJ_ROOT/tests/test_graph_loader.cpp")

# Tensor tests
run_one "tensor_asan"  "-fsanitize=address -fsanitize=leak" "${T_SRCS[@]}"
run_one "tensor_ubsan"  "-fsanitize=undefined"               "${T_SRCS[@]}"
run_one "tensor_combo"  "-fsanitize=address,undefined"       "${T_SRCS[@]}"

# GraphLoader tests
run_one "graph_asan"   "-fsanitize=address -fsanitize=leak" "${G_SRCS[@]}"
run_one "graph_ubsan"  "-fsanitize=undefined"               "${G_SRCS[@]}"
run_one "graph_combo"  "-fsanitize=address,undefined"       "${G_SRCS[@]}"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Sanitizer Results: $PASS passed, $FAIL failed"
echo "══════════════════════════════════════════════════════════"

[[ $FAIL -eq 0 ]] && exit 0 || exit 1
