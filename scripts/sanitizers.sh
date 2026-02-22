#!/usr/bin/env bash
# scripts/sanitizers.sh
# ─────────────────────────────────────────────────────────────────────────────
# Builds and runs TinyGNN tests under three sanitizer configurations:
#   1. ASan+LSan  — AddressSanitizer + LeakSanitizer (memory errors, leaks)
#   2. UBSan      — UndefinedBehaviorSanitizer (integer overflow, OOB, etc.)
#   3. ASan+UBSan — Combined for maximum coverage
#
# Run from project root inside WSL Ubuntu-24.04:
#   bash scripts/sanitizers.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJ_ROOT/build/sanitizers"
INC="-I $PROJ_ROOT/include"
SRCS_TENSOR="$PROJ_ROOT/src/tensor.cpp $PROJ_ROOT/tests/test_tensor.cpp"
SRCS_GRAPH="$PROJ_ROOT/src/tensor.cpp $PROJ_ROOT/src/graph_loader.cpp $PROJ_ROOT/tests/test_graph_loader.cpp"
SRCS_MATMUL="$PROJ_ROOT/src/tensor.cpp $PROJ_ROOT/src/ops.cpp $PROJ_ROOT/tests/test_matmul.cpp"
SRCS_SPMM="$PROJ_ROOT/src/tensor.cpp $PROJ_ROOT/src/ops.cpp $PROJ_ROOT/tests/test_spmm.cpp"
CXXFLAGS="-std=c++17 -g -O1 -fno-omit-frame-pointer"

mkdir -p "$BUILD_DIR"

PASS=0
FAIL=0

run_config() {
    local name="$1"
    local flags="$2"
    local srcs="$3"
    local bin="$BUILD_DIR/test_${name}"

    echo ""
    echo "──────────────────────────────────────────────────────"
    printf "  Build: %-35s\n" "$name"
    echo "──────────────────────────────────────────────────────"

    # Build
    g++ $CXXFLAGS $flags $INC $srcs -o "$bin" 2>&1

    # Run — capture output + exit code
    local out
    local exit_code=0
    out=$(ASAN_OPTIONS="detect_leaks=1:halt_on_error=1" \
          UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1" \
          "$bin" 2>&1) || exit_code=$?

    local summary
    summary=$(echo "$out" | grep -E "Total|Passed|Failed" || true)

    if [[ $exit_code -eq 0 ]]; then
        echo "  [PASS] $name — $summary"
        ((PASS++)) || true
    else
        echo "  [FAIL] $name — exit code $exit_code"
        echo "$out" | tail -20
        ((FAIL++)) || true
    fi
}

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  TinyGNN — Sanitizer Test Suite"
echo "══════════════════════════════════════════════════════════"

# ── Tensor tests ──
# 1. AddressSanitizer + LeakSanitizer
run_config "tensor_asan_lsan" "-fsanitize=address -fsanitize=leak" "$SRCS_TENSOR"

# 2. UndefinedBehaviorSanitizer
run_config "tensor_ubsan" "-fsanitize=undefined" "$SRCS_TENSOR"

# 3. Combined ASan + UBSan
run_config "tensor_asan_ubsan" "-fsanitize=address,undefined" "$SRCS_TENSOR"

# ── GraphLoader tests ──
# 4. AddressSanitizer + LeakSanitizer
run_config "graph_asan_lsan" "-fsanitize=address -fsanitize=leak" "$SRCS_GRAPH"

# 5. UndefinedBehaviorSanitizer
run_config "graph_ubsan" "-fsanitize=undefined" "$SRCS_GRAPH"

# 6. Combined ASan + UBSan
run_config "graph_asan_ubsan" "-fsanitize=address,undefined" "$SRCS_GRAPH"

# ── Matmul (GEMM) tests ──
# 7. AddressSanitizer + LeakSanitizer
run_config "matmul_asan_lsan" "-fsanitize=address -fsanitize=leak" "$SRCS_MATMUL"

# 8. UndefinedBehaviorSanitizer
run_config "matmul_ubsan" "-fsanitize=undefined" "$SRCS_MATMUL"

# 9. Combined ASan + UBSan
run_config "matmul_asan_ubsan" "-fsanitize=address,undefined" "$SRCS_MATMUL"

# ── SpMM tests ──
# 10. AddressSanitizer + LeakSanitizer
run_config "spmm_asan_lsan" "-fsanitize=address -fsanitize=leak" "$SRCS_SPMM"

# 11. UndefinedBehaviorSanitizer
run_config "spmm_ubsan" "-fsanitize=undefined" "$SRCS_SPMM"

# 12. Combined ASan + UBSan
run_config "spmm_asan_ubsan" "-fsanitize=address,undefined" "$SRCS_SPMM"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Sanitizer Results: $PASS passed, $FAIL failed"
echo "══════════════════════════════════════════════════════════"
echo ""

[[ $FAIL -eq 0 ]] && exit 0 || exit 1
