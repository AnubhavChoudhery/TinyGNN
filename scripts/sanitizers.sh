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
SRCS_ACTIVATIONS="$PROJ_ROOT/src/tensor.cpp $PROJ_ROOT/src/ops.cpp $PROJ_ROOT/tests/test_activations.cpp"
SRCS_GCN="$PROJ_ROOT/src/tensor.cpp $PROJ_ROOT/src/ops.cpp $PROJ_ROOT/src/layers.cpp $PROJ_ROOT/tests/test_gcn.cpp"
SRCS_GRAPHSAGE="$PROJ_ROOT/src/tensor.cpp $PROJ_ROOT/src/ops.cpp $PROJ_ROOT/src/layers.cpp $PROJ_ROOT/tests/test_graphsage.cpp"
SRCS_GAT="$PROJ_ROOT/src/tensor.cpp $PROJ_ROOT/src/ops.cpp $PROJ_ROOT/src/layers.cpp $PROJ_ROOT/tests/test_gat.cpp"
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

# ── Activations tests ──
# 13. AddressSanitizer + LeakSanitizer
run_config "activations_asan_lsan" "-fsanitize=address -fsanitize=leak" "$SRCS_ACTIVATIONS"

# 14. UndefinedBehaviorSanitizer
run_config "activations_ubsan" "-fsanitize=undefined" "$SRCS_ACTIVATIONS"

# 15. Combined ASan + UBSan
run_config "activations_asan_ubsan" "-fsanitize=address,undefined" "$SRCS_ACTIVATIONS"

# ── GCN Layer tests ──
# 16. AddressSanitizer + LeakSanitizer
run_config "gcn_asan_lsan" "-fsanitize=address -fsanitize=leak" "$SRCS_GCN"

# 17. UndefinedBehaviorSanitizer
run_config "gcn_ubsan" "-fsanitize=undefined" "$SRCS_GCN"

# 18. Combined ASan + UBSan
run_config "gcn_asan_ubsan" "-fsanitize=address,undefined" "$SRCS_GCN"

# ── GraphSAGE Layer tests ──
# 19. AddressSanitizer + LeakSanitizer
run_config "graphsage_asan_lsan" "-fsanitize=address -fsanitize=leak" "$SRCS_GRAPHSAGE"

# 20. UndefinedBehaviorSanitizer
run_config "graphsage_ubsan" "-fsanitize=undefined" "$SRCS_GRAPHSAGE"

# 21. Combined ASan + UBSan
run_config "graphsage_asan_ubsan" "-fsanitize=address,undefined" "$SRCS_GRAPHSAGE"

# ── GAT Layer tests ──
# 22. AddressSanitizer + LeakSanitizer
run_config "gat_asan_lsan" "-fsanitize=address -fsanitize=leak" "$SRCS_GAT"

# 23. UndefinedBehaviorSanitizer
run_config "gat_ubsan" "-fsanitize=undefined" "$SRCS_GAT"

# 24. Combined ASan + UBSan
run_config "gat_asan_ubsan" "-fsanitize=address,undefined" "$SRCS_GAT"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Sanitizer Results: $PASS passed, $FAIL failed"
echo "══════════════════════════════════════════════════════════"
echo ""

[[ $FAIL -eq 0 ]] && exit 0 || exit 1
