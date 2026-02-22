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
SRCS="$PROJ_ROOT/src/tensor.cpp $PROJ_ROOT/tests/test_tensor.cpp"
CXXFLAGS="-std=c++17 -g -O1 -fno-omit-frame-pointer"

mkdir -p "$BUILD_DIR"

PASS=0
FAIL=0

run_config() {
    local name="$1"
    local flags="$2"
    local bin="$BUILD_DIR/test_${name}"

    echo ""
    echo "──────────────────────────────────────────────────────"
    printf "  Build: %-35s\n" "$name"
    echo "──────────────────────────────────────────────────────"

    # Build
    g++ $CXXFLAGS $flags $INC $SRCS -o "$bin" 2>&1

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

# 1. AddressSanitizer + LeakSanitizer
run_config "asan_lsan" "-fsanitize=address -fsanitize=leak"

# 2. UndefinedBehaviorSanitizer
run_config "ubsan" "-fsanitize=undefined"

# 3. Combined ASan + UBSan
run_config "asan_ubsan" "-fsanitize=address,undefined"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Sanitizer Results: $PASS passed, $FAIL failed"
echo "══════════════════════════════════════════════════════════"
echo ""

[[ $FAIL -eq 0 ]] && exit 0 || exit 1
