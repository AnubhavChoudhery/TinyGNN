#!/usr/bin/env bash
# =============================================================================
#  scripts/run_install_test.sh
#
#  Build & test the TinyGNN C++ install integration suite.
#  Mirrors run_sanitizers_phase2.sh in spirit.
#
#  Steps:
#    1. Configure + build + install the library into /tmp/tg_install
#    2. Build tests/test_install.cpp as a standalone CMake consumer
#       (i.e. uses find_package(tinygnn CONFIG REQUIRED), NOT the source tree)
#    3. Run the binary and report pass/fail
#
#  Usage:
#    bash scripts/run_install_test.sh
#    bash scripts/run_install_test.sh --no-reinstall   # skip library reinstall
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR=/tmp/tg_install_build
INSTALL_DIR=/tmp/tg_install
CONSUMER_DIR=/tmp/tg_install_consumer

REINSTALL=1
if [[ "${1:-}" == "--no-reinstall" ]]; then
    REINSTALL=0
fi

BOLD="\033[1m"
GREEN="\033[0;32m"
RED="\033[0;31m"
CYAN="\033[0;36m"
RESET="\033[0m"

echo -e "${BOLD}TinyGNN — C++ Install Integration Test${RESET}"
echo "========================================"
echo "  Repo:    $REPO_DIR"
echo "  Install: $INSTALL_DIR"
echo "  Consumer:$CONSUMER_DIR"
echo ""

# ── Step 1: Install the library ───────────────────────────────────────────────
if [[ $REINSTALL -eq 1 ]]; then
    echo -e "${CYAN}[1/3] Configure + build + install library${RESET}"
    rm -rf "$BUILD_DIR" "$INSTALL_DIR"
    cmake -S "$REPO_DIR" -B "$BUILD_DIR" \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
          -DTINYGNN_BUILD_TESTS=OFF \
          -DTINYGNN_BUILD_BENCHMARKS=OFF \
          -Wno-dev \
          -DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON \
          > /dev/null 2>&1

    cmake --build "$BUILD_DIR" -j"$(nproc)" > /dev/null 2>&1
    cmake --install "$BUILD_DIR" > /dev/null 2>&1
    echo -e "  ${GREEN}Library installed → $INSTALL_DIR${RESET}"
else
    echo -e "${CYAN}[1/3] Skipping library reinstall (--no-reinstall)${RESET}"
fi

# Verify install tree
if [[ ! -f "$INSTALL_DIR/share/tinygnn/tinygnn-config.cmake" ]]; then
    echo -e "${RED}ERROR: install tree incomplete — missing tinygnn-config.cmake${RESET}" >&2
    exit 1
fi

# ── Step 2: Build consumer project ───────────────────────────────────────────
echo -e "${CYAN}[2/3] Build test_install.cpp as standalone consumer${RESET}"
rm -rf "$CONSUMER_DIR"
mkdir -p "$CONSUMER_DIR"

# Write consumer CMakeLists.txt
cat > "$CONSUMER_DIR/CMakeLists.txt" << 'CMEOF'
cmake_minimum_required(VERSION 3.16)
project(tg_install_test CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Pull in the installed TinyGNN package
find_package(tinygnn CONFIG REQUIRED)

add_executable(test_install test_install.cpp)
target_link_libraries(test_install PRIVATE tinygnn::tinygnn_core)
CMEOF

# Copy the test source into the consumer directory
cp "$REPO_DIR/tests/test_install.cpp" "$CONSUMER_DIR/test_install.cpp"

cmake -S "$CONSUMER_DIR" -B "$CONSUMER_DIR/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$INSTALL_DIR" \
      -Wno-dev \
      > /dev/null 2>&1

cmake --build "$CONSUMER_DIR/build" -j"$(nproc)" > /dev/null 2>&1
echo -e "  ${GREEN}Consumer binary built → $CONSUMER_DIR/build/test_install${RESET}"

# ── Step 3: Run the tests ─────────────────────────────────────────────────────
echo -e "${CYAN}[3/3] Running install integration tests${RESET}"
echo ""

set +e
"$CONSUMER_DIR/build/test_install"
EXIT_CODE=$?
set -e

echo ""
echo "========================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}All install tests PASSED${RESET}"
else
    echo -e "${RED}${BOLD}Some install tests FAILED (exit $EXIT_CODE)${RESET}" >&2
fi
echo ""
exit $EXIT_CODE
