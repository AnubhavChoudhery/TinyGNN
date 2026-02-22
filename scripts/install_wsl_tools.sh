#!/usr/bin/env bash
# scripts/install_wsl_tools.sh
# Run this ONCE inside WSL (Ubuntu-24.04) to install all analysis tools.
# Usage (from project root, inside WSL terminal):
#   bash scripts/install_wsl_tools.sh
set -euo pipefail

echo "==> Updating package lists..."
sudo apt-get update -qq

echo "==> Installing g++, valgrind, gprof, kcachegrind deps..."
sudo apt-get install -y \
    g++ \
    valgrind \
    binutils \
    linux-tools-common \
    linux-tools-generic \
    2>&1 | grep -v "^Get\|^Hit\|^Ign\|^Reading\|^Building\|^Selecting"

echo ""
echo "Installed versions:"
g++ --version | head -1
valgrind --version

echo ""
echo "==> Done. Now run: bash scripts/valgrind_all.sh"
