#!/usr/bin/env python3
"""
TinyGNN — Build helper script for Python extension.

Builds the pybind11 C++ extension and runs all Python tests.

Usage:
    python scripts/build_python.py          # Build + test
    python scripts/build_python.py --build  # Build only
    python scripts/build_python.py --test   # Test only (assumes built)
"""

import os
import sys
import subprocess
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable


def run(cmd, **kwargs):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=ROOT, **kwargs)
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        sys.exit(1)
    return result


def build():
    print("\n═══ Building Python extension ═══")
    run(f'"{PYTHON}" setup.py build_ext --inplace --compiler=mingw32')
    print("  Build complete!")

    # Verify the .pyd exists
    import glob
    pyds = glob.glob(os.path.join(ROOT, "*.pyd"))
    if pyds:
        for p in pyds:
            size = os.path.getsize(p) / 1024
            print(f"  {os.path.basename(p)} ({size:.0f} KB)")
    else:
        print("  ERROR: No .pyd file found!")
        sys.exit(1)


def test():
    print("\n═══ Running Python binding tests ═══")
    run(f'"{PYTHON}" -m pytest tests/test_python_bindings.py -v --tb=short')
    print("  All tests passed!")


def main():
    parser = argparse.ArgumentParser(description="Build TinyGNN Python extension")
    parser.add_argument("--build", action="store_true", help="Build only")
    parser.add_argument("--test", action="store_true", help="Test only")
    args = parser.parse_args()

    if args.build:
        build()
    elif args.test:
        test()
    else:
        build()
        test()


if __name__ == "__main__":
    main()
