# Contributing to TinyGNN

Thank you for your interest in contributing to TinyGNN! This document provides guidelines and information for contributors.

## Authors

**Jai Ansh Singh Bindra and Anubhav Choudhery (under JBAC EdTech)**

## Getting Started

### Prerequisites

- **C++17 compiler**: GCC 8+, Clang 7+, or MSVC 2019+
- **CMake 3.16+**
- **Python 3.8+** (for Python bindings)
- **pybind11 ≥ 2.11** (for Python bindings)

### Setting Up the Development Environment

```bash
# Clone the repository
git clone https://github.com/JaiAnshSB/TinyGNN.git
cd TinyGNN

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install development dependencies
pip install -e ".[dev]"

# Build and test C++
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure

# Run Python tests
python -m pytest tests/test_python_bindings.py -v
```

## Development Workflow

### Code Style

**C++ (backend)**:
- C++17 standard
- 4-space indentation
- `snake_case` for functions and variables
- `PascalCase` for types and classes
- Comprehensive documentation comments (Doxygen-style)
- All public functions must have `@throws` documentation

**Python (frontend)**:
- Follow PEP 8
- Type hints where practical
- Docstrings for all public functions

### Adding a New Operation

1. **Declare** in the appropriate header (`include/tinygnn/ops.hpp` or `layers.hpp`)
2. **Implement** in the corresponding source (`src/ops.cpp` or `src/layers.cpp`)
3. **Bind** in `python/tinygnn_ext.cpp`
4. **Export** in `python/tinygnn/__init__.py`
5. **Test** with both C++ assertions and Python pytest

### Running Tests

```bash
# C++ tests (all 9 test suites, 18,000+ assertions)
cmake --build build --parallel
ctest --test-dir build --output-on-failure

# Python tests (49 tests)
python -m pytest tests/test_python_bindings.py -v

# Memory safety (Linux/WSL only)
bash scripts/sanitizers.sh       # ASan + UBSan
bash scripts/valgrind_all.sh     # Memcheck + Helgrind
```

### Building Documentation

```bash
# C++ docs (Doxygen)
doxygen Doxyfile
# Output: docs/doxygen/html/index.html

# Python docs (Sphinx)
pip install sphinx sphinx-rtd-theme breathe
cd docs && make html
# Output: docs/_build/html/index.html
```

## Pull Request Guidelines

1. **Branch from `main`** and target `main` for your PR
2. **Keep commits focused** — one logical change per commit
3. **Write descriptive commit messages** following the format:
   ```
   Phase N: Brief description of change

   Detailed explanation of what changed and why.
   Include test results where applicable.
   ```
4. **All existing tests must pass** — CI will verify this automatically
5. **Add tests for new functionality** — aim for comprehensive coverage
6. **Update documentation** when adding public API surface

## Architecture Overview

```
include/tinygnn/     → C++ public headers (tensor, ops, layers, model)
src/                 → C++ implementations
python/              → pybind11 bindings + Python package
tests/               → C++ test suites + Python pytest
benchmarks/          → Performance benchmark programs
scripts/             → Build helpers, training, validation
docs/                → Sphinx + Doxygen documentation
.github/workflows/   → GitHub Actions CI/CD
```

### Design Principles

- **Zero dependencies** — no Eigen, BLAS, Boost, or external libraries
- **Fail-fast validation** — descriptive exceptions for all invalid inputs
- **In-place where possible** — minimize memory allocations
- **Cache-aware** — (i,k,j) loop order, AVX2 SIMD, OpenMP parallelism
- **Operator fusion** — fused GAT/SAGE kernels reduce memory bandwidth

## Reporting Issues

When reporting bugs, please include:
- Operating system and compiler version
- Python version (if applicable)
- Steps to reproduce
- Expected vs. actual behavior
- Relevant error messages or stack traces

## License

By contributing to TinyGNN, you agree that your contributions will be licensed under the MIT License.
