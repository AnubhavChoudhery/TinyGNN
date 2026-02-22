# TinyGNN

> A dependency-free C++ inference engine optimized for Sparse Graph Neural Networks.

TinyGNN pivots from a dense-only deep learning runtime to a **sparse-native architecture**, demonstrating low-level systems programming, memory architecture optimizations, and parallel computing — with zero external dependencies.

---

## Project Roadmap

| Phase | Title | Status |
|-------|-------|--------|
| **1** | Hybrid Tensor Core | ✅ Complete |
| 2 | Sparse Matrix Operations | 🔜 Planned |
| 3 | GNN Layer Primitives | 🔜 Planned |
| 4 | Parallel Execution Engine | 🔜 Planned |

---

## Phase 1 — Hybrid Tensor Core

### Goal
Establish the fundamental data structure by expanding a contiguous memory design to support sparse graphs natively, without any third-party libraries.

### Design

#### `StorageFormat` Enum
```cpp
enum class StorageFormat : uint8_t {
    Dense     = 0,   // Row-major contiguous storage
    SparseCSR = 1    // Compressed Sparse Row
};
```

#### `Tensor` Struct
The unified `Tensor` type handles both dense and sparse layouts through a single interface:

```
Dense layout:
  data_   → [v₀₀, v₀₁, ..., v_(R-1)(C-1)]   (rows × cols floats, row-major)
  strides_ → {cols, 1}

CSR layout:
  row_ptr_ → size (rows + 1): row_ptr[i] to row_ptr[i+1] = column indices of row i
  col_ind_ → size (nnz):      column index of each non-zero
  data_    → size (nnz):      value of each non-zero
```

#### Memory Footprint

| Format | Formula | 1000×1000 example |
|--------|---------|-------------------|
| Dense | `rows × cols × 4` bytes | **4,000,000 bytes** (≈3.8 MB) |
| SparseCSR | `nnz×4 + nnz×4 + (rows+1)×4` bytes | **44,004 bytes** with 5000 edges (≈43 KB) |

> **~99% memory reduction** for the 1000×1000 / 5000-edge benchmark.

### Results

```
Dense  1000×1000 memory          = 4,000,000 bytes
Sparse 1000×1000 (5000 nnz)      =    44,004 bytes
Memory ratio  (sparse / dense)   =      1.10%
Very sparse   (10 nnz in 10k×10k)=      0.01%
```

### Test Suite
104 assertions across 18 test functions — zero external dependencies, zero failures.

| Category | Tests |
|----------|-------|
| Dense construction & mutation | 3 |
| Memory footprint validation | 2 |
| Memory comparison / reduction ratios | 2 |
| Edge cases (empty, 1×1, 0-nnz, degenerate shapes, CSR data integrity) | 6 |
| Error handling (invalid args, out-of-bounds, wrong format access) | 4 |
| Utility (repr, enum values) | 1 |

```
Total : 104
Passed: 104
Failed: 0
```

---

## Project Structure

```
TinyGNN/
├── CMakeLists.txt              # C++17 build system (CMake)
├── include/
│   └── tinygnn/
│       └── tensor.hpp          # StorageFormat enum + Tensor struct (public API)
├── src/
│   └── tensor.cpp              # Tensor implementation
└── tests/
    └── test_tensor.cpp         # Dependency-free unit tests (104 assertions)
```

---

## Building

### Prerequisites
- C++17-capable compiler (GCC 8+, Clang 7+, MSVC 2019+)
- CMake 3.16+ *(or build manually with g++)*

### With CMake
```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

### Directly with g++ (MinGW / Linux)
```bash
# Compile
g++ -std=c++17 -Wall -Wextra -static -I include \
    src/tensor.cpp tests/test_tensor.cpp -o build/test_tensor

# Run tests
./build/test_tensor
```

---

## Design Principles

- **Zero dependencies** — no Eigen, no BLAS, no external libraries
- **Exact memory accounting** — `memory_footprint_bytes()` returns precise byte counts for both formats
- **Fail-fast validation** — CSR construction validates `row_ptr` monotonicity, column index bounds, and size consistency at construction time
- **Unified interface** — Dense and CSR tensors share the same `Tensor` type; format is a runtime tag