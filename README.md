# TinyGNN

> A dependency-free C++ inference engine optimized for Sparse Graph Neural Networks.

TinyGNN pivots from a dense-only deep learning runtime to a **sparse-native architecture**, demonstrating low-level systems programming, memory architecture optimizations, and parallel computing — with zero external dependencies.

---

## Project Roadmap

| Phase | Title | Status |
|-------|-------|--------|
| **1** | Hybrid Tensor Core | ✅ Complete |
| **2** | Graph Data Loader | ✅ Complete |
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
│       ├── tensor.hpp          # StorageFormat enum + Tensor struct
│       └── graph_loader.hpp    # GraphData + GraphLoader (CSV → CSR pipeline)
├── src/
│   ├── tensor.cpp              # Tensor implementation
│   └── graph_loader.cpp        # CSV parser + edge-list-to-CSR conversion
├── tests/
│   ├── test_tensor.cpp         # Phase 1 tests (104 assertions)
│   └── test_graph_loader.cpp   # Phase 2 tests (196 assertions)
└── scripts/
    ├── sanitizers.sh           # ASan + UBSan testing
    ├── valgrind_all.sh         # Memcheck + Helgrind + Callgrind
    ├── run_sanitizers_phase2.sh # WSL-compatible sanitizer runner
    └── run_valgrind_phase2.sh  # WSL-compatible valgrind runner
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
# Phase 1 — Tensor tests
g++ -std=c++17 -Wall -Wextra -static -I include \
    src/tensor.cpp tests/test_tensor.cpp -o build/test_tensor
./build/test_tensor

# Phase 2 — GraphLoader tests
g++ -std=c++17 -Wall -Wextra -static -I include \
    src/tensor.cpp src/graph_loader.cpp tests/test_graph_loader.cpp \
    -o build/test_graph_loader
./build/test_graph_loader
```

---

## Phase 2 — Graph Data Loader (Ingestion Pipeline)

### Goal
Load irregular graph structures from CSV and convert them to hardware-friendly sorted CSR format for GNN inference.

### Design

#### `GraphLoader` Class
```cpp
class GraphLoader {
public:
    static std::vector<std::pair<int32_t, int32_t>>
        parse_edges(const std::string& path);        // edge-list CSV → pairs

    static Tensor parse_features(const std::string& path);  // feature CSV → Dense tensor

    static Tensor edge_list_to_csr(                  // raw edges → sorted CSR
        const std::vector<std::pair<int32_t, int32_t>>& edges,
        std::size_t num_nodes);

    static GraphData load(const std::string& edges_path,    // full pipeline
                          const std::string& features_path);
};
```

#### CSR Conversion Algorithm — O(E + V + E·log(E/V))
```
Step 1: Count out-degree per node         — O(E)
Step 2: Build row_ptr via prefix sum      — O(V)
Step 3: Fill col_ind via offset insertion  — O(E)
Step 4: Sort col_ind within each row      — O(E·log(E/V)) amortised
```

#### CSV Format Support
- Automatic header row detection (skips non-numeric first lines)
- Both LF and CRLF line endings
- Whitespace-tolerant parsing
- Out-of-order node IDs with gap zero-filling

### Results (Cora-Scale)

```
Nodes:     2,708
Edges:     10,556
Features:  1,433 per node

Adjacency: Tensor(2708x2708, SparseCSR, 95,284 bytes)
Features:  Tensor(2708x1433, Dense, 15,522,256 bytes)
```

### Test Suite
196 assertions across 37 test functions.

| Category | Tests | Assertions |
|----------|-------|------------|
| Edge CSV parsing | 6 | header, no-header, CRLF, self-loops, trailing blanks |
| Feature CSV parsing | 6 | ordered, unordered, sparse IDs, negatives |
| CSR conversion | 8 | exact arrays, sorting, empty rows, memory footprint |
| Full load pipeline | 4 | node-0 neighbors, all-nodes verify, feature expansion |
| Cora-scale validation | 3 | 2708 nodes / 10556 edges / 1433 features |
| Error handling | 10 | file-not-found, malformed, negative IDs, bounds |

```
Total : 196
Passed: 196
Failed: 0
```

### Memory Safety Verification
- **AddressSanitizer + LeakSanitizer**: PASS
- **UndefinedBehaviorSanitizer**: PASS
- **Combined ASan + UBSan**: PASS
- **Valgrind Memcheck**: 0 errors, 0 leaks

---

## Design Principles

- **Zero dependencies** — no Eigen, no BLAS, no external libraries
- **Exact memory accounting** — `memory_footprint_bytes()` returns precise byte counts for both formats
- **Fail-fast validation** — CSR construction validates `row_ptr` monotonicity, column index bounds, and size consistency at construction time
- **Unified interface** — Dense and CSR tensors share the same `Tensor` type; format is a runtime tag