# TinyGNN

A dependency-free C++ inference engine optimized for Sparse Graph Neural Networks.

TinyGNN pivots from a dense-only deep learning runtime to a **sparse-native architecture**, demonstrating low-level systems programming, memory architecture optimizations, and parallel computing -- with zero external dependencies.

---

## Project Roadmap

| Phase | Title | Status |
|-------|-------|--------|
| 1 | Hybrid Tensor Core | Complete |
| 2 | Graph Data Loader | Complete |
| 3 | Dense Compute Kernels (GEMM) | Complete |
| 4 | Parallel Execution Engine | Planned |

---

## Project Structure

```
TinyGNN/
├── CMakeLists.txt
├── include/
│   └── tinygnn/
│       ├── tensor.hpp          # StorageFormat enum + Tensor struct
│       ├── graph_loader.hpp    # GraphData + GraphLoader (CSV -> CSR pipeline)
│       └── ops.hpp             # compute kernels: matmul (dense GEMM)
├── src/
│   ├── tensor.cpp
│   ├── graph_loader.cpp        # CSV parser + edge-list-to-CSR conversion
│   └── ops.cpp                 # matmul implementation (i,k,j loop order)
├── tests/
│   ├── test_tensor.cpp         # Phase 1 -- 104 assertions
│   ├── test_graph_loader.cpp   # Phase 2 -- 196 assertions
│   └── test_matmul.cpp         # Phase 3 -- 268 assertions
└── scripts/
    ├── install_wsl_tools.sh    # one-time WSL tooling setup
    ├── sanitizers.sh           # ASan + UBSan (9 configs, all 3 phases)
    └── valgrind_all.sh         # Memcheck + Helgrind + Callgrind (all 3 phases)
```

---

## Building

### Prerequisites
- C++17-capable compiler (GCC 8+, Clang 7+, MSVC 2019+)
- CMake 3.16+  *(optional -- g++ directly also works)*

### With CMake
```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

### Directly with g++
```bash
# Phase 1 -- Tensor
g++ -std=c++17 -Wall -Wextra -I include \
    src/tensor.cpp tests/test_tensor.cpp \
    -o build/test_tensor && ./build/test_tensor

# Phase 2 -- Graph Data Loader
g++ -std=c++17 -Wall -Wextra -I include \
    src/tensor.cpp src/graph_loader.cpp tests/test_graph_loader.cpp \
    -o build/test_graph_loader && ./build/test_graph_loader

# Phase 3 -- Dense GEMM
g++ -std=c++17 -Wall -Wextra -I include \
    src/tensor.cpp src/ops.cpp tests/test_matmul.cpp \
    -o build/test_matmul && ./build/test_matmul
```

### Memory safety (WSL / Linux)
```bash
bash scripts/sanitizers.sh       # 9 sanitizer configs across all phases
bash scripts/valgrind_all.sh     # Memcheck + Helgrind + Callgrind
```

---

## Phase 1 -- Hybrid Tensor Core

### Goal
Establish the fundamental data structure by expanding a contiguous memory design to support both dense and sparse graph layouts, without any third-party libraries.

### Design

#### StorageFormat enum
```cpp
enum class StorageFormat : uint8_t {
    Dense     = 0,   // Row-major contiguous storage
    SparseCSR = 1    // Compressed Sparse Row
};
```

#### Tensor struct layout

```
Dense:
  data_    -> [v00, v01, ..., v(R-1)(C-1)]   rows*cols floats, row-major
  strides_ -> {cols, 1}

SparseCSR:
  row_ptr_ -> size (rows+1)   row_ptr[i]..row_ptr[i+1] = col indices of row i
  col_ind_ -> size nnz        column index of each non-zero
  data_    -> size nnz        value of each non-zero
```

#### Memory footprint

| Format | Formula | 1000x1000 example |
|--------|---------|-------------------|
| Dense | rows * cols * 4 bytes | 4,000,000 bytes (3.8 MB) |
| SparseCSR | nnz*4 + nnz*4 + (rows+1)*4 bytes | 44,004 bytes with 5000 edges (43 KB) |

Sparse uses ~1.1% of the memory of the equivalent dense matrix in this benchmark.

### Results

```
Dense  1000x1000 memory          = 4,000,000 bytes
Sparse 1000x1000 (5000 nnz)      =    44,004 bytes
Memory ratio (sparse / dense)    =      1.10%
Very sparse (10 nnz in 10k x10k) =      0.01%
```

### Test suite -- 104 assertions, 18 test functions, 0 failures

| Category | Functions |
|----------|-----------|
| Dense construction and mutation | 3 |
| Memory footprint validation | 2 |
| Memory comparison / reduction ratios | 2 |
| Edge cases (empty, 1x1, 0-nnz, degenerate shapes, CSR data integrity) | 6 |
| Error handling (invalid args, out-of-bounds, wrong format access) | 4 |
| Utility (repr, enum values) | 1 |

```
Total : 104
Passed: 104
Failed: 0
```

---

## Phase 2 -- Graph Data Loader

### Goal
Load irregular graph structures from CSV files and convert them to hardware-friendly sorted CSR format, ready for GNN inference.

### Design

#### GraphLoader class
```cpp
class GraphLoader {
public:
    // Edge-list CSV  ->  vector of (src, dst) pairs
    static std::vector<std::pair<int32_t, int32_t>>
        parse_edges(const std::string& path);

    // Feature CSV  ->  Dense tensor (num_nodes x num_features)
    static Tensor parse_features(const std::string& path);

    // Raw edge list  ->  sorted SparseCSR adjacency tensor
    static Tensor edge_list_to_csr(
        const std::vector<std::pair<int32_t, int32_t>>& edges,
        std::size_t num_nodes);

    // Full pipeline: CSV files  ->  GraphData (adjacency + features)
    static GraphData load(const std::string& edges_path,
                          const std::string& features_path);
};
```

#### CSR conversion algorithm -- O(E + V + E*log(E/V))

```
Step 1: Count out-degree per node         O(E)
Step 2: Build row_ptr via prefix sum      O(V)
Step 3: Fill col_ind via offset cursors   O(E)
Step 4: Sort col_ind within each row      O(E*log(E/V)) amortised
```

#### CSV format support
- Automatic header row detection (skips non-numeric first lines)
- Both LF and CRLF line endings
- Whitespace-tolerant parsing
- Out-of-order node IDs with gap zero-filling in feature tensor

### Results (Cora-scale synthetic benchmark)

```
Nodes:     2,708
Edges:     10,556
Features:  1,433 per node

Adjacency: Tensor(2708x2708, SparseCSR, 95,284 bytes)
Features:  Tensor(2708x1433, Dense, 15,522,256 bytes)
```

### Test suite -- 196 assertions, 37 test functions, 0 failures

| Category | Functions | Covers |
|----------|-----------|--------|
| Edge CSV parsing | 6 | header, no-header, CRLF, self-loops, trailing blanks |
| Feature CSV parsing | 6 | ordered, unordered, sparse IDs, negative values |
| CSR conversion | 8 | exact arrays, sort invariant, empty rows, memory footprint |
| Full load pipeline | 4 | node-0 neighbors, all-nodes verify, feature zero-expansion |
| Cora-scale validation | 3 | 2708 nodes / 10556 edges / 1433 features / row_ptr invariants |
| Error handling | 10 | file-not-found, malformed lines, negative IDs, out-of-range |

```
Total : 196
Passed: 196
Failed: 0
```

---

## Phase 3 -- Dense Compute Kernels (GEMM)

### Goal
Implement the baseline dense matrix multiplication needed for the GNN feature transformation H' = H x W, where H is the node feature matrix and W is the learned weight matrix.

### Design

#### matmul function
```cpp
// C = A x B
// A: Dense (M x K)   B: Dense (K x N)   C: Dense (M x N), newly allocated
Tensor matmul(const Tensor& A, const Tensor& B);
```

#### Loop order -- (i, k, j)

The implementation uses the **(i, k, j)** traversal order rather than the naive (i, j, k):

```
for i in [0, M):           // row of A and C
  for k in [0, K):         // shared dimension
    a_ik = A[i][k]         // hoisted into scalar register
    for j in [0, N):       // row of B, column of C
      C[i][j] += a_ik * B[k][j]
```

**Why (i, k, j):** `A[i][k]` is loop-invariant across the inner j-loop and lives in a register. `B[k][j]` is accessed row-sequentially (cache-friendly). The compiler can auto-vectorise the inner j-loop because the scalar `a_ik` eliminates a load dependency. `__restrict__` pointers are used to communicate the no-aliasing guarantee to the compiler.

#### Complexity

| Metric | Value |
|--------|-------|
| Time complexity | O(M * K * N) |
| FLOPs | 2 * M * K * N (1 multiply + 1 add per A element) |
| Output allocation | M * N * 4 bytes |

#### Preconditions enforced
- Both operands must be StorageFormat::Dense (sparse-dense SpMM is Phase 4)
- A.cols() must equal B.rows() -- throws std::invalid_argument with both dimensions in the message
- Zero-dimension tensors produce a valid zero-filled output

### Results

```
matmul(128x128, 128x128)         =    1 ms
matmul(1024x32, 32x256)          =    4 ms  (output = 1024 KB)
matmul(256x256, I_256)           spot checks: 8/8 pass
matmul(5x8, 8x6) footprint       = 120 bytes (5*6*4)
```

GNN feature transform (H x W, one-hot nodes selecting rows of W):

```
H (3x4)  x  W (4x2)  ->  H' (3x2)
Node 0: [1, 2]   Node 1: [3, 4]   Node 2: [5, 6]
```

4x4 hardcoded result (spec required):

```
A = reshape([1..16], 4, 4)    B = reshape([17..32], 4, 4)

C = A x B:
  [  250,  260,  270,  280 ]
  [  618,  644,  670,  696 ]
  [  986, 1028, 1070, 1112 ]
  [ 1354, 1412, 1470, 1528 ]
```

### Test suite -- 268 assertions, 30 test functions, 0 failures

| Category | Functions | Covers |
|----------|-----------|--------|
| 4x4 hardcoded result (spec) | 3 | full matrix, element spot-checks, 2x2 sub-block cross-check |
| Non-square shapes | 3 | (2x3)x(3x4), matrix x column-vector, row-vector x matrix |
| Identity and zero properties | 4 | A*I=A, I*A=A, non-square identity, zero matrix output |
| GNN feature transform | 1 | H x W with one-hot nodes verifying row selection from W |
| Algebraic properties | 3 | non-commutativity, associativity, scalar distributivity |
| Output tensor properties | 3 | format=Dense, shape MxN, footprint = M*N*4 bytes |
| Stress / scale | 3 | 128x128 all-ones, 1024x32x256 rectangle, 256x256 identity |
| Dimension mismatch errors | 5 | basic throw, message content, 4 bad-shape combos, degenerate 0-dim |
| Sparse input errors | 2 | sparse A throws, sparse B throws (SparseCSR in message) |
| Edge / degenerate cases | 3 | 1x1 scalar, all-zeros output, GNN layer chain (AB)C = A(BC) |

```
Total : 268
Passed: 268
Failed: 0
```

---

## Cumulative Test Statistics

| Phase | Test file | Functions | Assertions | Result |
|-------|-----------|-----------|------------|--------|
| 1 | test_tensor.cpp | 18 | 104 | 104 / 104 |
| 2 | test_graph_loader.cpp | 37 | 196 | 196 / 196 |
| 3 | test_matmul.cpp | 30 | 268 | 268 / 268 |
| **Total** | | **85** | **568** | **568 / 568** |

---

## Memory Safety

All three phases pass the full sanitizer matrix with zero errors:

| Config | Tensor | Graph Loader | Matmul |
|--------|--------|--------------|--------|
| ASan + LSan | Pass | Pass | Pass |
| UBSan | Pass | Pass | Pass |
| ASan + UBSan combined | Pass | Pass | Pass |
| Valgrind Memcheck | 0 errors, 0 leaks | 0 errors, 0 leaks | 0 errors, 0 leaks |
| Valgrind Helgrind | 0 races | 0 races | 0 races |

---

## Design Principles

- **Zero dependencies** -- no Eigen, no BLAS, no Boost, no external libraries of any kind
- **Exact memory accounting** -- memory_footprint_bytes() returns precise byte counts for both storage formats
- **Fail-fast validation** -- CSR construction validates row_ptr monotonicity, column index bounds, and size consistency; matmul validates format and inner dimensions before allocating output
- **Unified interface** -- Dense and SparseCSR tensors share the same Tensor type; format is a runtime tag, not a separate class hierarchy
- **Cache-aware loop ordering** -- GEMM uses (i,k,j) order with a register-hoisted scalar to enable compiler auto-vectorisation on the inner loop
