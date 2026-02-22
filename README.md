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
| 4 | Sparse Compute Kernels (SpMM) | Complete |
| 5 | Activations & Utilities | Complete |

---

## Project Structure

```
TinyGNN/
├── CMakeLists.txt
├── include/
│   └── tinygnn/
│       ├── tensor.hpp          # StorageFormat enum + Tensor struct
│       ├── graph_loader.hpp    # GraphData + GraphLoader (CSV -> CSR pipeline)
│       └── ops.hpp             # compute kernels + activations: matmul, spmm, relu, softmax, etc.
├── src/
│   ├── tensor.cpp
│   ├── graph_loader.cpp        # CSV parser + edge-list-to-CSR conversion
│   └── ops.cpp                 # matmul, spmm, relu, leaky_relu, elu, sigmoid, tanh, gelu, softmax, log_softmax, add_bias
├── tests/
│   ├── test_tensor.cpp         # Phase 1 -- 104 assertions
│   ├── test_graph_loader.cpp   # Phase 2 -- 269 assertions
│   ├── test_matmul.cpp         # Phase 3 -- 268 assertions
│   ├── test_spmm.cpp           # Phase 4 -- 306 assertions
│   └── test_activations.cpp    # Phase 5 -- 266 assertions
├── datasets/                   # downloaded via scripts/fetch_datasets.py
│   ├── cora/                   # 2,708 nodes, 5,429 edges, 1,433 features
│   └── reddit/                 # 232,965 nodes, 114,615,892 edges, 602 features
└── scripts/
    ├── install_wsl_tools.sh    # one-time WSL tooling setup
    ├── fetch_datasets.py       # download Cora + Reddit, convert to CSV
    ├── sanitizers.sh           # ASan + UBSan (15 configs, all 5 phases)
    └── valgrind_all.sh         # Memcheck + Helgrind + Callgrind (all 5 phases)
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

# Phase 4 -- Sparse-Dense SpMM
g++ -std=c++17 -Wall -Wextra -I include \
    src/tensor.cpp src/ops.cpp tests/test_spmm.cpp \
    -o build/test_spmm && ./build/test_spmm

# Phase 5 -- Activations & Utilities
g++ -std=c++17 -Wall -Wextra -I include \
    src/tensor.cpp src/ops.cpp tests/test_activations.cpp \
    -o build/test_activations && ./build/test_activations
```

### Memory safety (WSL / Linux)
```bash
bash scripts/sanitizers.sh       # 15 sanitizer configs across all phases
bash scripts/valgrind_all.sh     # Memcheck + Helgrind + Callgrind
```

---

## Datasets

TinyGNN tests against two real-world graph datasets. Download them with the included Python script:

```bash
# Cora only (no extra Python deps, ~170 KB download)
python3 scripts/fetch_datasets.py --cora-only

# Both Cora and Reddit (needs numpy + scipy, ~200 MB download)
pip install numpy scipy
python3 scripts/fetch_datasets.py
```

| Dataset | Source | Nodes | Edges | Features | CSV size |
|---------|--------|-------|-------|----------|----------|
| Cora | LINQS (McCallum et al. 2000) | 2,708 | 5,429 | 1,433 (binary bag-of-words) | 7.5 MB |
| Reddit | DGL / GraphSAGE (Hamilton et al. 2017) | 232,965 | 114,615,892 | 602 (GloVe embeddings) | 2.7 GB |

Files are placed in `datasets/cora/` and `datasets/reddit/` (gitignored).
Tests that require actual datasets skip gracefully when the files are not present.

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

### Results (Reddit-scale synthetic benchmark)

Reddit (Hamilton et al. 2017, GraphSAGE paper): 232,965 nodes, 114,615,892 directed edges, 602 features.
Generated synthetically with a fixed seed; full file pipeline writes ~1.4 GB edge CSV + ~282 MB feature CSV.

```
Nodes:     232,965
Edges:     114,615,892
Features:  602 per node

Adjacency: Tensor(232965x232965, SparseCSR, 917,859,000 bytes)  (~875 MB)
Features:  Tensor(232965x602,   Dense,      560,979,720 bytes)  (~535 MB)

In-memory CSR algorithm test:
  Edge generation (114M pairs):   1,350 ms
  CSR construction + sort:        5,591 ms
  Total:                          6,979 ms

Full file-pipeline test:
  Edge file write (1.4 GB):      25,491 ms
  Feature file write (282 MB):    3,748 ms
  GraphLoader::load():          119,047 ms
  Total:                        148,286 ms

Node 0 neighbors: 455 (CSR traversal matches raw CSV exactly)
```

### Results (actual Cora dataset)

The real Cora citation network (McCallum et al. 2000): 2,708 papers linked by 5,429 directed citations, with 1,433 binary bag-of-words features per paper.

```
Nodes:     2,708
Edges:     5,429
Features:  1,433 per node (binary)

Adjacency: Tensor(2708x2708, SparseCSR, 54,268 bytes)     (~53 KB)
Features:  Tensor(2708x1433, Dense,     15,522,256 bytes)  (~14.8 MB)

Feature density: 49,216 / 3,880,564 entries are 1  (1.27%)
Node 0 neighbors: 3 (CSR traversal matches raw CSV exactly)
```

### Results (actual Reddit dataset)

The real Reddit post network (Hamilton et al. 2017, GraphSAGE): 232,965 posts linked by 114,615,892 directed edges, with 602-dimensional GloVe word embeddings per post.

```
Nodes:     232,965
Edges:     114,615,892
Features:  602 per node (GloVe float)

Adjacency: Tensor(232965x232965, SparseCSR, 917,859,000 bytes)  (~875 MB)
Features:  Tensor(232965x602,   Dense,      560,979,720 bytes)  (~535 MB)

GraphLoader::load() time: 157 s  (WSL, Ubuntu 24.04, GCC 13.3)
Node 0 neighbors: 2,204
```

### Test suite -- 269 assertions, 49 test functions, 0 failures

| Category | Functions | Covers |
|----------|-----------|--------|
| Edge CSV parsing | 6 | header, no-header, CRLF, self-loops, trailing blanks |
| Feature CSV parsing | 6 | ordered, unordered, sparse IDs, negative values |
| CSR conversion | 8 | exact arrays, sort invariant, empty rows, memory footprint |
| Full load pipeline | 4 | node-0 neighbors, all-nodes verify, feature zero-expansion |
| Cora-scale validation | 3 | 2708 nodes / 10556 edges / 1433 features / row_ptr invariants |
| Reddit-scale validation | 4 | 232,965 nodes / 114,615,892 edges / in-memory CSR + full pipeline + node-0 match |
| Actual Cora dataset | 4 | 2,708 nodes / 5,429 real citation edges / binary features / node-0 match |
| Actual Reddit dataset | 4 | 232,965 nodes / 114,615,892 real edges / GloVe features / CSR invariants |
| Error handling | 10 | file-not-found, malformed lines, negative IDs, out-of-range |

```
Total : 269
Passed: 269
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

## Phase 4 -- Sparse Compute Kernels (SpMM)

### Goal
Implement the sparse-dense matrix multiplication kernel that is the heart of GNN message-passing: `H_agg = Adj × H`, aggregating each node's neighbor features via the sparse adjacency matrix.

### Design

#### spmm function
```cpp
// C = A × B
// A: SparseCSR (M × K)   B: Dense (K × N)   C: Dense (M × N), newly allocated
Tensor spmm(const Tensor& A, const Tensor& B);
```

#### Algorithm -- CSR-SpMM

```
for i in [0, M):                        // row of A (node)
  for nz in [row_ptr[i], row_ptr[i+1]): // non-zeros in row i (neighbors)
    k = col_ind[nz]                     // column (neighbor node ID)
    a_val = data[nz]                    // edge weight (1.0 for unweighted)
    for j in [0, N):                    // feature dimension
      C[i][j] += a_val * B[k][j]       // accumulate neighbor's features
```

**Why CSR-SpMM:** The outer loop walks rows (nodes). The middle loop walks non-zero columns (graph neighbors) — this *is* the message-passing. The inner loop accumulates dense features — cache-friendly because `B[k][j]` is contiguous in memory and the compiler can auto-vectorise with the scalar `a_val` hoisted to a register.

#### Complexity

| Metric | Value |
|--------|-------|
| Time complexity | O(nnz × N) — proportional to edges × features |
| FLOPs | 2 × nnz × N (1 multiply + 1 add per edge per feature) |
| Output allocation | M × N × 4 bytes |

#### Preconditions enforced
- A must be StorageFormat::SparseCSR (Dense A → "use matmul() instead")
- B must be StorageFormat::Dense (Sparse B → "Sparse × Sparse not supported")
- A.cols() must equal B.rows() — throws std::invalid_argument with dimensions in message
- Early exit for degenerate cases (M=0, N=0, or nnz=0) producing valid zero-filled output

### Results

3×3 hand-calculated SpMM (spec required):
```
A (SparseCSR):                   B (Dense):
  [1 1 0]  row_ptr=[0,2,3,5]      [1 2]
  [0 1 0]  col_ind=[0,1,1,0,2]    [3 4]
  [1 0 1]  values =[1,1,1,1,1]    [5 6]

C = spmm(A, B):
  [4, 6]   ← 1·[1,2] + 1·[3,4]
  [3, 4]   ← 1·[3,4]
  [6, 8]   ← 1·[1,2] + 1·[5,6]
```

4×4 weighted hand-calculated:
```
A (SparseCSR):                B (Dense):
  [2 0 0 0]  values=[2,3,     [1  0  2]
  [0 3 1 0]   1,1,2,1,1]      [0  1  0]
  [1 0 0 2]                    [3  0  1]
  [0 1 1 0]                    [0  2  0]

C = spmm(A, B):
  [2, 0, 4]  [3, 3, 1]  [1, 4, 2]  [3, 1, 1]
```

GNN message-passing (star graph, 5 nodes):
```
H_agg = spmm(Adj_star, H)
  Hub (node 0): sum of all features = [10, 40]
  Leaf i:       H[0] + H[i]  (receives from hub + self)
```

Equivalence with dense matmul verified for:
- 32×32 sparse (~9% density) × 32×16 dense
- Cora-scale: 2,708×2,708 sparse (3 nnz/row) × 2,708×32 dense

### Test suite -- 306 assertions, 35 test functions, 0 failures

| Category | Functions | Covers |
|----------|-----------|--------|
| 3×3 hand-calculated (spec) | 3 | full matrix, element checks, column vector |
| 4×4 weighted CSR | 2 | full result, element derivations |
| Non-square shapes | 3 | (5×3)×(3×4), (2×5)×(5×1), (1×4)×(4×3) |
| Identity & zero properties | 4 | sparse I × B = B, 64×64, zero-nnz, all-ones |
| GNN message-passing | 3 | triangle (fully connected), star graph, two-hop path chain |
| Dense matmul equivalence | 2 | small 3×3, medium 32×32 deterministic sparse |
| Output tensor properties | 3 | format=Dense, shape M×N, memory footprint |
| Stress / scale | 3 | 512×512 identity, 2708-node Cora-scale, 1024×128 |
| Wrong format errors | 4 | dense A throws, message check, sparse B throws, message check |
| Dimension mismatch | 3 | basic throw, message "dimension mismatch", multiple combos |
| Edge / degenerate cases | 5 | single nnz, empty rows, diagonal/self-loops, negative weights, 1×1 |

```
Total : 306
Passed: 306
Failed: 0
```

---

## Phase 5 -- Activations & Utilities

### Goal
Implement the element-wise activation functions and utilities needed between GNN layers: `H' = Activation(Adj × H × W + b)`.  These operate in-place on Dense tensors for zero-copy efficiency.

### Design

#### Activation functions

| Function | Formula | GNN Use Case |
|----------|---------|--------------|
| `relu_inplace(X)` | max(0, x) | GCN, GraphSAGE, GIN — standard hidden layer activation |
| `leaky_relu_inplace(X, α)` | x if x≥0, else αx (default α=0.01) | GAT attention coefficients (α=0.2) |
| `elu_inplace(X, α)` | x if x≥0, else α(eˣ−1) (default α=1.0) | EGNN, PNA — smooth negative region |
| `sigmoid_inplace(X)` | 1/(1+e⁻ˣ) | GGNN gating, link prediction, binary classification |
| `tanh_inplace(X)` | tanh(x) | GRU/LSTM graph networks (GGNN, MPNN) |
| `gelu_inplace(X)` | x·Φ(x) ≈ 0.5x(1+tanh(√(2/π)(x+0.044715x³))) | GPS, Graphormer, TokenGT — transformer GNNs |
| `softmax_inplace(X)` | row-wise softmax with max-subtraction stability | GAT attention normalization, classification output |
| `log_softmax_inplace(X)` | row-wise log-softmax (numerically stable) | NLL loss for node classification |

#### Utility functions

| Function | Operation | GNN Use Case |
|----------|-----------|--------------|
| `add_bias(X, bias)` | X[i][j] += bias[0][j] — broadcast (1×N) across M rows | Linear layer bias: H' = AHW + b |

#### Common properties
- All activations modify the tensor **in-place** (O(1) extra memory)
- All validate **Dense format** and throw `std::invalid_argument` on SparseCSR
- Softmax / log-softmax use the **max-subtraction trick** for numerical stability
- Sigmoid uses a **two-branch implementation** (x≥0 vs x<0) to prevent overflow
- GELU uses the **tanh approximation** (same as PyTorch `F.gelu(approximate='tanh')`)

#### Complexity

| Function | Time | Extra Memory |
|----------|------|-------------|
| relu, leaky_relu, sigmoid, tanh | O(M × N) | O(1) |
| elu, gelu | O(M × N) with exp/tanh on subset | O(1) |
| softmax, log_softmax | O(M × N) — 3 passes per row | O(1) |
| add_bias | O(M × N) | O(1) |

### Results

GCN layer pipeline (identity Adj, 3 nodes, 2 features):
```
H' = ReLU(Adj × H × W + b)
  Node 0: [1.5, 0.0]   (relu clamps negative bias-shifted value)
  Node 1: [2.5, 0.0]
  Node 2: [3.5, 0.0]
```

GAT-style attention (LeakyReLU + Softmax):
```
Scores:  [0.5, -0.3, 1.2]
LeakyReLU(α=0.2): [0.5, -0.06, 1.2]
Softmax: attention weights sum to 1.0, largest score gets most weight
```

Node classification (Log-Softmax for NLL loss):
```
3 nodes × 4 classes
All log-probabilities ≤ 0, exp(row) sums to 1.0
Argmax preserved through log-softmax
```

Numerical stability verified:
```
softmax([1000, 1001])     — no NaN/Inf, correct probabilities
log_softmax([1000, 1001, 999]) — no NaN/Inf, exp sums to 1.0
```

### Test suite -- 266 assertions, 70 test functions, 0 failures

| Category | Functions | Covers |
|----------|-----------|--------|
| ReLU | 5 | basic, 2D matrix, all-positive, all-negative, zero preservation |
| Leaky ReLU | 5 | default α=0.01, GAT α=0.2, 2D, α=0→ReLU, α=1→identity |
| ELU | 4 | basic, custom α=2.0, saturation at −α, 2D mixed signs |
| Sigmoid | 5 | basic values, σ(x)+σ(−x)=1 symmetry, bounds (0,1), hand-calculated, 2D |
| Tanh | 5 | basic, odd symmetry, bounds (−1,1), hand-calculated, 2D |
| GELU | 4 | basic (0→0, large→x, neg→0), hand-calculated vs reference, zero symmetry, 2D |
| Softmax | 7 | row sums=1, bounds [0,1], uniform dist, hand-calculated, multi-row, numerical stability, argmax preserved |
| Log-Softmax | 6 | exp sums=1, all ≤ 0, consistent with softmax, uniform, numerical stability, multi-row |
| Add bias | 6 | basic broadcasting, zero bias, negative bias, single-row, wrong-rows throw, wrong-cols throw |
| GNN pipeline integration | 3 | full GCN layer, GAT attention, node classification |
| Error handling (SparseCSR) | 10 | all 9 activations + add_bias reject sparse input |
| Stress / scale | 5 | relu 1024×256, softmax 1024×128, sigmoid 512×512, bias 2708×32, gelu 256×256 |
| Edge / degenerate | 5 | 1×1 through all activations, zero-cols rejected, zero-rows no-op, double-apply idempotency |

```
Total : 266
Passed: 266
Failed: 0
```

---

## Cumulative Test Statistics

| Phase | Test file | Functions | Assertions | Result |
|-------|-----------|-----------|------------|--------|
| 1 | test_tensor.cpp | 18 | 104 | 104 / 104 |
| 2 | test_graph_loader.cpp | 49 | 269 | 269 / 269 |
| 3 | test_matmul.cpp | 30 | 268 | 268 / 268 |
| 4 | test_spmm.cpp | 35 | 306 | 306 / 306 |
| 5 | test_activations.cpp | 70 | 266 | 266 / 266 |
| **Total** | | **202** | **1,213** | **1,213 / 1,213** |

---

## Memory Safety

All five phases pass the full sanitizer matrix with zero errors:

| Config | Tensor | Graph Loader | Matmul | SpMM | Activations |
|--------|--------|--------------|--------|------|-------------|
| ASan + LSan | Pass | Pass | Pass | Pass | Pass |
| UBSan | Pass | Pass | Pass | Pass | Pass |
| ASan + UBSan combined | Pass | Pass | Pass | Pass | Pass |
| Valgrind Memcheck | 0 errors, 0 leaks | 0 errors, 0 leaks | 0 errors, 0 leaks | 0 errors, 0 leaks | 0 errors, 0 leaks |
| Valgrind Helgrind | 0 races | 0 races | 0 races | 0 races | 0 races |

---

## Design Principles

- **Zero dependencies** -- no Eigen, no BLAS, no Boost, no external libraries of any kind
- **Exact memory accounting** -- memory_footprint_bytes() returns precise byte counts for both storage formats
- **Fail-fast validation** -- CSR construction validates row_ptr monotonicity, column index bounds, and size consistency; matmul validates format and inner dimensions before allocating output
- **Unified interface** -- Dense and SparseCSR tensors share the same Tensor type; format is a runtime tag, not a separate class hierarchy
- **Cache-aware loop ordering** -- GEMM uses (i,k,j) order with a register-hoisted scalar to enable compiler auto-vectorisation on the inner loop
- **Sparse-native message passing** -- SpMM walks the CSR structure directly (row_ptr → col_ind → accumulate), avoiding any sparse-to-dense conversion
- **In-place activations** -- all 8 activation functions modify tensors in-place with O(1) extra memory, avoiding unnecessary allocations in GNN inference pipelines
- **Numerical stability** -- softmax/log-softmax use the max-subtraction trick; sigmoid uses a two-branch formula to prevent overflow for extreme inputs
