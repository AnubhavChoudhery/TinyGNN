#pragma once

// ============================================================================
//  TinyGNN — Compute Kernels & Activations  (Phase 3 + Phase 4 + Phase 5)
//  include/tinygnn/ops.hpp
//
//  This header declares the compute operations used inside GNN layers:
//
//    matmul(A, B)               — dense general matrix multiply (GEMM)
//    spmm(A, B)                 — sparse-dense matrix multiply  (SpMM)
//    relu_inplace(X)            — in-place ReLU activation
//    leaky_relu_inplace(X, α)   — in-place Leaky ReLU (GAT)
//    elu_inplace(X, α)          — in-place ELU activation
//    sigmoid_inplace(X)         — in-place sigmoid activation
//    tanh_inplace(X)            — in-place tanh activation
//    gelu_inplace(X)            — in-place GELU (transformer GNNs)
//    softmax_inplace(X)         — in-place row-wise softmax (attention)
//    log_softmax_inplace(X)     — in-place row-wise log-softmax (NLL loss)
//    add_bias(X, bias)          — broadcasting bias addition
//
//  Design principles:
//    • matmul: Dense × Dense GEMM (Phase 3).
//    • spmm:  SparseCSR × Dense SpMM — message passing (Phase 4).
//    • Activations & utilities: in-place transformations on Dense
//      tensors used between GNN layers (Phase 5).
//    • All functions operate in single precision (float32).
//    • Zero external dependencies (no BLAS, no Eigen, no LAPACK).
//    • Clear, descriptive exceptions for every invalid input.
//
//  Internal implementation notes  (see ops.cpp for details):
//    matmul uses the canonical three-loop order (i, k, j) which
//    iterates over B in column order and A in row order, keeping the
//    innermost accumulation in a register.
//
//    spmm iterates over the sparse rows using row_ptr to find each row's
//    non-zero columns in col_ind, then accumulates the corresponding
//    dense rows from B into the output C.  This is the standard
//    CSR-SpMM algorithm used in GNN message passing.
//
//    Activations are applied element-wise in-place (or row-wise for
//    softmax / log-softmax).  All validate Dense format and throw on
//    SparseCSR input.
//
//  Future optimisation hooks:
//    • L1/L2 cache tiling (block / micro-kernel decomposition)
//    • SIMD vectorisation via compiler auto-vec or SSE/AVX intrinsics
//    • Parallel outer loop with OpenMP  #pragma omp parallel for
// ============================================================================

#include "tinygnn/tensor.hpp"

#include <cstddef>

namespace tinygnn {

// ============================================================================
//  matmul — Dense General Matrix-Matrix Multiply (GEMM)
// ============================================================================
//  Computes C = A × B where:
//    A  must be a Dense tensor of shape (M, K)
//    B  must be a Dense tensor of shape (K, N)
//    C  is  a  Dense tensor of shape (M, N), newly allocated
//
//  All values are float32; computation accumulates in float32.
//
//  Complexity: O(M × K × N)
//  Memory:     O(M × N)  for the output (A and B are read-only)
//
//  GNN usage:  Linear feature transformation  H' = H × W
//    H — (num_nodes × in_features)  node feature matrix
//
//  @throws std::invalid_argument  if A or B is not Dense
//  @throws std::invalid_argument  if A.cols() != B.rows() (dimension mismatch)
//  @throws std::invalid_argument  if either tensor has zero dimensions
// ============================================================================
Tensor matmul(const Tensor& A, const Tensor& B);

// ============================================================================
//  spmm — Sparse-Dense Matrix-Matrix Multiply (SpMM)
// ============================================================================
//  Computes C = A × B where:
//    A  must be a SparseCSR tensor of shape (M, K)
//    B  must be a Dense     tensor of shape (K, N)
//    C  is  a  Dense     tensor of shape (M, N), newly allocated
//
//  All values are float32; computation accumulates in float32.
//
//  Complexity: O(nnz(A) × N)       — proportional to edges × features
//  Memory:     O(M × N)            — for the output (A and B are read-only)
//
//  GNN usage:  Message-passing aggregation  H_agg = Adj × H
//    Adj — (num_nodes × num_nodes)  sparse adjacency matrix
//    H   — (num_nodes × features)   dense node feature matrix
//    H_agg[i] = Σ_j Adj[i][j] * H[j]  (sum neighbor features)
//
//  Algorithm (CSR-SpMM):
//    for each row i of A:
//      for each non-zero A[i][k] (found via row_ptr[i]..row_ptr[i+1]):
//        val = A.data[nz_idx]           // edge weight (1.0 for unweighted)
//        for each feature j in [0, N):
//          C[i][j] += val * B[k][j]     // accumulate neighbor k's features
//
//  The outer loop walks CSR rows (nodes).  The middle loop walks non-zero
//  columns (neighbors) — this is the graph traversal.  The inner loop
//  accumulates dense features — this is vectorisable and cache-friendly
//  because B[k][j] is contiguous in memory.
//
//  @throws std::invalid_argument  if A is not SparseCSR
//  @throws std::invalid_argument  if B is not Dense
//  @throws std::invalid_argument  if A.cols() != B.rows() (dimension mismatch)
// ============================================================================
Tensor spmm(const Tensor& A, const Tensor& B);

// ============================================================================
//                  Phase 5 — Activations & Utilities
// ============================================================================

// ============================================================================
//  relu_inplace — Rectified Linear Unit (in-place)
// ============================================================================
//  Applies f(x) = max(0, x) element-wise.
//
//  This is the most widely used activation in GNNs (GCN, GraphSAGE, GIN).
//  In a GNN layer: H' = ReLU(Adj × H × W + b)
//
//  Complexity: O(rows × cols)      — single pass over all elements
//  Memory:     O(1) extra          — modifies X in-place
//
//  @param X  Dense tensor to modify in-place
//  @throws std::invalid_argument  if X is not Dense
// ============================================================================
void relu_inplace(Tensor& X);

// ============================================================================
//  leaky_relu_inplace — Leaky ReLU (in-place)
// ============================================================================
//  Applies f(x) = x if x >= 0, else alpha * x.
//
//  Critical for Graph Attention Networks (GAT), where LeakyReLU(α=0.2) is
//  applied to attention coefficients before softmax normalization.
//
//  Complexity: O(rows × cols)
//  Memory:     O(1) extra
//
//  @param X      Dense tensor to modify in-place
//  @param alpha  Negative slope (default 0.01; GAT uses 0.2)
//  @throws std::invalid_argument  if X is not Dense
// ============================================================================
void leaky_relu_inplace(Tensor& X, float alpha = 0.01f);

// ============================================================================
//  elu_inplace — Exponential Linear Unit (in-place)
// ============================================================================
//  Applies f(x) = x if x >= 0, else alpha * (exp(x) - 1).
//
//  ELU provides a smooth, differentiable transition through zero, avoiding
//  the "dying ReLU" problem.  Used in some GNN architectures (EGNN, PNA).
//
//  Complexity: O(rows × cols)     — exp() on negative elements only
//  Memory:     O(1) extra
//
//  @param X      Dense tensor to modify in-place
//  @param alpha  Scale for negative region (default 1.0)
//  @throws std::invalid_argument  if X is not Dense
// ============================================================================
void elu_inplace(Tensor& X, float alpha = 1.0f);

// ============================================================================
//  sigmoid_inplace — Logistic Sigmoid (in-place)
// ============================================================================
//  Applies f(x) = 1 / (1 + exp(-x)).
//
//  Used for:
//    • Binary node/edge classification output
//    • Gating mechanisms (GRU-based GNNs like GGNN)
//    • Edge existence probability in link prediction
//
//  Output range: (0, 1) for all finite inputs.
//
//  Complexity: O(rows × cols)
//  Memory:     O(1) extra
//
//  @param X  Dense tensor to modify in-place
//  @throws std::invalid_argument  if X is not Dense
// ============================================================================
void sigmoid_inplace(Tensor& X);

// ============================================================================
//  tanh_inplace — Hyperbolic Tangent (in-place)
// ============================================================================
//  Applies f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
//
//  Used in LSTM/GRU-based graph networks (GGNN, MPNN) and some attention
//  mechanisms.  Output range: (-1, 1).
//
//  Complexity: O(rows × cols)
//  Memory:     O(1) extra
//
//  @param X  Dense tensor to modify in-place
//  @throws std::invalid_argument  if X is not Dense
// ============================================================================
void tanh_inplace(Tensor& X);

// ============================================================================
//  gelu_inplace — Gaussian Error Linear Unit (in-place)
// ============================================================================
//  Applies f(x) = x * Φ(x), approximated as:
//    f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
//
//  The standard activation in transformer-based GNNs (GPS, Graphormer,
//  TokenGT).  Provides a smooth, non-monotonic gate.
//
//  Complexity: O(rows × cols)
//  Memory:     O(1) extra
//
//  @param X  Dense tensor to modify in-place
//  @throws std::invalid_argument  if X is not Dense
// ============================================================================
void gelu_inplace(Tensor& X);

// ============================================================================
//  softmax_inplace — Row-wise Softmax (in-place)
// ============================================================================
//  For each row i of X (shape M × N):
//    X[i][j] = exp(X[i][j] - max_j(X[i])) / Σ_k exp(X[i][k] - max_k(X[i]))
//
//  Row-wise softmax is essential for:
//    • GAT attention weight normalization (per-node softmax over neighbors)
//    • Multi-class node classification output layer
//
//  The max-subtraction trick ensures numerical stability (prevents overflow).
//
//  Guarantees: each row sums to 1.0; all values in [0, 1].
//
//  Complexity: O(rows × cols)     — 3 passes per row (max, exp+sum, divide)
//  Memory:     O(1) extra
//
//  @param X  Dense tensor to modify in-place
//  @throws std::invalid_argument  if X is not Dense
//  @throws std::invalid_argument  if X has zero columns
// ============================================================================
void softmax_inplace(Tensor& X);

// ============================================================================
//  log_softmax_inplace — Row-wise Log-Softmax (in-place)
// ============================================================================
//  For each row i of X (shape M × N):
//    X[i][j] = (X[i][j] - max_j(X[i])) - log(Σ_k exp(X[i][k] - max_k(X[i])))
//
//  Equivalent to log(softmax(X)) but numerically stable.  Used with
//  negative log-likelihood (NLL) loss for multi-class node classification.
//
//  Guarantees: each row's values are all <= 0; exp of each row sums to 1.0.
//
//  Complexity: O(rows × cols)     — 3 passes per row (max, exp+sum, subtract)
//  Memory:     O(1) extra
//
//  @param X  Dense tensor to modify in-place
//  @throws std::invalid_argument  if X is not Dense
//  @throws std::invalid_argument  if X has zero columns
// ============================================================================
void log_softmax_inplace(Tensor& X);

// ============================================================================
//  add_bias — Broadcasting Bias Addition (in-place)
// ============================================================================
//  Adds a bias vector to every row of X:
//    X[i][j] += bias[0][j]   for all i in [0, rows), j in [0, cols)
//
//  The bias tensor must be shape (1 × N) where N == X.cols().
//  This broadcasts the single bias row across all M rows of X.
//
//  GNN usage:  H' = Activation(A × H × W + b)
//    After the linear transform (matmul/spmm), add a learned bias vector
//    to every node's feature vector before applying the activation.
//
//  Complexity: O(rows × cols)
//  Memory:     O(1) extra
//
//  @param X     Dense tensor to modify in-place (M × N)
//  @param bias  Dense tensor of shape (1 × N) — the bias row vector
//  @throws std::invalid_argument  if X is not Dense
//  @throws std::invalid_argument  if bias is not Dense
//  @throws std::invalid_argument  if bias is not shape (1 × N) with N == X.cols()
// ============================================================================
void add_bias(Tensor& X, const Tensor& bias);

}  // namespace tinygnn
