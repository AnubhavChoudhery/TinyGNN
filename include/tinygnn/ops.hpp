#pragma once

// ============================================================================
//  TinyGNN — Compute Kernels  (Phase 3 + Phase 4)
//  include/tinygnn/ops.hpp
//
//  This header declares the compute operations used inside GNN layers:
//
//    matmul(A, B)  — dense general matrix multiply (GEMM):  C = A × B
//    spmm(A, B)    — sparse-dense matrix multiply  (SpMM):  C = A × B
//
//  Design principles:
//    • matmul: Dense × Dense GEMM (Phase 3).
//    • spmm:  SparseCSR × Dense SpMM — the heart of GNN message passing
//             (Phase 4).  SpMM(Adj, H) aggregates neighbor features.
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

}  // namespace tinygnn
