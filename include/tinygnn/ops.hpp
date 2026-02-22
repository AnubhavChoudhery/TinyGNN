#pragma once

// ============================================================================
//  TinyGNN — Dense Compute Kernels  (Phase 3)
//  include/tinygnn/ops.hpp
//
//  This header declares the compute operations used inside GNN layers:
//
//    matmul(A, B)  — dense general matrix multiply (GEMM):  C = A × B
//
//  Design principles:
//    • Dense-only in this phase; sparse-dense matmul comes in Phase 4.
//    • All functions operate in single precision (float32).
//    • Zero external dependencies (no BLAS, no Eigen, no LAPACK).
//    • Clear, descriptive exceptions for every invalid input.
//
//  Internal implementation notes  (see ops.cpp for details):
//    The baseline uses the canonical three-loop order (i, k, j) which
//    iterates over B in column order and A in row order, keeping the
//    innermost accumulation in a register.  This is loop-order (i,k,j)
//    — also called the "row-panel × column-panel" traversal — and is a
//    natural starting point before cache-tiling and SIMD are applied.
//
//    Loop order justification:
//      Outer i   — walks output row  i of C and input  row  i of A
//      Middle k  — walks shared dimension; A[i][k] is row-sequential
//      Inner j   — walks output col  j of C and input  col  j of B
//      A[i][k] is reused across all j → fits in a register.
//      B[k][j] is accessed row-sequentially (cache-friendly for small K).
//
//  Future optimisation hooks (left as Phase 4+ work):
//    • L1/L2 cache tiling (block / micro-kernel decomposition)
//    • SIMD vectorisation via compiler auto-vec or SSE/AVX intrinsics
//    • Parallel outer loop with OpenMP  #pragma omp parallel for
//    • Sparse-dense GEMM (SpMM) for aggregation in GNN message passing
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

}  // namespace tinygnn
