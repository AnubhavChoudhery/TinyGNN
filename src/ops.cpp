// ============================================================================
//  TinyGNN — Compute Kernels  (Phase 3 + Phase 4)
//  src/ops.cpp
// ============================================================================

#include "tinygnn/ops.hpp"

#include <stdexcept>
#include <string>

namespace tinygnn {

// ============================================================================
//  matmul  — C = A × B  (dense GEMM, float32, row-major storage)
// ============================================================================
//
//  Algorithm: baseline triple-loop GEMM in (i, k, j) order.
//
//  Memory access pattern (row-major layout):
//    A[i][k]  →  A.data()[i * K + k]   — walks A row-by-row (sequential)
//    B[k][j]  →  B.data()[k * N + j]   — walks B row-by-row (sequential)
//    C[i][j]  →  C.data()[i * N + j]   — written row-by-row (sequential)
//
//  The key optimisation compared to the naïve (i, j, k) order is hoisting
//  A[i][k] out of the inner j-loop into a scalar register (`a_ik`).
//  This eliminates one memory load per inner iteration and lets the compiler
//  vectorise the inner j-loop over B's row k more aggressively.
//
//  Performance characteristics:
//    • Complexity : O(M × K × N)
//    • Flops      : 2 × M × K × N  (1 multiply + 1 add per element of A×B)
//    • Read bytes : M×K×4 + K×N×4  (A and B, each read once per outer loop)
//    • Write bytes: M×N×4           (C written once)
//
//  Cache behaviour (without tiling):
//    For small matrices that fit in L1/L2 cache this is near-optimal.
//    For large matrices (L > sqrt(cache_size / sizeof(float))) the inner
//    k-loop will cause B to be evicted between outer i iterations.
//    Phase 4 will add L1 cache blocking (tile sizes Mc × Kc × Nc).
//
// ============================================================================
Tensor matmul(const Tensor& A, const Tensor& B) {

    // ── Precondition 1: both tensors must be Dense ───────────────────────────
    if (A.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            "matmul: tensor A must be Dense (got SparseCSR). "
            "Sparse-dense matmul is planned for Phase 4 (SpMM kernel).");
    }
    if (B.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            "matmul: tensor B must be Dense (got SparseCSR). "
            "Sparse-dense matmul is planned for Phase 4 (SpMM kernel).");
    }

    const std::size_t M = A.rows();
    const std::size_t K = A.cols();
    const std::size_t K2 = B.rows();
    const std::size_t N = B.cols();

    // ── Precondition 2: inner dimensions must agree ──────────────────────────
    if (K != K2) {
        throw std::invalid_argument(
            "matmul: dimension mismatch — A is (" +
            std::to_string(M) + "×" + std::to_string(K) +
            ") but B is (" +
            std::to_string(K2) + "×" + std::to_string(N) +
            "). Required: A.cols() == B.rows() (both = " +
            std::to_string(K) + " vs " + std::to_string(K2) + ").");
    }

    // ── Precondition 3: no zero-dimension operands ───────────────────────────
    //  A 0-row or 0-col multiply is mathematically defined but almost always
    //  indicates a caller bug in a GNN context.  We allow it and return an
    //  empty tensor (rows * cols == 0 is valid Dense construction).

    // ── Allocate output C  (M × N, zero-initialised) ────────────────────────
    Tensor C = Tensor::dense(M, N);   // data_ filled with 0.0f

    // Direct pointer access avoids repeated bounds-checked vector::operator[]
    // overhead inside the hot triple loop.
    const float* __restrict__ a = A.data().data();
    const float* __restrict__ b = B.data().data();
    float*       __restrict__ c = C.data().data();

    // ── Core GEMM: loop order (i, k, j) ─────────────────────────────────────
    //
    //  Outer loop  i ∈ [0, M):   one row of A → one row of C
    //  Middle loop k ∈ [0, K):   shared dimension
    //    Load A[i][k] once into scalar register a_ik.
    //    Inner loop  j ∈ [0, N):   one row of B, one element of C
    //      C[i][j] += A[i][k] * B[k][j]
    //
    //  The compiler can auto-vectorise the inner j-loop because:
    //    • a_ik is a loop-invariant scalar
    //    • b[k*N + j] and c[i*N + j] are contiguous in memory
    //    • __restrict__ informs the compiler there is no aliasing
    //
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const float a_ik = a[i * K + k];   // hoist A[i][k] into register

            // Inner j-loop: compiler-vectorisable with -O2 / -O3
            for (std::size_t j = 0; j < N; ++j) {
                c[i * N + j] += a_ik * b[k * N + j];
            }
        }
    }

    return C;
}

// ============================================================================
//  spmm  — C = A × B  (sparse-dense, CSR-SpMM, float32)
// ============================================================================
//
//  Algorithm: CSR row-wise SpMM
//
//  For each row i of the sparse matrix A:
//    Walk the non-zeros in A[i, :] using row_ptr[i]..row_ptr[i+1].
//    For each non-zero at column k with value a_val:
//      Accumulate a_val * B[k, :] into C[i, :].
//
//  Memory access pattern:
//    A.row_ptr  — sequential read, size (M+1)
//    A.col_ind  — sequential read within each row segment
//    A.data     — sequential read, aligned with col_ind
//    B[k][j]    — accessed row-by-row for each neighbor k; contiguous in j
//    C[i][j]    — written row-by-row; contiguous in j
//
//  The inner j-loop is the same streaming pattern as in dense GEMM:
//    a_val is loop-invariant (hoisted into a register), B[k][j] and C[i][j]
//    are contiguous → compiler can auto-vectorise with -O2.
//
//  Performance characteristics:
//    • Complexity : O(nnz × N)  where nnz = number of non-zeros in A
//    • Flops      : 2 × nnz × N  (1 multiply + 1 add per non-zero per feature)
//    • Read bytes : nnz×4 (values) + nnz×4 (col_ind) + (M+1)×4 (row_ptr)
//                   + nnz×N×4 (rows of B, with reuse across rows)
//    • Write bytes: M×N×4  (output C, written once)
//
//  This is the heart of GNN message-passing:
//    H_agg = Adj × H
//    For each node i, H_agg[i] = Σ over neighbors j of A[i][j] * H[j]
//
// ============================================================================
Tensor spmm(const Tensor& A, const Tensor& B) {

    // ── Precondition 1: A must be SparseCSR ──────────────────────────────────
    if (A.format() != StorageFormat::SparseCSR) {
        throw std::invalid_argument(
            "spmm: tensor A must be SparseCSR (got Dense). "
            "For Dense × Dense, use matmul() instead.");
    }

    // ── Precondition 2: B must be Dense ──────────────────────────────────────
    if (B.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            "spmm: tensor B must be Dense (got SparseCSR). "
            "Sparse × Sparse multiplication is not supported.");
    }

    const std::size_t M  = A.rows();   // number of nodes (output rows)
    const std::size_t K  = A.cols();   // shared dimension
    const std::size_t K2 = B.rows();
    const std::size_t N  = B.cols();   // feature dimension (output cols)

    // ── Precondition 3: inner dimensions must agree ──────────────────────────
    if (K != K2) {
        throw std::invalid_argument(
            "spmm: dimension mismatch — A is (" +
            std::to_string(M) + "×" + std::to_string(K) +
            ", SparseCSR) but B is (" +
            std::to_string(K2) + "×" + std::to_string(N) +
            ", Dense). Required: A.cols() == B.rows() (" +
            std::to_string(K) + " vs " + std::to_string(K2) + ").");
    }

    // ── Allocate output C  (M × N, zero-initialised) ────────────────────────
    Tensor C = Tensor::dense(M, N);   // data_ filled with 0.0f

    // Early exit for degenerate cases (M=0 or N=0 or nnz=0)
    if (M == 0 || N == 0 || A.nnz() == 0) {
        return C;
    }

    // Direct pointer access for the hot loop
    const int32_t* __restrict__ rp  = A.row_ptr().data();
    const int32_t* __restrict__ ci  = A.col_ind().data();
    const float*   __restrict__ av  = A.data().data();
    const float*   __restrict__ b   = B.data().data();
    float*         __restrict__ c   = C.data().data();

    // ── Core CSR-SpMM ────────────────────────────────────────────────────────
    //
    //  Outer loop  i ∈ [0, M):      one row of A → one row of C (one node)
    //  Middle loop nz ∈ [rp[i], rp[i+1]):  non-zeros in row i (neighbors)
    //    k    = ci[nz]             column index (neighbor node ID)
    //    a_val = av[nz]            edge weight (1.0 for unweighted graphs)
    //    Inner loop  j ∈ [0, N):   feature dimension
    //      C[i][j] += a_val * B[k][j]
    //
    //  The inner j-loop is vectorisable: a_val is a scalar constant,
    //  b[k*N + j] and c[i*N + j] are contiguous in memory.
    //
    for (std::size_t i = 0; i < M; ++i) {
        const int32_t row_start = rp[i];
        const int32_t row_end   = rp[i + 1];

        for (int32_t nz = row_start; nz < row_end; ++nz) {
            const auto k     = static_cast<std::size_t>(ci[nz]);
            const float a_val = av[nz];

            // Accumulate a_val * B[k, :] into C[i, :]
            // Compiler can auto-vectorise this with -O2 / -O3
            for (std::size_t j = 0; j < N; ++j) {
                c[i * N + j] += a_val * b[k * N + j];
            }
        }
    }

    return C;
}

}  // namespace tinygnn
