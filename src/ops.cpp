// ============================================================================
//  TinyGNN — Dense Compute Kernels  (Phase 3)
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

}  // namespace tinygnn
