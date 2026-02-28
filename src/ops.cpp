// ============================================================================
//  TinyGNN — Compute Kernels & Activations  (Phase 3–5 + Phase 8 parallelism)
//  src/ops.cpp
//
//  Phase 8 additions:
//    • OpenMP  — #pragma omp parallel for on outer (row) loops of matmul,
//                spmm, softmax, log_softmax, add_bias, and element-wise
//                activations.
//    • AVX2   — _mm256 FMA intrinsics on the inner feature-dimension loops
//                of matmul and spmm for 8-wide SIMD throughput.
// ============================================================================

#include "tinygnn/ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

// ── SIMD headers (Phase 8) ──────────────────────────────────────────────────
#ifdef __AVX2__
#include <immintrin.h>
#endif

// ── OpenMP (Phase 8) ────────────────────────────────────────────────────────
#ifdef _OPENMP
#include <omp.h>
#endif

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
    //  Phase 8: outer i-loop is parallelised with OpenMP.
    //  Inner j-loop uses AVX2 FMA intrinsics (8 floats per cycle).
    //
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const float a_ik = a[i * K + k];
            float* __restrict__ ci = c + i * N;
            const float* __restrict__ bk = b + k * N;

#ifdef __AVX2__
            // AVX2 path: 8-wide FMA
            const __m256 va = _mm256_set1_ps(a_ik);
            std::size_t j = 0;
            for (; j + 8 <= N; j += 8) {
                __m256 vc = _mm256_loadu_ps(ci + j);
                __m256 vb = _mm256_loadu_ps(bk + j);
                vc = _mm256_fmadd_ps(va, vb, vc);
                _mm256_storeu_ps(ci + j, vc);
            }
            // Scalar tail
            for (; j < N; ++j) {
                ci[j] += a_ik * bk[j];
            }
#else
            for (std::size_t j = 0; j < N; ++j) {
                ci[j] += a_ik * bk[j];
            }
#endif
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
    //  Phase 8: outer i-loop parallelised with OpenMP (schedule(dynamic)
    //  because row lengths vary widely in power-law graphs).
    //  Inner j-loop uses AVX2 FMA intrinsics.
    //
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < M; ++i) {
        const int32_t row_start = rp[i];
        const int32_t row_end   = rp[i + 1];
        float* __restrict__ crow = c + i * N;

        for (int32_t nz = row_start; nz < row_end; ++nz) {
            const auto k     = static_cast<std::size_t>(ci[nz]);
            const float a_val = av[nz];
            const float* __restrict__ brow = b + k * N;

#ifdef __AVX2__
            const __m256 va = _mm256_set1_ps(a_val);
            std::size_t j = 0;
            for (; j + 8 <= N; j += 8) {
                __m256 vc = _mm256_loadu_ps(crow + j);
                __m256 vb = _mm256_loadu_ps(brow + j);
                vc = _mm256_fmadd_ps(va, vb, vc);
                _mm256_storeu_ps(crow + j, vc);
            }
            for (; j < N; ++j) {
                crow[j] += a_val * brow[j];
            }
#else
            for (std::size_t j = 0; j < N; ++j) {
                crow[j] += a_val * brow[j];
            }
#endif
        }
    }

    return C;
}

// ============================================================================
//                  Phase 5 — Activations & Utilities
// ============================================================================

// ── Helper: validate that a tensor is Dense ─────────────────────────────────
static void require_dense(const Tensor& X, const char* func_name) {
    if (X.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            std::string(func_name) +
            ": tensor must be Dense (got SparseCSR). "
            "Activations operate on dense feature matrices only.");
    }
}

// ============================================================================
//  relu_inplace  — f(x) = max(0, x)
// ============================================================================
//
//  Single pass over all elements.  std::max(0, x) compiles to a single
//  MAXSS (scalar) or MAXPS (packed/SIMD) instruction on x86 with -O2,
//  avoiding any branch and allowing the loop to be auto-vectorised over
//  4 or 8 floats per iteration using SSE/AVX.
//
//  Used in virtually every GNN architecture:
//    GCN:        H' = ReLU(  D^{-½} A D^{-½} H W  )
//    GraphSAGE:  H' = ReLU(  W · CONCAT(h_v, AGG({h_u}))  )
//    GIN:        H' = ReLU(  MLP( (1+ε) h_v + Σ h_u )  )
//
// ============================================================================
void relu_inplace(Tensor& X) {
    require_dense(X, "relu_inplace");

    float* __restrict__ d = X.data().data();
    const std::size_t n = X.data().size();

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        d[i] = std::max(0.0f, d[i]);
    }
}

// ============================================================================
//  leaky_relu_inplace  — f(x) = x if x >= 0, else alpha * x
// ============================================================================
//
//  The default alpha=0.01 is the standard LeakyReLU.
//  GAT (Veličković et al. 2018) uses alpha=0.2 for attention coefficients.
//
//  Unlike ReLU, Leaky ReLU allows a small gradient for negative inputs,
//  preventing "dead neuron" problems in deep GNN stacks.
//
//  The ternary form (instead of if) enables the compiler to emit a
//  conditional-move (CMOV) — branchless, SIMD-friendly — since both
//  branches are cheap arithmetic with no function calls.
//
// ============================================================================
void leaky_relu_inplace(Tensor& X, float alpha) {
    require_dense(X, "leaky_relu_inplace");

    float* __restrict__ d = X.data().data();
    const std::size_t n = X.data().size();

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        d[i] = d[i] >= 0.0f ? d[i] : alpha * d[i];
    }
}

// ============================================================================
//  elu_inplace  — f(x) = x if x >= 0, else alpha * (exp(x) - 1)
// ============================================================================
//
//  ELU (Clevert et al. 2016) provides:
//    • Smooth transition through zero (continuous first derivative)
//    • Negative saturation at -alpha (pushes mean activations toward zero)
//    • No "dying neuron" problem
//
//  Used in EGNN (equivariant GNNs), PNA, and some message-passing designs.
//  exp() is only computed for negative elements (positive elements are cheap).
//
// ============================================================================
void elu_inplace(Tensor& X, float alpha) {
    require_dense(X, "elu_inplace");

    float* __restrict__ d = X.data().data();
    const std::size_t n = X.data().size();

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        if (d[i] < 0.0f) {
            d[i] = alpha * (std::exp(d[i]) - 1.0f);
        }
    }
}

// ============================================================================
//  sigmoid_inplace  — f(x) = 1 / (1 + exp(-x))
// ============================================================================
//
//  Numerically stable implementation:
//    x >= 0:  f(x) = 1 / (1 + exp(-x))           — standard form
//    x <  0:  f(x) = exp(x) / (1 + exp(x))        — avoids exp(large positive)
//
//  Both branches produce identical mathematical results but the split
//  prevents float overflow for extreme values.
//
//  GNN usage:
//    • GGNN gate:  z_v = σ(W_z h_v + U_z m_v)
//    • Link prediction:  P(edge) = σ(h_u · h_v)
//    • Binary node classification output
//
// ============================================================================
void sigmoid_inplace(Tensor& X) {
    require_dense(X, "sigmoid_inplace");

    float* __restrict__ d = X.data().data();
    const std::size_t n = X.data().size();

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        if (d[i] >= 0.0f) {
            d[i] = 1.0f / (1.0f + std::exp(-d[i]));
        } else {
            const float ex = std::exp(d[i]);
            d[i] = ex / (1.0f + ex);
        }
    }
}

// ============================================================================
//  tanh_inplace  — f(x) = tanh(x)
// ============================================================================
//
//  Uses std::tanh which is already numerically stable for all float inputs.
//  Output range: (-1, 1).
//
//  GNN usage:
//    • GGNN (Li et al. 2016): GRU cell state update
//    • MPNN (Gilmer et al. 2017): message function
//    • Some attention score clamping mechanisms
//
// ============================================================================
void tanh_inplace(Tensor& X) {
    require_dense(X, "tanh_inplace");

    float* __restrict__ d = X.data().data();
    const std::size_t n = X.data().size();

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        d[i] = std::tanh(d[i]);
    }
}

// ============================================================================
//  gelu_inplace  — f(x) = x * Φ(x)  (tanh approximation)
// ============================================================================
//
//  Approximation (Hendrycks & Gimpel 2016):
//    f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
//
//  This is the same approximation used by PyTorch's F.gelu(approximate='tanh')
//  and is the standard in transformer-based GNNs.
//
//  GNN usage:
//    • GPS (Rampášek et al. 2022): feed-forward layers
//    • Graphormer (Ying et al. 2021): transformer FFN blocks
//    • TokenGT (Kim et al. 2022): all MLP layers
//
// ============================================================================
void gelu_inplace(Tensor& X) {
    require_dense(X, "gelu_inplace");

    float* __restrict__ d = X.data().data();
    const std::size_t n = X.data().size();

    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float COEFF          = 0.044715f;

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        const float x = d[i];
        const float inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        d[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

// ============================================================================
//  softmax_inplace  — row-wise softmax  (numerically stable)
// ============================================================================
//
//  For each row i of X (M × N):
//    1. Find max_val = max_j X[i][j]           (prevent overflow)
//    2. Compute exp(X[i][j] - max_val)         (shifted exponentials)
//    3. Sum the exponentials → row_sum
//    4. Divide each element by row_sum          (normalize to probability)
//
//  3 passes per row: find max, compute exp+sum, divide.
//
//  GNN usage:
//    • GAT:    α_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))
//    • Multi-class node classification output (C classes → C-dim softmax)
//
//  Zero-row tensors (M=0) are valid no-ops.  Zero-column tensors are
//  rejected because softmax over an empty set is undefined.
//
// ============================================================================
void softmax_inplace(Tensor& X) {
    require_dense(X, "softmax_inplace");

    const std::size_t M = X.rows();
    const std::size_t N = X.cols();

    if (N == 0) {
        throw std::invalid_argument(
            "softmax_inplace: tensor has 0 columns — softmax over "
            "an empty set is undefined.");
    }

    if (M == 0) return;  // no rows → nothing to do

    float* __restrict__ d = X.data().data();

    // Phase 8: parallelize over rows — each row is independent
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < M; ++i) {
        float* row = d + i * N;

        // Pass 1: find row maximum (for numerical stability)
        float max_val = row[0];
        for (std::size_t j = 1; j < N; ++j) {
            if (row[j] > max_val) max_val = row[j];
        }

        // Pass 2: compute shifted exponentials and their sum
        float row_sum = 0.0f;
        for (std::size_t j = 0; j < N; ++j) {
            row[j] = std::exp(row[j] - max_val);
            row_sum += row[j];
        }

        // Pass 3: normalize
        const float inv_sum = 1.0f / row_sum;
        for (std::size_t j = 0; j < N; ++j) {
            row[j] *= inv_sum;
        }
    }
}

// ============================================================================
//  log_softmax_inplace  — row-wise log-softmax  (numerically stable)
// ============================================================================
//
//  For each row i of X (M × N):
//    1. Find max_val = max_j X[i][j]
//    2. Compute log_sum_exp = max_val + log(Σ_k exp(X[i][k] - max_val))
//    3. X[i][j] = X[i][j] - log_sum_exp
//
//  This is mathematically equivalent to log(softmax(X)) but avoids
//  computing softmax probabilities that may underflow to zero.
//
//  GNN usage:
//    • Node classification:  loss = NLL(log_softmax(scores), labels)
//    • Numerically stable log-probability output for graph-level tasks
//
// ============================================================================
void log_softmax_inplace(Tensor& X) {
    require_dense(X, "log_softmax_inplace");

    const std::size_t M = X.rows();
    const std::size_t N = X.cols();

    if (N == 0) {
        throw std::invalid_argument(
            "log_softmax_inplace: tensor has 0 columns — log-softmax over "
            "an empty set is undefined.");
    }

    if (M == 0) return;  // no rows → nothing to do

    float* __restrict__ d = X.data().data();

    // Phase 8: parallelize over rows — each row is independent
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < M; ++i) {
        float* row = d + i * N;

        // Pass 1: find row maximum
        float max_val = row[0];
        for (std::size_t j = 1; j < N; ++j) {
            if (row[j] > max_val) max_val = row[j];
        }

        // Pass 2: compute sum of shifted exponentials
        float sum_exp = 0.0f;
        for (std::size_t j = 0; j < N; ++j) {
            sum_exp += std::exp(row[j] - max_val);
        }

        // log_sum_exp = max_val + log(sum_exp)
        const float log_sum_exp = max_val + std::log(sum_exp);

        // Pass 3: subtract log_sum_exp from each element
        for (std::size_t j = 0; j < N; ++j) {
            row[j] -= log_sum_exp;
        }
    }
}

// ============================================================================
//  add_bias  — broadcasting bias addition  (in-place)
// ============================================================================
//
//  X[i][j] += bias[0][j]   for all rows i.
//
//  The bias must be shape (1 × N) where N == X.cols().  This broadcasts
//  the single bias row across all M rows of X, implementing the bias
//  term in a GNN linear layer:  H' = Adj × H × W + b
//
//  Complexity: O(M × N)  — one pass over all elements of X
//  Memory:     O(1) extra — modifies X in-place
//
// ============================================================================
void add_bias(Tensor& X, const Tensor& bias) {
    require_dense(X, "add_bias (X)");

    if (bias.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            "add_bias: bias tensor must be Dense (got SparseCSR).");
    }

    const std::size_t M = X.rows();
    const std::size_t N = X.cols();

    if (bias.rows() != 1 || bias.cols() != N) {
        throw std::invalid_argument(
            "add_bias: bias must be shape (1×" + std::to_string(N) +
            ") to match X columns, but got (" +
            std::to_string(bias.rows()) + "×" +
            std::to_string(bias.cols()) + ").");
    }

    float*       __restrict__ x = X.data().data();
    const float* __restrict__ b = bias.data().data();

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            x[i * N + j] += b[j];
        }
    }
}

}  // namespace tinygnn
