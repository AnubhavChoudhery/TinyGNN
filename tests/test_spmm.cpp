// ============================================================================
//  TinyGNN — Sparse-Dense Compute Kernels Unit Tests  (Phase 4)
//  tests/test_spmm.cpp
//
//  Test categories:
//    1.  Correctness — 3×3 hand-calculated SpMM        (tests  1 –  3)
//    2.  Correctness — 4×4 weighted CSR                 (tests  4 –  5)
//    3.  Correctness — non-square shapes                (tests  6 –  8)
//    4.  Identity & zero properties                     (tests  9 – 12)
//    5.  GNN message-passing scenarios                  (tests 13 – 15)
//    6.  Equivalence with dense matmul                  (tests 16 – 17)
//    7.  Output tensor properties                       (tests 18 – 20)
//    8.  Stress / scale tests                           (tests 21 – 23)
//    9.  Error handling — wrong formats                 (tests 24 – 27)
//   10.  Error handling — dimension mismatch            (tests 28 – 30)
//   11.  Edge / degenerate cases                        (tests 31 – 35)
//
//  All sparse matrices are constructed with Tensor::sparse_csr().
//  All expected values are computed by hand or verified against
//  the dense matmul() path as a reference implementation.
// ============================================================================

#include "tinygnn/ops.hpp"
#include "tinygnn/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ── Minimal dependency-free test framework (same idiom as test_matmul.cpp) ──

static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define ASSERT_TRUE(cond)                                                      \
    do {                                                                       \
        ++g_tests_run;                                                         \
        if (!(cond)) {                                                         \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__            \
                      << " — ASSERT_TRUE(" #cond ")\n";                       \
            ++g_tests_failed;                                                  \
        } else { ++g_tests_passed; }                                           \
    } while (0)

#define ASSERT_EQ(a, b)                                                        \
    do {                                                                       \
        ++g_tests_run;                                                         \
        if ((a) != (b)) {                                                      \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__            \
                      << " — ASSERT_EQ(" #a ", " #b ") → "                    \
                      << (a) << " != " << (b) << "\n";                        \
            ++g_tests_failed;                                                  \
        } else { ++g_tests_passed; }                                           \
    } while (0)

#define ASSERT_FLOAT_EQ(a, b)                                                  \
    do {                                                                       \
        ++g_tests_run;                                                         \
        if (std::fabs(static_cast<double>(a) - static_cast<double>(b)) > 1e-4) {\
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__            \
                      << " — ASSERT_FLOAT_EQ(" #a ", " #b ") → "              \
                      << (a) << " != " << (b) << "\n";                        \
            ++g_tests_failed;                                                  \
        } else { ++g_tests_passed; }                                           \
    } while (0)

#define ASSERT_THROWS(expr, exc_type)                                          \
    do {                                                                       \
        ++g_tests_run;                                                         \
        bool caught_ = false;                                                  \
        try { expr; } catch (const exc_type&) { caught_ = true; }             \
        if (!caught_) {                                                        \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__            \
                      << " — ASSERT_THROWS(" #expr ", " #exc_type             \
                      ") did not throw\n";                                     \
            ++g_tests_failed;                                                  \
        } else { ++g_tests_passed; }                                           \
    } while (0)

#define ASSERT_THROWS_MSG(expr, exc_type, substr)                              \
    do {                                                                       \
        ++g_tests_run;                                                         \
        bool caught_ = false;                                                  \
        std::string msg_;                                                      \
        try { expr; } catch (const exc_type& e) {                              \
            caught_ = true; msg_ = e.what();                                   \
        }                                                                      \
        if (!caught_) {                                                        \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__            \
                      << " — ASSERT_THROWS_MSG did not throw\n";              \
            ++g_tests_failed;                                                  \
        } else if (msg_.find(substr) == std::string::npos) {                  \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__            \
                      << " — ASSERT_THROWS_MSG: \"" << substr                 \
                      << "\" not found in:\n    \"" << msg_ << "\"\n";        \
            ++g_tests_failed;                                                  \
        } else { ++g_tests_passed; }                                           \
    } while (0)

#define RUN_TEST(fn)                                                           \
    do {                                                                       \
        std::cout << "  Running " #fn "...\n";                                 \
        fn();                                                                  \
    } while (0)

using namespace tinygnn;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Build a (rows × cols) dense tensor from a braced initialiser list.
static Tensor make_dense(std::size_t rows, std::size_t cols,
                         std::vector<float> data) {
    return Tensor::dense(rows, cols, std::move(data));
}

/// Build a sparse CSR tensor.
static Tensor make_csr(std::size_t rows, std::size_t cols,
                       std::vector<int32_t> row_ptr,
                       std::vector<int32_t> col_ind,
                       std::vector<float> values) {
    return Tensor::sparse_csr(rows, cols, std::move(row_ptr),
                              std::move(col_ind), std::move(values));
}

/// Build a sparse identity matrix I_n as CSR.
static Tensor sparse_identity(std::size_t n) {
    std::vector<int32_t> rp(n + 1);
    std::vector<int32_t> ci(n);
    std::vector<float>   vals(n, 1.0f);
    for (std::size_t i = 0; i < n; ++i) {
        rp[i]  = static_cast<int32_t>(i);
        ci[i]  = static_cast<int32_t>(i);
    }
    rp[n] = static_cast<int32_t>(n);
    return Tensor::sparse_csr(n, n, std::move(rp), std::move(ci),
                              std::move(vals));
}

/// Build a sparse all-ones n×n matrix (every element = 1.0).
static Tensor sparse_all_ones(std::size_t n) {
    std::vector<int32_t> rp(n + 1);
    std::vector<int32_t> ci;
    std::vector<float>   vals;
    ci.reserve(n * n);
    vals.reserve(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        rp[i] = static_cast<int32_t>(i * n);
        for (std::size_t j = 0; j < n; ++j) {
            ci.push_back(static_cast<int32_t>(j));
            vals.push_back(1.0f);
        }
    }
    rp[n] = static_cast<int32_t>(n * n);
    return Tensor::sparse_csr(n, n, std::move(rp), std::move(ci),
                              std::move(vals));
}

/// Approximate equality check for two tensors.
static bool approx_equal(const Tensor& A, const Tensor& B,
                         float tol = 1e-4f) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
    for (std::size_t i = 0; i < A.data().size(); ++i) {
        if (std::fabs(A.data()[i] - B.data()[i]) > tol) return false;
    }
    return true;
}

// ============================================================================
//  1.  3×3 hand-calculated SpMM (spec requirement)
// ============================================================================
//
//  Sparse adjacency A (3 nodes, 5 edges):
//    node 0 → {0, 1}     (self-loop + edge to 1)
//    node 1 → {1}         (self-loop only)
//    node 2 → {0, 2}     (edges to 0 and self-loop)
//
//    A = [1 1 0]   row_ptr = [0, 2, 3, 5]
//        [0 1 0]   col_ind = [0, 1, 1, 0, 2]
//        [1 0 1]   values  = [1, 1, 1, 1, 1]
//
//  Dense features B (3 nodes × 2 features):
//    B = [1 2]
//        [3 4]
//        [5 6]
//
//  C = A × B:
//    C[0] = 1·[1,2] + 1·[3,4]         = [4,  6]
//    C[1] = 1·[3,4]                    = [3,  4]
//    C[2] = 1·[1,2] + 1·[5,6]         = [6,  8]
//
void test_3x3_hand_calculated() {
    auto A = make_csr(3, 3,
        {0, 2, 3, 5},       // row_ptr
        {0, 1, 1, 0, 2},    // col_ind
        {1, 1, 1, 1, 1});   // values (all 1.0)

    auto B = make_dense(3, 2, {1, 2,
                                3, 4,
                                5, 6});

    auto C = spmm(A, B);

    ASSERT_EQ(C.rows(), 3u);
    ASSERT_EQ(C.cols(), 2u);
    ASSERT_TRUE(C.format() == StorageFormat::Dense);

    // Row 0: [4, 6]
    ASSERT_FLOAT_EQ(C.data()[0], 4.0f);
    ASSERT_FLOAT_EQ(C.data()[1], 6.0f);
    // Row 1: [3, 4]
    ASSERT_FLOAT_EQ(C.data()[2], 3.0f);
    ASSERT_FLOAT_EQ(C.data()[3], 4.0f);
    // Row 2: [6, 8]
    ASSERT_FLOAT_EQ(C.data()[4], 6.0f);
    ASSERT_FLOAT_EQ(C.data()[5], 8.0f);
}

// ============================================================================
//  2.  Verify individual elements with named positions
// ============================================================================
void test_3x3_element_spot_checks() {
    // Same A, B as test 1
    auto A = make_csr(3, 3,
        {0, 2, 3, 5}, {0, 1, 1, 0, 2}, {1, 1, 1, 1, 1});
    auto B = make_dense(3, 2, {1, 2, 3, 4, 5, 6});
    auto C = spmm(A, B);

    // C[0][0] = 1*1 + 1*3 = 4
    ASSERT_FLOAT_EQ(C.at(0, 0), 4.0f);
    // C[0][1] = 1*2 + 1*4 = 6
    ASSERT_FLOAT_EQ(C.at(0, 1), 6.0f);
    // C[1][0] = 1*3 = 3
    ASSERT_FLOAT_EQ(C.at(1, 0), 3.0f);
    // C[2][1] = 1*2 + 1*6 = 8
    ASSERT_FLOAT_EQ(C.at(2, 1), 8.0f);
}

// ============================================================================
//  3.  3×3 sparse × 3×1 column vector (message-passing: aggregate to scalar)
// ============================================================================
void test_3x3_times_column_vector() {
    // Same adjacency as tests 1-2
    auto A = make_csr(3, 3,
        {0, 2, 3, 5}, {0, 1, 1, 0, 2}, {1, 1, 1, 1, 1});
    // Single feature per node
    auto B = make_dense(3, 1, {10, 20, 30});
    auto C = spmm(A, B);

    ASSERT_EQ(C.rows(), 3u);
    ASSERT_EQ(C.cols(), 1u);
    // C[0] = 1*10 + 1*20 = 30
    ASSERT_FLOAT_EQ(C.data()[0], 30.0f);
    // C[1] = 1*20 = 20
    ASSERT_FLOAT_EQ(C.data()[1], 20.0f);
    // C[2] = 1*10 + 1*30 = 40
    ASSERT_FLOAT_EQ(C.data()[2], 40.0f);
}

// ============================================================================
//  4.  4×4 hand-calculated with weighted edges
// ============================================================================
//
//  Sparse A (4×4, 6 nnz, weighted):
//    A = [2  0  0  0]   row_ptr = [0, 1, 3, 5, 7]
//        [0  3  1  0]   col_ind = [0, 1, 2, 0, 3, 1, 2]
//        [1  0  0  2]   values  = [2, 3, 1, 1, 2, 1, 1]
//        [0  1  1  0]
//
//  Dense B (4×3):
//    B = [1  0  2]
//        [0  1  0]
//        [3  0  1]
//        [0  2  0]
//
//  C = A × B (4×3):
//    C[0] = 2·[1,0,2]                     = [2, 0, 4]
//    C[1] = 3·[0,1,0] + 1·[3,0,1]         = [3, 3, 1]
//    C[2] = 1·[1,0,2] + 2·[0,2,0]         = [1, 4, 2]
//    C[3] = 1·[0,1,0] + 1·[3,0,1]         = [3, 1, 1]
//
void test_4x4_weighted_hand_calculated() {
    auto A = make_csr(4, 4,
        {0, 1, 3, 5, 7},
        {0, 1, 2, 0, 3, 1, 2},
        {2, 3, 1, 1, 2, 1, 1});

    auto B = make_dense(4, 3, {1, 0, 2,
                                0, 1, 0,
                                3, 0, 1,
                                0, 2, 0});
    auto C = spmm(A, B);

    ASSERT_EQ(C.rows(), 4u);
    ASSERT_EQ(C.cols(), 3u);

    // Row 0: [2, 0, 4]
    ASSERT_FLOAT_EQ(C.at(0, 0), 2.0f);
    ASSERT_FLOAT_EQ(C.at(0, 1), 0.0f);
    ASSERT_FLOAT_EQ(C.at(0, 2), 4.0f);
    // Row 1: [3, 3, 1]
    ASSERT_FLOAT_EQ(C.at(1, 0), 3.0f);
    ASSERT_FLOAT_EQ(C.at(1, 1), 3.0f);
    ASSERT_FLOAT_EQ(C.at(1, 2), 1.0f);
    // Row 2: [1, 4, 2]
    ASSERT_FLOAT_EQ(C.at(2, 0), 1.0f);
    ASSERT_FLOAT_EQ(C.at(2, 1), 4.0f);
    ASSERT_FLOAT_EQ(C.at(2, 2), 2.0f);
    // Row 3: [3, 1, 1]
    ASSERT_FLOAT_EQ(C.at(3, 0), 3.0f);
    ASSERT_FLOAT_EQ(C.at(3, 1), 1.0f);
    ASSERT_FLOAT_EQ(C.at(3, 2), 1.0f);
}

// ============================================================================
//  5.  4×4 weighted — verify specific element derivations
// ============================================================================
void test_4x4_weighted_derivations() {
    auto A = make_csr(4, 4,
        {0, 1, 3, 5, 7},
        {0, 1, 2, 0, 3, 1, 2},
        {2, 3, 1, 1, 2, 1, 1});
    auto B = make_dense(4, 3, {1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0});
    auto C = spmm(A, B);

    // C[0][0] = 2*1 = 2
    ASSERT_FLOAT_EQ(C.at(0, 0), 2.0f);
    // C[1][1] = 3*1 + 1*0 = 3
    ASSERT_FLOAT_EQ(C.at(1, 1), 3.0f);
    // C[2][1] = 1*0 + 2*2 = 4
    ASSERT_FLOAT_EQ(C.at(2, 1), 4.0f);
    // C[3][2] = 1*0 + 1*1 = 1
    ASSERT_FLOAT_EQ(C.at(3, 2), 1.0f);
}

// ============================================================================
//  6.  Non-square: 5×3 sparse × 3×4 dense
// ============================================================================
//
//  A (5×3, 7 nnz):
//    Row 0: col 0 = 1.0, col 2 = 2.0
//    Row 1: col 1 = 3.0
//    Row 2: (empty — disconnected node)
//    Row 3: col 0 = 1.0, col 1 = 1.0, col 2 = 1.0
//    Row 4: col 2 = 4.0
//
//    row_ptr = [0, 2, 3, 3, 6, 7]
//    col_ind = [0, 2, 1, 0, 1, 2, 2]
//    values  = [1, 2, 3, 1, 1, 1, 4]
//
//  B (3×4):
//    B = [1 0 0 1]
//        [0 2 0 0]
//        [0 0 3 1]
//
//  C = A × B (5×4):
//    C[0] = 1·[1,0,0,1] + 2·[0,0,3,1] = [1, 0, 6, 3]
//    C[1] = 3·[0,2,0,0]               = [0, 6, 0, 0]
//    C[2] = (empty)                    = [0, 0, 0, 0]
//    C[3] = 1·[1,0,0,1]+1·[0,2,0,0]+1·[0,0,3,1] = [1, 2, 3, 2]
//    C[4] = 4·[0,0,3,1]               = [0, 0, 12, 4]
//
void test_nonsquare_5x3_times_3x4() {
    auto A = make_csr(5, 3,
        {0, 2, 3, 3, 6, 7},
        {0, 2, 1, 0, 1, 2, 2},
        {1, 2, 3, 1, 1, 1, 4});

    auto B = make_dense(3, 4, {1, 0, 0, 1,
                                0, 2, 0, 0,
                                0, 0, 3, 1});
    auto C = spmm(A, B);

    ASSERT_EQ(C.rows(), 5u);
    ASSERT_EQ(C.cols(), 4u);

    // Row 0: [1, 0, 6, 3]
    ASSERT_FLOAT_EQ(C.at(0, 0), 1.0f);
    ASSERT_FLOAT_EQ(C.at(0, 1), 0.0f);
    ASSERT_FLOAT_EQ(C.at(0, 2), 6.0f);
    ASSERT_FLOAT_EQ(C.at(0, 3), 3.0f);
    // Row 1: [0, 6, 0, 0]
    ASSERT_FLOAT_EQ(C.at(1, 0), 0.0f);
    ASSERT_FLOAT_EQ(C.at(1, 1), 6.0f);
    ASSERT_FLOAT_EQ(C.at(1, 2), 0.0f);
    ASSERT_FLOAT_EQ(C.at(1, 3), 0.0f);
    // Row 2: [0, 0, 0, 0] (disconnected node)
    ASSERT_FLOAT_EQ(C.at(2, 0), 0.0f);
    ASSERT_FLOAT_EQ(C.at(2, 1), 0.0f);
    ASSERT_FLOAT_EQ(C.at(2, 2), 0.0f);
    ASSERT_FLOAT_EQ(C.at(2, 3), 0.0f);
    // Row 3: [1, 2, 3, 2]
    ASSERT_FLOAT_EQ(C.at(3, 0), 1.0f);
    ASSERT_FLOAT_EQ(C.at(3, 1), 2.0f);
    ASSERT_FLOAT_EQ(C.at(3, 2), 3.0f);
    ASSERT_FLOAT_EQ(C.at(3, 3), 2.0f);
    // Row 4: [0, 0, 12, 4]
    ASSERT_FLOAT_EQ(C.at(4, 0), 0.0f);
    ASSERT_FLOAT_EQ(C.at(4, 1), 0.0f);
    ASSERT_FLOAT_EQ(C.at(4, 2), 12.0f);
    ASSERT_FLOAT_EQ(C.at(4, 3), 4.0f);
}

// ============================================================================
//  7.  Non-square: 2×5 sparse × 5×1 (very wide A, single feature)
// ============================================================================
void test_nonsquare_2x5_times_5x1() {
    // Row 0 has cols {0,2,4}, Row 1 has cols {1,3}
    auto A = make_csr(2, 5,
        {0, 3, 5},
        {0, 2, 4, 1, 3},
        {1, 1, 1, 2, 2});

    auto B = make_dense(5, 1, {10, 20, 30, 40, 50});
    auto C = spmm(A, B);

    ASSERT_EQ(C.rows(), 2u);
    ASSERT_EQ(C.cols(), 1u);
    // C[0] = 1*10 + 1*30 + 1*50 = 90
    ASSERT_FLOAT_EQ(C.data()[0], 90.0f);
    // C[1] = 2*20 + 2*40 = 120
    ASSERT_FLOAT_EQ(C.data()[1], 120.0f);
}

// ============================================================================
//  8.  Non-square: 1×4 sparse × 4×3 (single-row sparse — one node)
// ============================================================================
void test_nonsquare_1x4_times_4x3() {
    // Single row: cols {1, 3}, vals {0.5, 2.0}
    auto A = make_csr(1, 4,
        {0, 2},
        {1, 3},
        {0.5f, 2.0f});

    auto B = make_dense(4, 3, {1, 2, 3,
                                4, 5, 6,
                                7, 8, 9,
                                10, 11, 12});
    auto C = spmm(A, B);

    ASSERT_EQ(C.rows(), 1u);
    ASSERT_EQ(C.cols(), 3u);
    // C[0] = 0.5*[4,5,6] + 2.0*[10,11,12] = [2+20, 2.5+22, 3+24] = [22, 24.5, 27]
    ASSERT_FLOAT_EQ(C.at(0, 0), 22.0f);
    ASSERT_FLOAT_EQ(C.at(0, 1), 24.5f);
    ASSERT_FLOAT_EQ(C.at(0, 2), 27.0f);
}

// ============================================================================
//  9.  Sparse identity × dense = dense  (spmm(I, B) == B)
// ============================================================================
void test_sparse_identity_right() {
    auto I = sparse_identity(4);
    auto B = make_dense(4, 3, {1, 2, 3,
                                4, 5, 6,
                                7, 8, 9,
                                10, 11, 12});
    auto C = spmm(I, B);

    ASSERT_EQ(C.rows(), 4u);
    ASSERT_EQ(C.cols(), 3u);
    ASSERT_TRUE(approx_equal(C, B));
}

// ============================================================================
//  10. Sparse identity (larger) — 64×64
// ============================================================================
void test_sparse_identity_64() {
    auto I = sparse_identity(64);
    // B = sequential values 64×8
    std::vector<float> bdata(64 * 8);
    std::iota(bdata.begin(), bdata.end(), 1.0f);
    auto B = make_dense(64, 8, std::move(bdata));
    auto C = spmm(I, B);
    ASSERT_TRUE(approx_equal(C, B));
}

// ============================================================================
//  11. Zero-nnz sparse → all-zero output
// ============================================================================
void test_zero_nnz_sparse() {
    // 3×3 sparse with no non-zeros
    auto A = make_csr(3, 3,
        {0, 0, 0, 0},
        {},
        {});

    auto B = make_dense(3, 2, {1, 2, 3, 4, 5, 6});
    auto C = spmm(A, B);

    ASSERT_EQ(C.rows(), 3u);
    ASSERT_EQ(C.cols(), 2u);
    for (std::size_t i = 0; i < 6; ++i) {
        ASSERT_FLOAT_EQ(C.data()[i], 0.0f);
    }
}

// ============================================================================
//  12. Sparse all-ones → each row of C = sum of all rows of B
// ============================================================================
void test_all_ones_sparse() {
    auto A = sparse_all_ones(3);
    auto B = make_dense(3, 2, {1, 2,
                                3, 4,
                                5, 6});
    auto C = spmm(A, B);

    // Each row of C should be [1+3+5, 2+4+6] = [9, 12]
    ASSERT_EQ(C.rows(), 3u);
    ASSERT_EQ(C.cols(), 2u);
    for (std::size_t i = 0; i < 3; ++i) {
        ASSERT_FLOAT_EQ(C.at(i, 0), 9.0f);
        ASSERT_FLOAT_EQ(C.at(i, 1), 12.0f);
    }
}

// ============================================================================
//  13. GNN message-passing: triangle graph
// ============================================================================
//  Three nodes forming a triangle with self-loops.
//  Adjacency (unweighted):
//    0 → {0, 1, 2}
//    1 → {0, 1, 2}
//    2 → {0, 1, 2}
//  This is a fully connected graph — the all-ones matrix.
//
//  Features: H = [[1, 0], [0, 1], [1, 1]]
//  H_agg = A × H:  every row = sum(H) = [2, 2]
//
void test_gnn_triangle_graph() {
    auto Adj = sparse_all_ones(3);
    auto H = make_dense(3, 2, {1, 0,
                                0, 1,
                                1, 1});
    auto H_agg = spmm(Adj, H);

    ASSERT_EQ(H_agg.rows(), 3u);
    ASSERT_EQ(H_agg.cols(), 2u);
    for (std::size_t i = 0; i < 3; ++i) {
        ASSERT_FLOAT_EQ(H_agg.at(i, 0), 2.0f);
        ASSERT_FLOAT_EQ(H_agg.at(i, 1), 2.0f);
    }
}

// ============================================================================
//  14. GNN message-passing: star graph (node 0 is hub)
// ============================================================================
//  5 nodes.  Node 0 connects to all (including self-loop).
//  Nodes 1–4 connect only to node 0 and themselves.
//
//    Adj:  0→{0,1,2,3,4}  1→{0,1}  2→{0,2}  3→{0,3}  4→{0,4}
//
//  Features H (5×2): each node's feature = [node_id, 10 - node_id]
//    H = [[0, 10], [1, 9], [2, 8], [3, 7], [4, 6]]
//
//  H_agg:
//    Row 0: sum of all features = [0+1+2+3+4, 10+9+8+7+6] = [10, 40]
//    Row 1: H[0] + H[1] = [0+1, 10+9] = [1, 19]
//    Row 2: H[0] + H[2] = [0+2, 10+8] = [2, 18]
//    Row 3: H[0] + H[3] = [0+3, 10+7] = [3, 17]
//    Row 4: H[0] + H[4] = [0+4, 10+6] = [4, 16]
//
void test_gnn_star_graph() {
    // Row 0: 5 entries (0-4), Row 1: 2 entries (0,1), ...
    auto Adj = make_csr(5, 5,
        {0, 5, 7, 9, 11, 13},
        {0, 1, 2, 3, 4,
         0, 1,
         0, 2,
         0, 3,
         0, 4},
        {1,1,1,1,1, 1,1, 1,1, 1,1, 1,1});

    auto H = make_dense(5, 2, {0, 10,
                                1,  9,
                                2,  8,
                                3,  7,
                                4,  6});
    auto H_agg = spmm(Adj, H);

    ASSERT_EQ(H_agg.rows(), 5u);
    ASSERT_EQ(H_agg.cols(), 2u);

    ASSERT_FLOAT_EQ(H_agg.at(0, 0), 10.0f);
    ASSERT_FLOAT_EQ(H_agg.at(0, 1), 40.0f);
    ASSERT_FLOAT_EQ(H_agg.at(1, 0), 1.0f);
    ASSERT_FLOAT_EQ(H_agg.at(1, 1), 19.0f);
    ASSERT_FLOAT_EQ(H_agg.at(2, 0), 2.0f);
    ASSERT_FLOAT_EQ(H_agg.at(2, 1), 18.0f);
    ASSERT_FLOAT_EQ(H_agg.at(3, 0), 3.0f);
    ASSERT_FLOAT_EQ(H_agg.at(3, 1), 17.0f);
    ASSERT_FLOAT_EQ(H_agg.at(4, 0), 4.0f);
    ASSERT_FLOAT_EQ(H_agg.at(4, 1), 16.0f);
}

// ============================================================================
//  15. GNN two-hop: spmm(Adj, spmm(Adj, H)) — chained SpMM
// ============================================================================
//  Path graph: 0 → 1 → 2 (with self-loops)
//    Adj:  0→{0,1}  1→{0,1,2}  2→{1,2}
//    H = [[1,0], [0,1], [0,0]]
//
//  1-hop: H1 = Adj × H
//    H1[0] = H[0] + H[1]   = [1, 1]
//    H1[1] = H[0]+H[1]+H[2] = [1, 1]
//    H1[2] = H[1] + H[2]   = [0, 1]
//
//  2-hop: H2 = Adj × H1
//    H2[0] = H1[0]+H1[1]       = [2, 2]
//    H2[1] = H1[0]+H1[1]+H1[2] = [2, 3]
//    H2[2] = H1[1]+H1[2]       = [1, 2]
//
void test_gnn_two_hop_path_graph() {
    auto Adj = make_csr(3, 3,
        {0, 2, 5, 7},
        {0, 1, 0, 1, 2, 1, 2},
        {1, 1, 1, 1, 1, 1, 1});

    auto H = make_dense(3, 2, {1, 0, 0, 1, 0, 0});

    // 1-hop
    auto H1 = spmm(Adj, H);
    ASSERT_FLOAT_EQ(H1.at(0, 0), 1.0f);
    ASSERT_FLOAT_EQ(H1.at(0, 1), 1.0f);
    ASSERT_FLOAT_EQ(H1.at(1, 0), 1.0f);
    ASSERT_FLOAT_EQ(H1.at(1, 1), 1.0f);
    ASSERT_FLOAT_EQ(H1.at(2, 0), 0.0f);
    ASSERT_FLOAT_EQ(H1.at(2, 1), 1.0f);

    // 2-hop
    auto H2 = spmm(Adj, H1);
    ASSERT_FLOAT_EQ(H2.at(0, 0), 2.0f);
    ASSERT_FLOAT_EQ(H2.at(0, 1), 2.0f);
    ASSERT_FLOAT_EQ(H2.at(1, 0), 2.0f);
    ASSERT_FLOAT_EQ(H2.at(1, 1), 3.0f);
    ASSERT_FLOAT_EQ(H2.at(2, 0), 1.0f);
    ASSERT_FLOAT_EQ(H2.at(2, 1), 2.0f);
}

// ============================================================================
//  16. Equivalence with dense matmul — small
// ============================================================================
//  Convert a small sparse matrix to dense form, multiply both ways,
//  and verify results match.
//
void test_equivalence_with_dense_matmul_small() {
    // Sparse A (3×3): A = [[1,0,2],[0,3,0],[4,0,5]]
    auto A_sparse = make_csr(3, 3,
        {0, 2, 3, 5},
        {0, 2, 1, 0, 2},
        {1, 2, 3, 4, 5});

    // Equivalent dense A
    auto A_dense = make_dense(3, 3, {1, 0, 2,
                                      0, 3, 0,
                                      4, 0, 5});

    auto B = make_dense(3, 4, {1, 2, 3, 4,
                                5, 6, 7, 8,
                                9, 10, 11, 12});

    auto C_spmm  = spmm(A_sparse, B);
    auto C_dense = matmul(A_dense, B);

    ASSERT_EQ(C_spmm.rows(), C_dense.rows());
    ASSERT_EQ(C_spmm.cols(), C_dense.cols());
    ASSERT_TRUE(approx_equal(C_spmm, C_dense));
}

// ============================================================================
//  17. Equivalence with dense matmul — medium (32×32 sparse, ~25% density)
// ============================================================================
void test_equivalence_with_dense_matmul_medium() {
    const std::size_t N = 32;
    const std::size_t F = 16;

    // Build a deterministic sparse matrix: A[i][(i+k) % N] = k+1
    // for k in {0, 1, 2} — 3 nnz per row = 3*32 = 96 nnz (~9.4% density)
    std::vector<int32_t> rp(N + 1, 0);
    std::vector<int32_t> ci;
    std::vector<float>   vals;
    std::vector<float>   dense_data(N * N, 0.0f);

    for (std::size_t i = 0; i < N; ++i) {
        rp[i] = static_cast<int32_t>(ci.size());

        // Collect columns for this row (must be sorted for valid CSR)
        std::vector<std::pair<int32_t, float>> entries;
        for (int k = 0; k < 3; ++k) {
            auto col = static_cast<int32_t>((i + static_cast<std::size_t>(k)) % N);
            float val = static_cast<float>(k + 1);
            entries.push_back({col, val});
        }
        std::sort(entries.begin(), entries.end());

        for (auto& [c, v] : entries) {
            ci.push_back(c);
            vals.push_back(v);
            dense_data[i * N + static_cast<std::size_t>(c)] = v;
        }
    }
    rp[N] = static_cast<int32_t>(ci.size());

    auto A_sparse = Tensor::sparse_csr(N, N, rp, ci, vals);
    auto A_dense  = Tensor::dense(N, N, dense_data);

    // B: sequential values
    std::vector<float> bdata(N * F);
    for (std::size_t i = 0; i < N * F; ++i) {
        bdata[i] = static_cast<float>(i % 17) * 0.1f;
    }
    auto B = Tensor::dense(N, F, bdata);

    auto C_spmm  = spmm(A_sparse, B);
    auto C_dense = matmul(A_dense, B);

    ASSERT_TRUE(approx_equal(C_spmm, C_dense, 1e-3f));
}

// ============================================================================
//  18. Output is Dense format
// ============================================================================
void test_output_is_dense() {
    auto A = make_csr(2, 2, {0, 1, 2}, {0, 1}, {1, 1});
    auto B = make_dense(2, 2, {1, 2, 3, 4});
    auto C = spmm(A, B);
    ASSERT_TRUE(C.format() == StorageFormat::Dense);
}

// ============================================================================
//  19. Output shape correctness
// ============================================================================
void test_output_shape() {
    // A is 5×3, B is 3×7 → C is 5×7
    auto A = make_csr(5, 3,
        {0, 1, 2, 3, 4, 5},
        {0, 1, 2, 0, 1},
        {1, 1, 1, 1, 1});
    auto B = make_dense(3, 7, std::vector<float>(3 * 7, 1.0f));
    auto C = spmm(A, B);

    ASSERT_EQ(C.rows(), 5u);
    ASSERT_EQ(C.cols(), 7u);
    ASSERT_EQ(C.data().size(), 35u);
}

// ============================================================================
//  20. Output memory footprint
// ============================================================================
void test_output_memory_footprint() {
    auto A = make_csr(10, 8,
        {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20},
        {0,1, 2,3, 4,5, 6,7, 0,1, 2,3, 4,5, 6,7, 0,1, 2,3},
        std::vector<float>(20, 1.0f));
    auto B = make_dense(8, 6, std::vector<float>(8 * 6, 1.0f));
    auto C = spmm(A, B);

    // C is 10×6 dense → 60 floats → 240 bytes
    ASSERT_EQ(C.memory_footprint_bytes(), 10u * 6u * sizeof(float));
}

// ============================================================================
//  21. Stress: sparse identity 512×512 × dense 512×64
// ============================================================================
void test_stress_sparse_identity_512() {
    auto I = sparse_identity(512);
    std::vector<float> bdata(512 * 64);
    for (std::size_t i = 0; i < bdata.size(); ++i)
        bdata[i] = static_cast<float>(i % 100) * 0.01f;
    auto B = make_dense(512, 64, std::move(bdata));
    auto C = spmm(I, B);
    ASSERT_TRUE(approx_equal(C, B));
}

// ============================================================================
//  22. Stress: Cora-scale — 2708 nodes, ~5% sparse adj × 1433 features
// ============================================================================
//  This simulates a real GNN workload (Cora dataset dimensions).
//  We verify spmm(sparse_A, B) matches matmul(dense_A, B).
//
void test_stress_cora_scale_equivalence() {
    const std::size_t N = 2708;
    const std::size_t F = 32;   // Reduced features for speed

    // Build a deterministic sparse adjacency: each node connects to
    // itself and the next 2 nodes (circular), giving 3*N = 8124 nnz
    std::vector<int32_t> rp(N + 1);
    std::vector<int32_t> ci;
    std::vector<float>   vals;
    std::vector<float>   dense_data(N * N, 0.0f);

    for (std::size_t i = 0; i < N; ++i) {
        rp[i] = static_cast<int32_t>(ci.size());

        std::vector<std::pair<int32_t, float>> entries;
        for (int k = 0; k < 3; ++k) {
            auto col = static_cast<int32_t>((i + static_cast<std::size_t>(k)) % N);
            entries.push_back({col, 1.0f});
        }
        std::sort(entries.begin(), entries.end());

        for (auto& [c, v] : entries) {
            ci.push_back(c);
            vals.push_back(v);
            dense_data[i * N + static_cast<std::size_t>(c)] = v;
        }
    }
    rp[N] = static_cast<int32_t>(ci.size());

    auto A_sparse = Tensor::sparse_csr(N, N, rp, ci, vals);
    auto A_dense  = Tensor::dense(N, N, dense_data);

    // Feature vector: deterministic
    std::vector<float> bdata(N * F);
    for (std::size_t i = 0; i < N * F; ++i)
        bdata[i] = static_cast<float>((i * 7 + 3) % 101) * 0.01f;
    auto B = Tensor::dense(N, F, bdata);

    auto C_spmm  = spmm(A_sparse, B);
    auto C_dense = matmul(A_dense,  B);

    ASSERT_EQ(C_spmm.rows(), N);
    ASSERT_EQ(C_spmm.cols(), F);
    ASSERT_TRUE(approx_equal(C_spmm, C_dense, 1e-2f));
}

// ============================================================================
//  23. Stress: 1024×1024 × 1024×128 (higher nnz per row)
// ============================================================================
void test_stress_1024x128() {
    const std::size_t N = 1024;
    const std::size_t F = 128;
    const int nnz_per_row = 5;

    std::vector<int32_t> rp(N + 1);
    std::vector<int32_t> ci;
    std::vector<float>   vals;

    for (std::size_t i = 0; i < N; ++i) {
        rp[i] = static_cast<int32_t>(ci.size());
        std::vector<int32_t> cols;
        for (int k = 0; k < nnz_per_row; ++k) {
            cols.push_back(static_cast<int32_t>((i + static_cast<std::size_t>(k * 7)) % N));
        }
        std::sort(cols.begin(), cols.end());
        // Remove duplicates
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        for (auto c : cols) {
            ci.push_back(c);
            vals.push_back(1.0f);
        }
    }
    rp[N] = static_cast<int32_t>(ci.size());

    auto A = Tensor::sparse_csr(N, N, rp, ci, vals);

    std::vector<float> bdata(N * F, 1.0f);
    auto B = Tensor::dense(N, F, bdata);
    auto C = spmm(A, B);

    // With all-ones B, each row of C should have value = (unique nnz in that row)
    // for every column. Verify a spot:
    ASSERT_EQ(C.rows(), N);
    ASSERT_EQ(C.cols(), F);

    // Row 0 has nnz_per_row unique entries (0, 7, 14, 21, 28)
    // Each column of C[0] should be nnz_per_row * 1.0 = 5.0
    for (std::size_t j = 0; j < F; ++j) {
        ASSERT_FLOAT_EQ(C.at(0, j), static_cast<float>(nnz_per_row));
    }
}

// ============================================================================
//  24. Error: Dense A → should throw (use matmul instead)
// ============================================================================
void test_error_dense_A() {
    auto A = make_dense(3, 3, {1,0,0, 0,1,0, 0,0,1});
    auto B = make_dense(3, 2, {1,2, 3,4, 5,6});
    ASSERT_THROWS(spmm(A, B), std::invalid_argument);
}

// ============================================================================
//  25. Error: Dense A — check error message
// ============================================================================
void test_error_dense_A_message() {
    auto A = make_dense(3, 3, {1,0,0, 0,1,0, 0,0,1});
    auto B = make_dense(3, 2, {1,2, 3,4, 5,6});
    ASSERT_THROWS_MSG(spmm(A, B), std::invalid_argument, "SparseCSR");
}

// ============================================================================
//  26. Error: Sparse B → should throw
// ============================================================================
void test_error_sparse_B() {
    auto A = make_csr(3, 3, {0,1,2,3}, {0,1,2}, {1,1,1});
    auto B = make_csr(3, 3, {0,1,2,3}, {0,1,2}, {1,1,1});
    ASSERT_THROWS(spmm(A, B), std::invalid_argument);
}

// ============================================================================
//  27. Error: Sparse B — check error message
// ============================================================================
void test_error_sparse_B_message() {
    auto A = make_csr(3, 3, {0,1,2,3}, {0,1,2}, {1,1,1});
    auto B = make_csr(3, 3, {0,1,2,3}, {0,1,2}, {1,1,1});
    ASSERT_THROWS_MSG(spmm(A, B), std::invalid_argument, "Dense");
}

// ============================================================================
//  28. Error: Dimension mismatch — A.cols() != B.rows()
// ============================================================================
void test_error_dimension_mismatch() {
    auto A = make_csr(3, 4, {0,1,2,3}, {0,1,2}, {1,1,1});
    auto B = make_dense(3, 2, {1,2, 3,4, 5,6});
    ASSERT_THROWS(spmm(A, B), std::invalid_argument);
}

// ============================================================================
//  29. Error: Dimension mismatch — check error message content
// ============================================================================
void test_error_dimension_mismatch_message() {
    auto A = make_csr(3, 5, {0,1,2,3}, {0,1,2}, {1,1,1});
    auto B = make_dense(3, 2, {1,2, 3,4, 5,6});
    ASSERT_THROWS_MSG(spmm(A, B), std::invalid_argument, "dimension mismatch");
}

// ============================================================================
//  30. Error: Various dimension mismatches
// ============================================================================
void test_error_dimension_mismatch_various() {
    // A is 4×2 sparse, B is 3×2 dense → mismatch (2 != 3)
    auto A = make_csr(4, 2, {0,1,1,2,2}, {0, 1}, {1, 1});
    auto B = make_dense(3, 2, {1,2, 3,4, 5,6});
    ASSERT_THROWS(spmm(A, B), std::invalid_argument);

    // A is 1×10 sparse, B is 5×1 dense → mismatch (10 != 5)
    auto A2 = make_csr(1, 10, {0, 0}, {}, {});
    auto B2 = make_dense(5, 1, {1,2,3,4,5});
    ASSERT_THROWS(spmm(A2, B2), std::invalid_argument);
}

// ============================================================================
//  31. Edge case: single non-zero element
// ============================================================================
void test_single_nonzero() {
    // 3×3 sparse with only A[1][2] = 7.0
    auto A = make_csr(3, 3,
        {0, 0, 1, 1},
        {2},
        {7.0f});

    auto B = make_dense(3, 2, {1, 2, 3, 4, 5, 6});
    auto C = spmm(A, B);

    // Only C[1] should be non-zero: 7.0 * B[2] = 7.0 * [5, 6] = [35, 42]
    ASSERT_FLOAT_EQ(C.at(0, 0), 0.0f);
    ASSERT_FLOAT_EQ(C.at(0, 1), 0.0f);
    ASSERT_FLOAT_EQ(C.at(1, 0), 35.0f);
    ASSERT_FLOAT_EQ(C.at(1, 1), 42.0f);
    ASSERT_FLOAT_EQ(C.at(2, 0), 0.0f);
    ASSERT_FLOAT_EQ(C.at(2, 1), 0.0f);
}

// ============================================================================
//  32. Edge case: row with many empty rows (disconnected nodes)
// ============================================================================
void test_empty_rows_disconnected_nodes() {
    // 5×5 sparse: only row 2 has entries (cols 0, 4), rest are empty
    auto A = make_csr(5, 5,
        {0, 0, 0, 2, 2, 2},
        {0, 4},
        {1.0f, 1.0f});

    auto B = make_dense(5, 3, {1,2,3,
                                4,5,6,
                                7,8,9,
                                10,11,12,
                                13,14,15});
    auto C = spmm(A, B);

    // Rows 0,1,3,4 should be zero
    for (std::size_t i : {0u, 1u, 3u, 4u}) {
        for (std::size_t j = 0; j < 3; ++j) {
            ASSERT_FLOAT_EQ(C.at(i, j), 0.0f);
        }
    }
    // Row 2: B[0] + B[4] = [1+13, 2+14, 3+15] = [14, 16, 18]
    ASSERT_FLOAT_EQ(C.at(2, 0), 14.0f);
    ASSERT_FLOAT_EQ(C.at(2, 1), 16.0f);
    ASSERT_FLOAT_EQ(C.at(2, 2), 18.0f);
}

// ============================================================================
//  33. Edge case: self-loops only (diagonal sparse matrix)
// ============================================================================
void test_self_loops_only() {
    // Diagonal: A[i][i] = (i+1) for i = 0..3
    auto A = make_csr(4, 4,
        {0, 1, 2, 3, 4},
        {0, 1, 2, 3},
        {1.0f, 2.0f, 3.0f, 4.0f});

    auto B = make_dense(4, 2, {10, 20,
                                30, 40,
                                50, 60,
                                70, 80});
    auto C = spmm(A, B);

    // C[i] = A[i][i] * B[i]
    ASSERT_FLOAT_EQ(C.at(0, 0), 10.0f);    // 1*10
    ASSERT_FLOAT_EQ(C.at(0, 1), 20.0f);    // 1*20
    ASSERT_FLOAT_EQ(C.at(1, 0), 60.0f);    // 2*30
    ASSERT_FLOAT_EQ(C.at(1, 1), 80.0f);    // 2*40
    ASSERT_FLOAT_EQ(C.at(2, 0), 150.0f);   // 3*50
    ASSERT_FLOAT_EQ(C.at(2, 1), 180.0f);   // 3*60
    ASSERT_FLOAT_EQ(C.at(3, 0), 280.0f);   // 4*70
    ASSERT_FLOAT_EQ(C.at(3, 1), 320.0f);   // 4*80
}

// ============================================================================
//  34. Edge case: fractional / negative weights
// ============================================================================
void test_fractional_negative_weights() {
    // A (2×3): row 0 = {col0: 0.5, col2: -1.5}, row 1 = {col1: 2.5}
    auto A = make_csr(2, 3,
        {0, 2, 3},
        {0, 2, 1},
        {0.5f, -1.5f, 2.5f});

    auto B = make_dense(3, 2, {4, 6,
                                2, 8,
                                10, 12});
    auto C = spmm(A, B);

    // C[0] = 0.5*[4,6] + (-1.5)*[10,12] = [2-15, 3-18] = [-13, -15]
    ASSERT_FLOAT_EQ(C.at(0, 0), -13.0f);
    ASSERT_FLOAT_EQ(C.at(0, 1), -15.0f);
    // C[1] = 2.5*[2,8] = [5, 20]
    ASSERT_FLOAT_EQ(C.at(1, 0), 5.0f);
    ASSERT_FLOAT_EQ(C.at(1, 1), 20.0f);
}

// ============================================================================
//  35. Edge case: 1×1 scalar-like SpMM
// ============================================================================
void test_1x1_scalar() {
    auto A = make_csr(1, 1, {0, 1}, {0}, {3.0f});
    auto B = make_dense(1, 1, {7.0f});
    auto C = spmm(A, B);

    ASSERT_EQ(C.rows(), 1u);
    ASSERT_EQ(C.cols(), 1u);
    ASSERT_FLOAT_EQ(C.data()[0], 21.0f);  // 3 * 7
}

// ============================================================================
//  main — run all tests
// ============================================================================
int main() {
    std::cout << "\n"
        "+=================================================================+\n"
        "|   TinyGNN — SpMM Unit Tests (Phase 4)                          |\n"
        "|   Testing: spmm(A, B) = C = A × B  (SparseCSR × Dense)        |\n"
        "+=================================================================+\n\n";

    std::cout << "── 1. 3×3 Hand-Calculated SpMM (spec required) ────────────────\n";
    RUN_TEST(test_3x3_hand_calculated);
    RUN_TEST(test_3x3_element_spot_checks);
    RUN_TEST(test_3x3_times_column_vector);

    std::cout << "\n── 2. 4×4 Weighted CSR ──────────────────────────────────────\n";
    RUN_TEST(test_4x4_weighted_hand_calculated);
    RUN_TEST(test_4x4_weighted_derivations);

    std::cout << "\n── 3. Non-Square Shapes ─────────────────────────────────────\n";
    RUN_TEST(test_nonsquare_5x3_times_3x4);
    RUN_TEST(test_nonsquare_2x5_times_5x1);
    RUN_TEST(test_nonsquare_1x4_times_4x3);

    std::cout << "\n── 4. Identity & Zero Properties ────────────────────────────\n";
    RUN_TEST(test_sparse_identity_right);
    RUN_TEST(test_sparse_identity_64);
    RUN_TEST(test_zero_nnz_sparse);
    RUN_TEST(test_all_ones_sparse);

    std::cout << "\n── 5. GNN Message-Passing Scenarios ─────────────────────────\n";
    RUN_TEST(test_gnn_triangle_graph);
    RUN_TEST(test_gnn_star_graph);
    RUN_TEST(test_gnn_two_hop_path_graph);

    std::cout << "\n── 6. Equivalence with Dense matmul ─────────────────────────\n";
    RUN_TEST(test_equivalence_with_dense_matmul_small);
    RUN_TEST(test_equivalence_with_dense_matmul_medium);

    std::cout << "\n── 7. Output Tensor Properties ──────────────────────────────\n";
    RUN_TEST(test_output_is_dense);
    RUN_TEST(test_output_shape);
    RUN_TEST(test_output_memory_footprint);

    std::cout << "\n── 8. Stress / Scale Tests ──────────────────────────────────\n";
    RUN_TEST(test_stress_sparse_identity_512);
    RUN_TEST(test_stress_cora_scale_equivalence);
    RUN_TEST(test_stress_1024x128);

    std::cout << "\n── 9. Error Handling — Wrong Formats ────────────────────────\n";
    RUN_TEST(test_error_dense_A);
    RUN_TEST(test_error_dense_A_message);
    RUN_TEST(test_error_sparse_B);
    RUN_TEST(test_error_sparse_B_message);

    std::cout << "\n── 10. Error Handling — Dimension Mismatch ──────────────────\n";
    RUN_TEST(test_error_dimension_mismatch);
    RUN_TEST(test_error_dimension_mismatch_message);
    RUN_TEST(test_error_dimension_mismatch_various);

    std::cout << "\n── 11. Edge / Degenerate Cases ──────────────────────────────\n";
    RUN_TEST(test_single_nonzero);
    RUN_TEST(test_empty_rows_disconnected_nodes);
    RUN_TEST(test_self_loops_only);
    RUN_TEST(test_fractional_negative_weights);
    RUN_TEST(test_1x1_scalar);

    // ── Summary ──────────────────────────────────────────────────────────────
    std::cout << "\n=================================================================\n";
    std::cout << "  Total : " << g_tests_run    << "\n";
    std::cout << "  Passed: " << g_tests_passed << "\n";
    std::cout << "  Failed: " << g_tests_failed << "\n";
    std::cout << "=================================================================\n\n";

    return g_tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
