// ============================================================================
//  TinyGNN — Dense Compute Kernels Unit Tests  (Phase 3)
//  tests/test_matmul.cpp
//
//  Test categories:
//    1.  Correctness — 4×4 hardcoded result         (tests  1 –  3)
//    2.  Correctness — non-square shapes             (tests  4 –  6)
//    3.  Correctness — identity & zero properties    (tests  7 – 10)
//    4.  Correctness — GNN feature transform         (test  11)
//    5.  Algebraic properties                        (tests 12 – 14)
//    6.  Output tensor properties                    (tests 15 – 17)
//    7.  Stress / scale tests                        (tests 18 – 20)
//    8.  Error handling — dimension mismatch         (tests 21 – 25)
//    9.  Error handling — sparse inputs              (tests 26 – 27)
//   10.  Edge / degenerate cases                     (tests 28 – 30)
// ============================================================================

#include "tinygnn/ops.hpp"
#include "tinygnn/tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ── Minimal dependency-free test framework (same idiom as test_tensor.cpp) ──

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
static Tensor make(std::size_t rows, std::size_t cols,
                   std::vector<float> data) {
    return Tensor::dense(rows, cols, std::move(data));
}

/// Build a square identity matrix of size n.
static Tensor identity(std::size_t n) {
    Tensor I = Tensor::dense(n, n);
    for (std::size_t i = 0; i < n; ++i)
        I.data()[i * n + i] = 1.0f;
    return I;
}

/// Build a (rows × cols) dense tensor filled with a constant value.
static Tensor filled(std::size_t rows, std::size_t cols, float v) {
    return Tensor::dense(rows, cols,
                         std::vector<float>(rows * cols, v));
}

/// Check that two tensors have the same shape and all elements are within tol.
static bool approx_equal(const Tensor& A, const Tensor& B,
                         float tol = 1e-4f) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
    for (std::size_t i = 0; i < A.data().size(); ++i) {
        if (std::fabs(A.data()[i] - B.data()[i]) > tol) return false;
    }
    return true;
}

// ============================================================================
//  1.  4×4 hardcoded result  (required by spec)
// ============================================================================
//
//  A = reshape([1 .. 16], 4, 4)  B = reshape([17 .. 32], 4, 4)
//
//  Expected C (derived by hand and verified below):
//    C[0] = [250, 260, 270, 280]
//    C[1] = [618, 644, 670, 696]
//    C[2] = [986, 1028, 1070, 1112]
//    C[3] = [1354, 1412, 1470, 1528]
//
//  Derivation of C[0][0]:
//    1·17 + 2·21 + 3·25 + 4·29 = 17 + 42 + 75 + 116 = 250
//  Derivation of C[3][3]:
//    13·20 + 14·24 + 15·28 + 16·32 = 260 + 336 + 420 + 512 = 1528
//
void test_4x4_hardcoded_result() {
    // clang-format off
    auto A = make(4, 4, { 1,  2,  3,  4,
                           5,  6,  7,  8,
                           9, 10, 11, 12,
                          13, 14, 15, 16 });

    auto B = make(4, 4, { 17, 18, 19, 20,
                           21, 22, 23, 24,
                           25, 26, 27, 28,
                           29, 30, 31, 32 });

    // Expected result, computed by hand:
    //   C[i][j] = sum_k A[i][k] * B[k][j]
    const std::vector<float> expected = {
         250,  260,  270,  280,   // row 0
         618,  644,  670,  696,   // row 1
         986, 1028, 1070, 1112,   // row 2
        1354, 1412, 1470, 1528    // row 3
    };
    // clang-format on

    auto C = matmul(A, B);

    ASSERT_EQ(C.rows(), 4u);
    ASSERT_EQ(C.cols(), 4u);

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            ASSERT_FLOAT_EQ(C.data()[i * 4 + j], expected[i * 4 + j]);
        }
    }
}

// ============================================================================
//  2.  4×4 — verify every element individually with named positions
// ============================================================================
void test_4x4_element_spot_checks() {
    // Same A and B as test 1; check specific named positions.
    auto A = make(4, 4, { 1,  2,  3,  4,
                           5,  6,  7,  8,
                           9, 10, 11, 12,
                          13, 14, 15, 16 });
    auto B = make(4, 4, { 17, 18, 19, 20,
                           21, 22, 23, 24,
                           25, 26, 27, 28,
                           29, 30, 31, 32 });
    auto C = matmul(A, B);

    // Top-left corner
    ASSERT_FLOAT_EQ(C.at(0, 0), 250.0f);
    // Top-right corner
    ASSERT_FLOAT_EQ(C.at(0, 3), 280.0f);
    // Bottom-left corner
    ASSERT_FLOAT_EQ(C.at(3, 0), 1354.0f);
    // Bottom-right corner
    ASSERT_FLOAT_EQ(C.at(3, 3), 1528.0f);
    // Centre elements
    ASSERT_FLOAT_EQ(C.at(1, 1), 644.0f);
    ASSERT_FLOAT_EQ(C.at(2, 2), 1070.0f);
}

// ============================================================================
//  3.  4×4 — 2×2 sub-block (manual cross-check with smaller arithmetic)
// ============================================================================
//  Use simpler 2×2 matrices first and embed in a 4×4 block-diagonal to
//  independently verify: [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
void test_4x4_subblock_verification() {
    // clang-format off
    // 2×2 test: [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
    auto A2 = make(2, 2, { 1.0f, 2.0f,
                            3.0f, 4.0f });
    auto B2 = make(2, 2, { 5.0f, 6.0f,
                            7.0f, 8.0f });
    // clang-format on

    auto C2 = matmul(A2, B2);

    // 1·5 + 2·7 = 19
    ASSERT_FLOAT_EQ(C2.at(0, 0), 19.0f);
    // 1·6 + 2·8 = 22
    ASSERT_FLOAT_EQ(C2.at(0, 1), 22.0f);
    // 3·5 + 4·7 = 43
    ASSERT_FLOAT_EQ(C2.at(1, 0), 43.0f);
    // 3·6 + 4·8 = 50
    ASSERT_FLOAT_EQ(C2.at(1, 1), 50.0f);
}

// ============================================================================
//  4.  Non-square: (2×3) × (3×4) = (2×4)
// ============================================================================
//  A = [[1,2,3],[4,5,6]]    B = [[7,8,9,10],[11,12,13,14],[15,16,17,18]]
//
//  C[0][0] = 1·7 + 2·11 + 3·15 = 7 + 22 + 45 = 74
//  C[0][3] = 1·10 + 2·14 + 3·18 = 10 + 28 + 54 = 92
//  C[1][1] = 4·8 + 5·12 + 6·16 = 32 + 60 + 96 = 188
//  C[1][3] = 4·10 + 5·14 + 6·18 = 40 + 70 + 108 = 218
//
void test_2x3_times_3x4() {
    // clang-format off
    auto A = make(2, 3, { 1.0f, 2.0f,  3.0f,
                           4.0f, 5.0f,  6.0f });
    auto B = make(3, 4, {  7.0f,  8.0f,  9.0f, 10.0f,
                           11.0f, 12.0f, 13.0f, 14.0f,
                           15.0f, 16.0f, 17.0f, 18.0f });
    // clang-format on

    auto C = matmul(A, B);

    ASSERT_EQ(C.rows(), 2u);
    ASSERT_EQ(C.cols(), 4u);

    ASSERT_FLOAT_EQ(C.at(0, 0),  74.0f);
    ASSERT_FLOAT_EQ(C.at(0, 1),  80.0f);
    ASSERT_FLOAT_EQ(C.at(0, 2),  86.0f);
    ASSERT_FLOAT_EQ(C.at(0, 3),  92.0f);
    ASSERT_FLOAT_EQ(C.at(1, 0), 173.0f);
    ASSERT_FLOAT_EQ(C.at(1, 1), 188.0f);
    ASSERT_FLOAT_EQ(C.at(1, 2), 203.0f);
    ASSERT_FLOAT_EQ(C.at(1, 3), 218.0f);
}

// ============================================================================
//  5.  Non-square: (3×1) column-vector output  — matrix × column vector
// ============================================================================
//  A = [[1,2,3],[4,5,6],[7,8,9]]    v = [[1],[0],[-1]]
//
//  C[0][0] = 1 + 0 - 3 = -2
//  C[1][0] = 4 + 0 - 6 = -2
//  C[2][0] = 7 + 0 - 9 = -2
//
void test_matrix_times_column_vector() {
    // clang-format off
    auto A = make(3, 3, { 1.0f, 2.0f, 3.0f,
                           4.0f, 5.0f, 6.0f,
                           7.0f, 8.0f, 9.0f });
    auto v = make(3, 1, { 1.0f, 0.0f, -1.0f });
    // clang-format on

    auto C = matmul(A, v);

    ASSERT_EQ(C.rows(), 3u);
    ASSERT_EQ(C.cols(), 1u);

    ASSERT_FLOAT_EQ(C.at(0, 0), -2.0f);
    ASSERT_FLOAT_EQ(C.at(1, 0), -2.0f);
    ASSERT_FLOAT_EQ(C.at(2, 0), -2.0f);
}

// ============================================================================
//  6.  Non-square: (1×4) row vector × (4×3) matrix
// ============================================================================
//  v = [[1, 2, 3, 4]]
//  B = [[1,0,0],[0,1,0],[0,0,1],[0,0,0]]  (4×3 non-square)
//
//  C = [[1, 2, 3]]
//
void test_row_vector_times_matrix() {
    // clang-format off
    auto v = make(1, 4, { 1.0f, 2.0f, 3.0f, 4.0f });
    auto B = make(4, 3, { 1.0f, 0.0f, 0.0f,
                           0.0f, 1.0f, 0.0f,
                           0.0f, 0.0f, 1.0f,
                           0.0f, 0.0f, 0.0f });
    // clang-format on

    auto C = matmul(v, B);

    ASSERT_EQ(C.rows(), 1u);
    ASSERT_EQ(C.cols(), 3u);

    ASSERT_FLOAT_EQ(C.at(0, 0), 1.0f);
    ASSERT_FLOAT_EQ(C.at(0, 1), 2.0f);
    ASSERT_FLOAT_EQ(C.at(0, 2), 3.0f);
}

// ============================================================================
//  7.  Right-identity: A × I = A
// ============================================================================
void test_right_identity() {
    auto A = make(4, 4, { 1,  2,  3,  4,
                           5,  6,  7,  8,
                           9, 10, 11, 12,
                          13, 14, 15, 16 });
    auto I4 = identity(4);

    auto C = matmul(A, I4);

    ASSERT_EQ(C.rows(), 4u);
    ASSERT_EQ(C.cols(), 4u);
    ASSERT_TRUE(approx_equal(C, A));
}

// ============================================================================
//  8.  Left-identity: I × A = A
// ============================================================================
void test_left_identity() {
    auto A = make(4, 4, { 1,  2,  3,  4,
                           5,  6,  7,  8,
                           9, 10, 11, 12,
                          13, 14, 15, 16 });
    auto I4 = identity(4);

    auto C = matmul(I4, A);

    ASSERT_TRUE(approx_equal(C, A));
}

// ============================================================================
//  9.  Non-square identity-like: (3×5) × I_5 = (3×5)
// ============================================================================
void test_nonsquare_right_identity() {
    auto A = make(3, 5, { 1, 2, 3, 4, 5,
                           6, 7, 8, 9, 10,
                          11,12,13,14,15 });
    auto I5 = identity(5);

    auto C = matmul(A, I5);

    ASSERT_EQ(C.rows(), 3u);
    ASSERT_EQ(C.cols(), 5u);
    ASSERT_TRUE(approx_equal(C, A));
}

// ============================================================================
//  10. Zero matrix: A × 0 = 0  (zero output)
// ============================================================================
void test_multiply_by_zero_matrix() {
    auto A = make(3, 4, { 1,2,3,4, 5,6,7,8, 9,10,11,12 });
    auto Z = filled(4, 2, 0.0f);

    auto C = matmul(A, Z);

    ASSERT_EQ(C.rows(), 3u);
    ASSERT_EQ(C.cols(), 2u);

    for (float v : C.data()) {
        ASSERT_FLOAT_EQ(v, 0.0f);
    }
}

// ============================================================================
//  11. GNN feature transform: H × W  (primary inference use case)
// ============================================================================
//  H = node feature matrix  (num_nodes × in_features)
//  W = weight matrix        (in_features × out_features)
//  H' = transformed features (num_nodes × out_features)
//
//  Here: 3 nodes, 4 input features, 2 output features.
//  H uses one-hot encoding for simplicity so the expected output is just
//  the rows of W indexed by the hot column.
//
//  H = [[1,0,0,0],   W = [[1,2],    H' = [[1,2],
//       [0,1,0,0],        [3,4],          [3,4],
//       [0,0,1,0]]        [5,6],          [5,6]]
//                         [7,8]]
//
void test_gnn_feature_transform() {
    // clang-format off
    auto H = make(3, 4, { 1.0f, 0.0f, 0.0f, 0.0f,    // node 0
                           0.0f, 1.0f, 0.0f, 0.0f,    // node 1
                           0.0f, 0.0f, 1.0f, 0.0f }); // node 2

    auto W = make(4, 2, { 1.0f, 2.0f,    // feature 0 → out
                           3.0f, 4.0f,    // feature 1 → out
                           5.0f, 6.0f,    // feature 2 → out
                           7.0f, 8.0f }); // feature 3 → out
    // clang-format on

    auto Hp = matmul(H, W);   // H' = H × W

    ASSERT_EQ(Hp.rows(), 3u);
    ASSERT_EQ(Hp.cols(), 2u);

    // node 0: selects row 0 of W → [1, 2]
    ASSERT_FLOAT_EQ(Hp.at(0, 0), 1.0f);
    ASSERT_FLOAT_EQ(Hp.at(0, 1), 2.0f);

    // node 1: selects row 1 of W → [3, 4]
    ASSERT_FLOAT_EQ(Hp.at(1, 0), 3.0f);
    ASSERT_FLOAT_EQ(Hp.at(1, 1), 4.0f);

    // node 2: selects row 2 of W → [5, 6]
    ASSERT_FLOAT_EQ(Hp.at(2, 0), 5.0f);
    ASSERT_FLOAT_EQ(Hp.at(2, 1), 6.0f);

    std::cout << "    H' (3×2): "
              << Hp.at(0,0) << "," << Hp.at(0,1) << " | "
              << Hp.at(1,0) << "," << Hp.at(1,1) << " | "
              << Hp.at(2,0) << "," << Hp.at(2,1) << "\n";
}

// ============================================================================
//  12. Algebraic property — NOT commutative: AB ≠ BA
// ============================================================================
//  A = [[1,2],[3,4]]   B = [[0,1],[1,0]]
//  AB = [[2,1],[4,3]]  BA = [[3,4],[1,2]]
//
void test_not_commutative() {
    auto A = make(2, 2, { 1.0f, 2.0f,
                           3.0f, 4.0f });
    auto B = make(2, 2, { 0.0f, 1.0f,
                           1.0f, 0.0f });

    auto AB = matmul(A, B);
    auto BA = matmul(B, A);

    // AB = [[2,1],[4,3]]
    ASSERT_FLOAT_EQ(AB.at(0, 0), 2.0f);
    ASSERT_FLOAT_EQ(AB.at(0, 1), 1.0f);
    ASSERT_FLOAT_EQ(AB.at(1, 0), 4.0f);
    ASSERT_FLOAT_EQ(AB.at(1, 1), 3.0f);

    // BA = [[3,4],[1,2]]
    ASSERT_FLOAT_EQ(BA.at(0, 0), 3.0f);
    ASSERT_FLOAT_EQ(BA.at(0, 1), 4.0f);
    ASSERT_FLOAT_EQ(BA.at(1, 0), 1.0f);
    ASSERT_FLOAT_EQ(BA.at(1, 1), 2.0f);

    // Verify AB ≠ BA element-wise
    ASSERT_TRUE(!approx_equal(AB, BA));
}

// ============================================================================
//  13. Algebraic property — associativity: (AB)C ≈ A(BC)
// ============================================================================
//  A = [[1,1],[0,1]]   B = [[2,0],[1,3]]   C = [[1,2],[3,1]]
//
//  AB = [[3,3],[1,3]]
//  (AB)C = [[12,9],[10,5]]
//
//  BC = [[2,4],[10,5]]
//  A(BC) = [[12,9],[10,5]]
//
void test_associativity() {
    auto A = make(2, 2, { 1.0f, 1.0f,
                           0.0f, 1.0f });
    auto B = make(2, 2, { 2.0f, 0.0f,
                           1.0f, 3.0f });
    auto C = make(2, 2, { 1.0f, 2.0f,
                           3.0f, 1.0f });

    auto AB_C = matmul(matmul(A, B), C);
    auto A_BC = matmul(A, matmul(B, C));

    ASSERT_TRUE(approx_equal(AB_C, A_BC));

    // Verify exact values: [[12,9],[10,5]]
    ASSERT_FLOAT_EQ(AB_C.at(0, 0), 12.0f);
    ASSERT_FLOAT_EQ(AB_C.at(0, 1),  9.0f);
    ASSERT_FLOAT_EQ(AB_C.at(1, 0), 10.0f);
    ASSERT_FLOAT_EQ(AB_C.at(1, 1),  5.0f);
}

// ============================================================================
//  14. Algebraic property — scalar multiplication equivalence
// ============================================================================
//  k · (A × B) ≈ (k·A) × B ≈ A × (k·B)
//
void test_scalar_distributivity() {
    auto A = make(3, 3, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    auto B = make(3, 3, { 9, 8, 7, 6, 5, 4, 3, 2, 1 });

    const float k = 3.0f;

    // k · (A × B)
    auto AB = matmul(A, B);
    std::vector<float> kAB_data = AB.data();
    for (auto& v : kAB_data) v *= k;
    auto kAB = Tensor::dense(AB.rows(), AB.cols(), std::move(kAB_data));

    // (k·A) × B
    std::vector<float> kA_data = A.data();
    for (auto& v : kA_data) v *= k;
    auto kA_x_B = matmul(Tensor::dense(3, 3, std::move(kA_data)), B);

    // A × (k·B)
    std::vector<float> kB_data = B.data();
    for (auto& v : kB_data) v *= k;
    auto A_x_kB = matmul(A, Tensor::dense(3, 3, std::move(kB_data)));

    ASSERT_TRUE(approx_equal(kAB, kA_x_B));
    ASSERT_TRUE(approx_equal(kAB, A_x_kB));
}

// ============================================================================
//  15. Output tensor properties — format must be Dense
// ============================================================================
void test_output_is_dense() {
    auto A = make(3, 3, { 1,2,3, 4,5,6, 7,8,9 });
    auto B = make(3, 3, { 9,8,7, 6,5,4, 3,2,1 });

    auto C = matmul(A, B);

    ASSERT_TRUE(C.format() == StorageFormat::Dense);
}

// ============================================================================
//  16. Output tensor properties — shape must be (M × N)
// ============================================================================
void test_output_shape() {
    // (7 × 3) × (3 × 11) = (7 × 11)
    auto A = filled(7, 3, 1.0f);
    auto B = filled(3, 11, 1.0f);

    auto C = matmul(A, B);

    ASSERT_EQ(C.rows(), 7u);
    ASSERT_EQ(C.cols(), 11u);
    ASSERT_EQ(C.numel(), 77u);

    // row_ptr and col_ind must be empty (it's Dense)
    ASSERT_TRUE(C.row_ptr().empty());
    ASSERT_TRUE(C.col_ind().empty());
}

// ============================================================================
//  17. Output tensor properties — memory footprint is exactly M×N×4 bytes
// ============================================================================
void test_output_memory_footprint() {
    auto A = filled(5, 8, 1.0f);   // 5×8
    auto B = filled(8, 6, 1.0f);   // 8×6
    auto C = matmul(A, B);         // 5×6

    const std::size_t expected = 5u * 6u * sizeof(float);  // 120 bytes
    ASSERT_EQ(C.memory_footprint_bytes(), expected);

    std::cout << "    matmul(5×8, 8×6) output = " << C.memory_footprint_bytes()
              << " bytes (" << C.rows() << "×" << C.cols() << " Dense)\n";
}

// ============================================================================
//  18. Stress test — all-ones matrix: every element of C = K
// ============================================================================
//  If A = 1_{M×K}  and  B = 1_{K×N}, then C[i][j] = K  for all i, j.
//  This gives a simple closed-form check for large matrices.
//
void test_stress_all_ones_128x128() {
    constexpr std::size_t M = 128, K = 128, N = 128;

    auto A = filled(M, K, 1.0f);
    auto B = filled(K, N, 1.0f);

    auto start = std::chrono::steady_clock::now();
    auto C = matmul(A, B);
    auto end   = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  end - start).count();

    ASSERT_EQ(C.rows(), M);
    ASSERT_EQ(C.cols(), N);

    // Every element should equal K = 128
    bool all_correct = true;
    for (float v : C.data()) {
        if (std::fabs(v - static_cast<float>(K)) > 1e-3f) {
            all_correct = false;
            break;
        }
    }
    ASSERT_TRUE(all_correct);

    std::cout << "    matmul(128×128, 128×128) completed in " << ms << " ms\n";
}

// ============================================================================
//  19. Stress test — all-ones rectangle: (1024 × 32) × (32 × 256)
// ============================================================================
//  Each output element = 32 (the inner dimension K).
//
void test_stress_rectangle_1024x32x256() {
    constexpr std::size_t M = 1024, K = 32, N = 256;

    auto A = filled(M, K, 1.0f);
    auto B = filled(K, N, 1.0f);

    auto start = std::chrono::steady_clock::now();
    auto C = matmul(A, B);
    auto end   = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  end - start).count();

    ASSERT_EQ(C.rows(), M);
    ASSERT_EQ(C.cols(), N);

    // Sample 5 spread positions instead of all M×N to keep assertion count low
    for (std::size_t i : {0UL, 256UL, 512UL, 768UL, 1023UL}) {
        for (std::size_t j : {0UL, 63UL, 127UL, 255UL}) {
            ASSERT_FLOAT_EQ(C.data()[i * N + j], static_cast<float>(K));
        }
    }

    std::cout << "    matmul(1024×32, 32×256) completed in " << ms << " ms, "
              << "output = " << C.memory_footprint_bytes() / 1024u << " KB\n";
}

// ============================================================================
//  20. Stress test — identity property on large matrix
// ============================================================================
//  A × I_{256} = A  (exercises large square identity)
//
void test_stress_identity_256() {
    constexpr std::size_t N = 256;

    // Fill A with values 0..N*N-1
    std::vector<float> data(N * N);
    std::iota(data.begin(), data.end(), 0.0f);
    auto A = Tensor::dense(N, N, data);

    auto IN = identity(N);
    auto C  = matmul(A, IN);

    ASSERT_EQ(C.rows(), N);
    ASSERT_EQ(C.cols(), N);

    // Spot-check 8 elements scattered across the matrix
    for (std::size_t idx : {0UL, 100UL, 1000UL, 5000UL,
                             10000UL, 30000UL, 50000UL, 65535UL}) {
        ASSERT_FLOAT_EQ(C.data()[idx], data[idx]);
    }

    std::cout << "    matmul(256×256, I_256): spot checks passed\n";
}

// ============================================================================
//  21. Error — incompatible shapes: (3×4) × (3×2) — inner dims 4 ≠ 3
// ============================================================================
void test_error_shape_mismatch_basic() {
    auto A = filled(3, 4, 1.0f);   // cols = 4
    auto B = filled(3, 2, 1.0f);   // rows = 3  →  4 ≠ 3

    ASSERT_THROWS(matmul(A, B), std::invalid_argument);
}

// ============================================================================
//  22. Error — exception message must mention the conflicting dimensions
// ============================================================================
void test_error_shape_mismatch_message() {
    auto A = filled(2, 5, 1.0f);   // cols = 5
    auto B = filled(7, 3, 1.0f);   // rows = 7

    // The message should say something about 5 vs 7
    ASSERT_THROWS_MSG(
        matmul(A, B),
        std::invalid_argument,
        "mismatch"
    );
    ASSERT_THROWS_MSG(
        matmul(A, B),
        std::invalid_argument,
        "5"   // A.cols()
    );
    ASSERT_THROWS_MSG(
        matmul(A, B),
        std::invalid_argument,
        "7"   // B.rows()
    );
}

// ============================================================================
//  23. Error — several invalid shape combinations
// ============================================================================
void test_error_shape_mismatch_various() {
    // (1×1) × (2×1) — inner 1 ≠ 2
    ASSERT_THROWS(matmul(filled(1, 1, 0), filled(2, 1, 0)),
                  std::invalid_argument);

    // (4×3) × (4×3) — inner 3 ≠ 4
    ASSERT_THROWS(matmul(filled(4, 3, 0), filled(4, 3, 0)),
                  std::invalid_argument);

    // (100×50) × (51×200) — inner 50 ≠ 51
    ASSERT_THROWS(matmul(filled(100, 50, 0), filled(51, 200, 0)),
                  std::invalid_argument);

    // (1×1000) × (999×1) — inner 1000 ≠ 999
    ASSERT_THROWS(matmul(filled(1, 1000, 0), filled(999, 1, 0)),
                  std::invalid_argument);
}

// ============================================================================
//  24. Error — (N×M) × (M×0) should succeed (zero-column output is valid)
//             (N×0) × (0×M) should succeed (zero inner-dimension, result = 0)
// ============================================================================
void test_degenerate_zero_dimension() {
    // (3×4) × (4×0) → (3×0)  — valid: empty output
    auto A = filled(3, 4, 1.0f);
    auto B = Tensor::dense(4, 0);        // 4 rows, 0 cols

    auto C = matmul(A, B);
    ASSERT_EQ(C.rows(), 3u);
    ASSERT_EQ(C.cols(), 0u);
    ASSERT_EQ(C.numel(), 0u);

    // (3×0) × (0×5) → (3×5) all zeros — sum over empty K is 0
    auto A2 = Tensor::dense(3, 0);
    auto B2 = Tensor::dense(0, 5);

    auto C2 = matmul(A2, B2);
    ASSERT_EQ(C2.rows(), 3u);
    ASSERT_EQ(C2.cols(), 5u);
    for (float v : C2.data()) ASSERT_FLOAT_EQ(v, 0.0f);
}

// ============================================================================
//  25. Error — (1×1) correct and (1×2)×(1×1) incorrect
// ============================================================================
void test_error_single_element_mismatch() {
    // Valid: (1×1) × (1×1) = [[scalar]]
    auto A = make(1, 1, { 5.0f });
    auto B = make(1, 1, { 7.0f });
    auto C = matmul(A, B);
    ASSERT_EQ(C.rows(), 1u);
    ASSERT_EQ(C.cols(), 1u);
    ASSERT_FLOAT_EQ(C.at(0, 0), 35.0f);   // 5 × 7 = 35

    // Invalid: (1×2) × (1×1) — inner 2 ≠ 1
    ASSERT_THROWS(matmul(make(1, 2, {1.0f, 2.0f}), make(1, 1, {3.0f})),
                  std::invalid_argument);
}

// ============================================================================
//  26. Error — sparse A must throw
// ============================================================================
void test_error_sparse_A() {
    auto A = Tensor::sparse_csr(3, 3,
                                {0, 1, 2, 3},
                                {0, 1, 2},
                                {1.0f, 1.0f, 1.0f});
    auto B = filled(3, 3, 1.0f);

    ASSERT_THROWS_MSG(matmul(A, B), std::invalid_argument, "SparseCSR");
}

// ============================================================================
//  27. Error — sparse B must throw
// ============================================================================
void test_error_sparse_B() {
    auto A = filled(3, 3, 1.0f);
    auto B = Tensor::sparse_csr(3, 3,
                                {0, 1, 2, 3},
                                {0, 1, 2},
                                {1.0f, 1.0f, 1.0f});

    ASSERT_THROWS_MSG(matmul(A, B), std::invalid_argument, "SparseCSR");
}

// ============================================================================
//  28. Edge case — 1×1 scalar multiplication
// ============================================================================
void test_1x1_scalar() {
    // [[k]] × [[v]] = [[k*v]]
    for (float k : {0.0f, 1.0f, -3.0f, 7.5f}) {
        for (float v : {0.0f, 1.0f, 2.0f, -0.5f}) {
            auto K = make(1, 1, {k});
            auto V = make(1, 1, {v});
            auto C = matmul(K, V);
            ASSERT_EQ(C.rows(), 1u);
            ASSERT_EQ(C.cols(), 1u);
            ASSERT_FLOAT_EQ(C.at(0, 0), k * v);
        }
    }
}

// ============================================================================
//  29. Edge case — multiplying by all-zeros gives all-zeros output
// ============================================================================
void test_multiply_gives_zeros() {
    auto A = make(5, 5, { 1,2,3,4,5, 6,7,8,9,10,
                           11,12,13,14,15, 16,17,18,19,20,
                           21,22,23,24,25 });
    auto Z = filled(5, 5, 0.0f);

    auto C1 = matmul(A, Z);  // A × 0
    auto C2 = matmul(Z, A);  // 0 × A

    for (float v : C1.data()) ASSERT_FLOAT_EQ(v, 0.0f);
    for (float v : C2.data()) ASSERT_FLOAT_EQ(v, 0.0f);
}

// ============================================================================
//  30. Edge case — chain of matmuls matches direct deep result
// ============================================================================
//  Verifies that stacking layers (a common GNN pattern) produces a stable
//  numeric result: H'' = (H × W1) × W2  must equal H × (W1 × W2)
//  (by associativity, but also tests no mutation of input tensors).
//
void test_chain_matmuls_gnn_pattern() {
    // 4 nodes, 6 → 4 → 2 feature dims
    auto H  = make(4, 6, { 1,0,0,0,0,0,
                            0,1,0,0,0,0,
                            0,0,1,0,0,0,
                            0,0,0,1,0,0 });

    auto W1 = make(6, 4, { 1,0,0,0,
                            0,1,0,0,
                            0,0,1,0,
                            0,0,0,1,
                            0,0,0,0,
                            0,0,0,0 });

    auto W2 = make(4, 2, { 1,0,
                            0,1,
                            1,1,
                            0,0 });

    // Path 1: H'' = (H × W1) × W2
    auto HW1    = matmul(H, W1);
    auto HW1W2a = matmul(HW1, W2);

    // Path 2: H'' = H × (W1 × W2)
    auto W1W2   = matmul(W1, W2);
    auto HW1W2b = matmul(H, W1W2);

    ASSERT_EQ(HW1W2a.rows(), 4u);
    ASSERT_EQ(HW1W2a.cols(), 2u);
    ASSERT_TRUE(approx_equal(HW1W2a, HW1W2b));

    // Spot check: node 0 mapped through identity → W2 row 0 = [1,0]
    ASSERT_FLOAT_EQ(HW1W2a.at(0, 0), 1.0f);
    ASSERT_FLOAT_EQ(HW1W2a.at(0, 1), 0.0f);
    // node 2 → W2 row 2 = [1,1]
    ASSERT_FLOAT_EQ(HW1W2a.at(2, 0), 1.0f);
    ASSERT_FLOAT_EQ(HW1W2a.at(2, 1), 1.0f);
}

// ============================================================================
//  main — run all tests
// ============================================================================
int main() {
    std::cout << "\n"
        "+=================================================================+\n"
        "|   TinyGNN — Dense GEMM Unit Tests (Phase 3)                    |\n"
        "|   Testing: matmul(A, B) = C = A × B                           |\n"
        "+=================================================================+\n\n";

    std::cout << "── 1. 4×4 Hardcoded Result (spec required) ─────────────────\n";
    RUN_TEST(test_4x4_hardcoded_result);
    RUN_TEST(test_4x4_element_spot_checks);
    RUN_TEST(test_4x4_subblock_verification);

    std::cout << "\n── 2. Non-Square Shapes ────────────────────────────────────\n";
    RUN_TEST(test_2x3_times_3x4);
    RUN_TEST(test_matrix_times_column_vector);
    RUN_TEST(test_row_vector_times_matrix);

    std::cout << "\n── 3. Identity & Zero Properties ───────────────────────────\n";
    RUN_TEST(test_right_identity);
    RUN_TEST(test_left_identity);
    RUN_TEST(test_nonsquare_right_identity);
    RUN_TEST(test_multiply_by_zero_matrix);

    std::cout << "\n── 4. GNN Feature Transform ─────────────────────────────────\n";
    RUN_TEST(test_gnn_feature_transform);

    std::cout << "\n── 5. Algebraic Properties ──────────────────────────────────\n";
    RUN_TEST(test_not_commutative);
    RUN_TEST(test_associativity);
    RUN_TEST(test_scalar_distributivity);

    std::cout << "\n── 6. Output Tensor Properties ──────────────────────────────\n";
    RUN_TEST(test_output_is_dense);
    RUN_TEST(test_output_shape);
    RUN_TEST(test_output_memory_footprint);

    std::cout << "\n── 7. Stress / Scale Tests ──────────────────────────────────\n";
    RUN_TEST(test_stress_all_ones_128x128);
    RUN_TEST(test_stress_rectangle_1024x32x256);
    RUN_TEST(test_stress_identity_256);

    std::cout << "\n── 8. Error Handling — Dimension Mismatch ───────────────────\n";
    RUN_TEST(test_error_shape_mismatch_basic);
    RUN_TEST(test_error_shape_mismatch_message);
    RUN_TEST(test_error_shape_mismatch_various);
    RUN_TEST(test_degenerate_zero_dimension);
    RUN_TEST(test_error_single_element_mismatch);

    std::cout << "\n── 9. Error Handling — Sparse Inputs ────────────────────────\n";
    RUN_TEST(test_error_sparse_A);
    RUN_TEST(test_error_sparse_B);

    std::cout << "\n── 10. Edge / Degenerate Cases ──────────────────────────────\n";
    RUN_TEST(test_1x1_scalar);
    RUN_TEST(test_multiply_gives_zeros);
    RUN_TEST(test_chain_matmuls_gnn_pattern);

    // ── Summary ──────────────────────────────────────────────────────────────
    std::cout << "\n=================================================================\n";
    std::cout << "  Total : " << g_tests_run    << "\n";
    std::cout << "  Passed: " << g_tests_passed << "\n";
    std::cout << "  Failed: " << g_tests_failed << "\n";
    std::cout << "=================================================================\n\n";

    return g_tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
