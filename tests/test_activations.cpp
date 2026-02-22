// ============================================================================
//  TinyGNN — Activations & Utilities Unit Tests  (Phase 5)
//  tests/test_activations.cpp
//
//  Test categories:
//    1.  ReLU — basic cases, zero/negative/positive         (tests  1 –  5)
//    2.  Leaky ReLU — default alpha, GAT alpha              (tests  6 – 10)
//    3.  ELU — smooth negative region                       (tests 11 – 14)
//    4.  Sigmoid — bounds, symmetry, special values         (tests 15 – 19)
//    5.  Tanh — bounds, symmetry, special values            (tests 20 – 24)
//    6.  GELU — approximation, special values               (tests 25 – 28)
//    7.  Softmax — row sums, probability properties         (tests 29 – 35)
//    8.  Log-Softmax — log-probability properties           (tests 36 – 41)
//    9.  Add bias — broadcasting, shapes                    (tests 42 – 47)
//   10.  GNN pipeline integration                           (tests 48 – 50)
//   11.  Error handling — SparseCSR rejected                (tests 51 – 60)
//   12.  Stress / scale tests                               (tests 61 – 65)
//   13.  Edge / degenerate cases                            (tests 66 – 70)
//
//  All expected values are computed by hand or verified against
//  known mathematical properties of each function.
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

// ── Minimal dependency-free test framework (same idiom as other tests) ──────

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

// Wider tolerance for transcendental functions (exp, tanh, etc.)
#define ASSERT_NEAR(a, b, tol)                                                 \
    do {                                                                       \
        ++g_tests_run;                                                         \
        if (std::fabs(static_cast<double>(a) - static_cast<double>(b)) > (tol)) {\
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__            \
                      << " — ASSERT_NEAR(" #a ", " #b ", " #tol ") → "        \
                      << (a) << " vs " << (b)                                 \
                      << " (diff=" << std::fabs(static_cast<double>(a)         \
                                                - static_cast<double>(b))      \
                      << ")\n";                                                \
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
                      << " — ASSERT_THROWS_MSG: message \""                   \
                      << msg_ << "\" does not contain \"" << substr            \
                      << "\"\n";                                               \
            ++g_tests_failed;                                                  \
        } else { ++g_tests_passed; }                                           \
    } while (0)

using namespace tinygnn;

// ── Helper: make a simple sparse CSR tensor for error-handling tests ────────
static Tensor make_sparse_3x3() {
    return Tensor::sparse_csr(3, 3,
        {0, 1, 2, 3},  // row_ptr
        {0, 1, 2},      // col_ind  (diagonal)
        {1.0f, 1.0f, 1.0f});
}

// ============================================================================
//  Category 1: ReLU  (tests 1–5)
// ============================================================================

// Test 1: ReLU basic — positive values unchanged, negatives zeroed
void test_relu_basic() {
    //  Input:  [-3, -1, 0, 1, 3]
    //  Expect: [ 0,  0, 0, 1, 3]
    auto X = Tensor::dense(1, 5, {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f});
    relu_inplace(X);

    ASSERT_FLOAT_EQ(X.at(0, 0), 0.0f);
    ASSERT_FLOAT_EQ(X.at(0, 1), 0.0f);
    ASSERT_FLOAT_EQ(X.at(0, 2), 0.0f);
    ASSERT_FLOAT_EQ(X.at(0, 3), 1.0f);
    ASSERT_FLOAT_EQ(X.at(0, 4), 3.0f);
}

// Test 2: ReLU 2D matrix — hand-calculated
void test_relu_2d_matrix() {
    //  [ -2   3 ]     [ 0  3 ]
    //  [  5  -7 ]  →  [ 5  0 ]
    //  [ -1   0 ]     [ 0  0 ]
    auto X = Tensor::dense(3, 2, {-2.0f, 3.0f, 5.0f, -7.0f, -1.0f, 0.0f});
    relu_inplace(X);

    ASSERT_FLOAT_EQ(X.at(0, 0), 0.0f);
    ASSERT_FLOAT_EQ(X.at(0, 1), 3.0f);
    ASSERT_FLOAT_EQ(X.at(1, 0), 5.0f);
    ASSERT_FLOAT_EQ(X.at(1, 1), 0.0f);
    ASSERT_FLOAT_EQ(X.at(2, 0), 0.0f);
    ASSERT_FLOAT_EQ(X.at(2, 1), 0.0f);
}

// Test 3: ReLU all-positive — no change
void test_relu_all_positive() {
    auto X = Tensor::dense(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    relu_inplace(X);
    ASSERT_FLOAT_EQ(X.at(0, 0), 1.0f);
    ASSERT_FLOAT_EQ(X.at(0, 1), 2.0f);
    ASSERT_FLOAT_EQ(X.at(1, 2), 6.0f);
}

// Test 4: ReLU all-negative — all zeros
void test_relu_all_negative() {
    auto X = Tensor::dense(2, 2, {-1.0f, -2.0f, -3.0f, -4.0f});
    relu_inplace(X);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            ASSERT_FLOAT_EQ(X.at(i, j), 0.0f);
}

// Test 5: ReLU preserves exact zero
void test_relu_preserves_zero() {
    auto X = Tensor::dense(1, 1, {0.0f});
    relu_inplace(X);
    ASSERT_FLOAT_EQ(X.at(0, 0), 0.0f);
}

// ============================================================================
//  Category 2: Leaky ReLU  (tests 6–10)
// ============================================================================

// Test 6: Leaky ReLU default alpha=0.01
void test_leaky_relu_default_alpha() {
    //  Input:  [-100, -1, 0, 1, 100]
    //  Expect: [-1.0, -0.01, 0, 1, 100]
    auto X = Tensor::dense(1, 5, {-100.0f, -1.0f, 0.0f, 1.0f, 100.0f});
    leaky_relu_inplace(X);

    ASSERT_FLOAT_EQ(X.at(0, 0), -1.0f);
    ASSERT_FLOAT_EQ(X.at(0, 1), -0.01f);
    ASSERT_FLOAT_EQ(X.at(0, 2), 0.0f);
    ASSERT_FLOAT_EQ(X.at(0, 3), 1.0f);
    ASSERT_FLOAT_EQ(X.at(0, 4), 100.0f);
}

// Test 7: Leaky ReLU GAT alpha=0.2
void test_leaky_relu_gat_alpha() {
    //  Input:  [-5, -1, 0, 1, 5]
    //  Expect: [-1.0, -0.2, 0, 1, 5]
    auto X = Tensor::dense(1, 5, {-5.0f, -1.0f, 0.0f, 1.0f, 5.0f});
    leaky_relu_inplace(X, 0.2f);

    ASSERT_FLOAT_EQ(X.at(0, 0), -1.0f);
    ASSERT_FLOAT_EQ(X.at(0, 1), -0.2f);
    ASSERT_FLOAT_EQ(X.at(0, 2), 0.0f);
    ASSERT_FLOAT_EQ(X.at(0, 3), 1.0f);
    ASSERT_FLOAT_EQ(X.at(0, 4), 5.0f);
}

// Test 8: Leaky ReLU 2D matrix
void test_leaky_relu_2d() {
    //  alpha=0.1
    //  [ -2   3 ]     [ -0.2   3 ]
    //  [  5  -7 ]  →  [  5   -0.7 ]
    auto X = Tensor::dense(2, 2, {-2.0f, 3.0f, 5.0f, -7.0f});
    leaky_relu_inplace(X, 0.1f);

    ASSERT_FLOAT_EQ(X.at(0, 0), -0.2f);
    ASSERT_FLOAT_EQ(X.at(0, 1), 3.0f);
    ASSERT_FLOAT_EQ(X.at(1, 0), 5.0f);
    ASSERT_FLOAT_EQ(X.at(1, 1), -0.7f);
}

// Test 9: Leaky ReLU alpha=0 degenerates to ReLU
void test_leaky_relu_alpha_zero_is_relu() {
    auto X1 = Tensor::dense(1, 4, {-2.0f, 0.0f, 1.0f, -3.0f});
    auto X2 = Tensor::dense(1, 4, {-2.0f, 0.0f, 1.0f, -3.0f});

    leaky_relu_inplace(X1, 0.0f);
    relu_inplace(X2);

    for (std::size_t j = 0; j < 4; ++j)
        ASSERT_FLOAT_EQ(X1.at(0, j), X2.at(0, j));
}

// Test 10: Leaky ReLU alpha=1 degenerates to identity
void test_leaky_relu_alpha_one_is_identity() {
    auto X = Tensor::dense(1, 4, {-2.0f, 0.0f, 1.0f, -3.0f});
    leaky_relu_inplace(X, 1.0f);

    ASSERT_FLOAT_EQ(X.at(0, 0), -2.0f);
    ASSERT_FLOAT_EQ(X.at(0, 1), 0.0f);
    ASSERT_FLOAT_EQ(X.at(0, 2), 1.0f);
    ASSERT_FLOAT_EQ(X.at(0, 3), -3.0f);
}

// ============================================================================
//  Category 3: ELU  (tests 11–14)
// ============================================================================

// Test 11: ELU basic — positive unchanged, negative region smooth
void test_elu_basic() {
    //  alpha=1.0 (default)
    //  f(-1) = 1.0 * (exp(-1) - 1) ≈ -0.6321
    //  f(0)  = 0
    //  f(2)  = 2
    auto X = Tensor::dense(1, 3, {-1.0f, 0.0f, 2.0f});
    elu_inplace(X);

    ASSERT_NEAR(X.at(0, 0), std::exp(-1.0f) - 1.0f, 1e-5);
    ASSERT_FLOAT_EQ(X.at(0, 1), 0.0f);
    ASSERT_FLOAT_EQ(X.at(0, 2), 2.0f);
}

// Test 12: ELU custom alpha=2.0
void test_elu_custom_alpha() {
    //  f(-1) = 2.0 * (exp(-1) - 1) ≈ -1.2642
    auto X = Tensor::dense(1, 2, {-1.0f, 3.0f});
    elu_inplace(X, 2.0f);

    ASSERT_NEAR(X.at(0, 0), 2.0f * (std::exp(-1.0f) - 1.0f), 1e-5);
    ASSERT_FLOAT_EQ(X.at(0, 1), 3.0f);
}

// Test 13: ELU saturates at -alpha for very negative input
void test_elu_saturation() {
    //  f(-100) ≈ alpha * (0 - 1) = -alpha = -1.0
    auto X = Tensor::dense(1, 1, {-100.0f});
    elu_inplace(X, 1.0f);

    ASSERT_NEAR(X.at(0, 0), -1.0f, 1e-4);
}

// Test 14: ELU 2D matrix with mixed signs
void test_elu_2d_matrix() {
    auto X = Tensor::dense(2, 2, {-2.0f, 1.0f, 0.0f, -0.5f});
    elu_inplace(X, 1.0f);

    ASSERT_NEAR(X.at(0, 0), std::exp(-2.0f) - 1.0f, 1e-5);
    ASSERT_FLOAT_EQ(X.at(0, 1), 1.0f);
    ASSERT_FLOAT_EQ(X.at(1, 0), 0.0f);
    ASSERT_NEAR(X.at(1, 1), std::exp(-0.5f) - 1.0f, 1e-5);
}

// ============================================================================
//  Category 4: Sigmoid  (tests 15–19)
// ============================================================================

// Test 15: Sigmoid basic values
void test_sigmoid_basic() {
    //  σ(0) = 0.5
    //  σ(large) → 1
    //  σ(-large) → 0
    auto X = Tensor::dense(1, 3, {0.0f, 100.0f, -100.0f});
    sigmoid_inplace(X);

    ASSERT_NEAR(X.at(0, 0), 0.5f, 1e-5);
    ASSERT_NEAR(X.at(0, 1), 1.0f, 1e-5);
    ASSERT_NEAR(X.at(0, 2), 0.0f, 1e-5);
}

// Test 16: Sigmoid symmetry — σ(x) + σ(-x) = 1
void test_sigmoid_symmetry() {
    auto X1 = Tensor::dense(1, 4, {-2.0f, -0.5f, 0.5f, 2.0f});
    auto X2 = Tensor::dense(1, 4, { 2.0f,  0.5f, -0.5f, -2.0f});

    sigmoid_inplace(X1);
    sigmoid_inplace(X2);

    for (std::size_t j = 0; j < 4; ++j) {
        ASSERT_NEAR(X1.at(0, j) + X2.at(0, j), 1.0f, 1e-5);
    }
}

// Test 17: Sigmoid output always in (0, 1)
void test_sigmoid_output_bounds() {
    auto X = Tensor::dense(1, 6, {-50.0f, -10.0f, -1.0f, 1.0f, 10.0f, 50.0f});
    sigmoid_inplace(X);

    for (std::size_t j = 0; j < 6; ++j) {
        ASSERT_TRUE(X.at(0, j) >= 0.0f);
        ASSERT_TRUE(X.at(0, j) <= 1.0f);
    }
}

// Test 18: Sigmoid hand-calculated
void test_sigmoid_hand_calculated() {
    //  σ(1) = 1/(1+e^-1) ≈ 0.7310586
    //  σ(-1) = 1/(1+e^1) ≈ 0.2689414
    auto X = Tensor::dense(1, 2, {1.0f, -1.0f});
    sigmoid_inplace(X);

    ASSERT_NEAR(X.at(0, 0), 0.7310586f, 1e-4);
    ASSERT_NEAR(X.at(0, 1), 0.2689414f, 1e-4);
}

// Test 19: Sigmoid 2D matrix
void test_sigmoid_2d_matrix() {
    auto X = Tensor::dense(2, 2, {0.0f, 1.0f, -1.0f, 0.0f});
    sigmoid_inplace(X);

    ASSERT_NEAR(X.at(0, 0), 0.5f, 1e-5);
    ASSERT_NEAR(X.at(0, 1), 0.7310586f, 1e-4);
    ASSERT_NEAR(X.at(1, 0), 0.2689414f, 1e-4);
    ASSERT_NEAR(X.at(1, 1), 0.5f, 1e-5);
}

// ============================================================================
//  Category 5: Tanh  (tests 20–24)
// ============================================================================

// Test 20: Tanh basic values
void test_tanh_basic() {
    //  tanh(0) = 0
    //  tanh(large) → 1
    //  tanh(-large) → -1
    auto X = Tensor::dense(1, 3, {0.0f, 100.0f, -100.0f});
    tanh_inplace(X);

    ASSERT_NEAR(X.at(0, 0), 0.0f, 1e-5);
    ASSERT_NEAR(X.at(0, 1), 1.0f, 1e-5);
    ASSERT_NEAR(X.at(0, 2), -1.0f, 1e-5);
}

// Test 21: Tanh symmetry — tanh(-x) = -tanh(x)
void test_tanh_symmetry() {
    auto X1 = Tensor::dense(1, 3, {-2.0f, 0.5f, 3.0f});
    auto X2 = Tensor::dense(1, 3, { 2.0f, -0.5f, -3.0f});

    tanh_inplace(X1);
    tanh_inplace(X2);

    for (std::size_t j = 0; j < 3; ++j) {
        ASSERT_NEAR(X1.at(0, j) + X2.at(0, j), 0.0f, 1e-5);
    }
}

// Test 22: Tanh output always in (-1, 1)
void test_tanh_output_bounds() {
    auto X = Tensor::dense(1, 6, {-50.0f, -5.0f, -0.1f, 0.1f, 5.0f, 50.0f});
    tanh_inplace(X);

    for (std::size_t j = 0; j < 6; ++j) {
        ASSERT_TRUE(X.at(0, j) >= -1.0f);
        ASSERT_TRUE(X.at(0, j) <= 1.0f);
    }
}

// Test 23: Tanh hand-calculated
void test_tanh_hand_calculated() {
    //  tanh(1) ≈ 0.7615942
    //  tanh(0.5) ≈ 0.4621172
    auto X = Tensor::dense(1, 2, {1.0f, 0.5f});
    tanh_inplace(X);

    ASSERT_NEAR(X.at(0, 0), 0.7615942f, 1e-4);
    ASSERT_NEAR(X.at(0, 1), 0.4621172f, 1e-4);
}

// Test 24: Tanh 2D matrix
void test_tanh_2d_matrix() {
    auto X = Tensor::dense(2, 2, {0.0f, 1.0f, -1.0f, 2.0f});
    tanh_inplace(X);

    ASSERT_NEAR(X.at(0, 0), 0.0f, 1e-5);
    ASSERT_NEAR(X.at(0, 1), std::tanh(1.0f), 1e-5);
    ASSERT_NEAR(X.at(1, 0), std::tanh(-1.0f), 1e-5);
    ASSERT_NEAR(X.at(1, 1), std::tanh(2.0f), 1e-5);
}

// ============================================================================
//  Category 6: GELU  (tests 25–28)
// ============================================================================

// Reference GELU using the tanh approximation
static float gelu_ref(float x) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float COEFF = 0.044715f;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

// Test 25: GELU basic values
void test_gelu_basic() {
    //  GELU(0) = 0 (exact)
    //  GELU(large positive) ≈ x
    //  GELU(large negative) ≈ 0
    auto X = Tensor::dense(1, 3, {0.0f, 10.0f, -10.0f});
    gelu_inplace(X);

    ASSERT_NEAR(X.at(0, 0), 0.0f, 1e-5);
    ASSERT_NEAR(X.at(0, 1), 10.0f, 1e-3);  // large positive ≈ x
    ASSERT_NEAR(X.at(0, 2), 0.0f, 1e-3);   // large negative ≈ 0
}

// Test 26: GELU hand-calculated against reference
void test_gelu_hand_calculated() {
    std::vector<float> vals = {-2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f};
    auto X = Tensor::dense(1, 6, vals);
    gelu_inplace(X);

    for (std::size_t j = 0; j < 6; ++j) {
        ASSERT_NEAR(X.at(0, j), gelu_ref(vals[j]), 1e-4);
    }
}

// Test 27: GELU near-symmetry — GELU(x) + GELU(-x) ≈ x for moderate x
//  Because GELU(x) = x·Φ(x) and GELU(-x) = -x·Φ(-x) = -x·(1-Φ(x))
//  so GELU(x) + GELU(-x) = x·Φ(x) - x + x·Φ(x) = x(2Φ(x)-1)
//  This is NOT exactly x, but we verify GELU(0) = 0 symmetry holds.
void test_gelu_zero_symmetry() {
    auto X = Tensor::dense(1, 1, {0.0f});
    gelu_inplace(X);
    ASSERT_NEAR(X.at(0, 0), 0.0f, 1e-6);
}

// Test 28: GELU 2D matrix
void test_gelu_2d_matrix() {
    auto X = Tensor::dense(2, 2, {-1.0f, 0.0f, 1.0f, 2.0f});
    gelu_inplace(X);

    ASSERT_NEAR(X.at(0, 0), gelu_ref(-1.0f), 1e-4);
    ASSERT_NEAR(X.at(0, 1), 0.0f, 1e-5);
    ASSERT_NEAR(X.at(1, 0), gelu_ref(1.0f), 1e-4);
    ASSERT_NEAR(X.at(1, 1), gelu_ref(2.0f), 1e-4);
}

// ============================================================================
//  Category 7: Softmax  (tests 29–35)
// ============================================================================

// Test 29: Softmax single row — sums to 1
void test_softmax_row_sums_to_one() {
    auto X = Tensor::dense(1, 4, {1.0f, 2.0f, 3.0f, 4.0f});
    softmax_inplace(X);

    float sum = 0.0f;
    for (std::size_t j = 0; j < 4; ++j) sum += X.at(0, j);
    ASSERT_NEAR(sum, 1.0f, 1e-5);
}

// Test 30: Softmax — all values in [0, 1]
void test_softmax_output_bounds() {
    auto X = Tensor::dense(1, 4, {-10.0f, 0.0f, 5.0f, 20.0f});
    softmax_inplace(X);

    for (std::size_t j = 0; j < 4; ++j) {
        ASSERT_TRUE(X.at(0, j) >= 0.0f);
        ASSERT_TRUE(X.at(0, j) <= 1.0f);
    }
}

// Test 31: Softmax — equal inputs produce uniform distribution
void test_softmax_uniform() {
    auto X = Tensor::dense(1, 4, {5.0f, 5.0f, 5.0f, 5.0f});
    softmax_inplace(X);

    for (std::size_t j = 0; j < 4; ++j) {
        ASSERT_NEAR(X.at(0, j), 0.25f, 1e-5);
    }
}

// Test 32: Softmax — hand-calculated 3-class
void test_softmax_hand_calculated() {
    //  Input:  [1, 2, 3]
    //  max = 3, shifted = [-2, -1, 0]
    //  exp: [e^-2, e^-1, 1]  ≈ [0.1353, 0.3679, 1.0]
    //  sum ≈ 1.5032
    //  Result: [0.0900, 0.2447, 0.6652]
    auto X = Tensor::dense(1, 3, {1.0f, 2.0f, 3.0f});
    softmax_inplace(X);

    float e1 = std::exp(-2.0f), e2 = std::exp(-1.0f), e3 = 1.0f;
    float s = e1 + e2 + e3;
    ASSERT_NEAR(X.at(0, 0), e1 / s, 1e-5);
    ASSERT_NEAR(X.at(0, 1), e2 / s, 1e-5);
    ASSERT_NEAR(X.at(0, 2), e3 / s, 1e-5);
}

// Test 33: Softmax — multiple rows, each sums to 1
void test_softmax_multi_row() {
    auto X = Tensor::dense(3, 3, {
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.0f, 1.0f,
        10.0f, 20.0f, 30.0f
    });
    softmax_inplace(X);

    for (std::size_t i = 0; i < 3; ++i) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < 3; ++j) sum += X.at(i, j);
        ASSERT_NEAR(sum, 1.0f, 1e-5);
    }
}

// Test 34: Softmax — numerical stability with large values
void test_softmax_numerical_stability() {
    //  Without max-subtraction, exp(1000) would overflow.
    //  After subtraction: [0, 1] → exp: [1, e^1] → sum = 1+e
    auto X = Tensor::dense(1, 2, {1000.0f, 1001.0f});
    softmax_inplace(X);

    float expected_0 = 1.0f / (1.0f + std::exp(1.0f));
    float expected_1 = std::exp(1.0f) / (1.0f + std::exp(1.0f));

    ASSERT_NEAR(X.at(0, 0), expected_0, 1e-5);
    ASSERT_NEAR(X.at(0, 1), expected_1, 1e-5);

    // Verify no NaN/Inf
    ASSERT_TRUE(!std::isnan(X.at(0, 0)));
    ASSERT_TRUE(!std::isnan(X.at(0, 1)));
    ASSERT_TRUE(!std::isinf(X.at(0, 0)));
    ASSERT_TRUE(!std::isinf(X.at(0, 1)));
}

// Test 35: Softmax — largest input gets largest probability
void test_softmax_argmax_preserved() {
    auto X = Tensor::dense(1, 5, {1.0f, 5.0f, 2.0f, 4.0f, 3.0f});
    softmax_inplace(X);

    // Element 1 (value 5) should have the highest softmax probability
    float max_prob = X.at(0, 1);
    for (std::size_t j = 0; j < 5; ++j) {
        ASSERT_TRUE(X.at(0, j) <= max_prob + 1e-6f);
    }
}

// ============================================================================
//  Category 8: Log-Softmax  (tests 36–41)
// ============================================================================

// Test 36: Log-softmax — exp of output sums to 1
void test_log_softmax_exp_sums_to_one() {
    auto X = Tensor::dense(1, 4, {1.0f, 2.0f, 3.0f, 4.0f});
    log_softmax_inplace(X);

    float sum = 0.0f;
    for (std::size_t j = 0; j < 4; ++j) sum += std::exp(X.at(0, j));
    ASSERT_NEAR(sum, 1.0f, 1e-4);
}

// Test 37: Log-softmax — all values <= 0
void test_log_softmax_all_nonpositive() {
    auto X = Tensor::dense(1, 4, {1.0f, 2.0f, 3.0f, 4.0f});
    log_softmax_inplace(X);

    for (std::size_t j = 0; j < 4; ++j) {
        ASSERT_TRUE(X.at(0, j) <= 0.0f + 1e-6f);
    }
}

// Test 38: Log-softmax — consistent with softmax (element-wise log)
void test_log_softmax_vs_softmax() {
    auto X_soft = Tensor::dense(1, 4, {1.0f, -2.0f, 3.0f, 0.5f});
    auto X_log  = Tensor::dense(1, 4, {1.0f, -2.0f, 3.0f, 0.5f});

    softmax_inplace(X_soft);
    log_softmax_inplace(X_log);

    for (std::size_t j = 0; j < 4; ++j) {
        ASSERT_NEAR(X_log.at(0, j), std::log(X_soft.at(0, j)), 1e-4);
    }
}

// Test 39: Log-softmax — equal inputs
void test_log_softmax_uniform() {
    //  log(1/4) = -ln(4) ≈ -1.3863
    auto X = Tensor::dense(1, 4, {0.0f, 0.0f, 0.0f, 0.0f});
    log_softmax_inplace(X);

    float expected = -std::log(4.0f);
    for (std::size_t j = 0; j < 4; ++j) {
        ASSERT_NEAR(X.at(0, j), expected, 1e-5);
    }
}

// Test 40: Log-softmax — numerical stability with large values
void test_log_softmax_numerical_stability() {
    auto X = Tensor::dense(1, 3, {1000.0f, 1001.0f, 999.0f});
    log_softmax_inplace(X);

    // Verify no NaN/Inf
    for (std::size_t j = 0; j < 3; ++j) {
        ASSERT_TRUE(!std::isnan(X.at(0, j)));
        ASSERT_TRUE(!std::isinf(X.at(0, j)));
    }

    // exp of output should sum to 1
    float sum = 0.0f;
    for (std::size_t j = 0; j < 3; ++j) sum += std::exp(X.at(0, j));
    ASSERT_NEAR(sum, 1.0f, 1e-4);
}

// Test 41: Log-softmax — multi-row
void test_log_softmax_multi_row() {
    auto X = Tensor::dense(2, 3, {
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.0f, 1.0f
    });
    log_softmax_inplace(X);

    for (std::size_t i = 0; i < 2; ++i) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < 3; ++j) {
            ASSERT_TRUE(X.at(i, j) <= 0.0f + 1e-6f);
            sum += std::exp(X.at(i, j));
        }
        ASSERT_NEAR(sum, 1.0f, 1e-4);
    }
}

// ============================================================================
//  Category 9: Add Bias  (tests 42–47)
// ============================================================================

// Test 42: Add bias — basic broadcasting
void test_add_bias_basic() {
    //  X (3×2):           bias (1×2):      result:
    //  [1, 2]             [10, 20]         [11, 22]
    //  [3, 4]                              [13, 24]
    //  [5, 6]                              [15, 26]
    auto X = Tensor::dense(3, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto bias = Tensor::dense(1, 2, {10.0f, 20.0f});
    add_bias(X, bias);

    ASSERT_FLOAT_EQ(X.at(0, 0), 11.0f);
    ASSERT_FLOAT_EQ(X.at(0, 1), 22.0f);
    ASSERT_FLOAT_EQ(X.at(1, 0), 13.0f);
    ASSERT_FLOAT_EQ(X.at(1, 1), 24.0f);
    ASSERT_FLOAT_EQ(X.at(2, 0), 15.0f);
    ASSERT_FLOAT_EQ(X.at(2, 1), 26.0f);
}

// Test 43: Add bias — zero bias is identity
void test_add_bias_zero() {
    auto X = Tensor::dense(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto bias = Tensor::dense(1, 3, {0.0f, 0.0f, 0.0f});
    add_bias(X, bias);

    ASSERT_FLOAT_EQ(X.at(0, 0), 1.0f);
    ASSERT_FLOAT_EQ(X.at(1, 2), 6.0f);
}

// Test 44: Add bias — negative bias
void test_add_bias_negative() {
    auto X = Tensor::dense(2, 2, {10.0f, 20.0f, 30.0f, 40.0f});
    auto bias = Tensor::dense(1, 2, {-5.0f, -10.0f});
    add_bias(X, bias);

    ASSERT_FLOAT_EQ(X.at(0, 0), 5.0f);
    ASSERT_FLOAT_EQ(X.at(0, 1), 10.0f);
    ASSERT_FLOAT_EQ(X.at(1, 0), 25.0f);
    ASSERT_FLOAT_EQ(X.at(1, 1), 30.0f);
}

// Test 45: Add bias — single row X (edge case)
void test_add_bias_single_row() {
    auto X = Tensor::dense(1, 3, {1.0f, 2.0f, 3.0f});
    auto bias = Tensor::dense(1, 3, {10.0f, 20.0f, 30.0f});
    add_bias(X, bias);

    ASSERT_FLOAT_EQ(X.at(0, 0), 11.0f);
    ASSERT_FLOAT_EQ(X.at(0, 1), 22.0f);
    ASSERT_FLOAT_EQ(X.at(0, 2), 33.0f);
}

// Test 46: Add bias — wrong bias shape (rows != 1)
void test_add_bias_wrong_rows() {
    auto X = Tensor::dense(3, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto bad_bias = Tensor::dense(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});

    ASSERT_THROWS(add_bias(X, bad_bias), std::invalid_argument);
}

// Test 47: Add bias — wrong bias cols
void test_add_bias_wrong_cols() {
    auto X = Tensor::dense(3, 4, std::vector<float>(12, 1.0f));
    auto bad_bias = Tensor::dense(1, 3, {1.0f, 2.0f, 3.0f});

    ASSERT_THROWS_MSG(add_bias(X, bad_bias), std::invalid_argument, "bias must be shape");
}

// ============================================================================
//  Category 10: GNN Pipeline Integration  (tests 48–50)
// ============================================================================

// Test 48: Full GCN layer — H' = ReLU(Adj × H × W + b)
void test_gcn_layer_pipeline() {
    //  3-node graph, 2 features in, 2 features out
    //  Adj (3×3 sparse identity → each node sees only itself)
    auto Adj = Tensor::sparse_csr(3, 3,
        {0, 1, 2, 3}, {0, 1, 2}, {1.0f, 1.0f, 1.0f});

    //  H (3×2): features
    auto H = Tensor::dense(3, 2, {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f});

    //  W (2×2): weights (identity)
    auto W = Tensor::dense(2, 2, {1.0f, 0.0f, 0.0f, 1.0f});

    //  b (1×2): bias
    auto bias = Tensor::dense(1, 2, {0.5f, 0.5f});

    // Step 1: Message passing  H_agg = Adj × H  (identity → H unchanged)
    auto H_agg = spmm(Adj, H);

    // Step 2: Feature transform  H_trans = H_agg × W  (identity → unchanged)
    auto H_trans = matmul(H_agg, W);

    // Step 3: Add bias
    add_bias(H_trans, bias);

    // Step 4: Activation
    relu_inplace(H_trans);

    //  Expected (after bias + relu):
    //  [1+0.5, -1+0.5] → [1.5, 0]   (relu clamps -0.5 to 0)
    //  [2+0.5, -2+0.5] → [2.5, 0]
    //  [3+0.5, -3+0.5] → [3.5, 0]
    ASSERT_FLOAT_EQ(H_trans.at(0, 0), 1.5f);
    ASSERT_FLOAT_EQ(H_trans.at(0, 1), 0.0f);
    ASSERT_FLOAT_EQ(H_trans.at(1, 0), 2.5f);
    ASSERT_FLOAT_EQ(H_trans.at(1, 1), 0.0f);
    ASSERT_FLOAT_EQ(H_trans.at(2, 0), 3.5f);
    ASSERT_FLOAT_EQ(H_trans.at(2, 1), 0.0f);
}

// Test 49: GAT-style attention — LeakyReLU + Softmax
void test_gat_attention_pipeline() {
    //  Attention scores for node 0 attending to 3 neighbors
    //  e = [0.5, -0.3, 1.2]
    //  Step 1: LeakyReLU(e, alpha=0.2)
    //  Step 2: Softmax to get attention weights
    auto scores = Tensor::dense(1, 3, {0.5f, -0.3f, 1.2f});

    leaky_relu_inplace(scores, 0.2f);
    // After leaky relu: [0.5, -0.06, 1.2]
    ASSERT_FLOAT_EQ(scores.at(0, 0), 0.5f);
    ASSERT_NEAR(scores.at(0, 1), -0.06f, 1e-5);
    ASSERT_FLOAT_EQ(scores.at(0, 2), 1.2f);

    softmax_inplace(scores);
    // Verify attention weights sum to 1
    float sum = scores.at(0, 0) + scores.at(0, 1) + scores.at(0, 2);
    ASSERT_NEAR(sum, 1.0f, 1e-5);

    // Largest score (1.2) should get the most attention
    ASSERT_TRUE(scores.at(0, 2) > scores.at(0, 0));
    ASSERT_TRUE(scores.at(0, 0) > scores.at(0, 1));
}

// Test 50: Node classification output — log-softmax for NLL loss
void test_classification_pipeline() {
    //  3 nodes, 4 classes
    //  Raw scores (logits) after final layer
    auto logits = Tensor::dense(3, 4, {
        1.0f, 2.0f, 3.0f, 4.0f,    // node 0: class 3 should win
        5.0f, 1.0f, 2.0f, 0.5f,    // node 1: class 0 should win
        0.1f, 0.1f, 0.1f, 0.1f     // node 2: uniform (all equal)
    });

    log_softmax_inplace(logits);

    // All values should be non-positive (log-probabilities)
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            ASSERT_TRUE(logits.at(i, j) <= 0.0f + 1e-6f);

    // Each row's exp should sum to 1
    for (std::size_t i = 0; i < 3; ++i) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < 4; ++j) sum += std::exp(logits.at(i, j));
        ASSERT_NEAR(sum, 1.0f, 1e-4);
    }

    // Node 0: class 3 (index 3) should have highest log-probability
    ASSERT_TRUE(logits.at(0, 3) > logits.at(0, 0));
    ASSERT_TRUE(logits.at(0, 3) > logits.at(0, 1));
    ASSERT_TRUE(logits.at(0, 3) > logits.at(0, 2));

    // Node 1: class 0 should have highest log-probability
    ASSERT_TRUE(logits.at(1, 0) > logits.at(1, 1));
    ASSERT_TRUE(logits.at(1, 0) > logits.at(1, 2));
    ASSERT_TRUE(logits.at(1, 0) > logits.at(1, 3));

    // Node 2: uniform → all equal → log(1/4) ≈ -1.3863
    float expected = -std::log(4.0f);
    for (std::size_t j = 0; j < 4; ++j)
        ASSERT_NEAR(logits.at(2, j), expected, 1e-5);
}

// ============================================================================
//  Category 11: Error Handling — SparseCSR Rejected  (tests 51–60)
// ============================================================================

// Test 51: ReLU rejects sparse
void test_relu_rejects_sparse() {
    auto S = make_sparse_3x3();
    ASSERT_THROWS(relu_inplace(S), std::invalid_argument);
}

// Test 52: ReLU error message mentions Dense
void test_relu_error_message() {
    auto S = make_sparse_3x3();
    ASSERT_THROWS_MSG(relu_inplace(S), std::invalid_argument, "Dense");
}

// Test 53: Leaky ReLU rejects sparse
void test_leaky_relu_rejects_sparse() {
    auto S = make_sparse_3x3();
    ASSERT_THROWS(leaky_relu_inplace(S), std::invalid_argument);
}

// Test 54: ELU rejects sparse
void test_elu_rejects_sparse() {
    auto S = make_sparse_3x3();
    ASSERT_THROWS(elu_inplace(S), std::invalid_argument);
}

// Test 55: Sigmoid rejects sparse
void test_sigmoid_rejects_sparse() {
    auto S = make_sparse_3x3();
    ASSERT_THROWS(sigmoid_inplace(S), std::invalid_argument);
}

// Test 56: Tanh rejects sparse
void test_tanh_rejects_sparse() {
    auto S = make_sparse_3x3();
    ASSERT_THROWS(tanh_inplace(S), std::invalid_argument);
}

// Test 57: GELU rejects sparse
void test_gelu_rejects_sparse() {
    auto S = make_sparse_3x3();
    ASSERT_THROWS(gelu_inplace(S), std::invalid_argument);
}

// Test 58: Softmax rejects sparse
void test_softmax_rejects_sparse() {
    auto S = make_sparse_3x3();
    ASSERT_THROWS(softmax_inplace(S), std::invalid_argument);
}

// Test 59: Log-softmax rejects sparse
void test_log_softmax_rejects_sparse() {
    auto S = make_sparse_3x3();
    ASSERT_THROWS(log_softmax_inplace(S), std::invalid_argument);
}

// Test 60: Add bias rejects sparse X and sparse bias
void test_add_bias_rejects_sparse() {
    auto S = make_sparse_3x3();
    auto D = Tensor::dense(1, 3, {1.0f, 2.0f, 3.0f});

    // Sparse X
    ASSERT_THROWS_MSG(add_bias(S, D), std::invalid_argument, "Dense");
    // Sparse bias
    auto X = Tensor::dense(3, 3, std::vector<float>(9, 1.0f));
    ASSERT_THROWS_MSG(add_bias(X, S), std::invalid_argument, "bias tensor must be Dense");
}

// ============================================================================
//  Category 12: Stress / Scale Tests  (tests 61–65)
// ============================================================================

// Test 61: ReLU on large tensor (1024×256)
void test_relu_stress() {
    const std::size_t M = 1024, N = 256;
    std::vector<float> data(M * N);
    for (std::size_t i = 0; i < M * N; ++i)
        data[i] = static_cast<float>(static_cast<int>(i % 200) - 100);  // [-100, 99]

    auto X = Tensor::dense(M, N, data);
    relu_inplace(X);

    // All values should be >= 0 after ReLU
    bool all_ok = true;
    for (std::size_t i = 0; i < M * N; ++i)
        if (X.data()[i] < 0.0f) all_ok = false;
    ASSERT_TRUE(all_ok);

    // Count: values 0-99 survive (100 values), values -100 to -1 become 0 (100 values)
    // Positive values:  1..99 (99 per 200-cycle)
    std::size_t positive_count = 0;
    for (std::size_t i = 0; i < M * N; ++i)
        if (X.data()[i] > 0.0f) ++positive_count;

    // Verify manually: in each 200-cycle, values 101..199 (i%200) map to 1..99
    std::size_t expected = 0;
    for (std::size_t i = 0; i < M * N; ++i) {
        int val = static_cast<int>(i % 200) - 100;
        if (val > 0) ++expected;
    }
    ASSERT_EQ(positive_count, expected);
}

// Test 62: Softmax on large tensor (1024×128) — each row sums to 1
void test_softmax_stress() {
    const std::size_t M = 1024, N = 128;
    std::vector<float> data(M * N);
    for (std::size_t i = 0; i < M * N; ++i)
        data[i] = static_cast<float>(i % 100) * 0.1f - 5.0f;

    auto X = Tensor::dense(M, N, data);
    softmax_inplace(X);

    bool all_ok = true;
    for (std::size_t i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < N; ++j) {
            float v = X.at(i, j);
            if (v < 0.0f || v > 1.0f) all_ok = false;
            sum += v;
        }
        if (std::fabs(sum - 1.0f) > 1e-4f) all_ok = false;
    }
    ASSERT_TRUE(all_ok);
}

// Test 63: Sigmoid on large tensor — all outputs in (0, 1)
void test_sigmoid_stress() {
    const std::size_t M = 512, N = 512;
    std::vector<float> data(M * N);
    for (std::size_t i = 0; i < M * N; ++i)
        data[i] = static_cast<float>(static_cast<int>(i % 200) - 100) * 0.5f;

    auto X = Tensor::dense(M, N, data);
    sigmoid_inplace(X);

    bool all_ok = true;
    for (std::size_t i = 0; i < M * N; ++i) {
        float v = X.data()[i];
        if (v < 0.0f || v > 1.0f) all_ok = false;
    }
    ASSERT_TRUE(all_ok);
}

// Test 64: Add bias on large tensor (2708×32, Cora-scale)
void test_add_bias_stress() {
    const std::size_t M = 2708, N = 32;
    std::vector<float> data(M * N, 1.0f);
    std::vector<float> bias_data(N);
    for (std::size_t j = 0; j < N; ++j)
        bias_data[j] = static_cast<float>(j);

    auto X = Tensor::dense(M, N, data);
    auto bias = Tensor::dense(1, N, bias_data);
    add_bias(X, bias);

    // Every row should be [1+0, 1+1, 1+2, ..., 1+31]
    bool all_ok = true;
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
            if (std::fabs(X.at(i, j) - (1.0f + static_cast<float>(j))) > 1e-5f)
                all_ok = false;
    ASSERT_TRUE(all_ok);
}

// Test 65: GELU on large tensor — spot check against reference
void test_gelu_stress() {
    const std::size_t M = 256, N = 256;
    std::vector<float> data(M * N);
    for (std::size_t i = 0; i < M * N; ++i)
        data[i] = static_cast<float>(static_cast<int>(i % 100) - 50) * 0.1f;

    auto X = Tensor::dense(M, N, data);

    // Compute references before applying gelu
    std::vector<float> refs(M * N);
    for (std::size_t i = 0; i < M * N; ++i)
        refs[i] = gelu_ref(data[i]);

    gelu_inplace(X);

    bool all_ok = true;
    for (std::size_t i = 0; i < M * N; ++i)
        if (std::fabs(X.data()[i] - refs[i]) > 1e-4f) all_ok = false;
    ASSERT_TRUE(all_ok);
}

// ============================================================================
//  Category 13: Edge / Degenerate Cases  (tests 66–70)
// ============================================================================

// Test 66: 1×1 tensor through all activations
void test_1x1_all_activations() {
    {
        auto X = Tensor::dense(1, 1, {-5.0f});
        relu_inplace(X);
        ASSERT_FLOAT_EQ(X.at(0, 0), 0.0f);
    }
    {
        auto X = Tensor::dense(1, 1, {-5.0f});
        leaky_relu_inplace(X, 0.1f);
        ASSERT_FLOAT_EQ(X.at(0, 0), -0.5f);
    }
    {
        auto X = Tensor::dense(1, 1, {0.0f});
        sigmoid_inplace(X);
        ASSERT_NEAR(X.at(0, 0), 0.5f, 1e-5);
    }
    {
        auto X = Tensor::dense(1, 1, {0.0f});
        tanh_inplace(X);
        ASSERT_NEAR(X.at(0, 0), 0.0f, 1e-5);
    }
    {
        auto X = Tensor::dense(1, 1, {0.0f});
        gelu_inplace(X);
        ASSERT_NEAR(X.at(0, 0), 0.0f, 1e-5);
    }
    {
        auto X = Tensor::dense(1, 1, {5.0f});
        softmax_inplace(X);
        ASSERT_NEAR(X.at(0, 0), 1.0f, 1e-5);
    }
    {
        auto X = Tensor::dense(1, 1, {5.0f});
        log_softmax_inplace(X);
        ASSERT_NEAR(X.at(0, 0), 0.0f, 1e-5);
    }
}

// Test 67: Softmax rejects zero-column tensor
void test_softmax_zero_cols() {
    auto X = Tensor::dense(3, 0);
    ASSERT_THROWS_MSG(softmax_inplace(X), std::invalid_argument, "0 columns");
}

// Test 68: Log-softmax rejects zero-column tensor
void test_log_softmax_zero_cols() {
    auto X = Tensor::dense(3, 0);
    ASSERT_THROWS_MSG(log_softmax_inplace(X), std::invalid_argument, "0 columns");
}

// Test 69: Activations on zero-row tensor are no-ops
void test_zero_row_activations() {
    {
        auto X = Tensor::dense(0, 5);
        relu_inplace(X);
        ASSERT_EQ(X.rows(), std::size_t(0));
        ASSERT_EQ(X.cols(), std::size_t(5));
    }
    {
        auto X = Tensor::dense(0, 5);
        sigmoid_inplace(X);
        ASSERT_EQ(X.data().size(), std::size_t(0));
    }
}

// Test 70: Double-apply idempotency and composition
void test_double_apply() {
    // ReLU is idempotent: relu(relu(X)) = relu(X)
    auto X1 = Tensor::dense(1, 4, {-2.0f, -1.0f, 1.0f, 2.0f});
    relu_inplace(X1);
    auto after_first = std::vector<float>(X1.data().begin(), X1.data().end());
    relu_inplace(X1);

    for (std::size_t j = 0; j < 4; ++j)
        ASSERT_FLOAT_EQ(X1.at(0, j), after_first[j]);

    // Softmax is NOT idempotent, but second softmax is still valid:
    // each row still sums to 1 and values in [0,1]
    auto X2 = Tensor::dense(1, 3, {1.0f, 2.0f, 3.0f});
    softmax_inplace(X2);
    softmax_inplace(X2);

    float sum = 0.0f;
    for (std::size_t j = 0; j < 3; ++j) {
        ASSERT_TRUE(X2.at(0, j) >= 0.0f);
        ASSERT_TRUE(X2.at(0, j) <= 1.0f);
        sum += X2.at(0, j);
    }
    ASSERT_NEAR(sum, 1.0f, 1e-5);
}

// ============================================================================
//  Main
// ============================================================================

int main() {
    std::cout << "\n"
        "+=================================================================+\n"
        "|   TinyGNN — Activations & Utilities Unit Tests (Phase 5)       |\n"
        "|   Testing: relu, leaky_relu, elu, sigmoid, tanh, gelu,         |\n"
        "|            softmax, log_softmax, add_bias                       |\n"
        "+=================================================================+\n\n";

    // Category 1: ReLU
    std::cout << "── 1. ReLU ─────────────────────────────────────────────────\n";
    std::cout << "  Running test_relu_basic...\n";                test_relu_basic();
    std::cout << "  Running test_relu_2d_matrix...\n";            test_relu_2d_matrix();
    std::cout << "  Running test_relu_all_positive...\n";         test_relu_all_positive();
    std::cout << "  Running test_relu_all_negative...\n";         test_relu_all_negative();
    std::cout << "  Running test_relu_preserves_zero...\n";       test_relu_preserves_zero();

    // Category 2: Leaky ReLU
    std::cout << "\n── 2. Leaky ReLU ───────────────────────────────────────────\n";
    std::cout << "  Running test_leaky_relu_default_alpha...\n";  test_leaky_relu_default_alpha();
    std::cout << "  Running test_leaky_relu_gat_alpha...\n";      test_leaky_relu_gat_alpha();
    std::cout << "  Running test_leaky_relu_2d...\n";             test_leaky_relu_2d();
    std::cout << "  Running test_leaky_relu_alpha_zero_is_relu...\n"; test_leaky_relu_alpha_zero_is_relu();
    std::cout << "  Running test_leaky_relu_alpha_one_is_identity...\n"; test_leaky_relu_alpha_one_is_identity();

    // Category 3: ELU
    std::cout << "\n── 3. ELU ──────────────────────────────────────────────────\n";
    std::cout << "  Running test_elu_basic...\n";                 test_elu_basic();
    std::cout << "  Running test_elu_custom_alpha...\n";          test_elu_custom_alpha();
    std::cout << "  Running test_elu_saturation...\n";            test_elu_saturation();
    std::cout << "  Running test_elu_2d_matrix...\n";             test_elu_2d_matrix();

    // Category 4: Sigmoid
    std::cout << "\n── 4. Sigmoid ──────────────────────────────────────────────\n";
    std::cout << "  Running test_sigmoid_basic...\n";             test_sigmoid_basic();
    std::cout << "  Running test_sigmoid_symmetry...\n";          test_sigmoid_symmetry();
    std::cout << "  Running test_sigmoid_output_bounds...\n";     test_sigmoid_output_bounds();
    std::cout << "  Running test_sigmoid_hand_calculated...\n";   test_sigmoid_hand_calculated();
    std::cout << "  Running test_sigmoid_2d_matrix...\n";         test_sigmoid_2d_matrix();

    // Category 5: Tanh
    std::cout << "\n── 5. Tanh ─────────────────────────────────────────────────\n";
    std::cout << "  Running test_tanh_basic...\n";                test_tanh_basic();
    std::cout << "  Running test_tanh_symmetry...\n";             test_tanh_symmetry();
    std::cout << "  Running test_tanh_output_bounds...\n";        test_tanh_output_bounds();
    std::cout << "  Running test_tanh_hand_calculated...\n";      test_tanh_hand_calculated();
    std::cout << "  Running test_tanh_2d_matrix...\n";            test_tanh_2d_matrix();

    // Category 6: GELU
    std::cout << "\n── 6. GELU ─────────────────────────────────────────────────\n";
    std::cout << "  Running test_gelu_basic...\n";                test_gelu_basic();
    std::cout << "  Running test_gelu_hand_calculated...\n";      test_gelu_hand_calculated();
    std::cout << "  Running test_gelu_zero_symmetry...\n";        test_gelu_zero_symmetry();
    std::cout << "  Running test_gelu_2d_matrix...\n";            test_gelu_2d_matrix();

    // Category 7: Softmax
    std::cout << "\n── 7. Softmax ──────────────────────────────────────────────\n";
    std::cout << "  Running test_softmax_row_sums_to_one...\n";   test_softmax_row_sums_to_one();
    std::cout << "  Running test_softmax_output_bounds...\n";     test_softmax_output_bounds();
    std::cout << "  Running test_softmax_uniform...\n";           test_softmax_uniform();
    std::cout << "  Running test_softmax_hand_calculated...\n";   test_softmax_hand_calculated();
    std::cout << "  Running test_softmax_multi_row...\n";         test_softmax_multi_row();
    std::cout << "  Running test_softmax_numerical_stability...\n"; test_softmax_numerical_stability();
    std::cout << "  Running test_softmax_argmax_preserved...\n";  test_softmax_argmax_preserved();

    // Category 8: Log-Softmax
    std::cout << "\n── 8. Log-Softmax ──────────────────────────────────────────\n";
    std::cout << "  Running test_log_softmax_exp_sums_to_one...\n"; test_log_softmax_exp_sums_to_one();
    std::cout << "  Running test_log_softmax_all_nonpositive...\n"; test_log_softmax_all_nonpositive();
    std::cout << "  Running test_log_softmax_vs_softmax...\n";    test_log_softmax_vs_softmax();
    std::cout << "  Running test_log_softmax_uniform...\n";       test_log_softmax_uniform();
    std::cout << "  Running test_log_softmax_numerical_stability...\n"; test_log_softmax_numerical_stability();
    std::cout << "  Running test_log_softmax_multi_row...\n";     test_log_softmax_multi_row();

    // Category 9: Add Bias
    std::cout << "\n── 9. Add Bias ─────────────────────────────────────────────\n";
    std::cout << "  Running test_add_bias_basic...\n";            test_add_bias_basic();
    std::cout << "  Running test_add_bias_zero...\n";             test_add_bias_zero();
    std::cout << "  Running test_add_bias_negative...\n";         test_add_bias_negative();
    std::cout << "  Running test_add_bias_single_row...\n";       test_add_bias_single_row();
    std::cout << "  Running test_add_bias_wrong_rows...\n";       test_add_bias_wrong_rows();
    std::cout << "  Running test_add_bias_wrong_cols...\n";       test_add_bias_wrong_cols();

    // Category 10: GNN Pipeline Integration
    std::cout << "\n── 10. GNN Pipeline Integration ────────────────────────────\n";
    std::cout << "  Running test_gcn_layer_pipeline...\n";        test_gcn_layer_pipeline();
    std::cout << "  Running test_gat_attention_pipeline...\n";    test_gat_attention_pipeline();
    std::cout << "  Running test_classification_pipeline...\n";   test_classification_pipeline();

    // Category 11: Error Handling
    std::cout << "\n── 11. Error Handling — SparseCSR Rejected ─────────────────\n";
    std::cout << "  Running test_relu_rejects_sparse...\n";       test_relu_rejects_sparse();
    std::cout << "  Running test_relu_error_message...\n";        test_relu_error_message();
    std::cout << "  Running test_leaky_relu_rejects_sparse...\n"; test_leaky_relu_rejects_sparse();
    std::cout << "  Running test_elu_rejects_sparse...\n";        test_elu_rejects_sparse();
    std::cout << "  Running test_sigmoid_rejects_sparse...\n";    test_sigmoid_rejects_sparse();
    std::cout << "  Running test_tanh_rejects_sparse...\n";       test_tanh_rejects_sparse();
    std::cout << "  Running test_gelu_rejects_sparse...\n";       test_gelu_rejects_sparse();
    std::cout << "  Running test_softmax_rejects_sparse...\n";    test_softmax_rejects_sparse();
    std::cout << "  Running test_log_softmax_rejects_sparse...\n"; test_log_softmax_rejects_sparse();
    std::cout << "  Running test_add_bias_rejects_sparse...\n";   test_add_bias_rejects_sparse();

    // Category 12: Stress / Scale
    std::cout << "\n── 12. Stress / Scale Tests ────────────────────────────────\n";
    std::cout << "  Running test_relu_stress...\n";               test_relu_stress();
    std::cout << "  Running test_softmax_stress...\n";            test_softmax_stress();
    std::cout << "  Running test_sigmoid_stress...\n";            test_sigmoid_stress();
    std::cout << "  Running test_add_bias_stress...\n";           test_add_bias_stress();
    std::cout << "  Running test_gelu_stress...\n";               test_gelu_stress();

    // Category 13: Edge / Degenerate Cases
    std::cout << "\n── 13. Edge / Degenerate Cases ─────────────────────────────\n";
    std::cout << "  Running test_1x1_all_activations...\n";       test_1x1_all_activations();
    std::cout << "  Running test_softmax_zero_cols...\n";         test_softmax_zero_cols();
    std::cout << "  Running test_log_softmax_zero_cols...\n";     test_log_softmax_zero_cols();
    std::cout << "  Running test_zero_row_activations...\n";      test_zero_row_activations();
    std::cout << "  Running test_double_apply...\n";              test_double_apply();

    // ── Summary ──────────────────────────────────────────────────────────────
    std::cout << "\n"
        "=================================================================\n"
        "  Total : " << g_tests_run << "\n"
        "  Passed: " << g_tests_passed << "\n"
        "  Failed: " << g_tests_failed << "\n"
        "=================================================================\n\n";

    return g_tests_failed == 0 ? 0 : 1;
}
