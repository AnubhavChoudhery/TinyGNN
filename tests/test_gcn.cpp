// ============================================================================
//  TinyGNN — GCN Layer Unit Tests  (Phase 6, Part 1)
//  tests/test_gcn.cpp
//
//  Test categories:
//    1.  add_self_loops — basic, already-has-diag, empty      (tests  1 –  7)
//    2.  gcn_norm — hand-verified normalization values         (tests  8 – 14)
//    3.  GCNLayer construction and weight management           (tests 15 – 22)
//    4.  GCNLayer forward — 3-node hand-computed               (tests 23 – 28)
//    5.  GCNLayer forward — no bias, no activation             (tests 29 – 33)
//    6.  Two-layer GCN stacking                                (tests 34 – 37)
//    7.  10-node ring graph — full pipeline                    (tests 38 – 48)
//    8.  Error handling                                        (tests 49 – 60)
//    9.  Edge / degenerate cases                               (tests 61 – 68)
//
//  Hand-computed values verified against the GCN formula:
//    H' = σ( D̃^{-1/2} (A+I) D̃^{-1/2} · H · W + b )
//
//  All floating-point comparisons use tolerance ≤ 1e-5.
// ============================================================================

#include "tinygnn/layers.hpp"
#include "tinygnn/ops.hpp"
#include "tinygnn/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ── Minimal test framework (same idiom as other TinyGNN tests) ──────────────

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

// ── Helper: build a small undirected graph in CSR format ────────────────────
// edges: list of (src, dst) pairs for a directed graph
// N: number of nodes
static Tensor make_csr(std::size_t N,
                       const std::vector<std::pair<int,int>>& edges) {
    // Count edges per row
    std::vector<int32_t> rp(N + 1, 0);
    for (auto& [s, d] : edges) {
        rp[s + 1]++;
    }
    for (std::size_t i = 1; i <= N; ++i) rp[i] += rp[i - 1];

    std::vector<int32_t> ci(edges.size());
    std::vector<float>   vals(edges.size(), 1.0f);

    // Use a copy of rp as write pointers
    std::vector<int32_t> write_pos(rp.begin(), rp.end());
    for (auto& [s, d] : edges) {
        ci[write_pos[s]++] = d;
    }

    // Sort col_ind within each row (required for CSR)
    for (std::size_t i = 0; i < N; ++i) {
        std::sort(ci.begin() + rp[i], ci.begin() + rp[i + 1]);
    }

    return Tensor::sparse_csr(N, N,
                              std::move(rp), std::move(ci), std::move(vals));
}

// Helper: make an undirected CSR from undirected edges
static Tensor make_undirected_csr(std::size_t N,
                                  const std::vector<std::pair<int,int>>& edges) {
    std::vector<std::pair<int,int>> directed;
    for (auto& [u, v] : edges) {
        directed.push_back({u, v});
        directed.push_back({v, u});
    }
    return make_csr(N, directed);
}


// ============================================================================
//  Category 1: add_self_loops
// ============================================================================

// Test 1: Simple 3-node path graph  0 - 1 - 2
void test_add_self_loops_basic() {
    // Adjacency (undirected): 0↔1, 1↔2
    auto A = make_undirected_csr(3, {{0,1}, {1,2}});
    ASSERT_EQ(A.nnz(), 4u);  // 4 directed edges

    auto A_hat = add_self_loops(A);
    ASSERT_EQ(A_hat.rows(), 3u);
    ASSERT_EQ(A_hat.cols(), 3u);
    ASSERT_EQ(A_hat.nnz(), 7u);  // 4 edges + 3 self-loops

    // Check diagonal entries exist with value 1.0
    const auto& rp = A_hat.row_ptr();
    const auto& ci = A_hat.col_ind();
    const auto& v  = A_hat.data();

    for (int i = 0; i < 3; ++i) {
        bool found = false;
        for (int32_t nz = rp[i]; nz < rp[i + 1]; ++nz) {
            if (ci[nz] == i) {
                ASSERT_NEAR(v[nz], 1.0f, 1e-6);
                found = true;
            }
        }
        ASSERT_TRUE(found);
    }
}

// Test 2: Graph that already has self-loops → values should be incremented
void test_add_self_loops_existing_diag() {
    // A = [[2, 1], [1, 3]]  — already has diagonal entries
    auto A = Tensor::sparse_csr(2, 2,
        {0, 2, 4},
        {0, 1, 0, 1},
        {2.0f, 1.0f, 1.0f, 3.0f});

    auto A_hat = add_self_loops(A);
    ASSERT_EQ(A_hat.nnz(), 4u);  // same structure

    // Diagonal should be original + 1.0
    const auto& v = A_hat.data();
    (void)A_hat.col_ind();
    (void)A_hat.row_ptr();

    // Row 0: entries at cols [0, 1], values should be [3.0, 1.0]
    ASSERT_NEAR(v[0], 3.0f, 1e-6);  // 2.0 + 1.0
    ASSERT_NEAR(v[1], 1.0f, 1e-6);  // unchanged

    // Row 1: entries at cols [0, 1], values should be [1.0, 4.0]
    ASSERT_NEAR(v[2], 1.0f, 1e-6);  // unchanged
    ASSERT_NEAR(v[3], 4.0f, 1e-6);  // 3.0 + 1.0
}

// Test 3: Empty graph (no edges) — self-loops create identity
void test_add_self_loops_empty() {
    auto A = Tensor::sparse_csr(3, 3,
        {0, 0, 0, 0}, {}, {});

    auto A_hat = add_self_loops(A);
    ASSERT_EQ(A_hat.nnz(), 3u);  // 3 diagonal entries

    const auto& ci = A_hat.col_ind();
    const auto& v  = A_hat.data();
    for (int i = 0; i < 3; ++i) {
        ASSERT_EQ(ci[i], i);
        ASSERT_NEAR(v[i], 1.0f, 1e-6);
    }
}

// Test 4: Single node
void test_add_self_loops_single_node() {
    auto A = Tensor::sparse_csr(1, 1, {0, 0}, {}, {});
    auto A_hat = add_self_loops(A);
    ASSERT_EQ(A_hat.nnz(), 1u);
    ASSERT_NEAR(A_hat.data()[0], 1.0f, 1e-6);
    ASSERT_EQ(A_hat.col_ind()[0], 0);
}

// Test 5: Sorted col_ind is maintained
void test_add_self_loops_sorted_order() {
    // Row 0 connects to col 2 only (no self-loop, no col < 0)
    // Row 1 connects to col 0 and col 2 (no self-loop)
    // Row 2 connects to col 0 only
    auto A = Tensor::sparse_csr(3, 3,
        {0, 1, 3, 4},
        {2, 0, 2, 0},
        {1.0f, 1.0f, 1.0f, 1.0f});

    auto A_hat = add_self_loops(A);

    // Check sorted order within each row
    const auto& rp = A_hat.row_ptr();
    const auto& ci = A_hat.col_ind();
    for (int i = 0; i < 3; ++i) {
        for (int32_t nz = rp[i]; nz < rp[i + 1] - 1; ++nz) {
            ASSERT_TRUE(ci[nz] <= ci[nz + 1]);
        }
    }
}

// Test 6: add_self_loops rejects Dense input
void test_add_self_loops_rejects_dense() {
    auto D = Tensor::dense(3, 3);
    ASSERT_THROWS(add_self_loops(D), std::invalid_argument);
}

// Test 7: add_self_loops rejects non-square
void test_add_self_loops_rejects_nonsquare() {
    auto A = Tensor::sparse_csr(3, 4,
        {0, 1, 2, 3}, {0, 1, 2}, {1.0f, 1.0f, 1.0f});
    ASSERT_THROWS(add_self_loops(A), std::invalid_argument);
}


// ============================================================================
//  Category 2: gcn_norm
// ============================================================================

// Test 8: 3-node path graph (0-1-2)
//   A = [[0,1,0],[1,0,1],[0,1,0]]
//   Ã = A + I = [[1,1,0],[1,1,1],[0,1,1]]
//   deg = [2, 3, 2]
//   Â[i][j] = Ã[i][j] / (√deg[i] * √deg[j])
//
//   Â[0][0] = 1/(√2·√2) = 0.5
//   Â[0][1] = 1/(√2·√3) = 1/√6 ≈ 0.408248
//   Â[1][0] = 1/(√3·√2) = 1/√6 ≈ 0.408248
//   Â[1][1] = 1/(√3·√3) = 1/3  ≈ 0.333333
//   Â[1][2] = 1/(√3·√2) = 1/√6 ≈ 0.408248
//   Â[2][1] = 1/(√2·√3) = 1/√6 ≈ 0.408248
//   Â[2][2] = 1/(√2·√2) = 0.5
void test_gcn_norm_3node_path() {
    auto A = make_undirected_csr(3, {{0,1}, {1,2}});
    auto A_norm = gcn_norm(A);

    ASSERT_EQ(A_norm.rows(), 3u);
    ASSERT_EQ(A_norm.cols(), 3u);
    ASSERT_EQ(A_norm.nnz(), 7u);

    const auto& rp = A_norm.row_ptr();
    const auto& ci = A_norm.col_ind();
    const auto& v  = A_norm.data();

    const double inv_sqrt6 = 1.0 / std::sqrt(6.0);

    // Row 0: cols [0, 1]
    ASSERT_EQ(rp[0], 0);
    ASSERT_EQ(ci[0], 0);
    ASSERT_NEAR(v[0], 0.5f, 1e-5);
    ASSERT_EQ(ci[1], 1);
    ASSERT_NEAR(v[1], static_cast<float>(inv_sqrt6), 1e-5);

    // Row 1: cols [0, 1, 2]
    ASSERT_EQ(ci[2], 0);
    ASSERT_NEAR(v[2], static_cast<float>(inv_sqrt6), 1e-5);
    ASSERT_EQ(ci[3], 1);
    ASSERT_NEAR(v[3], 1.0f / 3.0f, 1e-5);
    ASSERT_EQ(ci[4], 2);
    ASSERT_NEAR(v[4], static_cast<float>(inv_sqrt6), 1e-5);

    // Row 2: cols [1, 2]
    ASSERT_EQ(ci[5], 1);
    ASSERT_NEAR(v[5], static_cast<float>(inv_sqrt6), 1e-5);
    ASSERT_EQ(ci[6], 2);
    ASSERT_NEAR(v[6], 0.5f, 1e-5);
}

// Test 9: Complete graph K3
//   A = [[0,1,1],[1,0,1],[1,1,0]]
//   Ã = [[1,1,1],[1,1,1],[1,1,1]]
//   deg = [3, 3, 3]
//   Â[i][j] = 1/(√3·√3) = 1/3 for all entries
void test_gcn_norm_complete_graph() {
    auto A = make_undirected_csr(3, {{0,1}, {0,2}, {1,2}});
    auto A_norm = gcn_norm(A);

    ASSERT_EQ(A_norm.nnz(), 9u);  // 3×3 dense
    const auto& v = A_norm.data();
    for (std::size_t i = 0; i < 9; ++i) {
        ASSERT_NEAR(v[i], 1.0f / 3.0f, 1e-5);
    }
}

// Test 10: Isolated node (degree 0 after self-loop → degree 1)
//   A = [[0,0],[0,0]] (2 isolated nodes)
//   Ã = [[1,0],[0,1]] (identity)
//   deg = [1, 1]
//   Â[i][j] = Ã[i][j] / (1·1) = identity
void test_gcn_norm_isolated_nodes() {
    auto A = Tensor::sparse_csr(2, 2, {0, 0, 0}, {}, {});
    auto A_norm = gcn_norm(A);

    ASSERT_EQ(A_norm.nnz(), 2u);
    ASSERT_NEAR(A_norm.data()[0], 1.0f, 1e-5);
    ASSERT_NEAR(A_norm.data()[1], 1.0f, 1e-5);
}

// Test 11: Single node with no edges
void test_gcn_norm_single_node() {
    auto A = Tensor::sparse_csr(1, 1, {0, 0}, {}, {});
    auto A_norm = gcn_norm(A);
    ASSERT_EQ(A_norm.nnz(), 1u);
    ASSERT_NEAR(A_norm.data()[0], 1.0f, 1e-5);
}

// Test 12: Row sums of normalized adjacency (property test)
//   For symmetric normalization, row sums are not necessarily 1
//   but should be within [0, sqrt(N)] for reasonable graphs.
void test_gcn_norm_row_sums_bounded() {
    auto A = make_undirected_csr(5, {{0,1},{1,2},{2,3},{3,4},{0,4}});
    auto A_norm = gcn_norm(A);

    const auto& rp = A_norm.row_ptr();
    const auto& v  = A_norm.data();

    for (std::size_t i = 0; i < 5; ++i) {
        float row_sum = 0.0f;
        for (int32_t nz = rp[i]; nz < rp[i + 1]; ++nz) {
            row_sum += v[nz];
        }
        ASSERT_TRUE(row_sum > 0.0f);
        ASSERT_TRUE(row_sum < 5.0f);
    }
}

// Test 13: gcn_norm rejects Dense
void test_gcn_norm_rejects_dense() {
    auto D = Tensor::dense(3, 3);
    ASSERT_THROWS(gcn_norm(D), std::invalid_argument);
}

// Test 14: gcn_norm rejects non-square
void test_gcn_norm_rejects_nonsquare() {
    auto A = Tensor::sparse_csr(3, 4,
        {0, 1, 2, 3}, {0, 1, 2}, {1.0f, 1.0f, 1.0f});
    ASSERT_THROWS(gcn_norm(A), std::invalid_argument);
}


// ============================================================================
//  Category 3: GCNLayer construction and weight management
// ============================================================================

// Test 15: Basic construction
void test_gcn_layer_construction() {
    GCNLayer layer(16, 32);
    ASSERT_EQ(layer.in_features(), 16u);
    ASSERT_EQ(layer.out_features(), 32u);
    ASSERT_TRUE(layer.has_bias());
    ASSERT_TRUE(layer.activation() == Activation::ReLU);
    ASSERT_EQ(layer.weight().rows(), 16u);
    ASSERT_EQ(layer.weight().cols(), 32u);
    ASSERT_EQ(layer.bias().rows(), 1u);
    ASSERT_EQ(layer.bias().cols(), 32u);
}

// Test 16: Construction without bias
void test_gcn_layer_no_bias() {
    GCNLayer layer(8, 4, false, Activation::None);
    ASSERT_TRUE(!layer.has_bias());
    ASSERT_TRUE(layer.activation() == Activation::None);
}

// Test 17: set_weight
void test_gcn_layer_set_weight() {
    GCNLayer layer(2, 3);
    auto W = Tensor::dense(2, 3, {1,2,3,4,5,6});
    layer.set_weight(std::move(W));
    ASSERT_NEAR(layer.weight().at(0, 0), 1.0f, 1e-6);
    ASSERT_NEAR(layer.weight().at(1, 2), 6.0f, 1e-6);
}

// Test 18: set_bias
void test_gcn_layer_set_bias() {
    GCNLayer layer(2, 3);
    auto b = Tensor::dense(1, 3, {0.1f, 0.2f, 0.3f});
    layer.set_bias(std::move(b));
    ASSERT_NEAR(layer.bias().at(0, 0), 0.1f, 1e-6);
    ASSERT_NEAR(layer.bias().at(0, 2), 0.3f, 1e-6);
}

// Test 19: set_weight wrong shape
void test_gcn_layer_set_weight_wrong_shape() {
    GCNLayer layer(4, 8);
    auto W = Tensor::dense(3, 8);  // wrong rows
    ASSERT_THROWS(layer.set_weight(std::move(W)), std::invalid_argument);
}

// Test 20: set_bias when use_bias=false
void test_gcn_layer_set_bias_no_bias() {
    GCNLayer layer(4, 8, false);
    auto b = Tensor::dense(1, 8);
    ASSERT_THROWS(layer.set_bias(std::move(b)), std::invalid_argument);
}

// Test 21: set_bias wrong shape
void test_gcn_layer_set_bias_wrong_shape() {
    GCNLayer layer(4, 8);
    auto b = Tensor::dense(1, 4);  // wrong cols
    ASSERT_THROWS(layer.set_bias(std::move(b)), std::invalid_argument);
}

// Test 22: Zero in/out features rejected
void test_gcn_layer_zero_features() {
    ASSERT_THROWS(GCNLayer(0, 8), std::invalid_argument);
    ASSERT_THROWS(GCNLayer(8, 0), std::invalid_argument);
}


// ============================================================================
//  Category 4: GCNLayer forward — 3-node hand-computed
// ============================================================================
//
//  Graph: 0 -- 1 -- 2  (path graph)
//  Adjacency (undirected, no self-loops):
//    A = [[0,1,0],[1,0,1],[0,1,0]]
//
//  After gcn_norm:
//    Â[0][0]=0.5  Â[0][1]=1/√6  Â[1][0]=1/√6  Â[1][1]=1/3
//    Â[1][2]=1/√6  Â[2][1]=1/√6  Â[2][2]=0.5
//
//  H = [[1.0, 0.0],
//       [0.0, 1.0],
//       [1.0, 1.0]]
//
//  W = [[1.0, 0.5],
//       [0.5, 1.0]]
//
//  bias = [0.1, -0.1]
//
//  Step 1: HW = H * W
//    HW[0] = [1.0, 0.5]
//    HW[1] = [0.5, 1.0]
//    HW[2] = [1.5, 1.5]
//
//  Step 2: out = Â * HW
//    out[0] = 0.5*[1.0,0.5] + (1/√6)*[0.5,1.0]
//           = [0.5+0.204124, 0.25+0.408248]
//           = [0.704124, 0.658248]
//
//    out[1] = (1/√6)*[1.0,0.5] + (1/3)*[0.5,1.0] + (1/√6)*[1.5,1.5]
//           = [0.408248+0.166667+0.612372, 0.204124+0.333333+0.612372]
//           = [1.187287, 1.149829]
//
//    out[2] = (1/√6)*[0.5,1.0] + 0.5*[1.5,1.5]
//           = [0.204124+0.75, 0.408248+0.75]
//           = [0.954124, 1.158248]
//
//  Step 3: out += bias
//    out[0] = [0.804124, 0.558248]
//    out[1] = [1.287287, 1.049829]
//    out[2] = [1.054124, 1.058248]
//
//  Step 4: ReLU (all positive → no change)
//

// Test 23: Full forward pass matches hand-computed values
void test_gcn_forward_3node() {
    // Build adjacency
    auto A = make_undirected_csr(3, {{0,1}, {1,2}});
    auto A_norm = gcn_norm(A);

    // Node features (3 × 2)
    auto H = Tensor::dense(3, 2, {1.0f, 0.0f,
                                   0.0f, 1.0f,
                                   1.0f, 1.0f});

    // Layer: 2 → 2
    GCNLayer layer(2, 2, true, Activation::ReLU);
    layer.set_weight(Tensor::dense(2, 2, {1.0f, 0.5f,
                                           0.5f, 1.0f}));
    layer.set_bias(Tensor::dense(1, 2, {0.1f, -0.1f}));

    auto out = layer.forward(A_norm, H);

    ASSERT_EQ(out.rows(), 3u);
    ASSERT_EQ(out.cols(), 2u);

    // Check against hand-computed values
    ASSERT_NEAR(out.at(0, 0), 0.804124f, 1e-4);
    ASSERT_NEAR(out.at(0, 1), 0.558248f, 1e-4);
    ASSERT_NEAR(out.at(1, 0), 1.287287f, 1e-4);
    ASSERT_NEAR(out.at(1, 1), 1.049829f, 1e-4);
    ASSERT_NEAR(out.at(2, 0), 1.054124f, 1e-4);
    ASSERT_NEAR(out.at(2, 1), 1.058248f, 1e-4);
}

// Test 24: Output dimensions correct
void test_gcn_forward_output_shape() {
    auto A = make_undirected_csr(3, {{0,1}, {1,2}});
    auto A_norm = gcn_norm(A);
    auto H = Tensor::dense(3, 4);   // 3 nodes, 4 features

    GCNLayer layer(4, 8);
    auto out = layer.forward(A_norm, H);

    ASSERT_EQ(out.rows(), 3u);
    ASSERT_EQ(out.cols(), 8u);
    ASSERT_TRUE(out.format() == StorageFormat::Dense);
}

// Test 25: With identity weight (W = I), no bias
//   Out should be spmm(A_norm, H)
void test_gcn_forward_identity_weight() {
    auto A = make_undirected_csr(3, {{0,1}, {1,2}});
    auto A_norm = gcn_norm(A);
    auto H = Tensor::dense(3, 2, {1.0f, 0.0f,
                                   0.0f, 1.0f,
                                   1.0f, 1.0f});

    GCNLayer layer(2, 2, false, Activation::None);
    layer.set_weight(Tensor::dense(2, 2, {1.0f, 0.0f,
                                           0.0f, 1.0f}));

    auto out = layer.forward(A_norm, H);

    // With W=I and no bias, no activation: out = Â * H
    auto expected = spmm(A_norm, H);
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 2; ++c) {
            ASSERT_NEAR(out.at(r, c), expected.at(r, c), 1e-5);
        }
    }
}

// Test 26: ReLU actually clips negative values
void test_gcn_forward_relu_clips() {
    auto A = make_undirected_csr(2, {{0,1}});
    auto A_norm = gcn_norm(A);

    // H with values that produce negative output
    auto H = Tensor::dense(2, 1, {1.0f, 1.0f});

    GCNLayer layer(1, 1, true, Activation::ReLU);
    layer.set_weight(Tensor::dense(1, 1, {1.0f}));
    layer.set_bias(Tensor::dense(1, 1, {-10.0f}));  // large negative bias

    auto out = layer.forward(A_norm, H);

    // After large negative bias, ReLU should clip to 0
    ASSERT_NEAR(out.at(0, 0), 0.0f, 1e-6);
    ASSERT_NEAR(out.at(1, 0), 0.0f, 1e-6);
}

// Test 27: Zero weight → output is just bias (or 0 if no bias)
void test_gcn_forward_zero_weight() {
    auto A = make_undirected_csr(3, {{0,1}, {1,2}});
    auto A_norm = gcn_norm(A);
    auto H = Tensor::dense(3, 2, {1,2,3,4,5,6});

    GCNLayer layer(2, 2, true, Activation::None);
    // Weight is zero-initialized by default
    layer.set_bias(Tensor::dense(1, 2, {0.5f, -0.5f}));

    auto out = layer.forward(A_norm, H);

    // matmul(H, 0) = 0, spmm(A_norm, 0) = 0, + bias = bias
    for (std::size_t r = 0; r < 3; ++r) {
        ASSERT_NEAR(out.at(r, 0),  0.5f, 1e-6);
        ASSERT_NEAR(out.at(r, 1), -0.5f, 1e-6);
    }
}

// Test 28: Verify reuse of A_norm (same norm matrix, different features)
void test_gcn_forward_reuse_norm() {
    auto A = make_undirected_csr(3, {{0,1}, {1,2}});
    auto A_norm = gcn_norm(A);

    GCNLayer layer(2, 2, false, Activation::None);
    layer.set_weight(Tensor::dense(2, 2, {1,0, 0,1}));

    auto H1 = Tensor::dense(3, 2, {1,0, 0,1, 1,1});
    auto H2 = Tensor::dense(3, 2, {2,0, 0,2, 2,2});

    auto out1 = layer.forward(A_norm, H1);
    auto out2 = layer.forward(A_norm, H2);

    // H2 = 2*H1, so out2 should be 2*out1 (linear layer, no activation)
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 2; ++c) {
            ASSERT_NEAR(out2.at(r, c), 2.0f * out1.at(r, c), 1e-5);
        }
    }
}


// ============================================================================
//  Category 5: GCNLayer forward — no bias, no activation variants
// ============================================================================

// Test 29: No bias mode
void test_gcn_forward_no_bias() {
    auto A = make_undirected_csr(3, {{0,1}, {1,2}});
    auto A_norm = gcn_norm(A);
    auto H = Tensor::dense(3, 2, {1,0, 0,1, 1,1});

    GCNLayer layer_bias(2, 2, true, Activation::None);
    layer_bias.set_weight(Tensor::dense(2, 2, {1,0.5, 0.5,1}));
    // bias stays zero

    GCNLayer layer_nobias(2, 2, false, Activation::None);
    layer_nobias.set_weight(Tensor::dense(2, 2, {1,0.5, 0.5,1}));

    auto out_bias = layer_bias.forward(A_norm, H);
    auto out_nobias = layer_nobias.forward(A_norm, H);

    // With zero bias, both should match
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 2; ++c) {
            ASSERT_NEAR(out_bias.at(r, c), out_nobias.at(r, c), 1e-6);
        }
    }
}

// Test 30: No activation (identity)
void test_gcn_forward_no_activation() {
    auto A = make_undirected_csr(3, {{0,1}, {1,2}});
    auto A_norm = gcn_norm(A);
    auto H = Tensor::dense(3, 2, {1,0, 0,1, 1,1});

    GCNLayer layer(2, 2, true, Activation::None);
    layer.set_weight(Tensor::dense(2, 2, {1,0.5, 0.5,1}));
    layer.set_bias(Tensor::dense(1, 2, {0.1f, -0.1f}));

    auto out = layer.forward(A_norm, H);

    // Same as hand-computed in test 23 but without ReLU
    // Since all values are positive, they should match ReLU result
    ASSERT_NEAR(out.at(0, 0), 0.804124f, 1e-4);
    ASSERT_NEAR(out.at(0, 1), 0.558248f, 1e-4);
}

// Test 31: No ReLU, negative outputs preserved
void test_gcn_forward_negative_output_no_activation() {
    auto A = make_undirected_csr(2, {{0,1}});
    auto A_norm = gcn_norm(A);
    auto H = Tensor::dense(2, 1, {1.0f, 1.0f});

    GCNLayer layer(1, 1, true, Activation::None);
    layer.set_weight(Tensor::dense(1, 1, {1.0f}));
    layer.set_bias(Tensor::dense(1, 1, {-5.0f}));

    auto out = layer.forward(A_norm, H);

    // Values should be negative
    ASSERT_TRUE(out.at(0, 0) < 0.0f);
    ASSERT_TRUE(out.at(1, 0) < 0.0f);
}

// Test 32: Dimension reduction (in_features > out_features)
void test_gcn_forward_dim_reduction() {
    auto A = make_undirected_csr(4, {{0,1}, {1,2}, {2,3}});
    auto A_norm = gcn_norm(A);
    auto H = Tensor::dense(4, 8);  // 8 features in

    GCNLayer layer(8, 2, false, Activation::None);  // reduce to 2
    auto out = layer.forward(A_norm, H);

    ASSERT_EQ(out.rows(), 4u);
    ASSERT_EQ(out.cols(), 2u);
}

// Test 33: Dimension expansion (in_features < out_features)
void test_gcn_forward_dim_expansion() {
    auto A = make_undirected_csr(4, {{0,1}, {1,2}, {2,3}});
    auto A_norm = gcn_norm(A);
    auto H = Tensor::dense(4, 2);  // 2 features in

    GCNLayer layer(2, 16, true, Activation::ReLU);  // expand to 16
    auto out = layer.forward(A_norm, H);

    ASSERT_EQ(out.rows(), 4u);
    ASSERT_EQ(out.cols(), 16u);
}


// ============================================================================
//  Category 6: Two-layer GCN stacking
// ============================================================================

// Test 34: Two-layer pipeline dimensions
void test_gcn_two_layer_dims() {
    auto A = make_undirected_csr(5, {{0,1},{1,2},{2,3},{3,4}});
    auto A_norm = gcn_norm(A);
    auto H = Tensor::dense(5, 4);  // 5 nodes, 4 features

    GCNLayer layer1(4, 8, true, Activation::ReLU);
    GCNLayer layer2(8, 2, true, Activation::None);

    auto h1 = layer1.forward(A_norm, H);
    ASSERT_EQ(h1.rows(), 5u);
    ASSERT_EQ(h1.cols(), 8u);

    auto h2 = layer2.forward(A_norm, h1);
    ASSERT_EQ(h2.rows(), 5u);
    ASSERT_EQ(h2.cols(), 2u);
}

// Test 35: Two-layer pipeline with known weights
void test_gcn_two_layer_values() {
    auto A = make_undirected_csr(3, {{0,1}, {1,2}});
    auto A_norm = gcn_norm(A);
    auto H = Tensor::dense(3, 2, {1,0, 0,1, 1,1});

    // Layer 1: 2 → 2, no bias, no activation (linear)
    GCNLayer layer1(2, 2, false, Activation::None);
    layer1.set_weight(Tensor::dense(2, 2, {1,0, 0,1}));  // identity

    // Layer 2: 2 → 1, bias, no activation
    GCNLayer layer2(2, 1, true, Activation::None);
    layer2.set_weight(Tensor::dense(2, 1, {1, 1}));  // sum features
    layer2.set_bias(Tensor::dense(1, 1, {0.0f}));

    auto h1 = layer1.forward(A_norm, H);
    auto h2 = layer2.forward(A_norm, h1);

    ASSERT_EQ(h2.rows(), 3u);
    ASSERT_EQ(h2.cols(), 1u);

    // Verify non-zero output (just a sanity check)
    ASSERT_TRUE(std::fabs(h2.at(0, 0)) > 1e-6);
    ASSERT_TRUE(std::fabs(h2.at(1, 0)) > 1e-6);
    ASSERT_TRUE(std::fabs(h2.at(2, 0)) > 1e-6);
}

// Test 36: Three-layer deep GCN
void test_gcn_three_layer() {
    auto A = make_undirected_csr(4, {{0,1},{1,2},{2,3},{0,3}});
    auto A_norm = gcn_norm(A);
    auto H = Tensor::dense(4, 8);  // 4 nodes, 8 features

    GCNLayer l1(8, 16, true, Activation::ReLU);
    GCNLayer l2(16, 8, true, Activation::ReLU);
    GCNLayer l3(8, 2, false, Activation::None);

    auto h1 = l1.forward(A_norm, H);
    auto h2 = l2.forward(A_norm, h1);
    auto h3 = l3.forward(A_norm, h2);

    ASSERT_EQ(h3.rows(), 4u);
    ASSERT_EQ(h3.cols(), 2u);
}

// Test 37: Same norm matrix for all layers
void test_gcn_shared_norm() {
    auto A = make_undirected_csr(3, {{0,1},{1,2}});
    auto A_norm = gcn_norm(A);
    auto H = Tensor::dense(3, 2, {1,2,3,4,5,6});

    GCNLayer l1(2, 4);
    GCNLayer l2(4, 2);

    auto h1 = l1.forward(A_norm, H);
    auto h2 = l2.forward(A_norm, h1);

    // Just verify it doesn't crash and dimensions are correct
    ASSERT_EQ(h2.rows(), 3u);
    ASSERT_EQ(h2.cols(), 2u);
}


// ============================================================================
//  Category 7: 10-node ring graph — full pipeline verification
// ============================================================================
//
//  Ring graph: 0-1-2-3-4-5-6-7-8-9-0
//  Each node has degree 2. After self-loop: degree 3.
//  Â[i][i] = 1/3, Â[i][j] = 1/3 for neighbors (since deg[i]=deg[j]=3)
//

// Test 38: 10-node ring — gcn_norm properties
void test_10node_ring_norm_properties() {
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < 10; ++i) {
        edges.push_back({i, (i + 1) % 10});
    }
    auto A = make_undirected_csr(10, edges);
    auto A_norm = gcn_norm(A);

    ASSERT_EQ(A_norm.rows(), 10u);
    ASSERT_EQ(A_norm.cols(), 10u);

    // Each node has degree 3 (2 neighbors + self-loop = 3)
    // So Â[i][j] = 1/(√3·√3) = 1/3 for all entries
    const auto& v = A_norm.data();
    for (std::size_t i = 0; i < A_norm.nnz(); ++i) {
        ASSERT_NEAR(v[i], 1.0f / 3.0f, 1e-5);
    }
}

// Test 39: 10-node ring — number of non-zeros
void test_10node_ring_nnz() {
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < 10; ++i) {
        edges.push_back({i, (i + 1) % 10});
    }
    auto A = make_undirected_csr(10, edges);
    auto A_norm = gcn_norm(A);

    // 20 directed edges + 10 self-loops = 30
    ASSERT_EQ(A_norm.nnz(), 30u);
}

// Test 40: 10-node ring — forward pass with identity weight
void test_10node_ring_identity_forward() {
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < 10; ++i) {
        edges.push_back({i, (i + 1) % 10});
    }
    auto A = make_undirected_csr(10, edges);
    auto A_norm = gcn_norm(A);

    // Features: node i has feature [i, 10-i]
    std::vector<float> h_data;
    for (int i = 0; i < 10; ++i) {
        h_data.push_back(static_cast<float>(i));
        h_data.push_back(static_cast<float>(10 - i));
    }
    auto H = Tensor::dense(10, 2, h_data);

    GCNLayer layer(2, 2, false, Activation::None);
    layer.set_weight(Tensor::dense(2, 2, {1,0, 0,1}));  // identity

    auto out = layer.forward(A_norm, H);

    // With W=I, no bias, no activation: out = Â * H
    // For ring graph with all Â entries = 1/3:
    // out[i] = (1/3)*(H[i-1] + H[i] + H[i+1])
    for (int i = 0; i < 10; ++i) {
        int prev = (i + 9) % 10;
        int next = (i + 1) % 10;
        float expected_f0 = (h_data[prev*2] + h_data[i*2] + h_data[next*2]) / 3.0f;
        float expected_f1 = (h_data[prev*2+1] + h_data[i*2+1] + h_data[next*2+1]) / 3.0f;

        ASSERT_NEAR(out.at(i, 0), expected_f0, 1e-5);
        ASSERT_NEAR(out.at(i, 1), expected_f1, 1e-5);
    }
}

// Test 41: 10-node ring — forward with specific weights, verify step by step
void test_10node_ring_full_pipeline() {
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < 10; ++i) {
        edges.push_back({i, (i + 1) % 10});
    }
    auto A = make_undirected_csr(10, edges);
    auto A_norm = gcn_norm(A);

    // All features = [1.0, 1.0]
    std::vector<float> h_data(20, 1.0f);
    auto H = Tensor::dense(10, 2, h_data);

    // W = [[2.0, 0.0], [0.0, 3.0]], bias = [0.1, -0.2]
    GCNLayer layer(2, 2, true, Activation::ReLU);
    layer.set_weight(Tensor::dense(2, 2, {2.0f, 0.0f, 0.0f, 3.0f}));
    layer.set_bias(Tensor::dense(1, 2, {0.1f, -0.2f}));

    auto out = layer.forward(A_norm, H);

    // HW = H * W = [[2, 3], [2, 3], ...]  (all rows identical)
    // spmm(Â, HW): since all HW rows are the same and Â rows sum to 1/3*3=1:
    //   out[i] = (1/3)*[2,3] + (1/3)*[2,3] + (1/3)*[2,3] = [2, 3]
    // + bias: [2.1, 2.8]
    // ReLU: [2.1, 2.8]
    for (int i = 0; i < 10; ++i) {
        ASSERT_NEAR(out.at(i, 0), 2.1f, 1e-4);
        ASSERT_NEAR(out.at(i, 1), 2.8f, 1e-4);
    }
}

// Test 42: 10-node ring — two-layer GCN
void test_10node_ring_two_layer() {
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < 10; ++i) {
        edges.push_back({i, (i + 1) % 10});
    }
    auto A = make_undirected_csr(10, edges);
    auto A_norm = gcn_norm(A);

    std::vector<float> h_data;
    for (int i = 0; i < 10; ++i) {
        h_data.push_back(static_cast<float>(i) / 10.0f);
        h_data.push_back(static_cast<float>(10 - i) / 10.0f);
    }
    auto H = Tensor::dense(10, 2, h_data);

    GCNLayer l1(2, 4, true, Activation::ReLU);
    l1.set_weight(Tensor::dense(2, 4, {0.5f, 0.1f, -0.3f, 0.2f,
                                        0.1f, 0.5f,  0.2f, -0.1f}));
    l1.set_bias(Tensor::dense(1, 4, {0.0f, 0.0f, 0.0f, 0.0f}));

    GCNLayer l2(4, 2, false, Activation::None);
    l2.set_weight(Tensor::dense(4, 2, {1,0, 0,1, 1,0, 0,1}));

    auto h1 = l1.forward(A_norm, H);
    auto h2 = l2.forward(A_norm, h1);

    ASSERT_EQ(h2.rows(), 10u);
    ASSERT_EQ(h2.cols(), 2u);

    // Verify all outputs are finite
    for (std::size_t r = 0; r < 10; ++r) {
        for (std::size_t c = 0; c < 2; ++c) {
            ASSERT_TRUE(std::isfinite(h2.at(r, c)));
        }
    }
}

// Test 43: 10-node with step-by-step manual verification
void test_10node_manual_step_by_step() {
    // Star graph: node 0 connected to all others
    std::vector<std::pair<int,int>> edges;
    for (int i = 1; i < 10; ++i) {
        edges.push_back({0, i});
    }
    auto A = make_undirected_csr(10, edges);
    auto A_norm = gcn_norm(A);

    // Node 0 has deg_tilde = 9+1 = 10 (9 neighbors + self)
    // Node i (i>0) has deg_tilde = 1+1 = 2 (1 neighbor + self)
    // Â[0][0] = 1/sqrt(10)*1/sqrt(10) = 1/10 = 0.1
    // Â[0][i] = 1/sqrt(10)*1/sqrt(2)  for i>0
    // Â[i][0] = 1/sqrt(2)*1/sqrt(10)  for i>0
    // Â[i][i] = 1/sqrt(2)*1/sqrt(2) = 0.5  for i>0

    // Verify norm values for node 0
    const auto& rp = A_norm.row_ptr();
    const auto& ci = A_norm.col_ind();
    const auto& v  = A_norm.data();

    float expected_self_0 = 1.0f / 10.0f;
    float expected_cross  = 1.0f / std::sqrt(20.0f);  // 1/(sqrt(10)*sqrt(2))
    float expected_self_i = 0.5f;

    // Check a few key values
    // Row 0 should have 10 entries (self + 9 neighbors)
    ASSERT_EQ(rp[1] - rp[0], 10);

    // The diagonal of row 0 (col 0)
    ASSERT_EQ(ci[rp[0]], 0);
    ASSERT_NEAR(v[rp[0]], expected_self_0, 1e-5);

    // A cross-edge from row 0 to some neighbor
    ASSERT_NEAR(v[rp[0] + 1], expected_cross, 1e-5);

    // Row 1 (one of the leaf nodes): 2 entries (col 0 and col 1)
    ASSERT_EQ(rp[2] - rp[1], 2);
    ASSERT_NEAR(v[rp[1]], expected_cross, 1e-5);   // Â[1][0]
    ASSERT_NEAR(v[rp[1] + 1], expected_self_i, 1e-5);  // Â[1][1]
}

// Test 44-48: Additional property tests for 10-node graphs
void test_10node_symmetric_norm() {
    // For undirected graph, Â should be symmetric: Â[i][j] = Â[j][i]
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < 10; ++i) {
        edges.push_back({i, (i + 1) % 10});
    }
    auto A = make_undirected_csr(10, edges);
    auto A_norm = gcn_norm(A);

    const auto& rp = A_norm.row_ptr();
    const auto& ci = A_norm.col_ind();
    const auto& v  = A_norm.data();

    // Build dense matrix to check symmetry
    std::vector<float> dense(100, 0.0f);
    for (std::size_t i = 0; i < 10; ++i) {
        for (int32_t nz = rp[i]; nz < rp[i + 1]; ++nz) {
            dense[i * 10 + ci[nz]] = v[nz];
        }
    }

    for (int i = 0; i < 10; ++i) {
        for (int j = i + 1; j < 10; ++j) {
            ASSERT_NEAR(dense[i * 10 + j], dense[j * 10 + i], 1e-5);
        }
    }
}

void test_10node_norm_preserves_structure() {
    // All non-zeros in A_norm should correspond to edges in A+I
    std::vector<std::pair<int,int>> edges = {{0,1},{1,2},{3,4},{5,6},{7,8}};
    auto A = make_undirected_csr(10, edges);
    auto A_norm = gcn_norm(A);

    // nnz = 20 directed edges + 10 self-loops = 30?
    // Wait: edges are just 5 undirected = 10 directed, plus 10 self = 20
    ASSERT_EQ(A_norm.nnz(), 20u);

    // All values should be positive
    for (std::size_t i = 0; i < A_norm.nnz(); ++i) {
        ASSERT_TRUE(A_norm.data()[i] > 0.0f);
    }
}


// ============================================================================
//  Category 8: Error handling
// ============================================================================

// Test 49: forward with Dense A_norm
void test_gcn_forward_rejects_dense_A() {
    GCNLayer layer(2, 2);
    auto A = Tensor::dense(3, 3);
    auto H = Tensor::dense(3, 2);
    ASSERT_THROWS(layer.forward(A, H), std::invalid_argument);
}

// Test 50: forward with SparseCSR H
void test_gcn_forward_rejects_sparse_H() {
    auto A_norm = gcn_norm(make_undirected_csr(3, {{0,1},{1,2}}));
    GCNLayer layer(2, 2);
    auto H = Tensor::sparse_csr(3, 2, {0,1,2,3}, {0,1,0}, {1,1,1});
    ASSERT_THROWS(layer.forward(A_norm, H), std::invalid_argument);
}

// Test 51: forward with non-square A_norm
void test_gcn_forward_rejects_nonsquare_A() {
    GCNLayer layer(2, 2);
    auto A = Tensor::sparse_csr(3, 4,
        {0,1,2,3}, {0,1,2}, {1,1,1});
    auto H = Tensor::dense(4, 2);
    ASSERT_THROWS(layer.forward(A, H), std::invalid_argument);
}

// Test 52: forward with wrong feature dimension
void test_gcn_forward_wrong_features() {
    auto A_norm = gcn_norm(make_undirected_csr(3, {{0,1},{1,2}}));
    GCNLayer layer(4, 2);  // expects 4 input features
    auto H = Tensor::dense(3, 2);  // has 2 features
    ASSERT_THROWS(layer.forward(A_norm, H), std::invalid_argument);
}

// Test 53: forward with mismatched A and H rows
void test_gcn_forward_row_mismatch() {
    auto A_norm = gcn_norm(make_undirected_csr(3, {{0,1},{1,2}}));
    GCNLayer layer(2, 2);
    auto H = Tensor::dense(5, 2);  // 5 nodes but A_norm is 3×3
    ASSERT_THROWS(layer.forward(A_norm, H), std::invalid_argument);
}

// Test 54: Error message content for Dense A
void test_gcn_forward_error_msg_dense_A() {
    GCNLayer layer(2, 2);
    auto A = Tensor::dense(3, 3);
    auto H = Tensor::dense(3, 2);
    ASSERT_THROWS_MSG(layer.forward(A, H), std::invalid_argument, "SparseCSR");
}

// Test 55: Error message for wrong features
void test_gcn_forward_error_msg_features() {
    auto A_norm = gcn_norm(make_undirected_csr(3, {{0,1},{1,2}}));
    GCNLayer layer(4, 2);
    auto H = Tensor::dense(3, 2);
    ASSERT_THROWS_MSG(layer.forward(A_norm, H), std::invalid_argument, "in_features");
}

// Test 56: set_weight with sparse tensor
void test_gcn_set_weight_rejects_sparse() {
    GCNLayer layer(2, 2);
    auto W = Tensor::sparse_csr(2, 2, {0,1,2}, {0,1}, {1,1});
    ASSERT_THROWS(layer.set_weight(std::move(W)), std::invalid_argument);
}

// Test 57: set_bias with sparse tensor
void test_gcn_set_bias_rejects_sparse() {
    GCNLayer layer(2, 2);
    auto b = Tensor::sparse_csr(1, 2, {0,1}, {0}, {1});
    ASSERT_THROWS(layer.set_bias(std::move(b)), std::invalid_argument);
}

// Test 58: set_bias with wrong number of rows
void test_gcn_set_bias_wrong_rows() {
    GCNLayer layer(2, 3);
    auto b = Tensor::dense(2, 3);  // 2 rows instead of 1
    ASSERT_THROWS(layer.set_bias(std::move(b)), std::invalid_argument);
}

// Tests 59-60: add_self_loops / gcn_norm error messages
void test_add_self_loops_error_msg() {
    auto D = Tensor::dense(3, 3);
    ASSERT_THROWS_MSG(add_self_loops(D), std::invalid_argument, "SparseCSR");
}

void test_gcn_norm_error_msg() {
    auto A = Tensor::sparse_csr(3, 4,
        {0,1,2,3}, {0,1,2}, {1,1,1});
    ASSERT_THROWS_MSG(gcn_norm(A), std::invalid_argument, "square");
}


// ============================================================================
//  Category 9: Edge / degenerate cases
// ============================================================================

// Test 61: Single node, single feature
void test_gcn_single_node() {
    auto A = Tensor::sparse_csr(1, 1, {0, 0}, {}, {});
    auto A_norm = gcn_norm(A);
    auto H = Tensor::dense(1, 1, {3.0f});

    GCNLayer layer(1, 1, true, Activation::None);
    layer.set_weight(Tensor::dense(1, 1, {2.0f}));
    layer.set_bias(Tensor::dense(1, 1, {0.5f}));

    auto out = layer.forward(A_norm, H);

    // A_norm for single node with self-loop: [[1.0]]
    // HW = 3.0 * 2.0 = 6.0
    // spmm([[1.0]], [[6.0]]) = [[6.0]]
    // + bias: 6.5
    ASSERT_NEAR(out.at(0, 0), 6.5f, 1e-5);
}

// Test 62: Disconnected components (2 separate edges)
void test_gcn_disconnected_components() {
    // 4 nodes: 0-1, 2-3 (two separate edges)
    auto A = make_undirected_csr(4, {{0,1}, {2,3}});
    auto A_norm = gcn_norm(A);
    auto H = Tensor::dense(4, 1, {1.0f, 2.0f, 10.0f, 20.0f});

    GCNLayer layer(1, 1, false, Activation::None);
    layer.set_weight(Tensor::dense(1, 1, {1.0f}));

    auto out = layer.forward(A_norm, H);

    // Nodes 0,1 form one component; nodes 2,3 form another
    // Information from component 1 should NOT leak to component 2
    // For ring of 2 nodes with self-loops: each has degree 2
    // Â[0][0] = 0.5, Â[0][1] = 0.5, Â[1][0] = 0.5, Â[1][1] = 0.5
    // out[0] = 0.5*1 + 0.5*2 = 1.5
    // out[1] = 0.5*1 + 0.5*2 = 1.5
    ASSERT_NEAR(out.at(0, 0), 1.5f, 1e-5);
    ASSERT_NEAR(out.at(1, 0), 1.5f, 1e-5);

    // out[2] = 0.5*10 + 0.5*20 = 15
    // out[3] = 0.5*10 + 0.5*20 = 15
    ASSERT_NEAR(out.at(2, 0), 15.0f, 1e-5);
    ASSERT_NEAR(out.at(3, 0), 15.0f, 1e-5);
}

// Test 63: Fully connected graph K5
void test_gcn_fully_connected() {
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j)
            edges.push_back({i, j});
    auto A = make_undirected_csr(5, edges);
    auto A_norm = gcn_norm(A);

    // All const features → output should be uniform
    auto H = Tensor::dense(5, 1, {1,1,1,1,1});
    GCNLayer layer(1, 1, false, Activation::None);
    layer.set_weight(Tensor::dense(1, 1, {1.0f}));

    auto out = layer.forward(A_norm, H);

    // All nodes see the same neighborhood → same output
    float val = out.at(0, 0);
    for (int i = 1; i < 5; ++i) {
        ASSERT_NEAR(out.at(i, 0), val, 1e-5);
    }
}

// Test 64: Large-ish graph (100 nodes, chain)
void test_gcn_100_node_chain() {
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < 99; ++i) {
        edges.push_back({i, i + 1});
    }
    auto A = make_undirected_csr(100, edges);
    auto A_norm = gcn_norm(A);

    std::vector<float> h_data(200, 1.0f);
    auto H = Tensor::dense(100, 2, h_data);

    GCNLayer layer(2, 4, true, Activation::ReLU);
    auto out = layer.forward(A_norm, H);

    ASSERT_EQ(out.rows(), 100u);
    ASSERT_EQ(out.cols(), 4u);

    // All outputs should be non-negative (ReLU)
    for (std::size_t r = 0; r < 100; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            ASSERT_TRUE(out.at(r, c) >= 0.0f);
        }
    }
}

// Test 65: Weighted graph (non-binary edges)
void test_gcn_weighted_graph() {
    // A weighted graph: edge weights ≠ 1.0
    auto A = Tensor::sparse_csr(2, 2,
        {0, 1, 2},
        {1, 0},
        {0.5f, 2.0f});  // asymmetric weights

    auto A_norm = gcn_norm(A);

    // After self-loops: Ã = [[1, 0.5], [2, 1]]
    // deg[0] = 1 + 0.5 = 1.5
    // deg[1] = 2 + 1 = 3
    // Â[0][0] = 1 / (√1.5 * √1.5) = 1/1.5
    // Â[0][1] = 0.5 / (√1.5 * √3)
    // etc.

    float d0 = 1.5f, d1 = 3.0f;
    float expected_00 = 1.0f / d0;
    float expected_01 = 0.5f / (std::sqrt(d0) * std::sqrt(d1));
    float expected_10 = 2.0f / (std::sqrt(d1) * std::sqrt(d0));
    float expected_11 = 1.0f / d1;

    const auto& v = A_norm.data();
    ASSERT_NEAR(v[0], expected_00, 1e-5);
    ASSERT_NEAR(v[1], expected_01, 1e-5);
    ASSERT_NEAR(v[2], expected_10, 1e-5);
    ASSERT_NEAR(v[3], expected_11, 1e-5);
}

// Test 66: Directed graph still works (not symmetric)
void test_gcn_directed_graph() {
    // Directed: 0→1, 1→2 (no reverse edges)
    auto A = make_csr(3, {{0,1}, {1,2}});

    // gcn_norm should still work (the math is defined for any square CSR)
    auto A_norm = gcn_norm(A);
    ASSERT_EQ(A_norm.rows(), 3u);

    auto H = Tensor::dense(3, 2, {1,0, 0,1, 1,1});
    GCNLayer layer(2, 2, false, Activation::None);
    layer.set_weight(Tensor::dense(2, 2, {1,0, 0,1}));

    auto out = layer.forward(A_norm, H);
    ASSERT_EQ(out.rows(), 3u);
    ASSERT_EQ(out.cols(), 2u);
}

// Test 67: Multiple features, verify matmul ordering
void test_gcn_feature_transform_ordering() {
    // Verify that matmul is applied before spmm (transform-then-propagate)
    auto A = make_undirected_csr(2, {{0,1}});
    auto A_norm = gcn_norm(A);

    // H = [[1, 0], [0, 1]], W = [[1, 2], [3, 4]]
    auto H = Tensor::dense(2, 2, {1,0, 0,1});

    GCNLayer layer(2, 2, false, Activation::None);
    layer.set_weight(Tensor::dense(2, 2, {1,2, 3,4}));

    auto out = layer.forward(A_norm, H);

    // Step 1: HW = [[1,2],[3,4]]
    // Step 2: spmm(A_norm, HW)
    // For 2-node complete graph + self-loops: each node degree is 2
    // Â = [[0.5, 0.5], [0.5, 0.5]]
    // out[0] = 0.5*[1,2] + 0.5*[3,4] = [2, 3]
    // out[1] = 0.5*[1,2] + 0.5*[3,4] = [2, 3]
    ASSERT_NEAR(out.at(0, 0), 2.0f, 1e-5);
    ASSERT_NEAR(out.at(0, 1), 3.0f, 1e-5);
    ASSERT_NEAR(out.at(1, 0), 2.0f, 1e-5);
    ASSERT_NEAR(out.at(1, 1), 3.0f, 1e-5);
}

// Test 68: GCN output matches manual spmm + matmul
void test_gcn_matches_manual_ops() {
    auto A = make_undirected_csr(3, {{0,1},{1,2}});
    auto A_norm = gcn_norm(A);
    auto H = Tensor::dense(3, 2, {0.5f, -0.3f, 1.2f, 0.8f, -0.1f, 0.6f});
    auto W = Tensor::dense(2, 3, {0.1f, -0.2f, 0.3f,
                                   0.4f, 0.5f, -0.6f});
    auto bias = Tensor::dense(1, 3, {0.01f, 0.02f, 0.03f});

    // Manual computation
    auto HW = matmul(H, W);
    auto agg = spmm(A_norm, HW);
    add_bias(agg, bias);
    relu_inplace(agg);

    // Layer computation
    GCNLayer layer(2, 3, true, Activation::ReLU);
    layer.set_weight(Tensor::dense(2, 3, {0.1f, -0.2f, 0.3f,
                                           0.4f, 0.5f, -0.6f}));
    layer.set_bias(Tensor::dense(1, 3, {0.01f, 0.02f, 0.03f}));
    auto out = layer.forward(A_norm, H);

    // Both should produce identical results
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            ASSERT_NEAR(out.at(r, c), agg.at(r, c), 1e-5);
        }
    }
}


// ============================================================================
//  main
// ============================================================================
int main() {
    std::cout << "\n"
        "+=================================================================+\n"
        "|   TinyGNN — GCN Layer Unit Tests (Phase 6)                     |\n"
        "|   Testing: add_self_loops, gcn_norm, GCNLayer                  |\n"
        "+=================================================================+\n\n";

    // Category 1: add_self_loops
    std::cout << "── 1. add_self_loops ────────────────────────────────────────\n";
    std::cout << "  Running test_add_self_loops_basic...\n";              test_add_self_loops_basic();
    std::cout << "  Running test_add_self_loops_existing_diag...\n";      test_add_self_loops_existing_diag();
    std::cout << "  Running test_add_self_loops_empty...\n";              test_add_self_loops_empty();
    std::cout << "  Running test_add_self_loops_single_node...\n";        test_add_self_loops_single_node();
    std::cout << "  Running test_add_self_loops_sorted_order...\n";       test_add_self_loops_sorted_order();
    std::cout << "  Running test_add_self_loops_rejects_dense...\n";      test_add_self_loops_rejects_dense();
    std::cout << "  Running test_add_self_loops_rejects_nonsquare...\n";  test_add_self_loops_rejects_nonsquare();

    // Category 2: gcn_norm
    std::cout << "\n── 2. gcn_norm ──────────────────────────────────────────────\n";
    std::cout << "  Running test_gcn_norm_3node_path...\n";               test_gcn_norm_3node_path();
    std::cout << "  Running test_gcn_norm_complete_graph...\n";           test_gcn_norm_complete_graph();
    std::cout << "  Running test_gcn_norm_isolated_nodes...\n";           test_gcn_norm_isolated_nodes();
    std::cout << "  Running test_gcn_norm_single_node...\n";              test_gcn_norm_single_node();
    std::cout << "  Running test_gcn_norm_row_sums_bounded...\n";         test_gcn_norm_row_sums_bounded();
    std::cout << "  Running test_gcn_norm_rejects_dense...\n";            test_gcn_norm_rejects_dense();
    std::cout << "  Running test_gcn_norm_rejects_nonsquare...\n";        test_gcn_norm_rejects_nonsquare();

    // Category 3: GCNLayer construction
    std::cout << "\n── 3. GCNLayer Construction ─────────────────────────────────\n";
    std::cout << "  Running test_gcn_layer_construction...\n";            test_gcn_layer_construction();
    std::cout << "  Running test_gcn_layer_no_bias...\n";                 test_gcn_layer_no_bias();
    std::cout << "  Running test_gcn_layer_set_weight...\n";              test_gcn_layer_set_weight();
    std::cout << "  Running test_gcn_layer_set_bias...\n";                test_gcn_layer_set_bias();
    std::cout << "  Running test_gcn_layer_set_weight_wrong_shape...\n";  test_gcn_layer_set_weight_wrong_shape();
    std::cout << "  Running test_gcn_layer_set_bias_no_bias...\n";        test_gcn_layer_set_bias_no_bias();
    std::cout << "  Running test_gcn_layer_set_bias_wrong_shape...\n";    test_gcn_layer_set_bias_wrong_shape();
    std::cout << "  Running test_gcn_layer_zero_features...\n";           test_gcn_layer_zero_features();

    // Category 4: GCNLayer forward (3-node hand-computed)
    std::cout << "\n── 4. GCNLayer Forward (3-node) ─────────────────────────────\n";
    std::cout << "  Running test_gcn_forward_3node...\n";                 test_gcn_forward_3node();
    std::cout << "  Running test_gcn_forward_output_shape...\n";          test_gcn_forward_output_shape();
    std::cout << "  Running test_gcn_forward_identity_weight...\n";       test_gcn_forward_identity_weight();
    std::cout << "  Running test_gcn_forward_relu_clips...\n";            test_gcn_forward_relu_clips();
    std::cout << "  Running test_gcn_forward_zero_weight...\n";           test_gcn_forward_zero_weight();
    std::cout << "  Running test_gcn_forward_reuse_norm...\n";            test_gcn_forward_reuse_norm();

    // Category 5: No bias / no activation
    std::cout << "\n── 5. No Bias / No Activation ───────────────────────────────\n";
    std::cout << "  Running test_gcn_forward_no_bias...\n";               test_gcn_forward_no_bias();
    std::cout << "  Running test_gcn_forward_no_activation...\n";         test_gcn_forward_no_activation();
    std::cout << "  Running test_gcn_forward_negative_output_no_activation...\n"; test_gcn_forward_negative_output_no_activation();
    std::cout << "  Running test_gcn_forward_dim_reduction...\n";         test_gcn_forward_dim_reduction();
    std::cout << "  Running test_gcn_forward_dim_expansion...\n";         test_gcn_forward_dim_expansion();

    // Category 6: Two/three-layer stacking
    std::cout << "\n── 6. Multi-layer GCN ───────────────────────────────────────\n";
    std::cout << "  Running test_gcn_two_layer_dims...\n";                test_gcn_two_layer_dims();
    std::cout << "  Running test_gcn_two_layer_values...\n";              test_gcn_two_layer_values();
    std::cout << "  Running test_gcn_three_layer...\n";                   test_gcn_three_layer();
    std::cout << "  Running test_gcn_shared_norm...\n";                   test_gcn_shared_norm();

    // Category 7: 10-node ring graph
    std::cout << "\n── 7. 10-node Graph Tests ───────────────────────────────────\n";
    std::cout << "  Running test_10node_ring_norm_properties...\n";       test_10node_ring_norm_properties();
    std::cout << "  Running test_10node_ring_nnz...\n";                   test_10node_ring_nnz();
    std::cout << "  Running test_10node_ring_identity_forward...\n";      test_10node_ring_identity_forward();
    std::cout << "  Running test_10node_ring_full_pipeline...\n";         test_10node_ring_full_pipeline();
    std::cout << "  Running test_10node_ring_two_layer...\n";             test_10node_ring_two_layer();
    std::cout << "  Running test_10node_manual_step_by_step...\n";        test_10node_manual_step_by_step();
    std::cout << "  Running test_10node_symmetric_norm...\n";             test_10node_symmetric_norm();
    std::cout << "  Running test_10node_norm_preserves_structure...\n";   test_10node_norm_preserves_structure();

    // Category 8: Error handling
    std::cout << "\n── 8. Error Handling ────────────────────────────────────────\n";
    std::cout << "  Running test_gcn_forward_rejects_dense_A...\n";       test_gcn_forward_rejects_dense_A();
    std::cout << "  Running test_gcn_forward_rejects_sparse_H...\n";      test_gcn_forward_rejects_sparse_H();
    std::cout << "  Running test_gcn_forward_rejects_nonsquare_A...\n";   test_gcn_forward_rejects_nonsquare_A();
    std::cout << "  Running test_gcn_forward_wrong_features...\n";        test_gcn_forward_wrong_features();
    std::cout << "  Running test_gcn_forward_row_mismatch...\n";          test_gcn_forward_row_mismatch();
    std::cout << "  Running test_gcn_forward_error_msg_dense_A...\n";     test_gcn_forward_error_msg_dense_A();
    std::cout << "  Running test_gcn_forward_error_msg_features...\n";    test_gcn_forward_error_msg_features();
    std::cout << "  Running test_gcn_set_weight_rejects_sparse...\n";     test_gcn_set_weight_rejects_sparse();
    std::cout << "  Running test_gcn_set_bias_rejects_sparse...\n";       test_gcn_set_bias_rejects_sparse();
    std::cout << "  Running test_gcn_set_bias_wrong_rows...\n";           test_gcn_set_bias_wrong_rows();
    std::cout << "  Running test_add_self_loops_error_msg...\n";          test_add_self_loops_error_msg();
    std::cout << "  Running test_gcn_norm_error_msg...\n";                test_gcn_norm_error_msg();

    // Category 9: Edge / degenerate cases
    std::cout << "\n── 9. Edge / Degenerate Cases ───────────────────────────────\n";
    std::cout << "  Running test_gcn_single_node...\n";                   test_gcn_single_node();
    std::cout << "  Running test_gcn_disconnected_components...\n";       test_gcn_disconnected_components();
    std::cout << "  Running test_gcn_fully_connected...\n";               test_gcn_fully_connected();
    std::cout << "  Running test_gcn_100_node_chain...\n";                test_gcn_100_node_chain();
    std::cout << "  Running test_gcn_weighted_graph...\n";                test_gcn_weighted_graph();
    std::cout << "  Running test_gcn_directed_graph...\n";                test_gcn_directed_graph();
    std::cout << "  Running test_gcn_feature_transform_ordering...\n";    test_gcn_feature_transform_ordering();
    std::cout << "  Running test_gcn_matches_manual_ops...\n";            test_gcn_matches_manual_ops();

    // ── Summary ──
    std::cout << "\n"
        "=================================================================\n"
        "  Total : " << g_tests_run << "\n"
        "  Passed: " << g_tests_passed << "\n"
        "  Failed: " << g_tests_failed << "\n"
        "=================================================================\n\n";

    return g_tests_failed == 0 ? 0 : 1;
}
