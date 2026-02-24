// ============================================================================
//  TinyGNN — GraphSAGE Layer Unit Tests  (Phase 6, Part 2)
//  tests/test_graphsage.cpp
//
//  Test categories:
//    1.  sage_max_aggregate — basic, empty, single node         (tests  1 –  8)
//    2.  SAGELayer construction & weight management              (tests  9 – 17)
//    3.  SAGELayer Mean forward — 3-node hand-computed           (tests 18 – 24)
//    4.  SAGELayer Max forward — 3-node hand-computed            (tests 25 – 31)
//    5.  Multi-layer GraphSAGE                                   (tests 32 – 35)
//    6.  10-node graph tests                                     (tests 36 – 42)
//    7.  Error handling                                          (tests 43 – 52)
//    8.  Edge / degenerate cases                                 (tests 53 – 60)
//
//  GraphSAGE formula (PyG convention):
//    h_v' = σ( W_neigh · AGG({h_u : u ∈ N(v)}) + W_self · h_v + b )
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

// ── Minimal test framework ──────────────────────────────────────────────────

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

// ── Helper: build CSR from directed edges ────────────────────────────────────
static Tensor make_csr(std::size_t N,
                       const std::vector<std::pair<int,int>>& edges) {
    std::vector<int32_t> rp(N + 1, 0);
    for (auto& [s, d] : edges) rp[s + 1]++;
    for (std::size_t i = 1; i <= N; ++i) rp[i] += rp[i - 1];

    std::vector<int32_t> ci(edges.size());
    std::vector<float>   vals(edges.size(), 1.0f);
    std::vector<int32_t> wp(rp.begin(), rp.end());
    for (auto& [s, d] : edges) ci[wp[s]++] = d;

    for (std::size_t i = 0; i < N; ++i)
        std::sort(ci.begin() + rp[i], ci.begin() + rp[i + 1]);

    return Tensor::sparse_csr(N, N,
                              std::move(rp), std::move(ci), std::move(vals));
}

static Tensor make_undirected_csr(std::size_t N,
                                  const std::vector<std::pair<int,int>>& edges) {
    std::vector<std::pair<int,int>> dir;
    for (auto& [u, v] : edges) {
        dir.push_back({u, v});
        dir.push_back({v, u});
    }
    return make_csr(N, dir);
}


// ============================================================================
//  Category 1: sage_max_aggregate
// ============================================================================

// Test 1: Basic max aggregate — 3-node path 0-1-2
//   A (undirected, no self-loops): 0↔1, 1↔2
//   H = [[1,0],[0,2],[3,1]]
//   Node 0 neighbors: {1} → max = [0, 2]
//   Node 1 neighbors: {0, 2} → max = [max(1,3), max(0,1)] = [3, 1]
//   Node 2 neighbors: {1} → max = [0, 2]
void test_sage_max_basic() {
    auto A = make_undirected_csr(3, {{0,1}, {1,2}});
    auto H = Tensor::dense(3, 2, {1,0, 0,2, 3,1});

    auto agg = sage_max_aggregate(A, H);

    ASSERT_EQ(agg.rows(), 3u);
    ASSERT_EQ(agg.cols(), 2u);

    ASSERT_NEAR(agg.at(0, 0), 0.0f, 1e-6);  // max over {H[1]} = [0]
    ASSERT_NEAR(agg.at(0, 1), 2.0f, 1e-6);  // max over {H[1]} = [2]
    ASSERT_NEAR(agg.at(1, 0), 3.0f, 1e-6);  // max(H[0][0], H[2][0]) = max(1,3) = 3
    ASSERT_NEAR(agg.at(1, 1), 1.0f, 1e-6);  // max(H[0][1], H[2][1]) = max(0,1) = 1
    ASSERT_NEAR(agg.at(2, 0), 0.0f, 1e-6);  // max over {H[1]} = [0]
    ASSERT_NEAR(agg.at(2, 1), 2.0f, 1e-6);  // max over {H[1]} = [2]
}

// Test 2: Isolated node gets zero
void test_sage_max_isolated_node() {
    // 3 nodes: only edge 0→1, 1→0. Node 2 isolated.
    auto A = make_undirected_csr(3, {{0,1}});
    auto H = Tensor::dense(3, 2, {5,5, 10,10, 99,99});

    auto agg = sage_max_aggregate(A, H);
    ASSERT_NEAR(agg.at(2, 0), 0.0f, 1e-6);
    ASSERT_NEAR(agg.at(2, 1), 0.0f, 1e-6);
}

// Test 3: Single neighbor → max equals that neighbor
void test_sage_max_single_neighbor() {
    auto A = make_undirected_csr(2, {{0,1}});
    auto H = Tensor::dense(2, 3, {1,2,3, 4,5,6});

    auto agg = sage_max_aggregate(A, H);
    // Node 0's only neighbor is 1
    ASSERT_NEAR(agg.at(0, 0), 4.0f, 1e-6);
    ASSERT_NEAR(agg.at(0, 1), 5.0f, 1e-6);
    ASSERT_NEAR(agg.at(0, 2), 6.0f, 1e-6);
    // Node 1's only neighbor is 0
    ASSERT_NEAR(agg.at(1, 0), 1.0f, 1e-6);
    ASSERT_NEAR(agg.at(1, 1), 2.0f, 1e-6);
    ASSERT_NEAR(agg.at(1, 2), 3.0f, 1e-6);
}

// Test 4: All neighbors have same features → max = that value
void test_sage_max_uniform_neighbors() {
    auto A = make_undirected_csr(4, {{0,1},{0,2},{0,3}});
    auto H = Tensor::dense(4, 2, {0,0, 5,5, 5,5, 5,5});

    auto agg = sage_max_aggregate(A, H);
    // Node 0 has 3 neighbors all with [5,5]
    ASSERT_NEAR(agg.at(0, 0), 5.0f, 1e-6);
    ASSERT_NEAR(agg.at(0, 1), 5.0f, 1e-6);
}

// Test 5: Negative features
void test_sage_max_negative_features() {
    auto A = make_undirected_csr(3, {{0,1},{0,2}});
    auto H = Tensor::dense(3, 2, {0,0, -3,-1, -5,-2});

    auto agg = sage_max_aggregate(A, H);
    // Node 0 neighbors: {1,2} → max([-3,-1],[-5,-2]) = [-3,-1]
    ASSERT_NEAR(agg.at(0, 0), -3.0f, 1e-6);
    ASSERT_NEAR(agg.at(0, 1), -1.0f, 1e-6);
}

// Test 6: Empty graph (no edges)
void test_sage_max_empty_graph() {
    auto A = Tensor::sparse_csr(3, 3, {0,0,0,0}, {}, {});
    auto H = Tensor::dense(3, 2, {1,2,3,4,5,6});

    auto agg = sage_max_aggregate(A, H);
    // All nodes isolated → all zeros
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 2; ++c) {
            ASSERT_NEAR(agg.at(r, c), 0.0f, 1e-6);
        }
    }
}

// Test 7: sage_max_aggregate rejects Dense A
void test_sage_max_rejects_dense_A() {
    auto A = Tensor::dense(3, 3);
    auto H = Tensor::dense(3, 2);
    ASSERT_THROWS(sage_max_aggregate(A, H), std::invalid_argument);
}

// Test 8: sage_max_aggregate rejects SparseCSR H
void test_sage_max_rejects_sparse_H() {
    auto A = make_undirected_csr(3, {{0,1}});
    auto H = Tensor::sparse_csr(3, 2, {0,1,2,3}, {0,1,0}, {1,1,1});
    ASSERT_THROWS(sage_max_aggregate(A, H), std::invalid_argument);
}


// ============================================================================
//  Category 2: SAGELayer construction & weight management
// ============================================================================

// Test 9: Basic construction (Mean)
void test_sage_layer_construction_mean() {
    SAGELayer layer(16, 32, SAGELayer::Aggregator::Mean);
    ASSERT_EQ(layer.in_features(), 16u);
    ASSERT_EQ(layer.out_features(), 32u);
    ASSERT_TRUE(layer.aggregator() == SAGELayer::Aggregator::Mean);
    ASSERT_TRUE(layer.has_bias());
    ASSERT_TRUE(layer.activation() == Activation::ReLU);
}

// Test 10: Construction (Max)
void test_sage_layer_construction_max() {
    SAGELayer layer(8, 4, SAGELayer::Aggregator::Max, false, Activation::None);
    ASSERT_TRUE(layer.aggregator() == SAGELayer::Aggregator::Max);
    ASSERT_TRUE(!layer.has_bias());
    ASSERT_TRUE(layer.activation() == Activation::None);
}

// Test 11: Weight shapes
void test_sage_layer_weight_shapes() {
    SAGELayer layer(4, 8);
    ASSERT_EQ(layer.weight_neigh().rows(), 4u);
    ASSERT_EQ(layer.weight_neigh().cols(), 8u);
    ASSERT_EQ(layer.weight_self().rows(), 4u);
    ASSERT_EQ(layer.weight_self().cols(), 8u);
    ASSERT_EQ(layer.bias().rows(), 1u);
    ASSERT_EQ(layer.bias().cols(), 8u);
}

// Test 12: set_weight_neigh
void test_sage_layer_set_weight_neigh() {
    SAGELayer layer(2, 2);
    layer.set_weight_neigh(Tensor::dense(2, 2, {1,2,3,4}));
    ASSERT_NEAR(layer.weight_neigh().at(1, 1), 4.0f, 1e-6);
}

// Test 13: set_weight_self
void test_sage_layer_set_weight_self() {
    SAGELayer layer(2, 2);
    layer.set_weight_self(Tensor::dense(2, 2, {5,6,7,8}));
    ASSERT_NEAR(layer.weight_self().at(0, 0), 5.0f, 1e-6);
}

// Test 14: set_bias
void test_sage_layer_set_bias() {
    SAGELayer layer(2, 3);
    layer.set_bias(Tensor::dense(1, 3, {0.1f, 0.2f, 0.3f}));
    ASSERT_NEAR(layer.bias().at(0, 1), 0.2f, 1e-6);
}

// Test 15: set_weight_neigh wrong shape
void test_sage_set_weight_neigh_wrong_shape() {
    SAGELayer layer(4, 8);
    ASSERT_THROWS(layer.set_weight_neigh(Tensor::dense(3, 8)), std::invalid_argument);
}

// Test 16: set_bias when no bias
void test_sage_set_bias_no_bias() {
    SAGELayer layer(4, 8, SAGELayer::Aggregator::Mean, false);
    ASSERT_THROWS(layer.set_bias(Tensor::dense(1, 8)), std::invalid_argument);
}

// Test 17: Zero features rejected
void test_sage_zero_features() {
    ASSERT_THROWS(SAGELayer(0, 8), std::invalid_argument);
    ASSERT_THROWS(SAGELayer(8, 0), std::invalid_argument);
}


// ============================================================================
//  Category 3: SAGELayer Mean forward — 3-node path
// ============================================================================
//
//  Graph: 0 -- 1 -- 2  (undirected, NO self-loops)
//  A = [[0,1,0],[1,0,1],[0,1,0]]
//
//  H = [[1, 0],
//       [0, 1],
//       [1, 1]]
//
//  W_neigh = [[1, 0.5], [0.5, 1]]
//  W_self  = [[0.5, 0], [0, 0.5]]
//  bias    = [0.1, -0.1]
//
//  Step 1: Mean aggregation
//    Node 0 neighbors: {1}, deg=1 → AGG[0] = [0,1]/1 = [0, 1]
//    Node 1 neighbors: {0,2}, deg=2 → AGG[1] = ([1,0]+[1,1])/2 = [1, 0.5]
//    Node 2 neighbors: {1}, deg=1 → AGG[2] = [0,1]/1 = [0, 1]
//
//  Step 2: h_neigh = matmul(AGG, W_neigh)
//    h_neigh[0] = [0*1+1*0.5, 0*0.5+1*1] = [0.5, 1.0]
//    h_neigh[1] = [1*1+0.5*0.5, 1*0.5+0.5*1] = [1.25, 1.0]
//    h_neigh[2] = [0.5, 1.0]
//
//  Step 3: h_self = matmul(H, W_self)
//    h_self[0] = [1*0.5+0*0, 1*0+0*0.5] = [0.5, 0.0]
//    h_self[1] = [0*0.5+1*0, 0*0+1*0.5] = [0.0, 0.5]
//    h_self[2] = [1*0.5+1*0, 1*0+1*0.5] = [0.5, 0.5]
//
//  Step 4: out = h_neigh + h_self
//    out[0] = [1.0, 1.0]
//    out[1] = [1.25, 1.5]
//    out[2] = [1.0, 1.5]
//
//  Step 5: + bias
//    out[0] = [1.1, 0.9]
//    out[1] = [1.35, 1.4]
//    out[2] = [1.1, 1.4]
//
//  Step 6: ReLU (all positive)
//

// Test 18: Full mean forward 3-node
void test_sage_mean_forward_3node() {
    auto A = make_undirected_csr(3, {{0,1}, {1,2}});
    auto H = Tensor::dense(3, 2, {1,0, 0,1, 1,1});

    SAGELayer layer(2, 2, SAGELayer::Aggregator::Mean, true, Activation::ReLU);
    layer.set_weight_neigh(Tensor::dense(2, 2, {1,0.5, 0.5,1}));
    layer.set_weight_self(Tensor::dense(2, 2, {0.5,0, 0,0.5}));
    layer.set_bias(Tensor::dense(1, 2, {0.1f, -0.1f}));

    auto out = layer.forward(A, H);

    ASSERT_EQ(out.rows(), 3u);
    ASSERT_EQ(out.cols(), 2u);

    ASSERT_NEAR(out.at(0, 0), 1.1f, 1e-4);
    ASSERT_NEAR(out.at(0, 1), 0.9f, 1e-4);
    ASSERT_NEAR(out.at(1, 0), 1.35f, 1e-4);
    ASSERT_NEAR(out.at(1, 1), 1.4f, 1e-4);
    ASSERT_NEAR(out.at(2, 0), 1.1f, 1e-4);
    ASSERT_NEAR(out.at(2, 1), 1.4f, 1e-4);
}

// Test 19: Output shape
void test_sage_mean_output_shape() {
    auto A = make_undirected_csr(5, {{0,1},{1,2},{2,3},{3,4}});
    auto H = Tensor::dense(5, 4);

    SAGELayer layer(4, 8, SAGELayer::Aggregator::Mean);
    auto out = layer.forward(A, H);

    ASSERT_EQ(out.rows(), 5u);
    ASSERT_EQ(out.cols(), 8u);
}

// Test 20: Identity weights, no bias, no activation
void test_sage_mean_identity() {
    auto A = make_undirected_csr(3, {{0,1},{1,2}});
    auto H = Tensor::dense(3, 2, {1,0, 0,1, 1,1});

    SAGELayer layer(2, 2, SAGELayer::Aggregator::Mean, false, Activation::None);
    layer.set_weight_neigh(Tensor::dense(2, 2, {1,0, 0,1}));
    layer.set_weight_self(Tensor::dense(2, 2, {1,0, 0,1}));

    auto out = layer.forward(A, H);

    // out = AGG_mean + H (both with identity weight)
    // AGG[0] = [0,1], H[0] = [1,0] → [1,1]
    // AGG[1] = [1,0.5], H[1] = [0,1] → [1,1.5]
    // AGG[2] = [0,1], H[2] = [1,1] → [1,2]
    ASSERT_NEAR(out.at(0, 0), 1.0f, 1e-5);
    ASSERT_NEAR(out.at(0, 1), 1.0f, 1e-5);
    ASSERT_NEAR(out.at(1, 0), 1.0f, 1e-5);
    ASSERT_NEAR(out.at(1, 1), 1.5f, 1e-5);
    ASSERT_NEAR(out.at(2, 0), 1.0f, 1e-5);
    ASSERT_NEAR(out.at(2, 1), 2.0f, 1e-5);
}

// Test 21: Zero weights → output is just bias
void test_sage_mean_zero_weights() {
    auto A = make_undirected_csr(3, {{0,1},{1,2}});
    auto H = Tensor::dense(3, 2, {5,5,5,5,5,5});

    SAGELayer layer(2, 2, SAGELayer::Aggregator::Mean, true, Activation::None);
    // Weights are zero by default
    layer.set_bias(Tensor::dense(1, 2, {0.5f, -0.5f}));

    auto out = layer.forward(A, H);

    for (std::size_t r = 0; r < 3; ++r) {
        ASSERT_NEAR(out.at(r, 0), 0.5f, 1e-6);
        ASSERT_NEAR(out.at(r, 1), -0.5f, 1e-6);
    }
}

// Test 22: ReLU clips negative output
void test_sage_mean_relu_clips() {
    auto A = make_undirected_csr(2, {{0,1}});
    auto H = Tensor::dense(2, 1, {1,1});

    SAGELayer layer(1, 1, SAGELayer::Aggregator::Mean, true, Activation::ReLU);
    layer.set_weight_neigh(Tensor::dense(1, 1, {1.0f}));
    layer.set_weight_self(Tensor::dense(1, 1, {1.0f}));
    layer.set_bias(Tensor::dense(1, 1, {-20.0f}));

    auto out = layer.forward(A, H);
    ASSERT_NEAR(out.at(0, 0), 0.0f, 1e-6);
    ASSERT_NEAR(out.at(1, 0), 0.0f, 1e-6);
}

// Test 23: No bias mode
void test_sage_mean_no_bias() {
    auto A = make_undirected_csr(3, {{0,1},{1,2}});
    auto H = Tensor::dense(3, 2, {1,0, 0,1, 1,1});

    SAGELayer layer(2, 2, SAGELayer::Aggregator::Mean, false, Activation::None);
    layer.set_weight_neigh(Tensor::dense(2, 2, {1,0, 0,1}));
    layer.set_weight_self(Tensor::dense(2, 2, {1,0, 0,1}));

    auto out = layer.forward(A, H);

    // Same as identity test
    ASSERT_NEAR(out.at(0, 0), 1.0f, 1e-5);
    ASSERT_NEAR(out.at(0, 1), 1.0f, 1e-5);
}

// Test 24: Linearity — 2*H should give 2*output (when no activation)
void test_sage_mean_linearity() {
    auto A = make_undirected_csr(3, {{0,1},{1,2}});

    SAGELayer layer(2, 2, SAGELayer::Aggregator::Mean, false, Activation::None);
    layer.set_weight_neigh(Tensor::dense(2, 2, {0.5,0.1, 0.2,0.3}));
    layer.set_weight_self(Tensor::dense(2, 2, {0.1,0.4, 0.3,0.2}));

    auto H1 = Tensor::dense(3, 2, {1,2, 3,4, 5,6});
    auto H2 = Tensor::dense(3, 2, {2,4, 6,8, 10,12});

    auto out1 = layer.forward(A, H1);
    auto out2 = layer.forward(A, H2);

    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 2; ++c) {
            ASSERT_NEAR(out2.at(r, c), 2.0f * out1.at(r, c), 1e-4);
        }
    }
}


// ============================================================================
//  Category 4: SAGELayer Max forward — 3-node path
// ============================================================================
//
//  Same graph and features as Mean test:
//    A = [[0,1,0],[1,0,1],[0,1,0]]
//    H = [[1,0],[0,1],[1,1]]
//
//  Max aggregation:
//    Node 0 neighbors: {1} → AGG[0] = [0, 1]
//    Node 1 neighbors: {0,2} → AGG[1] = [max(1,1), max(0,1)] = [1, 1]
//    Node 2 neighbors: {1} → AGG[2] = [0, 1]
//
//  W_neigh = [[1,0],[0,1]], W_self = [[1,0],[0,1]], no bias, no activation
//
//  h_neigh = AGG = [[0,1],[1,1],[0,1]]
//  h_self = H = [[1,0],[0,1],[1,1]]
//  out = h_neigh + h_self = [[1,1],[1,2],[1,2]]

// Test 25: Full max forward 3-node
void test_sage_max_forward_3node() {
    auto A = make_undirected_csr(3, {{0,1}, {1,2}});
    auto H = Tensor::dense(3, 2, {1,0, 0,1, 1,1});

    SAGELayer layer(2, 2, SAGELayer::Aggregator::Max, false, Activation::None);
    layer.set_weight_neigh(Tensor::dense(2, 2, {1,0, 0,1}));
    layer.set_weight_self(Tensor::dense(2, 2, {1,0, 0,1}));

    auto out = layer.forward(A, H);

    ASSERT_NEAR(out.at(0, 0), 1.0f, 1e-5);
    ASSERT_NEAR(out.at(0, 1), 1.0f, 1e-5);
    ASSERT_NEAR(out.at(1, 0), 1.0f, 1e-5);
    ASSERT_NEAR(out.at(1, 1), 2.0f, 1e-5);
    ASSERT_NEAR(out.at(2, 0), 1.0f, 1e-5);
    ASSERT_NEAR(out.at(2, 1), 2.0f, 1e-5);
}

// Test 26: Max with bias and ReLU
void test_sage_max_forward_with_bias_relu() {
    auto A = make_undirected_csr(3, {{0,1}, {1,2}});
    auto H = Tensor::dense(3, 2, {1,0, 0,1, 1,1});

    SAGELayer layer(2, 2, SAGELayer::Aggregator::Max, true, Activation::ReLU);
    layer.set_weight_neigh(Tensor::dense(2, 2, {1,0, 0,1}));
    layer.set_weight_self(Tensor::dense(2, 2, {1,0, 0,1}));
    layer.set_bias(Tensor::dense(1, 2, {0.1f, -0.1f}));

    auto out = layer.forward(A, H);

    // Base: [[1,1],[1,2],[1,2]] + bias → [[1.1,0.9],[1.1,1.9],[1.1,1.9]]
    // ReLU: all positive → unchanged
    ASSERT_NEAR(out.at(0, 0), 1.1f, 1e-5);
    ASSERT_NEAR(out.at(0, 1), 0.9f, 1e-5);
    ASSERT_NEAR(out.at(1, 0), 1.1f, 1e-5);
    ASSERT_NEAR(out.at(1, 1), 1.9f, 1e-5);
}

// Test 27: Max vs Mean — max should be >= mean for non-negative features
void test_sage_max_vs_mean() {
    auto A = make_undirected_csr(4, {{0,1},{0,2},{0,3}});
    auto H = Tensor::dense(4, 2, {0,0, 1,2, 3,1, 2,4});

    SAGELayer layer_mean(2, 2, SAGELayer::Aggregator::Mean, false, Activation::None);
    layer_mean.set_weight_neigh(Tensor::dense(2, 2, {1,0, 0,1}));
    layer_mean.set_weight_self(Tensor::dense(2, 2, {0,0, 0,0}));  // zero self

    SAGELayer layer_max(2, 2, SAGELayer::Aggregator::Max, false, Activation::None);
    layer_max.set_weight_neigh(Tensor::dense(2, 2, {1,0, 0,1}));
    layer_max.set_weight_self(Tensor::dense(2, 2, {0,0, 0,0}));

    auto out_mean = layer_mean.forward(A, H);
    auto out_max = layer_max.forward(A, H);

    // For node 0 (with 3 neighbors having non-negative features):
    // max_agg[0] >= mean_agg[0]
    ASSERT_TRUE(out_max.at(0, 0) >= out_mean.at(0, 0) - 1e-6);
    ASSERT_TRUE(out_max.at(0, 1) >= out_mean.at(0, 1) - 1e-6);
}

// Test 28: Max with negative features — picks least negative
void test_sage_max_negative_picks_least() {
    auto A = make_undirected_csr(3, {{0,1},{0,2}});
    auto H = Tensor::dense(3, 1, {0, -3, -1});

    SAGELayer layer(1, 1, SAGELayer::Aggregator::Max, false, Activation::None);
    layer.set_weight_neigh(Tensor::dense(1, 1, {1.0f}));
    layer.set_weight_self(Tensor::dense(1, 1, {0.0f}));  // suppress self

    auto out = layer.forward(A, H);
    // Node 0 neighbors: {1,2} → max(-3,-1) = -1
    ASSERT_NEAR(out.at(0, 0), -1.0f, 1e-5);
}

// Test 29: Isolated node with max — should get self contribution only
void test_sage_max_isolated_node_forward() {
    // Node 2 is isolated (no edges to it)
    auto A = make_undirected_csr(3, {{0,1}});
    auto H = Tensor::dense(3, 2, {1,2, 3,4, 5,6});

    SAGELayer layer(2, 2, SAGELayer::Aggregator::Max, false, Activation::None);
    layer.set_weight_neigh(Tensor::dense(2, 2, {1,0, 0,1}));
    layer.set_weight_self(Tensor::dense(2, 2, {1,0, 0,1}));

    auto out = layer.forward(A, H);

    // Node 2: max_agg = [0,0] (no neighbors), self = [5,6]
    // out[2] = [0,0] + [5,6] = [5,6]
    ASSERT_NEAR(out.at(2, 0), 5.0f, 1e-5);
    ASSERT_NEAR(out.at(2, 1), 6.0f, 1e-5);
}

// Test 30: Max forward output dimensions
void test_sage_max_output_dims() {
    auto A = make_undirected_csr(5, {{0,1},{1,2},{2,3},{3,4}});
    auto H = Tensor::dense(5, 4);

    SAGELayer layer(4, 8, SAGELayer::Aggregator::Max);
    auto out = layer.forward(A, H);
    ASSERT_EQ(out.rows(), 5u);
    ASSERT_EQ(out.cols(), 8u);
}

// Test 31: Max with dimension change
void test_sage_max_dim_change() {
    auto A = make_undirected_csr(4, {{0,1},{1,2},{2,3}});
    auto H = Tensor::dense(4, 8);

    SAGELayer layer(8, 2, SAGELayer::Aggregator::Max, true, Activation::ReLU);
    auto out = layer.forward(A, H);
    ASSERT_EQ(out.rows(), 4u);
    ASSERT_EQ(out.cols(), 2u);
}


// ============================================================================
//  Category 5: Multi-layer GraphSAGE
// ============================================================================

// Test 32: Two-layer Mean SAGE
void test_sage_two_layer_mean() {
    auto A = make_undirected_csr(5, {{0,1},{1,2},{2,3},{3,4}});
    auto H = Tensor::dense(5, 4);

    SAGELayer l1(4, 8, SAGELayer::Aggregator::Mean, true, Activation::ReLU);
    SAGELayer l2(8, 2, SAGELayer::Aggregator::Mean, false, Activation::None);

    auto h1 = l1.forward(A, H);
    ASSERT_EQ(h1.rows(), 5u);
    ASSERT_EQ(h1.cols(), 8u);

    auto h2 = l2.forward(A, h1);
    ASSERT_EQ(h2.rows(), 5u);
    ASSERT_EQ(h2.cols(), 2u);
}

// Test 33: Two-layer Max SAGE
void test_sage_two_layer_max() {
    auto A = make_undirected_csr(5, {{0,1},{1,2},{2,3},{3,4}});
    auto H = Tensor::dense(5, 4);

    SAGELayer l1(4, 8, SAGELayer::Aggregator::Max, true, Activation::ReLU);
    SAGELayer l2(8, 2, SAGELayer::Aggregator::Max, true, Activation::None);

    auto h1 = l1.forward(A, H);
    auto h2 = l2.forward(A, h1);
    ASSERT_EQ(h2.rows(), 5u);
    ASSERT_EQ(h2.cols(), 2u);
}

// Test 34: Mixed Mean+Max layers
void test_sage_mixed_layers() {
    auto A = make_undirected_csr(4, {{0,1},{1,2},{2,3},{0,3}});
    auto H = Tensor::dense(4, 4);

    SAGELayer l1(4, 8, SAGELayer::Aggregator::Mean, true, Activation::ReLU);
    SAGELayer l2(8, 4, SAGELayer::Aggregator::Max, true, Activation::ReLU);
    SAGELayer l3(4, 2, SAGELayer::Aggregator::Mean, false, Activation::None);

    auto h1 = l1.forward(A, H);
    auto h2 = l2.forward(A, h1);
    auto h3 = l3.forward(A, h2);

    ASSERT_EQ(h3.rows(), 4u);
    ASSERT_EQ(h3.cols(), 2u);
}

// Test 35: GraphSAGE → GCN stacking (mixed layer types)
void test_sage_gcn_stacking() {
    auto A = make_undirected_csr(4, {{0,1},{1,2},{2,3}});
    auto A_norm = gcn_norm(A);
    auto H = Tensor::dense(4, 4);

    SAGELayer sage(4, 8, SAGELayer::Aggregator::Mean, true, Activation::ReLU);
    GCNLayer gcn(8, 2, true, Activation::None);

    auto h1 = sage.forward(A, H);
    auto h2 = gcn.forward(A_norm, h1);

    ASSERT_EQ(h2.rows(), 4u);
    ASSERT_EQ(h2.cols(), 2u);
}


// ============================================================================
//  Category 6: 10-node graph tests
// ============================================================================

// Test 36: 10-node ring, Mean SAGE, uniform features
void test_10node_sage_mean_uniform() {
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < 10; ++i)
        edges.push_back({i, (i + 1) % 10});
    auto A = make_undirected_csr(10, edges);

    // All features = [1, 1]
    auto H = Tensor::dense(10, 2, std::vector<float>(20, 1.0f));

    SAGELayer layer(2, 2, SAGELayer::Aggregator::Mean, false, Activation::None);
    layer.set_weight_neigh(Tensor::dense(2, 2, {1,0, 0,1}));
    layer.set_weight_self(Tensor::dense(2, 2, {1,0, 0,1}));

    auto out = layer.forward(A, H);

    // Mean of 2 neighbors (both [1,1]) = [1,1]
    // + self [1,1] = [2,2]
    for (int i = 0; i < 10; ++i) {
        ASSERT_NEAR(out.at(i, 0), 2.0f, 1e-5);
        ASSERT_NEAR(out.at(i, 1), 2.0f, 1e-5);
    }
}

// Test 37: 10-node ring, Max SAGE, varying features
void test_10node_sage_max_varying() {
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < 10; ++i)
        edges.push_back({i, (i + 1) % 10});
    auto A = make_undirected_csr(10, edges);

    // Node i has feature [i, 10-i]
    std::vector<float> h_data;
    for (int i = 0; i < 10; ++i) {
        h_data.push_back(static_cast<float>(i));
        h_data.push_back(static_cast<float>(10 - i));
    }
    auto H = Tensor::dense(10, 2, h_data);

    SAGELayer layer(2, 2, SAGELayer::Aggregator::Max, false, Activation::None);
    layer.set_weight_neigh(Tensor::dense(2, 2, {1,0, 0,1}));
    layer.set_weight_self(Tensor::dense(2, 2, {0,0, 0,0}));  // suppress self

    auto out = layer.forward(A, H);

    // Node 0 neighbors: {1, 9} → max([1,9],[9,1]) = [9, 9]
    ASSERT_NEAR(out.at(0, 0), 9.0f, 1e-5);
    ASSERT_NEAR(out.at(0, 1), 9.0f, 1e-5);

    // Node 5 neighbors: {4, 6} → max([4,6],[6,4]) = [6, 6]
    ASSERT_NEAR(out.at(5, 0), 6.0f, 1e-5);
    ASSERT_NEAR(out.at(5, 1), 6.0f, 1e-5);
}

// Test 38: 10-node star graph, Mean
void test_10node_star_mean() {
    std::vector<std::pair<int,int>> edges;
    for (int i = 1; i < 10; ++i)
        edges.push_back({0, i});
    auto A = make_undirected_csr(10, edges);

    // Node features: node i has [i, 0]
    std::vector<float> h_data;
    for (int i = 0; i < 10; ++i) {
        h_data.push_back(static_cast<float>(i));
        h_data.push_back(0.0f);
    }
    auto H = Tensor::dense(10, 2, h_data);

    SAGELayer layer(2, 2, SAGELayer::Aggregator::Mean, false, Activation::None);
    layer.set_weight_neigh(Tensor::dense(2, 2, {1,0, 0,1}));
    layer.set_weight_self(Tensor::dense(2, 2, {0,0, 0,0}));

    auto out = layer.forward(A, H);

    // Node 0 has 9 neighbors with features [1,0],[2,0],...,[9,0]
    // Mean = [(1+2+...+9)/9, 0] = [45/9, 0] = [5, 0]
    ASSERT_NEAR(out.at(0, 0), 5.0f, 1e-4);
    ASSERT_NEAR(out.at(0, 1), 0.0f, 1e-4);

    // Node 1 has 1 neighbor (node 0) with feature [0, 0]
    ASSERT_NEAR(out.at(1, 0), 0.0f, 1e-4);
}

// Test 39: 10-node two-layer pipeline
void test_10node_sage_two_layer() {
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < 10; ++i)
        edges.push_back({i, (i + 1) % 10});
    auto A = make_undirected_csr(10, edges);

    std::vector<float> h_data;
    for (int i = 0; i < 10; ++i) {
        h_data.push_back(static_cast<float>(i) / 10.0f);
        h_data.push_back(1.0f - static_cast<float>(i) / 10.0f);
    }
    auto H = Tensor::dense(10, 2, h_data);

    SAGELayer l1(2, 4, SAGELayer::Aggregator::Mean, true, Activation::ReLU);
    SAGELayer l2(4, 2, SAGELayer::Aggregator::Mean, false, Activation::None);

    auto h1 = l1.forward(A, H);
    auto h2 = l2.forward(A, h1);

    ASSERT_EQ(h2.rows(), 10u);
    ASSERT_EQ(h2.cols(), 2u);

    for (std::size_t r = 0; r < 10; ++r) {
        for (std::size_t c = 0; c < 2; ++c) {
            ASSERT_TRUE(std::isfinite(h2.at(r, c)));
        }
    }
}

// Test 40: 100-node chain
void test_sage_100node_chain() {
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < 99; ++i)
        edges.push_back({i, i + 1});
    auto A = make_undirected_csr(100, edges);

    auto H = Tensor::dense(100, 4, std::vector<float>(400, 0.5f));

    SAGELayer layer(4, 8, SAGELayer::Aggregator::Mean, true, Activation::ReLU);
    auto out = layer.forward(A, H);

    ASSERT_EQ(out.rows(), 100u);
    ASSERT_EQ(out.cols(), 8u);

    for (std::size_t r = 0; r < 100; ++r)
        for (std::size_t c = 0; c < 8; ++c)
            ASSERT_TRUE(out.at(r, c) >= 0.0f);  // ReLU
}

// Test 41: Mean degree normalization correctness
void test_sage_mean_degree_normalization() {
    // Hub node 0 with 4 neighbors, leaf nodes 1-4 with 1 neighbor each
    auto A = make_undirected_csr(5, {{0,1},{0,2},{0,3},{0,4}});
    auto H = Tensor::dense(5, 1, {0, 2, 4, 6, 8});

    SAGELayer layer(1, 1, SAGELayer::Aggregator::Mean, false, Activation::None);
    layer.set_weight_neigh(Tensor::dense(1, 1, {1.0f}));
    layer.set_weight_self(Tensor::dense(1, 1, {0.0f}));

    auto out = layer.forward(A, H);

    // Node 0: mean of {2,4,6,8}/4 = 5.0
    ASSERT_NEAR(out.at(0, 0), 5.0f, 1e-5);

    // Node 1: mean of {0}/1 = 0.0
    ASSERT_NEAR(out.at(1, 0), 0.0f, 1e-5);

    // Node 3: mean of {0}/1 = 0.0
    ASSERT_NEAR(out.at(3, 0), 0.0f, 1e-5);
}

// Test 42: Max aggregate with star graph
void test_sage_max_star_graph() {
    auto A = make_undirected_csr(5, {{0,1},{0,2},{0,3},{0,4}});
    auto H = Tensor::dense(5, 2, {0,0, 1,10, 3,5, 2,8, 4,1});

    auto agg = sage_max_aggregate(A, H);

    // Node 0 neighbors: {1,2,3,4} → max per feature
    // f0: max(1,3,2,4) = 4
    // f1: max(10,5,8,1) = 10
    ASSERT_NEAR(agg.at(0, 0), 4.0f, 1e-6);
    ASSERT_NEAR(agg.at(0, 1), 10.0f, 1e-6);
}


// ============================================================================
//  Category 7: Error handling
// ============================================================================

// Test 43: forward with Dense A
void test_sage_forward_rejects_dense_A() {
    SAGELayer layer(2, 2);
    auto A = Tensor::dense(3, 3);
    auto H = Tensor::dense(3, 2);
    ASSERT_THROWS(layer.forward(A, H), std::invalid_argument);
}

// Test 44: forward with SparseCSR H
void test_sage_forward_rejects_sparse_H() {
    auto A = make_undirected_csr(3, {{0,1}});
    SAGELayer layer(2, 2);
    auto H = Tensor::sparse_csr(3, 2, {0,1,2,3}, {0,1,0}, {1,1,1});
    ASSERT_THROWS(layer.forward(A, H), std::invalid_argument);
}

// Test 45: wrong feature dimension
void test_sage_forward_wrong_features() {
    auto A = make_undirected_csr(3, {{0,1}});
    SAGELayer layer(4, 2);
    auto H = Tensor::dense(3, 2);  // 2 features vs expected 4
    ASSERT_THROWS(layer.forward(A, H), std::invalid_argument);
}

// Test 46: A.cols() != H.rows()
void test_sage_forward_dim_mismatch() {
    auto A = make_undirected_csr(3, {{0,1},{1,2}});
    SAGELayer layer(2, 2);
    auto H = Tensor::dense(5, 2);  // 5 rows vs A's 3 cols
    ASSERT_THROWS(layer.forward(A, H), std::invalid_argument);
}

// Test 47: sage_max_aggregate dimension mismatch
void test_sage_max_dim_mismatch() {
    auto A = make_undirected_csr(3, {{0,1}});
    auto H = Tensor::dense(5, 2);  // wrong number of rows
    ASSERT_THROWS(sage_max_aggregate(A, H), std::invalid_argument);
}

// Test 48: set_weight_neigh sparse
void test_sage_set_weight_neigh_sparse() {
    SAGELayer layer(2, 2);
    auto W = Tensor::sparse_csr(2, 2, {0,1,2}, {0,1}, {1,1});
    ASSERT_THROWS(layer.set_weight_neigh(std::move(W)), std::invalid_argument);
}

// Test 49: set_weight_self sparse
void test_sage_set_weight_self_sparse() {
    SAGELayer layer(2, 2);
    auto W = Tensor::sparse_csr(2, 2, {0,1,2}, {0,1}, {1,1});
    ASSERT_THROWS(layer.set_weight_self(std::move(W)), std::invalid_argument);
}

// Test 50: set_bias sparse
void test_sage_set_bias_sparse() {
    SAGELayer layer(2, 2);
    auto b = Tensor::sparse_csr(1, 2, {0,1}, {0}, {1});
    ASSERT_THROWS(layer.set_bias(std::move(b)), std::invalid_argument);
}

// Test 51: Error message for Dense A
void test_sage_forward_error_msg() {
    SAGELayer layer(2, 2);
    auto A = Tensor::dense(3, 3);
    auto H = Tensor::dense(3, 2);
    ASSERT_THROWS_MSG(layer.forward(A, H), std::invalid_argument, "SparseCSR");
}

// Test 52: set_weight_self wrong cols
void test_sage_set_weight_self_wrong_cols() {
    SAGELayer layer(4, 8);
    auto W = Tensor::dense(4, 4);  // wrong out_features
    ASSERT_THROWS(layer.set_weight_self(std::move(W)), std::invalid_argument);
}


// ============================================================================
//  Category 8: Edge / degenerate cases
// ============================================================================

// Test 53: Single node, Mean
void test_sage_single_node_mean() {
    // Single node, no edges
    auto A = Tensor::sparse_csr(1, 1, {0, 0}, {}, {});
    auto H = Tensor::dense(1, 2, {3.0f, 4.0f});

    SAGELayer layer(2, 2, SAGELayer::Aggregator::Mean, false, Activation::None);
    layer.set_weight_neigh(Tensor::dense(2, 2, {1,0, 0,1}));
    layer.set_weight_self(Tensor::dense(2, 2, {1,0, 0,1}));

    auto out = layer.forward(A, H);

    // No neighbors → AGG = [0,0], self = [3,4]
    // out = [0,0] + [3,4] = [3,4]
    ASSERT_NEAR(out.at(0, 0), 3.0f, 1e-5);
    ASSERT_NEAR(out.at(0, 1), 4.0f, 1e-5);
}

// Test 54: Single node, Max
void test_sage_single_node_max() {
    auto A = Tensor::sparse_csr(1, 1, {0, 0}, {}, {});
    auto H = Tensor::dense(1, 2, {3.0f, 4.0f});

    SAGELayer layer(2, 2, SAGELayer::Aggregator::Max, false, Activation::None);
    layer.set_weight_neigh(Tensor::dense(2, 2, {1,0, 0,1}));
    layer.set_weight_self(Tensor::dense(2, 2, {1,0, 0,1}));

    auto out = layer.forward(A, H);
    ASSERT_NEAR(out.at(0, 0), 3.0f, 1e-5);
    ASSERT_NEAR(out.at(0, 1), 4.0f, 1e-5);
}

// Test 55: All-zero features
void test_sage_all_zero_features() {
    auto A = make_undirected_csr(3, {{0,1},{1,2}});
    auto H = Tensor::dense(3, 2);  // zero-init

    SAGELayer layer(2, 2, SAGELayer::Aggregator::Mean, true, Activation::None);
    layer.set_bias(Tensor::dense(1, 2, {0.5f, -0.5f}));

    auto out = layer.forward(A, H);

    // All zeros + bias
    for (std::size_t r = 0; r < 3; ++r) {
        ASSERT_NEAR(out.at(r, 0), 0.5f, 1e-6);
        ASSERT_NEAR(out.at(r, 1), -0.5f, 1e-6);
    }
}

// Test 56: Disconnected components
void test_sage_disconnected() {
    // 0-1, 2-3 (two separate edges)
    auto A = make_undirected_csr(4, {{0,1},{2,3}});
    auto H = Tensor::dense(4, 1, {1,2,100,200});

    SAGELayer layer(1, 1, SAGELayer::Aggregator::Mean, false, Activation::None);
    layer.set_weight_neigh(Tensor::dense(1, 1, {1.0f}));
    layer.set_weight_self(Tensor::dense(1, 1, {1.0f}));

    auto out = layer.forward(A, H);

    // Node 0: mean_neigh = 2/1 = 2, self = 1, total = 3
    ASSERT_NEAR(out.at(0, 0), 3.0f, 1e-5);
    // Node 1: mean_neigh = 1/1 = 1, self = 2, total = 3
    ASSERT_NEAR(out.at(1, 0), 3.0f, 1e-5);
    // Node 2: mean_neigh = 200/1 = 200, self = 100, total = 300
    ASSERT_NEAR(out.at(2, 0), 300.0f, 1e-5);
    // Node 3: mean_neigh = 100/1 = 100, self = 200, total = 300
    ASSERT_NEAR(out.at(3, 0), 300.0f, 1e-5);
}

// Test 57: Fully connected K5
void test_sage_fully_connected() {
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j)
            edges.push_back({i, j});
    auto A = make_undirected_csr(5, edges);

    auto H = Tensor::dense(5, 1, {1,1,1,1,1});

    SAGELayer layer(1, 1, SAGELayer::Aggregator::Mean, false, Activation::None);
    layer.set_weight_neigh(Tensor::dense(1, 1, {1.0f}));
    layer.set_weight_self(Tensor::dense(1, 1, {1.0f}));

    auto out = layer.forward(A, H);

    // All nodes identical, mean of 4 neighbors = 1, self = 1, total = 2
    for (int i = 0; i < 5; ++i) {
        ASSERT_NEAR(out.at(i, 0), 2.0f, 1e-5);
    }
}

// Test 58: Directed graph (not symmetric adjacency)
void test_sage_directed_graph() {
    auto A = make_csr(3, {{0,1}, {1,2}});
    auto H = Tensor::dense(3, 2, {1,0, 0,1, 1,1});

    SAGELayer layer(2, 2, SAGELayer::Aggregator::Mean, false, Activation::None);
    layer.set_weight_neigh(Tensor::dense(2, 2, {1,0, 0,1}));
    layer.set_weight_self(Tensor::dense(2, 2, {1,0, 0,1}));

    auto out = layer.forward(A, H);

    // Node 0 neighbors: {1} → mean = [0,1], + self [1,0] = [1,1]
    ASSERT_NEAR(out.at(0, 0), 1.0f, 1e-5);
    ASSERT_NEAR(out.at(0, 1), 1.0f, 1e-5);

    // Node 2: no outgoing neighbors (0 degree) → AGG = [0,0], self = [1,1]
    ASSERT_NEAR(out.at(2, 0), 1.0f, 1e-5);
    ASSERT_NEAR(out.at(2, 1), 1.0f, 1e-5);
}

// Test 59: Manual ops vs layer (Mean)
void test_sage_matches_manual_ops_mean() {
    auto A = make_undirected_csr(3, {{0,1},{1,2}});
    auto H = Tensor::dense(3, 2, {0.5f, -0.3f, 1.2f, 0.8f, -0.1f, 0.6f});
    auto W_n = Tensor::dense(2, 2, {0.1f, 0.2f, 0.3f, 0.4f});
    auto W_s = Tensor::dense(2, 2, {0.5f, -0.1f, -0.2f, 0.6f});
    auto bias = Tensor::dense(1, 2, {0.01f, -0.01f});

    // Manual computation
    auto agg_sum = spmm(A, H);  // sum aggregation
    // Normalize by degree
    const auto& rp = A.row_ptr();
    float* agg_d = agg_sum.data().data();
    for (std::size_t i = 0; i < 3; ++i) {
        float deg = static_cast<float>(rp[i + 1] - rp[i]);
        if (deg > 0) {
            agg_d[i * 2]     /= deg;
            agg_d[i * 2 + 1] /= deg;
        }
    }
    auto h_neigh = matmul(agg_sum, W_n);
    auto h_self = matmul(H, W_s);

    // Combine
    float* hn = h_neigh.data().data();
    const float* hs = h_self.data().data();
    for (std::size_t i = 0; i < 6; ++i) hn[i] += hs[i];

    add_bias(h_neigh, bias);
    relu_inplace(h_neigh);

    // Layer computation
    SAGELayer layer(2, 2, SAGELayer::Aggregator::Mean, true, Activation::ReLU);
    layer.set_weight_neigh(Tensor::dense(2, 2, {0.1f, 0.2f, 0.3f, 0.4f}));
    layer.set_weight_self(Tensor::dense(2, 2, {0.5f, -0.1f, -0.2f, 0.6f}));
    layer.set_bias(Tensor::dense(1, 2, {0.01f, -0.01f}));

    auto out = layer.forward(A, H);

    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 2; ++c) {
            ASSERT_NEAR(out.at(r, c), h_neigh.at(r, c), 1e-5);
        }
    }
}

// Test 60: Manual ops vs layer (Max)
void test_sage_matches_manual_ops_max() {
    auto A = make_undirected_csr(3, {{0,1},{1,2}});
    auto H = Tensor::dense(3, 2, {1,0, 0,1, 1,1});
    auto W_n = Tensor::dense(2, 2, {1,0, 0,1});
    auto W_s = Tensor::dense(2, 2, {1,0, 0,1});

    // Manual max aggregate
    auto agg = sage_max_aggregate(A, H);
    auto h_neigh = matmul(agg, W_n);
    auto h_self = matmul(H, W_s);
    float* hn = h_neigh.data().data();
    const float* hs = h_self.data().data();
    for (std::size_t i = 0; i < 6; ++i) hn[i] += hs[i];

    // Layer
    SAGELayer layer(2, 2, SAGELayer::Aggregator::Max, false, Activation::None);
    layer.set_weight_neigh(Tensor::dense(2, 2, {1,0, 0,1}));
    layer.set_weight_self(Tensor::dense(2, 2, {1,0, 0,1}));

    auto out = layer.forward(A, H);

    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 2; ++c) {
            ASSERT_NEAR(out.at(r, c), h_neigh.at(r, c), 1e-5);
        }
    }
}


// ============================================================================
//  main
// ============================================================================
int main() {
    std::cout << "\n"
        "+=================================================================+\n"
        "|   TinyGNN — GraphSAGE Layer Unit Tests (Phase 6)               |\n"
        "|   Testing: sage_max_aggregate, SAGELayer (Mean, Max)           |\n"
        "+=================================================================+\n\n";

    // Category 1: sage_max_aggregate
    std::cout << "── 1. sage_max_aggregate ───────────────────────────────────\n";
    std::cout << "  Running test_sage_max_basic...\n";                    test_sage_max_basic();
    std::cout << "  Running test_sage_max_isolated_node...\n";            test_sage_max_isolated_node();
    std::cout << "  Running test_sage_max_single_neighbor...\n";          test_sage_max_single_neighbor();
    std::cout << "  Running test_sage_max_uniform_neighbors...\n";        test_sage_max_uniform_neighbors();
    std::cout << "  Running test_sage_max_negative_features...\n";        test_sage_max_negative_features();
    std::cout << "  Running test_sage_max_empty_graph...\n";              test_sage_max_empty_graph();
    std::cout << "  Running test_sage_max_rejects_dense_A...\n";          test_sage_max_rejects_dense_A();
    std::cout << "  Running test_sage_max_rejects_sparse_H...\n";         test_sage_max_rejects_sparse_H();

    // Category 2: SAGELayer construction
    std::cout << "\n── 2. SAGELayer Construction ────────────────────────────────\n";
    std::cout << "  Running test_sage_layer_construction_mean...\n";      test_sage_layer_construction_mean();
    std::cout << "  Running test_sage_layer_construction_max...\n";       test_sage_layer_construction_max();
    std::cout << "  Running test_sage_layer_weight_shapes...\n";          test_sage_layer_weight_shapes();
    std::cout << "  Running test_sage_layer_set_weight_neigh...\n";       test_sage_layer_set_weight_neigh();
    std::cout << "  Running test_sage_layer_set_weight_self...\n";        test_sage_layer_set_weight_self();
    std::cout << "  Running test_sage_layer_set_bias...\n";               test_sage_layer_set_bias();
    std::cout << "  Running test_sage_set_weight_neigh_wrong_shape...\n"; test_sage_set_weight_neigh_wrong_shape();
    std::cout << "  Running test_sage_set_bias_no_bias...\n";             test_sage_set_bias_no_bias();
    std::cout << "  Running test_sage_zero_features...\n";                test_sage_zero_features();

    // Category 3: SAGELayer Mean forward
    std::cout << "\n── 3. SAGELayer Mean Forward ────────────────────────────────\n";
    std::cout << "  Running test_sage_mean_forward_3node...\n";           test_sage_mean_forward_3node();
    std::cout << "  Running test_sage_mean_output_shape...\n";            test_sage_mean_output_shape();
    std::cout << "  Running test_sage_mean_identity...\n";                test_sage_mean_identity();
    std::cout << "  Running test_sage_mean_zero_weights...\n";            test_sage_mean_zero_weights();
    std::cout << "  Running test_sage_mean_relu_clips...\n";              test_sage_mean_relu_clips();
    std::cout << "  Running test_sage_mean_no_bias...\n";                 test_sage_mean_no_bias();
    std::cout << "  Running test_sage_mean_linearity...\n";               test_sage_mean_linearity();

    // Category 4: SAGELayer Max forward
    std::cout << "\n── 4. SAGELayer Max Forward ─────────────────────────────────\n";
    std::cout << "  Running test_sage_max_forward_3node...\n";            test_sage_max_forward_3node();
    std::cout << "  Running test_sage_max_forward_with_bias_relu...\n";   test_sage_max_forward_with_bias_relu();
    std::cout << "  Running test_sage_max_vs_mean...\n";                  test_sage_max_vs_mean();
    std::cout << "  Running test_sage_max_negative_picks_least...\n";     test_sage_max_negative_picks_least();
    std::cout << "  Running test_sage_max_isolated_node_forward...\n";    test_sage_max_isolated_node_forward();
    std::cout << "  Running test_sage_max_output_dims...\n";              test_sage_max_output_dims();
    std::cout << "  Running test_sage_max_dim_change...\n";               test_sage_max_dim_change();

    // Category 5: Multi-layer
    std::cout << "\n── 5. Multi-layer GraphSAGE ─────────────────────────────────\n";
    std::cout << "  Running test_sage_two_layer_mean...\n";               test_sage_two_layer_mean();
    std::cout << "  Running test_sage_two_layer_max...\n";                test_sage_two_layer_max();
    std::cout << "  Running test_sage_mixed_layers...\n";                 test_sage_mixed_layers();
    std::cout << "  Running test_sage_gcn_stacking...\n";                 test_sage_gcn_stacking();

    // Category 6: 10-node graph tests
    std::cout << "\n── 6. 10-node Graph Tests ───────────────────────────────────\n";
    std::cout << "  Running test_10node_sage_mean_uniform...\n";          test_10node_sage_mean_uniform();
    std::cout << "  Running test_10node_sage_max_varying...\n";           test_10node_sage_max_varying();
    std::cout << "  Running test_10node_star_mean...\n";                  test_10node_star_mean();
    std::cout << "  Running test_10node_sage_two_layer...\n";             test_10node_sage_two_layer();
    std::cout << "  Running test_sage_100node_chain...\n";                test_sage_100node_chain();
    std::cout << "  Running test_sage_mean_degree_normalization...\n";    test_sage_mean_degree_normalization();
    std::cout << "  Running test_sage_max_star_graph...\n";               test_sage_max_star_graph();

    // Category 7: Error handling
    std::cout << "\n── 7. Error Handling ────────────────────────────────────────\n";
    std::cout << "  Running test_sage_forward_rejects_dense_A...\n";      test_sage_forward_rejects_dense_A();
    std::cout << "  Running test_sage_forward_rejects_sparse_H...\n";     test_sage_forward_rejects_sparse_H();
    std::cout << "  Running test_sage_forward_wrong_features...\n";       test_sage_forward_wrong_features();
    std::cout << "  Running test_sage_forward_dim_mismatch...\n";         test_sage_forward_dim_mismatch();
    std::cout << "  Running test_sage_max_dim_mismatch...\n";             test_sage_max_dim_mismatch();
    std::cout << "  Running test_sage_set_weight_neigh_sparse...\n";      test_sage_set_weight_neigh_sparse();
    std::cout << "  Running test_sage_set_weight_self_sparse...\n";       test_sage_set_weight_self_sparse();
    std::cout << "  Running test_sage_set_bias_sparse...\n";              test_sage_set_bias_sparse();
    std::cout << "  Running test_sage_forward_error_msg...\n";            test_sage_forward_error_msg();
    std::cout << "  Running test_sage_set_weight_self_wrong_cols...\n";   test_sage_set_weight_self_wrong_cols();

    // Category 8: Edge / degenerate cases
    std::cout << "\n── 8. Edge / Degenerate Cases ───────────────────────────────\n";
    std::cout << "  Running test_sage_single_node_mean...\n";             test_sage_single_node_mean();
    std::cout << "  Running test_sage_single_node_max...\n";              test_sage_single_node_max();
    std::cout << "  Running test_sage_all_zero_features...\n";            test_sage_all_zero_features();
    std::cout << "  Running test_sage_disconnected...\n";                 test_sage_disconnected();
    std::cout << "  Running test_sage_fully_connected...\n";              test_sage_fully_connected();
    std::cout << "  Running test_sage_directed_graph...\n";               test_sage_directed_graph();
    std::cout << "  Running test_sage_matches_manual_ops_mean...\n";      test_sage_matches_manual_ops_mean();
    std::cout << "  Running test_sage_matches_manual_ops_max...\n";       test_sage_matches_manual_ops_max();

    // ── Summary ──
    std::cout << "\n"
        "=================================================================\n"
        "  Total : " << g_tests_run << "\n"
        "  Passed: " << g_tests_passed << "\n"
        "  Failed: " << g_tests_failed << "\n"
        "=================================================================\n\n";

    return g_tests_failed == 0 ? 0 : 1;
}
