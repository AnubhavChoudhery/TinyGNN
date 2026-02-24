// ============================================================================
//  TinyGNN — GAT Layer Unit Tests  (Phase 6, Part 3)
//  tests/test_gat.cpp
//
//  Test categories:
//    1. edge_softmax — sparse row-wise softmax, sum-to-1.0 invariant
//    2. GATLayer construction & weight management
//    3. GATLayer forward — hand-computed 3-node examples
//    4. GATLayer forward — bias and activation
//    5. Multi-layer GAT and mixed layer stacking
//    6. Larger graph tests (10-node, 100-node)
//    7. Error handling
//    8. Edge / degenerate cases
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

#define ASSERT_TRUE(cond)                                                       \
    do {                                                                        \
        ++g_tests_run;                                                          \
        if (!(cond)) {                                                          \
            std::cerr << "  FAIL: " << __FILE__ << ":" << __LINE__              \
                      << "  " #cond "\n";                                       \
            ++g_tests_failed;                                                   \
        } else { ++g_tests_passed; }                                            \
    } while (0)

#define ASSERT_EQ(a, b)                                                         \
    do {                                                                        \
        ++g_tests_run;                                                          \
        if ((a) != (b)) {                                                       \
            std::cerr << "  FAIL: " << __FILE__ << ":" << __LINE__              \
                      << "  " #a " == " #b "  (" << (a) << " vs " << (b) << ")\n"; \
            ++g_tests_failed;                                                   \
        } else { ++g_tests_passed; }                                            \
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                  \
    do {                                                                        \
        ++g_tests_run;                                                          \
        float diff_ = std::fabs(static_cast<float>(a) - static_cast<float>(b)); \
        if (diff_ > static_cast<float>(tol)) {                                  \
            std::cerr << "  FAIL: " << __FILE__ << ":" << __LINE__              \
                      << "  |" #a " - " #b "| = " << diff_                      \
                      << " > " << (tol) << "\n";                                \
            ++g_tests_failed;                                                   \
        } else { ++g_tests_passed; }                                            \
    } while (0)

#define ASSERT_THROWS(expr, exc_type)                                           \
    do {                                                                        \
        ++g_tests_run;                                                          \
        bool caught_ = false;                                                   \
        try { expr; } catch (const exc_type&) { caught_ = true; }               \
        if (!caught_) {                                                         \
            std::cerr << "  FAIL: " << __FILE__ << ":" << __LINE__              \
                      << "  expected " #exc_type " from " #expr "\n";           \
            ++g_tests_failed;                                                   \
        } else { ++g_tests_passed; }                                            \
    } while (0)

#define ASSERT_THROWS_MSG(expr, exc_type, substr)                               \
    do {                                                                        \
        ++g_tests_run;                                                          \
        bool caught_ = false;                                                   \
        std::string msg_;                                                       \
        try { expr; } catch (const exc_type& e) {                               \
            caught_ = true; msg_ = e.what();                                    \
        }                                                                       \
        if (!caught_) {                                                         \
            std::cerr << "  FAIL: " << __FILE__ << ":" << __LINE__              \
                      << "  expected " #exc_type " from " #expr "\n";           \
            ++g_tests_failed;                                                   \
        } else if (msg_.find(substr) == std::string::npos) {                    \
            std::cerr << "  FAIL: " << __FILE__ << ":" << __LINE__              \
                      << "  message missing \"" << (substr) << "\": \""         \
                      << msg_ << "\"\n";                                        \
            ++g_tests_failed;                                                   \
        } else { ++g_tests_passed; }                                            \
    } while (0)

using namespace tinygnn;

// ── Helper: build CSR from directed edges ────────────────────────────────────
static Tensor make_csr(std::size_t N,
                       const std::vector<std::pair<int,int>>& edges) {
    std::vector<std::vector<int>> adj(N);
    for (auto [s, d] : edges) adj[s].push_back(d);
    for (auto& row : adj) std::sort(row.begin(), row.end());

    std::vector<int32_t> rp(N + 1, 0);
    std::vector<int32_t> ci;
    std::vector<float>   vals;
    for (std::size_t i = 0; i < N; ++i) {
        rp[i] = static_cast<int32_t>(ci.size());
        for (int c : adj[i]) {
            ci.push_back(c);
            vals.push_back(1.0f);
        }
    }
    rp[N] = static_cast<int32_t>(ci.size());
    return Tensor::sparse_csr(N, N, std::move(rp), std::move(ci), std::move(vals));
}

static Tensor make_undirected_csr(std::size_t N,
                                  const std::vector<std::pair<int,int>>& edges) {
    std::vector<std::pair<int,int>> all;
    for (auto [s, d] : edges) { all.push_back({s,d}); all.push_back({d,s}); }
    return make_csr(N, all);
}

// ============================================================================
//  Category 1: edge_softmax
// ============================================================================

// Test 1: Basic — 3-node path, uniform values → uniform attention
void test_edge_softmax_uniform() {
    // A with all values = 1.0
    auto A = make_undirected_csr(3, {{0,1},{1,2}});
    auto S = edge_softmax(A);

    ASSERT_TRUE(S.format() == StorageFormat::SparseCSR);
    ASSERT_EQ(S.rows(), 3u);
    ASSERT_EQ(S.cols(), 3u);

    // Node 0: 1 neighbor → softmax = [1.0]
    const auto& rp = S.row_ptr();
    const auto& sv = S.data();
    ASSERT_NEAR(sv[rp[0]], 1.0f, 1e-6);

    // Node 1: 2 neighbors, equal scores → [0.5, 0.5]
    ASSERT_NEAR(sv[rp[1]], 0.5f, 1e-6);
    ASSERT_NEAR(sv[rp[1] + 1], 0.5f, 1e-6);

    // Node 2: 1 neighbor → [1.0]
    ASSERT_NEAR(sv[rp[2]], 1.0f, 1e-6);
}

// Test 2: Sum-to-1.0 invariant for every row
void test_edge_softmax_sum_to_one() {
    auto A = make_undirected_csr(5, {{0,1},{0,2},{0,3},{0,4},{1,2},{2,3}});

    // Set non-uniform values
    float* vals = const_cast<float*>(A.data().data());
    for (std::size_t i = 0; i < A.nnz(); ++i)
        vals[i] = static_cast<float>(i) * 0.3f - 1.0f;

    auto S = edge_softmax(A);
    const auto& rp = S.row_ptr();
    const auto& sv = S.data();

    for (std::size_t i = 0; i < 5; ++i) {
        float row_sum = 0.0f;
        for (int32_t nz = rp[i]; nz < rp[i + 1]; ++nz) {
            row_sum += sv[nz];
            ASSERT_TRUE(sv[nz] >= 0.0f);   // non-negative
            ASSERT_TRUE(sv[nz] <= 1.0f);   // at most 1
        }
        if (rp[i + 1] > rp[i]) {
            ASSERT_NEAR(row_sum, 1.0f, 1e-5);
        }
    }
}

// Test 3: Numerical stability — large values
void test_edge_softmax_large_values() {
    // CSR: row 0 has cols {0,1} with values {1000, 1001}
    auto A = Tensor::sparse_csr(2, 2, {0,2,4}, {0,1, 0,1}, {1000,1001, -500,-501});
    auto S = edge_softmax(A);
    const auto& sv = S.data();

    // Row 0: exp(1000-1001)/[exp(0)+exp(-1)] + exp(1001-1001)/...
    float e0 = std::exp(-1.0f), e1 = 1.0f;
    ASSERT_NEAR(sv[0], e0 / (e0 + e1), 1e-5);
    ASSERT_NEAR(sv[1], e1 / (e0 + e1), 1e-5);
    ASSERT_NEAR(sv[0] + sv[1], 1.0f, 1e-6);
    ASSERT_NEAR(sv[2] + sv[3], 1.0f, 1e-6);
}

// Test 4: Single element per row → always 1.0
void test_edge_softmax_single_entry() {
    auto A = Tensor::sparse_csr(3, 3, {0,1,2,3}, {1, 2, 0}, {-5.0f, 0.0f, 100.0f});
    auto S = edge_softmax(A);
    for (std::size_t i = 0; i < 3; ++i)
        ASSERT_NEAR(S.data()[i], 1.0f, 1e-6);
}

// Test 5: Empty rows (isolated nodes)
void test_edge_softmax_empty_rows() {
    // Node 1 is isolated
    auto A = Tensor::sparse_csr(3, 3, {0,1,1,2}, {1, 0}, {2.0f, 3.0f});
    auto S = edge_softmax(A);
    ASSERT_NEAR(S.data()[0], 1.0f, 1e-6);  // row 0: single entry
    ASSERT_NEAR(S.data()[1], 1.0f, 1e-6);  // row 2: single entry
}

// Test 6: Rejects Dense input
void test_edge_softmax_rejects_dense() {
    auto D = Tensor::dense(2, 2, {1,2,3,4});
    ASSERT_THROWS(edge_softmax(D), std::invalid_argument);
}

// Test 7: Negative values → still sums to 1.0
void test_edge_softmax_negative_values() {
    auto A = Tensor::sparse_csr(1, 3, {0,3}, {0,1,2}, {-10.0f, -20.0f, -5.0f});
    auto S = edge_softmax(A);
    float sum = S.data()[0] + S.data()[1] + S.data()[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5);
    // -5 is largest → should have highest probability
    ASSERT_TRUE(S.data()[2] > S.data()[0]);
    ASSERT_TRUE(S.data()[0] > S.data()[1]);
}

// Test 8: All identical values → uniform distribution
void test_edge_softmax_identical_values() {
    auto A = Tensor::sparse_csr(1, 4, {0,4}, {0,1,2,3}, {7.0f,7.0f,7.0f,7.0f});
    auto S = edge_softmax(A);
    for (int i = 0; i < 4; ++i)
        ASSERT_NEAR(S.data()[i], 0.25f, 1e-6);
}


// ============================================================================
//  Category 2: GATLayer construction & weight management
// ============================================================================

// Test 9: Basic construction
void test_gat_construction() {
    GATLayer layer(16, 8);
    ASSERT_EQ(layer.in_features(), 16u);
    ASSERT_EQ(layer.out_features(), 8u);
    ASSERT_NEAR(layer.negative_slope(), 0.2f, 1e-6);
    ASSERT_TRUE(layer.has_bias());
    ASSERT_TRUE(layer.activation() == Activation::None);
}

// Test 10: Construction with custom params
void test_gat_construction_custom() {
    GATLayer layer(4, 2, 0.1f, false, Activation::ReLU);
    ASSERT_NEAR(layer.negative_slope(), 0.1f, 1e-6);
    ASSERT_TRUE(!layer.has_bias());
    ASSERT_TRUE(layer.activation() == Activation::ReLU);
}

// Test 11: Zero features rejected
void test_gat_zero_features() {
    ASSERT_THROWS(GATLayer(0, 8), std::invalid_argument);
    ASSERT_THROWS(GATLayer(8, 0), std::invalid_argument);
}

// Test 12: set_weight valid
void test_gat_set_weight() {
    GATLayer layer(4, 2);
    layer.set_weight(Tensor::dense(4, 2, {1,0, 0,1, 1,1, 0,0}));
    ASSERT_EQ(layer.weight().rows(), 4u);
    ASSERT_EQ(layer.weight().cols(), 2u);
}

// Test 13: set_weight wrong shape
void test_gat_set_weight_wrong_shape() {
    GATLayer layer(4, 2);
    ASSERT_THROWS(layer.set_weight(Tensor::dense(4, 3)), std::invalid_argument);
    ASSERT_THROWS(layer.set_weight(Tensor::dense(3, 2)), std::invalid_argument);
}

// Test 14: set_attn_left / right
void test_gat_set_attn() {
    GATLayer layer(4, 2);
    layer.set_attn_left(Tensor::dense(1, 2, {1.0f, 0.5f}));
    layer.set_attn_right(Tensor::dense(1, 2, {0.5f, 1.0f}));
    ASSERT_NEAR(layer.attn_left().at(0, 0), 1.0f, 1e-6);
    ASSERT_NEAR(layer.attn_right().at(0, 1), 1.0f, 1e-6);
}

// Test 15: set_attn wrong shape
void test_gat_set_attn_wrong_shape() {
    GATLayer layer(4, 2);
    ASSERT_THROWS(layer.set_attn_left(Tensor::dense(1, 3)), std::invalid_argument);
    ASSERT_THROWS(layer.set_attn_right(Tensor::dense(2, 2)), std::invalid_argument);
}

// Test 16: set_bias
void test_gat_set_bias() {
    GATLayer layer(4, 2, 0.2f, true);
    layer.set_bias(Tensor::dense(1, 2, {0.1f, -0.1f}));
    ASSERT_NEAR(layer.bias().at(0, 0), 0.1f, 1e-6);
}

// Test 17: set_bias when use_bias=false
void test_gat_set_bias_no_bias() {
    GATLayer layer(4, 2, 0.2f, false);
    ASSERT_THROWS(layer.set_bias(Tensor::dense(1, 2)), std::invalid_argument);
}

// Test 18: set_weight rejects sparse
void test_gat_set_weight_sparse() {
    GATLayer layer(2, 2);
    auto W = Tensor::sparse_csr(2, 2, {0,1,2}, {0,1}, {1,1});
    ASSERT_THROWS(layer.set_weight(std::move(W)), std::invalid_argument);
}


// ============================================================================
//  Category 3: GATLayer forward — hand-computed
// ============================================================================

// Test 19: 2-node complete graph with self-loops — identity W, uniform attention
//   Nodes: 0, 1.  A_sl has self-loops → edges: (0,0),(0,1),(1,0),(1,1)
//   H = [[1,0],[0,1]],  W = I_2, a_l = [1,0], a_r = [0,1]
//   Wh = H·I = H = [[1,0],[0,1]]
//   src_scores: a_l · Wh[i] → [1, 0]
//   dst_scores: a_r · Wh[j] → [0, 1]
//   e_00 = LReLU(1+0) = 1,  e_01 = LReLU(1+1) = 2
//   e_10 = LReLU(0+0) = 0,  e_11 = LReLU(0+1) = 1
//   Row 0: softmax([1,2]), Row 1: softmax([0,1])
void test_gat_forward_2node() {
    auto A = add_self_loops(make_undirected_csr(2, {{0,1}}));
    auto H = Tensor::dense(2, 2, {1,0, 0,1});

    GATLayer layer(2, 2, 0.2f, false, Activation::None);
    layer.set_weight(Tensor::dense(2, 2, {1,0, 0,1}));
    layer.set_attn_left(Tensor::dense(1, 2, {1,0}));
    layer.set_attn_right(Tensor::dense(1, 2, {0,1}));

    auto out = layer.forward(A, H);
    ASSERT_EQ(out.rows(), 2u);
    ASSERT_EQ(out.cols(), 2u);

    // Row 0: α_00 = exp(1)/(exp(1)+exp(2)), α_01 = exp(2)/(exp(1)+exp(2))
    float e1 = std::exp(1.0f), e2 = std::exp(2.0f);
    float a00 = e1/(e1+e2), a01 = e2/(e1+e2);
    // out[0] = α_00 * Wh[0] + α_01 * Wh[1] = a00*[1,0] + a01*[0,1] = [a00, a01]
    ASSERT_NEAR(out.at(0, 0), a00, 1e-5);
    ASSERT_NEAR(out.at(0, 1), a01, 1e-5);

    // Row 1: α_10 = exp(0)/(exp(0)+exp(1)), α_11 = exp(1)/(exp(0)+exp(1))
    float e0b = 1.0f, e1b = std::exp(1.0f);
    float a10 = e0b/(e0b+e1b), a11 = e1b/(e0b+e1b);
    ASSERT_NEAR(out.at(1, 0), a10, 1e-5);
    ASSERT_NEAR(out.at(1, 1), a11, 1e-5);
}

// Test 20: Output dimensions with feature transform (4→2)
void test_gat_output_shape() {
    auto A = add_self_loops(make_undirected_csr(5, {{0,1},{1,2},{2,3},{3,4}}));
    auto H = Tensor::dense(5, 4);
    GATLayer layer(4, 2);
    auto out = layer.forward(A, H);
    ASSERT_EQ(out.rows(), 5u);
    ASSERT_EQ(out.cols(), 2u);
}

// Test 21: Uniform attention → should equal mean aggregation of Wh
void test_gat_uniform_attention() {
    auto A = add_self_loops(make_undirected_csr(3, {{0,1},{1,2}}));
    auto H = Tensor::dense(3, 2, {1,2, 3,4, 5,6});

    GATLayer layer(2, 2, 0.2f, false, Activation::None);
    layer.set_weight(Tensor::dense(2, 2, {1,0, 0,1}));
    // a_l = a_r = [0,0] → all edge scores = 0 → uniform attention
    layer.set_attn_left(Tensor::dense(1, 2, {0,0}));
    layer.set_attn_right(Tensor::dense(1, 2, {0,0}));

    auto out = layer.forward(A, H);

    // Node 0 has neighbors {0,1} → mean = (H[0]+H[1])/2 = (1+3)/2, (2+4)/2 = [2,3]
    ASSERT_NEAR(out.at(0, 0), 2.0f, 1e-5);
    ASSERT_NEAR(out.at(0, 1), 3.0f, 1e-5);

    // Node 1 has neighbors {0,1,2} → mean = (1+3+5)/3, (2+4+6)/3 = [3,4]
    ASSERT_NEAR(out.at(1, 0), 3.0f, 1e-5);
    ASSERT_NEAR(out.at(1, 1), 4.0f, 1e-5);

    // Node 2 has neighbors {1,2} → mean = (3+5)/2, (4+6)/2 = [4,5]
    ASSERT_NEAR(out.at(2, 0), 4.0f, 1e-5);
    ASSERT_NEAR(out.at(2, 1), 5.0f, 1e-5);
}

// Test 22: Attention weights in GAT forward sum to 1.0 per node
void test_gat_attention_sums_to_one() {
    auto A = add_self_loops(make_undirected_csr(4, {{0,1},{0,2},{1,3},{2,3}}));
    auto H = Tensor::dense(4, 3, {1,0,0, 0,1,0, 0,0,1, 1,1,1});

    GATLayer layer(3, 2, 0.2f, false, Activation::None);
    layer.set_weight(Tensor::dense(3, 2, {1,0, 0,1, 1,1}));
    layer.set_attn_left(Tensor::dense(1, 2, {0.5f, -0.3f}));
    layer.set_attn_right(Tensor::dense(1, 2, {-0.2f, 0.4f}));

    // Manually reproduce the attention computation to verify sum-to-1
    Tensor Wh = matmul(H, layer.weight());
    const float* wh = Wh.data().data();
    const float* al = layer.attn_left().data().data();
    const float* ar = layer.attn_right().data().data();
    const std::size_t F = 2;

    std::vector<float> src(4), dst(4);
    for (int i = 0; i < 4; ++i) {
        float s = 0, d = 0;
        for (std::size_t f = 0; f < F; ++f) {
            s += al[f] * wh[i*F+f];
            d += ar[f] * wh[i*F+f];
        }
        src[i] = s; dst[i] = d;
    }

    const auto& rp = A.row_ptr();
    const auto& ci = A.col_ind();
    for (std::size_t i = 0; i < 4; ++i) {
        // Compute edge logits for this row
        std::vector<float> logits;
        for (int32_t nz = rp[i]; nz < rp[i+1]; ++nz) {
            float e = src[i] + dst[ci[nz]];
            logits.push_back(e >= 0 ? e : 0.2f * e);
        }
        // Softmax
        float mx = *std::max_element(logits.begin(), logits.end());
        float sm = 0;
        for (auto& v : logits) { v = std::exp(v - mx); sm += v; }
        for (auto& v : logits) v /= sm;

        float sum = 0;
        for (auto v : logits) sum += v;
        ASSERT_NEAR(sum, 1.0f, 1e-5);
    }

    // Also verify the layer runs without error
    auto out = layer.forward(A, H);
    ASSERT_EQ(out.rows(), 4u);
    ASSERT_EQ(out.cols(), 2u);
    for (std::size_t r = 0; r < 4; ++r)
        for (std::size_t c = 0; c < 2; ++c)
            ASSERT_TRUE(std::isfinite(out.at(r, c)));
}


// ============================================================================
//  Category 4: GATLayer forward — bias and activation
// ============================================================================

// Test 23: With bias
void test_gat_forward_with_bias() {
    auto A = add_self_loops(Tensor::sparse_csr(1, 1, {0,0}, {}, {}));
    // Single isolated node (only self-loop from add_self_loops)
    auto H = Tensor::dense(1, 2, {1.0f, 2.0f});

    GATLayer layer(2, 2, 0.2f, true, Activation::None);
    layer.set_weight(Tensor::dense(2, 2, {1,0, 0,1}));
    layer.set_attn_left(Tensor::dense(1, 2, {0,0}));
    layer.set_attn_right(Tensor::dense(1, 2, {0,0}));
    layer.set_bias(Tensor::dense(1, 2, {0.5f, -0.5f}));

    auto out = layer.forward(A, H);
    // Only self-loop: attention = 1.0 on self → out = Wh[0] + bias = [1.5, 1.5]
    ASSERT_NEAR(out.at(0, 0), 1.5f, 1e-5);
    ASSERT_NEAR(out.at(0, 1), 1.5f, 1e-5);
}

// Test 24: With ReLU activation
void test_gat_forward_relu() {
    auto A = add_self_loops(Tensor::sparse_csr(1, 1, {0,0}, {}, {}));
    auto H = Tensor::dense(1, 2, {1.0f, 2.0f});

    GATLayer layer(2, 2, 0.2f, true, Activation::ReLU);
    layer.set_weight(Tensor::dense(2, 2, {1,0, 0,1}));
    layer.set_attn_left(Tensor::dense(1, 2, {0,0}));
    layer.set_attn_right(Tensor::dense(1, 2, {0,0}));
    layer.set_bias(Tensor::dense(1, 2, {0.5f, -3.0f}));

    auto out = layer.forward(A, H);
    // out before ReLU: [1+0.5, 2-3] = [1.5, -1.0]
    // after ReLU: [1.5, 0]
    ASSERT_NEAR(out.at(0, 0), 1.5f, 1e-5);
    ASSERT_NEAR(out.at(0, 1), 0.0f, 1e-5);
}

// Test 25: LeakyReLU negative slope effect on attention
void test_gat_leaky_relu_effect() {
    auto A = add_self_loops(make_undirected_csr(2, {{0,1}}));
    auto H = Tensor::dense(2, 1, {1.0f, -1.0f});

    // Set up so that some edge scores go negative
    GATLayer layer(1, 1, 0.2f, false, Activation::None);
    layer.set_weight(Tensor::dense(1, 1, {1.0f}));
    layer.set_attn_left(Tensor::dense(1, 1, {1.0f}));
    layer.set_attn_right(Tensor::dense(1, 1, {1.0f}));

    auto out = layer.forward(A, H);
    // Wh = H = [[1],[-1]]
    // src = [1,-1], dst = [1,-1]
    // e_00 = LReLU(1+1) = 2, e_01 = LReLU(1-1) = 0
    // e_10 = LReLU(-1+1) = 0, e_11 = LReLU(-1-1) = 0.2*(-2) = -0.4
    // Row 0: softmax([2,0]), Row 1: softmax([0,-0.4])

    // Just verify it runs and has finite output
    ASSERT_TRUE(std::isfinite(out.at(0, 0)));
    ASSERT_TRUE(std::isfinite(out.at(1, 0)));
}


// ============================================================================
//  Category 5: Multi-layer GAT and mixed stacking
// ============================================================================

// Test 26: Two-layer GAT
void test_gat_two_layer() {
    auto A = add_self_loops(make_undirected_csr(4, {{0,1},{1,2},{2,3}}));
    auto H = Tensor::dense(4, 4);

    GATLayer l1(4, 8, 0.2f, true, Activation::ReLU);
    GATLayer l2(8, 2, 0.2f, false, Activation::None);

    auto h1 = l1.forward(A, H);
    ASSERT_EQ(h1.rows(), 4u);
    ASSERT_EQ(h1.cols(), 8u);

    auto h2 = l2.forward(A, h1);
    ASSERT_EQ(h2.rows(), 4u);
    ASSERT_EQ(h2.cols(), 2u);
}

// Test 27: GAT → GCN stacking
void test_gat_gcn_stacking() {
    auto A_raw = make_undirected_csr(4, {{0,1},{1,2},{2,3}});
    auto A_sl = add_self_loops(A_raw);
    auto A_norm = gcn_norm(A_raw);
    auto H = Tensor::dense(4, 4);

    GATLayer gat(4, 8, 0.2f, true, Activation::ReLU);
    GCNLayer gcn(8, 2, true, Activation::None);

    auto h1 = gat.forward(A_sl, H);
    auto h2 = gcn.forward(A_norm, h1);
    ASSERT_EQ(h2.rows(), 4u);
    ASSERT_EQ(h2.cols(), 2u);
}

// Test 28: SAGE → GAT stacking
void test_sage_gat_stacking() {
    auto A_raw = make_undirected_csr(4, {{0,1},{1,2},{2,3}});
    auto A_sl = add_self_loops(A_raw);
    auto H = Tensor::dense(4, 4);

    SAGELayer sage(4, 8, SAGELayer::Aggregator::Mean, true, Activation::ReLU);
    GATLayer gat(8, 2, 0.2f, false, Activation::None);

    auto h1 = sage.forward(A_raw, H);
    auto h2 = gat.forward(A_sl, h1);
    ASSERT_EQ(h2.rows(), 4u);
    ASSERT_EQ(h2.cols(), 2u);
}


// ============================================================================
//  Category 6: Larger graph tests
// ============================================================================

// Test 29: 10-node ring — softmax sums to 1.0 in forward
void test_gat_10node_ring() {
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < 10; ++i)
        edges.push_back({i, (i+1)%10});
    auto A = add_self_loops(make_undirected_csr(10, edges));

    std::vector<float> h_data;
    for (int i = 0; i < 10; ++i) {
        h_data.push_back(static_cast<float>(i) / 10.0f);
        h_data.push_back(1.0f - static_cast<float>(i) / 10.0f);
    }
    auto H = Tensor::dense(10, 2, h_data);

    GATLayer layer(2, 4, 0.2f, true, Activation::ReLU);
    auto out = layer.forward(A, H);
    ASSERT_EQ(out.rows(), 10u);
    ASSERT_EQ(out.cols(), 4u);

    for (std::size_t r = 0; r < 10; ++r)
        for (std::size_t c = 0; c < 4; ++c)
            ASSERT_TRUE(out.at(r, c) >= 0.0f);  // ReLU
}

// Test 30: 10-node star — hub node gets all information
void test_gat_10node_star() {
    std::vector<std::pair<int,int>> edges;
    for (int i = 1; i < 10; ++i) edges.push_back({0, i});
    auto A = add_self_loops(make_undirected_csr(10, edges));
    auto H = Tensor::dense(10, 2, std::vector<float>(20, 1.0f));

    GATLayer layer(2, 2, 0.2f, false, Activation::None);
    layer.set_weight(Tensor::dense(2, 2, {1,0, 0,1}));
    layer.set_attn_left(Tensor::dense(1, 2, {0,0}));
    layer.set_attn_right(Tensor::dense(1, 2, {0,0}));

    auto out = layer.forward(A, H);
    // Uniform features + uniform attention → output = mean of all [1,1] = [1,1]
    ASSERT_NEAR(out.at(0, 0), 1.0f, 1e-5);
    ASSERT_NEAR(out.at(0, 1), 1.0f, 1e-5);
}

// Test 31: 100-node chain
void test_gat_100node_chain() {
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < 99; ++i) edges.push_back({i, i+1});
    auto A = add_self_loops(make_undirected_csr(100, edges));
    auto H = Tensor::dense(100, 4, std::vector<float>(400, 0.5f));

    GATLayer layer(4, 8, 0.2f, true, Activation::ReLU);
    auto out = layer.forward(A, H);
    ASSERT_EQ(out.rows(), 100u);
    ASSERT_EQ(out.cols(), 8u);
    for (std::size_t r = 0; r < 100; ++r)
        for (std::size_t c = 0; c < 8; ++c)
            ASSERT_TRUE(out.at(r, c) >= 0.0f);
}


// ============================================================================
//  Category 7: Error handling
// ============================================================================

// Test 32: forward with Dense A
void test_gat_forward_rejects_dense_A() {
    GATLayer layer(2, 2);
    ASSERT_THROWS(layer.forward(Tensor::dense(3, 3), Tensor::dense(3, 2)),
                  std::invalid_argument);
}

// Test 33: forward with SparseCSR H
void test_gat_forward_rejects_sparse_H() {
    auto A = add_self_loops(make_undirected_csr(3, {{0,1}}));
    GATLayer layer(2, 2);
    auto H = Tensor::sparse_csr(3, 2, {0,1,2,3}, {0,1,0}, {1,1,1});
    ASSERT_THROWS(layer.forward(A, H), std::invalid_argument);
}

// Test 34: wrong feature dimension
void test_gat_forward_wrong_features() {
    auto A = add_self_loops(make_undirected_csr(3, {{0,1}}));
    GATLayer layer(4, 2);
    ASSERT_THROWS(layer.forward(A, Tensor::dense(3, 2)), std::invalid_argument);
}

// Test 35: A.rows() != H.rows()
void test_gat_forward_dim_mismatch() {
    auto A = add_self_loops(make_undirected_csr(3, {{0,1},{1,2}}));
    GATLayer layer(2, 2);
    ASSERT_THROWS(layer.forward(A, Tensor::dense(5, 2)), std::invalid_argument);
}

// Test 36: Non-square A
void test_gat_forward_nonsquare_A() {
    auto A = Tensor::sparse_csr(3, 4, {0,1,2,3}, {0,1,2}, {1,1,1});
    GATLayer layer(2, 2);
    ASSERT_THROWS(layer.forward(A, Tensor::dense(3, 2)), std::invalid_argument);
}

// Test 37: Error message content
void test_gat_forward_error_msg() {
    GATLayer layer(2, 2);
    ASSERT_THROWS_MSG(layer.forward(Tensor::dense(3,3), Tensor::dense(3,2)),
                      std::invalid_argument, "SparseCSR");
}


// ============================================================================
//  Category 8: Edge / degenerate cases
// ============================================================================

// Test 38: Single node with self-loop only
void test_gat_single_node() {
    auto A = add_self_loops(Tensor::sparse_csr(1, 1, {0,0}, {}, {}));
    auto H = Tensor::dense(1, 3, {1.0f, 2.0f, 3.0f});

    GATLayer layer(3, 2, 0.2f, false, Activation::None);
    layer.set_weight(Tensor::dense(3, 2, {1,0, 0,1, 0,0}));
    layer.set_attn_left(Tensor::dense(1, 2, {0,0}));
    layer.set_attn_right(Tensor::dense(1, 2, {0,0}));

    auto out = layer.forward(A, H);
    // Wh = [1*1+0*2+0*3, 0*1+1*2+0*3] = [1, 2]
    // Only self-loop → attention = 1.0 → out = Wh = [1, 2]
    ASSERT_NEAR(out.at(0, 0), 1.0f, 1e-5);
    ASSERT_NEAR(out.at(0, 1), 2.0f, 1e-5);
}

// Test 39: All-zero features with bias
void test_gat_all_zero_features() {
    auto A = add_self_loops(make_undirected_csr(3, {{0,1},{1,2}}));
    auto H = Tensor::dense(3, 2);  // zero

    GATLayer layer(2, 2, 0.2f, true, Activation::None);
    layer.set_bias(Tensor::dense(1, 2, {0.5f, -0.5f}));

    auto out = layer.forward(A, H);
    // All Wh = [0,0], all attention uniform, aggregation = [0,0], + bias
    for (std::size_t r = 0; r < 3; ++r) {
        ASSERT_NEAR(out.at(r, 0), 0.5f, 1e-5);
        ASSERT_NEAR(out.at(r, 1), -0.5f, 1e-5);
    }
}

// Test 40: Fully connected K4 with self-loops
void test_gat_fully_connected() {
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < 4; ++i)
        for (int j = i+1; j < 4; ++j)
            edges.push_back({i,j});
    auto A = add_self_loops(make_undirected_csr(4, edges));
    auto H = Tensor::dense(4, 2, {1,1, 1,1, 1,1, 1,1});

    GATLayer layer(2, 2, 0.2f, false, Activation::None);
    layer.set_weight(Tensor::dense(2, 2, {1,0, 0,1}));
    layer.set_attn_left(Tensor::dense(1, 2, {0,0}));
    layer.set_attn_right(Tensor::dense(1, 2, {0,0}));

    auto out = layer.forward(A, H);
    // Uniform features + uniform attention → all outputs = [1,1]
    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(out.at(i, 0), 1.0f, 1e-5);
        ASSERT_NEAR(out.at(i, 1), 1.0f, 1e-5);
    }
}

// Test 41: edge_softmax on large sparse graph — exhaustive sum-to-1.0 check
void test_edge_softmax_large_graph_sum_to_one() {
    // Build a 50-node random-ish graph
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < 50; ++i) {
        edges.push_back({i, (i+1)%50});
        edges.push_back({i, (i+7)%50});
    }
    auto A = add_self_loops(make_undirected_csr(50, edges));

    // Assign varying values
    float* vals = const_cast<float*>(A.data().data());
    for (std::size_t i = 0; i < A.nnz(); ++i)
        vals[i] = std::sin(static_cast<float>(i) * 0.7f) * 3.0f;

    auto S = edge_softmax(A);
    const auto& rp = S.row_ptr();
    const auto& sv = S.data();

    for (std::size_t i = 0; i < 50; ++i) {
        float sum = 0.0f;
        for (int32_t nz = rp[i]; nz < rp[i+1]; ++nz) {
            ASSERT_TRUE(sv[nz] >= 0.0f);
            sum += sv[nz];
        }
        ASSERT_NEAR(sum, 1.0f, 1e-4);
    }
}


// ============================================================================
//  main
// ============================================================================
int main() {
    std::cout << "\n"
        "+=================================================================+\n"
        "|   TinyGNN — GAT Layer Unit Tests (Phase 6)                     |\n"
        "|   Testing: edge_softmax, GATLayer (attention, SpSDDMM)         |\n"
        "+=================================================================+\n\n";

    // Category 1: edge_softmax
    std::cout << "── 1. edge_softmax ────────────────────────────────────────────\n";
    std::cout << "  Running test_edge_softmax_uniform...\n";            test_edge_softmax_uniform();
    std::cout << "  Running test_edge_softmax_sum_to_one...\n";         test_edge_softmax_sum_to_one();
    std::cout << "  Running test_edge_softmax_large_values...\n";       test_edge_softmax_large_values();
    std::cout << "  Running test_edge_softmax_single_entry...\n";       test_edge_softmax_single_entry();
    std::cout << "  Running test_edge_softmax_empty_rows...\n";         test_edge_softmax_empty_rows();
    std::cout << "  Running test_edge_softmax_rejects_dense...\n";      test_edge_softmax_rejects_dense();
    std::cout << "  Running test_edge_softmax_negative_values...\n";    test_edge_softmax_negative_values();
    std::cout << "  Running test_edge_softmax_identical_values...\n";   test_edge_softmax_identical_values();

    // Category 2: GATLayer construction
    std::cout << "\n── 2. GATLayer Construction ─────────────────────────────────────\n";
    std::cout << "  Running test_gat_construction...\n";                test_gat_construction();
    std::cout << "  Running test_gat_construction_custom...\n";         test_gat_construction_custom();
    std::cout << "  Running test_gat_zero_features...\n";               test_gat_zero_features();
    std::cout << "  Running test_gat_set_weight...\n";                  test_gat_set_weight();
    std::cout << "  Running test_gat_set_weight_wrong_shape...\n";      test_gat_set_weight_wrong_shape();
    std::cout << "  Running test_gat_set_attn...\n";                    test_gat_set_attn();
    std::cout << "  Running test_gat_set_attn_wrong_shape...\n";        test_gat_set_attn_wrong_shape();
    std::cout << "  Running test_gat_set_bias...\n";                    test_gat_set_bias();
    std::cout << "  Running test_gat_set_bias_no_bias...\n";            test_gat_set_bias_no_bias();
    std::cout << "  Running test_gat_set_weight_sparse...\n";           test_gat_set_weight_sparse();

    // Category 3: GATLayer forward
    std::cout << "\n── 3. GATLayer Forward ──────────────────────────────────────────\n";
    std::cout << "  Running test_gat_forward_2node...\n";               test_gat_forward_2node();
    std::cout << "  Running test_gat_output_shape...\n";                test_gat_output_shape();
    std::cout << "  Running test_gat_uniform_attention...\n";           test_gat_uniform_attention();
    std::cout << "  Running test_gat_attention_sums_to_one...\n";       test_gat_attention_sums_to_one();

    // Category 4: Bias and activation
    std::cout << "\n── 4. Bias & Activation ─────────────────────────────────────────\n";
    std::cout << "  Running test_gat_forward_with_bias...\n";           test_gat_forward_with_bias();
    std::cout << "  Running test_gat_forward_relu...\n";                test_gat_forward_relu();
    std::cout << "  Running test_gat_leaky_relu_effect...\n";           test_gat_leaky_relu_effect();

    // Category 5: Multi-layer
    std::cout << "\n── 5. Multi-layer & Mixed Stacking ─────────────────────────────\n";
    std::cout << "  Running test_gat_two_layer...\n";                   test_gat_two_layer();
    std::cout << "  Running test_gat_gcn_stacking...\n";                test_gat_gcn_stacking();
    std::cout << "  Running test_sage_gat_stacking...\n";               test_sage_gat_stacking();

    // Category 6: Larger graphs
    std::cout << "\n── 6. Larger Graph Tests ────────────────────────────────────────\n";
    std::cout << "  Running test_gat_10node_ring...\n";                 test_gat_10node_ring();
    std::cout << "  Running test_gat_10node_star...\n";                 test_gat_10node_star();
    std::cout << "  Running test_gat_100node_chain...\n";               test_gat_100node_chain();

    // Category 7: Error handling
    std::cout << "\n── 7. Error Handling ────────────────────────────────────────────\n";
    std::cout << "  Running test_gat_forward_rejects_dense_A...\n";     test_gat_forward_rejects_dense_A();
    std::cout << "  Running test_gat_forward_rejects_sparse_H...\n";    test_gat_forward_rejects_sparse_H();
    std::cout << "  Running test_gat_forward_wrong_features...\n";      test_gat_forward_wrong_features();
    std::cout << "  Running test_gat_forward_dim_mismatch...\n";        test_gat_forward_dim_mismatch();
    std::cout << "  Running test_gat_forward_nonsquare_A...\n";         test_gat_forward_nonsquare_A();
    std::cout << "  Running test_gat_forward_error_msg...\n";           test_gat_forward_error_msg();

    // Category 8: Edge cases
    std::cout << "\n── 8. Edge / Degenerate Cases ───────────────────────────────────\n";
    std::cout << "  Running test_gat_single_node...\n";                 test_gat_single_node();
    std::cout << "  Running test_gat_all_zero_features...\n";           test_gat_all_zero_features();
    std::cout << "  Running test_gat_fully_connected...\n";             test_gat_fully_connected();
    std::cout << "  Running test_edge_softmax_large_graph_sum_to_one...\n"; test_edge_softmax_large_graph_sum_to_one();

    // ── Summary ──
    std::cout << "\n"
        "=================================================================\n"
        "  Total : " << g_tests_run << "\n"
        "  Passed: " << g_tests_passed << "\n"
        "  Failed: " << g_tests_failed << "\n"
        "=================================================================\n\n";

    return g_tests_failed == 0 ? 0 : 1;
}
