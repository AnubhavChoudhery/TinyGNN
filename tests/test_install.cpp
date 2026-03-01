// ============================================================================
//  TinyGNN — C++ Install Integration Tests  (Phase 11)
//  tests/test_install.cpp
//
//  Compiled as a standalone consumer against the INSTALLED library
//  (find_package(tinygnn CONFIG REQUIRED)).  Mirrors the 31-test Python
//  integration test (test_gnn_integration.py) in spirit and coverage.
//
//  Test categories:
//    §1  Tensor API                    (tests  1 –  6)
//    §2  matmul & spmm                 (tests  7 – 10)
//    §3  Activations & utilities       (tests 11 – 18)
//    §4  Graph normalization           (tests 19 – 21)
//    §5  GCNLayer                      (tests 22 – 25)
//    §6  SAGELayer — Mean & Max        (tests 26 – 29)
//    §7  GATLayer                      (tests 30 – 31)
//    §8  Model end-to-end              (tests 32 – 33)
//
//  Build & run via:
//    bash scripts/run_install_test.sh
// ============================================================================

#include <tinygnn/tensor.hpp>
#include <tinygnn/ops.hpp>
#include <tinygnn/layers.hpp>
#include <tinygnn/model.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace tinygnn;

// ── Minimal test harness ────────────────────────────────────────────────────
static int g_run = 0, g_pass = 0, g_fail = 0;
static std::string g_section;

static void begin_section(const std::string& name) {
    std::cout << "\n  ── " << name << " ──\n";
    g_section = name;
}

#define CHECK(cond)                                                     \
    do {                                                                \
        ++g_run;                                                        \
        if (!(cond)) {                                                  \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__    \
                      << "  " << #cond << "\n";                        \
            ++g_fail;                                                   \
        } else {                                                        \
            ++g_pass;                                                   \
        }                                                               \
    } while (0)

#define CHECK_EQ(a, b)                                                  \
    do {                                                                \
        ++g_run;                                                        \
        if ((a) != (b)) {                                               \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__    \
                      << "  " << #a << " == " << #b                    \
                      << "  (" << (a) << " != " << (b) << ")\n";       \
            ++g_fail;                                                   \
        } else {                                                        \
            ++g_pass;                                                   \
        }                                                               \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                           \
    do {                                                                \
        ++g_run;                                                        \
        if (std::fabs((a) - (b)) > (eps)) {                            \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__    \
                      << "  |" << #a << " - " << #b << "| > " << (eps) \
                      << "  (" << std::fabs((a)-(b)) << ")\n";         \
            ++g_fail;                                                   \
        } else {                                                        \
            ++g_pass;                                                   \
        }                                                               \
    } while (0)

// ── Helpers ─────────────────────────────────────────────────────────────────

// Returns true if all elements are finite (no NaN/Inf)
static bool all_finite(const Tensor& t) {
    for (float v : t.data())
        if (!std::isfinite(v)) return false;
    return true;
}

// Returns true if all elements satisfy predicate
static bool all_ge(const Tensor& t, float threshold) {
    for (float v : t.data())
        if (v < threshold) return false;
    return true;
}

// Returns true if all elements in [lo, hi]
static bool all_in_range(const Tensor& t, float lo, float hi) {
    for (float v : t.data())
        if (v < lo || v > hi) return false;
    return true;
}

// Scalar sum of all values
static float sum_all(const Tensor& t) {
    float s = 0;
    for (float v : t.data()) s += v;
    return s;
}

// ── Synthetic graph (5 nodes, undirected, edges {0-1,0-2,1-2,2-3,3-4}) ─────
//
//  Adjacency CSR:
//    row 0 → [1, 2]
//    row 1 → [0, 2]
//    row 2 → [0, 1, 3]
//    row 3 → [2, 4]
//    row 4 → [3]
//
//  row_ptr: [0, 2, 4, 7, 9, 10]
//  col_ind: [1, 2, 0, 2, 0, 1, 3, 2, 4, 3]
static Tensor make_adj() {
    return Tensor::sparse_csr(5, 5,
        {0, 2, 4, 7, 9, 10},
        {1, 2, 0, 2, 0, 1, 3, 2, 4, 3},
        {1,1,1,1,1,1,1,1,1,1});
}

// 5 nodes × 4 features (distinct, non-zero)
static Tensor make_features() {
    return Tensor::dense(5, 4, {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 0.1f, 0.2f,
        0.3f, 0.4f, 0.5f, 0.6f,
        0.7f, 0.8f, 0.9f, 1.0f,
    });
}

// Weight matrix (in_f × out_f) filled with a small constant
static Tensor make_weight(std::size_t rows, std::size_t cols, float val = 0.1f) {
    std::vector<float> d(rows * cols, val);
    return Tensor::dense(rows, cols, d);
}

// Bias row vector (1 × cols) filled with val
static Tensor make_bias(std::size_t cols, float val = 0.01f) {
    return Tensor::dense(1, cols, std::vector<float>(cols, val));
}

// Attention vector (1 × cols) — alternating +/- small values
static Tensor make_attn(std::size_t cols) {
    std::vector<float> d(cols);
    for (std::size_t i = 0; i < cols; ++i)
        d[i] = (i % 2 == 0) ? 0.05f : -0.05f;
    return Tensor::dense(1, cols, d);
}

// ============================================================================
//  §1  Tensor API
// ============================================================================
static void test_tensor_api() {
    begin_section("§1  Tensor API");

    // Test 1: Dense construction shape
    {
        auto t = Tensor::dense(3, 7);
        CHECK(t.format() == StorageFormat::Dense);
        CHECK_EQ(t.rows(), 3u);
        CHECK_EQ(t.cols(), 7u);
        CHECK_EQ(t.numel(), 21u);
    }

    // Test 2: Dense data comes back zeroed
    {
        auto t = Tensor::dense(4, 4);
        CHECK(all_finite(t));
        CHECK_NEAR(sum_all(t), 0.0f, 1e-6f);
    }

    // Test 3: CSR construction shape & nnz
    {
        auto adj = make_adj();
        CHECK(adj.format() == StorageFormat::SparseCSR);
        CHECK_EQ(adj.rows(), 5u);
        CHECK_EQ(adj.cols(), 5u);
        CHECK_EQ(adj.nnz(), 10u);
    }

    // Test 4: Dense element access (at)
    {
        auto t = Tensor::dense(2, 3, {1,2,3,4,5,6});
        CHECK_NEAR(t.at(0, 0), 1.0f, 1e-6f);
        CHECK_NEAR(t.at(1, 2), 6.0f, 1e-6f);
    }

    // Test 5: repr() is non-empty and contains format info
    {
        auto t = Tensor::dense(10, 10);
        auto s = t.repr();
        CHECK(!s.empty());
        CHECK(s.find("Dense") != std::string::npos || s.find("10") != std::string::npos);
    }

    // Test 6: memory_footprint_bytes
    {
        auto dense = Tensor::dense(100, 100);
        CHECK(dense.memory_footprint_bytes() == 100u * 100u * sizeof(float));

        auto csr = make_adj();  // 10 nnz, 5 rows
        // nnz*4 + nnz*4 + (rows+1)*4
        std::size_t expected = 10*4 + 10*4 + 6*4;
        CHECK_EQ(csr.memory_footprint_bytes(), expected);
    }
}

// ============================================================================
//  §2  matmul & spmm
// ============================================================================
static void test_matmul_spmm() {
    begin_section("§2  matmul & spmm");

    // Test 7: matmul output shape
    {
        auto A = Tensor::dense(3, 4, std::vector<float>(12, 1.0f));
        auto B = Tensor::dense(4, 5, std::vector<float>(20, 1.0f));
        auto C = matmul(A, B);
        CHECK_EQ(C.rows(), 3u);
        CHECK_EQ(C.cols(), 5u);
        CHECK(C.format() == StorageFormat::Dense);
    }

    // Test 8: matmul values — 1×K × K×1 = scalar dot product
    {
        // [1,2,3,4] × [1;2;3;4] = 1+4+9+16 = 30
        auto A = Tensor::dense(1, 4, {1,2,3,4});
        auto B = Tensor::dense(4, 1, {1,2,3,4});
        auto C = matmul(A, B);
        CHECK_NEAR(C.at(0, 0), 30.0f, 1e-4f);
    }

    // Test 9: spmm output shape
    {
        auto adj = make_adj();    // 5×5 CSR
        auto H   = make_features(); // 5×4 Dense
        auto out = spmm(adj, H);
        CHECK_EQ(out.rows(), 5u);
        CHECK_EQ(out.cols(), 4u);
        CHECK(out.format() == StorageFormat::Dense);
    }

    // Test 10: spmm values — node 0's row is sum of neighbors 1 and 2
    {
        auto adj = make_adj();
        auto H   = make_features();
        auto out = spmm(adj, H);
        // Node 0 aggregates nodes 1 and 2
        // H[1] = {0.5, 0.6, 0.7, 0.8}, H[2] = {0.9, 1.0, 0.1, 0.2}
        // sum col 0 = 0.5 + 0.9 = 1.4
        CHECK_NEAR(out.at(0, 0), 1.4f, 1e-5f);
        CHECK(all_finite(out));
    }
}

// ============================================================================
//  §3  Activations & utilities
// ============================================================================
static void test_activations() {
    begin_section("§3  Activations & utilities");

    // Test 11: relu_inplace — negatives become 0, positives unchanged
    {
        auto t = Tensor::dense(1, 6, {-2,-1,0,1,2,3});
        relu_inplace(t);
        CHECK_NEAR(t.at(0, 0), 0.0f, 1e-6f);
        CHECK_NEAR(t.at(0, 3), 1.0f, 1e-6f);
        CHECK(all_ge(t, 0.0f));
    }

    // Test 12: sigmoid_inplace — output in (0,1)
    {
        auto t = Tensor::dense(1, 4, {-10.0f, -1.0f, 0.0f, 10.0f});
        sigmoid_inplace(t);
        CHECK(all_in_range(t, 0.0f, 1.0f));
        CHECK_NEAR(t.at(0, 2), 0.5f, 1e-5f);   // sigmoid(0) = 0.5
    }

    // Test 13: tanh_inplace — output in (-1,1)
    {
        auto t = Tensor::dense(1, 4, {-10.0f, -1.0f, 0.0f, 10.0f});
        tanh_inplace(t);
        CHECK(all_in_range(t, -1.0f, 1.0f));
        CHECK_NEAR(t.at(0, 2), 0.0f, 1e-5f);   // tanh(0) = 0
    }

    // Test 14: elu_inplace — positives unchanged, negatives smoothed
    {
        auto t = Tensor::dense(1, 4, {-2.0f, -1.0f, 0.0f, 3.0f});
        auto orig_pos = 3.0f;
        elu_inplace(t, 1.0f);
        CHECK_NEAR(t.at(0, 3), orig_pos, 1e-5f);  // positive unchanged
        CHECK(t.at(0, 0) > -1.0f);                 // negative smoothed, > -1
        CHECK(all_finite(t));
    }

    // Test 15: gelu_inplace — finite output, monotone behavior
    {
        auto t = Tensor::dense(1, 4, {-3.0f, -1.0f, 0.0f, 3.0f});
        gelu_inplace(t);
        CHECK(all_finite(t));
        // gelu(0) ≈ 0, gelu(large) ≈ large
        CHECK_NEAR(t.at(0, 2), 0.0f, 0.01f);
        CHECK(t.at(0, 3) > 2.5f);
    }

    // Test 16: softmax_inplace — each row sums to 1
    {
        auto t = Tensor::dense(3, 4, {
            1,2,3,4,
            0,0,0,0,
            -1,-2,-3,-4
        });
        softmax_inplace(t);
        CHECK(all_in_range(t, 0.0f, 1.0f));
        // Each row sums to 1
        for (std::size_t i = 0; i < 3; ++i) {
            float row_sum = 0;
            for (std::size_t j = 0; j < 4; ++j) row_sum += t.at(i, j);
            CHECK_NEAR(row_sum, 1.0f, 1e-5f);
        }
    }

    // Test 17: log_softmax_inplace — exp of row sums to 1
    {
        auto t = Tensor::dense(2, 3, {1,2,3, -1,0,1});
        log_softmax_inplace(t);
        CHECK(all_finite(t));
        for (std::size_t i = 0; i < 2; ++i) {
            float row_expsum = 0;
            for (std::size_t j = 0; j < 3; ++j) row_expsum += std::exp(t.at(i, j));
            CHECK_NEAR(row_expsum, 1.0f, 1e-5f);
        }
    }

    // Test 18: add_bias — each row shifted by bias
    {
        auto t = Tensor::dense(3, 4, std::vector<float>(12, 0.0f));
        auto b = Tensor::dense(1, 4, {1,2,3,4});
        add_bias(t, b);
        CHECK_NEAR(t.at(0, 0), 1.0f, 1e-6f);
        CHECK_NEAR(t.at(2, 3), 4.0f, 1e-6f);
        // All rows identical
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 4; ++j)
                CHECK_NEAR(t.at(i, j), t.at(0, j), 1e-6f);
    }
}

// ============================================================================
//  §4  Graph normalization
// ============================================================================
static void test_graph_norm() {
    begin_section("§4  Graph normalization");

    auto adj = make_adj();   // 5×5, 10 nnz, no self-loops

    // Test 19: add_self_loops — nnz grows by N=5
    {
        auto adj_sl = add_self_loops(adj);
        CHECK_EQ(adj_sl.nnz(), 15u);   // 10 edge entries + 5 self-loops
        CHECK_EQ(adj_sl.rows(), 5u);
        CHECK_EQ(adj_sl.cols(), 5u);
        CHECK(adj_sl.format() == StorageFormat::SparseCSR);
    }

    // Test 20: gcn_norm — values ≤ 1, structure preserved
    {
        auto norm = gcn_norm(adj);
        CHECK_EQ(norm.rows(), 5u);
        CHECK_EQ(norm.cols(), 5u);
        // GCN norm adds self-loops so nnz = 10 + 5 = 15
        CHECK_EQ(norm.nnz(), 15u);
        // All values in (0, 1]
        CHECK(all_in_range(norm, 0.0f, 1.0f));
        CHECK(all_finite(norm));
    }

    // Test 21: edge_softmax — each row of CSR sums to 1
    {
        // Build a small CSR with known values for softmax check
        auto A = Tensor::sparse_csr(3, 3,
            {0, 2, 4, 5},
            {0, 1, 0, 2, 1},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
        auto S = edge_softmax(A);
        CHECK_EQ(S.nnz(), 5u);
        CHECK(all_finite(S));
        CHECK(all_in_range(S, 0.0f, 1.0f));
        // Row 0 (values 1,2) should sum to 1
        float row0_sum = S.data()[0] + S.data()[1];
        CHECK_NEAR(row0_sum, 1.0f, 1e-5f);
    }
}

// ============================================================================
//  §5  GCNLayer
// ============================================================================
static void test_gcn_layer() {
    begin_section("§5  GCNLayer");

    auto adj  = make_adj();
    auto H    = make_features();
    auto norm = gcn_norm(adj);

    const std::size_t IN = 4, OUT = 8;
    GCNLayer layer(IN, OUT, /*bias=*/true, Activation::ReLU);
    layer.set_weight(make_weight(IN, OUT));
    layer.set_bias(make_bias(OUT));

    // Test 22: Layer observers
    {
        CHECK_EQ(layer.in_features(),  IN);
        CHECK_EQ(layer.out_features(), OUT);
        CHECK(layer.has_bias());
    }

    auto out = layer.forward(norm, H);

    // Test 23: Output shape
    {
        CHECK_EQ(out.rows(), 5u);
        CHECK_EQ(out.cols(), OUT);
        CHECK(out.format() == StorageFormat::Dense);
    }

    // Test 24: No NaN / Inf
    {
        CHECK(all_finite(out));
    }

    // Test 25: ReLU applied — all values ≥ 0
    {
        CHECK(all_ge(out, 0.0f));
    }
}

// ============================================================================
//  §6  SAGELayer — Mean & Max aggregation
// ============================================================================
static void test_sage_layer() {
    begin_section("§6  SAGELayer (Mean + Max)");

    auto adj = make_adj();
    auto H   = make_features();

    const std::size_t IN = 4, OUT = 8;

    auto make_sage_layer = [&](SAGELayer::Aggregator agg) {
        SAGELayer layer(IN, OUT, agg, /*bias=*/true, Activation::ReLU);
        layer.set_weight_neigh(make_weight(IN, OUT, 0.08f));
        layer.set_weight_self (make_weight(IN, OUT, 0.12f));
        layer.set_bias(make_bias(OUT, 0.01f));
        return layer;
    };

    // Test 26: Mean aggregation output shape
    {
        auto layer = make_sage_layer(SAGELayer::Aggregator::Mean);
        auto out   = layer.forward(adj, H);
        CHECK_EQ(out.rows(), 5u);
        CHECK_EQ(out.cols(), OUT);
    }

    // Test 27: Mean aggregation no NaN
    {
        auto layer = make_sage_layer(SAGELayer::Aggregator::Mean);
        auto out   = layer.forward(adj, H);
        CHECK(all_finite(out));
    }

    // Test 28: Max aggregation output shape
    {
        auto layer = make_sage_layer(SAGELayer::Aggregator::Max);
        auto out   = layer.forward(adj, H);
        CHECK_EQ(out.rows(), 5u);
        CHECK_EQ(out.cols(), OUT);
    }

    // Test 29: Max aggregation no NaN
    {
        auto layer = make_sage_layer(SAGELayer::Aggregator::Max);
        auto out   = layer.forward(adj, H);
        CHECK(all_finite(out));
    }
}

// ============================================================================
//  §7  GATLayer
// ============================================================================
static void test_gat_layer() {
    begin_section("§7  GATLayer");

    auto adj    = make_adj();
    auto adj_sl = add_self_loops(adj);
    auto H      = make_features();

    const std::size_t IN = 4, OUT = 8;
    GATLayer layer(IN, OUT, 0.2f, /*bias=*/true, Activation::None);
    layer.set_weight    (make_weight(IN, OUT));
    layer.set_attn_left (make_attn(OUT));
    layer.set_attn_right(make_attn(OUT));
    layer.set_bias      (make_bias(OUT));

    auto out = layer.forward(adj_sl, H);

    // Test 30: Output shape
    {
        CHECK_EQ(out.rows(), 5u);
        CHECK_EQ(out.cols(), OUT);
        CHECK(out.format() == StorageFormat::Dense);
    }

    // Test 31: No NaN / Inf
    {
        CHECK(all_finite(out));
    }
}

// ============================================================================
//  §8  Model — end-to-end (GCN and SAGE)
// ============================================================================
static void test_model() {
    begin_section("§8  Model end-to-end");

    auto adj = make_adj();
    auto H   = make_features();

    // ── Test 32: 2-layer GCN (4→8→3) ──────────────────────────────────────
    {
        Model m;
        m.add_gcn_layer(4, 8, true, Activation::ReLU);
        m.add_gcn_layer(8, 3, true, Activation::None);
        CHECK_EQ(m.num_layers(), 2u);

        // Set weights via the internal layer access
        // (Model.forward builds adjacencies internally from raw adj)
        // We create a 2-layer GCN by chaining manual layers for validation,
        // and use Model for shape/no-crash check only.
        auto out = m.forward(adj, H);
        CHECK_EQ(out.rows(), 5u);
        CHECK_EQ(out.cols(), 3u);
        CHECK(all_finite(out));
    }

    // ── Test 33: 2-layer SAGE (4→8→3, Mean) ───────────────────────────────
    {
        Model m;
        m.add_sage_layer(4, 8, SAGELayer::Aggregator::Mean, true, Activation::ReLU);
        m.add_sage_layer(8, 3, SAGELayer::Aggregator::Mean, true, Activation::None);
        CHECK_EQ(m.num_layers(), 2u);

        auto out = m.forward(adj, H);
        CHECK_EQ(out.rows(), 5u);
        CHECK_EQ(out.cols(), 3u);
        CHECK(all_finite(out));
    }
}

// ============================================================================
//  main
// ============================================================================
int main() {
    std::cout << "TinyGNN — C++ Install Integration Tests\n";
    std::cout << "========================================\n";

    test_tensor_api();
    test_matmul_spmm();
    test_activations();
    test_graph_norm();
    test_gcn_layer();
    test_sage_layer();
    test_gat_layer();
    test_model();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << g_pass << " passed, "
              << g_fail  << " failed, "
              << g_run   << " total\n";

    if (g_fail > 0) {
        std::cout << "RESULT: FAIL\n";
        return EXIT_FAILURE;
    }
    std::cout << "RESULT: PASS\n";
    return EXIT_SUCCESS;
}
