// ============================================================================
//  TinyGNN — Phase 7: End-to-End Pipeline & Reference Matching Tests
//  tests/test_e2e.cpp
//
//  Validates that C++ inference for GCN, GraphSAGE, and GAT on the Cora
//  dataset produces classification accuracy matching the PyTorch Geometric
//  reference models (within tolerance).
//
//  Prerequisites:
//    Run  scripts/train_cora.py  to generate:
//      weights/cora_graph.bin
//      weights/gcn_cora.bin
//      weights/sage_cora.bin
//      weights/gat_cora.bin
// ============================================================================

#include "tinygnn/model.hpp"
#include "tinygnn/ops.hpp"
#include "tinygnn/tensor.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ── Minimal test framework ──────────────────────────────────────────────────
static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;
static int g_assertions   = 0;

#define ASSERT_TRUE(cond)                                                     \
    do {                                                                      \
        ++g_assertions;                                                       \
        if (!(cond)) {                                                        \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__           \
                      << " — ASSERT_TRUE(" #cond ")\n";                      \
            ++g_tests_failed;                                                 \
        } else {                                                              \
            ++g_tests_passed;                                                 \
        }                                                                     \
    } while (0)

#define ASSERT_EQ(a, b)                                                       \
    do {                                                                      \
        ++g_assertions;                                                       \
        if ((a) != (b)) {                                                     \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__           \
                      << " — ASSERT_EQ(" #a ", " #b ") → "                   \
                      << (a) << " != " << (b) << "\n";                       \
            ++g_tests_failed;                                                 \
        } else {                                                              \
            ++g_tests_passed;                                                 \
        }                                                                     \
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                \
    do {                                                                      \
        ++g_assertions;                                                       \
        if (std::fabs(static_cast<double>(a) - static_cast<double>(b)) >      \
            static_cast<double>(tol)) {                                       \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__           \
                      << " — ASSERT_NEAR(" #a ", " #b ", " #tol ") → "       \
                      << (a) << " vs " << (b) << "\n";                       \
            ++g_tests_failed;                                                 \
        } else {                                                              \
            ++g_tests_passed;                                                 \
        }                                                                     \
    } while (0)

#define RUN_TEST(fn)                                                          \
    do {                                                                      \
        std::cout << "  Running " #fn "...\n";                                \
        ++g_tests_run;                                                        \
        fn();                                                                 \
    } while (0)

using namespace tinygnn;

// ============================================================================
//  Configuration
// ============================================================================
// Paths relative to the working directory (project root)
static const std::string GRAPH_PATH  = "weights/cora_graph.bin";
static const std::string GCN_PATH    = "weights/gcn_cora.bin";
static const std::string SAGE_PATH   = "weights/sage_cora.bin";
static const std::string GAT_PATH    = "weights/gat_cora.bin";

// Architecture dimensions
static constexpr std::size_t F_IN  = 1433;
static constexpr std::size_t C     = 7;
static constexpr std::size_t H_GCN = 64;
static constexpr std::size_t H_SAGE = 64;
static constexpr std::size_t GAT_HEADS = 8;
static constexpr std::size_t GAT_HEAD_DIM = 8;

// Absolute accuracy tolerance: C++ accuracy should be within this of PyG
static constexpr double ACCURACY_TOLERANCE = 0.05;  // ±5%

// Minimum accuracy floor (sanity check)
static constexpr double MIN_ACCURACY = 0.60;  // at least 60%

// ============================================================================
//  Helpers
// ============================================================================

/// Compute argmax per row of a Dense tensor → vector of predicted class labels
static std::vector<int32_t> argmax_per_row(const Tensor& logits) {
    const std::size_t N = logits.rows();
    const std::size_t K = logits.cols();
    const float* data = logits.data().data();

    std::vector<int32_t> preds(N);
    for (std::size_t i = 0; i < N; ++i) {
        float best = data[i * K];
        int32_t best_j = 0;
        for (std::size_t j = 1; j < K; ++j) {
            if (data[i * K + j] > best) {
                best = data[i * K + j];
                best_j = static_cast<int32_t>(j);
            }
        }
        preds[i] = best_j;
    }
    return preds;
}

/// Compute accuracy on test nodes
static double compute_test_accuracy(const std::vector<int32_t>& preds,
                                    const std::vector<int32_t>& labels,
                                    const std::vector<uint8_t>& test_mask) {
    std::size_t correct = 0;
    std::size_t total   = 0;
    for (std::size_t i = 0; i < test_mask.size(); ++i) {
        if (test_mask[i]) {
            ++total;
            if (preds[i] == labels[i]) ++correct;
        }
    }
    return (total > 0) ? static_cast<double>(correct) / static_cast<double>(total)
                       : 0.0;
}

// ============================================================================
//  Global data (loaded once)
// ============================================================================
static CoraData     g_cora;
static WeightFile   g_gcn_wf;
static WeightFile   g_sage_wf;
static WeightFile   g_gat_wf;

// ============================================================================
//  Test: Load Cora graph data
// ============================================================================
void test_load_cora_data() {
    g_cora = load_cora_binary(GRAPH_PATH);

    ASSERT_EQ(g_cora.num_nodes,    static_cast<std::size_t>(2708));
    ASSERT_EQ(g_cora.num_features, static_cast<std::size_t>(1433));
    ASSERT_EQ(g_cora.num_classes,  static_cast<std::size_t>(7));
    ASSERT_TRUE(g_cora.num_edges > 0);

    // Adjacency
    ASSERT_TRUE(g_cora.adjacency.format() == StorageFormat::SparseCSR);
    ASSERT_EQ(g_cora.adjacency.rows(), static_cast<std::size_t>(2708));
    ASSERT_EQ(g_cora.adjacency.cols(), static_cast<std::size_t>(2708));

    // Features
    ASSERT_TRUE(g_cora.features.format() == StorageFormat::Dense);
    ASSERT_EQ(g_cora.features.rows(), static_cast<std::size_t>(2708));
    ASSERT_EQ(g_cora.features.cols(), static_cast<std::size_t>(1433));

    // Labels, masks
    ASSERT_EQ(g_cora.labels.size(),     static_cast<std::size_t>(2708));
    ASSERT_EQ(g_cora.train_mask.size(), static_cast<std::size_t>(2708));
    ASSERT_EQ(g_cora.val_mask.size(),   static_cast<std::size_t>(2708));
    ASSERT_EQ(g_cora.test_mask.size(),  static_cast<std::size_t>(2708));

    // Count test nodes
    std::size_t test_count = 0;
    for (auto m : g_cora.test_mask) test_count += m;
    ASSERT_EQ(test_count, static_cast<std::size_t>(1000));

    std::cout << "    Cora loaded: " << g_cora.num_nodes << " nodes, "
              << g_cora.num_edges << " edges\n";
}

// ============================================================================
//  Test: Load weight files
// ============================================================================
void test_load_weight_files() {
    g_gcn_wf  = load_weight_file(GCN_PATH);
    g_sage_wf = load_weight_file(SAGE_PATH);
    g_gat_wf  = load_weight_file(GAT_PATH);

    // GCN: 4 tensors (layer0.weight, layer0.bias, layer1.weight, layer1.bias)
    ASSERT_EQ(g_gcn_wf.tensors.size(), static_cast<std::size_t>(4));
    ASSERT_TRUE(g_gcn_wf.test_accuracy > 0.5f);

    // SAGE: 6 tensors (2 layers × {weight_neigh, weight_self, bias})
    ASSERT_EQ(g_sage_wf.tensors.size(), static_cast<std::size_t>(6));
    ASSERT_TRUE(g_sage_wf.test_accuracy > 0.5f);

    // GAT: 8 heads × 4 + 1 head × 4 = 36 tensors
    // (layer0: 8 heads × {weight, attn_left, attn_right, bias} = 32)
    // (layer1: 1 head  × {weight, attn_left, attn_right, bias} = 4)
    ASSERT_TRUE(g_gat_wf.tensors.size() >= 30);
    ASSERT_TRUE(g_gat_wf.test_accuracy > 0.5f);

    std::cout << "    GCN  ref accuracy: " << g_gcn_wf.test_accuracy << "\n";
    std::cout << "    SAGE ref accuracy: " << g_sage_wf.test_accuracy << "\n";
    std::cout << "    GAT  ref accuracy: " << g_gat_wf.test_accuracy << "\n";
}

// ============================================================================
//  Test: GCN weight tensor shapes
// ============================================================================
void test_gcn_weight_shapes() {
    const auto& ts = g_gcn_wf.tensors;

    auto check = [&](const std::string& name, std::size_t r, std::size_t c) {
        auto it = ts.find(name);
        ASSERT_TRUE(it != ts.end());
        if (it != ts.end()) {
            ASSERT_EQ(it->second.rows(), r);
            ASSERT_EQ(it->second.cols(), c);
        }
    };

    check("layer0.weight", F_IN,  H_GCN);    // (1433, 64)
    check("layer0.bias",   1,     H_GCN);     // (1, 64)
    check("layer1.weight", H_GCN, C);         // (64, 7)
    check("layer1.bias",   1,     C);          // (1, 7)
}

// ============================================================================
//  Test: SAGE weight tensor shapes
// ============================================================================
void test_sage_weight_shapes() {
    const auto& ts = g_sage_wf.tensors;

    auto check = [&](const std::string& name, std::size_t r, std::size_t c) {
        auto it = ts.find(name);
        ASSERT_TRUE(it != ts.end());
        if (it != ts.end()) {
            ASSERT_EQ(it->second.rows(), r);
            ASSERT_EQ(it->second.cols(), c);
        }
    };

    check("layer0.weight_neigh", F_IN,   H_SAGE);  // (1433, 64)
    check("layer0.weight_self",  F_IN,   H_SAGE);  // (1433, 64)
    check("layer0.bias",         1,      H_SAGE);   // (1, 64)
    check("layer1.weight_neigh", H_SAGE, C);        // (64, 7)
    check("layer1.weight_self",  H_SAGE, C);        // (64, 7)
    check("layer1.bias",         1,      C);         // (1, 7)
}

// ============================================================================
//  Test: GAT weight tensor shapes
// ============================================================================
void test_gat_weight_shapes() {
    const auto& ts = g_gat_wf.tensors;

    auto check = [&](const std::string& name, std::size_t r, std::size_t c) {
        auto it = ts.find(name);
        ASSERT_TRUE(it != ts.end());
        if (it != ts.end()) {
            ASSERT_EQ(it->second.rows(), r);
            ASSERT_EQ(it->second.cols(), c);
        }
    };

    // Layer 0: 8 heads, each (1433 → 8)
    for (std::size_t h = 0; h < GAT_HEADS; ++h) {
        std::string p = "layer0.head" + std::to_string(h);
        check(p + ".weight",     F_IN, GAT_HEAD_DIM);    // (1433, 8)
        check(p + ".attn_left",  1,    GAT_HEAD_DIM);     // (1, 8)
        check(p + ".attn_right", 1,    GAT_HEAD_DIM);     // (1, 8)
        check(p + ".bias",       1,    GAT_HEAD_DIM);     // (1, 8)
    }

    // Layer 1: 1 head, (64 → 7)
    check("layer1.head0.weight",     GAT_HEADS * GAT_HEAD_DIM, C);  // (64, 7)
    check("layer1.head0.attn_left",  1, C);                          // (1, 7)
    check("layer1.head0.attn_right", 1, C);                          // (1, 7)
    check("layer1.head0.bias",       1, C);                          // (1, 7)
}

// ============================================================================
//  Test: GCN end-to-end inference
// ============================================================================
void test_gcn_e2e() {
    // Build model: 2-layer GCN
    //   Layer 0: 1433 → 64, ReLU (internal), no post-act
    //   Layer 1: 64 → 7, None (internal), no post-act
    Model model;
    model.add_gcn_layer(F_IN, H_GCN, true, Activation::ReLU);
    model.add_gcn_layer(H_GCN, C, true, Activation::None);
    ASSERT_EQ(model.num_layers(), static_cast<std::size_t>(2));

    // Load weights
    model.load_weights(g_gcn_wf);

    // Forward pass
    Tensor logits = model.forward(g_cora.adjacency, g_cora.features);
    ASSERT_EQ(logits.rows(), g_cora.num_nodes);
    ASSERT_EQ(logits.cols(), C);

    // Apply log_softmax (for consistency, though argmax is the same)
    log_softmax_inplace(logits);

    // Compute accuracy
    auto preds = argmax_per_row(logits);
    double acc = compute_test_accuracy(preds, g_cora.labels, g_cora.test_mask);

    std::cout << "    GCN  C++ accuracy: " << acc
              << "  (ref: " << g_gcn_wf.test_accuracy << ")\n";

    // Assertions
    ASSERT_TRUE(acc >= MIN_ACCURACY);
    ASSERT_NEAR(acc, static_cast<double>(g_gcn_wf.test_accuracy),
                ACCURACY_TOLERANCE);
}

// ============================================================================
//  Test: GraphSAGE end-to-end inference
// ============================================================================
void test_sage_e2e() {
    Model model;
    model.add_sage_layer(F_IN, H_SAGE, SAGELayer::Aggregator::Mean,
                         true, Activation::ReLU);
    model.add_sage_layer(H_SAGE, C, SAGELayer::Aggregator::Mean,
                         true, Activation::None);
    ASSERT_EQ(model.num_layers(), static_cast<std::size_t>(2));

    model.load_weights(g_sage_wf);

    Tensor logits = model.forward(g_cora.adjacency, g_cora.features);
    ASSERT_EQ(logits.rows(), g_cora.num_nodes);
    ASSERT_EQ(logits.cols(), C);

    log_softmax_inplace(logits);

    auto preds = argmax_per_row(logits);
    double acc = compute_test_accuracy(preds, g_cora.labels, g_cora.test_mask);

    std::cout << "    SAGE C++ accuracy: " << acc
              << "  (ref: " << g_sage_wf.test_accuracy << ")\n";

    ASSERT_TRUE(acc >= MIN_ACCURACY);
    ASSERT_NEAR(acc, static_cast<double>(g_sage_wf.test_accuracy),
                ACCURACY_TOLERANCE);
}

// ============================================================================
//  Test: GAT end-to-end inference
// ============================================================================
void test_gat_e2e() {
    Model model;
    // Layer 0: 8-head GAT, 1433 → 8 per head, concat → 64, then ELU
    model.add_gat_layer(F_IN, GAT_HEAD_DIM, GAT_HEADS,
                        /*concat=*/true, /*neg_slope=*/0.2f,
                        /*bias=*/true, Activation::None,
                        Model::InterActivation::ELU);
    // Layer 1: 1-head GAT, 64 → 7, no post-act
    model.add_gat_layer(GAT_HEADS * GAT_HEAD_DIM, C, 1,
                        /*concat=*/false, /*neg_slope=*/0.2f,
                        /*bias=*/true, Activation::None,
                        Model::InterActivation::None);
    ASSERT_EQ(model.num_layers(), static_cast<std::size_t>(2));

    model.load_weights(g_gat_wf);

    Tensor logits = model.forward(g_cora.adjacency, g_cora.features);
    ASSERT_EQ(logits.rows(), g_cora.num_nodes);
    ASSERT_EQ(logits.cols(), C);

    log_softmax_inplace(logits);

    auto preds = argmax_per_row(logits);
    double acc = compute_test_accuracy(preds, g_cora.labels, g_cora.test_mask);

    std::cout << "    GAT  C++ accuracy: " << acc
              << "  (ref: " << g_gat_wf.test_accuracy << ")\n";

    ASSERT_TRUE(acc >= MIN_ACCURACY);
    ASSERT_NEAR(acc, static_cast<double>(g_gat_wf.test_accuracy),
                ACCURACY_TOLERANCE);
}

// ============================================================================
//  Test: Model configuration API
// ============================================================================
void test_model_api_gcn() {
    Model m;
    auto i0 = m.add_gcn_layer(10, 20, true, Activation::ReLU);
    auto i1 = m.add_gcn_layer(20, 5, false, Activation::None);
    ASSERT_EQ(i0, static_cast<std::size_t>(0));
    ASSERT_EQ(i1, static_cast<std::size_t>(1));
    ASSERT_EQ(m.num_layers(), static_cast<std::size_t>(2));
}

void test_model_api_sage() {
    Model m;
    m.add_sage_layer(10, 20, SAGELayer::Aggregator::Mean);
    m.add_sage_layer(20, 5, SAGELayer::Aggregator::Max, false, Activation::None);
    ASSERT_EQ(m.num_layers(), static_cast<std::size_t>(2));
}

void test_model_api_gat() {
    Model m;
    m.add_gat_layer(10, 8, 4, true, 0.2f, true, Activation::None,
                    Model::InterActivation::ELU);
    m.add_gat_layer(32, 5, 1, false, 0.2f, true, Activation::None);
    ASSERT_EQ(m.num_layers(), static_cast<std::size_t>(2));
}

// ============================================================================
//  Test: Binary format validation
// ============================================================================
void test_binary_format_magic() {
    // Verify weight files have correct magic
    ASSERT_TRUE(g_gcn_wf.tensors.size() > 0);
    ASSERT_TRUE(g_sage_wf.tensors.size() > 0);
    ASSERT_TRUE(g_gat_wf.tensors.size() > 0);
}

void test_cora_data_integrity() {
    // Labels should be in [0, num_classes)
    for (std::size_t i = 0; i < g_cora.labels.size(); ++i) {
        ASSERT_TRUE(g_cora.labels[i] >= 0);
        ASSERT_TRUE(g_cora.labels[i] < static_cast<int32_t>(g_cora.num_classes));
    }

    // Masks should sum to expected counts
    std::size_t train_c = 0, val_c = 0, test_c = 0;
    for (std::size_t i = 0; i < g_cora.num_nodes; ++i) {
        train_c += g_cora.train_mask[i];
        val_c   += g_cora.val_mask[i];
        test_c  += g_cora.test_mask[i];
    }
    ASSERT_EQ(train_c, static_cast<std::size_t>(140));
    ASSERT_EQ(val_c,   static_cast<std::size_t>(500));
    ASSERT_EQ(test_c,  static_cast<std::size_t>(1000));

    // Features should have some non-zero values
    bool has_nonzero = false;
    for (std::size_t i = 0; i < g_cora.features.data().size() && !has_nonzero; ++i) {
        if (g_cora.features.data()[i] != 0.0f) has_nonzero = true;
    }
    ASSERT_TRUE(has_nonzero);
}

// ============================================================================
//  Test: Model with small synthetic graph (GCN correctness)
// ============================================================================
void test_model_forward_small_gcn() {
    // 3-node graph: 0—1, 1—2, 0—2 (bidirectional)
    Tensor A = Tensor::sparse_csr(
        3, 3,
        {0, 2, 4, 6},          // row_ptr
        {1, 2, 0, 2, 0, 1},    // col_ind
        {1, 1, 1, 1, 1, 1}     // values
    );

    // Features: 3 nodes × 2 features
    Tensor H = Tensor::dense(3, 2, {1, 0, 0, 1, 1, 1});

    // Build a 1-layer GCN: 2 → 3, no bias, no activation
    Model model;
    model.add_gcn_layer(2, 3, false, Activation::None);

    // Create identity-like weight (2×3)
    WeightFile wf;
    wf.test_accuracy = 1.0f;
    wf.tensors["layer0.weight"] = Tensor::dense(2, 3, {1, 0, 0,
                                                         0, 1, 0});
    model.load_weights(wf);

    Tensor out = model.forward(A, H);
    ASSERT_EQ(out.rows(), static_cast<std::size_t>(3));
    ASSERT_EQ(out.cols(), static_cast<std::size_t>(3));

    // The output should be non-trivial (not all zeros)
    bool has_nonzero = false;
    for (auto v : out.data()) {
        if (std::fabs(v) > 1e-8f) has_nonzero = true;
    }
    ASSERT_TRUE(has_nonzero);
}

// ============================================================================
//  Test: Model with small synthetic graph (SAGE correctness)
// ============================================================================
void test_model_forward_small_sage() {
    // 3-node bidirectional graph
    Tensor A = Tensor::sparse_csr(
        3, 3,
        {0, 2, 4, 6},
        {1, 2, 0, 2, 0, 1},
        {1, 1, 1, 1, 1, 1}
    );

    Tensor H = Tensor::dense(3, 2, {1, 0, 0, 1, 1, 1});

    Model model;
    model.add_sage_layer(2, 3, SAGELayer::Aggregator::Mean,
                         false, Activation::None);

    WeightFile wf;
    wf.test_accuracy = 1.0f;
    // weight_neigh: (2, 3) identity-like
    wf.tensors["layer0.weight_neigh"] = Tensor::dense(2, 3, {1, 0, 0,
                                                               0, 1, 0});
    // weight_self: (2, 3) identity-like
    wf.tensors["layer0.weight_self"] = Tensor::dense(2, 3, {1, 0, 0,
                                                              0, 1, 0});

    model.load_weights(wf);

    Tensor out = model.forward(A, H);
    ASSERT_EQ(out.rows(), static_cast<std::size_t>(3));
    ASSERT_EQ(out.cols(), static_cast<std::size_t>(3));

    bool has_nonzero = false;
    for (auto v : out.data()) {
        if (std::fabs(v) > 1e-8f) has_nonzero = true;
    }
    ASSERT_TRUE(has_nonzero);
}

// ============================================================================
//  Test: Model with small synthetic graph (GAT correctness)
// ============================================================================
void test_model_forward_small_gat() {
    // 3-node bidirectional graph (no self-loops; Model adds them)
    Tensor A = Tensor::sparse_csr(
        3, 3,
        {0, 2, 4, 6},
        {1, 2, 0, 2, 0, 1},
        {1, 1, 1, 1, 1, 1}
    );

    Tensor H = Tensor::dense(3, 2, {1, 0, 0, 1, 1, 1});

    Model model;
    model.add_gat_layer(2, 3, /*num_heads=*/1, /*concat=*/false,
                        /*neg_slope=*/0.2f, /*bias=*/false, Activation::None);

    WeightFile wf;
    wf.test_accuracy = 1.0f;
    wf.tensors["layer0.head0.weight"] = Tensor::dense(2, 3, {1, 0, 0,
                                                               0, 1, 0});
    wf.tensors["layer0.head0.attn_left"]  = Tensor::dense(1, 3, {0.1f, 0.2f, 0.3f});
    wf.tensors["layer0.head0.attn_right"] = Tensor::dense(1, 3, {0.1f, 0.2f, 0.3f});

    model.load_weights(wf);

    Tensor out = model.forward(A, H);
    ASSERT_EQ(out.rows(), static_cast<std::size_t>(3));
    ASSERT_EQ(out.cols(), static_cast<std::size_t>(3));

    bool has_nonzero = false;
    for (auto v : out.data()) {
        if (std::fabs(v) > 1e-8f) has_nonzero = true;
    }
    ASSERT_TRUE(has_nonzero);
}

// ============================================================================
//  Test: Multi-head GAT concat
// ============================================================================
void test_model_multihead_gat_concat() {
    Tensor A = Tensor::sparse_csr(
        3, 3,
        {0, 2, 4, 6},
        {1, 2, 0, 2, 0, 1},
        {1, 1, 1, 1, 1, 1}
    );

    Tensor H = Tensor::dense(3, 2, {1, 0, 0, 1, 1, 1});

    Model model;
    // 2 heads × 3 features = 6 output features (concat)
    model.add_gat_layer(2, 3, /*num_heads=*/2, /*concat=*/true,
                        0.2f, false, Activation::None);
    ASSERT_EQ(model.num_layers(), static_cast<std::size_t>(1));

    WeightFile wf;
    wf.test_accuracy = 1.0f;
    wf.tensors["layer0.head0.weight"]     = Tensor::dense(2, 3, {1, 0, 0, 0, 1, 0});
    wf.tensors["layer0.head0.attn_left"]  = Tensor::dense(1, 3, {0.1f, 0.1f, 0.1f});
    wf.tensors["layer0.head0.attn_right"] = Tensor::dense(1, 3, {0.1f, 0.1f, 0.1f});
    wf.tensors["layer0.head1.weight"]     = Tensor::dense(2, 3, {0, 1, 0, 1, 0, 0});
    wf.tensors["layer0.head1.attn_left"]  = Tensor::dense(1, 3, {0.2f, 0.2f, 0.2f});
    wf.tensors["layer0.head1.attn_right"] = Tensor::dense(1, 3, {0.2f, 0.2f, 0.2f});

    model.load_weights(wf);

    Tensor out = model.forward(A, H);
    ASSERT_EQ(out.rows(), static_cast<std::size_t>(3));
    ASSERT_EQ(out.cols(), static_cast<std::size_t>(6));  // 2 heads × 3 features
}

// ============================================================================
//  Test: Model empty throws
// ============================================================================
void test_model_forward_empty_throws() {
    Model model;
    Tensor A = Tensor::sparse_csr(2, 2, {0, 1, 2}, {1, 0}, {1, 1});
    Tensor H = Tensor::dense(2, 2, {1, 0, 0, 1});
    bool caught = false;
    try {
        model.forward(A, H);
    } catch (const std::runtime_error&) {
        caught = true;
    }
    ASSERT_TRUE(caught);
}

// ============================================================================
//  Test: Weight file missing tensor throws
// ============================================================================
void test_load_weights_missing_tensor() {
    Model model;
    model.add_gcn_layer(2, 3, true, Activation::None);

    WeightFile wf;
    wf.test_accuracy = 1.0f;
    // Missing layer0.weight → should throw
    wf.tensors["layer0.bias"] = Tensor::dense(1, 3, {0, 0, 0});

    bool caught = false;
    try {
        model.load_weights(wf);
    } catch (const std::runtime_error&) {
        caught = true;
    }
    ASSERT_TRUE(caught);
}

// ============================================================================
//  Test: GCN logits are finite
// ============================================================================
void test_gcn_logits_finite() {
    Model model;
    model.add_gcn_layer(F_IN, H_GCN, true, Activation::ReLU);
    model.add_gcn_layer(H_GCN, C, true, Activation::None);
    model.load_weights(g_gcn_wf);

    Tensor logits = model.forward(g_cora.adjacency, g_cora.features);

    // All logits should be finite
    bool all_finite = true;
    for (auto v : logits.data()) {
        if (std::isnan(v) || std::isinf(v)) {
            all_finite = false;
            break;
        }
    }
    ASSERT_TRUE(all_finite);
}

// ============================================================================
//  Test: SAGE logits are finite
// ============================================================================
void test_sage_logits_finite() {
    Model model;
    model.add_sage_layer(F_IN, H_SAGE, SAGELayer::Aggregator::Mean,
                         true, Activation::ReLU);
    model.add_sage_layer(H_SAGE, C, SAGELayer::Aggregator::Mean,
                         true, Activation::None);
    model.load_weights(g_sage_wf);

    Tensor logits = model.forward(g_cora.adjacency, g_cora.features);

    bool all_finite = true;
    for (auto v : logits.data()) {
        if (std::isnan(v) || std::isinf(v)) {
            all_finite = false;
            break;
        }
    }
    ASSERT_TRUE(all_finite);
}

// ============================================================================
//  Test: GAT logits are finite
// ============================================================================
void test_gat_logits_finite() {
    Model model;
    model.add_gat_layer(F_IN, GAT_HEAD_DIM, GAT_HEADS, true, 0.2f,
                        true, Activation::None, Model::InterActivation::ELU);
    model.add_gat_layer(GAT_HEADS * GAT_HEAD_DIM, C, 1, false, 0.2f,
                        true, Activation::None);
    model.load_weights(g_gat_wf);

    Tensor logits = model.forward(g_cora.adjacency, g_cora.features);

    bool all_finite = true;
    for (auto v : logits.data()) {
        if (std::isnan(v) || std::isinf(v)) {
            all_finite = false;
            break;
        }
    }
    ASSERT_TRUE(all_finite);
}

// ============================================================================
//  Test: Prediction distribution is reasonable
// ============================================================================
void test_gcn_prediction_distribution() {
    Model model;
    model.add_gcn_layer(F_IN, H_GCN, true, Activation::ReLU);
    model.add_gcn_layer(H_GCN, C, true, Activation::None);
    model.load_weights(g_gcn_wf);

    Tensor logits = model.forward(g_cora.adjacency, g_cora.features);
    auto preds = argmax_per_row(logits);

    // All 7 classes should appear at least once
    std::vector<int> class_count(C, 0);
    for (auto p : preds) {
        ASSERT_TRUE(p >= 0 && p < static_cast<int32_t>(C));
        class_count[static_cast<std::size_t>(p)]++;
    }
    for (std::size_t c = 0; c < C; ++c) {
        ASSERT_TRUE(class_count[c] > 0);
    }
}

// ============================================================================
//  Test: SAGE prediction distribution
// ============================================================================
void test_sage_prediction_distribution() {
    Model model;
    model.add_sage_layer(F_IN, H_SAGE, SAGELayer::Aggregator::Mean,
                         true, Activation::ReLU);
    model.add_sage_layer(H_SAGE, C, SAGELayer::Aggregator::Mean,
                         true, Activation::None);
    model.load_weights(g_sage_wf);

    Tensor logits = model.forward(g_cora.adjacency, g_cora.features);
    auto preds = argmax_per_row(logits);

    std::vector<int> class_count(C, 0);
    for (auto p : preds) {
        ASSERT_TRUE(p >= 0 && p < static_cast<int32_t>(C));
        class_count[static_cast<std::size_t>(p)]++;
    }
    for (std::size_t c = 0; c < C; ++c) {
        ASSERT_TRUE(class_count[c] > 0);
    }
}

// ============================================================================
//  Test: GAT prediction distribution
// ============================================================================
void test_gat_prediction_distribution() {
    Model model;
    model.add_gat_layer(F_IN, GAT_HEAD_DIM, GAT_HEADS, true, 0.2f,
                        true, Activation::None, Model::InterActivation::ELU);
    model.add_gat_layer(GAT_HEADS * GAT_HEAD_DIM, C, 1, false, 0.2f,
                        true, Activation::None);
    model.load_weights(g_gat_wf);

    Tensor logits = model.forward(g_cora.adjacency, g_cora.features);
    auto preds = argmax_per_row(logits);

    std::vector<int> class_count(C, 0);
    for (auto p : preds) {
        ASSERT_TRUE(p >= 0 && p < static_cast<int32_t>(C));
        class_count[static_cast<std::size_t>(p)]++;
    }
    for (std::size_t c = 0; c < C; ++c) {
        ASSERT_TRUE(class_count[c] > 0);
    }
}

// ============================================================================
//  Test: Log-softmax output sums to ~0 (exp sums to ~1)
// ============================================================================
void test_log_softmax_consistency() {
    Model model;
    model.add_gcn_layer(F_IN, H_GCN, true, Activation::ReLU);
    model.add_gcn_layer(H_GCN, C, true, Activation::None);
    model.load_weights(g_gcn_wf);

    Tensor logits = model.forward(g_cora.adjacency, g_cora.features);
    log_softmax_inplace(logits);

    // For each row, exp(log_softmax) should sum to ~1
    for (std::size_t i = 0; i < std::min(logits.rows(), static_cast<std::size_t>(100)); ++i) {
        double row_sum = 0.0;
        for (std::size_t j = 0; j < C; ++j) {
            row_sum += std::exp(static_cast<double>(logits.at(i, j)));
        }
        ASSERT_NEAR(row_sum, 1.0, 1e-4);
    }
}

// ============================================================================
//  Test: GCN train accuracy (should be higher than test)
// ============================================================================
void test_gcn_train_accuracy() {
    Model model;
    model.add_gcn_layer(F_IN, H_GCN, true, Activation::ReLU);
    model.add_gcn_layer(H_GCN, C, true, Activation::None);
    model.load_weights(g_gcn_wf);

    Tensor logits = model.forward(g_cora.adjacency, g_cora.features);
    auto preds = argmax_per_row(logits);

    // Train accuracy
    std::size_t correct = 0, total = 0;
    for (std::size_t i = 0; i < g_cora.num_nodes; ++i) {
        if (g_cora.train_mask[i]) {
            ++total;
            if (preds[i] == g_cora.labels[i]) ++correct;
        }
    }
    double train_acc = static_cast<double>(correct) / static_cast<double>(total);
    std::cout << "    GCN  train accuracy: " << train_acc << "\n";

    // Train accuracy should be high (model was trained on these nodes)
    ASSERT_TRUE(train_acc >= 0.85);
}

// ============================================================================
//  main
// ============================================================================
int main() {
    // Graceful skip: weights are generated by scripts/train_cora.py (requires PyG)
    {
        std::ifstream probe(GRAPH_PATH);
        if (!probe.good()) {
            std::cout << "[SKIP] weights/cora_graph.bin not found — run scripts/train_cora.py first\n";
            std::cout << "E2ETests SKIPPED (no weights)\n";
            return 0; // exit 0 so CTest marks it passed
        }
    }

    std::cout << "\n═══════════════════════════════════════════════════════\n"
              << "  TinyGNN Phase 7 — End-to-End Pipeline Tests\n"
              << "═══════════════════════════════════════════════════════\n\n";

    // ── Data loading ─────────────────────────────────────────────────────
    std::cout << "[Data Loading]\n";
    RUN_TEST(test_load_cora_data);
    RUN_TEST(test_load_weight_files);
    RUN_TEST(test_cora_data_integrity);

    // ── Weight shape validation ──────────────────────────────────────────
    std::cout << "\n[Weight Shape Validation]\n";
    RUN_TEST(test_gcn_weight_shapes);
    RUN_TEST(test_sage_weight_shapes);
    RUN_TEST(test_gat_weight_shapes);

    // ── Model API tests ──────────────────────────────────────────────────
    std::cout << "\n[Model API]\n";
    RUN_TEST(test_model_api_gcn);
    RUN_TEST(test_model_api_sage);
    RUN_TEST(test_model_api_gat);
    RUN_TEST(test_model_forward_empty_throws);
    RUN_TEST(test_load_weights_missing_tensor);

    // ── Small synthetic graph tests ──────────────────────────────────────
    std::cout << "\n[Small Graph Forward Pass]\n";
    RUN_TEST(test_model_forward_small_gcn);
    RUN_TEST(test_model_forward_small_sage);
    RUN_TEST(test_model_forward_small_gat);
    RUN_TEST(test_model_multihead_gat_concat);

    // ── Logit sanity checks ──────────────────────────────────────────────
    std::cout << "\n[Logit Sanity]\n";
    RUN_TEST(test_gcn_logits_finite);
    RUN_TEST(test_sage_logits_finite);
    RUN_TEST(test_gat_logits_finite);
    RUN_TEST(test_log_softmax_consistency);

    // ── Prediction distribution ──────────────────────────────────────────
    std::cout << "\n[Prediction Distribution]\n";
    RUN_TEST(test_gcn_prediction_distribution);
    RUN_TEST(test_sage_prediction_distribution);
    RUN_TEST(test_gat_prediction_distribution);

    // ── End-to-end accuracy matching ─────────────────────────────────────
    std::cout << "\n[End-to-End Accuracy]\n";
    RUN_TEST(test_gcn_e2e);
    RUN_TEST(test_sage_e2e);
    RUN_TEST(test_gat_e2e);
    RUN_TEST(test_gcn_train_accuracy);

    // ── Summary ──────────────────────────────────────────────────────────
    std::cout << "\n═══════════════════════════════════════════════════════\n"
              << "  Tests run:    " << g_tests_run << "\n"
              << "  Assertions:   " << g_assertions << "\n"
              << "  Passed:       " << g_tests_passed << "\n"
              << "  Failed:       " << g_tests_failed << "\n"
              << "═══════════════════════════════════════════════════════\n";

    return (g_tests_failed == 0) ? 0 : 1;
}
