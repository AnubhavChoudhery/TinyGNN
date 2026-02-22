// ============================================================================
//  TinyGNN — GraphLoader Unit Tests  (Phase 2)
//  Dependency-free test harness (same framework as test_tensor.cpp)
// ============================================================================
//
//  Test categories:
//    1. Edge CSV parsing                   (tests  1 –  6)
//    2. Feature CSV parsing                (tests  7 – 12)
//    3. Edge-list → sorted CSR conversion  (tests 13 – 20)
//    4. Full pipeline (load)               (tests 21 – 24)
//    5. Cora-scale validation              (tests 25 – 27)
//    6. Reddit-scale validation            (tests 28 – 31)
//    7. Actual Cora dataset                (tests 32 – 35)
//    8. Actual Reddit dataset              (tests 36 – 39)
//    9. Error handling                     (tests 40 – 49)
//
// ============================================================================

#include "tinygnn/graph_loader.hpp"
#include "tinygnn/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <chrono>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ── Minimal test framework (identical to test_tensor.cpp) ───────────────────
static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define ASSERT_TRUE(cond)                                                     \
    do {                                                                      \
        ++g_tests_run;                                                        \
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
        ++g_tests_run;                                                        \
        if ((a) != (b)) {                                                     \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__           \
                      << " — ASSERT_EQ(" #a ", " #b ") -> "                  \
                      << (a) << " != " << (b) << "\n";                       \
            ++g_tests_failed;                                                 \
        } else {                                                              \
            ++g_tests_passed;                                                 \
        }                                                                     \
    } while (0)

#define ASSERT_FLOAT_EQ(a, b)                                                 \
    do {                                                                      \
        ++g_tests_run;                                                        \
        if (std::fabs(static_cast<double>(a) - static_cast<double>(b))        \
            > 1e-6) {                                                         \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__           \
                      << " — ASSERT_FLOAT_EQ(" #a ", " #b ") -> "            \
                      << (a) << " != " << (b) << "\n";                       \
            ++g_tests_failed;                                                 \
        } else {                                                              \
            ++g_tests_passed;                                                 \
        }                                                                     \
    } while (0)

#define ASSERT_THROWS(expr, exception_type)                                   \
    do {                                                                      \
        ++g_tests_run;                                                        \
        bool caught = false;                                                  \
        try { expr; } catch (const exception_type&) { caught = true; }        \
        if (!caught) {                                                        \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__           \
                      << " — ASSERT_THROWS(" #expr ", "                      \
                      << #exception_type ")\n";                               \
            ++g_tests_failed;                                                 \
        } else {                                                              \
            ++g_tests_passed;                                                 \
        }                                                                     \
    } while (0)

#define RUN_TEST(fn)                                                          \
    do {                                                                      \
        std::cout << "  Running " #fn "...\n";                                \
        fn();                                                                 \
    } while (0)

using namespace tinygnn;

// ============================================================================
//  RAII helper: write text to a tmp file, auto-delete on scope exit.
//  Non-copyable; movable for convenience.
// ============================================================================
class TempFile {
public:
    explicit TempFile(const std::string& content,
                      const std::string& suffix = ".csv") {
        static int counter = 0;
        path_ = "tinygnn_test_" + std::to_string(counter++) + suffix;
        {
            std::ofstream f(path_);
            if (!f.is_open()) {
                std::cerr << "TempFile: failed to create '" << path_ << "'\n";
                std::abort();
            }
            f << content;
        }  // flush + close
    }

    ~TempFile() { std::remove(path_.c_str()); }

    const std::string& path() const noexcept { return path_; }

    TempFile(const TempFile&)            = delete;
    TempFile& operator=(const TempFile&) = delete;
    TempFile(TempFile&& o) noexcept : path_(std::move(o.path_)) {
        o.path_.clear();
    }
    TempFile& operator=(TempFile&& o) noexcept {
        if (this != &o) {
            if (!path_.empty()) std::remove(path_.c_str());
            path_ = std::move(o.path_);
            o.path_.clear();
        }
        return *this;
    }

private:
    std::string path_;
};

// ════════════════════════════════════════════════════════════════════════════
//  1.  EDGE CSV PARSING
// ════════════════════════════════════════════════════════════════════════════

// 1. Parse edges with a header row
void test_parse_edges_with_header() {
    TempFile f(
        "src,dst\n"
        "0,1\n"
        "0,2\n"
        "1,0\n"
        "1,3\n"
        "2,0\n"
        "2,3\n"
        "2,4\n"
        "3,1\n"
        "3,2\n"
        "4,2\n"
    );
    auto edges = GraphLoader::parse_edges(f.path());

    ASSERT_EQ(edges.size(), 10u);
    ASSERT_EQ(edges[0].first,  0);  ASSERT_EQ(edges[0].second, 1);
    ASSERT_EQ(edges[1].first,  0);  ASSERT_EQ(edges[1].second, 2);
    ASSERT_EQ(edges[6].first,  2);  ASSERT_EQ(edges[6].second, 4);
    ASSERT_EQ(edges[9].first,  4);  ASSERT_EQ(edges[9].second, 2);
}

// 2. Parse edges without a header (first line is numeric)
void test_parse_edges_no_header() {
    TempFile f("0,1\n1,2\n2,0\n");
    auto edges = GraphLoader::parse_edges(f.path());

    ASSERT_EQ(edges.size(), 3u);
    ASSERT_EQ(edges[0].first, 0);  ASSERT_EQ(edges[0].second, 1);
    ASSERT_EQ(edges[1].first, 1);  ASSERT_EQ(edges[1].second, 2);
    ASSERT_EQ(edges[2].first, 2);  ASSERT_EQ(edges[2].second, 0);
}

// 3. Parse edges with CRLF line endings and extra whitespace
void test_parse_edges_crlf() {
    TempFile f("src,dst\r\n0, 1\r\n1 ,0\r\n");
    auto edges = GraphLoader::parse_edges(f.path());

    ASSERT_EQ(edges.size(), 2u);
    ASSERT_EQ(edges[0].first, 0);  ASSERT_EQ(edges[0].second, 1);
    ASSERT_EQ(edges[1].first, 1);  ASSERT_EQ(edges[1].second, 0);
}

// 4. Parse single edge
void test_parse_edges_single() {
    TempFile f("5,10\n");
    auto edges = GraphLoader::parse_edges(f.path());

    ASSERT_EQ(edges.size(), 1u);
    ASSERT_EQ(edges[0].first, 5);
    ASSERT_EQ(edges[0].second, 10);
}

// 5. Parse edges with self-loop
void test_parse_edges_self_loop() {
    TempFile f("0,0\n1,1\n0,1\n");
    auto edges = GraphLoader::parse_edges(f.path());

    ASSERT_EQ(edges.size(), 3u);
    ASSERT_EQ(edges[0].first, 0);  ASSERT_EQ(edges[0].second, 0);
    ASSERT_EQ(edges[1].first, 1);  ASSERT_EQ(edges[1].second, 1);
}

// 6. Parse edges with trailing blank lines (should be ignored)
void test_parse_edges_trailing_blanks() {
    TempFile f("0,1\n2,3\n\n\n");
    auto edges = GraphLoader::parse_edges(f.path());

    ASSERT_EQ(edges.size(), 2u);
}

// ════════════════════════════════════════════════════════════════════════════
//  2.  FEATURE CSV PARSING
// ════════════════════════════════════════════════════════════════════════════

// 7. Parse features with header
void test_parse_features_with_header() {
    TempFile f(
        "node_id,feat_0,feat_1,feat_2\n"
        "0,1.0,0.0,0.5\n"
        "1,0.0,1.0,0.0\n"
        "2,0.5,0.5,1.0\n"
        "3,0.0,0.0,1.0\n"
        "4,1.0,1.0,0.0\n"
    );
    auto t = GraphLoader::parse_features(f.path());

    ASSERT_TRUE(t.format() == StorageFormat::Dense);
    ASSERT_EQ(t.rows(), 5u);
    ASSERT_EQ(t.cols(), 3u);

    // Spot-check values
    ASSERT_FLOAT_EQ(t.at(0, 0), 1.0f);
    ASSERT_FLOAT_EQ(t.at(0, 2), 0.5f);
    ASSERT_FLOAT_EQ(t.at(2, 1), 0.5f);
    ASSERT_FLOAT_EQ(t.at(4, 0), 1.0f);
    ASSERT_FLOAT_EQ(t.at(4, 2), 0.0f);
}

// 8. Parse features without header
void test_parse_features_no_header() {
    TempFile f(
        "0,1.0,2.0\n"
        "1,3.0,4.0\n"
    );
    auto t = GraphLoader::parse_features(f.path());

    ASSERT_EQ(t.rows(), 2u);
    ASSERT_EQ(t.cols(), 2u);
    ASSERT_FLOAT_EQ(t.at(0, 0), 1.0f);
    ASSERT_FLOAT_EQ(t.at(0, 1), 2.0f);
    ASSERT_FLOAT_EQ(t.at(1, 0), 3.0f);
    ASSERT_FLOAT_EQ(t.at(1, 1), 4.0f);
}

// 9. Parse features with non-contiguous node IDs (gaps are zero-filled)
void test_parse_features_sparse_ids() {
    TempFile f(
        "node_id,f0\n"
        "0,10.0\n"
        "5,50.0\n"
    );
    auto t = GraphLoader::parse_features(f.path());

    // num_nodes = max_id + 1 = 6
    ASSERT_EQ(t.rows(), 6u);
    ASSERT_EQ(t.cols(), 1u);
    ASSERT_FLOAT_EQ(t.at(0, 0), 10.0f);
    ASSERT_FLOAT_EQ(t.at(1, 0), 0.0f);   // gap → zero
    ASSERT_FLOAT_EQ(t.at(2, 0), 0.0f);
    ASSERT_FLOAT_EQ(t.at(5, 0), 50.0f);
}

// 10. Parse features — single node, single feature
void test_parse_features_single() {
    TempFile f("0,42.0\n");
    auto t = GraphLoader::parse_features(f.path());

    ASSERT_EQ(t.rows(), 1u);
    ASSERT_EQ(t.cols(), 1u);
    ASSERT_FLOAT_EQ(t.at(0, 0), 42.0f);
}

// 11. Parse features — out-of-order node IDs
void test_parse_features_unordered() {
    TempFile f(
        "node_id,f0,f1\n"
        "3,30.0,31.0\n"
        "1,10.0,11.0\n"
        "0, 0.0, 1.0\n"
        "2,20.0,21.0\n"
    );
    auto t = GraphLoader::parse_features(f.path());

    ASSERT_EQ(t.rows(), 4u);
    ASSERT_EQ(t.cols(), 2u);
    ASSERT_FLOAT_EQ(t.at(0, 0),  0.0f);
    ASSERT_FLOAT_EQ(t.at(0, 1),  1.0f);
    ASSERT_FLOAT_EQ(t.at(1, 0), 10.0f);
    ASSERT_FLOAT_EQ(t.at(3, 0), 30.0f);
    ASSERT_FLOAT_EQ(t.at(3, 1), 31.0f);
}

// 12. Parse features — negative values
void test_parse_features_negative_values() {
    TempFile f("0,-1.5,2.5\n1,3.0,-4.0\n");
    auto t = GraphLoader::parse_features(f.path());

    ASSERT_EQ(t.rows(), 2u);
    ASSERT_FLOAT_EQ(t.at(0, 0), -1.5f);
    ASSERT_FLOAT_EQ(t.at(1, 1), -4.0f);
}

// ════════════════════════════════════════════════════════════════════════════
//  3.  EDGE-LIST → SORTED CSR CONVERSION
// ════════════════════════════════════════════════════════════════════════════

// 13. Known 5-node graph — exact CSR verification
void test_csr_known_graph() {
    //  Node 0 → {1, 2}
    //  Node 1 → {0, 3}
    //  Node 2 → {0, 3, 4}
    //  Node 3 → {1, 2}
    //  Node 4 → {2}
    std::vector<std::pair<int32_t, int32_t>> edges = {
        {0,1}, {0,2}, {1,0}, {1,3}, {2,0}, {2,3}, {2,4}, {3,1}, {3,2}, {4,2}
    };

    auto adj = GraphLoader::edge_list_to_csr(edges, 5);

    ASSERT_TRUE(adj.format() == StorageFormat::SparseCSR);
    ASSERT_EQ(adj.rows(), 5u);
    ASSERT_EQ(adj.cols(), 5u);
    ASSERT_EQ(adj.nnz(), 10u);

    // row_ptr = [0, 2, 4, 7, 9, 10]
    const auto& rp = adj.row_ptr();
    ASSERT_EQ(rp.size(), 6u);
    ASSERT_EQ(rp[0], 0);
    ASSERT_EQ(rp[1], 2);
    ASSERT_EQ(rp[2], 4);
    ASSERT_EQ(rp[3], 7);
    ASSERT_EQ(rp[4], 9);
    ASSERT_EQ(rp[5], 10);

    // col_ind (sorted within each row)
    const auto& ci = adj.col_ind();
    // Node 0 → [1, 2]
    ASSERT_EQ(ci[0], 1);  ASSERT_EQ(ci[1], 2);
    // Node 1 → [0, 3]
    ASSERT_EQ(ci[2], 0);  ASSERT_EQ(ci[3], 3);
    // Node 2 → [0, 3, 4]
    ASSERT_EQ(ci[4], 0);  ASSERT_EQ(ci[5], 3);  ASSERT_EQ(ci[6], 4);
    // Node 3 → [1, 2]
    ASSERT_EQ(ci[7], 1);  ASSERT_EQ(ci[8], 2);
    // Node 4 → [2]
    ASSERT_EQ(ci[9], 2);

    // All values = 1.0
    for (std::size_t i = 0; i < adj.data().size(); ++i) {
        ASSERT_FLOAT_EQ(adj.data()[i], 1.0f);
    }
}

// 14. CSR from unsorted edge list — verify sorting within rows
void test_csr_unsorted_input() {
    // Edges given out of order: Node 0 → {3, 1, 2}
    std::vector<std::pair<int32_t, int32_t>> edges = {
        {0,3}, {0,1}, {0,2}
    };

    auto adj = GraphLoader::edge_list_to_csr(edges, 4);

    const auto& ci = adj.col_ind();
    // Must be sorted: [1, 2, 3]
    ASSERT_EQ(ci[0], 1);
    ASSERT_EQ(ci[1], 2);
    ASSERT_EQ(ci[2], 3);
}

// 15. CSR — node with no outgoing edges (empty row)
void test_csr_empty_row() {
    // 3 nodes; only node 0 has edges
    std::vector<std::pair<int32_t, int32_t>> edges = {
        {0,1}, {0,2}
    };

    auto adj = GraphLoader::edge_list_to_csr(edges, 3);

    const auto& rp = adj.row_ptr();
    ASSERT_EQ(rp[0], 0);
    ASSERT_EQ(rp[1], 2);    // node 0: 2 edges
    ASSERT_EQ(rp[2], 2);    // node 1: 0 edges (empty)
    ASSERT_EQ(rp[3], 2);    // node 2: 0 edges (empty)
}

// 16. CSR — graph with self-loops
void test_csr_self_loops() {
    std::vector<std::pair<int32_t, int32_t>> edges = {
        {0,0}, {0,1}, {1,1}
    };

    auto adj = GraphLoader::edge_list_to_csr(edges, 2);

    const auto& rp = adj.row_ptr();
    const auto& ci = adj.col_ind();
    ASSERT_EQ(rp[0], 0);
    ASSERT_EQ(rp[1], 2);    // node 0: self-loop + edge to 1
    ASSERT_EQ(rp[2], 3);    // node 1: self-loop
    ASSERT_EQ(ci[0], 0);    // sorted: self-loop first
    ASSERT_EQ(ci[1], 1);
    ASSERT_EQ(ci[2], 1);    // node 1 self-loop
}

// 17. CSR — empty graph (no edges)
void test_csr_no_edges() {
    std::vector<std::pair<int32_t, int32_t>> edges;

    auto adj = GraphLoader::edge_list_to_csr(edges, 3);

    ASSERT_EQ(adj.rows(), 3u);
    ASSERT_EQ(adj.cols(), 3u);
    ASSERT_EQ(adj.nnz(), 0u);

    const auto& rp = adj.row_ptr();
    ASSERT_EQ(rp.size(), 4u);
    for (auto v : rp) { ASSERT_EQ(v, 0); }
}

// 18. CSR — single edge
void test_csr_single_edge() {
    std::vector<std::pair<int32_t, int32_t>> edges = {{0, 1}};
    auto adj = GraphLoader::edge_list_to_csr(edges, 2);

    ASSERT_EQ(adj.nnz(), 1u);
    ASSERT_EQ(adj.row_ptr()[0], 0);
    ASSERT_EQ(adj.row_ptr()[1], 1);
    ASSERT_EQ(adj.row_ptr()[2], 1);
    ASSERT_EQ(adj.col_ind()[0], 1);
}

// 19. CSR — column indices are globally sorted per row
void test_csr_sorted_invariant() {
    // Build a larger graph with many edges out of order
    std::vector<std::pair<int32_t, int32_t>> edges;
    // Node 0 → {9, 7, 5, 3, 1}
    for (int d = 9; d >= 1; d -= 2) edges.emplace_back(0, d);
    // Node 1 → {8, 6, 4, 2, 0}
    for (int d = 8; d >= 0; d -= 2) edges.emplace_back(1, d);

    auto adj = GraphLoader::edge_list_to_csr(edges, 10);

    const auto& rp = adj.row_ptr();
    const auto& ci = adj.col_ind();

    // Verify sorted within each row
    for (std::size_t r = 0; r < adj.rows(); ++r) {
        for (int32_t j = rp[r]; j + 1 < rp[r + 1]; ++j) {
            ASSERT_TRUE(ci[static_cast<std::size_t>(j)] <=
                        ci[static_cast<std::size_t>(j + 1)]);
        }
    }
}

// 20. CSR — verify memory footprint matches expected formula
void test_csr_memory_footprint() {
    // 100 nodes, 500 edges → exact footprint
    const std::size_t V = 100, E = 500;

    // Generate E edges within [0, V)
    std::mt19937 rng(123);
    std::uniform_int_distribution<int32_t> dist(0, static_cast<int32_t>(V - 1));
    std::vector<std::pair<int32_t, int32_t>> edges;
    edges.reserve(E);
    for (std::size_t i = 0; i < E; ++i) {
        edges.emplace_back(dist(rng), dist(rng));
    }

    auto adj = GraphLoader::edge_list_to_csr(edges, V);

    const std::size_t expected =
        E * sizeof(float) +          // values
        E * sizeof(int32_t) +        // col_ind
        (V + 1) * sizeof(int32_t);   // row_ptr

    ASSERT_EQ(adj.memory_footprint_bytes(), expected);

    std::cout << "    CSR(" << V << " nodes, " << E << " edges) = "
              << adj.memory_footprint_bytes() << " bytes\n";
}

// ════════════════════════════════════════════════════════════════════════════
//  4.  FULL PIPELINE  (load)
// ════════════════════════════════════════════════════════════════════════════

// 21. Full pipeline — small graph, verify all fields
void test_load_small_graph() {
    TempFile ef(
        "src,dst\n"
        "0,1\n"
        "0,2\n"
        "1,0\n"
        "1,3\n"
        "2,0\n"
        "2,3\n"
        "2,4\n"
        "3,1\n"
        "3,2\n"
        "4,2\n"
    );
    TempFile ff(
        "node_id,feat_0,feat_1,feat_2\n"
        "0,1.0,0.0,0.5\n"
        "1,0.0,1.0,0.0\n"
        "2,0.5,0.5,1.0\n"
        "3,0.0,0.0,1.0\n"
        "4,1.0,1.0,0.0\n"
    );

    auto gd = GraphLoader::load(ef.path(), ff.path());

    ASSERT_EQ(gd.num_nodes, 5u);
    ASSERT_EQ(gd.num_edges, 10u);
    ASSERT_EQ(gd.num_features, 3u);

    // Adjacency checks
    ASSERT_TRUE(gd.adjacency.format() == StorageFormat::SparseCSR);
    ASSERT_EQ(gd.adjacency.rows(), 5u);
    ASSERT_EQ(gd.adjacency.cols(), 5u);
    ASSERT_EQ(gd.adjacency.nnz(), 10u);

    // Feature checks
    ASSERT_TRUE(gd.node_features.format() == StorageFormat::Dense);
    ASSERT_EQ(gd.node_features.rows(), 5u);
    ASSERT_EQ(gd.node_features.cols(), 3u);
    ASSERT_FLOAT_EQ(gd.node_features.at(2, 2), 1.0f);
}

// 22. Full pipeline — node 0 neighbor verification against raw CSV
void test_load_node0_neighbors() {
    TempFile ef(
        "src,dst\n"
        "0,1\n"
        "0,2\n"
        "1,0\n"
        "1,3\n"
        "2,0\n"
        "2,3\n"
        "2,4\n"
        "3,1\n"
        "3,2\n"
        "4,2\n"
    );
    TempFile ff(
        "node_id,f0\n"
        "0,1.0\n"
        "1,2.0\n"
        "2,3.0\n"
        "3,4.0\n"
        "4,5.0\n"
    );

    // First, parse edges separately to know ground truth for node 0
    auto raw_edges = GraphLoader::parse_edges(ef.path());
    std::vector<int32_t> expected_neighbors_of_0;
    for (const auto& [src, dst] : raw_edges) {
        if (src == 0) expected_neighbors_of_0.push_back(dst);
    }
    std::sort(expected_neighbors_of_0.begin(), expected_neighbors_of_0.end());

    // Load full graph
    auto gd = GraphLoader::load(ef.path(), ff.path());

    // Traverse CSR row_ptr for Node 0
    const auto& rp = gd.adjacency.row_ptr();
    const auto& ci = gd.adjacency.col_ind();
    std::vector<int32_t> actual_neighbors_of_0;
    for (int32_t j = rp[0]; j < rp[1]; ++j) {
        actual_neighbors_of_0.push_back(ci[static_cast<std::size_t>(j)]);
    }

    // Assert exact match
    ASSERT_EQ(actual_neighbors_of_0.size(), expected_neighbors_of_0.size());
    for (std::size_t i = 0; i < actual_neighbors_of_0.size(); ++i) {
        ASSERT_EQ(actual_neighbors_of_0[i], expected_neighbors_of_0[i]);
    }

    std::cout << "    Node 0 neighbors: [";
    for (std::size_t i = 0; i < actual_neighbors_of_0.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << actual_neighbors_of_0[i];
    }
    std::cout << "]\n";
}

// 23. Full pipeline — every node's neighbors match raw CSV
void test_load_all_nodes_neighbors() {
    TempFile ef(
        "0,1\n0,2\n1,0\n1,3\n2,0\n2,3\n2,4\n3,1\n3,2\n4,2\n"
    );
    TempFile ff(
        "0,1.0\n1,2.0\n2,3.0\n3,4.0\n4,5.0\n"
    );

    auto raw_edges = GraphLoader::parse_edges(ef.path());
    auto gd = GraphLoader::load(ef.path(), ff.path());

    const auto& rp = gd.adjacency.row_ptr();
    const auto& ci = gd.adjacency.col_ind();

    // For every node, collect expected neighbors from raw edges, sort, compare
    for (std::size_t node = 0; node < gd.num_nodes; ++node) {
        std::vector<int32_t> expected;
        for (const auto& [src, dst] : raw_edges) {
            if (static_cast<std::size_t>(src) == node)
                expected.push_back(dst);
        }
        std::sort(expected.begin(), expected.end());

        std::vector<int32_t> actual;
        for (int32_t j = rp[node]; j < rp[node + 1]; ++j) {
            actual.push_back(ci[static_cast<std::size_t>(j)]);
        }

        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i) {
            ASSERT_EQ(actual[i], expected[i]);
        }
    }
}

// 24. Full pipeline — edges reference higher node ID than features
void test_load_edge_expands_features() {
    TempFile ef("0,5\n5,0\n");
    TempFile ff(
        "node_id,f0\n"
        "0,1.0\n"
        "1,2.0\n"
    );

    auto gd = GraphLoader::load(ef.path(), ff.path());

    // num_nodes = max(6 from edges, 2 from features) = 6
    ASSERT_EQ(gd.num_nodes, 6u);
    ASSERT_EQ(gd.num_edges, 2u);

    // Features should be zero-padded to 6 rows
    ASSERT_EQ(gd.node_features.rows(), 6u);
    ASSERT_FLOAT_EQ(gd.node_features.at(0, 0), 1.0f);
    ASSERT_FLOAT_EQ(gd.node_features.at(1, 0), 2.0f);
    ASSERT_FLOAT_EQ(gd.node_features.at(5, 0), 0.0f);  // zero-padded
}

// ════════════════════════════════════════════════════════════════════════════
//  5.  CORA-SCALE VALIDATION
// ════════════════════════════════════════════════════════════════════════════

// Helper: generate a synthetic Cora-scale dataset
// 2708 nodes, 10556 unique directed edges, 1433 binary features
struct CoraFiles {
    TempFile edges;
    TempFile features;
};

static CoraFiles generate_cora_scale_data() {
    const std::size_t NUM_NODES    = 2708;
    const std::size_t NUM_EDGES    = 10556;
    const std::size_t NUM_FEATURES = 1433;

    std::mt19937 rng(2026);  // fixed seed for determinism

    // Generate exactly NUM_EDGES unique directed edges (no self-loops)
    std::set<std::pair<int32_t, int32_t>> edge_set;
    std::uniform_int_distribution<int32_t> node_dist(
        0, static_cast<int32_t>(NUM_NODES - 1));
    while (edge_set.size() < NUM_EDGES) {
        int32_t src = node_dist(rng);
        int32_t dst = node_dist(rng);
        if (src != dst) edge_set.insert({src, dst});
    }

    // Write edges CSV
    std::ostringstream ess;
    ess << "src,dst\n";
    for (const auto& [s, d] : edge_set) {
        ess << s << "," << d << "\n";
    }

    // Write features CSV (binary features, smaller representation)
    std::ostringstream fss;
    fss << "node_id";
    for (std::size_t j = 0; j < NUM_FEATURES; ++j) {
        fss << ",f" << j;
    }
    fss << "\n";

    std::uniform_int_distribution<int> bit_dist(0, 1);
    for (std::size_t i = 0; i < NUM_NODES; ++i) {
        fss << i;
        for (std::size_t j = 0; j < NUM_FEATURES; ++j) {
            fss << "," << bit_dist(rng);
        }
        fss << "\n";
    }

    return {TempFile(ess.str()), TempFile(fss.str())};
}

// 25. Cora-scale: assert num_nodes == 2708, num_edges == 10556
void test_cora_scale_counts() {
    std::cout << "    Generating Cora-scale data (2708 nodes, 10556 edges)...\n";
    auto cora = generate_cora_scale_data();

    auto gd = GraphLoader::load(cora.edges.path(), cora.features.path());

    ASSERT_EQ(gd.num_nodes,   2708u);
    ASSERT_EQ(gd.num_edges,   10556u);
    ASSERT_EQ(gd.num_features, 1433u);

    // Adjacency shape
    ASSERT_EQ(gd.adjacency.rows(), 2708u);
    ASSERT_EQ(gd.adjacency.cols(), 2708u);
    ASSERT_EQ(gd.adjacency.nnz(),  10556u);
    ASSERT_TRUE(gd.adjacency.format() == StorageFormat::SparseCSR);

    // Feature shape
    ASSERT_EQ(gd.node_features.rows(), 2708u);
    ASSERT_EQ(gd.node_features.cols(), 1433u);
    ASSERT_TRUE(gd.node_features.format() == StorageFormat::Dense);

    std::cout << "    Adjacency: " << gd.adjacency.repr() << "\n";
    std::cout << "    Features:  " << gd.node_features.repr() << "\n";
}

// 26. Cora-scale: verify CSR row_ptr invariants
void test_cora_scale_csr_invariants() {
    auto cora = generate_cora_scale_data();
    auto gd = GraphLoader::load(cora.edges.path(), cora.features.path());

    const auto& rp = gd.adjacency.row_ptr();
    const auto& ci = gd.adjacency.col_ind();

    // row_ptr must have exactly num_nodes+1 entries
    ASSERT_EQ(rp.size(), 2709u);

    // row_ptr[0] == 0
    ASSERT_EQ(rp[0], 0);

    // row_ptr[last] == num_edges
    ASSERT_EQ(rp[2708], static_cast<int32_t>(10556));

    // row_ptr is non-decreasing
    bool non_decreasing = true;
    for (std::size_t i = 1; i < rp.size(); ++i) {
        if (rp[i] < rp[i - 1]) { non_decreasing = false; break; }
    }
    ASSERT_TRUE(non_decreasing);

    // All column indices in [0, 2707]
    bool cols_in_range = true;
    for (auto c : ci) {
        if (c < 0 || c >= 2708) { cols_in_range = false; break; }
    }
    ASSERT_TRUE(cols_in_range);

    // Column indices sorted within each row
    bool sorted_per_row = true;
    for (std::size_t r = 0; r < 2708; ++r) {
        for (int32_t j = rp[r]; j + 1 < rp[r + 1]; ++j) {
            if (ci[static_cast<std::size_t>(j)] >
                ci[static_cast<std::size_t>(j + 1)]) {
                sorted_per_row = false;
                break;
            }
        }
        if (!sorted_per_row) break;
    }
    ASSERT_TRUE(sorted_per_row);
}

// 27. Cora-scale: Node 0 neighbors match raw CSV exactly
void test_cora_scale_node0_neighbors() {
    auto cora = generate_cora_scale_data();

    // Parse raw edges to get ground truth for node 0
    auto raw_edges = GraphLoader::parse_edges(cora.edges.path());
    std::vector<int32_t> expected;
    for (const auto& [src, dst] : raw_edges) {
        if (src == 0) expected.push_back(dst);
    }
    std::sort(expected.begin(), expected.end());

    // Load full graph and traverse CSR
    auto gd = GraphLoader::load(cora.edges.path(), cora.features.path());
    const auto& rp = gd.adjacency.row_ptr();
    const auto& ci = gd.adjacency.col_ind();

    std::vector<int32_t> actual;
    for (int32_t j = rp[0]; j < rp[1]; ++j) {
        actual.push_back(ci[static_cast<std::size_t>(j)]);
    }

    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        ASSERT_EQ(actual[i], expected[i]);
    }

    std::cout << "    Cora Node 0 has " << actual.size() << " neighbors\n";
}

// ════════════════════════════════════════════════════════════════════════════
//  6.  REDDIT-SCALE VALIDATION
// ════════════════════════════════════════════════════════════════════════════
//
//  Reddit (Hamilton et al. 2017, GraphSAGE paper):
//    Nodes    : 232,965   (Reddit posts)
//    Edges    : 114,615,892  (directed post-to-post links)
//    Features : 602 per node
//
//  Data is generated synthetically with a fixed seed so the counts are exact
//  and results are deterministic across runs.
//
//  Tests are split into two groups:
//    a) In-memory algorithm test — calls edge_list_to_csr() directly,
//       bypassing file I/O entirely.  Validates the CSR engine at full
//       Reddit scale (peak RAM: ~1.8 GB).
//
//    b) Full file-pipeline tests — writes edge and feature CSV files using
//       direct buffered I/O, then loads with GraphLoader::load().
//       Reddit features use 602 binary values per node; the edge file
//       contains all 114,615,892 directed edges.
//       Expected runtime: 30–120 s depending on storage and CPU speed.
//
// ════════════════════════════════════════════════════════════════════════════

// ── Buffered file helpers (needed for files too large for ostringstream) ─────

namespace {

/// Write the decimal representation of v into buf.
static void append_int32(std::vector<char>& buf, int32_t v) {
    char tmp[12];
    int  len = std::snprintf(tmp, sizeof(tmp), "%d", v);
    buf.insert(buf.end(), tmp, tmp + len);
}

/// Flush buf to f and clear it.
static void flush_buf(std::ofstream& f, std::vector<char>& buf) {
    f.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    buf.clear();
}

/// Write a directed edge CSV with exactly num_edges random (src,dst) pairs.
/// Uses a 4 MB write buffer for speed.  Returns the temp-file path.
static std::string write_edge_csv(std::size_t num_nodes,
                                  std::size_t num_edges,
                                  uint32_t    seed) {
    static int counter = 0;
    std::string path = "tinygnn_reddit_edges_" + std::to_string(counter++) + ".csv";

    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("write_edge_csv: cannot create '" + path + "'");
    }

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> nd(
        0, static_cast<int32_t>(num_nodes - 1));

    constexpr std::size_t BUF_CAP = 1u << 22;   // 4 MB
    std::vector<char> buf;
    buf.reserve(BUF_CAP);

    // header
    const char hdr[] = "src,dst\n";
    buf.insert(buf.end(), hdr, hdr + sizeof(hdr) - 1);

    for (std::size_t i = 0; i < num_edges; ++i) {
        append_int32(buf, nd(rng));
        buf.push_back(',');
        append_int32(buf, nd(rng));
        buf.push_back('\n');
        if (buf.size() >= BUF_CAP) flush_buf(f, buf);
    }
    if (!buf.empty()) flush_buf(f, buf);
    return path;
}

/// Write a feature CSV: nodes 0..num_nodes-1, num_features binary values each.
/// Uses a 4 MB write buffer.  Returns the temp-file path.
static std::string write_feature_csv(std::size_t num_nodes,
                                     std::size_t num_features,
                                     uint32_t    seed) {
    static int counter = 0;
    std::string path = "tinygnn_reddit_features_" + std::to_string(counter++) + ".csv";

    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("write_feature_csv: cannot create '" + path + "'");
    }

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> bit(0, 1);

    constexpr std::size_t BUF_CAP = 1u << 22;   // 4 MB
    std::vector<char> buf;
    buf.reserve(BUF_CAP);

    // header: "node_id,f0,f1,...,f{N-1}\n"
    {
        const char hdr_id[] = "node_id";
        buf.insert(buf.end(), hdr_id, hdr_id + 7);
        for (std::size_t j = 0; j < num_features; ++j) {
            buf.push_back(',');
            buf.push_back('f');
            append_int32(buf, static_cast<int32_t>(j));
        }
        buf.push_back('\n');
    }

    for (std::size_t i = 0; i < num_nodes; ++i) {
        append_int32(buf, static_cast<int32_t>(i));
        for (std::size_t j = 0; j < num_features; ++j) {
            buf.push_back(',');
            buf.push_back(static_cast<char>('0' + bit(rng)));
        }
        buf.push_back('\n');
        if (buf.size() >= BUF_CAP) flush_buf(f, buf);
    }
    if (!buf.empty()) flush_buf(f, buf);
    return path;
}

/// RAII handle for one or two large temp files.
struct LargeFiles {
    std::string edges_path;
    std::string features_path;

    LargeFiles() = default;

    ~LargeFiles() {
        if (!edges_path.empty())    std::remove(edges_path.c_str());
        if (!features_path.empty()) std::remove(features_path.c_str());
    }

    LargeFiles(const LargeFiles&)            = delete;
    LargeFiles& operator=(const LargeFiles&) = delete;
    LargeFiles(LargeFiles&&)                 = default;
};

}  // anonymous namespace

// Reddit dataset constants
constexpr std::size_t REDDIT_NODES    = 232965;
constexpr std::size_t REDDIT_EDGES    = 114615892;
constexpr std::size_t REDDIT_FEATURES = 602;

// ─────────────────────────────────────────────────────────────────────────────
// 28. Reddit-scale: in-memory CSR algorithm test (no file I/O)
//     Directly calls edge_list_to_csr with 114,615,892 randomly generated
//     (src, dst) pairs.  Validates nnz, row_ptr invariants, column bounds,
//     and per-row sort order.
//     Peak RAM: ~1.8 GB (edge vector 875 MB + CSR arrays ~917 MB).
// ─────────────────────────────────────────────────────────────────────────────
void test_reddit_scale_csr_algorithm() {
    std::cout << "    [Reddit] Generating " << REDDIT_EDGES
              << " edges in memory (peak RAM ~1.8 GB)...\n";

    auto t0 = std::chrono::steady_clock::now();

    std::mt19937 rng(2026);
    std::uniform_int_distribution<int32_t> nd(
        0, static_cast<int32_t>(REDDIT_NODES - 1));

    std::vector<std::pair<int32_t, int32_t>> edges;
    edges.reserve(REDDIT_EDGES);
    for (std::size_t i = 0; i < REDDIT_EDGES; ++i) {
        edges.emplace_back(nd(rng), nd(rng));
    }

    auto t1 = std::chrono::steady_clock::now();
    std::cout << "    [Reddit] Edge generation: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms\n";

    auto adj = GraphLoader::edge_list_to_csr(edges, REDDIT_NODES);

    auto t2 = std::chrono::steady_clock::now();
    std::cout << "    [Reddit] CSR construction: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " ms\n";

    // ── Shape ──────────────────────────────────────────────────────────────
    ASSERT_EQ(adj.rows(), REDDIT_NODES);
    ASSERT_EQ(adj.cols(), REDDIT_NODES);
    ASSERT_EQ(adj.nnz(),  REDDIT_EDGES);
    ASSERT_TRUE(adj.format() == StorageFormat::SparseCSR);

    // ── row_ptr size and sentinels ──────────────────────────────────────────
    ASSERT_EQ(adj.row_ptr().size(), REDDIT_NODES + 1);
    ASSERT_EQ(adj.row_ptr().front(), 0);
    ASSERT_EQ(adj.row_ptr().back(),  static_cast<int32_t>(REDDIT_EDGES));

    // ── row_ptr non-decreasing ──────────────────────────────────────────────
    bool non_dec = true;
    for (std::size_t i = 1; i < adj.row_ptr().size(); ++i) {
        if (adj.row_ptr()[i] < adj.row_ptr()[i - 1]) { non_dec = false; break; }
    }
    ASSERT_TRUE(non_dec);

    // ── Column indices in bounds ────────────────────────────────────────────
    bool cols_ok = true;
    for (auto c : adj.col_ind()) {
        if (c < 0 || static_cast<std::size_t>(c) >= REDDIT_NODES) {
            cols_ok = false; break;
        }
    }
    ASSERT_TRUE(cols_ok);

    // ── Per-row sort invariant (sample every 1000th row to bound runtime) ───
    const auto& rp = adj.row_ptr();
    const auto& ci = adj.col_ind();
    bool sorted_ok = true;
    for (std::size_t r = 0; r < REDDIT_NODES; r += 1000) {
        for (int32_t j = rp[r]; j + 1 < rp[r + 1]; ++j) {
            if (ci[static_cast<std::size_t>(j)] >
                ci[static_cast<std::size_t>(j + 1)]) {
                sorted_ok = false;
                break;
            }
        }
        if (!sorted_ok) break;
    }
    ASSERT_TRUE(sorted_ok);

    // ── Memory footprint ────────────────────────────────────────────────────
    //  nnz × 4 (values) + nnz × 4 (col_ind) + (nodes+1) × 4 (row_ptr)
    const std::size_t expected_bytes =
        REDDIT_EDGES * sizeof(float) +
        REDDIT_EDGES * sizeof(int32_t) +
        (REDDIT_NODES + 1) * sizeof(int32_t);
    ASSERT_EQ(adj.memory_footprint_bytes(), expected_bytes);

    auto t3 = std::chrono::steady_clock::now();
    std::cout << "    [Reddit] CSR footprint: "
              << adj.memory_footprint_bytes() / (1024 * 1024)
              << " MB  |  total time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t0).count()
              << " ms\n";
    std::cout << "    [Reddit] Adjacency: " << adj.repr() << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 29. Reddit-scale: full file pipeline — counts and shape
//     Writes edge CSV (114,615,892 rows) and feature CSV (232,965 × 602)
//     to disk using buffered I/O, then loads via GraphLoader::load().
//     Assert num_nodes == 232,965 and num_edges == 114,615,892.
// ─────────────────────────────────────────────────────────────────────────────
void test_reddit_scale_pipeline_counts() {
    auto t0 = std::chrono::steady_clock::now();

    std::cout << "    [Reddit] Writing edge file (" << REDDIT_EDGES
              << " rows)...\n";
    LargeFiles files;
    files.edges_path    = write_edge_csv(REDDIT_NODES, REDDIT_EDGES, /*seed=*/1337);

    auto t1 = std::chrono::steady_clock::now();
    std::cout << "    [Reddit] Edge file written: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms\n";

    std::cout << "    [Reddit] Writing feature file ("
              << REDDIT_NODES << " nodes x " << REDDIT_FEATURES << " features)...\n";
    files.features_path = write_feature_csv(REDDIT_NODES, REDDIT_FEATURES, /*seed=*/42);

    auto t2 = std::chrono::steady_clock::now();
    std::cout << "    [Reddit] Feature file written: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " ms\n";

    std::cout << "    [Reddit] Loading via GraphLoader::load()...\n";
    auto gd = GraphLoader::load(files.edges_path, files.features_path);

    auto t3 = std::chrono::steady_clock::now();
    std::cout << "    [Reddit] Load complete: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << " ms  |  total: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t0).count()
              << " ms\n";

    // ── Counts ──────────────────────────────────────────────────────────────
    ASSERT_EQ(gd.num_nodes,    REDDIT_NODES);
    ASSERT_EQ(gd.num_edges,    REDDIT_EDGES);
    ASSERT_EQ(gd.num_features, REDDIT_FEATURES);

    // ── Adjacency shape ─────────────────────────────────────────────────────
    ASSERT_EQ(gd.adjacency.rows(), REDDIT_NODES);
    ASSERT_EQ(gd.adjacency.cols(), REDDIT_NODES);
    ASSERT_EQ(gd.adjacency.nnz(),  REDDIT_EDGES);
    ASSERT_TRUE(gd.adjacency.format() == StorageFormat::SparseCSR);

    // ── Feature shape ────────────────────────────────────────────────────────
    ASSERT_EQ(gd.node_features.rows(), REDDIT_NODES);
    ASSERT_EQ(gd.node_features.cols(), REDDIT_FEATURES);
    ASSERT_TRUE(gd.node_features.format() == StorageFormat::Dense);

    std::cout << "    [Reddit] Adjacency: " << gd.adjacency.repr() << "\n";
    std::cout << "    [Reddit] Features:  " << gd.node_features.repr() << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 30. Reddit-scale: CSR row_ptr invariants on the loaded full-pipeline graph
// ─────────────────────────────────────────────────────────────────────────────
void test_reddit_scale_csr_invariants() {
    // Reuse the same edge/feature files (same seeds as test 29).
    LargeFiles files;
    files.edges_path    = write_edge_csv(REDDIT_NODES, REDDIT_EDGES, /*seed=*/1337);
    files.features_path = write_feature_csv(REDDIT_NODES, REDDIT_FEATURES, /*seed=*/42);

    std::cout << "    [Reddit] Loading graph for CSR invariant checks...\n";
    auto gd = GraphLoader::load(files.edges_path, files.features_path);

    const auto& rp = gd.adjacency.row_ptr();
    const auto& ci = gd.adjacency.col_ind();

    // row_ptr has exactly num_nodes + 1 entries
    ASSERT_EQ(rp.size(), REDDIT_NODES + 1);

    // row_ptr sentinels
    ASSERT_EQ(rp.front(), 0);
    ASSERT_EQ(rp.back(),  static_cast<int32_t>(REDDIT_EDGES));

    // row_ptr non-decreasing
    bool non_dec = true;
    for (std::size_t i = 1; i < rp.size(); ++i) {
        if (rp[i] < rp[i - 1]) { non_dec = false; break; }
    }
    ASSERT_TRUE(non_dec);

    // All column indices in [0, REDDIT_NODES)
    bool cols_ok = true;
    for (auto c : ci) {
        if (c < 0 || static_cast<std::size_t>(c) >= REDDIT_NODES) {
            cols_ok = false; break;
        }
    }
    ASSERT_TRUE(cols_ok);

    // Per-row sort invariant (sample every 1000th row)
    bool sorted_ok = true;
    for (std::size_t r = 0; r < REDDIT_NODES; r += 1000) {
        for (int32_t j = rp[r]; j + 1 < rp[r + 1]; ++j) {
            if (ci[static_cast<std::size_t>(j)] >
                ci[static_cast<std::size_t>(j + 1)]) {
                sorted_ok = false;
                break;
            }
        }
        if (!sorted_ok) break;
    }
    ASSERT_TRUE(sorted_ok);

    std::cout << "    [Reddit] row_ptr[0]=" << rp.front()
              << "  row_ptr[" << REDDIT_NODES << "]="
              << rp.back() << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 31. Reddit-scale: Node 0 neighbors match raw CSV exactly
//     Re-parses the edge file to build the ground truth for node 0, then
//     traverses row 0 of the CSR and asserts an exact match.
// ─────────────────────────────────────────────────────────────────────────────
void test_reddit_scale_node0_neighbors() {
    // Write edges with the same seed as test 29 so results are identical.
    LargeFiles files;
    files.edges_path    = write_edge_csv(REDDIT_NODES, REDDIT_EDGES, /*seed=*/1337);
    files.features_path = write_feature_csv(REDDIT_NODES, REDDIT_FEATURES, /*seed=*/42);

    std::cout << "    [Reddit] Parsing edges for ground-truth node-0 neighbors...\n";
    auto raw = GraphLoader::parse_edges(files.edges_path);

    std::vector<int32_t> expected;
    for (const auto& [src, dst] : raw) {
        if (src == 0) expected.push_back(dst);
    }
    std::sort(expected.begin(), expected.end());

    std::cout << "    [Reddit] Loading full graph...\n";
    auto gd = GraphLoader::load(files.edges_path, files.features_path);

    const auto& rp = gd.adjacency.row_ptr();
    const auto& ci = gd.adjacency.col_ind();

    std::vector<int32_t> actual;
    actual.reserve(static_cast<std::size_t>(rp[1] - rp[0]));
    for (int32_t j = rp[0]; j < rp[1]; ++j) {
        actual.push_back(ci[static_cast<std::size_t>(j)]);
    }

    ASSERT_EQ(actual.size(), expected.size());
    bool match = true;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) { match = false; break; }
    }
    ASSERT_TRUE(match);

    std::cout << "    [Reddit] Node 0 has " << actual.size()
              << " neighbors — CSR traversal matches raw CSV exactly\n";
}

// ════════════════════════════════════════════════════════════════════════════
//  7.  ACTUAL CORA DATASET
// ════════════════════════════════════════════════════════════════════════════
//
//  Tests against the real Cora citation dataset (McCallum et al. 2000).
//  Requires: datasets/cora/edges.csv and datasets/cora/node_features.csv
//  (Run `python3 scripts/fetch_datasets.py` to download.)
//
//  Known properties:
//    Nodes    : 2,708  (unique papers)
//    Edges    : 5,429  (directed citation links)
//    Features : 1,433  (binary bag-of-words indicators)
//
// ════════════════════════════════════════════════════════════════════════════

namespace {

bool dataset_available(const std::string& name) {
    std::ifstream ef("datasets/" + name + "/edges.csv");
    std::ifstream ff("datasets/" + name + "/node_features.csv");
    return ef.is_open() && ff.is_open();
}

GraphData& cached_cora() {
    static GraphData gd = GraphLoader::load(
        "datasets/cora/edges.csv", "datasets/cora/node_features.csv");
    return gd;
}

GraphData& cached_reddit() {
    static GraphData gd = []() -> GraphData {
        std::cout << "    [Loading actual Reddit dataset — may take several minutes]\n";
        auto t0 = std::chrono::steady_clock::now();
        auto g = GraphLoader::load(
            "datasets/reddit/edges.csv",
            "datasets/reddit/node_features.csv");
        auto t1 = std::chrono::steady_clock::now();
        auto sec = std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count();
        std::cout << "    [Reddit loaded in " << sec << " s]\n";
        return g;
    }();
    return gd;
}

}  // anonymous namespace

// 32. Actual Cora: load and verify dataset dimensions
void test_cora_actual_load() {
    if (!dataset_available("cora")) {
        std::cout << "    [SKIP] datasets/cora/ not found "
                     "(run: python3 scripts/fetch_datasets.py)\n";
        return;
    }

    const auto& gd = cached_cora();

    ASSERT_EQ(gd.num_nodes,    2708u);
    ASSERT_EQ(gd.num_edges,    5429u);
    ASSERT_EQ(gd.num_features, 1433u);

    ASSERT_TRUE(gd.adjacency.format() == StorageFormat::SparseCSR);
    ASSERT_EQ(gd.adjacency.rows(), 2708u);
    ASSERT_EQ(gd.adjacency.cols(), 2708u);
    ASSERT_EQ(gd.adjacency.nnz(),  5429u);

    ASSERT_TRUE(gd.node_features.format() == StorageFormat::Dense);
    ASSERT_EQ(gd.node_features.rows(), 2708u);
    ASSERT_EQ(gd.node_features.cols(), 1433u);

    std::cout << "    Adjacency: " << gd.adjacency.repr() << "\n";
    std::cout << "    Features:  " << gd.node_features.repr() << "\n";
}

// 33. Actual Cora: CSR structural invariants
void test_cora_actual_csr_invariants() {
    if (!dataset_available("cora")) {
        std::cout << "    [SKIP] datasets/cora/ not found\n";
        return;
    }

    const auto& gd = cached_cora();
    const auto& rp = gd.adjacency.row_ptr();
    const auto& ci = gd.adjacency.col_ind();

    ASSERT_EQ(rp.size(), 2709u);
    ASSERT_EQ(rp.front(), 0);
    ASSERT_EQ(rp.back(),  static_cast<int32_t>(5429));

    bool non_dec = true;
    for (std::size_t i = 1; i < rp.size(); ++i) {
        if (rp[i] < rp[i - 1]) { non_dec = false; break; }
    }
    ASSERT_TRUE(non_dec);

    bool cols_ok = true;
    for (auto c : ci) {
        if (c < 0 || c >= 2708) { cols_ok = false; break; }
    }
    ASSERT_TRUE(cols_ok);

    bool sorted_ok = true;
    for (std::size_t r = 0; r < 2708; ++r) {
        for (int32_t j = rp[r]; j + 1 < rp[r + 1]; ++j) {
            if (ci[static_cast<std::size_t>(j)] >
                ci[static_cast<std::size_t>(j + 1)]) {
                sorted_ok = false;
                break;
            }
        }
        if (!sorted_ok) break;
    }
    ASSERT_TRUE(sorted_ok);
}

// 34. Actual Cora: features are all binary (0 or 1 bag-of-words)
void test_cora_actual_features_binary() {
    if (!dataset_available("cora")) {
        std::cout << "    [SKIP] datasets/cora/ not found\n";
        return;
    }

    const auto& gd = cached_cora();
    const auto& data = gd.node_features.data();

    ASSERT_EQ(gd.node_features.rows(), 2708u);
    ASSERT_EQ(gd.node_features.cols(), 1433u);

    bool all_binary = true;
    for (float v : data) {
        if (v != 0.0f && v != 1.0f) { all_binary = false; break; }
    }
    ASSERT_TRUE(all_binary);

    // Sum > 0: at least some words are present
    double sum = 0.0;
    for (float v : data) sum += static_cast<double>(v);
    ASSERT_TRUE(sum > 0.0);

    std::cout << "    Feature sum: " << static_cast<std::size_t>(sum)
              << " / " << data.size() << " entries are 1\n";
}

// 35. Actual Cora: Node 0 neighbors match raw edge file
void test_cora_actual_node0_neighbors() {
    if (!dataset_available("cora")) {
        std::cout << "    [SKIP] datasets/cora/ not found\n";
        return;
    }

    // Ground truth from raw CSV
    auto raw = GraphLoader::parse_edges("datasets/cora/edges.csv");
    std::vector<int32_t> expected;
    for (const auto& [src, dst] : raw) {
        if (src == 0) expected.push_back(dst);
    }
    std::sort(expected.begin(), expected.end());

    // CSR from loaded graph
    const auto& gd = cached_cora();
    const auto& rp = gd.adjacency.row_ptr();
    const auto& ci = gd.adjacency.col_ind();

    std::vector<int32_t> actual;
    for (int32_t j = rp[0]; j < rp[1]; ++j) {
        actual.push_back(ci[static_cast<std::size_t>(j)]);
    }

    ASSERT_EQ(actual.size(), expected.size());
    ASSERT_TRUE(actual == expected);

    std::cout << "    Cora Node 0: " << actual.size() << " neighbors\n";
}

// ════════════════════════════════════════════════════════════════════════════
//  8.  ACTUAL REDDIT DATASET
// ════════════════════════════════════════════════════════════════════════════
//
//  Tests against the real Reddit dataset (Hamilton et al. 2017, GraphSAGE).
//  Requires: datasets/reddit/edges.csv and datasets/reddit/node_features.csv
//  (Run `python3 scripts/fetch_datasets.py` to download.)
//
//  Known properties:
//    Nodes    : 232,965  (Reddit posts)
//    Edges    : 114,615,892  (directed post-to-post links)
//    Features : 602  (GloVe word embeddings)
//
//  NOTE: Loading the actual Reddit dataset takes several minutes and ~2 GB RAM.
//        The GraphData is cached: loaded once, shared across all Reddit tests.
//
// ════════════════════════════════════════════════════════════════════════════

// 36. Actual Reddit: load and verify dataset dimensions
void test_reddit_actual_load() {
    if (!dataset_available("reddit")) {
        std::cout << "    [SKIP] datasets/reddit/ not found "
                     "(run: python3 scripts/fetch_datasets.py)\n";
        return;
    }

    const auto& gd = cached_reddit();

    ASSERT_EQ(gd.num_nodes,    232965u);
    ASSERT_EQ(gd.num_edges,    114615892u);
    ASSERT_EQ(gd.num_features, 602u);

    ASSERT_TRUE(gd.adjacency.format() == StorageFormat::SparseCSR);
    ASSERT_EQ(gd.adjacency.rows(), 232965u);
    ASSERT_EQ(gd.adjacency.cols(), 232965u);
    ASSERT_EQ(gd.adjacency.nnz(),  114615892u);

    ASSERT_TRUE(gd.node_features.format() == StorageFormat::Dense);
    ASSERT_EQ(gd.node_features.rows(), 232965u);
    ASSERT_EQ(gd.node_features.cols(), 602u);

    std::cout << "    Adjacency: " << gd.adjacency.repr() << "\n";
    std::cout << "    Features:  " << gd.node_features.repr() << "\n";
}

// 37. Actual Reddit: CSR structural invariants
void test_reddit_actual_csr_invariants() {
    if (!dataset_available("reddit")) {
        std::cout << "    [SKIP] datasets/reddit/ not found\n";
        return;
    }

    const auto& gd = cached_reddit();
    const auto& rp = gd.adjacency.row_ptr();
    const auto& ci = gd.adjacency.col_ind();

    ASSERT_EQ(rp.size(), 232966u);
    ASSERT_EQ(rp.front(), 0);
    ASSERT_EQ(rp.back(),  static_cast<int32_t>(114615892));

    bool non_dec = true;
    for (std::size_t i = 1; i < rp.size(); ++i) {
        if (rp[i] < rp[i - 1]) { non_dec = false; break; }
    }
    ASSERT_TRUE(non_dec);

    bool cols_ok = true;
    for (auto c : ci) {
        if (c < 0 || static_cast<std::size_t>(c) >= 232965) {
            cols_ok = false; break;
        }
    }
    ASSERT_TRUE(cols_ok);

    // Per-row sort (sample every 1000th row)
    bool sorted_ok = true;
    for (std::size_t r = 0; r < 232965; r += 1000) {
        for (int32_t j = rp[r]; j + 1 < rp[r + 1]; ++j) {
            if (ci[static_cast<std::size_t>(j)] >
                ci[static_cast<std::size_t>(j + 1)]) {
                sorted_ok = false;
                break;
            }
        }
        if (!sorted_ok) break;
    }
    ASSERT_TRUE(sorted_ok);
}

// 38. Actual Reddit: features are finite GloVe embeddings
void test_reddit_actual_feature_properties() {
    if (!dataset_available("reddit")) {
        std::cout << "    [SKIP] datasets/reddit/ not found\n";
        return;
    }

    const auto& gd = cached_reddit();
    const auto& data = gd.node_features.data();

    ASSERT_EQ(gd.node_features.rows(), 232965u);
    ASSERT_EQ(gd.node_features.cols(), 602u);

    // All values must be finite (not NaN or Inf)
    bool all_finite = true;
    for (float v : data) {
        if (!std::isfinite(v)) { all_finite = false; break; }
    }
    ASSERT_TRUE(all_finite);

    // GloVe embeddings have real-valued entries — at least some non-zero
    bool has_nonzero = false;
    for (float v : data) {
        if (v != 0.0f) { has_nonzero = true; break; }
    }
    ASSERT_TRUE(has_nonzero);
}

// 39. Actual Reddit: Node 0 has neighbors in sorted order within range
void test_reddit_actual_node0_neighbors() {
    if (!dataset_available("reddit")) {
        std::cout << "    [SKIP] datasets/reddit/ not found\n";
        return;
    }

    const auto& gd = cached_reddit();
    const auto& rp = gd.adjacency.row_ptr();
    const auto& ci = gd.adjacency.col_ind();

    const auto count = static_cast<std::size_t>(rp[1] - rp[0]);
    ASSERT_TRUE(count > 0u);

    // Verify sorted and in range
    bool ok = true;
    for (int32_t j = rp[0]; j < rp[1]; ++j) {
        const auto c = ci[static_cast<std::size_t>(j)];
        if (c < 0 || static_cast<std::size_t>(c) >= 232965) { ok = false; break; }
        if (j > rp[0] &&
            ci[static_cast<std::size_t>(j - 1)] > c) {
            ok = false; break;
        }
    }
    ASSERT_TRUE(ok);

    std::cout << "    Reddit Node 0: " << count << " neighbors\n";
}

// ════════════════════════════════════════════════════════════════════════════
//  9.  ERROR HANDLING
// ════════════════════════════════════════════════════════════════════════════

// 40. File not found
void test_error_file_not_found() {
    ASSERT_THROWS(
        GraphLoader::parse_edges("nonexistent_edges_file.csv"),
        std::runtime_error
    );
    ASSERT_THROWS(
        GraphLoader::parse_features("nonexistent_features_file.csv"),
        std::runtime_error
    );
}

// 41. Empty file
void test_error_empty_file() {
    TempFile f("");
    ASSERT_THROWS(
        GraphLoader::parse_edges(f.path()),
        std::runtime_error
    );
}

// 42. Malformed edge line (no comma)
void test_error_malformed_edge() {
    TempFile f("0 1\n");
    ASSERT_THROWS(
        GraphLoader::parse_edges(f.path()),
        std::runtime_error
    );
}

// 43. Non-integer in edge file
void test_error_non_integer_edge() {
    TempFile f("0,abc\n");
    ASSERT_THROWS(
        GraphLoader::parse_edges(f.path()),
        std::runtime_error
    );
}

// 44. Negative node ID in edges
void test_error_negative_edge_id() {
    TempFile f("-1,2\n");
    ASSERT_THROWS(
        GraphLoader::parse_edges(f.path()),
        std::runtime_error
    );
}

// 45. Negative node ID in features
void test_error_negative_feature_id() {
    TempFile f("-1,1.0,2.0\n");
    ASSERT_THROWS(
        GraphLoader::parse_features(f.path()),
        std::runtime_error
    );
}

// 46. Inconsistent feature counts
void test_error_inconsistent_features() {
    TempFile f(
        "0,1.0,2.0\n"
        "1,3.0\n"          // only 1 feature instead of 2
    );
    ASSERT_THROWS(
        GraphLoader::parse_features(f.path()),
        std::runtime_error
    );
}

// 47. Edge out of range for CSR conversion
void test_error_edge_out_of_range() {
    std::vector<std::pair<int32_t, int32_t>> edges = {{0, 5}};
    ASSERT_THROWS(
        GraphLoader::edge_list_to_csr(edges, 3),  // node 5 >= 3
        std::invalid_argument
    );
}

// 48. Header-only edges file (no data rows)
void test_error_header_only_edges() {
    TempFile f("src,dst\n");
    // Should parse 0 edges (not throw), because an empty edge list
    // is technically valid.
    auto edges = GraphLoader::parse_edges(f.path());
    ASSERT_EQ(edges.size(), 0u);
}

// 49. Header-only features file (no data rows)
void test_error_header_only_features() {
    TempFile f("node_id,f0,f1\n");
    ASSERT_THROWS(
        GraphLoader::parse_features(f.path()),
        std::runtime_error
    );
}

// ════════════════════════════════════════════════════════════════════════════
//  MAIN
// ════════════════════════════════════════════════════════════════════════════
int main() {
    std::cout << "\n"
        "+==============================================+\n"
        "|  TinyGNN — GraphLoader Unit Tests (Phase 2)  |\n"
        "+==============================================+\n\n";

    std::cout << "-- Edge CSV Parsing ---------------------\n";
    RUN_TEST(test_parse_edges_with_header);
    RUN_TEST(test_parse_edges_no_header);
    RUN_TEST(test_parse_edges_crlf);
    RUN_TEST(test_parse_edges_single);
    RUN_TEST(test_parse_edges_self_loop);
    RUN_TEST(test_parse_edges_trailing_blanks);

    std::cout << "\n-- Feature CSV Parsing ------------------\n";
    RUN_TEST(test_parse_features_with_header);
    RUN_TEST(test_parse_features_no_header);
    RUN_TEST(test_parse_features_sparse_ids);
    RUN_TEST(test_parse_features_single);
    RUN_TEST(test_parse_features_unordered);
    RUN_TEST(test_parse_features_negative_values);

    std::cout << "\n-- Edge-List to CSR Conversion ----------\n";
    RUN_TEST(test_csr_known_graph);
    RUN_TEST(test_csr_unsorted_input);
    RUN_TEST(test_csr_empty_row);
    RUN_TEST(test_csr_self_loops);
    RUN_TEST(test_csr_no_edges);
    RUN_TEST(test_csr_single_edge);
    RUN_TEST(test_csr_sorted_invariant);
    RUN_TEST(test_csr_memory_footprint);

    std::cout << "\n-- Full Load Pipeline -------------------\n";
    RUN_TEST(test_load_small_graph);
    RUN_TEST(test_load_node0_neighbors);
    RUN_TEST(test_load_all_nodes_neighbors);
    RUN_TEST(test_load_edge_expands_features);

    std::cout << "\n-- Cora-Scale Validation ----------------\n";
    RUN_TEST(test_cora_scale_counts);
    RUN_TEST(test_cora_scale_csr_invariants);
    RUN_TEST(test_cora_scale_node0_neighbors);

    std::cout << "\n-- Reddit-Scale Validation --------------\n";
    std::cout << "  NOTE: Reddit tests are large (114M edges, ~1.8 GB peak RAM).\n";
    std::cout << "  Expected runtime: 30-120 s depending on hardware.\n";
    RUN_TEST(test_reddit_scale_csr_algorithm);
    RUN_TEST(test_reddit_scale_pipeline_counts);
    RUN_TEST(test_reddit_scale_csr_invariants);
    RUN_TEST(test_reddit_scale_node0_neighbors);

    std::cout << "\n-- Actual Cora Dataset ------------------\n";
    RUN_TEST(test_cora_actual_load);
    RUN_TEST(test_cora_actual_csr_invariants);
    RUN_TEST(test_cora_actual_features_binary);
    RUN_TEST(test_cora_actual_node0_neighbors);

    std::cout << "\n-- Actual Reddit Dataset ----------------\n";
    std::cout << "  NOTE: Loads ~2.7 GB of CSV data. May take 5-15 min.\n";
    RUN_TEST(test_reddit_actual_load);
    RUN_TEST(test_reddit_actual_csr_invariants);
    RUN_TEST(test_reddit_actual_feature_properties);
    RUN_TEST(test_reddit_actual_node0_neighbors);

    std::cout << "\n-- Error Handling -----------------------\n";
    RUN_TEST(test_error_file_not_found);
    RUN_TEST(test_error_empty_file);
    RUN_TEST(test_error_malformed_edge);
    RUN_TEST(test_error_non_integer_edge);
    RUN_TEST(test_error_negative_edge_id);
    RUN_TEST(test_error_negative_feature_id);
    RUN_TEST(test_error_inconsistent_features);
    RUN_TEST(test_error_edge_out_of_range);
    RUN_TEST(test_error_header_only_edges);
    RUN_TEST(test_error_header_only_features);

    // ── Summary ─────────────────────────────────────────────────────────
    std::cout << "\n==============================================\n";
    std::cout << "  Total : " << g_tests_run    << "\n";
    std::cout << "  Passed: " << g_tests_passed << "\n";
    std::cout << "  Failed: " << g_tests_failed << "\n";
    std::cout << "==============================================\n\n";

    return g_tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
