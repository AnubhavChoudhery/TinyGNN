// ============================================================================
//  TinyGNN — GraphLoader Unit Tests  (Phase 2)
//  Dependency-free test harness (same framework as test_tensor.cpp)
// ============================================================================
//
//  Test categories:
//    1. Edge CSV parsing                   (tests  1 – 6)
//    2. Feature CSV parsing                (tests  7 – 12)
//    3. Edge-list → sorted CSR conversion  (tests 13 – 20)
//    4. Full pipeline (load)               (tests 21 – 24)
//    5. Cora-scale validation              (tests 25 – 27)
//    6. Error handling                     (tests 28 – 37)
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
//  6.  ERROR HANDLING
// ════════════════════════════════════════════════════════════════════════════

// 28. File not found
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

// 29. Empty file
void test_error_empty_file() {
    TempFile f("");
    ASSERT_THROWS(
        GraphLoader::parse_edges(f.path()),
        std::runtime_error
    );
}

// 30. Malformed edge line (no comma)
void test_error_malformed_edge() {
    TempFile f("0 1\n");
    ASSERT_THROWS(
        GraphLoader::parse_edges(f.path()),
        std::runtime_error
    );
}

// 31. Non-integer in edge file
void test_error_non_integer_edge() {
    TempFile f("0,abc\n");
    ASSERT_THROWS(
        GraphLoader::parse_edges(f.path()),
        std::runtime_error
    );
}

// 32. Negative node ID in edges
void test_error_negative_edge_id() {
    TempFile f("-1,2\n");
    ASSERT_THROWS(
        GraphLoader::parse_edges(f.path()),
        std::runtime_error
    );
}

// 33. Negative node ID in features
void test_error_negative_feature_id() {
    TempFile f("-1,1.0,2.0\n");
    ASSERT_THROWS(
        GraphLoader::parse_features(f.path()),
        std::runtime_error
    );
}

// 34. Inconsistent feature counts
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

// 35. Edge out of range for CSR conversion
void test_error_edge_out_of_range() {
    std::vector<std::pair<int32_t, int32_t>> edges = {{0, 5}};
    ASSERT_THROWS(
        GraphLoader::edge_list_to_csr(edges, 3),  // node 5 >= 3
        std::invalid_argument
    );
}

// 36. Header-only edges file (no data rows)
void test_error_header_only_edges() {
    TempFile f("src,dst\n");
    // Should parse 0 edges (not throw), because an empty edge list
    // is technically valid.
    auto edges = GraphLoader::parse_edges(f.path());
    ASSERT_EQ(edges.size(), 0u);
}

// 37. Header-only features file (no data rows)
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
