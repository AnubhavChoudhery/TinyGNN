// ============================================================================
//  TinyGNN — Tensor Unit Tests
//  Dependency-free test harness (no gtest / catch2 required)
// ============================================================================

#include "tinygnn/tensor.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// ── Minimal test framework ──────────────────────────────────────────────────
static int  g_tests_run    = 0;
static int  g_tests_passed = 0;
static int  g_tests_failed = 0;

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
                      << " — ASSERT_EQ(" #a ", " #b ") → "                   \
                      << (a) << " != " << (b) << "\n";                       \
            ++g_tests_failed;                                                 \
        } else {                                                              \
            ++g_tests_passed;                                                 \
        }                                                                     \
    } while (0)

#define ASSERT_FLOAT_EQ(a, b)                                                 \
    do {                                                                      \
        ++g_tests_run;                                                        \
        if (std::fabs((a) - (b)) > 1e-6f) {                                  \
            std::cerr << "  [FAIL] " << __FILE__ << ":" << __LINE__           \
                      << " — ASSERT_FLOAT_EQ(" #a ", " #b ") → "             \
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
                      << " — ASSERT_THROWS(" #expr ", " #exception_type ")\n";\
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
//  1.  Dense tensor — basic construction & properties
// ============================================================================
void test_dense_basic() {
    auto t = Tensor::dense(3, 4);

    ASSERT_TRUE(t.format() == StorageFormat::Dense);
    ASSERT_EQ(t.rows(), 3u);
    ASSERT_EQ(t.cols(), 4u);
    ASSERT_EQ(t.numel(), 12u);
    ASSERT_EQ(t.ndim(), 2u);
    ASSERT_EQ(t.nnz(), 12u);

    // Shape & strides
    ASSERT_EQ(t.shape().size(), 2u);
    ASSERT_EQ(t.shape()[0], 3u);
    ASSERT_EQ(t.shape()[1], 4u);
    ASSERT_EQ(t.strides().size(), 2u);
    ASSERT_EQ(t.strides()[0], 4u);   // row stride
    ASSERT_EQ(t.strides()[1], 1u);   // col stride

    // All zeros
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 4; ++c)
            ASSERT_FLOAT_EQ(t.at(r, c), 0.0f);
}

// ============================================================================
//  2.  Dense tensor — construction from raw data
// ============================================================================
void test_dense_from_data() {
    // 2×3 matrix: [[1,2,3],[4,5,6]]
    auto t = Tensor::dense(2, 3, {1, 2, 3, 4, 5, 6});

    ASSERT_EQ(t.rows(), 2u);
    ASSERT_EQ(t.cols(), 3u);
    ASSERT_FLOAT_EQ(t.at(0, 0), 1.0f);
    ASSERT_FLOAT_EQ(t.at(0, 2), 3.0f);
    ASSERT_FLOAT_EQ(t.at(1, 0), 4.0f);
    ASSERT_FLOAT_EQ(t.at(1, 2), 6.0f);
}

// ============================================================================
//  3.  Dense tensor — element mutation
// ============================================================================
void test_dense_mutation() {
    auto t = Tensor::dense(2, 2);
    t.at(0, 1) = 42.0f;
    ASSERT_FLOAT_EQ(t.at(0, 1), 42.0f);

    t.at(1, 0) = -3.14f;
    ASSERT_FLOAT_EQ(t.at(1, 0), -3.14f);
}

// ============================================================================
//  4.  Dense memory footprint (1000×1000)
// ============================================================================
void test_dense_memory_1000x1000() {
    auto t = Tensor::dense(1000, 1000);

    const std::size_t expected = 1000u * 1000u * sizeof(float);  // 4,000,000
    ASSERT_EQ(t.memory_footprint_bytes(), expected);
    ASSERT_EQ(t.numel(), 1000u * 1000u);

    std::cout << "    Dense  1000x1000 memory = "
              << t.memory_footprint_bytes() << " bytes\n";
}

// ============================================================================
//  5.  CSR sparse tensor — basic construction
// ============================================================================
void test_sparse_basic() {
    // 3×3 identity matrix in CSR
    //   row_ptr = [0, 1, 2, 3]
    //   col_ind = [0, 1, 2]
    //   values  = [1, 1, 1]
    auto t = Tensor::sparse_csr(
        3, 3,
        {0, 1, 2, 3},
        {0, 1, 2},
        {1.0f, 1.0f, 1.0f}
    );

    ASSERT_TRUE(t.format() == StorageFormat::SparseCSR);
    ASSERT_EQ(t.rows(), 3u);
    ASSERT_EQ(t.cols(), 3u);
    ASSERT_EQ(t.nnz(), 3u);

    // CSR metadata sizes
    ASSERT_EQ(t.row_ptr().size(), 4u);
    ASSERT_EQ(t.col_ind().size(), 3u);
    ASSERT_EQ(t.data().size(), 3u);

    // Strides should be empty for CSR
    ASSERT_TRUE(t.strides().empty());
}

// ============================================================================
//  6.  Sparse memory footprint (1000×1000 with 5000 edges)
// ============================================================================
void test_sparse_memory_1000x1000_5000nnz() {
    // Build a CSR tensor with 1000 rows, 1000 cols, 5000 non-zeros
    const std::size_t rows = 1000;
    const std::size_t cols = 1000;
    const std::size_t nnz  = 5000;

    // Distribute nnz evenly: 5 per row
    std::vector<int32_t> row_ptr(rows + 1);
    std::vector<int32_t> col_ind(nnz);
    std::vector<float>   values(nnz, 1.0f);

    const std::size_t per_row = nnz / rows;  // 5
    for (std::size_t r = 0; r < rows; ++r) {
        row_ptr[r] = static_cast<int32_t>(r * per_row);
        for (std::size_t j = 0; j < per_row; ++j) {
            col_ind[r * per_row + j] = static_cast<int32_t>(j);
        }
    }
    row_ptr[rows] = static_cast<int32_t>(nnz);

    auto t = Tensor::sparse_csr(rows, cols,
                                std::move(row_ptr),
                                std::move(col_ind),
                                std::move(values));

    //  Expected footprint:
    //    values  : 5000 × 4 =  20,000 bytes
    //    col_ind : 5000 × 4 =  20,000 bytes
    //    row_ptr : 1001 × 4 =   4,004 bytes
    //    total   :             44,004 bytes
    const std::size_t expected_values  = nnz * sizeof(float);       // 20000
    const std::size_t expected_col_ind = nnz * sizeof(int32_t);     // 20000
    const std::size_t expected_row_ptr = (rows + 1) * sizeof(int32_t); // 4004
    const std::size_t expected_total   = expected_values + expected_col_ind + expected_row_ptr;

    ASSERT_EQ(t.memory_footprint_bytes(), expected_total);
    ASSERT_EQ(expected_total, 44004u);

    std::cout << "    Sparse 1000x1000 (5000 nnz) memory = "
              << t.memory_footprint_bytes() << " bytes\n";
}

// ============================================================================
//  7.  Memory footprint comparison — sparse vs. dense
// ============================================================================
void test_memory_reduction() {
    auto dense  = Tensor::dense(1000, 1000);

    const std::size_t rows = 1000, cols = 1000, nnz = 5000;
    std::vector<int32_t> rp(rows + 1);
    std::vector<int32_t> ci(nnz);
    std::vector<float>   vals(nnz, 1.0f);
    for (std::size_t r = 0; r < rows; ++r) {
        rp[r] = static_cast<int32_t>(r * 5);
        for (int j = 0; j < 5; ++j)
            ci[r * 5 + j] = static_cast<int32_t>(j);
    }
    rp[rows] = static_cast<int32_t>(nnz);

    auto sparse = Tensor::sparse_csr(rows, cols,
                                     std::move(rp), std::move(ci),
                                     std::move(vals));

    const double ratio = static_cast<double>(sparse.memory_footprint_bytes())
                       / static_cast<double>(dense.memory_footprint_bytes());

    std::cout << "    Memory ratio (sparse / dense) = " << ratio << "\n";

    // Sparse should use ≈ 1.1% of dense memory (44004 / 4000000)
    ASSERT_TRUE(ratio < 0.02);   // Must be less than 2%
    ASSERT_TRUE(sparse.memory_footprint_bytes() < dense.memory_footprint_bytes());
}

// ============================================================================
//  8.  Default (empty) tensor
// ============================================================================
void test_default_tensor() {
    Tensor t;
    ASSERT_TRUE(t.format() == StorageFormat::Dense);
    ASSERT_EQ(t.rows(), 0u);
    ASSERT_EQ(t.cols(), 0u);
    ASSERT_EQ(t.numel(), 0u);
    ASSERT_EQ(t.memory_footprint_bytes(), 0u);
    ASSERT_TRUE(t.data().empty());
    ASSERT_TRUE(t.row_ptr().empty());
    ASSERT_TRUE(t.col_ind().empty());
}

// ============================================================================
//  9.  Edge case — single element tensors
// ============================================================================
void test_single_element() {
    // Dense 1×1
    auto d = Tensor::dense(1, 1, {7.0f});
    ASSERT_EQ(d.numel(), 1u);
    ASSERT_FLOAT_EQ(d.at(0, 0), 7.0f);
    ASSERT_EQ(d.memory_footprint_bytes(), sizeof(float));

    // Sparse 1×1 with one non-zero
    auto s = Tensor::sparse_csr(1, 1, {0, 1}, {0}, {7.0f});
    ASSERT_EQ(s.nnz(), 1u);
    ASSERT_EQ(s.memory_footprint_bytes(),
              1 * sizeof(float) + 1 * sizeof(int32_t) + 2 * sizeof(int32_t));
}

// ============================================================================
//  10. Edge case — sparse tensor with zero non-zeros
// ============================================================================
void test_sparse_empty() {
    // 5×5 matrix with no non-zeros
    std::vector<int32_t> rp(6, 0);  // all zeros → no entries in any row
    auto t = Tensor::sparse_csr(5, 5, std::move(rp), {}, {});

    ASSERT_EQ(t.nnz(), 0u);
    ASSERT_EQ(t.rows(), 5u);
    ASSERT_EQ(t.cols(), 5u);
    ASSERT_EQ(t.data().size(), 0u);
    // Memory: only row_ptr overhead → 6 × 4 = 24 bytes
    ASSERT_EQ(t.memory_footprint_bytes(), 6u * sizeof(int32_t));
}

// ============================================================================
//  11. Edge case — large sparse ratio (very sparse)
// ============================================================================
void test_very_sparse() {
    // 10000×10000 with only 10 edges
    const std::size_t rows = 10000, cols = 10000, nnz = 10;

    std::vector<int32_t> rp(rows + 1, 0);
    std::vector<int32_t> ci;
    std::vector<float>   vals;

    // Put all 10 entries in row 0
    rp[0] = 0;
    for (std::size_t i = 0; i < nnz; ++i) {
        ci.push_back(static_cast<int32_t>(i));
        vals.push_back(1.0f);
    }
    for (std::size_t r = 1; r <= rows; ++r)
        rp[r] = static_cast<int32_t>(nnz);

    auto t = Tensor::sparse_csr(rows, cols, std::move(rp),
                                std::move(ci), std::move(vals));

    auto d = Tensor::dense(rows, cols);

    // Dense: 10000 * 10000 * 4 = 400,000,000 bytes
    // Sparse: 10*4 + 10*4 + 10001*4 = 40 + 40 + 40004 = 40,084 bytes
    ASSERT_EQ(d.memory_footprint_bytes(), 400000000u);
    ASSERT_EQ(t.memory_footprint_bytes(), 40084u);

    double ratio = static_cast<double>(t.memory_footprint_bytes())
                 / static_cast<double>(d.memory_footprint_bytes());
    std::cout << "    Very sparse ratio (10 nnz in 10k×10k) = " << ratio << "\n";
    ASSERT_TRUE(ratio < 0.001);  // < 0.1% memory
}

// ============================================================================
//  12. Error handling — invalid construction
// ============================================================================
void test_invalid_dense_construction() {
    // Wrong data size
    ASSERT_THROWS(
        Tensor::dense(2, 3, {1, 2, 3}),
        std::invalid_argument
    );
}

void test_invalid_sparse_construction() {
    // row_ptr wrong size
    ASSERT_THROWS(
        Tensor::sparse_csr(3, 3, {0, 1, 2}, {0, 1, 2}, {1, 1, 1}),
        std::invalid_argument
    );

    // col_ind / values size mismatch
    ASSERT_THROWS(
        Tensor::sparse_csr(3, 3, {0, 1, 2, 3}, {0, 1}, {1, 1, 1}),
        std::invalid_argument
    );

    // row_ptr.back() != values.size()
    ASSERT_THROWS(
        Tensor::sparse_csr(3, 3, {0, 1, 2, 5}, {0, 1, 2}, {1, 1, 1}),
        std::invalid_argument
    );

    // col_ind out of bounds
    ASSERT_THROWS(
        Tensor::sparse_csr(3, 3, {0, 1, 2, 3}, {0, 1, 99}, {1, 1, 1}),
        std::invalid_argument
    );

    // row_ptr not non-decreasing
    ASSERT_THROWS(
        Tensor::sparse_csr(3, 3, {0, 2, 1, 3}, {0, 1, 2}, {1, 1, 1}),
        std::invalid_argument
    );
}

// ============================================================================
//  13. Error handling — out-of-bounds access
// ============================================================================
void test_out_of_bounds() {
    auto t = Tensor::dense(3, 3);

    ASSERT_THROWS(t.at(3, 0), std::out_of_range);
    ASSERT_THROWS(t.at(0, 3), std::out_of_range);
    ASSERT_THROWS(t.at(100, 100), std::out_of_range);
}

// ============================================================================
//  14. Error handling — at() on sparse tensor
// ============================================================================
void test_at_on_sparse() {
    auto t = Tensor::sparse_csr(3, 3, {0, 1, 2, 3}, {0, 1, 2}, {1, 1, 1});

    ASSERT_THROWS(t.at(0, 0), std::runtime_error);
}

// ============================================================================
//  15. repr() string output
// ============================================================================
void test_repr() {
    auto d = Tensor::dense(10, 20);
    std::string r = d.repr();
    ASSERT_TRUE(r.find("10x20") != std::string::npos);
    ASSERT_TRUE(r.find("Dense") != std::string::npos);
    ASSERT_TRUE(r.find("800") != std::string::npos);  // 10*20*4=800

    auto s = Tensor::sparse_csr(3, 3, {0, 1, 2, 3}, {0, 1, 2}, {1, 1, 1});
    std::string sr = s.repr();
    ASSERT_TRUE(sr.find("3x3") != std::string::npos);
    ASSERT_TRUE(sr.find("SparseCSR") != std::string::npos);
}

// ============================================================================
//  16. Dense zero-column / zero-row edge cases
// ============================================================================
void test_degenerate_shapes() {
    // 0×5 matrix
    auto t1 = Tensor::dense(0, 5);
    ASSERT_EQ(t1.numel(), 0u);
    ASSERT_EQ(t1.memory_footprint_bytes(), 0u);
    ASSERT_EQ(t1.rows(), 0u);
    ASSERT_EQ(t1.cols(), 5u);

    // 5×0 matrix
    auto t2 = Tensor::dense(5, 0);
    ASSERT_EQ(t2.numel(), 0u);
    ASSERT_EQ(t2.memory_footprint_bytes(), 0u);
}

// ============================================================================
//  17. CSR data integrity — verify values are stored correctly
// ============================================================================
void test_csr_data_integrity() {
    // 4×4 matrix:
    //  [ 0  0  3  0 ]
    //  [ 0  0  0  0 ]
    //  [ 1  0  0  2 ]
    //  [ 0  5  0  0 ]
    auto t = Tensor::sparse_csr(
        4, 4,
        {0, 1, 1, 3, 4},         // row_ptr
        {2, 0, 3, 1},            // col_ind
        {3.0f, 1.0f, 2.0f, 5.0f} // values
    );

    ASSERT_EQ(t.nnz(), 4u);

    // Verify row_ptr
    ASSERT_EQ(t.row_ptr()[0], 0);
    ASSERT_EQ(t.row_ptr()[1], 1);
    ASSERT_EQ(t.row_ptr()[2], 1);  // empty row
    ASSERT_EQ(t.row_ptr()[3], 3);
    ASSERT_EQ(t.row_ptr()[4], 4);

    // Verify col_ind
    ASSERT_EQ(t.col_ind()[0], 2);
    ASSERT_EQ(t.col_ind()[1], 0);
    ASSERT_EQ(t.col_ind()[2], 3);
    ASSERT_EQ(t.col_ind()[3], 1);

    // Verify values
    ASSERT_FLOAT_EQ(t.data()[0], 3.0f);
    ASSERT_FLOAT_EQ(t.data()[1], 1.0f);
    ASSERT_FLOAT_EQ(t.data()[2], 2.0f);
    ASSERT_FLOAT_EQ(t.data()[3], 5.0f);
}

// ============================================================================
//  18. StorageFormat enum values
// ============================================================================
void test_storage_format_enum() {
    ASSERT_EQ(static_cast<uint8_t>(StorageFormat::Dense), 0u);
    ASSERT_EQ(static_cast<uint8_t>(StorageFormat::SparseCSR), 1u);
}

// ============================================================================
//  main — run all tests
// ============================================================================
int main() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              <<   "║      TinyGNN — Tensor Unit Tests         ║\n"
              <<   "╚══════════════════════════════════════════╝\n\n";

    std::cout << "── Dense Tensor Tests ──────────────────────\n";
    RUN_TEST(test_dense_basic);
    RUN_TEST(test_dense_from_data);
    RUN_TEST(test_dense_mutation);
    RUN_TEST(test_dense_memory_1000x1000);

    std::cout << "\n── Sparse CSR Tensor Tests ─────────────────\n";
    RUN_TEST(test_sparse_basic);
    RUN_TEST(test_sparse_memory_1000x1000_5000nnz);

    std::cout << "\n── Memory Comparison Tests ─────────────────\n";
    RUN_TEST(test_memory_reduction);
    RUN_TEST(test_very_sparse);

    std::cout << "\n── Edge Case Tests ─────────────────────────\n";
    RUN_TEST(test_default_tensor);
    RUN_TEST(test_single_element);
    RUN_TEST(test_sparse_empty);
    RUN_TEST(test_degenerate_shapes);
    RUN_TEST(test_csr_data_integrity);
    RUN_TEST(test_storage_format_enum);

    std::cout << "\n── Error Handling Tests ─────────────────────\n";
    RUN_TEST(test_invalid_dense_construction);
    RUN_TEST(test_invalid_sparse_construction);
    RUN_TEST(test_out_of_bounds);
    RUN_TEST(test_at_on_sparse);
    RUN_TEST(test_repr);

    // ── Summary ─────────────────────────────────────────────────────────
    std::cout << "\n══════════════════════════════════════════════\n";
    std::cout << "  Total : " << g_tests_run    << "\n";
    std::cout << "  Passed: " << g_tests_passed << "\n";
    std::cout << "  Failed: " << g_tests_failed << "\n";
    std::cout << "══════════════════════════════════════════════\n\n";

    return g_tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
