#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace tinygnn {

// ============================================================================
//  StorageFormat – distinguishes dense vs. sparse layouts
// ============================================================================
enum class StorageFormat : uint8_t {
    Dense     = 0,  // Row-major contiguous storage
    SparseCSR = 1   // Compressed Sparse Row
};

// ============================================================================
//  Tensor – hybrid dense / sparse tensor
// ============================================================================
//  • Dense   : data_ holds rows × cols floats in row-major order
//  • CSR     : row_ptr_ (size rows+1), col_ind_ (size nnz), data_ (size nnz)
// ============================================================================
struct Tensor {
    // ── Construction ────────────────────────────────────────────────────────
    /// Create an empty tensor (0×0, Dense)
    Tensor();

    /// Create a dense tensor filled with zeros
    static Tensor dense(std::size_t rows, std::size_t cols);

    /// Create a dense tensor from raw data (row-major)
    static Tensor dense(std::size_t rows, std::size_t cols,
                        std::vector<float> data);

    /// Create a CSR sparse tensor
    /// @param rows       Number of rows
    /// @param cols       Number of columns
    /// @param row_ptr    Row pointer array  (size = rows + 1)
    /// @param col_ind    Column index array  (size = nnz)
    /// @param values     Non-zero values     (size = nnz)
    static Tensor sparse_csr(std::size_t rows, std::size_t cols,
                             std::vector<int32_t> row_ptr,
                             std::vector<int32_t> col_ind,
                             std::vector<float> values);

    // ── Observers ───────────────────────────────────────────────────────────
    StorageFormat format()  const noexcept { return format_; }
    std::size_t   rows()    const noexcept { return shape_[0]; }
    std::size_t   cols()    const noexcept { return shape_[1]; }
    std::size_t   numel()   const noexcept { return data_.size(); }
    std::size_t   ndim()    const noexcept { return shape_.size(); }

    /// Number of non-zero elements (== numel() for Dense)
    std::size_t   nnz()     const noexcept;

    /// Total bytes consumed by the tensor's data vectors (not sizeof(Tensor))
    /// Dense  : rows * cols * sizeof(float)
    /// CSR    : nnz * sizeof(float) + nnz * sizeof(int32_t) + (rows+1) * sizeof(int32_t)
    std::size_t   memory_footprint_bytes() const noexcept;

    // ── Accessors ───────────────────────────────────────────────────────────
    const std::vector<std::size_t>& shape()   const noexcept { return shape_; }
    const std::vector<std::size_t>& strides() const noexcept { return strides_; }

    // Raw data (values for both Dense and CSR)
    const std::vector<float>&   data()    const noexcept { return data_; }
    std::vector<float>&         data()          noexcept { return data_; }

    // CSR-specific accessors (empty for Dense tensors)
    const std::vector<int32_t>& row_ptr() const noexcept { return row_ptr_; }
    const std::vector<int32_t>& col_ind() const noexcept { return col_ind_; }

    // ── Element access ──────────────────────────────────────────────────────
    /// Access element at (row, col)  — Dense only, throws for CSR
    float  at(std::size_t row, std::size_t col) const;
    float& at(std::size_t row, std::size_t col);

    // ── Utility ─────────────────────────────────────────────────────────────
    /// Human-readable summary  "Tensor(1000x1000, Dense, 4000000 bytes)"
    std::string repr() const;

private:
    StorageFormat             format_  = StorageFormat::Dense;
    std::vector<std::size_t>  shape_;    // {rows, cols}
    std::vector<std::size_t>  strides_;  // row-major strides (Dense only)

    std::vector<float>        data_;     // Dense: all values · CSR: non-zero values

    // CSR metadata (empty for Dense)
    std::vector<int32_t>      row_ptr_;
    std::vector<int32_t>      col_ind_;
};

}  // namespace tinygnn
