#include "tinygnn/tensor.hpp"

#include <sstream>
#include <stdexcept>

namespace tinygnn {

// ============================================================================
//  Construction
// ============================================================================

Tensor::Tensor()
    : format_(StorageFormat::Dense),
      shape_{0, 0},
      strides_{0, 0} {}

Tensor Tensor::dense(std::size_t rows, std::size_t cols) {
    Tensor t;
    t.format_  = StorageFormat::Dense;
    t.shape_   = {rows, cols};
    t.strides_ = {cols, 1};                       // row-major
    t.data_.resize(rows * cols, 0.0f);
    return t;
}

Tensor Tensor::dense(std::size_t rows, std::size_t cols,
                     std::vector<float> data) {
    if (data.size() != rows * cols) {
        throw std::invalid_argument(
            "Tensor::dense: data.size() (" + std::to_string(data.size()) +
            ") != rows*cols (" + std::to_string(rows * cols) + ")");
    }
    Tensor t;
    t.format_  = StorageFormat::Dense;
    t.shape_   = {rows, cols};
    t.strides_ = {cols, 1};
    t.data_    = std::move(data);
    return t;
}

Tensor Tensor::sparse_csr(std::size_t rows, std::size_t cols,
                          std::vector<int32_t> row_ptr,
                          std::vector<int32_t> col_ind,
                          std::vector<float> values) {
    // ── validation ──────────────────────────────────────────────────────
    if (row_ptr.size() != rows + 1) {
        throw std::invalid_argument(
            "Tensor::sparse_csr: row_ptr.size() (" +
            std::to_string(row_ptr.size()) +
            ") must equal rows+1 (" + std::to_string(rows + 1) + ")");
    }
    if (col_ind.size() != values.size()) {
        throw std::invalid_argument(
            "Tensor::sparse_csr: col_ind.size() (" +
            std::to_string(col_ind.size()) +
            ") must equal values.size() (" +
            std::to_string(values.size()) + ")");
    }
    const auto nnz = static_cast<std::size_t>(row_ptr.back());
    if (nnz != values.size()) {
        throw std::invalid_argument(
            "Tensor::sparse_csr: row_ptr.back() (" +
            std::to_string(nnz) +
            ") must equal values.size() (" +
            std::to_string(values.size()) + ")");
    }
    // Validate column indices are in bounds
    for (std::size_t i = 0; i < col_ind.size(); ++i) {
        if (col_ind[i] < 0 || static_cast<std::size_t>(col_ind[i]) >= cols) {
            throw std::invalid_argument(
                "Tensor::sparse_csr: col_ind[" + std::to_string(i) +
                "] = " + std::to_string(col_ind[i]) +
                " is out of bounds for cols=" + std::to_string(cols));
        }
    }
    // Validate row_ptr is non-decreasing
    for (std::size_t i = 1; i < row_ptr.size(); ++i) {
        if (row_ptr[i] < row_ptr[i - 1]) {
            throw std::invalid_argument(
                "Tensor::sparse_csr: row_ptr must be non-decreasing; "
                "row_ptr[" + std::to_string(i) + "] = " +
                std::to_string(row_ptr[i]) + " < row_ptr[" +
                std::to_string(i - 1) + "] = " +
                std::to_string(row_ptr[i - 1]));
        }
    }

    Tensor t;
    t.format_  = StorageFormat::SparseCSR;
    t.shape_   = {rows, cols};
    t.strides_ = {};                // strides are meaningless for CSR
    t.row_ptr_ = std::move(row_ptr);
    t.col_ind_ = std::move(col_ind);
    t.data_    = std::move(values);
    return t;
}

// ============================================================================
//  Observers
// ============================================================================

std::size_t Tensor::nnz() const noexcept {
    switch (format_) {
        case StorageFormat::Dense:
            return data_.size();       // every element counts
        case StorageFormat::SparseCSR:
            return data_.size();       // only non-zeros stored
    }
    return data_.size(); // unreachable, silences warnings
}

std::size_t Tensor::memory_footprint_bytes() const noexcept {
    switch (format_) {
        case StorageFormat::Dense:
            // All values stored contiguously
            return data_.size() * sizeof(float);

        case StorageFormat::SparseCSR:
            // values (nnz floats) + col_ind (nnz int32s) + row_ptr ((rows+1) int32s)
            return data_.size()    * sizeof(float)
                 + col_ind_.size() * sizeof(int32_t)
                 + row_ptr_.size() * sizeof(int32_t);
    }
    return 0; // unreachable
}

// ============================================================================
//  Element access
// ============================================================================

float Tensor::at(std::size_t row, std::size_t col) const {
    if (format_ != StorageFormat::Dense) {
        throw std::runtime_error(
            "Tensor::at() is only supported for Dense tensors");
    }
    if (row >= shape_[0] || col >= shape_[1]) {
        throw std::out_of_range(
            "Tensor::at(" + std::to_string(row) + ", " +
            std::to_string(col) + ") out of range for shape (" +
            std::to_string(shape_[0]) + ", " +
            std::to_string(shape_[1]) + ")");
    }
    return data_[row * strides_[0] + col * strides_[1]];
}

float& Tensor::at(std::size_t row, std::size_t col) {
    if (format_ != StorageFormat::Dense) {
        throw std::runtime_error(
            "Tensor::at() is only supported for Dense tensors");
    }
    if (row >= shape_[0] || col >= shape_[1]) {
        throw std::out_of_range(
            "Tensor::at(" + std::to_string(row) + ", " +
            std::to_string(col) + ") out of range for shape (" +
            std::to_string(shape_[0]) + ", " +
            std::to_string(shape_[1]) + ")");
    }
    return data_[row * strides_[0] + col * strides_[1]];
}

// ============================================================================
//  Utility
// ============================================================================

std::string Tensor::repr() const {
    std::ostringstream os;
    os << "Tensor(" << shape_[0] << "x" << shape_[1] << ", ";
    switch (format_) {
        case StorageFormat::Dense:     os << "Dense";     break;
        case StorageFormat::SparseCSR: os << "SparseCSR"; break;
    }
    os << ", " << memory_footprint_bytes() << " bytes)";
    return os.str();
}

}  // namespace tinygnn
