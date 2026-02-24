// ============================================================================
//  TinyGNN — GNN Layer Implementations  (Phase 6)
//  src/layers.cpp
//
//  Implements:
//    • add_self_loops   — Ã = A + I
//    • gcn_norm         — D̃^{-1/2} Ã D̃^{-1/2}
//    • GCNLayer         — GCN forward pass (Kipf & Welling, 2017)
//    • SAGELayer        — GraphSAGE forward pass (Hamilton et al., 2017)
//    • sage_max_aggregate — element-wise max pooling over neighbors
// ============================================================================

#include "tinygnn/layers.hpp"
#include "tinygnn/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tinygnn {

// ============================================================================
//  Helper — validate square SparseCSR matrix
// ============================================================================
static void require_square_csr(const Tensor& A, const char* fn_name) {
    if (A.format() != StorageFormat::SparseCSR) {
        throw std::invalid_argument(
            std::string(fn_name) + ": A must be SparseCSR (got Dense).");
    }
    if (A.rows() != A.cols()) {
        throw std::invalid_argument(
            std::string(fn_name) + ": A must be square — got (" +
            std::to_string(A.rows()) + "×" + std::to_string(A.cols()) + ").");
    }
}

// ============================================================================
//  add_self_loops — Ã = A + I
// ============================================================================
//
//  Algorithm:
//    1. First pass:  for each row i, scan col_ind to detect whether the
//       diagonal entry (i, i) already exists.  Count how many new entries
//       we need to insert.
//    2. Allocate output CSR arrays with new_nnz = old_nnz + extra_diags.
//    3. Second pass:  for each row i, copy existing entries and insert /
//       update the diagonal entry in sorted column order.
//
//  The output CSR preserves sorted col_ind order per row.
//
// ============================================================================
Tensor add_self_loops(const Tensor& A) {
    require_square_csr(A, "add_self_loops");

    const std::size_t N = A.rows();
    const auto& rp_in  = A.row_ptr();
    const auto& ci_in  = A.col_ind();
    const auto& v_in   = A.data();

    // ── Pass 1: detect existing diagonal entries ────────────────────────────
    std::vector<bool> has_diag(N, false);
    std::size_t extra = 0;

    for (std::size_t i = 0; i < N; ++i) {
        for (int32_t nz = rp_in[i]; nz < rp_in[i + 1]; ++nz) {
            if (ci_in[nz] == static_cast<int32_t>(i)) {
                has_diag[i] = true;
                break;
            }
        }
        if (!has_diag[i]) ++extra;
    }

    const std::size_t new_nnz = A.nnz() + extra;

    // ── Pass 2: build output CSR ────────────────────────────────────────────
    std::vector<int32_t> rp_out(N + 1);
    std::vector<int32_t> ci_out;
    std::vector<float>   v_out;
    ci_out.reserve(new_nnz);
    v_out.reserve(new_nnz);

    for (std::size_t i = 0; i < N; ++i) {
        rp_out[i] = static_cast<int32_t>(ci_out.size());

        const int32_t row_start = rp_in[i];
        const int32_t row_end   = rp_in[i + 1];
        bool diag_written = false;

        for (int32_t nz = row_start; nz < row_end; ++nz) {
            const int32_t col = ci_in[nz];

            // Insert new diagonal entry before any column > i (sorted order)
            if (!diag_written && !has_diag[i] &&
                col > static_cast<int32_t>(i)) {
                ci_out.push_back(static_cast<int32_t>(i));
                v_out.push_back(1.0f);
                diag_written = true;
            }

            if (col == static_cast<int32_t>(i)) {
                // Existing diagonal: A[i][i] + 1.0
                ci_out.push_back(col);
                v_out.push_back(v_in[nz] + 1.0f);
                diag_written = true;
            } else {
                // Non-diagonal: copy unchanged
                ci_out.push_back(col);
                v_out.push_back(v_in[nz]);
            }
        }

        // Diagonal still not written — all cols in this row are < i, or empty
        if (!diag_written) {
            ci_out.push_back(static_cast<int32_t>(i));
            v_out.push_back(1.0f);
        }
    }
    rp_out[N] = static_cast<int32_t>(ci_out.size());

    return Tensor::sparse_csr(N, N,
                              std::move(rp_out),
                              std::move(ci_out),
                              std::move(v_out));
}

// ============================================================================
//  gcn_norm — D̃^{-1/2} · (A + I) · D̃^{-1/2}  #Already planning for an optimized solution.
// ============================================================================
//
//  Algorithm:
//    1. Compute Ã = add_self_loops(A)
//    2. Compute degree d̃[i] = Σ_j Ã[i][j] for each row
//    3. Compute deg_inv_sqrt[i] = 1 / √d̃[i]  (0 if d̃[i] == 0)
//    4. For each non-zero Ã[i][j], scale by deg_inv_sqrt[i] * deg_inv_sqrt[j]
//    5. Return new CSR with same structure, scaled values
//
// ============================================================================
Tensor gcn_norm(const Tensor& A) {
    require_square_csr(A, "gcn_norm");

    // Step 1: add self-loops
    Tensor A_hat = add_self_loops(A);

    const std::size_t N  = A_hat.rows();
    const auto& rp       = A_hat.row_ptr();
    const auto& ci       = A_hat.col_ind();
    const auto& vals     = A_hat.data();

    // Step 2: compute degree d̃[i] = Σ_j Ã[i][j]
    std::vector<float> deg(N, 0.0f);
    for (std::size_t i = 0; i < N; ++i) {
        for (int32_t nz = rp[i]; nz < rp[i + 1]; ++nz) {
            deg[i] += vals[nz];
        }
    }

    // Step 3: deg_inv_sqrt[i] = 1 / √deg[i]
    std::vector<float> deg_inv_sqrt(N);
    for (std::size_t i = 0; i < N; ++i) {
        deg_inv_sqrt[i] = (deg[i] > 0.0f)
                          ? (1.0f / std::sqrt(deg[i]))
                          : 0.0f;
    }

    // Step 4: scale each non-zero  Â[i][j] = Ã[i][j] * dis[i] * dis[j]
    std::vector<float> new_vals(vals.size());
    for (std::size_t i = 0; i < N; ++i) {
        const float dis_i = deg_inv_sqrt[i];
        for (int32_t nz = rp[i]; nz < rp[i + 1]; ++nz) {
            const auto j = static_cast<std::size_t>(ci[nz]);
            new_vals[nz] = vals[nz] * dis_i * deg_inv_sqrt[j];
        }
    }

    // Step 5: build output CSR (same structure, new values)
    std::vector<int32_t> rp_copy(rp.begin(), rp.end());
    std::vector<int32_t> ci_copy(ci.begin(), ci.end());

    return Tensor::sparse_csr(N, N,
                              std::move(rp_copy),
                              std::move(ci_copy),
                              std::move(new_vals));
}

// ============================================================================
//  GCNLayer — constructor
// ============================================================================
GCNLayer::GCNLayer(std::size_t in_features, std::size_t out_features,
                   bool use_bias, Activation act)
    : in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias),
      activation_(act),
      weight_(Tensor::dense(in_features, out_features)),   // zero-init
      bias_(Tensor::dense(1, out_features))                // zero-init
{
    if (in_features == 0) {
        throw std::invalid_argument(
            "GCNLayer: in_features must be > 0.");
    }
    if (out_features == 0) {
        throw std::invalid_argument(
            "GCNLayer: out_features must be > 0.");
    }
}

// ============================================================================
//  GCNLayer — set_weight
// ============================================================================
void GCNLayer::set_weight(Tensor w) {
    if (w.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            "GCNLayer::set_weight: weight must be Dense.");
    }
    if (w.rows() != in_features_ || w.cols() != out_features_) {
        throw std::invalid_argument(
            "GCNLayer::set_weight: expected shape (" +
            std::to_string(in_features_) + "×" +
            std::to_string(out_features_) + "), got (" +
            std::to_string(w.rows()) + "×" +
            std::to_string(w.cols()) + ").");
    }
    weight_ = std::move(w);
}

// ============================================================================
//  GCNLayer — set_bias
// ============================================================================
void GCNLayer::set_bias(Tensor b) {
    if (!use_bias_) {
        throw std::invalid_argument(
            "GCNLayer::set_bias: layer was constructed with use_bias=false.");
    }
    if (b.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            "GCNLayer::set_bias: bias must be Dense.");
    }
    if (b.rows() != 1 || b.cols() != out_features_) {
        throw std::invalid_argument(
            "GCNLayer::set_bias: expected shape (1×" +
            std::to_string(out_features_) + "), got (" +
            std::to_string(b.rows()) + "×" +
            std::to_string(b.cols()) + ").");
    }
    bias_ = std::move(b);
}

// ============================================================================
//  GCNLayer — forward
// ============================================================================
//
//  H' = σ( A_norm · (H · W) + b )
//
//  Steps:
//    1. HW = matmul(H, W)            —  (N × F_in) × (F_in × F_out) → (N × F_out)
//    2. out = spmm(A_norm, HW)       —  (N × N) × (N × F_out) → (N × F_out)
//    3. add_bias(out, bias)           —  in-place broadcast add
//    4. activation(out)               —  in-place non-linearity
//
// ============================================================================
Tensor GCNLayer::forward(const Tensor& A_norm, const Tensor& H) const {
    // ── Validate A_norm ──────────────────────────────────────────────────────
    if (A_norm.format() != StorageFormat::SparseCSR) {
        throw std::invalid_argument(
            "GCNLayer::forward: A_norm must be SparseCSR (got Dense). "
            "Use gcn_norm() to precompute the normalized adjacency.");
    }
    if (A_norm.rows() != A_norm.cols()) {
        throw std::invalid_argument(
            "GCNLayer::forward: A_norm must be square — got (" +
            std::to_string(A_norm.rows()) + "×" +
            std::to_string(A_norm.cols()) + ").");
    }

    // ── Validate H ───────────────────────────────────────────────────────────
    if (H.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            "GCNLayer::forward: H must be Dense (got SparseCSR).");
    }
    if (H.cols() != in_features_) {
        throw std::invalid_argument(
            "GCNLayer::forward: H has " + std::to_string(H.cols()) +
            " columns but layer expects in_features=" +
            std::to_string(in_features_) + ".");
    }
    if (A_norm.rows() != H.rows()) {
        throw std::invalid_argument(
            "GCNLayer::forward: A_norm has " +
            std::to_string(A_norm.rows()) + " rows but H has " +
            std::to_string(H.rows()) + " rows. They must match (N nodes).");
    }

    // ── Step 1: Linear transform  HW = H · W ────────────────────────────────
    Tensor HW = matmul(H, weight_);

    // ── Step 2: Message passing   out = A_norm · HW ─────────────────────────
    Tensor out = spmm(A_norm, HW);

    // ── Step 3: Bias addition ────────────────────────────────────────────────
    if (use_bias_) {
        add_bias(out, bias_);
    }

    // ── Step 4: Activation (Important) ───────────────────────────────────────────────────
    switch (activation_) {
        case Activation::ReLU:
            relu_inplace(out);
            break;
        case Activation::None:
            break;
    }

    return out;
}

// ============================================================================
//  sage_max_aggregate — element-wise max pooling over neighbor features
// ============================================================================
//
//  For each node i:
//    for each feature f:
//      result[i][f] = max_{j ∈ N(i)} H[j][f]
//
//  Nodes with degree 0 (no neighbors) get a zero vector.
//
//  Implementation:
//    1. Initialize result to -inf for nodes with neighbors, 0 for d=0 nodes
//    2. Walk CSR row_ptr/col_ind to iterate neighbors
//    3. For each neighbor j, do element-wise max with result[i]
//    4. After processing, replace any remaining -inf with 0 (shouldn't happen
//       if step 1 is correct, but defensive)
//
// ============================================================================
Tensor sage_max_aggregate(const Tensor& A, const Tensor& H) {
    if (A.format() != StorageFormat::SparseCSR) {
        throw std::invalid_argument(
            "sage_max_aggregate: A must be SparseCSR (got Dense).");
    }
    if (H.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            "sage_max_aggregate: H must be Dense (got SparseCSR).");
    }
    if (A.cols() != H.rows()) {
        throw std::invalid_argument(
            "sage_max_aggregate: A.cols() (" + std::to_string(A.cols()) +
            ") != H.rows() (" + std::to_string(H.rows()) + ").");
    }

    const std::size_t N = A.rows();
    const std::size_t F = H.cols();
    const auto& rp      = A.row_ptr();
    const auto& ci      = A.col_ind();
    const float* h_data  = H.data().data();

    Tensor result = Tensor::dense(N, F);   // zero-init
    float* r_data = result.data().data();

    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

    for (std::size_t i = 0; i < N; ++i) {
        const int32_t row_start = rp[i];
        const int32_t row_end   = rp[i + 1];

        if (row_start == row_end) {
            // No neighbors → zero vector (already zero-initialized)
            continue;
        }

        // Initialize row to -inf, then take max over neighbors
        float* ri = r_data + i * F;
        for (std::size_t f = 0; f < F; ++f) {
            ri[f] = NEG_INF;
        }

        for (int32_t nz = row_start; nz < row_end; ++nz) {
            const auto j = static_cast<std::size_t>(ci[nz]);
            const float* hj = h_data + j * F;

            for (std::size_t f = 0; f < F; ++f) {
                ri[f] = std::max(ri[f], hj[f]);
            }
        }
    }

    return result;
}

// ============================================================================
//  SAGELayer — constructor
// ============================================================================
SAGELayer::SAGELayer(std::size_t in_features, std::size_t out_features,
                     Aggregator agg, bool use_bias, Activation act)
    : in_features_(in_features),
      out_features_(out_features),
      aggregator_(agg),
      use_bias_(use_bias),
      activation_(act),
      weight_neigh_(Tensor::dense(in_features, out_features)),   // zero-init
      weight_self_(Tensor::dense(in_features, out_features)),    // zero-init
      bias_(Tensor::dense(1, out_features))                      // zero-init
{
    if (in_features == 0) {
        throw std::invalid_argument(
            "SAGELayer: in_features must be > 0.");
    }
    if (out_features == 0) {
        throw std::invalid_argument(
            "SAGELayer: out_features must be > 0.");
    }
}

// ============================================================================
//  SAGELayer — set_weight_neigh
// ============================================================================
void SAGELayer::set_weight_neigh(Tensor w) {
    if (w.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            "SAGELayer::set_weight_neigh: weight must be Dense.");
    }
    if (w.rows() != in_features_ || w.cols() != out_features_) {
        throw std::invalid_argument(
            "SAGELayer::set_weight_neigh: expected shape (" +
            std::to_string(in_features_) + "×" +
            std::to_string(out_features_) + "), got (" +
            std::to_string(w.rows()) + "×" +
            std::to_string(w.cols()) + ").");
    }
    weight_neigh_ = std::move(w);
}

// ============================================================================
//  SAGELayer — set_weight_self
// ============================================================================
void SAGELayer::set_weight_self(Tensor w) {
    if (w.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            "SAGELayer::set_weight_self: weight must be Dense.");
    }
    if (w.rows() != in_features_ || w.cols() != out_features_) {
        throw std::invalid_argument(
            "SAGELayer::set_weight_self: expected shape (" +
            std::to_string(in_features_) + "×" +
            std::to_string(out_features_) + "), got (" +
            std::to_string(w.rows()) + "×" +
            std::to_string(w.cols()) + ").");
    }
    weight_self_ = std::move(w);
}

// ============================================================================
//  SAGELayer — set_bias
// ============================================================================
void SAGELayer::set_bias(Tensor b) {
    if (!use_bias_) {
        throw std::invalid_argument(
            "SAGELayer::set_bias: layer was constructed with use_bias=false.");
    }
    if (b.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            "SAGELayer::set_bias: bias must be Dense.");
    }
    if (b.rows() != 1 || b.cols() != out_features_) {
        throw std::invalid_argument(
            "SAGELayer::set_bias: expected shape (1×" +
            std::to_string(out_features_) + "), got (" +
            std::to_string(b.rows()) + "×" +
            std::to_string(b.cols()) + ").");
    }
    bias_ = std::move(b);
}

// ============================================================================
//  SAGELayer — forward
// ============================================================================
//
//  h_v' = σ( W_neigh · AGG({h_u : u ∈ N(v)}) + W_self · h_v + b )
//
//  Steps:
//    1. Compute AGG (mean or max aggregation)
//    2. h_neigh = matmul(AGG, W_neigh)      — transform aggregated features
//    3. h_self  = matmul(H, W_self)          — transform self features
//    4. out = h_neigh + h_self               — combine (element-wise add)
//    5. add_bias(out, bias)                  — optional
//    6. activation(out)                      — in-place
//
// ============================================================================
Tensor SAGELayer::forward(const Tensor& A, const Tensor& H) const {
    // ── Validate A ───────────────────────────────────────────────────────────
    if (A.format() != StorageFormat::SparseCSR) {
        throw std::invalid_argument(
            "SAGELayer::forward: A must be SparseCSR (got Dense).");
    }

    // ── Validate H ───────────────────────────────────────────────────────────
    if (H.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            "SAGELayer::forward: H must be Dense (got SparseCSR).");
    }
    if (H.cols() != in_features_) {
        throw std::invalid_argument(
            "SAGELayer::forward: H has " + std::to_string(H.cols()) +
            " columns but layer expects in_features=" +
            std::to_string(in_features_) + ".");
    }
    if (A.cols() != H.rows()) {
        throw std::invalid_argument(
            "SAGELayer::forward: A.cols() (" + std::to_string(A.cols()) +
            ") != H.rows() (" + std::to_string(H.rows()) + ").");
    }

    const std::size_t N = H.rows();

    // ── Step 1: Aggregation ──────────────────────────────────────────────────
    Tensor agg;

    switch (aggregator_) {
        case Aggregator::Mean: {
            // Mean aggregation: spmm(A, H) then divide by degree
            agg = spmm(A, H);   // (N × F_in) — sum of neighbors

            // Compute degree per row and divide
            const auto& rp = A.row_ptr();
            float* agg_d   = agg.data().data();
            const std::size_t F = H.cols();

            for (std::size_t i = 0; i < N; ++i) {
                const float deg = static_cast<float>(rp[i + 1] - rp[i]);
                if (deg > 0.0f) {
                    const float inv_deg = 1.0f / deg;
                    float* row = agg_d + i * F;
                    for (std::size_t f = 0; f < F; ++f) {
                        row[f] *= inv_deg;
                    }
                }
                // deg == 0 → agg[i] stays 0 (from spmm zero-init)
            }
            break;
        }
        case Aggregator::Max: {
            agg = sage_max_aggregate(A, H);
            break;
        }
    }

    // ── Step 2: Transform neighbor aggregation  h_neigh = AGG · W_neigh ─────
    Tensor h_neigh = matmul(agg, weight_neigh_);

    // ── Step 3: Transform self features  h_self = H · W_self ────────────────
    Tensor h_self = matmul(H, weight_self_);

    // ── Step 4: Combine  out = h_neigh + h_self  (element-wise addition) ────
    const std::size_t total = N * out_features_;
    float*       hn = h_neigh.data().data();
    const float* hs = h_self.data().data();
    for (std::size_t idx = 0; idx < total; ++idx) {
        hn[idx] += hs[idx];
    }
    // h_neigh now holds the combined result; use it as `out`
    Tensor& out = h_neigh;

    // ── Step 5: Bias addition ────────────────────────────────────────────────
    if (use_bias_) {
        add_bias(out, bias_);
    }

    // ── Step 6: Activation ───────────────────────────────────────────────────
    switch (activation_) {
        case Activation::ReLU:
            relu_inplace(out);
            break;
        case Activation::None:
            break;
    }

    return out;
}

// ============================================================================
//  edge_softmax — sparse row-wise softmax over CSR values
// ============================================================================
//
//  For each row i, computes softmax over the non-zero entries only:
//    α_ij = exp(v_ij - max_k(v_ik)) / Σ_k exp(v_ik - max_k(v_ik))
//
//  Two-pass algorithm per row:
//    Pass 1: find row-max M_i, compute sum S_i = Σ exp(v - M_i)
//    Pass 2: α = exp(v - M_i) / S_i
//
//  Empty rows (degree 0) are left unchanged (no entries to normalize).
//
// ============================================================================
Tensor edge_softmax(const Tensor& A) {
    if (A.format() != StorageFormat::SparseCSR) {
        throw std::invalid_argument(
            "edge_softmax: input must be SparseCSR (got Dense).");
    }

    const std::size_t N = A.rows();
    const auto& rp     = A.row_ptr();
    const auto& ci     = A.col_ind();
    const auto& vals   = A.data();
    const std::size_t nnz = vals.size();

    // New values for the softmax-normalized CSR
    std::vector<float> new_vals(nnz);

    for (std::size_t i = 0; i < N; ++i) {
        const int32_t row_start = rp[i];
        const int32_t row_end   = rp[i + 1];

        if (row_start == row_end) continue;  // empty row

        // Pass 1a: find max in this row
        float row_max = vals[row_start];
        for (int32_t nz = row_start + 1; nz < row_end; ++nz) {
            row_max = std::max(row_max, vals[nz]);
        }

        // Pass 1b: compute exp(v - max) and sum
        float row_sum = 0.0f;
        for (int32_t nz = row_start; nz < row_end; ++nz) {
            float e = std::exp(vals[nz] - row_max);
            new_vals[nz] = e;
            row_sum += e;
        }

        // Pass 2: normalize
        if (row_sum > 0.0f) {
            const float inv_sum = 1.0f / row_sum;
            for (int32_t nz = row_start; nz < row_end; ++nz) {
                new_vals[nz] *= inv_sum;
            }
        }
    }

    // Build output CSR with same structure, new values
    std::vector<int32_t> rp_copy(rp.begin(), rp.end());
    std::vector<int32_t> ci_copy(ci.begin(), ci.end());

    return Tensor::sparse_csr(N, A.cols(),
                              std::move(rp_copy),
                              std::move(ci_copy),
                              std::move(new_vals));
}

// ============================================================================
//  GATLayer — constructor
// ============================================================================
GATLayer::GATLayer(std::size_t in_features, std::size_t out_features,
                   float negative_slope, bool use_bias, Activation act)
    : in_features_(in_features),
      out_features_(out_features),
      negative_slope_(negative_slope),
      use_bias_(use_bias),
      activation_(act),
      weight_(Tensor::dense(in_features, out_features)),       // zero-init
      attn_left_(Tensor::dense(1, out_features)),              // zero-init
      attn_right_(Tensor::dense(1, out_features)),             // zero-init
      bias_(Tensor::dense(1, out_features))                    // zero-init
{
    if (in_features == 0) {
        throw std::invalid_argument(
            "GATLayer: in_features must be > 0.");
    }
    if (out_features == 0) {
        throw std::invalid_argument(
            "GATLayer: out_features must be > 0.");
    }
}

// ============================================================================
//  GATLayer — set_weight
// ============================================================================
void GATLayer::set_weight(Tensor w) {
    if (w.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            "GATLayer::set_weight: weight must be Dense.");
    }
    if (w.rows() != in_features_ || w.cols() != out_features_) {
        throw std::invalid_argument(
            "GATLayer::set_weight: expected shape (" +
            std::to_string(in_features_) + "×" +
            std::to_string(out_features_) + "), got (" +
            std::to_string(w.rows()) + "×" +
            std::to_string(w.cols()) + ").");
    }
    weight_ = std::move(w);
}

// ============================================================================
//  GATLayer — set_attn_left
// ============================================================================
void GATLayer::set_attn_left(Tensor a) {
    if (a.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            "GATLayer::set_attn_left: attention vector must be Dense.");
    }
    if (a.rows() != 1 || a.cols() != out_features_) {
        throw std::invalid_argument(
            "GATLayer::set_attn_left: expected shape (1×" +
            std::to_string(out_features_) + "), got (" +
            std::to_string(a.rows()) + "×" +
            std::to_string(a.cols()) + ").");
    }
    attn_left_ = std::move(a);
}

// ============================================================================
//  GATLayer — set_attn_right
// ============================================================================
void GATLayer::set_attn_right(Tensor a) {
    if (a.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            "GATLayer::set_attn_right: attention vector must be Dense.");
    }
    if (a.rows() != 1 || a.cols() != out_features_) {
        throw std::invalid_argument(
            "GATLayer::set_attn_right: expected shape (1×" +
            std::to_string(out_features_) + "), got (" +
            std::to_string(a.rows()) + "×" +
            std::to_string(a.cols()) + ").");
    }
    attn_right_ = std::move(a);
}

// ============================================================================
//  GATLayer — set_bias
// ============================================================================
void GATLayer::set_bias(Tensor b) {
    if (!use_bias_) {
        throw std::invalid_argument(
            "GATLayer::set_bias: layer was constructed with use_bias=false.");
    }
    if (b.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            "GATLayer::set_bias: bias must be Dense.");
    }
    if (b.rows() != 1 || b.cols() != out_features_) {
        throw std::invalid_argument(
            "GATLayer::set_bias: expected shape (1×" +
            std::to_string(out_features_) + "), got (" +
            std::to_string(b.rows()) + "×" +
            std::to_string(b.cols()) + ").");
    }
    bias_ = std::move(b);
}

// ============================================================================
//  GATLayer — forward
// ============================================================================
//
//  Complete GAT forward pass (single attention head):
//
//    1. Wh = matmul(H, W)                        [N × F_out]
//    2. src_scores[i] = Σ_f  a_l[f] * Wh[i][f]   (dot product per node)
//       dst_scores[j] = Σ_f  a_r[f] * Wh[j][f]
//    3. For each edge (i,j) in CSR:
//         e_ij = LeakyReLU(src_scores[i] + dst_scores[j])
//       → Build a CSR with these attention logits as values  (SpSDDMM)
//    4. α = edge_softmax(e_csr)                   (sparse softmax per row)
//    5. out = spmm(α, Wh) + b                     (attention-weighted aggregation)
//    6. activation(out)
//
// ============================================================================
Tensor GATLayer::forward(const Tensor& A, const Tensor& H) const {
    // ── Validate A ───────────────────────────────────────────────────────────
    if (A.format() != StorageFormat::SparseCSR) {
        throw std::invalid_argument(
            "GATLayer::forward: A must be SparseCSR (got Dense). "
            "Use add_self_loops() to prepare the adjacency.");
    }
    if (A.rows() != A.cols()) {
        throw std::invalid_argument(
            "GATLayer::forward: A must be square — got (" +
            std::to_string(A.rows()) + "×" + std::to_string(A.cols()) + ").");
    }

    // ── Validate H ───────────────────────────────────────────────────────────
    if (H.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            "GATLayer::forward: H must be Dense (got SparseCSR).");
    }
    if (H.cols() != in_features_) {
        throw std::invalid_argument(
            "GATLayer::forward: H has " + std::to_string(H.cols()) +
            " columns but layer expects in_features=" +
            std::to_string(in_features_) + ".");
    }
    if (A.rows() != H.rows()) {
        throw std::invalid_argument(
            "GATLayer::forward: A has " +
            std::to_string(A.rows()) + " rows but H has " +
            std::to_string(H.rows()) + " rows. They must match (N nodes).");
    }

    const std::size_t N = H.rows();
    const std::size_t F_out = out_features_;
    const auto& rp = A.row_ptr();
    const auto& ci = A.col_ind();

    // ── Step 1: Linear transform  Wh = H · W ────────────────────────────────
    Tensor Wh = matmul(H, weight_);   // (N × F_out)

    // ── Step 2: Compute per-node attention scores (SpSDDMM preparation) ──────
    //   src_scores[i] = a_l^T · Wh[i]   (source contribution)
    //   dst_scores[j] = a_r^T · Wh[j]   (target contribution)
    const float* wh_data  = Wh.data().data();
    const float* al_data  = attn_left_.data().data();
    const float* ar_data  = attn_right_.data().data();

    std::vector<float> src_scores(N, 0.0f);
    std::vector<float> dst_scores(N, 0.0f);

    for (std::size_t i = 0; i < N; ++i) {
        float s = 0.0f, d = 0.0f;
        const float* whi = wh_data + i * F_out;
        for (std::size_t f = 0; f < F_out; ++f) {
            s += al_data[f] * whi[f];
            d += ar_data[f] * whi[f];
        }
        src_scores[i] = s;
        dst_scores[i] = d;
    }

    // ── Step 3: SpSDDMM — compute edge attention logits ─────────────────────
    //   For each edge (i, j):  e_ij = LeakyReLU(src_scores[i] + dst_scores[j])
    const std::size_t nnz = ci.size();
    std::vector<float> edge_logits(nnz);

    for (std::size_t i = 0; i < N; ++i) {
        const float si = src_scores[i];
        for (int32_t nz = rp[i]; nz < rp[i + 1]; ++nz) {
            const auto j = static_cast<std::size_t>(ci[nz]);
            float e = si + dst_scores[j];
            // LeakyReLU
            edge_logits[nz] = (e >= 0.0f) ? e : negative_slope_ * e;
        }
    }

    // Build CSR with attention logits as values
    std::vector<int32_t> rp_copy(rp.begin(), rp.end());
    std::vector<int32_t> ci_copy(ci.begin(), ci.end());
    Tensor attn_csr = Tensor::sparse_csr(N, N,
                                         std::move(rp_copy),
                                         std::move(ci_copy),
                                         std::move(edge_logits));

    // ── Step 4: Sparse Softmax — normalize attention per neighborhood ────────
    Tensor alpha = edge_softmax(attn_csr);

    // ── Step 5: Attention-weighted message passing  out = spmm(α, Wh) ────────
    Tensor out = spmm(alpha, Wh);

    // Bias addition
    if (use_bias_) {
        add_bias(out, bias_);
    }

    // ── Step 6: Activation ───────────────────────────────────────────────────
    switch (activation_) {
        case Activation::ReLU:
            relu_inplace(out);
            break;
        case Activation::None:
            break;
    }

    return out;
}

}  // namespace tinygnn
