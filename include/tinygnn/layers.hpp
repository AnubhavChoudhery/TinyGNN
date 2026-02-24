#pragma once

// ============================================================================
//  TinyGNN — GNN Layer Implementations  (Phase 6)
//  include/tinygnn/layers.hpp
//
//  This header declares the graph neural network layer types that compose
//  the low-level ops (matmul, spmm, activations) from Phases 3–5 into
//  complete GNN message-passing layers:
//
//    Graph Normalization Utilities:
//      add_self_loops(A)         — Ã = A + I  (sparse adjacency + identity)
//      gcn_norm(A)               — D̃^{-1/2} Ã D̃^{-1/2}  (symmetric norm)
//      edge_softmax(A)           — row-wise (per-node) softmax over sparse values
//
//    Layer Implementations:
//      GCNLayer                  — Graph Convolutional Network (Kipf & Welling, ICLR 2017)
//      SAGELayer                 — GraphSAGE (Hamilton et al., NeurIPS 2017)
//      GATLayer                  — Graph Attention Network (Veličković et al., ICLR 2018)
//
//  Design principles:
//    • Inference-only — weights can be set explicitly (no autograd/training).
//    • Reuses existing ops: matmul, spmm, add_bias, relu_inplace.
//    • GCNLayer.forward() expects a pre-normalized adjacency (output of
//      gcn_norm) so normalization is a one-time preprocessing step.
//    • SAGELayer supports Mean and Max aggregation.  The adjacency should
//      NOT include self-loops; self-features are handled by a separate
//      weight matrix (W_self).
//    • GATLayer computes attention scores per edge using SpSDDMM, applies
//      LeakyReLU and Sparse Softmax over each node's neighbourhood, then
//      performs attention-weighted message passing.
//    • All functions operate in single precision (float32).
//    • Zero external dependencies.
//    • Clear, descriptive exceptions for every invalid input.
//
//  GCN forward pass (one layer):
//    H' = σ( Â_norm · (H · W) + b )
//    where Â_norm = gcn_norm(A) = D̃^{-1/2} (A + I) D̃^{-1/2}
//
//    Implementation order (follows PyG convention):
//      1. Linear transform:    HW = matmul(H, W)       [N × F_in → N × F_out]
//      2. Message passing:     out = spmm(Â_norm, HW)  [aggregate neighbors]
//      3. Bias addition:       out += b                 [optional]
//      4. Activation:          σ(out)                   [in-place]
//
//  GraphSAGE forward pass (one layer):
//    h_v' = σ( W_neigh · AGG({h_u : u ∈ N(v)}) + W_self · h_v + b )
//
//    Mean aggregation:
//      AGG_mean[i] = (1 / deg[i]) · Σ_{j ∈ N(i)} h_j
//
//    Max aggregation:
//      AGG_max[i][f] = max_{j ∈ N(i)} h_j[f]   (element-wise max)
//
//  GAT forward pass (single-head, one layer):
//    1. Linear transform:  Wh = H · W                (N × F_out)
//    2. Attention scores:  e_ij = LeakyReLU(a_l^T · Wh_i + a_r^T · Wh_j)
//       Computed via SpSDDMM: for each edge (i,j) in CSR, combine the
//       left/right attention vectors with the transformed features.
//    3. Sparse Softmax:    α_ij = softmax_j(e_ij) over N(i)
//       Computed per row using CSR row_ptr boundaries, with max-subtraction
//       for numerical stability.
//    4. Message passing:   h_i' = σ( Σ_j α_ij · Wh_j + b )
//       Implemented as spmm(α_sparse, Wh) where α_sparse is the CSR
//       adjacency with attention-weighted values.
//
//  References:
//    [1] Kipf & Welling, "Semi-Supervised Classification with Graph
//        Convolutional Networks", ICLR 2017.
//    [2] Hamilton, Ying & Leskovec, "Inductive Representation Learning
//        on Large Graphs", NeurIPS 2017.
//    [3] Veličković, Cucurull, Casanova, Romero, Liò & Bengio,
//        "Graph Attention Networks", ICLR 2018.
// ============================================================================

#include "tinygnn/tensor.hpp"

#include <cstddef>
#include <cstdint>

namespace tinygnn {

// ============================================================================
//                   Activation Selection
// ============================================================================
//  Enum for selecting the non-linearity applied after each layer's linear
//  transform + aggregation.  Kept minimal for GCN/SAGE; users needing
//  other activations can call the in-place functions from ops.hpp directly
//  on the layer output.
// ============================================================================
enum class Activation : uint8_t {
    None = 0,   // Identity — no activation
    ReLU = 1,   // ReLU: max(0, x)
};

// ============================================================================
//             Graph Normalization Utilities
// ============================================================================

// ============================================================================
//  add_self_loops — Add identity matrix to sparse adjacency
// ============================================================================
//  Computes Ã = A + I  where:
//    A  must be a square SparseCSR tensor of shape (N, N)
//    I  is the N×N identity matrix
//
//  For each row i:
//    • If A[i][i] already exists with value v, it becomes v + 1.0
//    • If A[i][i] does not exist, a new entry with value 1.0 is inserted
//
//  The output CSR maintains sorted column indices within each row.
//
//  Complexity: O(nnz + N)
//  Memory:     O(nnz + N)      — new CSR with up to N additional entries
//
//  @param A  Square SparseCSR adjacency matrix (N × N)
//  @return   New SparseCSR tensor Ã = A + I
//  @throws std::invalid_argument  if A is not SparseCSR
//  @throws std::invalid_argument  if A is not square
// ============================================================================
Tensor add_self_loops(const Tensor& A);

// ============================================================================
//  gcn_norm — Symmetric GCN normalization
// ============================================================================
//  Computes D̃^{-1/2} · Ã · D̃^{-1/2}  where:
//    Ã = A + I           (adjacency with self-loops)
//    D̃ = diag(d̃)       with d̃[i] = Σ_j Ã[i][j]  (degree of Ã)
//
//  For each non-zero entry Ã[i][j]:
//    Â[i][j] = Ã[i][j] / (√d̃[i] · √d̃[j])
//
//  This is the standard preprocessing for GCN (Kipf & Welling, 2017).
//  The normalized matrix should be computed once and reused across all
//  forward passes and all layers (it depends only on the graph topology).
//
//  Complexity: O(nnz + N)      — two passes over all non-zeros + degree comp
//  Memory:     O(nnz + N)      — new CSR tensor
//
//  @param A  Square SparseCSR adjacency matrix (N × N), without self-loops
//  @return   New SparseCSR tensor Â = D̃^{-1/2} (A + I) D̃^{-1/2}
//  @throws std::invalid_argument  if A is not SparseCSR
//  @throws std::invalid_argument  if A is not square
// ============================================================================
Tensor gcn_norm(const Tensor& A);

// ============================================================================
//                      GCN Layer
// ============================================================================
//  Graph Convolutional Network layer (Kipf & Welling, ICLR 2017).
//
//  Forward pass:
//    H' = σ( Â_norm · (H · W) + b )
//
//  Where:
//    Â_norm  — pre-normalized adjacency matrix (output of gcn_norm)
//    H       — input node features      (N × in_features)
//    W       — learnable weight matrix  (in_features × out_features)
//    b       — learnable bias vector    (1 × out_features), optional
//    σ       — activation function (ReLU or None)
//
//  The transform-then-propagate order (matmul first, then spmm) follows
//  the PyG convention and is optimal when out_features ≤ in_features.
//
//  Usage:
//    Tensor A_norm = gcn_norm(adjacency);   // precompute once
//    GCNLayer layer1(16, 32);               // 16 → 32 features
//    layer1.set_weight(my_W);               // set from PyTorch checkpoint
//    layer1.set_bias(my_b);
//    Tensor out = layer1.forward(A_norm, H);
// ============================================================================
struct GCNLayer {
    // ── Construction ────────────────────────────────────────────────────────
    /// @param in_features   Number of input features per node
    /// @param out_features  Number of output features per node
    /// @param use_bias      Whether to add a bias vector (default: true)
    /// @param act           Activation function (default: ReLU)
    GCNLayer(std::size_t in_features, std::size_t out_features,
             bool use_bias = true, Activation act = Activation::ReLU);

    // ── Weight management ───────────────────────────────────────────────────
    /// Set weight matrix W (must be in_features × out_features, Dense)
    void set_weight(Tensor w);

    /// Set bias vector b (must be 1 × out_features, Dense)
    /// @throws std::invalid_argument  if use_bias is false
    void set_bias(Tensor b);

    // ── Forward pass ────────────────────────────────────────────────────────
    /// Compute H' = σ( A_norm · (H · W) + b )
    /// @param A_norm  Pre-normalized adjacency (SparseCSR, N×N)
    /// @param H       Node feature matrix (Dense, N × in_features)
    /// @return        Output features (Dense, N × out_features)
    /// @throws std::invalid_argument  on format/dimension mismatches
    Tensor forward(const Tensor& A_norm, const Tensor& H) const;

    // ── Observers ───────────────────────────────────────────────────────────
    std::size_t in_features()  const noexcept { return in_features_; }
    std::size_t out_features() const noexcept { return out_features_; }
    bool        has_bias()     const noexcept { return use_bias_; }
    Activation  activation()   const noexcept { return activation_; }

    const Tensor& weight() const noexcept { return weight_; }
    const Tensor& bias()   const noexcept { return bias_; }

private:
    std::size_t in_features_;
    std::size_t out_features_;
    bool        use_bias_;
    Activation  activation_;

    Tensor weight_;   // Dense (in_features × out_features)
    Tensor bias_;     // Dense (1 × out_features)  — zero if !use_bias_
};

// ============================================================================
//                      GraphSAGE Layer
// ============================================================================
//  GraphSAGE layer (Hamilton, Ying & Leskovec, NeurIPS 2017).
//
//  Forward pass:
//    h_v' = σ( W_neigh · AGG({h_u : u ∈ N(v)}) + W_self · h_v + b )
//
//  Where:
//    A       — adjacency matrix WITHOUT self-loops (N × N, SparseCSR)
//    H       — input node features (N × in_features)
//    W_neigh — weight for aggregated neighbor features (in_features × out_features)
//    W_self  — weight for self features               (in_features × out_features)
//    b       — bias vector (1 × out_features), optional
//    AGG     — Mean or Max aggregation
//
//  Mean aggregation:
//    AGG_mean[i] = spmm(A, H)[i] / deg[i]
//    where deg[i] = number of neighbors of node i (row_ptr[i+1] - row_ptr[i])
//
//  Max aggregation:
//    AGG_max[i][f] = max_{j ∈ N(i)} H[j][f]
//    Element-wise max over neighbor features.  Nodes with no neighbors
//    get a zero vector.
//
//  Usage:
//    SAGELayer layer(16, 32, SAGELayer::Aggregator::Mean);
//    layer.set_weight_neigh(W_n);
//    layer.set_weight_self(W_s);
//    layer.set_bias(b);
//    Tensor out = layer.forward(adjacency, H);
// ============================================================================
struct SAGELayer {
    // ── Aggregation type ────────────────────────────────────────────────────
    enum class Aggregator : uint8_t {
        Mean = 0,   // Mean of neighbor features (normalized by degree)
        Max  = 1,   // Element-wise max of neighbor features
    };

    // ── Construction ────────────────────────────────────────────────────────
    /// @param in_features   Number of input features per node
    /// @param out_features  Number of output features per node
    /// @param agg           Aggregation type (default: Mean)
    /// @param use_bias      Whether to add a bias vector (default: true)
    /// @param act           Activation function (default: ReLU)
    SAGELayer(std::size_t in_features, std::size_t out_features,
              Aggregator agg = Aggregator::Mean,
              bool use_bias = true, Activation act = Activation::ReLU);

    // ── Weight management ───────────────────────────────────────────────────
    /// Set neighbor weight matrix (must be in_features × out_features, Dense)
    void set_weight_neigh(Tensor w);

    /// Set self weight matrix (must be in_features × out_features, Dense)
    void set_weight_self(Tensor w);

    /// Set bias vector (must be 1 × out_features, Dense)
    /// @throws std::invalid_argument  if use_bias is false
    void set_bias(Tensor b);

    // ── Forward pass ────────────────────────────────────────────────────────
    /// Compute h_v' = σ( W_neigh · AGG(neighbors) + W_self · h_v + b )
    /// @param A  Adjacency matrix WITHOUT self-loops (SparseCSR, N×N)
    /// @param H  Node feature matrix (Dense, N × in_features)
    /// @return   Output features (Dense, N × out_features)
    /// @throws std::invalid_argument  on format/dimension mismatches
    Tensor forward(const Tensor& A, const Tensor& H) const;

    // ── Observers ───────────────────────────────────────────────────────────
    std::size_t in_features()  const noexcept { return in_features_; }
    std::size_t out_features() const noexcept { return out_features_; }
    Aggregator  aggregator()   const noexcept { return aggregator_; }
    bool        has_bias()     const noexcept { return use_bias_; }
    Activation  activation()   const noexcept { return activation_; }

    const Tensor& weight_neigh() const noexcept { return weight_neigh_; }
    const Tensor& weight_self()  const noexcept { return weight_self_; }
    const Tensor& bias()         const noexcept { return bias_; }

private:
    std::size_t in_features_;
    std::size_t out_features_;
    Aggregator  aggregator_;
    bool        use_bias_;
    Activation  activation_;

    Tensor weight_neigh_;  // Dense (in_features × out_features)
    Tensor weight_self_;   // Dense (in_features × out_features)
    Tensor bias_;          // Dense (1 × out_features)
};

// ============================================================================
//  sage_max_aggregate — Element-wise max pooling over neighbor features
// ============================================================================
//  For each node i:
//    result[i][f] = max_{j ∈ N(i)} H[j][f]
//
//  Nodes with no neighbors (degree 0) receive a zero vector.
//
//  This is an internal utility for SAGELayer::forward() with Max aggregation,
//  but is also exposed publicly for testing and composition.
//
//  @param A  Adjacency matrix (SparseCSR, N × N) — defines neighborhood
//  @param H  Node feature matrix (Dense, N × F)
//  @return   Dense tensor (N × F) with max-pooled neighbor features
//  @throws std::invalid_argument  if A is not SparseCSR or H is not Dense
//  @throws std::invalid_argument  if A.cols() != H.rows()
// ============================================================================
Tensor sage_max_aggregate(const Tensor& A, const Tensor& H);

// ============================================================================
//  edge_softmax — Sparse row-wise softmax over CSR values
// ============================================================================
//  Computes softmax over each row of a SparseCSR tensor independently.
//  For each row i with non-zero entries j₁, j₂, …, jₖ:
//    α_ij = exp(A[i][j] - max_k(A[i][k])) / Σ_k exp(A[i][k] - max_k)
//
//  This is the "Sparse Softmax" required by GAT: it operates only on the
//  non-zero entries of each row (the node's neighborhood), not on all N
//  columns.  The max-subtraction trick ensures numerical stability.
//
//  Rows with no non-zeros (isolated nodes) are left empty.
//
//  Returns a NEW SparseCSR tensor with same structure (row_ptr, col_ind)
//  but softmax-normalized values.
//
//  Complexity: O(nnz)       — two passes over non-zeros (max+sum, normalize)
//  Memory:     O(nnz + N)   — new value array + per-row max/sum temporaries
//
//  @param A  SparseCSR tensor (values are raw attention scores / logits)
//  @return   New SparseCSR tensor with softmax-normalized values per row
//  @throws std::invalid_argument  if A is not SparseCSR
// ============================================================================
Tensor edge_softmax(const Tensor& A);

// ============================================================================
//                      GAT Layer
// ============================================================================
//  Graph Attention Network layer (Veličković et al., ICLR 2018).
//  Single-head attention (multi-head can be composed externally).
//
//  Forward pass:
//    1.  Wh = H · W                               (N × F_out)  linear transform
//    2.  e_ij = LeakyReLU(a_l · Wh_i + a_r · Wh_j) for each edge (i,j)
//        — SpSDDMM: compute source+target attention per edge in CSR
//    3.  α_ij = edge_softmax(e)                     sparse softmax per row
//    4.  out = spmm(α, Wh) + b                       attention-weighted aggregation
//    5.  σ(out)                                       optional activation
//
//  Weights:
//    W       — (in_features × out_features) linear projection
//    a_left  — (out_features) attention vector for source node
//    a_right — (out_features) attention vector for target node
//    bias    — (1 × out_features) optional bias
//
//  The adjacency matrix A should include self-loops if you want the node's
//  own features to participate in the attention mechanism (standard GAT
//  convention).  Use add_self_loops(A) before calling forward().
//
//  Usage:
//    Tensor A_sl = add_self_loops(adjacency);  // add self-loops once
//    GATLayer layer(16, 8);                    // 16 → 8 features
//    layer.set_weight(my_W);
//    layer.set_attn_left(my_a_l);
//    layer.set_attn_right(my_a_r);
//    layer.set_bias(my_b);
//    Tensor out = layer.forward(A_sl, H);
// ============================================================================
struct GATLayer {
    // ── Construction ────────────────────────────────────────────────────────
    /// @param in_features   Number of input features per node
    /// @param out_features  Number of output features per node
    /// @param negative_slope  LeakyReLU negative slope (default: 0.2, paper value)
    /// @param use_bias      Whether to add a bias vector (default: true)
    /// @param act           Activation function (default: None; GAT paper uses ELU
    ///                      but for testing simplicity we default to None)
    GATLayer(std::size_t in_features, std::size_t out_features,
             float negative_slope = 0.2f,
             bool use_bias = true, Activation act = Activation::None);

    // ── Weight management ───────────────────────────────────────────────────
    /// Set weight matrix W (must be in_features × out_features, Dense)
    void set_weight(Tensor w);

    /// Set left attention vector a_l (must be 1 × out_features, Dense)
    /// a_l is applied to the source node's transformed features:  a_l^T · Wh_i
    void set_attn_left(Tensor a);

    /// Set right attention vector a_r (must be 1 × out_features, Dense)
    /// a_r is applied to the target node's transformed features:  a_r^T · Wh_j
    void set_attn_right(Tensor a);

    /// Set bias vector b (must be 1 × out_features, Dense)
    /// @throws std::invalid_argument  if use_bias is false
    void set_bias(Tensor b);

    // ── Forward pass ────────────────────────────────────────────────────────
    /// Compute attention-weighted message passing.
    /// @param A  Adjacency matrix WITH self-loops (SparseCSR, N×N)
    /// @param H  Node feature matrix (Dense, N × in_features)
    /// @return   Output features (Dense, N × out_features)
    /// @throws std::invalid_argument  on format/dimension mismatches
    Tensor forward(const Tensor& A, const Tensor& H) const;

    // ── Observers ───────────────────────────────────────────────────────────
    std::size_t in_features()    const noexcept { return in_features_; }
    std::size_t out_features()   const noexcept { return out_features_; }
    float       negative_slope() const noexcept { return negative_slope_; }
    bool        has_bias()       const noexcept { return use_bias_; }
    Activation  activation()     const noexcept { return activation_; }

    const Tensor& weight()     const noexcept { return weight_; }
    const Tensor& attn_left()  const noexcept { return attn_left_; }
    const Tensor& attn_right() const noexcept { return attn_right_; }
    const Tensor& bias()       const noexcept { return bias_; }

private:
    std::size_t in_features_;
    std::size_t out_features_;
    float       negative_slope_;   // LeakyReLU α (typically 0.2)
    bool        use_bias_;
    Activation  activation_;

    Tensor weight_;      // Dense (in_features × out_features)
    Tensor attn_left_;   // Dense (1 × out_features) — source attention
    Tensor attn_right_;  // Dense (1 × out_features) — target attention
    Tensor bias_;        // Dense (1 × out_features)
};

}  // namespace tinygnn
