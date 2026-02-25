#pragma once

// ============================================================================
//  TinyGNN — Model & Weight Loading  (Phase 7)
//  include/tinygnn/model.hpp
//
//  This header declares:
//
//    CoraData          — Holds Cora graph data loaded from binary export
//    load_cora_binary  — Loads cora_graph.bin (features, labels, masks, edges)
//    load_weight_file  — Loads TGNN binary weight file into named tensor map
//    Model             — Dynamic execution graph that chains GNN layers
//
//  The Model class supports heterogeneous layer sequences:
//    • GCN layers    — use gcn_norm(A) adjacency preprocessing
//    • SAGE layers   — use raw adjacency (no self-loops)
//    • GAT layers    — use add_self_loops(A), supports multi-head concat/avg
//
//  Inter-layer activations (ELU, ReLU) applied outside the layer's own
//  activation, enabling patterns like GAT → ELU → next layer.
//
//  Binary weight file format:
//    Magic  "TGNN"  (4 bytes)
//    Version        uint32_le = 1
//    TestAccuracy   float32_le
//    NumTensors     uint32_le
//    ── per tensor ──
//    NameLen  uint32_le
//    Name     char[NameLen]
//    Rows     uint32_le
//    Cols     uint32_le
//    Data     float32_le[ Rows × Cols ]
// ============================================================================

#include "tinygnn/layers.hpp"
#include "tinygnn/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tinygnn {

// ============================================================================
//  CoraData — graph struct loaded from binary export
// ============================================================================
struct CoraData {
    std::size_t num_nodes    = 0;
    std::size_t num_features = 0;
    std::size_t num_classes  = 0;
    std::size_t num_edges    = 0;

    Tensor adjacency;          // SparseCSR  (N × N), values 1.0
    Tensor features;           // Dense      (N × F)

    std::vector<int32_t> labels;       // N int32
    std::vector<uint8_t> train_mask;   // N uint8 (0/1)
    std::vector<uint8_t> val_mask;
    std::vector<uint8_t> test_mask;
};

/// Load Cora graph data from the binary file exported by train_cora.py.
/// Builds a SparseCSR adjacency matrix from the directed edge list.
/// @param path  Path to cora_graph.bin
/// @throws std::runtime_error on I/O or format errors
CoraData load_cora_binary(const std::string& path);

// ============================================================================
//  Weight file loader
// ============================================================================

/// Result of loading a weight file: test accuracy + named tensors.
struct WeightFile {
    float test_accuracy = 0.0f;
    std::unordered_map<std::string, Tensor> tensors;
};

/// Load named tensors from a TGNN binary weight file.
/// @param path  Path to a .bin weight file
/// @throws std::runtime_error on I/O, magic, or version mismatch
WeightFile load_weight_file(const std::string& path);

// ============================================================================
//  Model — Dynamic GNN Execution Graph
// ============================================================================
class Model {
public:
    /// Activation applied AFTER a layer's output (outside the layer).
    enum class InterActivation : uint8_t {
        None = 0,
        ReLU = 1,
        ELU  = 2,
    };

    // ── Layer addition ──────────────────────────────────────────────────

    /// Add a GCN layer to the execution graph.
    /// @return Layer index (0-based)
    std::size_t add_gcn_layer(std::size_t in_f, std::size_t out_f,
                              bool bias = true,
                              Activation act = Activation::ReLU,
                              InterActivation post = InterActivation::None);

    /// Add a GraphSAGE layer.
    std::size_t add_sage_layer(std::size_t in_f, std::size_t out_f,
                               SAGELayer::Aggregator agg = SAGELayer::Aggregator::Mean,
                               bool bias = true,
                               Activation act = Activation::ReLU,
                               InterActivation post = InterActivation::None);

    /// Add a (possibly multi-head) GAT layer.
    /// Multi-head: creates `num_heads` independent GATLayer instances.
    /// @param concat  true = concatenate heads (out = num_heads × out_f),
    ///                false = average heads (out = out_f)
    std::size_t add_gat_layer(std::size_t in_f, std::size_t out_f,
                              std::size_t num_heads = 1,
                              bool concat = true,
                              float neg_slope = 0.2f,
                              bool bias = true,
                              Activation act = Activation::None,
                              InterActivation post = InterActivation::None);

    /// Number of logical layers in the execution graph.
    std::size_t num_layers() const noexcept { return layers_.size(); }

    // ── Weight management ───────────────────────────────────────────────

    /// Load weights from a TGNN binary file, mapping tensor names to layers.
    /// Naming convention:
    ///   GCN:  layer{i}.weight, layer{i}.bias
    ///   SAGE: layer{i}.weight_neigh, layer{i}.weight_self, layer{i}.bias
    ///   GAT:  layer{i}.head{j}.weight, .attn_left, .attn_right, .bias
    void load_weights(const std::string& path);

    /// Load weights from a pre-loaded WeightFile.
    void load_weights(const WeightFile& wf);

    // ── Forward pass ────────────────────────────────────────────────────

    /// Run the full execution graph.  Handles adjacency preprocessing
    /// (gcn_norm, add_self_loops) internally based on layer types.
    /// @param adjacency  Raw SparseCSR adjacency (N×N), NO self-loops
    /// @param features   Dense node features (N×F)
    /// @return           Dense output (N×out_features_of_last_layer)
    Tensor forward(const Tensor& adjacency, const Tensor& features) const;

private:
    struct LayerEntry {
        enum class Type : uint8_t { GCN, SAGE, GAT };
        Type          type;
        std::size_t   start_idx;    // index into gcn/sage/gat vector
        std::size_t   count;        // 1 for GCN/SAGE, num_heads for GAT
        bool          gat_concat;   // GAT only: concat vs. average
        InterActivation post_act;
    };

    std::vector<LayerEntry>  layers_;
    std::vector<GCNLayer>    gcn_layers_;
    std::vector<SAGELayer>   sage_layers_;
    std::vector<GATLayer>    gat_layers_;
};

}  // namespace tinygnn
