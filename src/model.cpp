// ============================================================================
//  TinyGNN — Model & Weight Loading  (Phase 7)
//  src/model.cpp
//
//  Implements:
//    • load_cora_binary  — Binary graph loader
//    • load_weight_file  — TGNN binary weight file reader
//    • Model             — Dynamic GNN execution graph
// ============================================================================

#include "tinygnn/model.hpp"
#include "tinygnn/ops.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tinygnn {

// ============================================================================
//  Helper — read exact bytes from a binary stream
// ============================================================================
static void read_exact(std::ifstream& f, void* buf, std::size_t n,
                       const char* context) {
    f.read(reinterpret_cast<char*>(buf), static_cast<std::streamsize>(n));
    if (!f) {
        throw std::runtime_error(
            std::string(context) + ": unexpected end of file or read error.");
    }
}

template <typename T>
static T read_le(std::ifstream& f, const char* ctx) {
    T val{};
    read_exact(f, &val, sizeof(T), ctx);
    return val;   // assumes little-endian host (x86/x64)
}

// ============================================================================
//  load_cora_binary — Load Cora graph from binary export
// ============================================================================
CoraData load_cora_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("load_cora_binary: cannot open " + path);
    }

    const char* ctx = "load_cora_binary";

    CoraData cd;
    cd.num_nodes    = read_le<uint32_t>(f, ctx);
    cd.num_features = read_le<uint32_t>(f, ctx);
    cd.num_classes  = read_le<uint32_t>(f, ctx);
    cd.num_edges    = read_le<uint32_t>(f, ctx);

    const std::size_t N = cd.num_nodes;
    const std::size_t F = cd.num_features;
    const std::size_t E = cd.num_edges;

    // ── Features: N × F float32 ──────────────────────────────────────────
    std::vector<float> feat_data(N * F);
    read_exact(f, feat_data.data(), N * F * sizeof(float), ctx);
    cd.features = Tensor::dense(N, F, std::move(feat_data));

    // ── Labels: N × int32 ────────────────────────────────────────────────
    cd.labels.resize(N);
    read_exact(f, cd.labels.data(), N * sizeof(int32_t), ctx);

    // ── Masks: N × uint8 each ────────────────────────────────────────────
    cd.train_mask.resize(N);
    cd.val_mask.resize(N);
    cd.test_mask.resize(N);
    read_exact(f, cd.train_mask.data(), N, ctx);
    read_exact(f, cd.val_mask.data(), N, ctx);
    read_exact(f, cd.test_mask.data(), N, ctx);

    // ── Edges: E × int32 (src) then E × int32 (dst) ─────────────────────
    std::vector<int32_t> edge_src(E), edge_dst(E);
    read_exact(f, edge_src.data(), E * sizeof(int32_t), ctx);
    read_exact(f, edge_dst.data(), E * sizeof(int32_t), ctx);

    // Build CSR adjacency matrix
    // Count per-row degree
    std::vector<int32_t> row_count(N, 0);
    for (std::size_t e = 0; e < E; ++e) {
        const auto src = static_cast<std::size_t>(edge_src[e]);
        if (src < N) ++row_count[src];
    }

    // Build row_ptr via prefix sum
    std::vector<int32_t> row_ptr(N + 1, 0);
    for (std::size_t i = 0; i < N; ++i) {
        row_ptr[i + 1] = row_ptr[i] + row_count[i];
    }

    // Fill col_ind
    const int32_t nnz = row_ptr[N];
    std::vector<int32_t> col_ind(static_cast<std::size_t>(nnz));
    std::vector<float>   values(static_cast<std::size_t>(nnz), 1.0f);
    std::vector<int32_t> offset(N, 0);

    for (std::size_t e = 0; e < E; ++e) {
        const auto src = static_cast<std::size_t>(edge_src[e]);
        const int32_t pos = row_ptr[src] + offset[src];
        col_ind[static_cast<std::size_t>(pos)] = edge_dst[e];
        ++offset[src];
    }

    // Sort col_ind within each row
    for (std::size_t i = 0; i < N; ++i) {
        const auto start = static_cast<std::size_t>(row_ptr[i]);
        const auto end   = static_cast<std::size_t>(row_ptr[i + 1]);
        std::sort(col_ind.begin() + static_cast<std::ptrdiff_t>(start),
                  col_ind.begin() + static_cast<std::ptrdiff_t>(end));
    }

    cd.adjacency = Tensor::sparse_csr(N, N,
                                      std::move(row_ptr),
                                      std::move(col_ind),
                                      std::move(values));
    return cd;
}

// ============================================================================
//  load_weight_file — Load TGNN binary weight file
// ============================================================================
WeightFile load_weight_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("load_weight_file: cannot open " + path);
    }

    const char* ctx = "load_weight_file";

    // ── Magic ────────────────────────────────────────────────────────────
    char magic[4]{};
    read_exact(f, magic, 4, ctx);
    if (std::memcmp(magic, "TGNN", 4) != 0) {
        throw std::runtime_error(
            "load_weight_file: invalid magic in " + path);
    }

    // ── Version ──────────────────────────────────────────────────────────
    const uint32_t version = read_le<uint32_t>(f, ctx);
    if (version != 1) {
        throw std::runtime_error(
            "load_weight_file: unsupported version " +
            std::to_string(version) + " in " + path);
    }

    // ── Test accuracy ────────────────────────────────────────────────────
    WeightFile wf;
    wf.test_accuracy = read_le<float>(f, ctx);

    // ── Tensors ──────────────────────────────────────────────────────────
    const uint32_t num_tensors = read_le<uint32_t>(f, ctx);

    for (uint32_t t = 0; t < num_tensors; ++t) {
        // Name
        const uint32_t name_len = read_le<uint32_t>(f, ctx);
        std::string name(name_len, '\0');
        read_exact(f, name.data(), name_len, ctx);

        // Shape
        const uint32_t rows = read_le<uint32_t>(f, ctx);
        const uint32_t cols = read_le<uint32_t>(f, ctx);

        // Data
        const std::size_t numel = static_cast<std::size_t>(rows) *
                                  static_cast<std::size_t>(cols);
        std::vector<float> data(numel);
        read_exact(f, data.data(), numel * sizeof(float), ctx);

        wf.tensors[name] = Tensor::dense(rows, cols, std::move(data));
    }

    return wf;
}

// ============================================================================
//  Model — Layer addition
// ============================================================================

std::size_t Model::add_gcn_layer(std::size_t in_f, std::size_t out_f,
                                 bool bias, Activation act,
                                 InterActivation post) {
    const std::size_t idx = gcn_layers_.size();
    gcn_layers_.emplace_back(in_f, out_f, bias, act);

    LayerEntry entry{};
    entry.type       = LayerEntry::Type::GCN;
    entry.start_idx  = idx;
    entry.count      = 1;
    entry.gat_concat = false;
    entry.post_act   = post;

    const std::size_t layer_idx = layers_.size();
    layers_.push_back(entry);
    return layer_idx;
}

std::size_t Model::add_sage_layer(std::size_t in_f, std::size_t out_f,
                                  SAGELayer::Aggregator agg, bool bias,
                                  Activation act, InterActivation post) {
    const std::size_t idx = sage_layers_.size();
    sage_layers_.emplace_back(in_f, out_f, agg, bias, act);

    LayerEntry entry{};
    entry.type       = LayerEntry::Type::SAGE;
    entry.start_idx  = idx;
    entry.count      = 1;
    entry.gat_concat = false;
    entry.post_act   = post;

    const std::size_t layer_idx = layers_.size();
    layers_.push_back(entry);
    return layer_idx;
}

std::size_t Model::add_gat_layer(std::size_t in_f, std::size_t out_f,
                                 std::size_t num_heads, bool concat,
                                 float neg_slope, bool bias,
                                 Activation act, InterActivation post) {
    const std::size_t idx = gat_layers_.size();

    for (std::size_t h = 0; h < num_heads; ++h) {
        gat_layers_.emplace_back(in_f, out_f, neg_slope, bias, act);
    }

    LayerEntry entry{};
    entry.type       = LayerEntry::Type::GAT;
    entry.start_idx  = idx;
    entry.count      = num_heads;
    entry.gat_concat = concat;
    entry.post_act   = post;

    const std::size_t layer_idx = layers_.size();
    layers_.push_back(entry);
    return layer_idx;
}

// ============================================================================
//  Model — Weight loading
// ============================================================================

void Model::load_weights(const std::string& path) {
    WeightFile wf = load_weight_file(path);
    load_weights(wf);
}

void Model::load_weights(const WeightFile& wf) {
    auto get = [&](const std::string& name) -> const Tensor& {
        auto it = wf.tensors.find(name);
        if (it == wf.tensors.end()) {
            throw std::runtime_error(
                "Model::load_weights: missing tensor '" + name + "'");
        }
        return it->second;
    };

    for (std::size_t li = 0; li < layers_.size(); ++li) {
        const auto& entry = layers_[li];
        const std::string prefix = "layer" + std::to_string(li);

        switch (entry.type) {
            case LayerEntry::Type::GCN: {
                auto& layer = gcn_layers_[entry.start_idx];
                layer.set_weight(get(prefix + ".weight"));
                if (layer.has_bias()) {
                    layer.set_bias(get(prefix + ".bias"));
                }
                break;
            }
            case LayerEntry::Type::SAGE: {
                auto& layer = sage_layers_[entry.start_idx];
                layer.set_weight_neigh(get(prefix + ".weight_neigh"));
                layer.set_weight_self(get(prefix + ".weight_self"));
                if (layer.has_bias()) {
                    layer.set_bias(get(prefix + ".bias"));
                }
                break;
            }
            case LayerEntry::Type::GAT: {
                for (std::size_t h = 0; h < entry.count; ++h) {
                    auto& layer = gat_layers_[entry.start_idx + h];
                    const std::string hp = prefix + ".head" + std::to_string(h);
                    layer.set_weight(get(hp + ".weight"));
                    layer.set_attn_left(get(hp + ".attn_left"));
                    layer.set_attn_right(get(hp + ".attn_right"));
                    if (layer.has_bias()) {
                        layer.set_bias(get(hp + ".bias"));
                    }
                }
                break;
            }
        }
    }
}

// ============================================================================
//  Helpers — horizontal concatenation and averaging of tensors
// ============================================================================

/// Concatenate tensors side-by-side:  [N×C1, N×C2, ...] → N×(C1+C2+...)
static Tensor concat_horizontal(const std::vector<Tensor>& ts) {
    const std::size_t N = ts[0].rows();
    std::size_t total_cols = 0;
    for (const auto& t : ts) total_cols += t.cols();

    std::vector<float> data(N * total_cols, 0.0f);
    std::size_t col_off = 0;

    for (const auto& t : ts) {
        const std::size_t C = t.cols();
        const float* src = t.data().data();
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < C; ++j) {
                data[i * total_cols + col_off + j] = src[i * C + j];
            }
        }
        col_off += C;
    }

    return Tensor::dense(N, total_cols, std::move(data));
}

/// Element-wise average of tensors (all same shape).
static Tensor average_tensors(const std::vector<Tensor>& ts) {
    const std::size_t N = ts[0].rows();
    const std::size_t C = ts[0].cols();
    const std::size_t total = N * C;

    std::vector<float> data(total, 0.0f);
    for (const auto& t : ts) {
        const float* src = t.data().data();
        for (std::size_t i = 0; i < total; ++i) {
            data[i] += src[i];
        }
    }

    const float inv = 1.0f / static_cast<float>(ts.size());
    for (std::size_t i = 0; i < total; ++i) {
        data[i] *= inv;
    }

    return Tensor::dense(N, C, std::move(data));
}

// ============================================================================
//  Model — Forward pass
// ============================================================================
Tensor Model::forward(const Tensor& adjacency, const Tensor& features) const {
    if (layers_.empty()) {
        throw std::runtime_error("Model::forward: no layers configured.");
    }

    // ── Determine which adjacency variants are needed ────────────────────
    bool need_gcn_norm   = false;
    bool need_self_loops = false;

    for (const auto& e : layers_) {
        if (e.type == LayerEntry::Type::GCN)  need_gcn_norm   = true;
        if (e.type == LayerEntry::Type::GAT)  need_self_loops = true;
    }

    Tensor A_gcn;
    if (need_gcn_norm) {
        A_gcn = gcn_norm(adjacency);
    }

    Tensor A_sl;
    if (need_self_loops) {
        A_sl = add_self_loops(adjacency);
    }

    const Tensor& A_raw = adjacency;   // for SAGE

    // ── Execute layers sequentially ──────────────────────────────────────
    Tensor H = features;

    for (const auto& entry : layers_) {
        switch (entry.type) {

            case LayerEntry::Type::GCN: {
                const auto& layer = gcn_layers_[entry.start_idx];
                H = layer.forward(A_gcn, H);
                break;
            }

            case LayerEntry::Type::SAGE: {
                const auto& layer = sage_layers_[entry.start_idx];
                H = layer.forward(A_raw, H);
                break;
            }

            case LayerEntry::Type::GAT: {
                if (entry.count == 1) {
                    const auto& layer = gat_layers_[entry.start_idx];
                    H = layer.forward(A_sl, H);
                } else {
                    // Multi-head: run each head, then concat or average
                    std::vector<Tensor> heads;
                    heads.reserve(entry.count);
                    for (std::size_t h = 0; h < entry.count; ++h) {
                        const auto& layer = gat_layers_[entry.start_idx + h];
                        heads.push_back(layer.forward(A_sl, H));
                    }
                    if (entry.gat_concat) {
                        H = concat_horizontal(heads);
                    } else {
                        H = average_tensors(heads);
                    }
                }
                break;
            }
        }

        // ── Post-layer activation ────────────────────────────────────────
        switch (entry.post_act) {
            case InterActivation::ReLU:
                relu_inplace(H);
                break;
            case InterActivation::ELU:
                elu_inplace(H);
                break;
            case InterActivation::None:
                break;
        }
    }

    return H;
}

}  // namespace tinygnn
