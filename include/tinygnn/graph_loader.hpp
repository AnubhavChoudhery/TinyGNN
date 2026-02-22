#pragma once

#include "tinygnn/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tinygnn {

// ============================================================================
//  GraphData — holds the loaded graph's adjacency matrix + node features
// ============================================================================
struct GraphData {
    Tensor      adjacency;       // SparseCSR (num_nodes × num_nodes), values 1.0
    Tensor      node_features;   // Dense     (num_nodes × num_features)
    std::size_t num_nodes    = 0;
    std::size_t num_edges    = 0;
    std::size_t num_features = 0;
};

// ============================================================================
//  GraphLoader — ingestion pipeline for graph CSV data
// ============================================================================
//  Loads graph structure from edge-list CSV and node features from CSV,
//  converting raw edge lists into hardware-friendly sorted CSR format.
//
//  Expected CSV formats:
//    edges.csv         : "src,dst" per line (optional header row)
//    node_features.csv : "node_id,f0,f1,...,fn" per line (optional header row)
//
//  Design principles:
//    • Zero external dependencies (only <fstream>, <sstream>, <algorithm>)
//    • Automatic header detection (skips rows starting with alphabetic chars)
//    • Handles both LF and CRLF line endings
//    • Column indices are sorted within each CSR row
//    • Robust error handling with descriptive exception messages
// ============================================================================
class GraphLoader {
public:
    // ── Parsing ─────────────────────────────────────────────────────────────

    /// Parse edge-list CSV → vector of (src, dst) pairs.
    /// Automatically detects and skips a header row.
    /// Supports both LF and CRLF line endings.
    /// @throws std::runtime_error on file-open failure, malformed lines,
    ///         or negative node IDs.
    static std::vector<std::pair<int32_t, int32_t>>
    parse_edges(const std::string& path);

    /// Parse node-feature CSV → Dense Tensor (num_nodes × num_features).
    /// Rows are placed by node_id so row i corresponds to node i.
    /// Missing node IDs are zero-filled.
    /// @throws std::runtime_error on file-open failure, invalid values,
    ///         or inconsistent feature counts.
    static Tensor parse_features(const std::string& path);

    // ── CSR Conversion ──────────────────────────────────────────────────────

    /// Convert raw (src, dst) edge list into a sorted CSR adjacency Tensor.
    /// The result is a (num_nodes × num_nodes) SparseCSR tensor where:
    ///   • Column indices are sorted within each row
    ///   • All values are 1.0f (unweighted adjacency)
    ///
    /// Algorithm complexity: O(E + V + E·log(E/V))
    ///   Step 1: Count out-degree per node        — O(E)
    ///   Step 2: Build row_ptr via prefix sum      — O(V)
    ///   Step 3: Fill col_ind via offset insertion  — O(E)
    ///   Step 4: Sort col_ind within each row      — O(E·log(E/V)) amortised
    ///
    /// @throws std::invalid_argument on out-of-range node IDs.
    static Tensor edge_list_to_csr(
        const std::vector<std::pair<int32_t, int32_t>>& edges,
        std::size_t num_nodes);

    // ── Full Pipeline ───────────────────────────────────────────────────────

    /// Load a complete graph (adjacency + features) from two CSV files.
    /// Infers num_nodes from the maximum node ID across both files.
    /// If the feature tensor has fewer rows than num_nodes (i.e. some nodes
    /// appear only in the edge list), it is zero-padded to match.
    static GraphData load(const std::string& edges_path,
                          const std::string& features_path);
};

}  // namespace tinygnn
