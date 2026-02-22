#include "tinygnn/graph_loader.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace tinygnn {

// ============================================================================
//  Internal helpers (anonymous namespace — no linkage outside this TU)
// ============================================================================
namespace {

/// Trim leading and trailing whitespace (space, tab, CR, LF)
std::string trim(const std::string& s) {
    const auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return {};
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

/// Return true if the line looks like a CSV header (first token contains a
/// letter or underscore — not a pure integer).
bool is_header(const std::string& line) {
    const std::string trimmed = trim(line);
    if (trimmed.empty()) return false;
    for (char c : trimmed) {
        if (c == ',' || c == ' ' || c == '\t') break;   // stop at separator
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_')
            return true;
    }
    return false;
}

/// Read all non-empty, trimmed lines from a file.
/// @throws std::runtime_error if the file cannot be opened.
std::vector<std::string> read_lines(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error(
            "GraphLoader: cannot open file '" + path + "'");
    }

    std::vector<std::string> lines;
    std::string raw;
    while (std::getline(file, raw)) {
        std::string trimmed = trim(raw);
        if (!trimmed.empty()) {
            lines.push_back(std::move(trimmed));
        }
    }
    return lines;
}

}  // anonymous namespace

// ============================================================================
//  parse_edges
// ============================================================================
std::vector<std::pair<int32_t, int32_t>>
GraphLoader::parse_edges(const std::string& path) {
    const auto lines = read_lines(path);
    if (lines.empty()) {
        throw std::runtime_error(
            "GraphLoader::parse_edges: file is empty: '" + path + "'");
    }

    const std::size_t start = is_header(lines[0]) ? 1 : 0;

    std::vector<std::pair<int32_t, int32_t>> edges;
    edges.reserve(lines.size() - start);

    for (std::size_t i = start; i < lines.size(); ++i) {
        const auto& line = lines[i];
        const auto comma = line.find(',');
        if (comma == std::string::npos) {
            throw std::runtime_error(
                "GraphLoader::parse_edges: malformed line " +
                std::to_string(i + 1) + ": '" + line + "'");
        }

        int32_t src = 0;
        int32_t dst = 0;
        try {
            src = static_cast<int32_t>(std::stoi(line.substr(0, comma)));
            dst = static_cast<int32_t>(std::stoi(line.substr(comma + 1)));
        } catch (const std::exception&) {
            throw std::runtime_error(
                "GraphLoader::parse_edges: invalid integer on line " +
                std::to_string(i + 1) + ": '" + line + "'");
        }

        if (src < 0 || dst < 0) {
            throw std::runtime_error(
                "GraphLoader::parse_edges: negative node ID on line " +
                std::to_string(i + 1) + ": src=" + std::to_string(src) +
                ", dst=" + std::to_string(dst));
        }

        edges.emplace_back(src, dst);
    }

    return edges;
}

// ============================================================================
//  parse_features
// ============================================================================
Tensor GraphLoader::parse_features(const std::string& path) {
    const auto lines = read_lines(path);
    if (lines.empty()) {
        throw std::runtime_error(
            "GraphLoader::parse_features: file is empty: '" + path + "'");
    }

    const std::size_t start = is_header(lines[0]) ? 1 : 0;
    const std::size_t num_data_lines = lines.size() - start;
    if (num_data_lines == 0) {
        throw std::runtime_error(
            "GraphLoader::parse_features: no data rows in '" + path + "'");
    }

    // First pass: parse everything and figure out dimensions
    std::size_t num_features = 0;
    int32_t max_node_id = -1;

    struct NodeRow {
        int32_t             node_id;
        std::vector<float>  features;
    };
    std::vector<NodeRow> rows;
    rows.reserve(num_data_lines);

    for (std::size_t i = start; i < lines.size(); ++i) {
        const auto& line = lines[i];
        std::istringstream ss(line);
        std::string token;

        // First token → node_id
        if (!std::getline(ss, token, ',')) {
            throw std::runtime_error(
                "GraphLoader::parse_features: empty line " +
                std::to_string(i + 1));
        }

        int32_t node_id = 0;
        try {
            node_id = static_cast<int32_t>(std::stoi(trim(token)));
        } catch (const std::exception&) {
            throw std::runtime_error(
                "GraphLoader::parse_features: invalid node_id on line " +
                std::to_string(i + 1) + ": '" + token + "'");
        }
        if (node_id < 0) {
            throw std::runtime_error(
                "GraphLoader::parse_features: negative node_id on line " +
                std::to_string(i + 1));
        }
        if (node_id > max_node_id) max_node_id = node_id;

        // Remaining tokens → features
        std::vector<float> feats;
        while (std::getline(ss, token, ',')) {
            try {
                feats.push_back(std::stof(trim(token)));
            } catch (const std::exception&) {
                throw std::runtime_error(
                    "GraphLoader::parse_features: invalid feature on line " +
                    std::to_string(i + 1) + ": '" + token + "'");
            }
        }

        // Validate / discover feature count
        if (num_features == 0) {
            num_features = feats.size();
            if (num_features == 0) {
                throw std::runtime_error(
                    "GraphLoader::parse_features: no features on line " +
                    std::to_string(i + 1));
            }
        } else if (feats.size() != num_features) {
            throw std::runtime_error(
                "GraphLoader::parse_features: inconsistent feature count "
                "on line " + std::to_string(i + 1) + ": expected " +
                std::to_string(num_features) + " but got " +
                std::to_string(feats.size()));
        }

        rows.push_back({node_id, std::move(feats)});
    }

    // Place features into a (max_node_id+1) × num_features dense matrix
    const auto num_nodes = static_cast<std::size_t>(max_node_id + 1);
    std::vector<float> data(num_nodes * num_features, 0.0f);

    for (const auto& row : rows) {
        const auto offset =
            static_cast<std::size_t>(row.node_id) * num_features;
        for (std::size_t j = 0; j < num_features; ++j) {
            data[offset + j] = row.features[j];
        }
    }

    return Tensor::dense(num_nodes, num_features, std::move(data));
}

// ============================================================================
//  edge_list_to_csr
// ============================================================================
Tensor GraphLoader::edge_list_to_csr(
    const std::vector<std::pair<int32_t, int32_t>>& edges,
    std::size_t num_nodes)
{
    // Handle trivial case: no nodes, no edges → 0×0 empty CSR
    if (num_nodes == 0) {
        if (!edges.empty()) {
            throw std::invalid_argument(
                "GraphLoader::edge_list_to_csr: num_nodes is 0 but "
                "edges is non-empty");
        }
        return Tensor::sparse_csr(0, 0, {0}, {}, {});
    }

    const auto num_edges = edges.size();

    // ── Validate all edge endpoints ─────────────────────────────────────
    for (std::size_t i = 0; i < num_edges; ++i) {
        const auto [src, dst] = edges[i];
        if (src < 0 || static_cast<std::size_t>(src) >= num_nodes) {
            throw std::invalid_argument(
                "GraphLoader::edge_list_to_csr: edge[" + std::to_string(i) +
                "].src = " + std::to_string(src) +
                " out of range [0, " + std::to_string(num_nodes) + ")");
        }
        if (dst < 0 || static_cast<std::size_t>(dst) >= num_nodes) {
            throw std::invalid_argument(
                "GraphLoader::edge_list_to_csr: edge[" + std::to_string(i) +
                "].dst = " + std::to_string(dst) +
                " out of range [0, " + std::to_string(num_nodes) + ")");
        }
    }

    // ── Step 1: Count out-degree of each source node ── O(E) ────────────
    std::vector<int32_t> degree(num_nodes, 0);
    for (const auto& [src, dst] : edges) {
        degree[static_cast<std::size_t>(src)]++;
    }

    // ── Step 2: Build row_ptr as prefix sum ── O(V) ─────────────────────
    std::vector<int32_t> row_ptr(num_nodes + 1, 0);
    for (std::size_t i = 0; i < num_nodes; ++i) {
        row_ptr[i + 1] = row_ptr[i] + degree[i];
    }

    // ── Step 3: Fill col_ind via offset insertion ── O(E) ───────────────
    std::vector<int32_t> col_ind(num_edges);
    std::vector<int32_t> offset(num_nodes, 0);   // per-row insertion cursor
    for (const auto& [src, dst] : edges) {
        const auto s = static_cast<std::size_t>(src);
        const auto pos = static_cast<std::size_t>(row_ptr[s] + offset[s]);
        col_ind[pos] = dst;
        offset[s]++;
    }

    // ── Step 4: Sort column indices within each row ── O(E·log(E/V)) ────
    for (std::size_t r = 0; r < num_nodes; ++r) {
        const auto begin = col_ind.begin() + row_ptr[r];
        const auto end   = col_ind.begin() + row_ptr[r + 1];
        std::sort(begin, end);
    }

    // ── Step 5: Build values array (1.0 for every edge) ── O(E) ────────
    std::vector<float> values(num_edges, 1.0f);

    return Tensor::sparse_csr(
        num_nodes, num_nodes,
        std::move(row_ptr),
        std::move(col_ind),
        std::move(values));
}

// ============================================================================
//  load  (full pipeline)
// ============================================================================
GraphData GraphLoader::load(const std::string& edges_path,
                            const std::string& features_path)
{
    // 1. Parse raw CSV data
    auto edges    = parse_edges(edges_path);
    auto features = parse_features(features_path);

    // 2. Determine num_nodes = max(nodes in edges, rows in features)
    int32_t max_from_edges = -1;
    for (const auto& [src, dst] : edges) {
        if (src > max_from_edges) max_from_edges = src;
        if (dst > max_from_edges) max_from_edges = dst;
    }
    const auto nodes_from_edges =
        edges.empty() ? std::size_t{0}
                      : static_cast<std::size_t>(max_from_edges + 1);
    const auto nodes_from_features = features.rows();
    const auto num_nodes = std::max(nodes_from_edges, nodes_from_features);

    // 3. Build sorted CSR adjacency matrix
    auto adjacency = edge_list_to_csr(edges, num_nodes);

    // 4. Zero-pad features if some nodes appear only in edges
    if (features.rows() < num_nodes) {
        const auto feat_cols = features.cols();
        std::vector<float> expanded(num_nodes * feat_cols, 0.0f);
        const auto& fdata = features.data();
        for (std::size_t i = 0; i < fdata.size(); ++i) {
            expanded[i] = fdata[i];
        }
        features = Tensor::dense(num_nodes, feat_cols, std::move(expanded));
    }

    GraphData gd;
    gd.adjacency     = std::move(adjacency);
    gd.node_features = std::move(features);
    gd.num_nodes     = num_nodes;
    gd.num_edges     = edges.size();
    gd.num_features  = gd.node_features.cols();
    return gd;
}

}  // namespace tinygnn
