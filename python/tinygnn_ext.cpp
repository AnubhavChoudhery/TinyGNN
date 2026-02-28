// ============================================================================
//  TinyGNN — pybind11 Bindings  (Phase 7: Python Bridge)
//  python/tinygnn_ext.cpp
//
//  Exposes the core TinyGNN C++ API to Python:
//
//    Tensor           — Dense + SparseCSR with zero-copy NumPy interop
//    StorageFormat    — enum (Dense, SparseCSR)
//    Activation       — enum (None_, ReLU)
//    Ops              — matmul, spmm, activations, add_bias
//    Layers           — GCNLayer, SAGELayer, GATLayer
//    Graph utilities  — add_self_loops, gcn_norm, edge_softmax
//    Model            — Dynamic execution graph
//    I/O              — load_cora_binary, load_weight_file
//
//  Zero-copy strategy:
//    • Tensor.to_numpy()   → NumPy array sharing the same memory
//    • Tensor.from_numpy() → Tensor wrapping (copying) a NumPy array
//    • Tensor.from_scipy_csr() → builds SparseCSR from scipy.sparse.csr_matrix
//    • Tensor.from_pyg_edge_index() → builds SparseCSR from PyG edge_index
//
//  pybind11 version: ≥ 2.11 (header-only, installed via pip)
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "tinygnn/tensor.hpp"
#include "tinygnn/ops.hpp"
#include "tinygnn/layers.hpp"
#include "tinygnn/model.hpp"
#include "tinygnn/graph_loader.hpp"

namespace py = pybind11;
using namespace tinygnn;

// ── Helper: create Dense Tensor from NumPy array (copy) ─────────────────────
static Tensor tensor_from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    if (buf.ndim != 2) {
        throw std::invalid_argument(
            "Tensor.from_numpy: expected 2D array, got " +
            std::to_string(buf.ndim) + "D");
    }
    auto rows = static_cast<std::size_t>(buf.shape[0]);
    auto cols = static_cast<std::size_t>(buf.shape[1]);
    const float* ptr = static_cast<const float*>(buf.ptr);
    std::vector<float> data(ptr, ptr + rows * cols);
    return Tensor::dense(rows, cols, std::move(data));
}

// ── Helper: create Dense Tensor from 1D NumPy (as 1×N bias) ────────────────
static Tensor tensor_from_numpy_1d(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    if (buf.ndim == 1) {
        auto cols = static_cast<std::size_t>(buf.shape[0]);
        const float* ptr = static_cast<const float*>(buf.ptr);
        std::vector<float> data(ptr, ptr + cols);
        return Tensor::dense(1, cols, std::move(data));
    } else if (buf.ndim == 2) {
        return tensor_from_numpy(arr);
    } else {
        throw std::invalid_argument(
            "Tensor.from_numpy_1d: expected 1D or 2D array, got " +
            std::to_string(buf.ndim) + "D");
    }
}

// ── Helper: Tensor → NumPy (copy for safety, since Tensor may be moved) ────
static py::array_t<float> tensor_to_numpy(const Tensor& t) {
    if (t.format() != StorageFormat::Dense) {
        throw std::invalid_argument(
            "to_numpy: only Dense tensors can be converted to NumPy.");
    }
    auto rows = static_cast<py::ssize_t>(t.rows());
    auto cols = static_cast<py::ssize_t>(t.cols());
    auto result = py::array_t<float>({rows, cols});
    auto buf = result.request();
    float* dst = static_cast<float*>(buf.ptr);
    const float* src = t.data().data();
    std::copy(src, src + t.rows() * t.cols(), dst);
    return result;
}

// ── Helper: SparseCSR from scipy.sparse.csr_matrix ──────────────────────────
static Tensor tensor_from_scipy_csr(py::object csr_mat) {
    // Extract indptr, indices, data, shape
    py::array_t<int32_t> indptr = csr_mat.attr("indptr").cast<py::array_t<int32_t>>();
    py::array_t<int32_t> indices = csr_mat.attr("indices").cast<py::array_t<int32_t>>();
    py::array_t<float, py::array::forcecast> data = csr_mat.attr("data").cast<py::array_t<float, py::array::forcecast>>();
    py::tuple shape = csr_mat.attr("shape").cast<py::tuple>();

    auto rows = shape[0].cast<std::size_t>();
    auto cols = shape[1].cast<std::size_t>();

    auto rp_buf = indptr.request();
    auto ci_buf = indices.request();
    auto v_buf = data.request();

    const int32_t* rp_ptr = static_cast<const int32_t*>(rp_buf.ptr);
    const int32_t* ci_ptr = static_cast<const int32_t*>(ci_buf.ptr);
    const float* v_ptr = static_cast<const float*>(v_buf.ptr);

    std::vector<int32_t> rp(rp_ptr, rp_ptr + rows + 1);
    std::vector<int32_t> ci(ci_ptr, ci_ptr + static_cast<std::size_t>(rp_ptr[rows]));
    std::vector<float> vals(v_ptr, v_ptr + static_cast<std::size_t>(rp_ptr[rows]));

    return Tensor::sparse_csr(rows, cols, std::move(rp), std::move(ci), std::move(vals));
}

// ── Helper: SparseCSR from PyG edge_index (COO) ────────────────────────────
static Tensor tensor_from_edge_index(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> edge_index,
    std::size_t num_nodes)
{
    auto buf = edge_index.request();
    if (buf.ndim != 2 || buf.shape[0] != 2) {
        throw std::invalid_argument(
            "from_edge_index: expected shape (2, E), got (" +
            std::to_string(buf.shape[0]) + ", " +
            std::to_string(buf.ndim > 1 ? buf.shape[1] : 0) + ")");
    }

    auto num_edges = static_cast<std::size_t>(buf.shape[1]);
    const int32_t* ptr = static_cast<const int32_t*>(buf.ptr);
    const int32_t* src_arr = ptr;
    const int32_t* dst_arr = ptr + num_edges;

    // Build COO → sorted CSR
    // Count per row
    std::vector<int32_t> rp(num_nodes + 1, 0);
    for (std::size_t e = 0; e < num_edges; ++e) {
        auto s = static_cast<std::size_t>(src_arr[e]);
        if (s < num_nodes) rp[s + 1]++;
    }
    for (std::size_t i = 1; i <= num_nodes; ++i) rp[i] += rp[i - 1];

    std::vector<int32_t> ci(num_edges);
    std::vector<float> vals(num_edges, 1.0f);
    std::vector<int32_t> wp(rp.begin(), rp.end());
    for (std::size_t e = 0; e < num_edges; ++e) {
        auto s = static_cast<std::size_t>(src_arr[e]);
        ci[wp[s]++] = dst_arr[e];
    }

    // Sort col_ind within each row
    for (std::size_t i = 0; i < num_nodes; ++i) {
        std::sort(ci.begin() + rp[i], ci.begin() + rp[i + 1]);
    }

    return Tensor::sparse_csr(num_nodes, num_nodes,
                              std::move(rp), std::move(ci), std::move(vals));
}

// ── Helper: get CSR arrays as numpy for Python inspection ────────────────────
static py::array_t<int32_t> tensor_row_ptr_numpy(const Tensor& t) {
    const auto& rp = t.row_ptr();
    return py::array_t<int32_t>(static_cast<py::ssize_t>(rp.size()), rp.data());
}

static py::array_t<int32_t> tensor_col_ind_numpy(const Tensor& t) {
    const auto& ci = t.col_ind();
    return py::array_t<int32_t>(static_cast<py::ssize_t>(ci.size()), ci.data());
}

static py::array_t<float> tensor_values_numpy(const Tensor& t) {
    const auto& v = t.data();
    return py::array_t<float>(static_cast<py::ssize_t>(v.size()), v.data());
}


// ============================================================================
//  Module definition
// ============================================================================
PYBIND11_MODULE(_tinygnn_core, m) {
    m.doc() = "TinyGNN — Zero-dependency GNN inference engine (C++17 + pybind11)";

    // ── StorageFormat enum ─────────────────────────────────────────────────
    py::enum_<StorageFormat>(m, "StorageFormat")
        .value("Dense", StorageFormat::Dense)
        .value("SparseCSR", StorageFormat::SparseCSR)
        .export_values();

    // ── Activation enum ────────────────────────────────────────────────────
    py::enum_<Activation>(m, "Activation")
        .value("NONE", Activation::None)
        .value("RELU", Activation::ReLU)
        .export_values();

    // ── Tensor ─────────────────────────────────────────────────────────────
    py::class_<Tensor>(m, "Tensor")
        // Constructors
        .def(py::init<>(), "Create an empty 0×0 Dense tensor")
        .def_static("dense",
            py::overload_cast<std::size_t, std::size_t>(&Tensor::dense),
            "Create a zero-filled Dense tensor",
            py::arg("rows"), py::arg("cols"))
        .def_static("dense_from_data",
            [](std::size_t rows, std::size_t cols, std::vector<float> data) {
                return Tensor::dense(rows, cols, std::move(data));
            },
            "Create a Dense tensor from data vector",
            py::arg("rows"), py::arg("cols"), py::arg("data"))
        .def_static("sparse_csr", &Tensor::sparse_csr,
            "Create a SparseCSR tensor",
            py::arg("rows"), py::arg("cols"),
            py::arg("row_ptr"), py::arg("col_ind"), py::arg("values"))

        // NumPy interop
        .def_static("from_numpy", &tensor_from_numpy,
            "Create Dense Tensor from 2D NumPy array (copies data)",
            py::arg("array"))
        .def_static("from_numpy_1d", &tensor_from_numpy_1d,
            "Create Dense Tensor from 1D NumPy (as 1×N) or 2D array",
            py::arg("array"))
        .def_static("from_scipy_csr", &tensor_from_scipy_csr,
            "Create SparseCSR Tensor from scipy.sparse.csr_matrix",
            py::arg("csr_matrix"))
        .def_static("from_edge_index", &tensor_from_edge_index,
            "Create SparseCSR Tensor from PyG edge_index (2×E int32)",
            py::arg("edge_index"), py::arg("num_nodes"))
        .def("to_numpy", &tensor_to_numpy,
            "Convert Dense Tensor to NumPy array (copies data)")

        // CSR accessors for Python
        .def("row_ptr_numpy", &tensor_row_ptr_numpy,
            "Get row_ptr as NumPy int32 array (copy)")
        .def("col_ind_numpy", &tensor_col_ind_numpy,
            "Get col_ind as NumPy int32 array (copy)")
        .def("values_numpy", &tensor_values_numpy,
            "Get values (data) as NumPy float32 array (copy)")

        // Observers
        .def_property_readonly("format", &Tensor::format)
        .def_property_readonly("rows", &Tensor::rows)
        .def_property_readonly("cols", &Tensor::cols)
        .def_property_readonly("numel", &Tensor::numel)
        .def_property_readonly("nnz", &Tensor::nnz)
        .def_property_readonly("memory_footprint_bytes",
                               &Tensor::memory_footprint_bytes)
        .def("at", py::overload_cast<std::size_t, std::size_t>(&Tensor::at, py::const_),
            "Element access (Dense only)",
            py::arg("row"), py::arg("col"))
        .def("repr", &Tensor::repr)
        .def("__repr__", &Tensor::repr);

    // ── Ops ────────────────────────────────────────────────────────────────
    m.def("matmul", &matmul, "Dense matrix multiply C = A × B",
          py::arg("A"), py::arg("B"));
    m.def("spmm", &spmm, "Sparse-Dense multiply C = A(CSR) × B(Dense)",
          py::arg("A"), py::arg("B"));

    // Activations
    m.def("relu_inplace", &relu_inplace, "In-place ReLU", py::arg("X"));
    m.def("leaky_relu_inplace", &leaky_relu_inplace,
          "In-place Leaky ReLU", py::arg("X"), py::arg("alpha") = 0.01f);
    m.def("elu_inplace", &elu_inplace,
          "In-place ELU", py::arg("X"), py::arg("alpha") = 1.0f);
    m.def("sigmoid_inplace", &sigmoid_inplace, "In-place sigmoid", py::arg("X"));
    m.def("tanh_inplace", &tanh_inplace, "In-place tanh", py::arg("X"));
    m.def("gelu_inplace", &gelu_inplace, "In-place GELU", py::arg("X"));
    m.def("softmax_inplace", &softmax_inplace, "In-place row-wise softmax", py::arg("X"));
    m.def("log_softmax_inplace", &log_softmax_inplace,
          "In-place row-wise log-softmax", py::arg("X"));
    m.def("add_bias", &add_bias, "Broadcast bias addition",
          py::arg("X"), py::arg("bias"));

    // ── Graph normalization utilities ──────────────────────────────────────
    m.def("add_self_loops", &add_self_loops,
          "Add self-loops: A_hat = A + I", py::arg("A"));
    m.def("gcn_norm", &gcn_norm,
          "Symmetric GCN normalization: D^{-1/2}(A+I)D^{-1/2}", py::arg("A"));
    m.def("edge_softmax", &edge_softmax,
          "Sparse row-wise softmax over CSR values", py::arg("A"));
    m.def("sage_max_aggregate", &sage_max_aggregate,
          "Element-wise max pooling over neighbors",
          py::arg("A"), py::arg("H"));

    // ── GCNLayer ───────────────────────────────────────────────────────────
    py::class_<GCNLayer>(m, "GCNLayer")
        .def(py::init<std::size_t, std::size_t, bool, Activation>(),
             "Construct GCN layer",
             py::arg("in_features"), py::arg("out_features"),
             py::arg("use_bias") = true,
             py::arg("activation") = Activation::ReLU)
        .def("set_weight", &GCNLayer::set_weight, py::arg("w"))
        .def("set_bias", &GCNLayer::set_bias, py::arg("b"))
        .def("forward", &GCNLayer::forward,
             py::arg("A_norm"), py::arg("H"))
        .def_property_readonly("in_features", &GCNLayer::in_features)
        .def_property_readonly("out_features", &GCNLayer::out_features)
        .def_property_readonly("has_bias", &GCNLayer::has_bias)
        .def_property_readonly("activation", &GCNLayer::activation)
        .def_property_readonly("weight", &GCNLayer::weight)
        .def_property_readonly("bias", &GCNLayer::bias);

    // ── SAGELayer ──────────────────────────────────────────────────────────
    py::class_<SAGELayer> sage(m, "SAGELayer");

    py::enum_<SAGELayer::Aggregator>(sage, "Aggregator")
        .value("Mean", SAGELayer::Aggregator::Mean)
        .value("Max", SAGELayer::Aggregator::Max)
        .export_values();

    sage.def(py::init<std::size_t, std::size_t, SAGELayer::Aggregator, bool, Activation>(),
             "Construct GraphSAGE layer",
             py::arg("in_features"), py::arg("out_features"),
             py::arg("aggregator") = SAGELayer::Aggregator::Mean,
             py::arg("use_bias") = true,
             py::arg("activation") = Activation::ReLU)
        .def("set_weight_neigh", &SAGELayer::set_weight_neigh, py::arg("w"))
        .def("set_weight_self", &SAGELayer::set_weight_self, py::arg("w"))
        .def("set_bias", &SAGELayer::set_bias, py::arg("b"))
        .def("forward", &SAGELayer::forward, py::arg("A"), py::arg("H"))
        .def_property_readonly("in_features", &SAGELayer::in_features)
        .def_property_readonly("out_features", &SAGELayer::out_features)
        .def_property_readonly("has_bias", &SAGELayer::has_bias)
        .def_property_readonly("activation", &SAGELayer::activation)
        .def_property_readonly("aggregator", &SAGELayer::aggregator)
        .def_property_readonly("weight_neigh", &SAGELayer::weight_neigh)
        .def_property_readonly("weight_self", &SAGELayer::weight_self)
        .def_property_readonly("bias", &SAGELayer::bias);

    // ── GATLayer ───────────────────────────────────────────────────────────
    py::class_<GATLayer>(m, "GATLayer")
        .def(py::init<std::size_t, std::size_t, float, bool, Activation>(),
             "Construct GAT layer (single head)",
             py::arg("in_features"), py::arg("out_features"),
             py::arg("negative_slope") = 0.2f,
             py::arg("use_bias") = true,
             py::arg("activation") = Activation::None)
        .def("set_weight", &GATLayer::set_weight, py::arg("w"))
        .def("set_attn_left", &GATLayer::set_attn_left, py::arg("a"))
        .def("set_attn_right", &GATLayer::set_attn_right, py::arg("a"))
        .def("set_bias", &GATLayer::set_bias, py::arg("b"))
        .def("forward", &GATLayer::forward, py::arg("A"), py::arg("H"))
        .def_property_readonly("in_features", &GATLayer::in_features)
        .def_property_readonly("out_features", &GATLayer::out_features)
        .def_property_readonly("negative_slope", &GATLayer::negative_slope)
        .def_property_readonly("has_bias", &GATLayer::has_bias)
        .def_property_readonly("activation", &GATLayer::activation)
        .def_property_readonly("weight", &GATLayer::weight)
        .def_property_readonly("attn_left", &GATLayer::attn_left)
        .def_property_readonly("attn_right", &GATLayer::attn_right)
        .def_property_readonly("bias", &GATLayer::bias);

    // ── Model ──────────────────────────────────────────────────────────────
    py::class_<Model> model(m, "Model");

    py::enum_<Model::InterActivation>(model, "InterActivation")
        .value("NONE", Model::InterActivation::None)
        .value("RELU", Model::InterActivation::ReLU)
        .value("ELU", Model::InterActivation::ELU)
        .export_values();

    model.def(py::init<>())
        .def("add_gcn_layer", &Model::add_gcn_layer,
             py::arg("in_f"), py::arg("out_f"),
             py::arg("bias") = true,
             py::arg("act") = Activation::ReLU,
             py::arg("post") = Model::InterActivation::None)
        .def("add_sage_layer", &Model::add_sage_layer,
             py::arg("in_f"), py::arg("out_f"),
             py::arg("agg") = SAGELayer::Aggregator::Mean,
             py::arg("bias") = true,
             py::arg("act") = Activation::ReLU,
             py::arg("post") = Model::InterActivation::None)
        .def("add_gat_layer", &Model::add_gat_layer,
             py::arg("in_f"), py::arg("out_f"),
             py::arg("num_heads") = static_cast<std::size_t>(1),
             py::arg("concat") = true,
             py::arg("neg_slope") = 0.2f,
             py::arg("bias") = true,
             py::arg("act") = Activation::None,
             py::arg("post") = Model::InterActivation::None)
        .def("load_weights",
             py::overload_cast<const std::string&>(&Model::load_weights),
             py::arg("path"))
        .def("load_weights_from_file",
             py::overload_cast<const WeightFile&>(&Model::load_weights),
             "Load weights from a pre-loaded WeightFile object",
             py::arg("weight_file"))
        .def("forward", &Model::forward,
             py::arg("adjacency"), py::arg("features"))
        .def_property_readonly("num_layers", &Model::num_layers);

    // ── CoraData ───────────────────────────────────────────────────────────
    py::class_<CoraData>(m, "CoraData")
        .def_readonly("num_nodes", &CoraData::num_nodes)
        .def_readonly("num_features", &CoraData::num_features)
        .def_readonly("num_classes", &CoraData::num_classes)
        .def_readonly("num_edges", &CoraData::num_edges)
        .def_readonly("adjacency", &CoraData::adjacency)
        .def_readonly("features", &CoraData::features)
        .def_readonly("labels", &CoraData::labels)
        .def_readonly("train_mask", &CoraData::train_mask)
        .def_readonly("val_mask", &CoraData::val_mask)
        .def_readonly("test_mask", &CoraData::test_mask);

    // ── WeightFile ─────────────────────────────────────────────────────────
    py::class_<WeightFile>(m, "WeightFile")
        .def_readonly("test_accuracy", &WeightFile::test_accuracy)
        .def_readonly("tensors", &WeightFile::tensors);

    // ── I/O functions ──────────────────────────────────────────────────────
    m.def("load_cora_binary", &load_cora_binary,
          "Load Cora graph from binary file", py::arg("path"));
    m.def("load_weight_file", &load_weight_file,
          "Load TGNN weight file", py::arg("path"));

    // ── GraphData ──────────────────────────────────────────────────────────
    py::class_<GraphData>(m, "GraphData")
        .def_readonly("adjacency", &GraphData::adjacency)
        .def_readonly("node_features", &GraphData::node_features)
        .def_readonly("num_nodes", &GraphData::num_nodes)
        .def_readonly("num_edges", &GraphData::num_edges)
        .def_readonly("num_features", &GraphData::num_features);

    // ── GraphLoader ────────────────────────────────────────────────────────
    py::class_<GraphLoader>(m, "GraphLoader")
        .def_static("parse_edges", &GraphLoader::parse_edges,
                    "Parse edge-list CSV → vector of (src, dst)", py::arg("path"))
        .def_static("parse_features", &GraphLoader::parse_features,
                    "Parse node-feature CSV → Dense Tensor", py::arg("path"))
        .def_static("edge_list_to_csr", &GraphLoader::edge_list_to_csr,
                    "Convert edge list to CSR adjacency",
                    py::arg("edges"), py::arg("num_nodes"))
        .def_static("load", &GraphLoader::load,
                    "Load graph from CSV edge + feature files",
                    py::arg("edges_path"), py::arg("features_path"));
}
