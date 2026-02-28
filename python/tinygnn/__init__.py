"""
TinyGNN — Zero-dependency GNN inference engine
================================================

Python front-end for the TinyGNN C++17 inference engine.
Provides seamless NumPy / SciPy / PyTorch interoperability.

Quick start
-----------
>>> import tinygnn
>>> import numpy as np
>>>
>>> # Create tensors from NumPy
>>> X = tinygnn.Tensor.from_numpy(np.random.randn(10, 16).astype(np.float32))
>>> print(X.rows, X.cols)
10 16
>>>
>>> # Load a pre-trained model and run inference
>>> model = tinygnn.Model()
>>> model.add_gcn_layer(1433, 16)
>>> model.add_gcn_layer(16, 7, activation=tinygnn.Activation.NONE)
>>> model.load_weights("weights_gcn.bin")
>>> output = model.forward(adj, features)
"""

from __future__ import annotations

# Re-export everything from the C++ extension
from _tinygnn_core import (
    # Enums
    StorageFormat,
    Activation,

    # Core tensor
    Tensor,

    # Ops
    matmul,
    spmm,
    relu_inplace,
    leaky_relu_inplace,
    elu_inplace,
    sigmoid_inplace,
    tanh_inplace,
    gelu_inplace,
    softmax_inplace,
    log_softmax_inplace,
    add_bias,

    # Graph utilities
    add_self_loops,
    gcn_norm,
    edge_softmax,
    sage_max_aggregate,

    # Layers
    GCNLayer,
    SAGELayer,
    GATLayer,

    # Model
    Model,

    # I/O
    CoraData,
    WeightFile,
    load_cora_binary,
    load_weight_file,

    # Graph loading
    GraphData,
    GraphLoader,
)

__version__ = "0.1.0"

__all__ = [
    # Enums
    "StorageFormat",
    "Activation",
    # Core
    "Tensor",
    # Ops
    "matmul",
    "spmm",
    "relu_inplace",
    "leaky_relu_inplace",
    "elu_inplace",
    "sigmoid_inplace",
    "tanh_inplace",
    "gelu_inplace",
    "softmax_inplace",
    "log_softmax_inplace",
    "add_bias",
    # Graph
    "add_self_loops",
    "gcn_norm",
    "edge_softmax",
    "sage_max_aggregate",
    # Layers
    "GCNLayer",
    "SAGELayer",
    "GATLayer",
    # Model
    "Model",
    # I/O
    "CoraData",
    "WeightFile",
    "load_cora_binary",
    "load_weight_file",
    "GraphData",
    "GraphLoader",
]
