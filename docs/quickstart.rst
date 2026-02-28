Quick Start
===========

Python Usage
------------

.. code-block:: python

   import tinygnn
   import numpy as np

   # --- Create tensors from NumPy ---
   features = tinygnn.Tensor.from_numpy(
       np.random.randn(100, 16).astype(np.float32)
   )

   # --- Build adjacency from edge list ---
   edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int32)
   adj = tinygnn.Tensor.from_edge_index(edge_index, num_nodes=100)

   # --- Use individual layers ---
   gcn = tinygnn.GCNLayer(16, 8, activation=tinygnn.Activation.RELU)
   W = tinygnn.Tensor.from_numpy(np.random.randn(16, 8).astype(np.float32))
   gcn.set_weight(W)

   adj_norm = tinygnn.gcn_norm(adj)
   output = gcn.forward(adj_norm, features)
   result = output.to_numpy()
   print(f"Output shape: {result.shape}")  # (100, 8)


Model-Based Inference
---------------------

.. code-block:: python

   import tinygnn

   # Build a 2-layer GCN model
   model = tinygnn.Model()
   model.add_gcn_layer(1433, 64, activation=tinygnn.Activation.RELU)
   model.add_gcn_layer(64, 7, activation=tinygnn.Activation.NONE)

   # Load pre-trained weights
   model.load_weights("weights/gcn_cora.bin")

   # Run inference
   cora = tinygnn.load_cora_binary("weights/cora_graph.bin")
   logits = model.forward(cora.adjacency, cora.features)
   predictions = logits.to_numpy().argmax(axis=1)


SciPy / PyTorch Interop
------------------------

.. code-block:: python

   import tinygnn
   import scipy.sparse as sp
   import numpy as np

   # From SciPy sparse matrix
   scipy_adj = sp.random(100, 100, density=0.05, format='csr', dtype=np.float32)
   adj = tinygnn.Tensor.from_scipy_csr(scipy_adj)

   # From PyTorch Geometric edge_index
   import torch
   edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int32).numpy()
   adj = tinygnn.Tensor.from_edge_index(edge_index, num_nodes=3)


C++ Usage
---------

.. code-block:: cpp

   #include "tinygnn/tensor.hpp"
   #include "tinygnn/ops.hpp"
   #include "tinygnn/layers.hpp"
   #include "tinygnn/model.hpp"

   using namespace tinygnn;

   // Build and run a GCN
   Model model;
   model.add_gcn_layer(1433, 64, true, Activation::ReLU);
   model.add_gcn_layer(64, 7, true, Activation::None);
   model.load_weights("weights/gcn_cora.bin");

   CoraData cora = load_cora_binary("weights/cora_graph.bin");
   Tensor logits = model.forward(cora.adjacency, cora.features);
