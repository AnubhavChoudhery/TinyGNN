Python API Reference
====================

.. module:: tinygnn
   :synopsis: Zero-dependency GNN inference engine

Enumerations
------------

.. class:: StorageFormat

   Tensor storage format.

   .. attribute:: Dense

      Row-major contiguous storage. Memory: ``rows × cols × 4`` bytes.

   .. attribute:: SparseCSR

      Compressed Sparse Row format. Memory: ``nnz × 8 + (rows+1) × 4`` bytes.

.. class:: Activation

   Activation function selector for GNN layers.

   .. attribute:: NONE

      Identity (no activation).

   .. attribute:: RELU

      Rectified Linear Unit: ``max(0, x)``.


Tensor
------

.. class:: Tensor

   Core data structure supporting both Dense and SparseCSR storage formats.

   **Static constructors:**

   .. staticmethod:: dense(rows, cols)

      Create a zero-filled Dense tensor of shape ``(rows, cols)``.

   .. staticmethod:: dense_from_data(rows, cols, data)

      Create a Dense tensor from a flat list of floats.

   .. staticmethod:: sparse_csr(rows, cols, row_ptr, col_ind, values)

      Create a SparseCSR tensor from CSR arrays.

   .. staticmethod:: from_numpy(array)

      Create a Dense Tensor from a 2D NumPy float32 array (copies data).

      :param array: 2D ``numpy.ndarray`` of dtype ``float32``
      :returns: Dense ``Tensor``

   .. staticmethod:: from_numpy_1d(array)

      Create a Dense Tensor from a 1D NumPy array as shape ``(1, N)``.
      Useful for bias vectors.

   .. staticmethod:: from_scipy_csr(csr_matrix)

      Create a SparseCSR Tensor from a ``scipy.sparse.csr_matrix``.

   .. staticmethod:: from_edge_index(edge_index, num_nodes)

      Create a SparseCSR adjacency from a PyG-style ``(2, E)`` edge index.

      :param edge_index: ``numpy.ndarray`` of shape ``(2, E)``, dtype ``int32``
      :param num_nodes: Number of nodes in the graph
      :returns: SparseCSR ``Tensor`` of shape ``(num_nodes, num_nodes)``

   **Instance methods:**

   .. method:: to_numpy()

      Convert a Dense Tensor to a 2D NumPy array (copies data).

   .. method:: row_ptr_numpy()

      Get CSR ``row_ptr`` as a NumPy ``int32`` array.

   .. method:: col_ind_numpy()

      Get CSR ``col_ind`` as a NumPy ``int32`` array.

   .. method:: values_numpy()

      Get tensor data/values as a NumPy ``float32`` array.

   .. method:: at(row, col)

      Element access (Dense tensors only).

   **Properties:**

   .. attribute:: format

      ``StorageFormat.Dense`` or ``StorageFormat.SparseCSR``.

   .. attribute:: rows

      Number of rows.

   .. attribute:: cols

      Number of columns.

   .. attribute:: numel

      Total number of elements (``rows × cols``).

   .. attribute:: nnz

      Number of non-zero entries (SparseCSR only; equals ``numel`` for Dense).

   .. attribute:: memory_footprint_bytes

      Actual memory consumption in bytes.


Operations
----------

.. function:: matmul(A, B)

   Dense matrix multiply: ``C = A × B``.

   :param A: Dense ``Tensor`` of shape ``(M, K)``
   :param B: Dense ``Tensor`` of shape ``(K, N)``
   :returns: Dense ``Tensor`` of shape ``(M, N)``

.. function:: spmm(A, B)

   Sparse-Dense matrix multiply: ``C = A × B``.

   :param A: SparseCSR ``Tensor`` of shape ``(M, K)``
   :param B: Dense ``Tensor`` of shape ``(K, N)``
   :returns: Dense ``Tensor`` of shape ``(M, N)``


Activations (In-Place)
----------------------

All activation functions modify the tensor in-place and return ``None``.

.. function:: relu_inplace(X)

   ``f(x) = max(0, x)``

.. function:: leaky_relu_inplace(X, alpha=0.01)

   ``f(x) = x if x >= 0, else alpha * x``

.. function:: elu_inplace(X, alpha=1.0)

   ``f(x) = x if x >= 0, else alpha * (exp(x) - 1)``

.. function:: sigmoid_inplace(X)

   ``f(x) = 1 / (1 + exp(-x))``

.. function:: tanh_inplace(X)

   ``f(x) = tanh(x)``

.. function:: gelu_inplace(X)

   ``f(x) = x * Φ(x)`` (tanh approximation)

.. function:: softmax_inplace(X)

   Row-wise softmax. Each row sums to 1.0.

.. function:: log_softmax_inplace(X)

   Row-wise log-softmax. Numerically stable.

.. function:: add_bias(X, bias)

   Broadcasting bias addition: ``X[i][j] += bias[0][j]``.


Graph Utilities
---------------

.. function:: add_self_loops(A)

   Add identity matrix to sparse adjacency: ``Ã = A + I``.

   :param A: Square SparseCSR ``Tensor``
   :returns: New SparseCSR ``Tensor``

.. function:: gcn_norm(A)

   Symmetric GCN normalization: ``D̃^{-1/2} (A + I) D̃^{-1/2}``.

   :param A: Square SparseCSR ``Tensor`` (without self-loops)
   :returns: Normalized SparseCSR ``Tensor``

.. function:: edge_softmax(A)

   Sparse row-wise softmax over CSR values.

.. function:: sage_max_aggregate(A, H)

   Element-wise max pooling over neighbors.


Layers
------

.. class:: GCNLayer(in_features, out_features, use_bias=True, activation=Activation.RELU)

   Graph Convolutional Network layer (Kipf & Welling, ICLR 2017).

   Forward: ``H' = σ(Â_norm · (H · W) + b)``

   .. method:: set_weight(w)

      Set weight matrix ``W`` of shape ``(in_features, out_features)``.

   .. method:: set_bias(b)

      Set bias vector ``b`` of shape ``(1, out_features)``.

   .. method:: forward(A_norm, H)

      Run forward pass.

      :param A_norm: Pre-normalized adjacency (output of ``gcn_norm``)
      :param H: Node features of shape ``(N, in_features)``
      :returns: Output features of shape ``(N, out_features)``


.. class:: SAGELayer(in_features, out_features, aggregator=SAGELayer.Aggregator.Mean, use_bias=True, activation=Activation.RELU)

   GraphSAGE layer (Hamilton et al., NeurIPS 2017).

   Forward: ``h' = σ(W_neigh · AGG(neighbors) + W_self · h + b)``

   .. class:: Aggregator

      .. attribute:: Mean
      .. attribute:: Max

   .. method:: set_weight_neigh(w)
   .. method:: set_weight_self(w)
   .. method:: set_bias(b)
   .. method:: forward(A, H)


.. class:: GATLayer(in_features, out_features, negative_slope=0.2, use_bias=True, activation=Activation.NONE)

   Graph Attention Network layer (Veličković et al., ICLR 2018).

   Uses fused SpSDDMM + edge_softmax + SpMM kernel for memory efficiency.

   .. method:: set_weight(w)
   .. method:: set_attn_left(a)
   .. method:: set_attn_right(a)
   .. method:: set_bias(b)
   .. method:: forward(A, H)


Model
-----

.. class:: Model

   Dynamic GNN execution graph supporting heterogeneous layer sequences.

   .. class:: InterActivation

      Activation applied between layers.

      .. attribute:: NONE
      .. attribute:: RELU
      .. attribute:: ELU

   .. method:: add_gcn_layer(in_f, out_f, bias=True, act=Activation.RELU, post=InterActivation.NONE)

      Add a GCN layer. Returns layer index.

   .. method:: add_sage_layer(in_f, out_f, agg=SAGELayer.Aggregator.Mean, bias=True, act=Activation.RELU, post=InterActivation.NONE)

      Add a GraphSAGE layer. Returns layer index.

   .. method:: add_gat_layer(in_f, out_f, num_heads=1, concat=True, neg_slope=0.2, bias=True, act=Activation.NONE, post=InterActivation.NONE)

      Add a GAT layer. Returns layer index.

   .. method:: load_weights(path)

      Load weights from a TGNN binary file.

   .. method:: forward(adjacency, features)

      Run the full execution graph.

      :param adjacency: Raw SparseCSR adjacency (no self-loops)
      :param features: Dense node features
      :returns: Dense output logits


I/O Functions
-------------

.. function:: load_cora_binary(path)

   Load Cora graph data from binary file.

   :returns: ``CoraData`` object with ``adjacency``, ``features``, ``labels``, masks

.. function:: load_weight_file(path)

   Load TGNN binary weight file.

   :returns: ``WeightFile`` object with ``test_accuracy`` and ``tensors`` dict

.. class:: GraphLoader

   CSV-based graph loading utilities.

   .. staticmethod:: parse_edges(path)
   .. staticmethod:: parse_features(path)
   .. staticmethod:: edge_list_to_csr(edges, num_nodes)
   .. staticmethod:: load(edges_path, features_path)
