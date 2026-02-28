C++ API Reference
=================

The C++ API is documented via Doxygen. Below are cross-references to the
main classes and functions generated from the annotated C++ headers.

Namespace: ``tinygnn``
----------------------

All TinyGNN C++ types and functions live in the ``tinygnn`` namespace.


Core Types
----------

.. doxygenclass:: tinygnn::Tensor
   :members:
   :undoc-members:

.. doxygenenum:: tinygnn::StorageFormat

.. doxygenenum:: tinygnn::Activation


Compute Operations (ops.hpp)
----------------------------

.. doxygenfunction:: tinygnn::matmul
.. doxygenfunction:: tinygnn::spmm
.. doxygenfunction:: tinygnn::relu_inplace
.. doxygenfunction:: tinygnn::leaky_relu_inplace
.. doxygenfunction:: tinygnn::elu_inplace
.. doxygenfunction:: tinygnn::sigmoid_inplace
.. doxygenfunction:: tinygnn::tanh_inplace
.. doxygenfunction:: tinygnn::gelu_inplace
.. doxygenfunction:: tinygnn::softmax_inplace
.. doxygenfunction:: tinygnn::log_softmax_inplace
.. doxygenfunction:: tinygnn::add_bias


GNN Layers (layers.hpp)
-----------------------

.. doxygenfunction:: tinygnn::add_self_loops
.. doxygenfunction:: tinygnn::gcn_norm
.. doxygenfunction:: tinygnn::edge_softmax
.. doxygenfunction:: tinygnn::sage_max_aggregate

.. doxygenstruct:: tinygnn::GCNLayer
   :members:
   :undoc-members:

.. doxygenstruct:: tinygnn::SAGELayer
   :members:
   :undoc-members:

.. doxygenstruct:: tinygnn::GATLayer
   :members:
   :undoc-members:


Model & I/O (model.hpp)
-----------------------

.. doxygenclass:: tinygnn::Model
   :members:
   :undoc-members:

.. doxygenstruct:: tinygnn::CoraData
   :members:

.. doxygenstruct:: tinygnn::WeightFile
   :members:

.. doxygenfunction:: tinygnn::load_cora_binary
.. doxygenfunction:: tinygnn::load_weight_file


Graph Loader (graph_loader.hpp)
-------------------------------

.. doxygenclass:: tinygnn::GraphLoader
   :members:
   :undoc-members:

.. doxygenstruct:: tinygnn::GraphData
   :members:
