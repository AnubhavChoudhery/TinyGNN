.. =============================================================================
..  TinyGNN Documentation — Main Index
.. =============================================================================

TinyGNN Documentation
=====================

**TinyGNN** is a zero-dependency C++17 inference engine for Sparse Graph Neural
Networks, with Python bindings via pybind11.

It provides complete GCN, GraphSAGE, and GAT inference with:

- **Zero external dependencies** — no Eigen, no BLAS, no Boost
- **Sparse-native CSR storage** — memory-efficient graph representation
- **AVX2 + OpenMP parallelism** — hardware-tuned SIMD and multi-threading
- **Operator fusion** — fused GAT and SAGE kernels for reduced memory/latency
- **Python interop** — seamless NumPy, SciPy, and PyTorch integration

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   python_api
   cpp_api
   benchmarks
   contributing


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
