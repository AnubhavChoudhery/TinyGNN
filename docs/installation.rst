Installation
============

From PyPI (recommended)
-----------------------

.. code-block:: bash

   pip install tinygnn

This installs the pre-built wheel with the C++ extension included.

From Source
-----------

Requirements:

- Python 3.8+
- C++17 compiler (GCC 8+, Clang 7+, or MSVC 2019+)
- pybind11 ≥ 2.11

.. code-block:: bash

   git clone https://github.com/JaiAnshSB/TinyGNN.git
   cd TinyGNN
   pip install -e ".[dev]"

This builds the C++ extension in-place and installs test dependencies.

C++ Only (No Python)
--------------------

If you only need the C++ library:

.. code-block:: bash

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --parallel
   ctest --test-dir build --output-on-failure

Optional Dependencies
---------------------

.. list-table::
   :header-rows: 1

   * - Package
     - Purpose
     - Install
   * - ``pytest``
     - Run Python tests
     - ``pip install pytest``
   * - ``scipy``
     - SciPy CSR interop
     - ``pip install scipy``
   * - ``torch``
     - PyTorch interop
     - ``pip install torch``
   * - ``torch_geometric``
     - PyG validation
     - ``pip install torch_geometric``
   * - OpenMP
     - Thread parallelism
     - Usually bundled with GCC; ``brew install libomp`` on macOS
   * - AVX2 CPU
     - SIMD acceleration
     - Intel Haswell+ / AMD Zen+ (scalar fallback provided)
