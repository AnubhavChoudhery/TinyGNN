Benchmarks
==========

Thread-Scaling (OpenMP + AVX2)
------------------------------

Tested on Intel Core 9 270H (20 threads), GCC 13.3, Ubuntu 24.04 (WSL2).

Cora-scale (2,708 nodes × 1,433 features):

.. list-table::
   :header-rows: 1

   * - Threads
     - SpMM
     - Speedup
     - GEMM
     - Speedup
   * - 1
     - 6.4 ms
     - 1.00×
     - 500 ms
     - 1.00×
   * - 4
     - 3.4 ms
     - 1.85×
     - 167 ms
     - 2.99×
   * - 8
     - 2.6 ms
     - 2.49×
     - 114 ms
     - 4.38×


Operator Fusion
---------------

GAT: Fused SpSDDMM + edge_softmax + SpMM

.. list-table::
   :header-rows: 1

   * - Configuration
     - Unfused
     - Fused
     - Speedup
     - Memory Ratio
   * - Cora-like
     - 4,867 µs
     - 5,391 µs
     - 0.90×
     - 2.8×
   * - Large (10K nodes, deg≈20)
     - 5,717 µs
     - 3,534 µs
     - 1.62×
     - 2.6×
   * - XL (20K nodes, deg≈30)
     - 12,323 µs
     - 4,726 µs
     - 2.61×
     - 5.6×

SAGE: Fused Aggregation + Dual-Matmul

.. list-table::
   :header-rows: 1

   * - Configuration
     - Unfused
     - Fused
     - Speedup
     - Memory Ratio
   * - Cora-like (F=1433)
     - 60,259 µs
     - 20,209 µs
     - 2.98×
     - 13.1×
   * - Medium (5K nodes)
     - 6,107 µs
     - 1,858 µs
     - 3.29×
     - 4.0×


Valgrind Massif Profiling
-------------------------

GAT heap usage (N=10000, F_in=64, F_out=32, deg≈20):

- **Unfused peak**: 10.35 MB
- **Fused peak**: 7.08 MB
- **Savings**: 3.44 MB (31.6% reduction)
