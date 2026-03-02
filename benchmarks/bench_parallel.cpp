// ============================================================================
//  TinyGNN — Phase 8: Parallel Benchmark Harness
//  benchmarks/bench_parallel.cpp
//
//  Benchmarks matmul, spmm, and softmax at varying thread counts to
//  demonstrate OpenMP + AVX2 scaling.  Outputs CSV to stdout for plotting.
//
//  Usage:
//    ./bench_parallel                     # default: 1,2,4,8 threads
//    ./bench_parallel --csv results.csv   # write CSV to file
//
//  Build:
//    g++ -std=c++17 -O2 -fopenmp -mavx2 -mfma -Iinclude
//        src/tensor.cpp src/ops.cpp benchmarks/bench_parallel.cpp
//        -o build/bench_parallel
// ============================================================================

#include "tinygnn/tensor.hpp"
#include "tinygnn/ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace tinygnn;
using Clock = std::chrono::high_resolution_clock;

// ============================================================================
//  Helpers
// ============================================================================

/// Simple pseudo-random dense tensor (deterministic seed)
static Tensor random_dense(std::size_t rows, std::size_t cols, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(rows * cols);
    for (auto& v : data) v = dist(rng);
    return Tensor::dense(rows, cols, std::move(data));
}

/// Build a synthetic sparse CSR matrix resembling a power-law graph.
///   N nodes, avg_degree average edges per node (Poisson sampled).
static Tensor random_sparse(std::size_t N, double avg_degree, unsigned seed) {
    std::mt19937 rng(seed);
    // Poisson-like: geometric distribution gives power-law-ish tail
    std::poisson_distribution<int> deg_dist(avg_degree);
    std::uniform_int_distribution<int32_t> col_dist(0, static_cast<int32_t>(N - 1));

    std::vector<int32_t> row_ptr(N + 1, 0);
    std::vector<int32_t> col_ind;
    std::vector<float>   values;

    for (std::size_t i = 0; i < N; ++i) {
        int degree = std::max(1, deg_dist(rng));  // at least 1 neighbor
        row_ptr[i] = static_cast<int32_t>(col_ind.size());

        // Generate random neighbors (allow duplicates for simplicity;
        // sort + deduplicate for valid CSR)
        std::vector<int32_t> neighbors;
        neighbors.reserve(static_cast<std::size_t>(degree));
        for (int d = 0; d < degree; ++d) {
            neighbors.push_back(col_dist(rng));
        }
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                        neighbors.end());

        for (auto c : neighbors) {
            col_ind.push_back(c);
            values.push_back(1.0f);  // unweighted
        }
    }
    row_ptr[N] = static_cast<int32_t>(col_ind.size());

    return Tensor::sparse_csr(N, N, std::move(row_ptr),
                               std::move(col_ind), std::move(values));
}

/// Time a callable (in seconds), averaging over `reps` repetitions.
/// Returns median of `reps` timings (less noisy than mean).
template <typename Fn>
static double bench(Fn&& fn, int reps) {
    // Warm-up
    fn();

    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(reps));
    for (int r = 0; r < reps; ++r) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        times.push_back(
            std::chrono::duration<double>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    return times[static_cast<std::size_t>(reps / 2)];  // median
}

// ============================================================================
//  Benchmark configurations
// ============================================================================
struct BenchConfig {
    const char* name;
    std::size_t N;           // matrix dimension (nodes)
    std::size_t F;           // features
    double avg_deg;          // average degree (for spmm)
    int reps;                // repetitions
};

static const BenchConfig configs[] = {
    // Cora-scale
    {"cora",     2708,    1433,  3.9,  20},
    // Medium: ~10K nodes, 128 features
    {"medium",  10000,     128, 10.0,  10},
    // Large: ~50K nodes, 256 features
    {"large",   50000,     256, 15.0,   5},
};

// ============================================================================
//  Main
// ============================================================================
int main(int argc, char* argv[]) {
    // Parse --csv flag
    std::string csv_path;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        }
    }

    // Thread counts to test — cover full range up to hw concurrency
    const int max_hw = []() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }();
    std::vector<int> thread_counts;
    for (int t : {1, 2, 4, 8, 16, 20, 32}) {
        if (t <= max_hw) thread_counts.push_back(t);
    }
    if (thread_counts.empty() || thread_counts.back() != max_hw)
        thread_counts.push_back(max_hw);

    // CSV header
    std::string csv;
    csv += "kernel,config,nodes,features,avg_degree,threads,time_s,speedup\n";

    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  TinyGNN Phase 8 — Parallel Benchmark               ║\n";
    std::cout << "╠══════════════════════════════════════════════════════╣\n";

#ifdef _OPENMP
    std::cout << "║  OpenMP:  enabled  (max threads: "
              << omp_get_max_threads() << ")" << std::string(
                  20 - std::to_string(omp_get_max_threads()).size(), ' ')
              << "║\n";
#else
    std::cout << "║  OpenMP:  disabled                                   ║\n";
#endif
#ifdef __AVX2__
    std::cout << "║  AVX2:    enabled                                    ║\n";
#else
    std::cout << "║  AVX2:    disabled                                   ║\n";
#endif
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    for (const auto& cfg : configs) {
        std::cout << "── Config: " << cfg.name
                  << " (N=" << cfg.N << ", F=" << cfg.F
                  << ", deg=" << cfg.avg_deg << ") ──\n";

        // Pre-build matrices
        auto A_sparse = random_sparse(cfg.N, cfg.avg_deg, 42);
        auto H_dense  = random_dense(cfg.N, cfg.F, 123);
        auto W_dense  = random_dense(cfg.F, cfg.F, 456);
        auto S_dense  = random_dense(cfg.N, cfg.F, 789);  // for softmax

        // Store 1-thread baseline times for speedup calculation
        double base_spmm  = 0.0;
        double base_matmul = 0.0;
        double base_softmax = 0.0;

        for (int t : thread_counts) {
#ifdef _OPENMP
            omp_set_num_threads(t);
#else
            if (t > 1) continue;  // no OpenMP — only run 1-thread
#endif

            // ── SpMM ──
            double spmm_time = bench([&]() {
                auto C = spmm(A_sparse, H_dense);
                (void)C;
            }, cfg.reps);

            // ── matmul ──
            // For large N, do a smaller matmul to keep time reasonable
            std::size_t mm_rows = std::min(cfg.N, (std::size_t)2000);
            auto mm_A = random_dense(mm_rows, cfg.F, 111);
            double matmul_time = bench([&]() {
                auto C = matmul(mm_A, W_dense);
                (void)C;
            }, cfg.reps);

            // ── Softmax ──
            double softmax_time = bench([&]() {
                auto Scopy = Tensor::dense(cfg.N, cfg.F, S_dense.data());
                softmax_inplace(Scopy);
            }, cfg.reps);

            if (t == 1) {
                base_spmm    = spmm_time;
                base_matmul  = matmul_time;
                base_softmax = softmax_time;
            }

            double sp_spmm    = base_spmm    / spmm_time;
            double sp_matmul  = base_matmul  / matmul_time;
            double sp_softmax = base_softmax / softmax_time;

            printf("  threads=%d  spmm=%.4fs (%.2fx)  matmul=%.4fs (%.2fx)  "
                   "softmax=%.4fs (%.2fx)\n",
                   t, spmm_time, sp_spmm, matmul_time, sp_matmul,
                   softmax_time, sp_softmax);

            // Append CSV rows
            auto row = [&](const char* kernel, double time_s, double speedup) {
                csv += std::string(kernel) + "," + cfg.name + ","
                     + std::to_string(cfg.N) + "," + std::to_string(cfg.F) + ","
                     + std::to_string(cfg.avg_deg) + ","
                     + std::to_string(t) + ","
                     + std::to_string(time_s) + ","
                     + std::to_string(speedup) + "\n";
            };
            row("spmm",    spmm_time,    sp_spmm);
            row("matmul",  matmul_time,  sp_matmul);
            row("softmax", softmax_time, sp_softmax);
        }
        std::cout << "\n";
    }

    // Write CSV
    if (!csv_path.empty()) {
        std::ofstream f(csv_path);
        f << csv;
        std::cout << "CSV written to: " << csv_path << "\n";
    } else {
        std::cout << "── Raw CSV (use --csv <file> to save) ──\n" << csv;
    }

    return 0;
}
