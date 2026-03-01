// ============================================================================
//  TinyGNN — Real-Dataset Benchmark  (Phase 11)
//  benchmarks/bench_datasets.cpp
//
//  Benchmarks GCN, GraphSAGE (fused vs unfused), and GAT (fused vs unfused)
//  on the actual Cora and Reddit datasets loaded from CSV.
//
//  Outputs CSV to stdout (or --csv FILE) for consumption by
//  scripts/plot_dataset_bench.py.
//
//  Build (WSL):
//    g++ -std=c++17 -O2 -fopenmp -mavx2 -mfma -Iinclude \
//        benchmarks/bench_datasets.cpp \
//        src/tensor.cpp src/graph_loader.cpp src/ops.cpp \
//        src/layers.cpp src/model.cpp \
//        -o build/bench/bench_datasets
//
//  Run:
//    OMP_NUM_THREADS=8 ./build/bench/bench_datasets \
//        --csv build/bench/dataset_results.csv
// ============================================================================

#include "tinygnn/graph_loader.hpp"
#include "tinygnn/layers.hpp"
#include "tinygnn/ops.hpp"
#include "tinygnn/tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace tinygnn;
using Clock = std::chrono::high_resolution_clock;

// ── Timing ──────────────────────────────────────────────────────────────────
static double elapsed_us(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

/// Median of `reps` measurements (µs)
template <typename Fn>
static double bench_us(Fn&& fn, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) { fn(); }
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(reps));
    for (int i = 0; i < reps; ++i) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        times.push_back(elapsed_us(t0, t1));
    }
    std::sort(times.begin(), times.end());
    return times[static_cast<std::size_t>(reps / 2)];
}

// ── Random dense tensor ─────────────────────────────────────────────────────
static Tensor random_dense(std::size_t rows, std::size_t cols,
                           std::mt19937& rng) {
    Tensor t = Tensor::dense(rows, cols);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    for (auto& v : t.data()) v = dist(rng);
    return t;
}

// ============================================================================
//  Unfused GAT forward — reconstructs original Phase 6 algorithm
// ============================================================================
static Tensor gat_forward_unfused(const Tensor& A, const Tensor& H,
                                  const Tensor& W, const Tensor& attn_l,
                                  const Tensor& attn_r, const Tensor& bias,
                                  float neg_slope) {
    const std::size_t N = H.rows();
    const std::size_t F_out = W.cols();
    const auto& rp = A.row_ptr();
    const auto& ci = A.col_ind();

    Tensor Wh = matmul(H, W);

    const float* wh_data = Wh.data().data();
    const float* al = attn_l.data().data();
    const float* ar = attn_r.data().data();

    std::vector<float> src(N, 0.0f), dst(N, 0.0f);
    for (std::size_t i = 0; i < N; ++i) {
        float s = 0.0f, d = 0.0f;
        const float* whi = wh_data + i * F_out;
        for (std::size_t f = 0; f < F_out; ++f) {
            s += al[f] * whi[f];
            d += ar[f] * whi[f];
        }
        src[i] = s;
        dst[i] = d;
    }

    const std::size_t nnz = ci.size();
    std::vector<float> edge_logits(nnz);
    for (std::size_t i = 0; i < N; ++i) {
        const float si = src[i];
        for (int32_t nz = rp[i]; nz < rp[i + 1]; ++nz) {
            const auto j = static_cast<std::size_t>(ci[nz]);
            float e = si + dst[j];
            edge_logits[nz] = (e >= 0.0f) ? e : neg_slope * e;
        }
    }

    std::vector<int32_t> rp_copy(rp.begin(), rp.end());
    std::vector<int32_t> ci_copy(ci.begin(), ci.end());
    Tensor attn_csr = Tensor::sparse_csr(
        N, N, std::move(rp_copy), std::move(ci_copy),
        std::move(edge_logits));

    Tensor alpha = edge_softmax(attn_csr);
    Tensor out = spmm(alpha, Wh);
    add_bias(out, bias);
    return out;
}

// ============================================================================
//  Unfused SAGE forward (Mean)
// ============================================================================
static Tensor sage_forward_unfused(const Tensor& A, const Tensor& H,
                                   const Tensor& W_neigh,
                                   const Tensor& W_self,
                                   const Tensor& bias) {
    const std::size_t N = H.rows();
    const auto& rp = A.row_ptr();

    Tensor agg = spmm(A, H);
    float* agg_d = agg.data().data();
    const std::size_t F = H.cols();

    for (std::size_t i = 0; i < N; ++i) {
        const float deg = static_cast<float>(rp[i + 1] - rp[i]);
        if (deg > 0.0f) {
            const float inv = 1.0f / deg;
            float* row = agg_d + i * F;
            for (std::size_t f = 0; f < F; ++f) row[f] *= inv;
        }
    }

    Tensor h_neigh = matmul(agg, W_neigh);
    Tensor h_self = matmul(H, W_self);

    const std::size_t total = N * W_neigh.cols();
    float* hn = h_neigh.data().data();
    const float* hs = h_self.data().data();
    for (std::size_t i = 0; i < total; ++i) hn[i] += hs[i];

    add_bias(h_neigh, bias);
    return h_neigh;
}

// ============================================================================
//  Dataset descriptor
// ============================================================================
struct Dataset {
    std::string name;
    std::string edges_path;
    std::string features_path;
};

// ============================================================================
//  Main
// ============================================================================
int main(int argc, char* argv[]) {
    // ── Parse args ──────────────────────────────────────────────────────────
    std::string csv_path;
    std::string data_dir = "datasets";
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--csv" && i + 1 < argc) csv_path = argv[++i];
        if (arg == "--data-dir" && i + 1 < argc) data_dir = argv[++i];
    }

    std::cout << "╔═══════════════════════════════════════════════════════════╗\n"
              << "║  TinyGNN — Real-Dataset Benchmark  (Phase 11)            ║\n"
              << "╚═══════════════════════════════════════════════════════════╝\n\n";

#ifdef _OPENMP
    std::cout << "  OpenMP threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "  OpenMP: disabled\n";
#endif
#ifdef __AVX2__
    std::cout << "  AVX2+FMA: enabled\n\n";
#else
    std::cout << "  AVX2: disabled\n\n";
#endif

    // ── Datasets ────────────────────────────────────────────────────────────
    Dataset datasets[] = {
        {"cora",   data_dir + "/cora/edges.csv",   data_dir + "/cora/node_features.csv"},
        {"reddit", data_dir + "/reddit/edges.csv", data_dir + "/reddit/node_features.csv"},
    };

    // ── CSV accumulator ─────────────────────────────────────────────────────
    std::string csv;
    csv += "dataset,num_nodes,num_edges,num_features,layer,variant,F_out,time_us,speedup\n";

    for (auto& ds : datasets) {
        std::cout << "━━━━ Loading " << ds.name << " ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

        GraphData g;
        try {
            g = GraphLoader::load(ds.edges_path, ds.features_path);
        } catch (const std::exception& e) {
            std::cerr << "  ⚠ Failed to load " << ds.name << ": " << e.what() << "\n";
            std::cerr << "  Skipping.\n\n";
            continue;
        }

        std::cout << "  Nodes: " << g.num_nodes
                  << "  Edges: " << g.num_edges
                  << "  Features: " << g.num_features << "\n";

        const std::size_t N = g.num_nodes;
        const std::size_t F_in = g.num_features;
        std::mt19937 rng(42);

        // Reduce feature dimension for Reddit to keep benchmarks manageable
        // (Reddit has 602 features, Cora has 1433)
        const std::size_t F_out_gat  = (N > 50000) ? 8  : 8;
        const std::size_t F_out_sage = (N > 50000) ? 32 : 128;
        const std::size_t F_out_gcn  = (N > 50000) ? 32 : 128;

        // Benchmark parameters — fewer iters for large graphs
        const int warmup = (N > 50000) ? 1 : 2;
        const int iters  = (N > 50000) ? 3 : 5;

        // ── Pre-compute GCN norm ────────────────────────────────────────────
        Tensor A_norm = gcn_norm(g.adjacency);

        // ================================================================
        //  GCN Benchmark
        // ================================================================
        {
            std::cout << "\n  ── GCN (F_in=" << F_in << " → F_out=" << F_out_gcn << ") ──\n";

            GCNLayer layer(F_in, F_out_gcn, true, Activation::ReLU);
            layer.set_weight(random_dense(F_in, F_out_gcn, rng));
            layer.set_bias(random_dense(1, F_out_gcn, rng));

            double gcn_time = bench_us([&]() {
                auto out = layer.forward(A_norm, g.node_features);
                (void)out;
            }, warmup, iters);

            std::cout << "    GCN forward: " << std::fixed << std::setprecision(0)
                      << gcn_time << " µs\n";

            csv += ds.name + "," + std::to_string(N) + ","
                 + std::to_string(g.num_edges) + ","
                 + std::to_string(F_in) + ","
                 + "GCN,fused," + std::to_string(F_out_gcn)
                 + "," + std::to_string(gcn_time) + ",1.0\n";
        }

        // ================================================================
        //  GAT Benchmark — fused vs unfused
        // ================================================================
        {
            std::cout << "\n  ── GAT (F_in=" << F_in << " → F_out=" << F_out_gat << ") ──\n";

            GATLayer layer(F_in, F_out_gat, 0.2f, true, Activation::None);
            layer.set_weight(random_dense(F_in, F_out_gat, rng));
            layer.set_attn_left(random_dense(1, F_out_gat, rng));
            layer.set_attn_right(random_dense(1, F_out_gat, rng));
            layer.set_bias(random_dense(1, F_out_gat, rng));

            // Extract weights for unfused
            Tensor W = Tensor::dense(F_in, F_out_gat);
            std::copy(layer.weight().data().begin(),
                      layer.weight().data().end(), W.data().begin());
            Tensor al = Tensor::dense(1, F_out_gat);
            std::copy(layer.attn_left().data().begin(),
                      layer.attn_left().data().end(), al.data().begin());
            Tensor ar = Tensor::dense(1, F_out_gat);
            std::copy(layer.attn_right().data().begin(),
                      layer.attn_right().data().end(), ar.data().begin());
            Tensor b = Tensor::dense(1, F_out_gat);
            std::copy(layer.bias().data().begin(),
                      layer.bias().data().end(), b.data().begin());

            // Fused (production forward)
            double fused_time = bench_us([&]() {
                auto out = layer.forward(g.adjacency, g.node_features);
                (void)out;
            }, warmup, iters);

            // Unfused
            double unfused_time = bench_us([&]() {
                auto out = gat_forward_unfused(g.adjacency, g.node_features,
                                              W, al, ar, b, 0.2f);
                (void)out;
            }, warmup, iters);

            double speedup = unfused_time / fused_time;

            std::cout << "    Unfused: " << std::fixed << std::setprecision(0)
                      << unfused_time << " µs\n"
                      << "    Fused:   " << fused_time << " µs\n"
                      << "    Speedup: " << std::setprecision(2) << speedup << "×\n";

            csv += ds.name + "," + std::to_string(N) + ","
                 + std::to_string(g.num_edges) + ","
                 + std::to_string(F_in) + ","
                 + "GAT,unfused," + std::to_string(F_out_gat)
                 + "," + std::to_string(unfused_time) + ",1.0\n";
            csv += ds.name + "," + std::to_string(N) + ","
                 + std::to_string(g.num_edges) + ","
                 + std::to_string(F_in) + ","
                 + "GAT,fused," + std::to_string(F_out_gat)
                 + "," + std::to_string(fused_time) + ","
                 + std::to_string(speedup) + "\n";
        }

        // ================================================================
        //  SAGE Benchmark — fused vs unfused
        // ================================================================
        {
            std::cout << "\n  ── SAGE (F_in=" << F_in << " → F_out=" << F_out_sage << ") ──\n";

            SAGELayer layer(F_in, F_out_sage, SAGELayer::Aggregator::Mean,
                            true, Activation::ReLU);
            layer.set_weight_neigh(random_dense(F_in, F_out_sage, rng));
            layer.set_weight_self(random_dense(F_in, F_out_sage, rng));
            layer.set_bias(random_dense(1, F_out_sage, rng));

            // Extract weights for unfused
            Tensor Wn = Tensor::dense(F_in, F_out_sage);
            std::copy(layer.weight_neigh().data().begin(),
                      layer.weight_neigh().data().end(), Wn.data().begin());
            Tensor Ws = Tensor::dense(F_in, F_out_sage);
            std::copy(layer.weight_self().data().begin(),
                      layer.weight_self().data().end(), Ws.data().begin());
            Tensor b = Tensor::dense(1, F_out_sage);
            std::copy(layer.bias().data().begin(),
                      layer.bias().data().end(), b.data().begin());

            // Fused
            double fused_time = bench_us([&]() {
                auto out = layer.forward(g.adjacency, g.node_features);
                (void)out;
            }, warmup, iters);

            // Unfused
            double unfused_time = bench_us([&]() {
                auto out = sage_forward_unfused(g.adjacency, g.node_features,
                                               Wn, Ws, b);
                (void)out;
            }, warmup, iters);

            double speedup = unfused_time / fused_time;

            std::cout << "    Unfused: " << std::fixed << std::setprecision(0)
                      << unfused_time << " µs\n"
                      << "    Fused:   " << fused_time << " µs\n"
                      << "    Speedup: " << std::setprecision(2) << speedup << "×\n";

            csv += ds.name + "," + std::to_string(N) + ","
                 + std::to_string(g.num_edges) + ","
                 + std::to_string(F_in) + ","
                 + "SAGE,unfused," + std::to_string(F_out_sage)
                 + "," + std::to_string(unfused_time) + ",1.0\n";
            csv += ds.name + "," + std::to_string(N) + ","
                 + std::to_string(g.num_edges) + ","
                 + std::to_string(F_in) + ","
                 + "SAGE,fused," + std::to_string(F_out_sage)
                 + "," + std::to_string(fused_time) + ","
                 + std::to_string(speedup) + "\n";
        }

        std::cout << "\n";
    }

    // ── Write CSV ───────────────────────────────────────────────────────────
    if (!csv_path.empty()) {
        std::ofstream ofs(csv_path);
        if (ofs.is_open()) {
            ofs << csv;
            std::cout << "CSV written to: " << csv_path << "\n";
        } else {
            std::cerr << "Error: could not write CSV to " << csv_path << "\n";
        }
    } else {
        std::cout << "\n── CSV Output ──────────────────────────────────────────\n";
        std::cout << csv;
    }

    std::cout << "\n  ✓ Dataset benchmark complete.\n";
    return 0;
}
