// ============================================================================
//  TinyGNN — Operator Fusion Benchmark  (Phase 9)
//  benchmarks/bench_fusion.cpp
//
//  Compares fused vs. unfused GAT and SAGE forward passes:
//    • Runtime (wall-clock, microseconds)
//    • Peak heap allocations (bytes allocated during forward pass)
//
//  The "unfused" path is reconstructed inline here (same algorithm as the
//  original Phase 6 implementation), while the "fused" path calls the
//  production forward() methods which now use fused kernels.
//
//  Build:
//    g++ -std=c++17 -O2 -fopenmp -mavx2 -mfma -Iinclude \
//        -o bench_fusion benchmarks/bench_fusion.cpp \
//        src/tensor.cpp src/graph_loader.cpp src/ops.cpp \
//        src/layers.cpp src/model.cpp
//
//  Run:
//    OMP_NUM_THREADS=8 ./bench_fusion
// ============================================================================

#include "tinygnn/layers.hpp"
#include "tinygnn/ops.hpp"
#include "tinygnn/tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

// ── Timing helper ───────────────────────────────────────────────────────────
using Clock = std::chrono::high_resolution_clock;

static double elapsed_us(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

// ── Random dense tensor ─────────────────────────────────────────────────────
static Tensor random_dense(std::size_t rows, std::size_t cols,
                           std::mt19937& rng) {
    Tensor t = Tensor::dense(rows, cols);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : t.data()) v = dist(rng);
    return t;
}

// ── Build random sparse adjacency (with self-loops) ─────────────────────────
static Tensor random_adj_with_self_loops(std::size_t N, std::size_t avg_deg,
                                         std::mt19937& rng) {
    // Build edge list
    std::vector<std::vector<int32_t>> adj(N);
    std::uniform_int_distribution<std::size_t> node_dist(0, N - 1);

    for (std::size_t i = 0; i < N; ++i) {
        adj[i].push_back(static_cast<int32_t>(i));  // self-loop
        for (std::size_t d = 0; d < avg_deg; ++d) {
            auto j = static_cast<int32_t>(node_dist(rng));
            adj[i].push_back(j);
        }
        // Sort and deduplicate
        std::sort(adj[i].begin(), adj[i].end());
        adj[i].erase(std::unique(adj[i].begin(), adj[i].end()),
                      adj[i].end());
    }

    // Build CSR
    std::vector<int32_t> rp(N + 1, 0);
    std::vector<int32_t> ci;
    std::vector<float> vals;

    for (std::size_t i = 0; i < N; ++i) {
        rp[i] = static_cast<int32_t>(ci.size());
        for (int32_t j : adj[i]) {
            ci.push_back(j);
            vals.push_back(1.0f);
        }
    }
    rp[N] = static_cast<int32_t>(ci.size());

    return Tensor::sparse_csr(N, N, std::move(rp), std::move(ci),
                              std::move(vals));
}

// ============================================================================
//  Unfused GAT forward — reconstructs the original Phase 6 algorithm
// ============================================================================
static Tensor gat_forward_unfused(const Tensor& A, const Tensor& H,
                                  const Tensor& W, const Tensor& attn_l,
                                  const Tensor& attn_r, const Tensor& bias,
                                  float neg_slope) {
    const std::size_t N = H.rows();
    const std::size_t F_out = W.cols();
    const auto& rp = A.row_ptr();
    const auto& ci = A.col_ind();

    // Step 1: Wh = H * W
    Tensor Wh = matmul(H, W);

    // Step 2: per-node attention scores
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

    // Step 3: SpSDDMM — build edge logits (nnz allocation)
    const std::size_t nnz = ci.size();
    std::vector<float> edge_logits(nnz);  // <-- KEY: nnz-sized alloc

    for (std::size_t i = 0; i < N; ++i) {
        const float si = src[i];
        for (int32_t nz = rp[i]; nz < rp[i + 1]; ++nz) {
            const auto j = static_cast<std::size_t>(ci[nz]);
            float e = si + dst[j];
            edge_logits[nz] = (e >= 0.0f) ? e : neg_slope * e;
        }
    }

    // Build attention CSR (copies rp, ci, edge_logits)
    std::vector<int32_t> rp_copy(rp.begin(), rp.end());
    std::vector<int32_t> ci_copy(ci.begin(), ci.end());
    Tensor attn_csr = Tensor::sparse_csr(
        N, N, std::move(rp_copy), std::move(ci_copy),
        std::move(edge_logits));

    // Step 4: edge_softmax (allocates another nnz-sized CSR)
    Tensor alpha = edge_softmax(attn_csr);

    // Step 5: spmm(alpha, Wh)
    Tensor out = spmm(alpha, Wh);

    // Bias
    add_bias(out, bias);

    return out;
}

// ============================================================================
//  Unfused SAGE forward (Mean) — reconstructs the original Phase 6 algorithm
// ============================================================================
static Tensor sage_forward_unfused(const Tensor& A, const Tensor& H,
                                   const Tensor& W_neigh,
                                   const Tensor& W_self,
                                   const Tensor& bias) {
    const std::size_t N = H.rows();
    const auto& rp = A.row_ptr();

    // Step 1: aggregate (spmm + degree normalize)
    Tensor agg = spmm(A, H);  // <-- KEY: N×F_in alloc
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

    // Step 2-3: separate matmuls
    Tensor h_neigh = matmul(agg, W_neigh);  // N×F_out
    Tensor h_self = matmul(H, W_self);      // N×F_out  <-- KEY: extra alloc

    // Step 4: element-wise add
    const std::size_t total = N * W_neigh.cols();
    float* hn = h_neigh.data().data();
    const float* hs = h_self.data().data();
    for (std::size_t i = 0; i < total; ++i) hn[i] += hs[i];

    // Bias
    add_bias(h_neigh, bias);

    return h_neigh;
}

// ============================================================================
//  Benchmark runner
// ============================================================================
struct BenchResult {
    double fused_us;
    double unfused_us;
    double speedup;
};

static BenchResult bench_gat(std::size_t N, std::size_t F_in,
                             std::size_t F_out, std::size_t avg_deg,
                             int warmup, int iters) {
    std::mt19937 rng(42);

    Tensor A = random_adj_with_self_loops(N, avg_deg, rng);
    Tensor H = random_dense(N, F_in, rng);

    GATLayer layer(F_in, F_out, 0.2f, true, Activation::None);
    layer.set_weight(random_dense(F_in, F_out, rng));
    layer.set_attn_left(random_dense(1, F_out, rng));
    layer.set_attn_right(random_dense(1, F_out, rng));
    layer.set_bias(random_dense(1, F_out, rng));

    // Extract weights for unfused path
    Tensor W = Tensor::dense(F_in, F_out);
    std::copy(layer.weight().data().begin(), layer.weight().data().end(),
              W.data().begin());
    Tensor al = Tensor::dense(1, F_out);
    std::copy(layer.attn_left().data().begin(),
              layer.attn_left().data().end(), al.data().begin());
    Tensor ar = Tensor::dense(1, F_out);
    std::copy(layer.attn_right().data().begin(),
              layer.attn_right().data().end(), ar.data().begin());
    Tensor b = Tensor::dense(1, F_out);
    std::copy(layer.bias().data().begin(), layer.bias().data().end(),
              b.data().begin());

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        auto out = layer.forward(A, H);
        (void)out;
    }

    // Fused timing
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        auto out = layer.forward(A, H);
        (void)out;
    }
    auto t1 = Clock::now();
    double fused = elapsed_us(t0, t1) / iters;

    // Unfused warmup
    for (int i = 0; i < warmup; ++i) {
        auto out = gat_forward_unfused(A, H, W, al, ar, b, 0.2f);
        (void)out;
    }

    // Unfused timing
    t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        auto out = gat_forward_unfused(A, H, W, al, ar, b, 0.2f);
        (void)out;
    }
    t1 = Clock::now();
    double unfused = elapsed_us(t0, t1) / iters;

    return {fused, unfused, unfused / fused};
}

static BenchResult bench_sage(std::size_t N, std::size_t F_in,
                              std::size_t F_out, std::size_t avg_deg,
                              int warmup, int iters) {
    std::mt19937 rng(42);

    Tensor A = random_adj_with_self_loops(N, avg_deg, rng);
    Tensor H = random_dense(N, F_in, rng);

    SAGELayer layer(F_in, F_out, SAGELayer::Aggregator::Mean, true,
                    Activation::ReLU);
    layer.set_weight_neigh(random_dense(F_in, F_out, rng));
    layer.set_weight_self(random_dense(F_in, F_out, rng));
    layer.set_bias(random_dense(1, F_out, rng));

    // Extract weights for unfused path
    Tensor Wn = Tensor::dense(F_in, F_out);
    std::copy(layer.weight_neigh().data().begin(),
              layer.weight_neigh().data().end(), Wn.data().begin());
    Tensor Ws = Tensor::dense(F_in, F_out);
    std::copy(layer.weight_self().data().begin(),
              layer.weight_self().data().end(), Ws.data().begin());
    Tensor b = Tensor::dense(1, F_out);
    std::copy(layer.bias().data().begin(), layer.bias().data().end(),
              b.data().begin());

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        auto out = layer.forward(A, H);
        (void)out;
    }

    // Fused timing
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        auto out = layer.forward(A, H);
        (void)out;
    }
    auto t1 = Clock::now();
    double fused = elapsed_us(t0, t1) / iters;

    // Unfused warmup
    for (int i = 0; i < warmup; ++i) {
        auto out = sage_forward_unfused(A, H, Wn, Ws, b);
        (void)out;
    }

    // Unfused timing
    t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        auto out = sage_forward_unfused(A, H, Wn, Ws, b);
        (void)out;
    }
    t1 = Clock::now();
    double unfused = elapsed_us(t0, t1) / iters;

    return {fused, unfused, unfused / fused};
}

// ============================================================================
//  Memory estimation (analytical)
// ============================================================================
struct MemEstimate {
    std::size_t unfused_bytes;
    std::size_t fused_bytes;
    double ratio;
};

static MemEstimate gat_memory(std::size_t N, std::size_t F_out,
                              std::size_t nnz, std::size_t max_deg) {
    // Unfused intermediates:
    //   Wh:          N × F_out × 4
    //   src_scores:  N × 4
    //   dst_scores:  N × 4
    //   edge_logits: nnz × 4
    //   attn_csr:    (N+1)×4 + nnz×4 + nnz×4  (rp + ci + vals)
    //   alpha_csr:   (N+1)×4 + nnz×4 + nnz×4
    //   out:         N × F_out × 4
    std::size_t unfused =
        N * F_out * 4 +         // Wh
        N * 4 + N * 4 +         // src, dst scores
        nnz * 4 +               // edge_logits
        (N + 1) * 4 + nnz * 4 + nnz * 4 +  // attn_csr
        (N + 1) * 4 + nnz * 4 + nnz * 4 +  // alpha_csr
        N * F_out * 4;          // out

    // Fused intermediates:
    //   Wh:          N × F_out × 4
    //   src_scores:  N × 4
    //   dst_scores:  N × 4
    //   attn buffer: max_deg × 4  (per-thread, single row)
    //   out:         N × F_out × 4
    std::size_t fused =
        N * F_out * 4 +         // Wh
        N * 4 + N * 4 +         // src, dst scores
        max_deg * 4 +           // per-row attn buffer (one thread)
        N * F_out * 4;          // out

    return {unfused, fused,
            static_cast<double>(unfused) / static_cast<double>(fused)};
}

static MemEstimate sage_memory(std::size_t N, std::size_t F_in,
                               std::size_t F_out) {
    // Unfused intermediates:
    //   agg:       N × F_in × 4
    //   h_neigh:   N × F_out × 4
    //   h_self:    N × F_out × 4
    std::size_t unfused =
        N * F_in * 4 +          // agg
        N * F_out * 4 +         // h_neigh
        N * F_out * 4;          // h_self

    // Fused intermediates:
    //   out:       N × F_out × 4
    //   agg_row:   F_in × 4  (per-thread, single row)
    std::size_t fused =
        N * F_out * 4 +         // out
        F_in * 4;               // per-row agg buffer (one thread)

    return {unfused, fused,
            static_cast<double>(unfused) / static_cast<double>(fused)};
}

// ============================================================================
//  Main
// ============================================================================
int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n"
              << "║  TinyGNN — Operator Fusion Benchmark  (Phase 9)         ║\n"
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

    // ── GAT Fusion Benchmarks ───────────────────────────────────────────────
    std::cout << "──── GAT: Fused SpSDDMM + edge_softmax + SpMM ────────────\n\n";

    struct GATConfig {
        std::size_t N, F_in, F_out, avg_deg;
        const char* label;
    };

    GATConfig gat_configs[] = {
        {2708,  1433, 8,   5,  "Cora-like (2708×1433→8, deg≈5)"},
        {5000,  128,  64,  10, "Medium (5000×128→64, deg≈10)"},
        {10000, 64,   32,  20, "Large (10000×64→32, deg≈20)"},
        {20000, 32,   16,  30, "XL (20000×32→16, deg≈30)"},
    };

    std::cout << std::fixed << std::setprecision(0);
    std::cout << "  " << std::left << std::setw(42) << "Configuration"
              << std::right << std::setw(10) << "Unfused"
              << std::setw(10) << "Fused"
              << std::setw(10) << "Speedup" << "\n";
    std::cout << "  " << std::string(72, '-') << "\n";

    for (auto& cfg : gat_configs) {
        BenchResult r = bench_gat(cfg.N, cfg.F_in, cfg.F_out, cfg.avg_deg,
                                  2, 5);
        std::cout << "  " << std::left << std::setw(42) << cfg.label
                  << std::right << std::setw(8) << r.unfused_us << "µs"
                  << std::setw(8) << r.fused_us << "µs"
                  << std::setprecision(2) << std::setw(8) << r.speedup << "×"
                  << std::setprecision(0) << "\n";
    }

    // ── GAT Memory Analysis ─────────────────────────────────────────────────
    std::cout << "\n  Memory analysis (intermediate allocations):\n";
    std::cout << "  " << std::left << std::setw(42) << "Configuration"
              << std::right << std::setw(12) << "Unfused"
              << std::setw(12) << "Fused"
              << std::setw(10) << "Ratio" << "\n";
    std::cout << "  " << std::string(76, '-') << "\n";

    for (auto& cfg : gat_configs) {
        // Estimate nnz and max_deg from avg_deg
        std::size_t nnz_approx = cfg.N * (cfg.avg_deg + 1); // +1 for self-loops
        std::size_t max_deg_approx = cfg.avg_deg * 3;  // rough estimate
        auto mem = gat_memory(cfg.N, cfg.F_out, nnz_approx, max_deg_approx);

        auto fmt_kb = [](std::size_t bytes) -> std::string {
            if (bytes < 1024 * 1024) {
                return std::to_string(bytes / 1024) + " KB";
            }
            return std::to_string(bytes / (1024 * 1024)) + " MB";
        };

        std::cout << "  " << std::left << std::setw(42) << cfg.label
                  << std::right << std::setw(12) << fmt_kb(mem.unfused_bytes)
                  << std::setw(12) << fmt_kb(mem.fused_bytes)
                  << std::setprecision(1) << std::setw(8) << mem.ratio << "×"
                  << std::setprecision(0) << "\n";
    }

    // ── SAGE Fusion Benchmarks ──────────────────────────────────────────────
    std::cout << "\n──── SAGE: Fused Aggregation + Dual-Matmul ──────────────\n\n";

    struct SAGEConfig {
        std::size_t N, F_in, F_out, avg_deg;
        const char* label;
    };

    SAGEConfig sage_configs[] = {
        {2708,  1433, 128, 5,  "Cora-like (2708×1433→128, deg≈5)"},
        {5000,  128,  64,  10, "Medium (5000×128→64, deg≈10)"},
        {10000, 64,   32,  20, "Large (10000×64→32, deg≈20)"},
        {20000, 128,  64,  30, "XL (20000×128→64, deg≈30)"},
    };

    std::cout << "  " << std::left << std::setw(42) << "Configuration"
              << std::right << std::setw(10) << "Unfused"
              << std::setw(10) << "Fused"
              << std::setw(10) << "Speedup" << "\n";
    std::cout << "  " << std::string(72, '-') << "\n";

    for (auto& cfg : sage_configs) {
        BenchResult r = bench_sage(cfg.N, cfg.F_in, cfg.F_out, cfg.avg_deg,
                                   2, 5);
        std::cout << "  " << std::left << std::setw(42) << cfg.label
                  << std::right << std::setw(8) << r.unfused_us << "µs"
                  << std::setw(8) << r.fused_us << "µs"
                  << std::setprecision(2) << std::setw(8) << r.speedup << "×"
                  << std::setprecision(0) << "\n";
    }

    // ── SAGE Memory Analysis ────────────────────────────────────────────────
    std::cout << "\n  Memory analysis (intermediate allocations):\n";
    std::cout << "  " << std::left << std::setw(42) << "Configuration"
              << std::right << std::setw(12) << "Unfused"
              << std::setw(12) << "Fused"
              << std::setw(10) << "Ratio" << "\n";
    std::cout << "  " << std::string(76, '-') << "\n";

    for (auto& cfg : sage_configs) {
        auto mem = sage_memory(cfg.N, cfg.F_in, cfg.F_out);

        auto fmt_kb = [](std::size_t bytes) -> std::string {
            if (bytes < 1024 * 1024) {
                return std::to_string(bytes / 1024) + " KB";
            }
            return std::to_string(bytes / (1024 * 1024)) + " MB";
        };

        std::cout << "  " << std::left << std::setw(42) << cfg.label
                  << std::right << std::setw(12) << fmt_kb(mem.unfused_bytes)
                  << std::setw(12) << fmt_kb(mem.fused_bytes)
                  << std::setprecision(1) << std::setw(8) << mem.ratio << "×"
                  << std::setprecision(0) << "\n";
    }

    std::cout << "\n  ✓ Benchmark complete.\n";
    return 0;
}
