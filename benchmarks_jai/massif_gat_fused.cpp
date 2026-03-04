// ============================================================================
//  TinyGNN — Massif Memory Profile: GAT Fused  (Phase 9)
//  benchmarks/massif_gat_fused.cpp
//
//  Runs the fused GAT forward to measure peak heap with Valgrind Massif.
//  Compare against massif_gat_unfused.cpp.
//
//  Build (no OpenMP — Massif + OMP don't mix well):
//    g++ -std=c++17 -O2 -g -Iinclude -o massif_gat_fused \
//        benchmarks/massif_gat_fused.cpp src/tensor.cpp src/graph_loader.cpp \
//        src/ops.cpp src/layers.cpp src/model.cpp
//
//  Run:
//    valgrind --tool=massif --pages-as-heap=no --stacks=no \
//             ./massif_gat_fused
//    ms_print massif.out.<pid>
// ============================================================================

#include "tinygnn/layers.hpp"
#include "tinygnn/tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using namespace tinygnn;

static Tensor random_dense(std::size_t r, std::size_t c, std::mt19937& rng) {
    Tensor t = Tensor::dense(r, c);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (auto& v : t.data()) v = d(rng);
    return t;
}

static Tensor random_adj(std::size_t N, std::size_t avg_deg,
                         std::mt19937& rng) {
    std::vector<std::vector<int32_t>> adj(N);
    std::uniform_int_distribution<std::size_t> nd(0, N - 1);
    for (std::size_t i = 0; i < N; ++i) {
        adj[i].push_back(static_cast<int32_t>(i));
        for (std::size_t d = 0; d < avg_deg; ++d)
            adj[i].push_back(static_cast<int32_t>(nd(rng)));
        std::sort(adj[i].begin(), adj[i].end());
        adj[i].erase(std::unique(adj[i].begin(), adj[i].end()),
                      adj[i].end());
    }
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

int main() {
    constexpr std::size_t N = 10000;
    constexpr std::size_t F_in = 64;
    constexpr std::size_t F_out = 32;
    constexpr std::size_t AVG_DEG = 20;

    std::mt19937 rng(42);
    Tensor A = random_adj(N, AVG_DEG, rng);
    Tensor H = random_dense(N, F_in, rng);

    GATLayer layer(F_in, F_out, 0.2f, true, Activation::None);
    layer.set_weight(random_dense(F_in, F_out, rng));
    layer.set_attn_left(random_dense(1, F_out, rng));
    layer.set_attn_right(random_dense(1, F_out, rng));
    layer.set_bias(random_dense(1, F_out, rng));

    std::cout << "GAT Fused: N=" << N << " F_in=" << F_in
              << " F_out=" << F_out << " avg_deg=" << AVG_DEG
              << " nnz=" << A.col_ind().size() << "\n";

    Tensor out = layer.forward(A, H);
    std::cout << "  out shape: " << out.rows() << "×" << out.cols() << "\n";
    std::cout << "  out[0][0] = " << out.data()[0] << "\n";
    return 0;
}
