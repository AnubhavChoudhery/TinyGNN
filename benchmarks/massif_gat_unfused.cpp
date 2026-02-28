// ============================================================================
//  TinyGNN — Massif Memory Profile: GAT Unfused  (Phase 9)
//  benchmarks/massif_gat_unfused.cpp
//
//  Runs the unfused (original Phase 6) GAT forward to measure peak heap
//  with Valgrind Massif.  Compare against massif_gat_fused.cpp.
//
//  Build (no OpenMP — Massif + OMP don't mix well):
//    g++ -std=c++17 -O2 -g -Iinclude -o massif_gat_unfused \
//        benchmarks/massif_gat_unfused.cpp src/tensor.cpp src/graph_loader.cpp \
//        src/ops.cpp src/layers.cpp src/model.cpp
//
//  Run:
//    valgrind --tool=massif --pages-as-heap=no --stacks=no \
//             ./massif_gat_unfused
//    ms_print massif.out.<pid>
// ============================================================================

#include "tinygnn/layers.hpp"
#include "tinygnn/ops.hpp"
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

// Reproduce the original unfused GAT forward
static Tensor gat_unfused(const Tensor& A, const Tensor& H,
                          const Tensor& W, const Tensor& al,
                          const Tensor& ar, const Tensor& bias,
                          float neg_slope) {
    const std::size_t N = H.rows();
    const std::size_t F_out = W.cols();
    const auto& rp = A.row_ptr();
    const auto& ci = A.col_ind();

    Tensor Wh = matmul(H, W);
    const float* wh_d = Wh.data().data();
    const float* ald = al.data().data();
    const float* ard = ar.data().data();

    std::vector<float> src(N, 0.f), dst(N, 0.f);
    for (std::size_t i = 0; i < N; ++i) {
        float s = 0.f, d = 0.f;
        const float* whi = wh_d + i * F_out;
        for (std::size_t f = 0; f < F_out; ++f) {
            s += ald[f] * whi[f];
            d += ard[f] * whi[f];
        }
        src[i] = s;
        dst[i] = d;
    }

    const std::size_t nnz = ci.size();
    std::vector<float> edge_logits(nnz);
    for (std::size_t i = 0; i < N; ++i) {
        const float si = src[i];
        for (int32_t nz = rp[i]; nz < rp[i + 1]; ++nz) {
            float e = si + dst[static_cast<std::size_t>(ci[nz])];
            edge_logits[nz] = (e >= 0.f) ? e : neg_slope * e;
        }
    }

    std::vector<int32_t> rpc(rp.begin(), rp.end());
    std::vector<int32_t> cic(ci.begin(), ci.end());
    Tensor attn_csr = Tensor::sparse_csr(N, N, std::move(rpc),
                                         std::move(cic),
                                         std::move(edge_logits));
    Tensor alpha = edge_softmax(attn_csr);
    Tensor out = spmm(alpha, Wh);
    add_bias(out, bias);
    return out;
}

int main() {
    constexpr std::size_t N = 10000;
    constexpr std::size_t F_in = 64;
    constexpr std::size_t F_out = 32;
    constexpr std::size_t AVG_DEG = 20;

    std::mt19937 rng(42);
    Tensor A = random_adj(N, AVG_DEG, rng);
    Tensor H = random_dense(N, F_in, rng);
    Tensor W = random_dense(F_in, F_out, rng);
    Tensor al = random_dense(1, F_out, rng);
    Tensor ar = random_dense(1, F_out, rng);
    Tensor b = random_dense(1, F_out, rng);

    std::cout << "GAT Unfused: N=" << N << " F_in=" << F_in
              << " F_out=" << F_out << " avg_deg=" << AVG_DEG
              << " nnz=" << A.col_ind().size() << "\n";

    Tensor out = gat_unfused(A, H, W, al, ar, b, 0.2f);
    std::cout << "  out shape: " << out.rows() << "×" << out.cols() << "\n";
    std::cout << "  out[0][0] = " << out.data()[0] << "\n";
    return 0;
}
