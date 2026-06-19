#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/zone_tersoff.cuh"      // tersoff_force_kernel, kMaxNbr, kZoneBlock
#include "tdmd/potentials/many_body.hpp"   // PassDecl/PassKind (the firewall)
#include "tdmd/potentials/tersoff.hpp"     // TersoffParams
#include "tdmd/potentials/tersoff_ring.hpp"  // TersoffPotential, TersoffRing (the policy-ctor)

// M6 / Tersoff-ladder Te5 — GpuTersoffWinForce: the GPU window-force POLICY that drops into
// TersoffRing (the Te3b policy-injected ctor) so GpuTersoffRing = TersoffRing<double,
// GpuTersoffWinForce<double>> inherits ALL the proven host orchestration unchanged. Mirrors
// GpuSwWinForce, with TWO genuine deltas: (1) the [BondOrder, Force] firewall (NOT SW's
// [Force,Force] — a GpuSwWinForce copy would wrongly reject pass 0 == BondOrder); (2) the
// HOST-SIDE CANONICAL ζ-SORT (sort the window by global key so the device kernel sums the FP64
// ζ canonically). MEASURED (the Te3 finding, S2 corrected): the sort is DEFENSIVE — GPU 1-vs-z is
// bitwise WITHOUT it (the block-order window is z-independent) and a single eval is sub-quantum;
// the sort is kept for window-permutation-bitwise-by-construction + canonical alignment with the
// CPU serial. CPU↔GPU is TOLERANCE (exp/pow/sin), GPU-internal is BITWISE — see zone_tersoff.cuh.
namespace tdmd::cuda {
namespace ters_wf_detail {
template <typename T>
inline T* ters_wf_malloc(std::size_t n) {
  void* d = nullptr;
  if (cudaMalloc(&d, n * sizeof(T)) != cudaSuccess)
    throw std::runtime_error("GpuTersoffWinForce: cudaMalloc failed");
  return static_cast<T*>(d);
}
inline int ters_ng(int n) { return (n + kZoneBlock - 1) / kZoneBlock; }
}  // namespace ters_wf_detail

struct GpuTersoffWindowState {
  potentials::TersoffParams tp{};  // by value — NO device tables (no spline)
  int cap_m = 0;
  double *wx = nullptr, *wy = nullptr, *wz = nullptr;
  long* wkey = nullptr;
  int* d_owned = nullptr;
  long long *d_fx = nullptr, *d_fy = nullptr, *d_fz = nullptr;
  long long *d_pe = nullptr, *d_nbonds = nullptr, *d_ntri = nullptr;
  unsigned long long* d_mr = nullptr;
  int* d_of = nullptr;
  unsigned long long sentinel = 0;  // pos_double_bits(1e300) — empty-window seed
  long long last_nbonds = 0, last_ntri = 0;  // TEST-ONLY; the ring SINKS counts (benign WAR race)
  std::mutex mu;

  explicit GpuTersoffWindowState(const potentials::TersoffParams& t) : tp(t) {
    double v = 1e300;
    std::memcpy(&sentinel, &v, 8);
    d_pe = ters_wf_detail::ters_wf_malloc<long long>(1);
    d_nbonds = ters_wf_detail::ters_wf_malloc<long long>(1);
    d_ntri = ters_wf_detail::ters_wf_malloc<long long>(1);
    d_mr = ters_wf_detail::ters_wf_malloc<unsigned long long>(1);
    d_of = ters_wf_detail::ters_wf_malloc<int>(1);
    grow(64);
  }
  void free_scratch() {
    for (void* p : {(void*)wx, (void*)wy, (void*)wz, (void*)wkey, (void*)d_owned,
                    (void*)d_fx, (void*)d_fy, (void*)d_fz})
      if (p) cudaFree(p);
    wx = wy = wz = nullptr; wkey = nullptr; d_owned = nullptr; d_fx = d_fy = d_fz = nullptr;
  }
  void grow(int m) {
    if (m <= cap_m) return;
    free_scratch();
    cap_m = m;
    wx = ters_wf_detail::ters_wf_malloc<double>(cap_m);
    wy = ters_wf_detail::ters_wf_malloc<double>(cap_m);
    wz = ters_wf_detail::ters_wf_malloc<double>(cap_m);
    wkey = ters_wf_detail::ters_wf_malloc<long>(cap_m);
    d_owned = ters_wf_detail::ters_wf_malloc<int>(cap_m);
    d_fx = ters_wf_detail::ters_wf_malloc<long long>(cap_m);
    d_fy = ters_wf_detail::ters_wf_malloc<long long>(cap_m);
    d_fz = ters_wf_detail::ters_wf_malloc<long long>(cap_m);
  }
  ~GpuTersoffWindowState() {
    free_scratch();
    for (void* p : {(void*)d_pe, (void*)d_nbonds, (void*)d_ntri, (void*)d_mr, (void*)d_of})
      if (p) cudaFree(p);
  }
};
static_assert(std::is_trivially_copyable_v<potentials::TersoffParams>,
              "TersoffParams is passed by value into the kernel; a future non-trivial member "
              "(e.g. a std::vector for multi-element) would silently slice — fail loudly here");

template <typename Real>
struct GpuTersoffWinForce {
  std::shared_ptr<GpuTersoffWindowState> st;
  bool skip_sort = false;  // TEST-ONLY (G-ζORDER): skip the canonical sort to prove it bites

  GpuTersoffWinForce(const potentials::TersoffParams& tp, const core::Box& box)
      : st(std::make_shared<GpuTersoffWindowState>(tp)) {
    for (int d = 0; d < 3; ++d)
      if (box.periodic[d] && box.len(d) < 2.0 * tp.rcut())
        throw std::invalid_argument("GpuTersoffWinForce: periodic box dim < 2·rcut — min-image ambiguous");
  }

  // FIREWALL — the INVERSE of GpuEamWindowForce (which THROWS on needs_transpose). Tersoff's
  // Force pass writes to a third atom k (needs_transpose) and the 4-role transpose-replay IS the
  // mechanism ⇒ ACCEPT it. THE genuine delta vs GpuSwWinForce: pass 0 is BondOrder (a copy of
  // GpuSwWinForce::assert_supported demands every pass be Force ⇒ would WRONGLY reject pass 0).
  static void assert_supported(std::span<const potentials::PassDecl> passes) {
    if (passes.size() != 2)
      throw std::runtime_error("GpuTersoffWinForce: Tersoff is the 2-pass [BondOrder ζ, Force] "
          "sequence — got " + std::to_string(passes.size()) + " passes");
    if (passes[0].kind != potentials::PassKind::BondOrder)
      throw std::runtime_error("GpuTersoffWinForce: pass 0 must be BondOrder (a Force here is the "
          "SW [Force,Force] descriptor, not Tersoff)");
    if (passes[1].kind != potentials::PassKind::Force)
      throw std::runtime_error("GpuTersoffWinForce: pass 1 must be Force");
    for (const auto& p : passes)
      if (p.iterative)
        throw std::runtime_error("GpuTersoffWinForce: iterative pass (QEq/CG, ReaxFF) unimplemented");
    if (passes[0].needs_transpose || !passes[1].needs_transpose)
      throw std::runtime_error("GpuTersoffWinForce: expected pass 0 symmetric (BondOrder ζ), "
          "pass 1 needs_transpose (Force) — descriptor does not match the Tersoff force structure");
  }

  void compute(const double* wx, const double* wy, const double* wz, const long* key,
               int m, const int* owned, int n_owned, const core::PairGeom& geom,
               double /*rho_cap — Tersoff has no density*/,
               std::vector<core::fixed::ForceAccum>& wFx,
               std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz,
               core::fixed::EnergyAccum& pe, double& min_r2) const {
    GpuTersoffWindowState& s = *st;
    std::lock_guard<std::mutex> lk(s.mu);
    if (m <= 0) return;
    s.grow(m);

    // CANONICAL ζ-ORDER (DEFENSIVE — mirrors TersoffWinForce::compute): sort the window by global
    // key so the device kernel sums the FP64 ζ canonically. The GPU 1-vs-z is bitwise WITHOUT this
    // (the block-order window is z-independent) and a single eval is sub-quantum (the Te3 finding),
    // so the sort is not strictly load-bearing for the GPU gates — kept for window-permutation-
    // bitwise-by-construction + canonical alignment with the CPU serial. skip_sort = test hook.
    std::vector<int> perm(m);
    std::iota(perm.begin(), perm.end(), 0);
    if (!skip_sort) std::sort(perm.begin(), perm.end(), [&](int a, int b) { return key[a] < key[b]; });
    std::vector<double> sx(m), sy(m), sz(m);
    std::vector<long> sk(m);
    std::vector<int> inv(m);
    for (int t = 0; t < m; ++t) { const int o = perm[t]; sx[t] = wx[o]; sy[t] = wy[o]; sz[t] = wz[o]; sk[t] = key[o]; inv[o] = t; }
    std::vector<int> sowned(n_owned);
    for (int t = 0; t < n_owned; ++t) sowned[t] = inv[owned[t]];

    cudaMemcpy(s.wx, sx.data(), m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wy, sy.data(), m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wz, sz.data(), m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wkey, sk.data(), m * sizeof(long), cudaMemcpyHostToDevice);
    if (n_owned > 0) cudaMemcpy(s.d_owned, sowned.data(), n_owned * sizeof(int), cudaMemcpyHostToDevice);
    const long long z64 = 0; const int z32 = 0;
    cudaMemcpy(s.d_pe, &z64, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_nbonds, &z64, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_ntri, &z64, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_mr, &s.sentinel, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_of, &z32, 4, cudaMemcpyHostToDevice);
    cudaMemsetAsync(s.d_fx, 0, m * sizeof(long long));  // initcheck fix (D2H reads all m)
    cudaMemsetAsync(s.d_fy, 0, m * sizeof(long long));
    cudaMemsetAsync(s.d_fz, 0, m * sizeof(long long));

    tersoff_force_kernel<<<ters_wf_detail::ters_ng(n_owned), kZoneBlock>>>(
        s.wx, s.wy, s.wz, s.wkey, m, s.d_owned, n_owned, geom, s.tp,
        s.d_fx, s.d_fy, s.d_fz, s.d_pe, s.d_nbonds, s.d_ntri, s.d_mr, s.d_of);

    std::vector<long long> hfx(m), hfy(m), hfz(m);
    cudaMemcpy(hfx.data(), s.d_fx, m * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(hfy.data(), s.d_fy, m * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(hfz.data(), s.d_fz, m * sizeof(long long), cudaMemcpyDeviceToHost);
    long long h_pe = 0, h_nb = 0, h_nt = 0;
    cudaMemcpy(&h_pe, s.d_pe, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_nb, s.d_nbonds, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_nt, s.d_ntri, 8, cudaMemcpyDeviceToHost);
    unsigned long long h_mr = 0;
    cudaMemcpy(&h_mr, s.d_mr, 8, cudaMemcpyDeviceToHost);
    int of = 0;
    cudaMemcpy(&of, s.d_of, 4, cudaMemcpyDeviceToHost);
    if (cudaDeviceSynchronize() != cudaSuccess)
      throw std::runtime_error("GpuTersoffWinForce: kernel launch/sync failed");
    if (of) throw std::runtime_error("GpuTersoffWinForce: force/neighbour overflow HALT");

    // un-permute: sorted slot t ↔ original window slot perm[t] (the kernel wrote the sorted layout).
    for (int t = 0; t < m; ++t) { wFx[perm[t]].raw = hfx[t]; wFy[perm[t]].raw = hfy[t]; wFz[perm[t]].raw = hfz[t]; }
    pe.raw += h_pe;
    double gpu_mr2;
    std::memcpy(&gpu_mr2, &h_mr, 8);
    if (gpu_mr2 < min_r2) min_r2 = gpu_mr2;
    s.last_nbonds = h_nb; s.last_ntri = h_nt;  // TEST-ONLY
  }
};

template <typename Real = double>
using GpuTersoffRing = potentials::TersoffRing<Real, GpuTersoffWinForce<Real>>;

}  // namespace tdmd::cuda
