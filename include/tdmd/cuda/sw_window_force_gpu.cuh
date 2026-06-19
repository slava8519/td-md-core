#pragma once
#include <cuda_runtime.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"     // Box
#include "tdmd/core/zones.hpp"   // PairGeom
#include "tdmd/cuda/zone_sw.cuh"  // sw_force_kernel, kMaxNbr, kZoneBlock
#include "tdmd/potentials/many_body.hpp"  // PassDecl/PassKind (the firewall)
#include "tdmd/potentials/sw.hpp"         // SwParams

// M6 / SW-ladder T5 — GpuSwWinForce: the GPU window-force POLICY that drops into SwRing
// (sw_ring.hpp's policy-injected ctor) so GpuSwRing = SwRing<double, GpuSwWinForce<double>>
// inherits ALL the proven host orchestration unchanged. Mirrors GpuEamWindowForce, STRIPPED
// (SW has no spline upload, no density/embedding pre-pass, no dens_scale/rho_cap, single
// Q24.40 format). Given ONE already-gathered 3-zone window it launches sw_force_kernel and
// writes the int64 accumulators decoded IDENTICALLY to the CPU SwWinForce. CPU↔GPU is
// TOLERANCE (exp/pow), GPU-internal is BITWISE — see the zone_sw.cuh contract banner.
namespace tdmd::cuda {
namespace sw_wf_detail {
template <typename T>
inline T* sw_wf_malloc(std::size_t n) {
  void* d = nullptr;
  if (cudaMalloc(&d, n * sizeof(T)) != cudaSuccess)
    throw std::runtime_error("GpuSwWinForce: cudaMalloc failed");
  return static_cast<T*>(d);
}
inline int sw_ng(int n) { return (n + kZoneBlock - 1) / kZoneBlock; }
}  // namespace sw_wf_detail

struct GpuSwWindowState {
  potentials::SwParams sp{};  // by value — NO device tables (no spline)
  int cap_m = 0;
  double *wx = nullptr, *wy = nullptr, *wz = nullptr;
  long* wkey = nullptr;
  int* d_owned = nullptr;
  long long *d_fx = nullptr, *d_fy = nullptr, *d_fz = nullptr, *d_pe = nullptr, *d_ntri = nullptr;
  unsigned long long* d_mr = nullptr;
  int* d_of = nullptr;
  unsigned long long sentinel = 0;  // pos_double_bits(1e300) — empty-window seed (CPU 1e300)
  long long last_ntri = 0;  // last window's triplet count — TEST-ONLY (G4); the ring SINKS
                            // n_triplets, so a write-without-read race across z is benign.
  std::mutex mu;

  explicit GpuSwWindowState(const potentials::SwParams& s) : sp(s) {
    double v = 1e300;
    std::memcpy(&sentinel, &v, 8);
    d_pe = sw_wf_detail::sw_wf_malloc<long long>(1);
    d_ntri = sw_wf_detail::sw_wf_malloc<long long>(1);
    d_mr = sw_wf_detail::sw_wf_malloc<unsigned long long>(1);
    d_of = sw_wf_detail::sw_wf_malloc<int>(1);
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
    wx = sw_wf_detail::sw_wf_malloc<double>(cap_m);
    wy = sw_wf_detail::sw_wf_malloc<double>(cap_m);
    wz = sw_wf_detail::sw_wf_malloc<double>(cap_m);
    wkey = sw_wf_detail::sw_wf_malloc<long>(cap_m);
    d_owned = sw_wf_detail::sw_wf_malloc<int>(cap_m);
    d_fx = sw_wf_detail::sw_wf_malloc<long long>(cap_m);
    d_fy = sw_wf_detail::sw_wf_malloc<long long>(cap_m);
    d_fz = sw_wf_detail::sw_wf_malloc<long long>(cap_m);
  }
  ~GpuSwWindowState() {
    free_scratch();
    for (void* p : {(void*)d_pe, (void*)d_ntri, (void*)d_mr, (void*)d_of})
      if (p) cudaFree(p);
  }
};

template <typename Real>
struct GpuSwWinForce {
  std::shared_ptr<GpuSwWindowState> st;

  GpuSwWinForce(const potentials::SwParams& sp, const core::Box& box)
      : st(std::make_shared<GpuSwWindowState>(sp)) {
    // SW two-wing min-image guard (host; mirrors SwRing ctor + sw_zone_pass): every
    // periodic box dim > 2·rcut, else a wing bond could wrap to a wrong image (the
    // device kernel trusts geom.reduce picked the unique image).
    for (int d = 0; d < 3; ++d)
      if (box.periodic[d] && box.len(d) < 2.0 * sp.rcut())
        throw std::invalid_argument("GpuSwWinForce: periodic box dim < 2·rcut — min-image ambiguous");
  }

  // FIREWALL — the INVERSE of GpuEamWindowForce::assert_supported (which THROWS on
  // needs_transpose). SW's φ₃ IS needs_transpose and THIS kernel implements the
  // transpose-replay ⇒ ACCEPT it; reject count≠2 / non-Force / iterative / the
  // symmetric-only-φ₃ trap. Copies SwWinForce::assert_supported (sw_ring.hpp) so the GPU
  // SW seam asserts its own descriptor from day one (closes the FIREWALL GAP for SW).
  static void assert_supported(std::span<const potentials::PassDecl> passes) {
    if (passes.size() != 2)
      throw std::runtime_error("GpuSwWinForce: SW is the 2-pass [φ2 Force, φ3 Force] "
          "sequence — got " + std::to_string(passes.size()) + " passes");
    for (const auto& p : passes) {
      if (p.kind != potentials::PassKind::Force)
        throw std::runtime_error("GpuSwWinForce: only PassKind::Force is implemented");
      if (p.iterative)
        throw std::runtime_error("GpuSwWinForce: iterative pass unimplemented");
    }
    if (passes[0].needs_transpose || !passes[1].needs_transpose)
      throw std::runtime_error("GpuSwWinForce: expected pass 0 symmetric (φ2), pass 1 "
          "needs_transpose (φ3) — descriptor does not match the SW force structure");
  }

  void compute(const double* wx, const double* wy, const double* wz, const long* key,
               int m, const int* owned, int n_owned, const core::PairGeom& geom,
               double /*rho_cap — SW has no density*/,
               std::vector<core::fixed::ForceAccum>& wFx,
               std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz,
               core::fixed::EnergyAccum& pe, double& min_r2) const {
    GpuSwWindowState& s = *st;
    std::lock_guard<std::mutex> lk(s.mu);  // serialize device work (correctness, like EAM)
    if (m <= 0) return;
    s.grow(m);

    cudaMemcpy(s.wx, wx, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wy, wy, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wz, wz, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wkey, key, m * sizeof(long), cudaMemcpyHostToDevice);
    if (n_owned > 0)
      cudaMemcpy(s.d_owned, owned, n_owned * sizeof(int), cudaMemcpyHostToDevice);
    const long long z64 = 0;
    const int z32 = 0;
    cudaMemcpy(s.d_pe, &z64, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_ntri, &z64, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_mr, &s.sentinel, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_of, &z32, 4, cudaMemcpyHostToDevice);
    // zero the force buffers before launch (the kernel writes only the n_owned owned
    // slots; the D2H below copies all m ⇒ without this the non-owned slots are read
    // uninitialized — benign, never consumed, but it trips initcheck. Acceptance fix).
    cudaMemsetAsync(s.d_fx, 0, m * sizeof(long long));
    cudaMemsetAsync(s.d_fy, 0, m * sizeof(long long));
    cudaMemsetAsync(s.d_fz, 0, m * sizeof(long long));

    sw_force_kernel<<<sw_wf_detail::sw_ng(n_owned), kZoneBlock>>>(
        s.wx, s.wy, s.wz, s.wkey, m, s.d_owned, n_owned, geom, s.sp,
        s.d_fx, s.d_fy, s.d_fz, s.d_pe, s.d_ntri, s.d_mr, s.d_of);

    std::vector<long long> hfx(m), hfy(m), hfz(m);
    cudaMemcpy(hfx.data(), s.d_fx, m * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(hfy.data(), s.d_fy, m * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(hfz.data(), s.d_fz, m * sizeof(long long), cudaMemcpyDeviceToHost);
    long long h_pe = 0, h_ntri = 0;  // h_ntri sunk (the MB2 count lives on the serial path)
    cudaMemcpy(&h_pe, s.d_pe, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_ntri, s.d_ntri, 8, cudaMemcpyDeviceToHost);
    unsigned long long h_mr = 0;
    cudaMemcpy(&h_mr, s.d_mr, 8, cudaMemcpyDeviceToHost);
    int of = 0;
    cudaMemcpy(&of, s.d_of, 4, cudaMemcpyDeviceToHost);
    if (cudaDeviceSynchronize() != cudaSuccess)
      throw std::runtime_error("GpuSwWinForce: kernel launch/sync failed");
    if (of) throw std::runtime_error("GpuSwWinForce: force/neighbour overflow HALT");

    for (int o = 0; o < n_owned; ++o) {
      const int loc = owned[o];
      wFx[loc].raw = hfx[loc]; wFy[loc].raw = hfy[loc]; wFz[loc].raw = hfz[loc];
    }
    pe.raw += h_pe;  // associative int64 add ⇒ == CPU multiset (up to the transcendental)
    double gpu_mr2;
    std::memcpy(&gpu_mr2, &h_mr, 8);
    if (gpu_mr2 < min_r2) min_r2 = gpu_mr2;
    s.last_ntri = h_ntri;  // TEST-ONLY (the ring never reads this)
  }
};

}  // namespace tdmd::cuda
