#pragma once
#include <cuda_runtime.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/zones.hpp"  // PairGeom
#include "tdmd/cuda/zone_eam.cuh"  // eam_density/embedding/force kernels, EamSetflView
#include "tdmd/potentials/eam_spline.hpp"

// M6 E5b-3b — the GPU WINDOW-FORCE POLICY for the streaming multi-node EAM ring.
//
// EamRing (eam_ring.hpp) is the PROVEN bitwise orchestration oracle; its ONLY
// moving part is the per-window EAM force. This policy runs that single piece on
// the GPU — given ONE already-gathered 3-zone window {wx,wy,wz,key,m,owned,
// n_owned} it launches the E5 kernels (eam_density → eam_embedding → eam_force,
// zone_eam.cuh, reused VERBATIM) and returns the int64 force accumulators + pe +
// min_r2, BITWISE-EQUAL to eam_window_force (eam_zone.hpp). test_cuda_eam already
// proves the kernels match per-window; this policy just wires the per-call
// upload → 3 kernels → download of the int64 raws, decoded identically.
//
// Why bitwise (not ~1e-12): the spline is transcendental-free (Horner + an int
// knot index) ⇒ CPU↔GPU is strictly == under --fmad=false (the TU MUST link
// tdmd_eam_cuda_flags). int64 accumulation is associative ⇒ the per-atom raw is
// order-free ⇒ depends only on the window MULTISET, which the gather preserves
// (key = atom id, identical on both sides). So EamGpuRing inherits ALL of
// EamRing's orchestration ⇒ bitwise ≡ the CPU ring by construction.
//
// CONCURRENCY: EamRing spawns one jthread per node, all sharing the single
// policy member ⇒ compute() is called concurrently. Correctness-first (E5b is a
// correctness gate, perf is E5c): a mutex serializes the device work; the shared
// scratch grows to the max window seen. The host orchestration (FSM, transport,
// staging) still overlaps — only the kernel launches serialize. A GPU-RESIDENT
// D2D StreamTransport + per-stream buffers is the perf optimization, deferred to
// E5c; THIS delivers the CORRECT streaming z>1 GPU EAM ring.
namespace tdmd::cuda {

template <typename T>
inline T* eam_wf_malloc(std::size_t n) {
  T* d = nullptr;
  if (cudaMalloc(&d, n * sizeof(T)) != cudaSuccess)
    throw std::runtime_error("GpuEamWindowForce: cudaMalloc failed");
  return d;
}

// Shared device state: the spline view (uploaded once) + growable per-call
// scratch + the serialization mutex. Held by shared_ptr so the policy is cheap
// to copy/move (EamRing stores it by value) yet all copies share one allocation.
struct GpuEamWindowState {
  EamSetflView view{};
  double *dF = nullptr, *dra = nullptr, *drp = nullptr;
  double dens_scale = 0.0, rho_cap = 0.0;

  // growable scratch (sized to the largest window m seen so far)
  int cap_m = 0;
  double *wx = nullptr, *wy = nullptr, *wz = nullptr;
  long *wkey = nullptr;
  int *d_owned = nullptr;
  long long *d_rho = nullptr, *d_fx = nullptr, *d_fy = nullptr, *d_fz = nullptr;
  double *d_fp = nullptr;
  // per-call scalars
  long long *d_pe = nullptr;
  unsigned long long *d_mr = nullptr;
  int *d_of = nullptr;
  unsigned long long sentinel = 0;  // pos_double_bits(1e300) — empty-window seed
  std::mutex mu;

  explicit GpuEamWindowState(const potentials::EamSetfl<double>& setfl) {
    std::vector<double> Fspl = setfl.Fspl, rhoaspl = setfl.rhoaspl, rphispl = setfl.rphispl;
    dF = eam_wf_malloc<double>(Fspl.size());
    dra = eam_wf_malloc<double>(rhoaspl.size());
    drp = eam_wf_malloc<double>(rphispl.size());
    cudaMemcpy(dF, Fspl.data(), Fspl.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dra, rhoaspl.data(), rhoaspl.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(drp, rphispl.data(), rphispl.size() * 8, cudaMemcpyHostToDevice);
    view = EamSetflView{dF, dra, drp, setfl.Nrho, setfl.Nr, setfl.rdrho, setfl.rdr, setfl.rcut};
    const int fb = setfl.density_fracbits();
    dens_scale = (fb == 44) ? core::fixed::FixedAccum<44>::kScale
                            : core::fixed::FixedAccum<40>::kScale;
    rho_cap = setfl.density_grid_max();
    // min_r2 seed = 1e300 (F5: NOT +inf — matches the CPU EAM oracle's 1e300, so
    // CPU↔GPU stays bitwise even on empty windows where no pair updates it).
    double v = 1e300;
    std::memcpy(&sentinel, &v, 8);
    d_pe = eam_wf_malloc<long long>(1);
    d_mr = eam_wf_malloc<unsigned long long>(1);
    d_of = eam_wf_malloc<int>(1);
    grow(64);  // initial scratch
  }

  void grow(int m) {
    if (m <= cap_m) return;
    free_scratch();
    cap_m = m;
    wx = eam_wf_malloc<double>(cap_m);
    wy = eam_wf_malloc<double>(cap_m);
    wz = eam_wf_malloc<double>(cap_m);
    wkey = eam_wf_malloc<long>(cap_m);
    d_owned = eam_wf_malloc<int>(cap_m);
    d_rho = eam_wf_malloc<long long>(cap_m);
    d_fp = eam_wf_malloc<double>(cap_m);
    d_fx = eam_wf_malloc<long long>(cap_m);
    d_fy = eam_wf_malloc<long long>(cap_m);
    d_fz = eam_wf_malloc<long long>(cap_m);
  }

  void free_scratch() {
    for (void* p : {(void*)wx, (void*)wy, (void*)wz, (void*)wkey, (void*)d_owned,
                    (void*)d_rho, (void*)d_fp, (void*)d_fx, (void*)d_fy, (void*)d_fz})
      if (p) cudaFree(p);
    wx = wy = wz = d_fp = nullptr;
    wkey = nullptr;
    d_owned = nullptr;
    d_rho = d_fx = d_fy = d_fz = nullptr;
  }

  ~GpuEamWindowState() {
    free_scratch();
    for (void* p : {(void*)dF, (void*)dra, (void*)drp, (void*)d_pe, (void*)d_mr, (void*)d_of})
      if (p) cudaFree(p);
  }
};

// The window-force policy object EamRing stores by value. Cheap to copy/move
// (shared_ptr to the device state). Matches the CpuEamWindowForce::compute
// signature exactly, so EamRing<Real,Math,GpuEamWindowForce> just works.
struct GpuEamWindowForce {
  std::shared_ptr<GpuEamWindowState> st;

  explicit GpuEamWindowForce(const potentials::EamSetfl<double>& setfl)
      : st(std::make_shared<GpuEamWindowState>(setfl)) {}

  void compute(const double* wx, const double* wy, const double* wz, const long* key,
               int m, const int* owned, int n_owned, const core::PairGeom& geom,
               double /*rho_cap_unused*/, std::vector<core::fixed::ForceAccum>& wFx,
               std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz,
               core::fixed::EnergyAccum& pe, double& min_r2) const {
    using namespace eam_sn_detail;  // kB, ng
    GpuEamWindowState& s = *st;
    std::lock_guard<std::mutex> lk(s.mu);  // serialize device work (correctness)
    if (m <= 0) return;  // empty window: no owned forces, pe/min_r2 unchanged
    s.grow(m);

    // upload the gathered window (key = the actual atom ids — the φ-once order,
    // IDENTICAL to the CPU eam_window_force call; NOT window-local indices).
    cudaMemcpy(s.wx, wx, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wy, wy, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wz, wz, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wkey, key, m * sizeof(long), cudaMemcpyHostToDevice);
    if (n_owned > 0)
      cudaMemcpy(s.d_owned, owned, n_owned * sizeof(int), cudaMemcpyHostToDevice);

    const long long zpe = 0;
    const int zof = 0;
    cudaMemcpy(s.d_pe, &zpe, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_mr, &s.sentinel, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_of, &zof, 4, cudaMemcpyHostToDevice);

    // 3 stream-ordered launches (verbatim E5 kernels) — ρ → F'(ρ) → force.
    eam_density_kernel<<<ng(m), kB>>>(s.wx, s.wy, s.wz, m, geom, s.view, s.dens_scale,
                                      s.d_rho, s.d_of);
    eam_embedding_kernel<<<ng(m), kB>>>(m, s.view, s.dens_scale, s.rho_cap, s.d_rho,
                                        s.d_fp, s.d_of);
    if (n_owned > 0)
      eam_force_kernel<<<ng(n_owned), kB>>>(s.wx, s.wy, s.wz, s.wkey, m, s.d_owned,
                                            n_owned, geom, s.view, s.dens_scale,
                                            s.d_rho, s.d_fp, s.d_fx, s.d_fy, s.d_fz,
                                            s.d_pe, s.d_mr, s.d_of);

    // download the int64 raws + scalars
    int of = 0;
    long long h_pe = 0;
    unsigned long long h_mr = 0;
    std::vector<long long> hfx, hfy, hfz;
    if (n_owned > 0) {
      hfx.resize(m); hfy.resize(m); hfz.resize(m);
      cudaMemcpy(hfx.data(), s.d_fx, m * 8, cudaMemcpyDeviceToHost);
      cudaMemcpy(hfy.data(), s.d_fy, m * 8, cudaMemcpyDeviceToHost);
      cudaMemcpy(hfz.data(), s.d_fz, m * 8, cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(&of, s.d_of, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_pe, s.d_pe, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_mr, s.d_mr, 8, cudaMemcpyDeviceToHost);
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
      throw std::runtime_error(std::string("GpuEamWindowForce: ") + cudaGetErrorString(err));

    // HALT symmetry: the CPU eam_window_force THROWS on ρ>rho_cap; the kernel
    // sets overflow bit 2 (rho-cap) / bit 1 (quantize). Throw ⇒ node_main maps
    // it to Halt::Internal, mirroring the CPU oracle's throw.
    if (of)
      throw std::runtime_error("GpuEamWindowForce: density/rho-cap overflow HALT");

    // merge results into the ring's running accumulators (decoded IDENTICALLY to
    // the CPU policy: ForceAccum.raw is the int64; pe.raw += d_pe; min_r2 = min).
    if (n_owned > 0) {
      for (int o = 0; o < n_owned; ++o) {
        const int loc = owned[o];
        wFx[std::size_t(loc)].raw = hfx[std::size_t(loc)];
        wFy[std::size_t(loc)].raw = hfy[std::size_t(loc)];
        wFz[std::size_t(loc)].raw = hfz[std::size_t(loc)];
      }
    }
    pe.raw += h_pe;  // int64 add is associative ⇒ == CPU's pe.add() multiset sum
    double gpu_mr2;
    std::memcpy(&gpu_mr2, &h_mr, 8);
    if (gpu_mr2 < min_r2) min_r2 = gpu_mr2;
  }
};

}  // namespace tdmd::cuda
