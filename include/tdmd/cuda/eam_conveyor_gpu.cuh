#pragma once
#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/zone_eam.cuh"
#include "tdmd/potentials/eam_spline.hpp"
#include "tdmd/potentials/eam_zone.hpp"  // zone_eam_window, eam_window_layout
#include "tdmd/units.hpp"

// M6 E5b — EAM on the GPU TimeConveyor ring.
//
// THIS FILE (E5b-2): the SINGLE-NODE (z=1) path — the degenerate ring where all
// zones are resident on one node, so there is no inter-node transport and no
// send-delay. It validates the load-bearing NEW EAM-on-GPU logic — the per-zone
// 3-zone window GATHER + the multi-pass (density→embedding→force) E5 kernels
// reused VERBATIM + the scatter + device velocity-Verlet — BITWISE against the
// serial zone_eam_pass oracle (the SingleNodeMatchesSerialVV gate). The window
// index lists come from zone_eam_window (the SAME windows the serial oracle uses);
// φ-once keys are window-local indices, which equal the global-index order
// because zone_eam_window returns a sorted window ⇒ identical PE gating to serial.
// The E5 kernels' int64 accumulation is order-free (B1/INV-9), so the result
// depends only on the window multiset ⇒ GPU ≡ CPU by construction.
//
// The STREAMING multi-node ring (z>1: per-zone DevSlots, StreamTransport, the
// second-forward-hop / one-zone send delay, the restricted-window superset
// oracles) is E5b-3 — see docs/_meta/E5B_GPU_EAM_RING_DESIGN_2026-06-17.md.
namespace tdmd::cuda {

namespace eam_sn_detail {
constexpr int kB = 128;
inline int ng(int n) { return (n + kB - 1) / kB; }

// gather window positions by the static index map; key = window-local index.
__global__ void eam_gather_kernel(const double* gx, const double* gy, const double* gz,
                                  const int* gmap, int m, double* wx, double* wy,
                                  double* wz, long* key) {
  const int aa = blockIdx.x * kB + threadIdx.x;
  if (aa >= m) return;
  const int g = gmap[aa];
  wx[aa] = gx[g]; wy[aa] = gy[g]; wz[aa] = gz[g];
  key[aa] = aa;
}

// scatter owned window-local int64 forces (decoded, Q24.40) to the global force
// array. Each global atom is owned by exactly one zone ⇒ written once ⇒ no
// contention; matches serial's `a.f[g] = wF[loc].value()` (forces pre-zeroed).
__global__ void eam_scatter_kernel(const long long* d_fx, const long long* d_fy,
                                   const long long* d_fz, const int* owned,
                                   const int* gmap, int n_owned, double* Fx,
                                   double* Fy, double* Fz) {
  const int o = blockIdx.x * kB + threadIdx.x;
  if (o >= n_owned) return;
  const int loc = owned[o];
  const int g = gmap[loc];
  Fx[g] = double(d_fx[loc]) / core::fixed::ForceAccum::kScale;
  Fy[g] = double(d_fy[loc]) / core::fixed::ForceAccum::kScale;
  Fz[g] = double(d_fz[loc]) / core::fixed::ForceAccum::kScale;
}

// velocity-Verlet — byte-for-byte the serial arithmetic (integrator.hpp /
// test_eam_ring serial_vv): inv_m=ftm2v/mass; v += 0.5·dt·inv_m·f; x += dt·v.
// --fmad=false ⇒ no FMA fusion ⇒ bitwise == CPU.
__global__ void eam_vv_first_half(double* x, double* y, double* z, double* vx,
                                  double* vy, double* vz, const double* fx,
                                  const double* fy, const double* fz,
                                  const double* mass, int n, double dt) {
  const int i = blockIdx.x * kB + threadIdx.x;
  if (i >= n) return;
  const double inv_m = units::ftm2v / mass[i];
  vx[i] += 0.5 * dt * inv_m * fx[i];
  vy[i] += 0.5 * dt * inv_m * fy[i];
  vz[i] += 0.5 * dt * inv_m * fz[i];
  x[i] += dt * vx[i];
  y[i] += dt * vy[i];
  z[i] += dt * vz[i];
}
__global__ void eam_vv_second_half(double* vx, double* vy, double* vz,
                                   const double* fx, const double* fy,
                                   const double* fz, const double* mass, int n,
                                   double dt) {
  const int i = blockIdx.x * kB + threadIdx.x;
  if (i >= n) return;
  const double inv_m = units::ftm2v / mass[i];
  vx[i] += 0.5 * dt * inv_m * fx[i];
  vy[i] += 0.5 * dt * inv_m * fy[i];
  vz[i] += 0.5 * dt * inv_m * fz[i];
}

template <typename T>
T* up(const std::vector<T>& v) {
  T* d = nullptr;
  cudaMalloc(&d, v.size() * sizeof(T));
  cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
}  // namespace eam_sn_detail

// Single-node (z=1) GPU EAM trajectory: advances `a` in place by `steps` velocity-
// Verlet steps. zd = static zone decomposition (membership fixed for the run,
// width>=2·rcut). setfl = the spline Math (the GPU kernels' contract). Bitwise ==
// serial zone_eam_pass-driven VV. Throws on rho-cap/overflow HALT (symmetric to
// the CPU oracle's throw).
// symmetric=true ⇒ the correct 3-zone window {j-1,j,j+1}; false ⇒ FORWARD-ONLY
// {j,j+1} (drops the lower donor) — the Oracle-A poison that proves a missing
// donor is DETECTABLE. out_f{x,y,z} (optional, size n) receive the forces at the
// final config (with steps=0, the step-0 forces — for the independent-oracle gate).
//
// FIREWALL GAP: this driver runs the EAM SYMMETRIC int64 accumulator (q(j)=−q(i))
// DIRECTLY from a raw EamSetfl — it has NO PassDecl, so it is NOT covered by
// GpuEamWindowForce::assert_supported (which gates only the EamRing path). It is the
// oldest/simplest GPU-EAM entry point (the physics suite copies it) ⇒ the LIKELY MEAM
// reuse site. A MEAM/Tersoff author MUST NOT run a needs_transpose (angular/bond-order,
// non-symmetric, force-to-third-atom-k) potential through this — build the transpose
// accumulator path instead. When MEAM lands, descriptor-gate this driver too (MB1).
template <typename Real>
void eam_gpu_run_singlenode(core::AtomSoA<Real>& a, const core::Box& box,
                            const core::ZoneDecomposition& zd,
                            const potentials::EamSetfl<double>& setfl, long steps,
                            double dt, bool symmetric = true,
                            std::vector<double>* out_fx = nullptr,
                            std::vector<double>* out_fy = nullptr,
                            std::vector<double>* out_fz = nullptr) {
  using namespace eam_sn_detail;
  const int n = a.n;
  const int nz = zd.n_zones;
  const core::PairGeom geom(box, setfl.rcut);
  const double rho_cap = setfl.density_grid_max();
  const int fb = setfl.density_fracbits();
  const double dens_scale = (fb == 44) ? core::fixed::FixedAccum<44>::kScale
                                       : core::fixed::FixedAccum<40>::kScale;

  // static per-zone windows (the SAME windows the serial oracle gathers).
  std::vector<int> gmap_flat, owned_flat, gmap_off{0}, owned_off{0};
  int max_m = 1;
  for (int zi = 0; zi < nz; ++zi) {
    const auto win = potentials::zone_eam_window(zd, zi, box.periodic[2], symmetric);
    std::vector<int> pos(n, -1);
    for (int aa = 0; aa < int(win.size()); ++aa) pos[win[aa]] = aa;
    for (int g : win) gmap_flat.push_back(g);
    for (int g : zd.members[zi]) owned_flat.push_back(pos[g]);  // window-local
    gmap_off.push_back(int(gmap_flat.size()));
    owned_off.push_back(int(owned_flat.size()));
    max_m = std::max(max_m, int(win.size()));
  }

  std::vector<double> hx(a.x), hy(a.y), hz(a.z), hmass(a.mass);
  std::vector<double> hvx(n), hvy(n), hvz(n);
  for (int i = 0; i < n; ++i) { hvx[i] = double(a.vx[i]); hvy[i] = double(a.vy[i]); hvz[i] = double(a.vz[i]); }
  double* dx = up(hx); double* dy = up(hy); double* dz = up(hz);
  double* dvx = up(hvx); double* dvy = up(hvy); double* dvz = up(hvz);
  double* dmass = up(hmass);
  double* dFx = up(std::vector<double>(n, 0.0));
  double* dFy = up(std::vector<double>(n, 0.0));
  double* dFz = up(std::vector<double>(n, 0.0));
  int* d_gmap = up(gmap_flat); int* d_owned = up(owned_flat);

  double* dF = up(setfl.Fspl); double* dra = up(setfl.rhoaspl); double* drp = up(setfl.rphispl);
  EamSetflView view{dF, dra, drp, setfl.Nrho, setfl.Nr, setfl.rdrho, setfl.rdr, setfl.rcut};
  double* wx = up(std::vector<double>(max_m, 0.0));
  double* wy = up(std::vector<double>(max_m, 0.0));
  double* wz = up(std::vector<double>(max_m, 0.0));
  long* wkey = up(std::vector<long>(max_m, 0));
  long long* d_rho = up(std::vector<long long>(max_m, 0));
  double* d_fp = up(std::vector<double>(max_m, 0.0));
  long long* d_fx = up(std::vector<long long>(max_m, 0));
  long long* d_fy = up(std::vector<long long>(max_m, 0));
  long long* d_fz = up(std::vector<long long>(max_m, 0));
  long long* d_pe = up(std::vector<long long>{0});
  unsigned long long sentinel;
  { double v = 1e300; std::memcpy(&sentinel, &v, 8); }
  unsigned long long* d_mr = up(std::vector<unsigned long long>{sentinel});
  int* d_of = up(std::vector<int>{0});

  auto compute_forces = [&]() {
    const long long zpe = 0;
    cudaMemcpy(d_pe, &zpe, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mr, &sentinel, 8, cudaMemcpyHostToDevice);
    for (int zi = 0; zi < nz; ++zi) {
      const int base = gmap_off[zi];
      const int m = gmap_off[zi + 1] - base;
      const int ob = owned_off[zi];
      const int no = owned_off[zi + 1] - ob;
      if (m == 0) continue;
      eam_gather_kernel<<<ng(m), kB>>>(dx, dy, dz, d_gmap + base, m, wx, wy, wz, wkey);
      eam_density_kernel<<<ng(m), kB>>>(wx, wy, wz, m, geom, view, dens_scale, d_rho, d_of);
      eam_embedding_kernel<<<ng(m), kB>>>(m, view, dens_scale, rho_cap, d_rho, d_fp, d_of);
      eam_force_kernel<<<ng(no ? no : 1), kB>>>(wx, wy, wz, wkey, m, d_owned + ob, no,
                                                geom, view, dens_scale, d_rho, d_fp,
                                                d_fx, d_fy, d_fz, d_pe, d_mr, d_of);
      if (no)
        eam_scatter_kernel<<<ng(no), kB>>>(d_fx, d_fy, d_fz, d_owned + ob, d_gmap + base,
                                           no, dFx, dFy, dFz);
    }
  };
  auto check_halt = [&]() {
    int of = 0;
    cudaMemcpy(&of, d_of, 4, cudaMemcpyDeviceToHost);
    if (of) throw std::runtime_error("eam_gpu_run_singlenode: density/rho-cap overflow HALT");
  };

  compute_forces();
  for (long h = 0; h < steps; ++h) {
    eam_vv_first_half<<<ng(n), kB>>>(dx, dy, dz, dvx, dvy, dvz, dFx, dFy, dFz, dmass, n, dt);
    compute_forces();
    eam_vv_second_half<<<ng(n), kB>>>(dvx, dvy, dvz, dFx, dFy, dFz, dmass, n, dt);
  }
  cudaDeviceSynchronize();
  check_halt();

  if (out_fx) { out_fx->resize(n); cudaMemcpy(out_fx->data(), dFx, n * 8, cudaMemcpyDeviceToHost); }
  if (out_fy) { out_fy->resize(n); cudaMemcpy(out_fy->data(), dFy, n * 8, cudaMemcpyDeviceToHost); }
  if (out_fz) { out_fz->resize(n); cudaMemcpy(out_fz->data(), dFz, n * 8, cudaMemcpyDeviceToHost); }
  cudaMemcpy(hx.data(), dx, n * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(hy.data(), dy, n * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(hz.data(), dz, n * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(hvx.data(), dvx, n * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(hvy.data(), dvy, n * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(hvz.data(), dvz, n * 8, cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; ++i) {
    a.x[i] = hx[i]; a.y[i] = hy[i]; a.z[i] = hz[i];
    a.vx[i] = Real(hvx[i]); a.vy[i] = Real(hvy[i]); a.vz[i] = Real(hvz[i]);
  }
  for (void* p : {(void*)dx, (void*)dy, (void*)dz, (void*)dvx, (void*)dvy, (void*)dvz,
                  (void*)dmass, (void*)dFx, (void*)dFy, (void*)dFz, (void*)d_gmap,
                  (void*)d_owned, (void*)dF, (void*)dra, (void*)drp, (void*)wx, (void*)wy,
                  (void*)wz, (void*)wkey, (void*)d_rho, (void*)d_fp, (void*)d_fx,
                  (void*)d_fy, (void*)d_fz, (void*)d_pe, (void*)d_mr, (void*)d_of})
    cudaFree(p);
}

}  // namespace tdmd::cuda
