#pragma once
#include "tdmd/core/zones.hpp"             // PairGeom (single-source min-image/cutoff)
#include "tdmd/cuda/zone_force.cuh"        // quantize, pos_double_bits, kZoneBlock
#include "tdmd/potentials/eam_spline.hpp"  // EamSetfl<double>::eval_spline (single source)

// M6 PR-E5 — GPU EAM 3-pass force (density → embedding → force) in int64
// fixed-point, BITWISE-EQUAL to CPU eam_window_force (eam_zone.hpp) in
// deterministic_fp64. Standalone (one already-gathered window); the GPU-ring
// integration (conveyor_gpu) is E5b. The spline is transcendental-free, so
// CPU↔GPU is strictly bitwise (==, the LJ bar) UNDER --fmad=false — every EAM
// .cu including this MUST link the `tdmd_eam_cuda_flags` INTERFACE target (the
// force-prefactor and the Horner steps fuse to fma otherwise and diverge 1 ULP).
//
// Per-atom thread, int64 accumulation in a register: density and force write
// only their OWN atom (no cross-thread write, no atomics) ⇒ the per-atom raw is
// order-free (integer add is associative, B1/INV-9) ⇒ identical to the CPU's
// per-atom multiset. Three SEPARATE stream-ordered launches give the
// producer→consumer barrier (ρ complete → F' complete → force).
namespace tdmd::cuda {

// POD device view of the EamSetfl spline coeffs (always double; passed by value).
struct EamSetflView {
  const double* Fspl;     // 7*(Nrho+1) doubles (device)
  const double* rhoaspl;  // 7*(Nr+1)
  const double* rphispl;  // 7*(Nr+1)
  int Nrho = 0, Nr = 0;
  double rdrho = 0, rdr = 0, rcut = 0;
  TDMD_HOST_DEVICE void eval_F(double rho, double& v, double& dv) const {
    potentials::EamSetfl<double>::eval_spline(Fspl, Nrho, rdrho, rho, v, dv);
  }
  TDMD_HOST_DEVICE void eval_rhoa(double r, double& v, double& dv) const {
    potentials::EamSetfl<double>::eval_spline(rhoaspl, Nr, rdr, r, v, dv);
  }
  TDMD_HOST_DEVICE void eval_phi(double r, double& v, double& dv) const {
    double z2, z2p;
    potentials::EamSetfl<double>::eval_spline(rphispl, Nr, rdr, r, z2, z2p);
    const double rec = 1.0 / r;       // FP-site: true IEEE divide (--prec-div=true)
    v = z2 * rec;
    dv = (z2p - v) * rec;
  }
};

// pass 1: ρ_aa = Σ_{bb≠aa} ρ_a(r) — int64 register, no cross-write (B1 order-free).
__global__ void eam_density_kernel(const double* wx, const double* wy, const double* wz,
                                   int m, core::PairGeom geom, EamSetflView eam,
                                   double dens_scale, long long* d_rho, int* overflow) {
  const int aa = blockIdx.x * kZoneBlock + threadIdx.x;
  if (aa >= m) return;
  long long qrho = 0;
  const double xi = wx[aa], yi = wy[aa], zi = wz[aa];
  for (int bb = 0; bb < m; ++bb) {
    if (bb == aa) continue;
    double dx = xi - wx[bb], dy = yi - wy[bb], dz = zi - wz[bb], r2;
    if (!geom.reduce(dx, dy, dz, r2)) continue;
    double v, dv;
    eam.eval_rhoa(sqrt(r2), v, dv);
    qrho += quantize(v, dens_scale, overflow);
  }
  d_rho[aa] = qrho;
}

// pass 2: F'_aa = F'(value(ρ_aa)) for ALL window atoms; rho-cap HALT (bit 2).
__global__ void eam_embedding_kernel(int m, EamSetflView eam, double dens_scale,
                                     double rho_cap, const long long* d_rho,
                                     double* d_fp, int* overflow) {
  const int aa = blockIdx.x * kZoneBlock + threadIdx.x;
  if (aa >= m) return;
  const double rho = double(d_rho[aa]) / dens_scale;  // == FixedAccum::value()
  if (rho > rho_cap) atomicOr(overflow, 2);
  double F, Fp;
  eam.eval_F(rho, F, Fp);
  d_fp[aa] = Fp;
}

// pass 3: full-neighbour force for OWNED atoms only + pe (embedding F once per
// owned + φ once per undirected pair under key[ii]<key[bb]).
__global__ void eam_force_kernel(const double* wx, const double* wy, const double* wz,
                                 const long* key, int m, const int* owned, int n_owned,
                                 core::PairGeom geom, EamSetflView eam, double dens_scale,
                                 const long long* d_rho, const double* d_fp,
                                 long long* d_fx, long long* d_fy, long long* d_fz,
                                 long long* d_pe, unsigned long long* d_min_r2,
                                 int* overflow) {
  __shared__ long long sred[kZoneBlock];
  const int o = blockIdx.x * kZoneBlock + threadIdx.x;
  long long qpe = 0;
  if (o < n_owned) {
    const int ii = owned[o];
    long long qfx = 0, qfy = 0, qfz = 0;
    double mr2 = 1e300;
    double Fi, Fpi;
    eam.eval_F(double(d_rho[ii]) / dens_scale, Fi, Fpi);  // embedding energy once per owned
    qpe += quantize(Fi, core::fixed::EnergyAccum::kScale, overflow);
    const double xi = wx[ii], yi = wy[ii], zi = wz[ii];
    const double fpi = d_fp[ii];
    const long ki = key[ii];
    for (int bb = 0; bb < m; ++bb) {
      if (bb == ii) continue;
      double dx = xi - wx[bb], dy = yi - wy[bb], dz = zi - wz[bb], r2;
      const bool ok = geom.reduce(dx, dy, dz, r2);
      mr2 = fmin(mr2, r2);  // BEFORE the cutoff (pass3 only) — overlap probe
      if (!ok) continue;
      const double r = sqrt(r2);
      double phi, dphi, ra, dra;
      eam.eval_phi(r, phi, dphi);
      eam.eval_rhoa(r, ra, dra);
      const double f_over_r = -(dphi + (fpi + d_fp[bb]) * dra) / r;  // FMA-killer site
      qfx += quantize(f_over_r * dx, core::fixed::ForceAccum::kScale, overflow);
      qfy += quantize(f_over_r * dy, core::fixed::ForceAccum::kScale, overflow);
      qfz += quantize(f_over_r * dz, core::fixed::ForceAccum::kScale, overflow);
      if (ki < key[bb]) qpe += quantize(phi, core::fixed::EnergyAccum::kScale, overflow);
    }
    d_fx[ii] = qfx; d_fy[ii] = qfy; d_fz[ii] = qfz;  // window-local ii
    atomicMin(d_min_r2, pos_double_bits(mr2));
  }
  // per-block int64 tree-reduce of qpe → one atomicAdd (zone_force.cuh epilogue)
  sred[threadIdx.x] = qpe;
  __syncthreads();
  for (int s = kZoneBlock / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicAdd(reinterpret_cast<unsigned long long*>(d_pe),
              static_cast<unsigned long long>(sred[0]));
}

}  // namespace tdmd::cuda
