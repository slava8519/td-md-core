#pragma once
// PERF TRACK — MIXED-PRECISION (production_mixed, B5) EAM GPU kernels.
//
// WHAT THIS IS. The expensive EAM work on the FP64:FP32=1:64 RTX 5080 is the
// per-pair transcendental-free spline Horner eval (eval_phi / eval_rhoa /
// eval_F) plus the IEEE divide in f_over_r — all 1:64-rate FP64. This header
// moves THAT math to FP32. It is a TWIN of zone_eam_verlet.cuh's tight-list
// kernels (the fastest fp64 path): copy-for-copy identical control flow,
// candidate set, φ-once gate, full-neighbour register accumulation and atomicMin,
// with exactly two changes —
//   (1) after the FP64 geometry, r = sqrt(r2) is CAST to float and the spline
//       value/derivative + f_over_r are computed in float32 (EamSetflViewF32);
//   (2) the float contribution is promoted to double (double(float) is EXACT)
//       and quantized with the SAME int64 FixedAccum scale.
//
// GEOMETRY STAYS FP64 (NON-NEGOTIABLE). dx/dy/dz/r2 = PairGeom::reduce min-image
// is FP64 — positions are FP64 by project contract. Only r→float and everything
// downstream of it is FP32.
//
// DETERMINISM (preserved). The accumulation is still int64 fixed-point. Integer
// add is associative ⇒ the per-atom sum is ORDER-FREE ⇒ the mixed kernel is
// GPU-INTERNAL bit-identical run-to-run and 1-vs-z (B1/INV-9), exactly like the
// fp64 path. What is GIVEN UP, ON PURPOSE, is the CPU↔GPU / fp64↔mixed BITWISE
// contract: FP32 spline math ≠ FP64 spline math, so mixed forces differ from the
// fp64 reference at the ~FP32 floor (~1e-6..1e-3 relative). That is the
// established B5 production_mixed precision class, not a bug.
//
// zone_eam.cuh / zone_eam_verlet.cuh / zone_eam_cells.cuh / the rings are
// BYTE-UNTOUCHED — this header only ADDS new kernels + a float spline view and
// reuses the verlet-list build (EamVerletList) verbatim. Still requires
// --fmad=false on the TU only for run-to-run determinism of the FP32 math
// (1-vs-z bitwise — same empirical guarantee as bench_conveyor_fma/bench_eam_fma;
// the CPU↔GPU bar is already abandoned here, so the flag is for self-consistency).
#include <cuda_runtime.h>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/zones.hpp"             // PairGeom (FP64 min-image — geometry stays FP64)
#include "tdmd/cuda/zone_eam.cuh"          // EamSetflView (double), quantize, pos_double_bits, kZoneBlock
#include "tdmd/cuda/zone_eam_verlet.cuh"   // EamVerletList + eam_build_verlet_list (reused verbatim)

namespace tdmd::cuda {

// ---------------------------------------------------------------------------
// Float device view of the setfl spline coeffs. The double coefficients are
// cast to float ONCE on the host (make_setfl_view_f32) into device float arrays;
// the eval is the SAME LAMMPS 7-coeff Horner + integer knot index, in float.
// rdr/rdrho/rcut are kept as float too (the knot-index arithmetic is float).
// ---------------------------------------------------------------------------
struct EamSetflViewF32 {
  const float* Fspl;     // 7*(Nrho+1) floats (device)
  const float* rhoaspl;  // 7*(Nr+1)
  const float* rphispl;  // 7*(Nr+1)
  int Nrho = 0, Nr = 0;
  float rdrho = 0, rdr = 0, rcut = 0;

  // Float twin of EamSetfl<double>::eval_spline — identical structure, float
  // arithmetic. Knot index is int (truncation toward zero; x>=0), no rounding.
  __device__ static void eval_spline(const float* spl, int n, float rd, float x,
                                     float& v, float& dv) {
    float p = x * rd + 1.0f;
    int m = int(p);
    if (m < 1) m = 1;
    if (m > n - 1) m = n - 1;
    p -= float(m);
    if (p > 1.0f) p = 1.0f;
    const float* c = spl + 7 * m;
    v = ((c[3] * p + c[4]) * p + c[5]) * p + c[6];
    dv = (c[0] * p + c[1]) * p + c[2];
  }
  __device__ void eval_F(float rho, float& v, float& dv) const {
    eval_spline(Fspl, Nrho, rdrho, rho, v, dv);
  }
  __device__ void eval_rhoa(float r, float& v, float& dv) const {
    eval_spline(rhoaspl, Nr, rdr, r, v, dv);
  }
  __device__ void eval_phi(float r, float& v, float& dv) const {
    float z2, z2p;
    eval_spline(rphispl, Nr, rdr, r, z2, z2p);
    const float rec = 1.0f / r;           // FP32 reciprocal (1:64-rate divide moved off FP64)
    v = z2 * rec;
    dv = (z2p - v) * rec;
  }
};

// Host-side: cast the double setfl coeffs to float, upload, return the view +
// the device pointers so the caller can free them. Allocates 3 float arrays.
struct DevSetflF32 {
  EamSetflViewF32 view;
  float *F = nullptr, *ra = nullptr, *rp = nullptr;
};
inline DevSetflF32 upload_setfl_f32(const EamSetflView& d64, int Nrho, int Nr,
                                    double rdrho, double rdr, double rcut,
                                    const std::vector<double>& Fspl,
                                    const std::vector<double>& rhoaspl,
                                    const std::vector<double>& rphispl) {
  (void)d64;
  DevSetflF32 d;
  auto cast_up = [](const std::vector<double>& src, float** dst) {
    std::vector<float> f(src.size());
    for (size_t i = 0; i < src.size(); ++i) f[i] = float(src[i]);
    cudaMalloc(dst, f.size() * sizeof(float));
    cudaMemcpy(*dst, f.data(), f.size() * sizeof(float), cudaMemcpyHostToDevice);
  };
  cast_up(Fspl, &d.F);
  cast_up(rhoaspl, &d.ra);
  cast_up(rphispl, &d.rp);
  d.view = EamSetflViewF32{d.F,         d.ra,       d.rp, Nrho, Nr,
                           float(rdrho), float(rdr), float(rcut)};
  return d;
}
inline void free_setfl_f32(DevSetflF32& d) {
  for (void* p : {(void*)d.F, (void*)d.ra, (void*)d.rp})
    if (p) cudaFree(p);
  d = DevSetflF32{};
}

// ===========================================================================
// pass 1 (MIXED / TIGHT LIST): ρ_aa = Σ ρ_a(r). TWIN of eam_density_verlet_kernel
// — same int64 register, same FP64 in-cutoff re-test (geom.reduce, r2 < rcut²),
// same dens_scale, single d_rho[aa] write. ONLY change: r→float, eval_rhoa in
// float, quantize(double(float v), ...). int64 accumulation ⇒ order-free.
// ===========================================================================
__global__ void eam_density_mixed_kernel(const double* wx, const double* wy,
                                         const double* wz, int m,
                                         core::PairGeom geom, EamSetflViewF32 eam,
                                         double dens_scale, const int* d_off,
                                         const int* d_idx, long long* d_rho,
                                         int* overflow) {
  const int aa = blockIdx.x * kZoneBlock + threadIdx.x;
  if (aa >= m) return;
  long long qrho = 0;
  const double xi = wx[aa], yi = wy[aa], zi = wz[aa];
  const int beg = d_off[aa], end = d_off[aa + 1];
  for (int t = beg; t < end; ++t) {
    const int bb = d_idx[t];
    double ddx = xi - wx[bb], ddy = yi - wy[bb], ddz = zi - wz[bb], r2;  // GEOMETRY FP64
    if (!geom.reduce(ddx, ddy, ddz, r2)) continue;                      // EXACT rcut re-test (FP64)
    const float r = float(sqrt(r2));                                    // r → float
    float v, dv;
    eam.eval_rhoa(r, v, dv);                                            // spline in FP32
    qrho += quantize(double(v), dens_scale, overflow);                  // exact float→double, same scale
  }
  d_rho[aa] = qrho;
}

// ===========================================================================
// pass 2 (MIXED): F'(ρ) for ALL window atoms + rho-cap HALT (bit 2). TWIN of
// eam_embedding_kernel — ρ is reconstructed from the int64 d_rho exactly as the
// fp64 path (double(d_rho[aa]) / dens_scale, the FixedAccum::value), then F'(ρ)
// is evaluated in FLOAT. d_fp carries Fp as a double (the float result widened —
// exact), consumed by the force kernel's float math.
// ===========================================================================
__global__ void eam_embedding_mixed_kernel(int m, EamSetflViewF32 eam,
                                           double dens_scale, double rho_cap,
                                           const long long* d_rho, double* d_fp,
                                           int* overflow) {
  const int aa = blockIdx.x * kZoneBlock + threadIdx.x;
  if (aa >= m) return;
  const double rho = double(d_rho[aa]) / dens_scale;   // == FixedAccum::value() (FP64 reconstruct)
  if (rho > rho_cap) atomicOr(overflow, 2);
  float F, Fp;
  eam.eval_F(float(rho), F, Fp);                       // embedding derivative in FP32
  d_fp[aa] = double(Fp);                               // widen (exact); the force kernel re-narrows
}

// ===========================================================================
// pass 3 (MIXED / TIGHT LIST): full-neighbour force for OWNED atoms + pe. TWIN of
// eam_force_verlet_kernel. IDENTICAL control flow (embedding once per owned,
// φ-once gate ki<key[bb], qfx/y/z registers, sred tree-reduce + one atomicAdd,
// atomicMin BEFORE the cutoff). The ONLY changes are FP32 math:
//   r → float; eval_F/eval_phi/eval_rhoa in float; f_over_r in float (FP32
//   reciprocal); quantize(double(float·dx)). Geometry (dx,r2) stays FP64.
// ===========================================================================
__global__ void eam_force_mixed_kernel(
    const double* wx, const double* wy, const double* wz, const long* key, int m,
    const int* owned, int n_owned, core::PairGeom geom, EamSetflViewF32 eam,
    double dens_scale, const long long* d_rho, const double* d_fp,
    const int* d_off, const int* d_idx, long long* d_fx, long long* d_fy,
    long long* d_fz, long long* d_pe, unsigned long long* d_min_r2,
    int* overflow) {
  __shared__ long long sred[kZoneBlock];
  const int o = blockIdx.x * kZoneBlock + threadIdx.x;
  long long qpe = 0;
  if (o < n_owned) {
    const int ii = owned[o];
    long long qfx = 0, qfy = 0, qfz = 0;
    double mr2 = 1e300;
    float Fi, Fpi;
    eam.eval_F(float(double(d_rho[ii]) / dens_scale), Fi, Fpi);  // embedding ENERGY once per owned (FP32)
    qpe += quantize(double(Fi), core::fixed::EnergyAccum::kScale, overflow);
    const double xi = wx[ii], yi = wy[ii], zi = wz[ii];
    const float fpi = float(d_fp[ii]);                            // F'(ρ_ii) (narrow back to FP32)
    const long ki = key[ii];
    const int beg = d_off[ii], end = d_off[ii + 1];
    for (int t = beg; t < end; ++t) {
      const int bb = d_idx[t];
      double ddx = xi - wx[bb], ddy = yi - wy[bb], ddz = zi - wz[bb], r2;  // GEOMETRY FP64
      const bool ok = geom.reduce(ddx, ddy, ddz, r2);
      mr2 = fmin(mr2, r2);                                        // overlap probe (FP64), before cutoff
      if (!ok) continue;
      const float r = float(sqrt(r2));                           // r → float
      float phi, dphi, ra, dra;
      eam.eval_phi(r, phi, dphi);                                // FP32 spline
      eam.eval_rhoa(r, ra, dra);                                 // FP32 spline
      const float fpj = float(d_fp[bb]);                         // F'(ρ_bb)
      const float f_over_r = -(dphi + (fpi + fpj) * dra) / r;    // FP32 force prefactor (FP32 divide)
      qfx += quantize(double(f_over_r) * ddx, core::fixed::ForceAccum::kScale, overflow);  // float math, FP64 dx*scale, exact widen
      qfy += quantize(double(f_over_r) * ddy, core::fixed::ForceAccum::kScale, overflow);
      qfz += quantize(double(f_over_r) * ddz, core::fixed::ForceAccum::kScale, overflow);
      if (ki < key[bb]) qpe += quantize(double(phi), core::fixed::EnergyAccum::kScale, overflow);
    }
    d_fx[ii] = qfx; d_fy[ii] = qfy; d_fz[ii] = qfz;  // window-local ii
    atomicMin(d_min_r2, pos_double_bits(mr2));
  }
  // per-block int64 tree-reduce of qpe → one atomicAdd (zone_eam.cuh epilogue)
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
