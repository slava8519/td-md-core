#pragma once
#include <cuda_runtime.h>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/zones.hpp"          // PairGeom (HOST_DEVICE reduce)
#include "tdmd/cuda/zone_force.cuh"     // quantize, pos_double_bits, kZoneBlock
#include "tdmd/potentials/tersoff.hpp"  // TersoffParams + all ters_* helpers (already HOST_DEVICE)

// M6 / Tersoff-ladder Te5 — GPU Tersoff force: the φ₃/zetaterm TRANSPOSE accumulator on the
// device (force to a third atom k, q(k)≠−q(i)) the symmetric EAM GPU int64 accumulator cannot
// run. Design of record: wf_d662cfdd-eca. [ENG].
//
// HONEST CONTRACT (the INVERSE of EAM's transcendental-free bitwise spline):
//   CPU↔GPU is TOLERANCE (~1e-9 force / ~1e-6 PE), NOT bitwise — ters_fc uses sin/cos, ters_fa/
//   ex_delr use exp, ters_bij uses pow/sqrt — ALL diverge libm-vs-CUDA ~1 ulp (the Morse/SW
//   situation). GPU-INTERNAL determinism (run-to-run, 1-vs-z) IS bitwise by B1 int64 register
//   order-freedom (write-once, no cross-atom force write, no force atomics) — the ring's
//   block-order window depends only on the zone id, NOT on the node count z. The host policy
//   ALSO pre-sorts the window by global key before the kernel — ζ_ij is an FP64 ORDER-SENSITIVE
//   sum (SW had none), so the sort (a) makes window-permutation bitwise BY CONSTRUCTION and
//   (b) keeps the GPU window canonically-aligned with the CPU serial (zone_eam_window sorts).
//   MEASURED (the Te3 finding, S2): a SINGLE force eval is sub-Q24.40-quantum sorted-vs-unsorted,
//   so the sort is DEFENSIVE (not strictly required for the current GPU gates) but robust + CPU-
//   consistent. --fmad=false is REQUIRED to keep GPU-internal determinism STRUCTURAL. The
//   independent FP64 oracle (tersoff_direct_fp64) <1e-9 is the only MB2 dropped-donor witness;
//   FD-of-energy + the LAMMPS golden (Te1) are the algebra witnesses.
namespace tdmd::cuda {

// per-thread cap on o's OWN in-rcut neighbour list (Role 1 + Role 2; NOT the window size m).
// Tersoff-Si rcut=bigr+bigd=3.2 Å ⇒ ~4 first-shell nbrs (2.35 Å); 2nd shell 3.84 Å is OUTSIDE
// ⇒ 16× headroom. A cap OVERFLOW is a guarded HALT (sticky bit 4, thrown BEFORE writeback),
// NEVER a silent truncation. (Roles 3/4 re-scan the window for a halo center i's neighbours into
// registers — no local buffer, so they cannot overflow; their cost is unbounded.)
inline constexpr int kMaxNbr = 64;

// ζ_{c,x} = Σ_{k∈window, k a nbr of c, k≠x} tersoff_zeta(...), in ASCENDING SLOT order (= ascending
// global, because the host pre-sorted the window). The device mirror of tersoff_window_force's
// zeta_replay (tersoff_zone.hpp). DO NOT reorder/filter/early-exit — the bitwise ζ-order invariant
// depends on the front-to-back walk over the host-sorted window. FP64 — NEVER a FixedAccum.
__device__ inline double zeta_center(const double* wx, const double* wy, const double* wz, int m,
                                     int c, int x_local, double rcx, const double* rcxh,
                                     core::PairGeom geom, const potentials::TersoffParams& p) {
  const double xc = wx[c], yc = wy[c], zc = wz[c];
  double zeta = 0.0;
  for (int k = 0; k < m; ++k) {
    if (k == c || k == x_local) continue;
    double dx = xc - wx[k], dy = yc - wy[k], dz = zc - wz[k], r2;
    if (!geom.reduce(dx, dy, dz, r2)) continue;
    const double rk = sqrt(r2), rkinv = 1.0 / rk;
    const double rckh[3] = {-dx * rkinv, -dy * rkinv, -dz * rkinv};  // (x_k−x_c)/r LAMMPS hat
    zeta += potentials::tersoff_zeta(rcx, rk, rcxh, rckh, p);
  }
  return zeta;
}

// One thread per OWNED atom o: the whole 4-role transpose-replay over the (host-sorted) gathered
// window into int64 REGISTERS, write-once. d_fx/fy/fz indexed by window-local o. PE/n_bonds/
// n_triplets cross threads via block tree-reduce + atomicAdd (test-only witnesses).
__global__ void tersoff_force_kernel(
    const double* wx, const double* wy, const double* wz, const long* key, int m,
    const int* owned, int n_owned, core::PairGeom geom, potentials::TersoffParams tp,
    long long* d_fx, long long* d_fy, long long* d_fz, long long* d_pe, long long* d_nbonds,
    long long* d_ntri, unsigned long long* d_min_r2, int* overflow) {
  __shared__ long long sred[kZoneBlock];
  const int t = blockIdx.x * kZoneBlock + threadIdx.x;
  const double fscale = core::fixed::ForceAccum::kScale;
  const double escale = core::fixed::EnergyAccum::kScale;
  long long qpe = 0, qnb = 0, qnt = 0;

  if (t < n_owned) {
    const int o = owned[t];
    long long qfx = 0, qfy = 0, qfz = 0;
    double mr2 = 1e300;
    const double xo = wx[o], yo = wy[o], zo = wz[o];
    const long ko = key[o];

    // o's own neighbour list, cached once (dx = xo − wx[b], PROJECT convention).
    int nb[kMaxNbr];
    double nbdx[kMaxNbr], nbdy[kMaxNbr], nbdz[kMaxNbr], nbr_[kMaxNbr];
    int cnt = 0;
    for (int b = 0; b < m; ++b) {
      if (b == o) continue;
      double dx = xo - wx[b], dy = yo - wy[b], dz = zo - wz[b], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
      mr2 = fmin(mr2, r2);
      if (cnt < kMaxNbr) { nb[cnt] = b; nbdx[cnt] = dx; nbdy[cnt] = dy; nbdz[cnt] = dz; nbr_[cnt] = sqrt(r2); ++cnt; }
      else atomicOr(overflow, 4);  // STICKY neighbour-cap HALT
    }

    // Role 1 — PAIR/repulsive: o's symmetric force; energy fc·fR + n_bonds by the LOWER GLOBAL KEY.
    for (int e = 0; e < cnt; ++e) {
      const double r = nbr_[e], tmp_exp = exp(-tp.lam1 * r);
      const double fforce = -tp.biga * tmp_exp * (potentials::ters_fc_d(r, tp) - potentials::ters_fc(r, tp) * tp.lam1) / r;
      qfx += quantize(nbdx[e] * fforce, fscale, overflow);
      qfy += quantize(nbdy[e] * fforce, fscale, overflow);
      qfz += quantize(nbdz[e] * fforce, fscale, overflow);
      if (ko < key[nb[e]]) { qpe += quantize(potentials::ters_fc(r, tp) * tp.biga * tmp_exp, escale, overflow); ++qnb; }
    }

    // Role 2 — CENTER-i of bond (o,j): attractive-radial f_i + angular f_i; OWNS attr-PE + count.
    for (int e = 0; e < cnt; ++e) {
      const double rij = nbr_[e], rijinv = 1.0 / rij;
      const double rijh[3] = {-nbdx[e] * rijinv, -nbdy[e] * rijinv, -nbdz[e] * rijinv};  // (x_j−x_o)/r
      const double zeta = zeta_center(wx, wy, wz, m, o, nb[e], rij, rijh, geom, tp);
      const double fa = potentials::ters_fa(rij, tp), fa_d = potentials::ters_fa_d(rij, tp), bij = potentials::ters_bij(zeta, tp);
      const double fpair_z = 0.5 * bij * fa_d * rijinv;
      const double prefactor = -0.5 * fa * potentials::ters_bij_d(zeta, tp);
      qpe += quantize(0.5 * bij * fa, escale, overflow);  // attractive energy, ONCE
      qfx += quantize(-nbdx[e] * fpair_z, fscale, overflow);
      qfy += quantize(-nbdy[e] * fpair_z, fscale, overflow);
      qfz += quantize(-nbdz[e] * fpair_z, fscale, overflow);
      for (int kk = 0; kk < cnt; ++kk) {
        if (kk == e) continue;
        const double rikinv = 1.0 / nbr_[kk];
        const double rikh[3] = {-nbdx[kk] * rikinv, -nbdy[kk] * rikinv, -nbdz[kk] * rikinv};
        double fi[3], fj[3], fk[3];
        potentials::tersoff_zetaterm_d(prefactor, rijh, rij, rijinv, rikh, nbr_[kk], rikinv, fi, fj, fk, tp);
        qfx += quantize(fi[0], fscale, overflow);
        qfy += quantize(fi[1], fscale, overflow);
        qfz += quantize(fi[2], fscale, overflow);
        ++qnt;  // count HERE only
      }
    }

    // Role 3 — ENDPOINT-j of bond (i,o): attractive-radial f_j + angular f_j; NO energy/count.
    for (int e = 0; e < cnt; ++e) {
      const int i = nb[e];
      const double rij = nbr_[e], rijinv = 1.0 / rij;
      const double rijh[3] = {nbdx[e] * rijinv, nbdy[e] * rijinv, nbdz[e] * rijinv};  // (x_o−x_i)/r (opp Role 2)
      const double zeta = zeta_center(wx, wy, wz, m, i, o, rij, rijh, geom, tp);
      const double fa = potentials::ters_fa(rij, tp), fa_d = potentials::ters_fa_d(rij, tp), bij = potentials::ters_bij(zeta, tp);
      const double fpair_z = 0.5 * bij * fa_d * rijinv;
      const double prefactor = -0.5 * fa * potentials::ters_bij_d(zeta, tp);
      qfx += quantize(-nbdx[e] * fpair_z, fscale, overflow);
      qfy += quantize(-nbdy[e] * fpair_z, fscale, overflow);
      qfz += quantize(-nbdz[e] * fpair_z, fscale, overflow);
      const double xi = wx[i], yi = wy[i], zi = wz[i];
      for (int b = 0; b < m; ++b) {
        if (b == i || b == o) continue;
        double dx = xi - wx[b], dy = yi - wy[b], dz = zi - wz[b], r2;
        if (!geom.reduce(dx, dy, dz, r2)) continue;
        const double rk = sqrt(r2), rkinv = 1.0 / rk;
        const double rikh[3] = {-dx * rkinv, -dy * rkinv, -dz * rkinv};
        double fi[3], fj[3], fk[3];
        potentials::tersoff_zetaterm_d(prefactor, rijh, rij, rijinv, rikh, rk, rkinv, fi, fj, fk, tp);
        qfx += quantize(fj[0], fscale, overflow);
        qfy += quantize(fj[1], fscale, overflow);
        qfz += quantize(fj[2], fscale, overflow);
      }
    }

    // Role 4 — THIRD-k of triplet (i;j,o): angular f_k only; NO energy/count.
    for (int e = 0; e < cnt; ++e) {
      const int i = nb[e];
      const double rio = nbr_[e], rioinv = 1.0 / rio;
      const double riohat[3] = {nbdx[e] * rioinv, nbdy[e] * rioinv, nbdz[e] * rioinv};  // (x_o−x_i)/r = k-hat
      const double xi = wx[i], yi = wy[i], zi = wz[i];
      for (int b = 0; b < m; ++b) {
        if (b == i || b == o) continue;
        double dx = xi - wx[b], dy = yi - wy[b], dz = zi - wz[b], r2;
        if (!geom.reduce(dx, dy, dz, r2)) continue;
        const double rij = sqrt(r2), rijinv = 1.0 / rij;
        const double rijh[3] = {-dx * rijinv, -dy * rijinv, -dz * rijinv};  // (x_j−x_i)/r
        const double zeta = zeta_center(wx, wy, wz, m, i, b, rij, rijh, geom, tp);
        const double prefactor = -0.5 * potentials::ters_fa(rij, tp) * potentials::ters_bij_d(zeta, tp);
        double fi[3], fj[3], fk[3];
        potentials::tersoff_zetaterm_d(prefactor, rijh, rij, rijinv, riohat, rio, rioinv, fi, fj, fk, tp);
        qfx += quantize(fk[0], fscale, overflow);
        qfy += quantize(fk[1], fscale, overflow);
        qfz += quantize(fk[2], fscale, overflow);
      }
    }

    d_fx[o] = qfx; d_fy[o] = qfy; d_fz[o] = qfz;  // window-local o, write-once
    atomicMin(d_min_r2, pos_double_bits(mr2));
  }

  // three block tree-reduces → atomicAdd (PE, n_bonds, n_triplets), reusing sred.
  sred[threadIdx.x] = qpe; __syncthreads();
  for (int s = kZoneBlock / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s]; __syncthreads(); }
  if (threadIdx.x == 0) atomicAdd(reinterpret_cast<unsigned long long*>(d_pe), (unsigned long long)sred[0]);
  __syncthreads();
  sred[threadIdx.x] = qnb; __syncthreads();
  for (int s = kZoneBlock / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s]; __syncthreads(); }
  if (threadIdx.x == 0) atomicAdd(reinterpret_cast<unsigned long long*>(d_nbonds), (unsigned long long)sred[0]);
  __syncthreads();
  sred[threadIdx.x] = qnt; __syncthreads();
  for (int s = kZoneBlock / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s]; __syncthreads(); }
  if (threadIdx.x == 0) atomicAdd(reinterpret_cast<unsigned long long*>(d_ntri), (unsigned long long)sred[0]);
}

}  // namespace tdmd::cuda
