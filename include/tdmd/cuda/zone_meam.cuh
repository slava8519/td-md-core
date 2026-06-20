#pragma once
#include <cuda_runtime.h>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/zones.hpp"        // PairGeom (HOST_DEVICE reduce)
#include "tdmd/cuda/zone_force.cuh"   // quantize, pos_double_bits, kZoneBlock
#include "tdmd/potentials/meam.hpp"   // MeamParamsView + all the HOST_DEVICE Me5 helpers

// MEAM-ladder Me5 — GPU MEAM: the screened many-body force on the device. THE HARDEST kernel of
// the project: screening S_ij = Π_k S(C_ijk) (the candidate set depends on the THIRD atom k) +
// the four angular partial densities ρ⁰..ρ³ + embedding F(ρ̄) + the 3rd-atom ∂S/∂x_k TRANSPOSE
// force (force to atom k, q(k)≠−q(i)) the symmetric EAM GPU int64 accumulator cannot run.
// Design of record: docs/_meta/M5_MEAM_GPU_DESIGN_2026-06-20.md. [ENG].
//
// THREE kernels (the EAM density→embedding→force split, screening folded in):
//   K1 meam_density_kernel — one thread per WINDOW atom c: scan c's window neighbours j, screen
//      each (meam_getscreen_d_device, the whole-window k-scan), accumulate the 27 angular density
//      lanes into thread-local int64 registers (Q24.40, ForceAccum::kScale — matching the CPU
//      FixedDens), write 27 lanes per atom. Order-free by B1.
//   K2 meam_embed_kernel — one thread per WINDOW atom c (owned AND HALO — Role B/C read ed[i] of a
//      halo center i): decode the 27 int64 lanes → MeamDensity, meam_dens_final_deriv_pod →
//      MeamEmbedDeriv → d_ed[c]; pe_embed for owned.
//   K3 meam_force_kernel — one thread per OWNED atom o: the 3-role transpose-replay in int64
//      REGISTERS, write-once (no cross-atom write, no force atomics ⇒ B1 order-free). kMaxNbr=64
//      cap on o's OWN list + guarded HALT.
//
// HONEST CONTRACT (the INVERSE of EAM's transcendental-free bitwise spline): CPU↔GPU is TOLERANCE
// (~1e-9 force / ~1e-6 PE) — exp/log/pow in rhoa*/G_gam/embedding diverge libm-vs-CUDA ~1 ulp.
// GPU-INTERNAL determinism (run-to-run, 1-vs-z) IS bitwise by B1 int64 register order-freedom
// (write-once; --fmad=false REQUIRED). The φ-spline (MeamParamsView) is itself bitwise. The
// independent FP64 oracle (meam_direct_fp64) on a PARTIAL-screening fixture is the SOLE
// dropped-screening-k witness (the diamond is binary-S ⇒ Role C structurally DEAD).
namespace tdmd::cuda {

// 27 density lanes, per-atom contiguous: d_dens[kDensLanes*c + lane].
//   0       rho0
//   1       arho2b
//   2..4    arho1[3]
//   5..10   arho2[6]
//   11..20  arho3[10]
//   21..23  arho3b[3]
//   24..26  t_ave[3]
inline constexpr int kDensLanes = 27;

// per-thread cap on o's OWN in-rc neighbour list (Role A/B/C re-scan i's neighbours from the
// window WITHOUT a buffer ⇒ cannot overflow; only o's own cached list hits kMaxNbr). MEAM-Si at
// rc=4.0 sees ~16 in-rc pre-screening (4×1NN@2.35 + 12×2NN@3.84<4.0); 64 = ~4× headroom. A cap
// overflow is a guarded HALT (sticky bit 4, thrown BEFORE writeback), NEVER a silent truncation.
inline constexpr int kMeamMaxNbr = 64;

// Decode the 27 int64 density lanes of window atom c into a stack MeamDensity (== FixedAccum::value()).
__device__ inline potentials::MeamDensity meam_decode_dens(const long long* d_dens, int c,
                                                           double dens_scale) {
  potentials::MeamDensity d;
  const long long* L = d_dens + kDensLanes * c;
  d.rho0 = double(L[0]) / dens_scale;
  d.arho2b = double(L[1]) / dens_scale;
  for (int m = 0; m < 3; ++m) d.arho1[m] = double(L[2 + m]) / dens_scale;
  for (int m = 0; m < 6; ++m) d.arho2[m] = double(L[5 + m]) / dens_scale;
  for (int m = 0; m < 10; ++m) d.arho3[m] = double(L[11 + m]) / dens_scale;
  for (int m = 0; m < 3; ++m) d.arho3b[m] = double(L[21 + m]) / dens_scale;
  for (int m = 0; m < 3; ++m) d.t_ave[m] = double(L[24 + m]) / dens_scale;
  return d;
}

// ----------------------------------------------------------------------------------------
// K1 — per WINDOW atom c: screened angular partial densities → 27 int64 lanes (Q24.40).
// ----------------------------------------------------------------------------------------
__global__ void meam_density_kernel(const double* wx, const double* wy, const double* wz, int m,
                                    core::PairGeom geom, potentials::MeamScreenCParams scp,
                                    potentials::MeamForceParams fp, double dens_scale,
                                    long long* d_dens, int* overflow) {
  const int c = blockIdx.x * kZoneBlock + threadIdx.x;
  if (c >= m) return;
  long long lane[kDensLanes];
  for (int l = 0; l < kDensLanes; ++l) lane[l] = 0;
  const double xc = wx[c], yc = wy[c], zc = wz[c];

  for (int jl = 0; jl < m; ++jl) {
    if (jl == c) continue;
    double djx = wx[jl] - xc, djy = wy[jl] - yc, djz = wz[jl] - zc, rij2;
    if (!geom.reduce(djx, djy, djz, rij2)) continue;  // c's window neighbour (rij2 < rc²)
    const double rij = sqrt(rij2);
    const potentials::MeamScreenD s =
        potentials::meam_getscreen_d_device(wx, wy, wz, m, c, jl, djx, djy, djz, rij2, rij, geom, scp);
    const double sij = s.scrfcn * s.fcpair;
    if (fabs(sij) < 1e-20) continue;

    const double aj = rij / fp.re - 1.0, ro0 = fp.rho0;
    const double rhoa0j = ro0 * exp(-fp.beta0 * aj) * sij;
    const double rhoa1j = ro0 * exp(-fp.beta1 * aj) * sij;
    const double rhoa2j = ro0 * exp(-fp.beta2 * aj) * sij;
    const double rhoa3j = ro0 * exp(-fp.beta3 * aj) * sij;
    lane[0] += quantize(rhoa0j, dens_scale, overflow);                 // rho0
    lane[24] += quantize(fp.t1_eff * rhoa0j, dens_scale, overflow);    // t_ave[0]
    lane[25] += quantize(fp.t2 * rhoa0j, dens_scale, overflow);        // t_ave[1]
    lane[26] += quantize(fp.t3 * rhoa0j, dens_scale, overflow);        // t_ave[2]
    lane[1] += quantize(rhoa2j, dens_scale, overflow);                 // arho2b
    const double A1j = rhoa1j / rij, A2j = rhoa2j / rij2, A3j = rhoa3j / (rij2 * rij);
    const double del[3] = {djx, djy, djz};
    int nv2 = 0, nv3 = 0;
    for (int mm = 0; mm < 3; ++mm) {
      lane[2 + mm] += quantize(A1j * del[mm], dens_scale, overflow);             // arho1
      lane[21 + mm] += quantize(rhoa3j * del[mm] / rij, dens_scale, overflow);   // arho3b
      for (int n = mm; n < 3; ++n) {
        lane[5 + nv2] += quantize(A2j * del[mm] * del[n], dens_scale, overflow); // arho2
        ++nv2;
        for (int pp = n; pp < 3; ++pp) {
          lane[11 + nv3] += quantize(A3j * del[mm] * del[n] * del[pp], dens_scale, overflow);
          ++nv3;
        }
      }
    }
  }
  long long* L = d_dens + kDensLanes * c;
  for (int l = 0; l < kDensLanes; ++l) L[l] = lane[l];
}

// ----------------------------------------------------------------------------------------
// K2 — per WINDOW atom c: decode 27 lanes → MeamDensity → MeamEmbedDeriv. pe_embed for owned.
// ----------------------------------------------------------------------------------------
__global__ void meam_embed_kernel(int m, const int* owned_flag,
                                  potentials::MeamForceParams fp, double A, double Ec, int ibar,
                                  double gsmooth, int emb_lin_neg, double rho_ref,
                                  double dens_scale, const long long* d_dens,
                                  potentials::MeamEmbedDeriv* d_ed, long long* d_pe_embed) {
  __shared__ long long sred[kZoneBlock];
  const int c = blockIdx.x * kZoneBlock + threadIdx.x;
  const double escale = core::fixed::EnergyAccum::kScale;
  long long qpe = 0;
  if (c < m) {
    const potentials::MeamDensity d = meam_decode_dens(d_dens, c, dens_scale);
    const potentials::MeamEmbedDeriv e =
        potentials::meam_dens_final_deriv_pod(d, fp.v2D, fp.v3D, A, Ec, ibar, gsmooth, emb_lin_neg,
                                              rho_ref);
    d_ed[c] = e;
    if (owned_flag[c]) qpe += quantize(e.F, escale, nullptr);  // owned-only pe_embed; F finite
  }
  // block tree-reduce → atomicAdd
  sred[threadIdx.x] = qpe;
  __syncthreads();
  for (int s = kZoneBlock / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicAdd(reinterpret_cast<unsigned long long*>(d_pe_embed), (unsigned long long)sred[0]);
}

// ----------------------------------------------------------------------------------------
// K3 — per OWNED atom o: the 3-role transpose-replay (int64 registers, write-once).
// drop_class>0 = POISON (Role C dropped — the MB2 teeth).
// ----------------------------------------------------------------------------------------
__global__ void meam_force_kernel(const double* wx, const double* wy, const double* wz,
                                  const long* key, int m, const int* owned, int n_owned,
                                  core::PairGeom geom, potentials::MeamScreenCParams scp,
                                  potentials::MeamForceParams fp, potentials::MeamParamsView view,
                                  double dens_scale, const long long* d_dens,
                                  const potentials::MeamEmbedDeriv* d_ed, long long* d_fx,
                                  long long* d_fy, long long* d_fz, long long* d_pe_pair,
                                  long long* d_npartial, long long* d_nzero,
                                  unsigned long long* d_min_r2, int* overflow, int drop_class) {
  __shared__ long long sred[kZoneBlock];
  const int t = blockIdx.x * kZoneBlock + threadIdx.x;
  const double fscale = core::fixed::ForceAccum::kScale;
  const double escale = core::fixed::EnergyAccum::kScale;
  long long qpe = 0, qnp = 0, qnz = 0;

  if (t < n_owned) {
    const int o = owned[t];
    long long qfx = 0, qfy = 0, qfz = 0;
    double mr2 = 1e300;
    const double xo = wx[o], yo = wy[o], zo = wz[o];
    const long ko = key[o];

    // o's OWN in-rc neighbour list, cached once (dx = wx[b] − wx[o], the MeamNbr / project sign).
    int nb[kMeamMaxNbr];
    double ndx[kMeamMaxNbr], ndy[kMeamMaxNbr], ndz[kMeamMaxNbr], nr2[kMeamMaxNbr], nr[kMeamMaxNbr];
    int cnt = 0;
    for (int b = 0; b < m; ++b) {
      if (b == o) continue;
      double dx = wx[b] - xo, dy = wy[b] - yo, dz = wz[b] - zo, r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
      mr2 = fmin(mr2, r2);
      if (cnt < kMeamMaxNbr) {
        nb[cnt] = b; ndx[cnt] = dx; ndy[cnt] = dy; ndz[cnt] = dz; nr2[cnt] = r2; nr[cnt] = sqrt(r2); ++cnt;
      } else atomicOr(overflow, 4);  // STICKY neighbour-cap HALT
    }

    const potentials::MeamDensity densO = meam_decode_dens(d_dens, o, dens_scale);
    const potentials::MeamEmbedDeriv& edO = d_ed[o];

    // ===== Role A — o is the LOWER-key center i of bond (o, j) =====
    for (int e = 0; e < cnt; ++e) {
      const int j = nb[e];
      if (!(ko < key[j])) continue;  // each undirected bond ONCE, owner = lower global key
      const double djx = ndx[e], djy = ndy[e], djz = ndz[e], rij2 = nr2[e], rij = nr[e];
      const potentials::MeamScreenD s =
          potentials::meam_getscreen_d_device(wx, wy, wz, m, o, j, djx, djy, djz, rij2, rij, geom, scp);
      // screening counts (owned + lower-key, BEFORE any sij skip — matches the CPU pass-1 gate).
      if (s.scrfcn == 0.0) ++qnz;
      else if (s.scrfcn > 0.0 && s.scrfcn < 1.0) ++qnp;
      if (fabs(s.scrfcn) < 1e-20) continue;
      const double sij0 = s.scrfcn * s.fcpair;
      const double phi = view.phi_spline(rij), phip = view.phip_spline(rij);
      qpe += quantize(phi * sij0, escale, overflow);  // pe_pair owned-once at the lower-key center
      const potentials::MeamBondForce bf = potentials::meam_bond_force_device(
          densO, meam_decode_dens(d_dens, j, dens_scale), edO, d_ed[j], djx, djy, djz, rij2, rij,
          s.scrfcn, s.fcpair, s.dscrfcn, phi, phip, fp);
      qfx += quantize(bf.fm[0], fscale, overflow);
      qfy += quantize(bf.fm[1], fscale, overflow);
      qfz += quantize(bf.fm[2], fscale, overflow);
      if (fabs(sij0) < 1e-20 || fabs(sij0 - 1.0) < 1e-20) continue;
      if (drop_class > 0) continue;
      for (int kk = 0; kk < cnt; ++kk) {
        if (kk == e) continue;
        const potentials::MeamScreenK sk = potentials::meam_screen_k_device(
            ndx[kk], ndy[kk], ndz[kk], nr2[kk], djx, djy, djz, rij2, sij0, bf.dUdsij, fp);
        if (!sk.active) continue;
        qfx += quantize(sk.force1 * sk.dik[0], fscale, overflow);  // f_i += force1·d_ik
        qfy += quantize(sk.force1 * sk.dik[1], fscale, overflow);
        qfz += quantize(sk.force1 * sk.dik[2], fscale, overflow);
      }
    }

    // ===== Role B — o is the HIGHER-key endpoint j of bond (i, o), i a neighbour of o =====
    for (int e = 0; e < cnt; ++e) {
      const int i = nb[e];
      if (!(key[i] < ko)) continue;  // i is the lower-key center
      // bond (i,o) in i's frame: d_io = x_o − x_i = −(x_i − x_o) = −nb-vector. nbd is wx[b]−wx[o]
      // so x_o − x_i = −(wx[i]−wx[o]) = −ndx[e]. rij2/rij are symmetric.
      const double diox = -ndx[e], dioy = -ndy[e], dioz = -ndz[e], rij2 = nr2[e], rij = nr[e];
      const potentials::MeamScreenD s =
          potentials::meam_getscreen_d_device(wx, wy, wz, m, i, o, diox, dioy, dioz, rij2, rij, geom, scp);
      if (fabs(s.scrfcn) < 1e-20) continue;
      const double sij0 = s.scrfcn * s.fcpair;
      const double phi = view.phi_spline(rij), phip = view.phip_spline(rij);
      // bond body in I'S frame (i is the center, o is j): reads dens[i]/ed[i] (i may be HALO).
      const potentials::MeamBondForce bf = potentials::meam_bond_force_device(
          meam_decode_dens(d_dens, i, dens_scale), densO, d_ed[i], edO, diox, dioy, dioz, rij2, rij,
          s.scrfcn, s.fcpair, s.dscrfcn, phi, phip, fp);
      qfx += quantize(-bf.fm[0], fscale, overflow);  // −fm to the higher endpoint o
      qfy += quantize(-bf.fm[1], fscale, overflow);
      qfz += quantize(-bf.fm[2], fscale, overflow);
      if (fabs(sij0) < 1e-20 || fabs(sij0 - 1.0) < 1e-20) continue;
      if (drop_class > 0) continue;
      // k-loop over i's window neighbours (re-scan): o receives force2·d_jk (o is the j endpoint).
      const double xi = wx[i], yi = wy[i], zi = wz[i];
      for (int b = 0; b < m; ++b) {
        if (b == i || b == o) continue;  // k != i (center), k != j (==o)
        double dikx = wx[b] - xi, diky = wy[b] - yi, dikz = wz[b] - zi, rik2;
        if (!geom.reduce(dikx, diky, dikz, rik2)) continue;
        const potentials::MeamScreenK sk = potentials::meam_screen_k_device(
            dikx, diky, dikz, rik2, diox, dioy, dioz, rij2, sij0, bf.dUdsij, fp);
        if (!sk.active) continue;
        qfx += quantize(sk.force2 * sk.djk[0], fscale, overflow);  // f_j += force2·d_jk (j == o)
        qfy += quantize(sk.force2 * sk.djk[1], fscale, overflow);
        qfz += quantize(sk.force2 * sk.djk[2], fscale, overflow);
      }
    }

    // ===== Role C — o is the SCREENING third-atom k of bond (i, j), i a neighbour of o =====
    for (int e = 0; e < cnt; ++e) {
      if (drop_class > 0) break;  // POISON: drop Role C entirely (atom-k's force collapses)
      const int i = nb[e];
      // bond (i,j): i is a neighbour of o (the center), j ranges over i's window neighbours with
      // key[i]<key[j]. o (==k) screens (i,j). i's frame: d_ij = x_j − x_i; d_ik = x_o − x_i.
      const double xi = wx[i], yi = wy[i], zi = wz[i];
      const double dikx = -ndx[e], diky = -ndy[e], dikz = -ndz[e], rik2 = nr2[e];  // x_o − x_i
      for (int b = 0; b < m; ++b) {
        if (b == i || b == o) continue;  // j != i, j != k(==o)
        if (!(key[i] < key[b])) continue;  // bond owner = lower global key (i < j)
        double dijx = wx[b] - xi, dijy = wy[b] - yi, dijz = wz[b] - zi, rij2;
        if (!geom.reduce(dijx, dijy, dijz, rij2)) continue;  // j a window neighbour of i
        const double rij = sqrt(rij2);
        const potentials::MeamScreenD s =
            potentials::meam_getscreen_d_device(wx, wy, wz, m, i, b, dijx, dijy, dijz, rij2, rij, geom, scp);
        if (fabs(s.scrfcn) < 1e-20) continue;
        const double sij0 = s.scrfcn * s.fcpair;
        if (fabs(sij0) < 1e-20 || fabs(sij0 - 1.0) < 1e-20) continue;  // binary-S: no k-force
        const double phi = view.phi_spline(rij), phip = view.phip_spline(rij);
        const potentials::MeamBondForce bf = potentials::meam_bond_force_device(
            meam_decode_dens(d_dens, i, dens_scale), meam_decode_dens(d_dens, b, dens_scale),
            d_ed[i], d_ed[b], dijx, dijy, dijz, rij2, rij, s.scrfcn, s.fcpair, s.dscrfcn, phi, phip,
            fp);
        const potentials::MeamScreenK sk = potentials::meam_screen_k_device(
            dikx, diky, dikz, rik2, dijx, dijy, dijz, rij2, sij0, bf.dUdsij, fp);
        if (!sk.active) continue;
        qfx += quantize(-(sk.force1 * sk.dik[0] + sk.force2 * sk.djk[0]), fscale, overflow);
        qfy += quantize(-(sk.force1 * sk.dik[1] + sk.force2 * sk.djk[1]), fscale, overflow);
        qfz += quantize(-(sk.force1 * sk.dik[2] + sk.force2 * sk.djk[2]), fscale, overflow);
      }
    }

    d_fx[o] = qfx; d_fy[o] = qfy; d_fz[o] = qfz;  // window-local o, write-once
    atomicMin(d_min_r2, pos_double_bits(mr2));
  }

  // block tree-reduces → atomicAdd (pe_pair, n_partial, n_zero), reusing sred.
  sred[threadIdx.x] = qpe; __syncthreads();
  for (int s = kZoneBlock / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s]; __syncthreads(); }
  if (threadIdx.x == 0) atomicAdd(reinterpret_cast<unsigned long long*>(d_pe_pair), (unsigned long long)sred[0]);
  __syncthreads();
  sred[threadIdx.x] = qnp; __syncthreads();
  for (int s = kZoneBlock / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s]; __syncthreads(); }
  if (threadIdx.x == 0) atomicAdd(reinterpret_cast<unsigned long long*>(d_npartial), (unsigned long long)sred[0]);
  __syncthreads();
  sred[threadIdx.x] = qnz; __syncthreads();
  for (int s = kZoneBlock / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s]; __syncthreads(); }
  if (threadIdx.x == 0) atomicAdd(reinterpret_cast<unsigned long long*>(d_nzero), (unsigned long long)sred[0]);
}

}  // namespace tdmd::cuda
