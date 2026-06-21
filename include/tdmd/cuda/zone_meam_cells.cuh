#pragma once
// MEAM-ladder Me5b — CELL-LIST culled MEAM GPU kernels. A culled variant of
// zone_meam.cuh's O(m) density scan (K1) + the four O(m) force re-scans (K3),
// RAW-int64-BITWISE-EQUAL to the all-window kernels so the bench can measure the
// culling win on the heavy screened density+force passes AND enable the first
// angular flagship-scale (10⁶) atom-steps/s number. K2 (embedding, no neighbour
// scan) is reused VERBATIM. CUDA-only header; --fmad=false MANDATORY (link
// tdmd_eam_cuda_flags). zone_meam.cuh / zone_cells.cuh / meam.hpp are
// BYTE-UNTOUCHED — this header only ADDS new kernels + one new screening helper
// and reuses the all-window bodies character-for-character.
//
// THE CRUX — THE CANONICAL-K CULL (the one genuine new machinery over EAM cells):
// UNLIKE EAM (int64 density lanes, order-free by B1), MEAM's screening
// sij = Π_k fcut(cikj) is an FP64 PRODUCT and dscrfcn = Σ_k coef1·dCikj an FP64
// SUM (meam.hpp meam_getscreen_d_device, lines ~608/643) — FP-order-SENSITIVE,
// NOT B1-protected. The all-window path scans the window in HOST-SORTED-KEY order
// (the canonical sort, meam_window_force_gpu.cuh). A cell-walk visits k in
// b_order[] (cell-scatter) order, which DIFFERS ⇒ the FP product/sum reassociate
// ⇒ cells would NOT be bitwise to all-window. THE FIX: meam_getscreen_d_cells_device
// GATHERS the stencil k-candidates into a small local buffer, SORTS that buffer by
// global key[] (≤~64 entries, register insertion sort), THEN runs the product +
// derivative passes in that order. This makes cells sij/dscrfcn BIT-IDENTICAL to
// the sorted all-window sij for ANY cell-scatter order ⇒ the shipped
// nondeterministic cell_scatter_kernel is reused UNCHANGED. This is the device
// analogue of Te3b's canonical-ζ-sort.
//
// G-A (cells ≡ all-window raw int64) is the proof this is correct. The cull is
// kept BY CONSTRUCTION (it makes cells bitwise to the sorted all-window path for
// any cell-scatter order). MEASURE-FIRST VERDICT (G-SORT): on the tested Si
// fixtures (diamond 16-screening-k/bond + the partial slab) the screening
// reassociation stays SUB-QUANTUM at Q24.40 ⇒ the canonical-k sort is DEFENSIVE
// here, NOT load-bearing — the same finding as Te3b / SW T3b / Te5 (the design
// predicted load-bearing; measurement found defensive). It remains kept for
// robustness against a denser multi-k partial fixture (deferred measure-first).
//
// THE REACH IS SUB-CELL: √ebound·rc ≈ 1.04·rc < 2·rc (MeamParams::recompute
// asserts it) ⇒ ONE rc-padded whole-window grid (make_zone_grid n_zones=1) covers
// density donors + force neighbours + screening-k. No second wider grid.
#include <cuda_runtime.h>

#include "tdmd/cuda/zone_cells.cuh"        // CellGrid, make_zone_grid, cell_count/scatter, CUB scan
#include "tdmd/cuda/zone_meam.cuh"         // K1/K2/K3 all-window, kDensLanes, kMeamMaxNbr, kZoneBlock
#include "tdmd/potentials/meam.hpp"        // MeamScreenCParams/D, meam_detail::fcut/dfcut/dCfunc

namespace tdmd::cuda {

// ---------------------------------------------------------------------------------------------
// THE CANONICAL-K CULL — the cells-aware screening helper (a sibling of
// meam_getscreen_d_device, meam.hpp:578). Walks the ±s_d cell-neighbourhood of the bond CENTER c
// (NOT jl — the screening k-loop centers on c, meam.hpp:586,592), applies the SAME skips/cuts
// (k==c||k==jl, geom.reduce, rjk2>rbound, rik2>rbound, a<=0), GATHERS surviving k into a small
// local buffer, SORTS it by global key[] (= the sorted all-window slot order), THEN runs the
// product pass + the dscrfcn pass in that order. Bit-identical to the SORTED all-window helper for
// ANY cell-scatter order. cxi/cyi/czi = c's cell coords (computed ONCE by the caller).
// ---------------------------------------------------------------------------------------------
__device__ inline potentials::MeamScreenD meam_getscreen_d_cells_device(
    const double* wx, const double* wy, const double* wz, const long* key, int m, int c, int jl,
    double djx, double djy, double djz, double rij2, double rij, const core::PairGeom& geom,
    const potentials::MeamScreenCParams& p, CellGrid g, const int* b_starts, const int* b_counts,
    const int* b_order, int cxi, int cyi, int czi, int* overflow) {
  const double cutforce = p.rc;
  const double drinv = 1.0 / p.delr;
  const double rbound = p.ebound * rij2;
  const double rnorm = (cutforce - rij) * drinv;
  const double xc = wx[c], yc = wy[c], zc = wz[c];

  // Gather the surviving screening-k candidates from the ±s_d stencil into a local buffer:
  // store the candidate's global key + the precomputed rik2 (the product pass + the dscrfcn pass
  // re-derive cikj from rik2/rjk2, so we cache both — single stencil walk, no double traversal).
  long kkey[kMeamMaxNbr];
  double krik2[kMeamMaxNbr], krjk2[kMeamMaxNbr];
  int kn = 0;

  const int dzlo = (g.nz == 1) ? 0 : -g.sz, dzhi = (g.nz == 1) ? 0 : g.sz;
  const int dylo = (g.ny == 1) ? 0 : -g.sy, dyhi = (g.ny == 1) ? 0 : g.sy;
  const int dxlo = (g.nx == 1) ? 0 : -g.sx, dxhi = (g.nx == 1) ? 0 : g.sx;
  for (int dz = dzlo; dz <= dzhi; ++dz) {
    int zci = czi + dz;
    if (g.wrapz) zci = ((zci % g.nz) + g.nz) % g.nz;  // |dz|>1-safe (nz>=2s+1 guaranteed)
    else if (zci < 0 || zci >= g.nz) continue;
    for (int dy = dylo; dy <= dyhi; ++dy) {
      int yci = cyi + dy;
      if (g.wrapy) yci = ((yci % g.ny) + g.ny) % g.ny;
      else if (yci < 0 || yci >= g.ny) continue;
      for (int dx = dxlo; dx <= dxhi; ++dx) {
        int xci = cxi + dx;
        if (g.wrapx) xci = ((xci % g.nx) + g.nx) % g.nx;
        else if (xci < 0 || xci >= g.nx) continue;
        const int cell = g.idx(xci, yci, zci);
        const int beg = b_starts[cell], cnt = b_counts[cell];
        for (int t = beg; t < beg + cnt; ++t) {
          const int k = b_order[t];
          if (k == c || k == jl) continue;  // k != c (self) and k != j (the bond's other endpoint)
          double dxik = wx[k] - xc, dyik = wy[k] - yc, dzik = wz[k] - zc, rik2;
          if (!geom.reduce(dxik, dyik, dzik, rik2)) continue;  // k a window neighbour of c
          const double dxjk = dxik - djx, dyjk = dyik - djy, dzjk = dzik - djz;
          const double rjk2 = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
          if (rjk2 > rbound) continue;
          if (rik2 > rbound) continue;
          const double xik = rik2 / rij2;
          const double xjk = rjk2 / rij2;
          const double a = 1.0 - (xik - xjk) * (xik - xjk);
          if (a <= 0.0) continue;  // ⭐ TRAP 1: the ONLY negative-C rejection (ellipse)
          // SURVIVOR — stash it (the Cmax/Cmin classification stays inside the canonical-order pass).
          // GUARDED like the nb[] build (M5b acceptance M1): a screening k-set exceeding the cap is a
          // STICKY HALT, NEVER a silent local-array OOB (the all-window helper streams with no buffer
          // ⇒ the cull introduced this hazard; the bound check restores "guarded HALT" — INV).
          if (kn < kMeamMaxNbr) {
            krik2[kn] = rik2; krjk2[kn] = rjk2; kkey[kn] = key[k]; ++kn;
          } else { atomicOr(overflow, 4); }
        }
      }
    }
  }

  // SORT the gathered survivors by global key (= the sorted all-window slot order). Insertion sort
  // — kn ≤ ~16 in-rc for MEAM-Si (the kMaxNbr margin), worst-case kMeamMaxNbr. Carries the LOAD-
  // BEARING canonical sort (Me3b) onto the device grid: the FP product/sum below reassociate in
  // exactly the order the host-sorted all-window helper walks ⇒ bit-identical sij/dscrfcn.
  for (int a = 1; a < kn; ++a) {
    const long kk = kkey[a]; const double r1 = krik2[a], r2 = krjk2[a];
    int b = a - 1;
    while (b >= 0 && kkey[b] > kk) {
      kkey[b + 1] = kkey[b]; krik2[b + 1] = krik2[b]; krjk2[b + 1] = krjk2[b]; --b;
    }
    kkey[b + 1] = kk; krik2[b + 1] = r1; krjk2[b + 1] = r2;
  }

  // First pass — the screening product itself, in canonical (global-key) order.
  double sij = 1.0;
  for (int e = 0; e < kn; ++e) {
    const double xik = krik2[e] / rij2;
    const double xjk = krjk2[e] / rij2;
    const double a = 1.0 - (xik - xjk) * (xik - xjk);
    double cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
    if (cikj >= p.Cmax) continue;
    else if (cikj <= p.Cmin) { sij = 0.0; break; }
    else {
      const double delc = p.Cmax - p.Cmin;
      cikj = (cikj - p.Cmin) / delc;
      sij *= potentials::meam_detail::fcut(cikj);
    }
  }

  double dfc;
  const double fc = potentials::meam_detail::dfcut(rnorm, dfc);
  const double fcij = fc;
  const double dfcij = dfc * drinv;

  // Second pass — dscrfcn (the radial screening derivative; partial-only), same canonical order.
  double dscrfcn = 0.0;
  const double sfcij = sij * fcij;
  if (fabs(sfcij) > 1e-20 && fabs(sfcij - 1.0) > 1e-20) {  // !iszero && !isone
    for (int e = 0; e < kn; ++e) {
      const double xik = krik2[e] / rij2;
      const double xjk = krjk2[e] / rij2;
      const double a = 1.0 - (xik - xjk) * (xik - xjk);
      double cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
      if (cikj >= p.Cmax) {
        continue;
      } else {
        const double delc = p.Cmax - p.Cmin;
        cikj = (cikj - p.Cmin) / delc;
        double dfikj;
        const double sikj = potentials::meam_detail::dfcut(cikj, dfikj);
        const double coef1 = dfikj / (delc * sikj);
        const double dCikj = potentials::meam_detail::dCfunc(rij2, krik2[e], krjk2[e]);
        dscrfcn += coef1 * dCikj;
      }
    }
    const double coef1 = sfcij;
    const double coef2 = sij * dfcij / rij;
    dscrfcn = dscrfcn * coef1 - coef2;  // ⭐ TRAP 2: the MINUS on coef2 (radial taper)
  }

  return {sij, fcij, dscrfcn};
}

// =============================================================================================
// K1 (CULLED) — meam_density_cells_kernel. FORK of meam_density_kernel (zone_meam.cuh:69-117):
// the outer for(jl<m) → the ±s_d stencil (copied character-for-character from
// eam_density_cells_kernel, zone_eam_cells.cuh:57-89); the body (the 27-lane quantize
// accumulation) byte-identical EXCEPT the screening call → the canonical-k cells variant. Density
// lanes int64 ⇒ order-free B1.
// =============================================================================================
__global__ void meam_density_cells_kernel(const double* wx, const double* wy, const double* wz,
                                          const long* key, int m, core::PairGeom geom,
                                          potentials::MeamScreenCParams scp,
                                          potentials::MeamForceParams fp, double dens_scale,
                                          CellGrid g, const int* b_starts, const int* b_counts,
                                          const int* b_order, long long* d_dens, int* overflow) {
  const int c = blockIdx.x * kZoneBlock + threadIdx.x;
  if (c >= m) return;
  long long lane[kDensLanes];
  for (int l = 0; l < kDensLanes; ++l) lane[l] = 0;
  const double xc = wx[c], yc = wy[c], zc = wz[c];
  int cxi, cyi, czi;
  g.coords(xc, yc, zc, cxi, cyi, czi);

  // ±s_d stencil over c's cell-neighbourhood (the candidate set for c's window neighbours j).
  const int dzlo = (g.nz == 1) ? 0 : -g.sz, dzhi = (g.nz == 1) ? 0 : g.sz;
  const int dylo = (g.ny == 1) ? 0 : -g.sy, dyhi = (g.ny == 1) ? 0 : g.sy;
  const int dxlo = (g.nx == 1) ? 0 : -g.sx, dxhi = (g.nx == 1) ? 0 : g.sx;
  for (int dz = dzlo; dz <= dzhi; ++dz) {
    int zci = czi + dz;
    if (g.wrapz) zci = ((zci % g.nz) + g.nz) % g.nz;
    else if (zci < 0 || zci >= g.nz) continue;
    for (int dy = dylo; dy <= dyhi; ++dy) {
      int yci = cyi + dy;
      if (g.wrapy) yci = ((yci % g.ny) + g.ny) % g.ny;
      else if (yci < 0 || yci >= g.ny) continue;
      for (int dx = dxlo; dx <= dxhi; ++dx) {
        int xci = cxi + dx;
        if (g.wrapx) xci = ((xci % g.nx) + g.nx) % g.nx;
        else if (xci < 0 || xci >= g.nx) continue;
        const int cell = g.idx(xci, yci, zci);
        const int beg = b_starts[cell], cnt = b_counts[cell];
        for (int t = beg; t < beg + cnt; ++t) {
          const int jl = b_order[t];
          if (jl == c) continue;
          double djx = wx[jl] - xc, djy = wy[jl] - yc, djz = wz[jl] - zc, rij2;
          if (!geom.reduce(djx, djy, djz, rij2)) continue;  // c's window neighbour (rij2 < rc²)
          const double rij = sqrt(rij2);
          const potentials::MeamScreenD s = meam_getscreen_d_cells_device(
              wx, wy, wz, key, m, c, jl, djx, djy, djz, rij2, rij, geom, scp, g, b_starts, b_counts,
              b_order, cxi, cyi, czi, overflow);
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
      }
    }
  }
  long long* L = d_dens + kDensLanes * c;
  for (int l = 0; l < kDensLanes; ++l) L[l] = lane[l];
}

// =============================================================================================
// K3 (CULLED) — meam_force_cells_kernel. FORK of meam_force_kernel (zone_meam.cuh:154-311) — FOUR
// sub-culls (1) o's nbr-build (for(b<m)) → ±s_d stencil centered on O; (2) Role A k-loop — already
// culled (iterates o's cached nb[]); (3) Role B re-scan → ±s_d stencil RE-CENTERED on I; (4) Role C
// double scan → ±s_d stencil RE-CENTERED on I (key[i]<key[b] owner gate + if(b==i||b==o) kept).
// Each role's screening call → the canonical-k cells variant. Force output int64 register
// write-once ⇒ bitwise to all-window by B1 PROVIDED every role re-scan visits the identical in-rc
// set AND every embedded sij uses the canonical-k cull. drop_class>0 = POISON (Role C dropped).
//
// A device __forceinline__ that walks the ±s_d cell-neighbourhood of an arbitrary CENTER (qx,qy,qz)
// in cell (qcx,qcy,qcz) and INVOKES the supplied lambda body(b) on every other window atom in it.
// Roles B/C re-center the grid query on i (NOT o), so the stencil center coords change per role-
// iteration (one extra g.coords each — negligible). The body re-derives geometry itself.
// =============================================================================================
template <typename Body>
__device__ inline void meam_cell_for_each(CellGrid g, const int* b_starts, const int* b_counts,
                                          const int* b_order, int qcx, int qcy, int qcz, Body body) {
  const int dzlo = (g.nz == 1) ? 0 : -g.sz, dzhi = (g.nz == 1) ? 0 : g.sz;
  const int dylo = (g.ny == 1) ? 0 : -g.sy, dyhi = (g.ny == 1) ? 0 : g.sy;
  const int dxlo = (g.nx == 1) ? 0 : -g.sx, dxhi = (g.nx == 1) ? 0 : g.sx;
  for (int dz = dzlo; dz <= dzhi; ++dz) {
    int zci = qcz + dz;
    if (g.wrapz) zci = ((zci % g.nz) + g.nz) % g.nz;
    else if (zci < 0 || zci >= g.nz) continue;
    for (int dy = dylo; dy <= dyhi; ++dy) {
      int yci = qcy + dy;
      if (g.wrapy) yci = ((yci % g.ny) + g.ny) % g.ny;
      else if (yci < 0 || yci >= g.ny) continue;
      for (int dx = dxlo; dx <= dxhi; ++dx) {
        int xci = qcx + dx;
        if (g.wrapx) xci = ((xci % g.nx) + g.nx) % g.nx;
        else if (xci < 0 || xci >= g.nx) continue;
        const int cell = g.idx(xci, yci, zci);
        const int beg = b_starts[cell], cnt = b_counts[cell];
        for (int t = beg; t < beg + cnt; ++t) body(b_order[t]);
      }
    }
  }
}

__global__ void meam_force_cells_kernel(const double* wx, const double* wy, const double* wz,
                                        const long* key, int m, const int* owned, int n_owned,
                                        core::PairGeom geom, potentials::MeamScreenCParams scp,
                                        potentials::MeamForceParams fp, potentials::MeamParamsView view,
                                        double dens_scale, const long long* d_dens,
                                        const potentials::MeamEmbedDeriv* d_ed, CellGrid g,
                                        const int* b_starts, const int* b_counts, const int* b_order,
                                        long long* d_fx, long long* d_fy, long long* d_fz,
                                        long long* d_pe_pair, long long* d_npartial,
                                        long long* d_nzero, unsigned long long* d_min_r2,
                                        int* overflow, int drop_class) {
  __shared__ long long sred[kZoneBlock];
  const int tid = blockIdx.x * kZoneBlock + threadIdx.x;
  const double fscale = core::fixed::ForceAccum::kScale;
  const double escale = core::fixed::EnergyAccum::kScale;
  long long qpe = 0, qnp = 0, qnz = 0;

  if (tid < n_owned) {
    const int o = owned[tid];
    long long qfx = 0, qfy = 0, qfz = 0;
    double mr2 = 1e300;
    const double xo = wx[o], yo = wy[o], zo = wz[o];
    const long ko = key[o];
    int ocx, ocy, ocz;
    g.coords(xo, yo, zo, ocx, ocy, ocz);

    // (1) o's OWN in-rc neighbour list, cached once — CULLED to o's cell-neighbourhood.
    int nb[kMeamMaxNbr];
    double ndx[kMeamMaxNbr], ndy[kMeamMaxNbr], ndz[kMeamMaxNbr], nr2[kMeamMaxNbr], nr[kMeamMaxNbr];
    int cnt = 0;
    meam_cell_for_each(g, b_starts, b_counts, b_order, ocx, ocy, ocz, [&](int b) {
      if (b == o) return;
      double dx = wx[b] - xo, dy = wy[b] - yo, dz = wz[b] - zo, r2;
      if (!geom.reduce(dx, dy, dz, r2)) return;
      mr2 = fmin(mr2, r2);
      if (cnt < kMeamMaxNbr) {
        nb[cnt] = b; ndx[cnt] = dx; ndy[cnt] = dy; ndz[cnt] = dz; nr2[cnt] = r2; nr[cnt] = sqrt(r2); ++cnt;
      } else atomicOr(overflow, 4);  // STICKY neighbour-cap HALT
    });

    const potentials::MeamDensity densO = meam_decode_dens(d_dens, o, dens_scale);
    const potentials::MeamEmbedDeriv& edO = d_ed[o];

    // ===== Role A — o is the LOWER-key center i of bond (o, j). k-loop iterates o's cached nb[]
    // (ALREADY culled — the buffer IS the cell-neighbourhood). Screening call → canonical-k cells. =
    for (int e = 0; e < cnt; ++e) {
      const int j = nb[e];
      if (!(ko < key[j])) continue;
      const double djx = ndx[e], djy = ndy[e], djz = ndz[e], rij2 = nr2[e], rij = nr[e];
      const potentials::MeamScreenD s = meam_getscreen_d_cells_device(
          wx, wy, wz, key, m, o, j, djx, djy, djz, rij2, rij, geom, scp, g, b_starts, b_counts,
          b_order, ocx, ocy, ocz, overflow);
      if (s.scrfcn == 0.0) ++qnz;
      else if (s.scrfcn > 0.0 && s.scrfcn < 1.0) ++qnp;
      if (fabs(s.scrfcn) < 1e-20) continue;
      const double sij0 = s.scrfcn * s.fcpair;
      const double phi = view.phi_spline(rij), phip = view.phip_spline(rij);
      qpe += quantize(phi * sij0, escale, overflow);
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
        qfx += quantize(sk.force1 * sk.dik[0], fscale, overflow);
        qfy += quantize(sk.force1 * sk.dik[1], fscale, overflow);
        qfz += quantize(sk.force1 * sk.dik[2], fscale, overflow);
      }
    }

    // ===== Role B — o is the HIGHER-key endpoint j of bond (i, o), i a neighbour of o. The k-loop
    // (re-scan of i's window neighbours) → ±s_d stencil RE-CENTERED on i. =====
    for (int e = 0; e < cnt; ++e) {
      const int i = nb[e];
      if (!(key[i] < ko)) continue;
      const double diox = -ndx[e], dioy = -ndy[e], dioz = -ndz[e], rij2 = nr2[e], rij = nr[e];
      const double xi = wx[i], yi = wy[i], zi = wz[i];
      int icx, icy, icz;
      g.coords(xi, yi, zi, icx, icy, icz);
      const potentials::MeamScreenD s = meam_getscreen_d_cells_device(
          wx, wy, wz, key, m, i, o, diox, dioy, dioz, rij2, rij, geom, scp, g, b_starts, b_counts,
          b_order, icx, icy, icz, overflow);
      if (fabs(s.scrfcn) < 1e-20) continue;
      const double sij0 = s.scrfcn * s.fcpair;
      const double phi = view.phi_spline(rij), phip = view.phip_spline(rij);
      const potentials::MeamBondForce bf = potentials::meam_bond_force_device(
          meam_decode_dens(d_dens, i, dens_scale), densO, d_ed[i], edO, diox, dioy, dioz, rij2, rij,
          s.scrfcn, s.fcpair, s.dscrfcn, phi, phip, fp);
      qfx += quantize(-bf.fm[0], fscale, overflow);
      qfy += quantize(-bf.fm[1], fscale, overflow);
      qfz += quantize(-bf.fm[2], fscale, overflow);
      if (fabs(sij0) < 1e-20 || fabs(sij0 - 1.0) < 1e-20) continue;
      if (drop_class > 0) continue;
      meam_cell_for_each(g, b_starts, b_counts, b_order, icx, icy, icz, [&](int b) {
        if (b == i || b == o) return;  // k != i (center), k != j (==o)
        double dikx = wx[b] - xi, diky = wy[b] - yi, dikz = wz[b] - zi, rik2;
        if (!geom.reduce(dikx, diky, dikz, rik2)) return;
        const potentials::MeamScreenK sk = potentials::meam_screen_k_device(
            dikx, diky, dikz, rik2, diox, dioy, dioz, rij2, sij0, bf.dUdsij, fp);
        if (!sk.active) return;
        qfx += quantize(sk.force2 * sk.djk[0], fscale, overflow);  // f_j += force2·d_jk (j == o)
        qfy += quantize(sk.force2 * sk.djk[1], fscale, overflow);
        qfz += quantize(sk.force2 * sk.djk[2], fscale, overflow);
      });
    }

    // ===== Role C — o is the SCREENING third-atom k of bond (i, j), i a neighbour of o. The inner
    // j-scan over i's window neighbours → ±s_d stencil RE-CENTERED on i. =====
    for (int e = 0; e < cnt; ++e) {
      if (drop_class > 0) break;  // POISON: drop Role C entirely
      const int i = nb[e];
      const double xi = wx[i], yi = wy[i], zi = wz[i];
      const double dikx = -ndx[e], diky = -ndy[e], dikz = -ndz[e], rik2 = nr2[e];  // x_o − x_i
      int icx, icy, icz;
      g.coords(xi, yi, zi, icx, icy, icz);
      meam_cell_for_each(g, b_starts, b_counts, b_order, icx, icy, icz, [&](int b) {
        if (b == i || b == o) return;  // j != i, j != k(==o)
        if (!(key[i] < key[b])) return;  // bond owner = lower global key (i < j)
        double dijx = wx[b] - xi, dijy = wy[b] - yi, dijz = wz[b] - zi, rij2;
        if (!geom.reduce(dijx, dijy, dijz, rij2)) return;  // j a window neighbour of i
        const double rij = sqrt(rij2);
        const potentials::MeamScreenD s = meam_getscreen_d_cells_device(
            wx, wy, wz, key, m, i, b, dijx, dijy, dijz, rij2, rij, geom, scp, g, b_starts, b_counts,
            b_order, icx, icy, icz, overflow);
        if (fabs(s.scrfcn) < 1e-20) return;
        const double sij0 = s.scrfcn * s.fcpair;
        if (fabs(sij0) < 1e-20 || fabs(sij0 - 1.0) < 1e-20) return;  // binary-S: no k-force
        const double phi = view.phi_spline(rij), phip = view.phip_spline(rij);
        const potentials::MeamBondForce bf = potentials::meam_bond_force_device(
            meam_decode_dens(d_dens, i, dens_scale), meam_decode_dens(d_dens, b, dens_scale),
            d_ed[i], d_ed[b], dijx, dijy, dijz, rij2, rij, s.scrfcn, s.fcpair, s.dscrfcn, phi, phip,
            fp);
        const potentials::MeamScreenK sk = potentials::meam_screen_k_device(
            dikx, diky, dikz, rik2, dijx, dijy, dijz, rij2, sij0, bf.dUdsij, fp);
        if (!sk.active) return;
        qfx += quantize(-(sk.force1 * sk.dik[0] + sk.force2 * sk.djk[0]), fscale, overflow);
        qfy += quantize(-(sk.force1 * sk.dik[1] + sk.force2 * sk.djk[1]), fscale, overflow);
        qfz += quantize(-(sk.force1 * sk.dik[2] + sk.force2 * sk.djk[2]), fscale, overflow);
      });
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

// --- host helper: build the WINDOW cell grid (the dropped-donor-safe extent, one rc-padded grid
// over ALL m atoms — covers density donors + force neighbours + screening-k; reach is sub-cell).
// STRUCTURAL COPY of EamCellGrid / eam_build_window_grid (zone_eam_cells.cuh:182-234), renamed. ---
struct MeamCellGrid {
  CellGrid g{};
  int* d_cell_of = nullptr;  // m  (window-local atom -> cell)
  int* d_counts = nullptr;   // ncells
  int* d_starts = nullptr;   // ncells (exclusive scan of counts)
  int* d_cursor = nullptr;   // ncells (scatter cursors, copy of starts)
  int* d_order = nullptr;    // m  (cell-sorted window-local atom indices)
  void* d_cub = nullptr;     // CUB scan scratch
  size_t cub_bytes = 0;
  int ncells = 0, m = 0;
};

inline void meam_cells_free(MeamCellGrid& c) {
  for (void* p : {(void*)c.d_cell_of, (void*)c.d_counts, (void*)c.d_starts,
                  (void*)c.d_cursor, (void*)c.d_order, c.d_cub})
    if (p) cudaFree(p);
  c = MeamCellGrid{};
}

inline MeamCellGrid meam_build_window_grid(const double* d_wx, const double* d_wy,
                                           const double* d_wz, int m, const double box_lo[3],
                                           const double box_len[3], const bool periodic[3],
                                           double rcut, int cell_div = 1) {
  MeamCellGrid c;
  c.m = m;
  c.g = make_zone_grid(box_lo, box_len, periodic, rcut, /*n_zones=*/1, /*zone_id=*/0, cell_div);
  c.ncells = c.g.ncells();
  cudaMalloc(&c.d_cell_of, size_t(m) * sizeof(int));
  cudaMalloc(&c.d_counts, size_t(c.ncells) * sizeof(int));
  cudaMalloc(&c.d_starts, size_t(c.ncells) * sizeof(int));
  cudaMalloc(&c.d_cursor, size_t(c.ncells) * sizeof(int));
  cudaMalloc(&c.d_order, size_t(m) * sizeof(int));
  cudaMemset(c.d_counts, 0, size_t(c.ncells) * sizeof(int));

  const int blk = kZoneBlock;
  const int cg = (m + blk - 1) / blk;
  cell_count_kernel<<<cg, blk>>>(d_wx, d_wy, d_wz, m, c.g, c.d_cell_of, c.d_counts);
  cub::DeviceScan::ExclusiveSum(nullptr, c.cub_bytes, c.d_counts, c.d_starts, c.ncells);
  cudaMalloc(&c.d_cub, c.cub_bytes);
  cub::DeviceScan::ExclusiveSum(c.d_cub, c.cub_bytes, c.d_counts, c.d_starts, c.ncells);
  cudaMemcpy(c.d_cursor, c.d_starts, size_t(c.ncells) * sizeof(int), cudaMemcpyDeviceToDevice);
  cell_scatter_kernel<<<cg, blk>>>(c.d_cell_of, m, c.d_cursor, c.d_order);
  return c;
}

}  // namespace tdmd::cuda
