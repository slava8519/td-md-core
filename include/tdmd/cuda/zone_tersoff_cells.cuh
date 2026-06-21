#pragma once
// Tersoff-ladder Te5b — CELL-LIST culled Tersoff GPU kernel (the bond-order 4-role transpose-
// replay). A culled variant of zone_tersoff.cuh's O(m) window re-scans (o's nbr-build + the
// four-role re-scans + the ζ k-loops), RAW-int64-BITWISE-EQUAL to the all-window kernel so the
// bench can measure the culling win on the heavy bond-order force AND enable the Tersoff entry of
// the angular flagship (10⁶) atom-steps/s table. CUDA-only header; --fmad=false MANDATORY (link
// tdmd_eam_cuda_flags) — NOT for CPU↔GPU bitwise (impossible: exp/pow/sin diverge libm-vs-CUDA),
// but to keep GPU-INTERNAL determinism STRUCTURAL (no codegen-varying FMA fusion). zone_tersoff.cuh
// / zone_cells.cuh are BYTE-UNTOUCHED — this header only ADDS one new kernel + one new ζ helper and
// reuses the all-window role bodies character-for-character.
//
// THE CRUX — THE CANONICAL-ζ-CULL (the one genuine new machinery over EAM/SW cells):
// TERSOFF IS MEAM-LIKE, NOT SW-LIKE. The bond-order ζ_ij = Σ_k fc·g·exp (the zeta_center sum in
// zone_tersoff.cuh) is an FP64 SUM over k ⇒ FP-order-SENSITIVE, NOT B1-protected (UNLIKE SW, whose
// every triplet is a SINGLE independent eval ⇒ int64-order-free). The all-window path sums ζ in the
// HOST-SORTED-KEY order (the canonical sort tersoff_window_force_gpu.cuh already does). A cell-walk
// visits k in b_order[] (cell-scatter) order, which DIFFERS ⇒ the FP ζ-sum reassociates ⇒ cells
// would NOT be bitwise to all-window. THE FIX: zeta_center_cells GATHERS the stencil k-candidates
// into a small local buffer, SORTS that buffer by global key[] (≤~64 entries, register insertion
// sort), THEN sums tersoff_zeta in that order. This makes cells ζ BIT-IDENTICAL to the sorted
// all-window ζ for ANY cell-scatter order ⇒ the shipped nondeterministic cell_scatter_kernel is
// reused UNCHANGED. This is the device analogue of Te3b's canonical-ζ-sort / Me5b's canonical-k cull.
//
// G-A (cells ≡ all-window raw int64) is the proof this is correct. The cull is kept BY CONSTRUCTION
// (it makes cells bitwise to the sorted all-window path for any cell-scatter order). MEASURE-FIRST
// VERDICT (G-SORT): on the tested Si fixtures the ζ reassociation stays SUB-QUANTUM at Q24.40 ⇒ the
// canonical-ζ sort is DEFENSIVE here, NOT load-bearing — the same finding as Te3b / Te5 / Me5b (the
// design predicted load-bearing; measurement found defensive). It remains kept for robustness.
//
// THE M1 BUFFER GUARD (M5b acceptance): GUARD every per-thread gather buffer (o's nb[] AND the ζ
// k-candidate buffer) with `if(cnt<kMax){...} else atomicOr(overflow,4);` (sticky HALT, NEVER a
// silent OOB — Me5b shipped an OOB bug in the cull k-buffer without it). G-OVERFLOW exercises it.
//
// THE REACH IS ONE WHOLE-WINDOW GRID: Roles 3/4 re-center the stencil on a NEIGHBOUR i of o and
// re-scan i's window neighbours — the {i; o, m'} triplet reaches 2·rcut from o (the same graph
// distance as EAM's density donor; effective_range_for("tersoff")={2,true}). One rc-padded whole-
// window grid (make_zone_grid n_zones=1) over ALL m atoms covers it (reach is sub-cell). No second
// wider grid.
#include <cuda_runtime.h>

#include "tdmd/cuda/zone_cells.cuh"     // CellGrid, make_zone_grid, cell_count/scatter, CUB scan
#include "tdmd/cuda/zone_tersoff.cuh"   // tersoff_force_kernel, zeta_center, kMaxNbr, kZoneBlock
#include "tdmd/potentials/tersoff.hpp"  // TersoffParams + tersoff_zeta/zetaterm_d (HOST_DEVICE)

namespace tdmd::cuda {

// per-thread cap on the gathered ζ k-candidate buffer (the canonical-ζ cull). Tersoff-Si rcut=3.2 Å
// ⇒ ~4 in-rcut neighbours of a center; the kMaxNbr=64 cap is the same 16× headroom as o's nb[]. A
// ζ-candidate buffer overflow is a guarded HALT (sticky bit 4), NEVER a silent local-array OOB.
inline constexpr int kZetaMax = kMaxNbr;  // 64

// A device __forceinline__ that walks the ±s_d cell-neighbourhood of an arbitrary CENTER cell
// (qcx,qcy,qcz) and INVOKES the supplied lambda body(b) on every window atom in it. Roles 3/4
// re-center the grid query on i (NOT o), so the stencil center coords change per role-iteration
// (one extra g.coords each — negligible). The body re-derives geometry itself. STRUCTURAL COPY of
// meam_cell_for_each / sw_cell_for_each (the eam_*_cells_kernel ±s_d loop).
template <typename Body>
__device__ inline void tersoff_cell_for_each(CellGrid g, const int* b_starts, const int* b_counts,
                                             const int* b_order, int qcx, int qcy, int qcz,
                                             Body body) {
  const int dzlo = (g.nz == 1) ? 0 : -g.sz, dzhi = (g.nz == 1) ? 0 : g.sz;
  const int dylo = (g.ny == 1) ? 0 : -g.sy, dyhi = (g.ny == 1) ? 0 : g.sy;
  const int dxlo = (g.nx == 1) ? 0 : -g.sx, dxhi = (g.nx == 1) ? 0 : g.sx;
  for (int dz = dzlo; dz <= dzhi; ++dz) {
    int zci = qcz + dz;
    if (g.wrapz) zci = ((zci % g.nz) + g.nz) % g.nz;  // |dz|>1-safe (nz>=2s+1 guaranteed)
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

// ---------------------------------------------------------------------------------------------
// THE CANONICAL-ζ-CULL — the cells-aware ζ helper (a sibling of zeta_center, zone_tersoff.cuh:40).
// ζ_{c,x} = Σ_{k∈window, k a nbr of c, k≠c, k≠x} tersoff_zeta(...). The all-window zeta_center
// streams the WHOLE window (front-to-back over the host-sorted slots). The cells variant walks the
// ±s_d cell-neighbourhood of the bond CENTER c, applies the SAME skips/cut (k==c||k==x, geom.reduce),
// GATHERS the surviving k into a small local buffer, SORTS it by global key[] (= the sorted all-
// window slot order), THEN sums tersoff_zeta in that order. Bit-identical to the SORTED all-window
// zeta_center for ANY cell-scatter order. ccx/ccy/ccz = c's cell coords (computed ONCE by the
// caller per role-iteration). FP64 — NEVER a FixedAccum. *overflow gets bit-4 (sticky HALT) if the
// in-rc ζ-candidate set exceeds kZetaMax (the M1 guard — the all-window helper streams with no
// buffer ⇒ the cull introduced this hazard).
__device__ inline double zeta_center_cells(const double* wx, const double* wy, const double* wz,
                                           const long* key, int c, int x_local, double rcx,
                                           const double* rcxh, core::PairGeom geom,
                                           const potentials::TersoffParams& p, CellGrid g,
                                           const int* b_starts, const int* b_counts,
                                           const int* b_order, int ccx, int ccy, int ccz,
                                           int* overflow) {
  const double xc = wx[c], yc = wy[c], zc = wz[c];

  // Gather the surviving ζ-k candidates from the ±s_d stencil. Cache the global key + the hat vector
  // + rk (so the canonical-order sum below re-derives nothing per candidate beyond the sort).
  long kkey[kZetaMax];
  double khx[kZetaMax], khy[kZetaMax], khz[kZetaMax], kr[kZetaMax];
  int kn = 0;
  tersoff_cell_for_each(g, b_starts, b_counts, b_order, ccx, ccy, ccz, [&](int k) {
    if (k == c || k == x_local) return;  // k != c (self) and k != x (the bond's other endpoint)
    double dx = xc - wx[k], dy = yc - wy[k], dz = zc - wz[k], r2;
    if (!geom.reduce(dx, dy, dz, r2)) return;  // k a window neighbour of c
    const double rk = sqrt(r2), rkinv = 1.0 / rk;
    if (kn < kZetaMax) {
      kkey[kn] = key[k];
      khx[kn] = -dx * rkinv; khy[kn] = -dy * rkinv; khz[kn] = -dz * rkinv;  // (x_k−x_c)/r LAMMPS hat
      kr[kn] = rk; ++kn;
    } else { atomicOr(overflow, 4); }  // STICKY ζ-candidate-cap HALT (M1 guard)
  });

  // SORT the gathered survivors by global key (= the sorted all-window slot order). Insertion sort —
  // kn ≤ ~4 in-rc for Tersoff-Si (the kMaxNbr margin). Carries the canonical-ζ order (Te3b/Te5) onto
  // the device grid: the FP ζ-sum below reassociates in exactly the order the host-sorted all-window
  // zeta_center walks ⇒ bit-identical ζ.
  for (int a = 1; a < kn; ++a) {
    const long kk = kkey[a]; const double hx = khx[a], hy = khy[a], hz = khz[a], rr = kr[a];
    int b = a - 1;
    while (b >= 0 && kkey[b] > kk) {
      kkey[b + 1] = kkey[b]; khx[b + 1] = khx[b]; khy[b + 1] = khy[b]; khz[b + 1] = khz[b];
      kr[b + 1] = kr[b]; --b;
    }
    kkey[b + 1] = kk; khx[b + 1] = hx; khy[b + 1] = hy; khz[b + 1] = hz; kr[b + 1] = rr;
  }

  double zeta = 0.0;
  for (int e = 0; e < kn; ++e) {
    const double rckh[3] = {khx[e], khy[e], khz[e]};
    zeta += potentials::tersoff_zeta(rcx, kr[e], rcxh, rckh, p);
  }
  return zeta;
}

// =============================================================================================
// CULLED Tersoff force kernel — tersoff_force_cells_kernel. FORK of tersoff_force_kernel
// (zone_tersoff.cuh:59-190). FIVE sub-culls: (0) o's nbr-build (`for(b<m)`) → ±s_d stencil centered
// on O; (Role 1 PAIR) iterates o's cached nb[] — ALREADY culled; (Role 2 CENTER-i) iterates o's
// cached nb[] (culled) + zeta_center → zeta_center_cells (centered on o); (Role 3 ENDPOINT-j) the
// k-loop re-scan of i's window neighbours → ±s_d stencil RE-CENTERED on i + zeta_center → cells
// (centered on i); (Role 4 THIRD-k) the j-scan over i's window neighbours → ±s_d stencil RE-CENTERED
// on i + zeta_center → cells (centered on i). Force output int64 register write-once ⇒ bitwise to
// all-window by B1 PROVIDED every role re-scan visits the identical in-rc set AND every embedded ζ
// uses the canonical-ζ cull. PE/n_bonds/n_triplets cross threads via block tree-reduce + atomicAdd.
// =============================================================================================
__global__ void tersoff_force_cells_kernel(
    const double* wx, const double* wy, const double* wz, const long* key, int m,
    const int* owned, int n_owned, core::PairGeom geom, potentials::TersoffParams tp,
    CellGrid g, const int* b_starts, const int* b_counts, const int* b_order,
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
    int ocx, ocy, ocz;
    g.coords(xo, yo, zo, ocx, ocy, ocz);

    // (0) o's own neighbour list, cached once — CULLED to o's cell-neighbourhood. dx = xo − wx[b].
    int nb[kMaxNbr];
    double nbdx[kMaxNbr], nbdy[kMaxNbr], nbdz[kMaxNbr], nbr_[kMaxNbr];
    int cnt = 0;
    tersoff_cell_for_each(g, b_starts, b_counts, b_order, ocx, ocy, ocz, [&](int b) {
      if (b == o) return;
      double dx = xo - wx[b], dy = yo - wy[b], dz = zo - wz[b], r2;
      if (!geom.reduce(dx, dy, dz, r2)) return;
      mr2 = fmin(mr2, r2);
      if (cnt < kMaxNbr) { nb[cnt] = b; nbdx[cnt] = dx; nbdy[cnt] = dy; nbdz[cnt] = dz; nbr_[cnt] = sqrt(r2); ++cnt; }
      else atomicOr(overflow, 4);  // STICKY neighbour-cap HALT (M1 guard)
    });

    // Role 1 — PAIR/repulsive: o's symmetric force; energy fc·fR + n_bonds by the LOWER GLOBAL KEY.
    // Iterates o's cached nb[] — ALREADY culled. Body byte-identical to the all-window kernel.
    for (int e = 0; e < cnt; ++e) {
      const double r = nbr_[e], tmp_exp = exp(-tp.lam1 * r);
      const double fforce = -tp.biga * tmp_exp * (potentials::ters_fc_d(r, tp) - potentials::ters_fc(r, tp) * tp.lam1) / r;
      qfx += quantize(nbdx[e] * fforce, fscale, overflow);
      qfy += quantize(nbdy[e] * fforce, fscale, overflow);
      qfz += quantize(nbdz[e] * fforce, fscale, overflow);
      if (ko < key[nb[e]]) { qpe += quantize(potentials::ters_fc(r, tp) * tp.biga * tmp_exp, escale, overflow); ++qnb; }
    }

    // Role 2 — CENTER-i of bond (o,j): attractive-radial f_i + angular f_i; OWNS attr-PE + count.
    // k-loop iterates o's cached nb[] (already culled); ζ → the canonical-ζ cull centered on o.
    for (int e = 0; e < cnt; ++e) {
      const double rij = nbr_[e], rijinv = 1.0 / rij;
      const double rijh[3] = {-nbdx[e] * rijinv, -nbdy[e] * rijinv, -nbdz[e] * rijinv};  // (x_j−x_o)/r
      const double zeta = zeta_center_cells(wx, wy, wz, key, o, nb[e], rij, rijh, geom, tp, g,
                                            b_starts, b_counts, b_order, ocx, ocy, ocz, overflow);
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

    // Role 3 — ENDPOINT-j of bond (i,o): attractive-radial f_j + angular f_j; NO energy/count. The
    // k-loop (re-scan of i's window neighbours) → ±s_d stencil RE-CENTERED on i; ζ → cells (on i).
    for (int e = 0; e < cnt; ++e) {
      const int i = nb[e];
      const double rij = nbr_[e], rijinv = 1.0 / rij;
      const double rijh[3] = {nbdx[e] * rijinv, nbdy[e] * rijinv, nbdz[e] * rijinv};  // (x_o−x_i)/r (opp Role 2)
      const double xi = wx[i], yi = wy[i], zi = wz[i];
      int icx, icy, icz;
      g.coords(xi, yi, zi, icx, icy, icz);
      const double zeta = zeta_center_cells(wx, wy, wz, key, i, o, rij, rijh, geom, tp, g,
                                            b_starts, b_counts, b_order, icx, icy, icz, overflow);
      const double fa = potentials::ters_fa(rij, tp), fa_d = potentials::ters_fa_d(rij, tp), bij = potentials::ters_bij(zeta, tp);
      const double fpair_z = 0.5 * bij * fa_d * rijinv;
      const double prefactor = -0.5 * fa * potentials::ters_bij_d(zeta, tp);
      qfx += quantize(-nbdx[e] * fpair_z, fscale, overflow);
      qfy += quantize(-nbdy[e] * fpair_z, fscale, overflow);
      qfz += quantize(-nbdz[e] * fpair_z, fscale, overflow);
      tersoff_cell_for_each(g, b_starts, b_counts, b_order, icx, icy, icz, [&](int b) {
        if (b == i || b == o) return;
        double dx = xi - wx[b], dy = yi - wy[b], dz = zi - wz[b], r2;
        if (!geom.reduce(dx, dy, dz, r2)) return;
        const double rk = sqrt(r2), rkinv = 1.0 / rk;
        const double rikh[3] = {-dx * rkinv, -dy * rkinv, -dz * rkinv};
        double fi[3], fj[3], fk[3];
        potentials::tersoff_zetaterm_d(prefactor, rijh, rij, rijinv, rikh, rk, rkinv, fi, fj, fk, tp);
        qfx += quantize(fj[0], fscale, overflow);
        qfy += quantize(fj[1], fscale, overflow);
        qfz += quantize(fj[2], fscale, overflow);
      });
    }

    // Role 4 — THIRD-k of triplet (i;j,o): angular f_k only; NO energy/count. The j-scan over i's
    // window neighbours → ±s_d stencil RE-CENTERED on i; ζ → cells (on i).
    for (int e = 0; e < cnt; ++e) {
      const int i = nb[e];
      const double rio = nbr_[e], rioinv = 1.0 / rio;
      const double riohat[3] = {nbdx[e] * rioinv, nbdy[e] * rioinv, nbdz[e] * rioinv};  // (x_o−x_i)/r = k-hat
      const double xi = wx[i], yi = wy[i], zi = wz[i];
      int icx, icy, icz;
      g.coords(xi, yi, zi, icx, icy, icz);
      tersoff_cell_for_each(g, b_starts, b_counts, b_order, icx, icy, icz, [&](int b) {
        if (b == i || b == o) return;
        double dx = xi - wx[b], dy = yi - wy[b], dz = zi - wz[b], r2;
        if (!geom.reduce(dx, dy, dz, r2)) return;
        const double rij = sqrt(r2), rijinv = 1.0 / rij;
        const double rijh[3] = {-dx * rijinv, -dy * rijinv, -dz * rijinv};  // (x_j−x_i)/r
        const double zeta = zeta_center_cells(wx, wy, wz, key, i, b, rij, rijh, geom, tp, g,
                                              b_starts, b_counts, b_order, icx, icy, icz, overflow);
        const double prefactor = -0.5 * potentials::ters_fa(rij, tp) * potentials::ters_bij_d(zeta, tp);
        double fi[3], fj[3], fk[3];
        potentials::tersoff_zetaterm_d(prefactor, rijh, rij, rijinv, riohat, rio, rioinv, fi, fj, fk, tp);
        qfx += quantize(fk[0], fscale, overflow);
        qfy += quantize(fk[1], fscale, overflow);
        qfz += quantize(fk[2], fscale, overflow);
      });
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

// --- host helper: build the WINDOW cell grid (the dropped-donor-safe extent, one rc-padded grid
// over ALL m atoms — covers o's neighbours + the Roles-3/4 2·rcut reach + the ζ-k; reach is sub-
// cell). STRUCTURAL COPY of SwCellGrid / sw_build_window_grid (zone_sw_cells.cuh), renamed. ---
struct TersoffCellGrid {
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

inline void tersoff_cells_free(TersoffCellGrid& c) {
  for (void* p : {(void*)c.d_cell_of, (void*)c.d_counts, (void*)c.d_starts,
                  (void*)c.d_cursor, (void*)c.d_order, c.d_cub})
    if (p) cudaFree(p);
  c = TersoffCellGrid{};
}

inline TersoffCellGrid tersoff_build_window_grid(const double* d_wx, const double* d_wy,
                                                 const double* d_wz, int m, const double box_lo[3],
                                                 const double box_len[3], const bool periodic[3],
                                                 double rcut, int cell_div = 1) {
  TersoffCellGrid c;
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
