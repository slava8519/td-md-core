#pragma once
// M6 E5c (bake-off, SPATIAL-SORT piece) — cell/Morton-SORTED per-atom VERLET
// neighbour-list EAM GPU kernels. A coalescing-optimised twin of
// zone_eam_verlet.cuh's tight-list density + force sweeps, BITWISE-EQUAL to the
// all-window kernels (zone_eam.cuh) by B1/INV-9 — once the per-atom forces are
// UN-SORTED back to the original window order. CUDA-only header; --fmad=false
// MANDATORY (link tdmd_eam_cuda_flags — the f_over_r prefactor and the Horner
// steps fuse to fma otherwise and break 1 ULP). zone_eam.cuh /
// zone_eam_verlet.cuh / zone_eam_cells.cuh / zone_eam_newton3.cuh / zone_cells.cuh
// / the rings are BYTE-UNTOUCHED — this header only ADDS new kernels + a host
// helper and reuses theirs verbatim.
//
// WHY SORTING (the coalescing lever): the verlet force/density kernels gather
// wx[bb], wy[bb], wz[bb], d_fp[bb], key[bb] for each neighbour bb in the CSR
// list. The FCC lattice is built in (ix,iy,iz,basis) order — NOT spatially
// sorted — so a given atom's neighbours have SCATTERED window-local indices ⇒
// uncoalesced global loads ⇒ memory-bound. Cell-sorting the atoms (positions +
// per-atom payload) clusters each atom's neighbours into contiguous memory ⇒
// the gathered loads coalesce. We also issue the neighbour gathers through the
// read-only data cache (__ldg).
//
// WHY THE SORT STAYS BITWISE-EQUAL (the B1 / key=original-index argument):
//   * Per-atom force = Σ over neighbours of quantize(...). int64 accumulation is
//     ORDER-FREE (associative, B1) ⇒ reordering the neighbour list does NOT
//     change the per-atom int64 sum. Same for ρ (Σ quantize(ρ_a)).
//   * The φ-once PE gate is key[aa] < key[bb]. If `key` carries the ORIGINAL
//     atom index (atom originally at window slot i keeps key=i regardless of its
//     sorted slot), the gated undirected-pair set is identical to all-window ⇒
//     identical PE. s_key[s] = original window index of the atom at sorted slot s.
//   * The in-cutoff acceptance is the EXACT same predicate (PairGeom::reduce,
//     r2 < rcut²) applied to the SAME pair of physical positions — a permutation
//     does not move atoms, only relabels slots ⇒ identical in-cutoff multiset.
//   * min_r2 is atomicMin (order-free); it agrees on the Overlap-HALT boolean but
//     is min-over-examined and never bit-compared across paths.
//   * The verlet list is built OVER THE SORTED positions; the indices it stores
//     are sorted slots whose neighbours are now spatially clustered.
// So: cell-sort (positions + carry s_key=original-index), build the list on the
// sorted positions, run the SAME math, then UN-SORT the per-atom int64
// ρ/fx/fy/fz back to original window order (scatter by order[]). The result is
// BITWISE == all-window. This is the load-bearing test in test_cuda_eam_cells.cu.
//
// FULL-NEIGHBOUR (NON-NEGOTIABLE): like zone_eam_verlet.cuh, the list contains
// BOTH directions (aa sees bb AND bb sees aa). eam_build_verlet_list builds a
// full list by construction; we reuse it unchanged on the sorted positions.
#include <cuda_runtime.h>

#include "tdmd/cuda/zone_cells.cuh"        // CellGrid, make_zone_grid, cell_count/scatter, CUB scan
#include "tdmd/cuda/zone_eam.cuh"          // EamSetflView, quantize, pos_double_bits, kZoneBlock
#include "tdmd/cuda/zone_eam_cells.cuh"    // eam_build_window_grid / EamCellGrid (the cell sort)
#include "tdmd/cuda/zone_eam_verlet.cuh"   // eam_build_verlet_list / EamVerletList (built on sorted pos)

namespace tdmd::cuda {

// Cell-SORTED window + the tight verlet list built over the sorted positions.
//   sx/sy/sz[m]  : positions in cell-sorted order (sorted slot s -> position)
//   s_key[m]     : ORIGINAL window index of the atom at sorted slot s (φ-once gate)
//   d_order[m]   : sorted slot s -> ORIGINAL window slot (the un-sort scatter map)
//   vl           : EamVerletList over sx/sy/sz (CSR neighbour indices are SORTED slots)
// Owned-membership: the bench/tests run every window atom owned, so the sorted
// force kernel treats every sorted slot as owned (n_owned == m). The owned remap
// is therefore the identity over sorted slots; no separate owned[] is carried.
struct EamSortedWindow {
  double* sx = nullptr;   // m
  double* sy = nullptr;   // m
  double* sz = nullptr;   // m
  long* s_key = nullptr;  // m  (original window index per sorted slot)
  int* d_order = nullptr; // m  (sorted slot -> original window slot; un-sort map)
  EamVerletList vl;       // verlet list over the SORTED positions
  int m = 0;
};

inline void eam_sorted_free(EamSortedWindow& s) {
  for (void* p : {(void*)s.sx, (void*)s.sy, (void*)s.sz, (void*)s.s_key,
                  (void*)s.d_order})
    if (p) cudaFree(p);
  eam_verlet_free(s.vl);
  s = EamSortedWindow{};
}

// Gather kernel: scatter the original window arrays into cell-sorted order.
//   order[s] = original window slot at sorted slot s (from the cell scatter).
// sorted slot s reads original slot o = order[s] and writes sx[s] = wx[o], etc.,
// s_key[s] = key[o] (the ORIGINAL index — for the φ-once gate after sorting).
__global__ void eam_sort_gather_kernel(const double* wx, const double* wy,
                                       const double* wz, const long* key, int m,
                                       const int* order, double* sx, double* sy,
                                       double* sz, long* s_key) {
  const int s = blockIdx.x * kZoneBlock + threadIdx.x;
  if (s >= m) return;
  const int o = order[s];
  sx[s] = wx[o];
  sy[s] = wy[o];
  sz[s] = wz[o];
  s_key[s] = key[o];
}

// Un-sort kernel: scatter a per-sorted-slot int64 array back to original order.
//   dst[order[s]] = src[s]  (sorted slot s -> original window slot order[s]).
// order[] is a permutation (every original slot appears exactly once) ⇒ the
// scatter is race-free (one writer per destination).
__global__ void eam_unsort_scatter_kernel(const long long* src, int m,
                                          const int* order, long long* dst) {
  const int s = blockIdx.x * kZoneBlock + threadIdx.x;
  if (s >= m) return;
  dst[order[s]] = src[s];
}

// pass 1 (SORTED + COALESCED): ρ_s = Σ_{bb in s's list} ρ_a(r), bb a SORTED slot.
// Bit-for-bit == eam_density_verlet_kernel / eam_density_kernel after the un-sort
// scatter: same int64 register, same EXACT in-cutoff re-test (geom.reduce,
// r2 < rcut²), same quantize, same dens_scale. The neighbour gathers (sx[bb],
// sy[bb], sz[bb]) go through __ldg (read-only data cache) and are now spatially
// clustered (the sort) ⇒ coalesced. d_rho_s is in SORTED-slot space (un-sorted
// by the caller).
__global__ void eam_density_sorted_kernel(const double* sx, const double* sy,
                                          const double* sz, int m,
                                          core::PairGeom geom, EamSetflView eam,
                                          double dens_scale, const int* d_off,
                                          const int* d_idx, long long* d_rho_s,
                                          int* overflow) {
  const int aa = blockIdx.x * kZoneBlock + threadIdx.x;
  if (aa >= m) return;
  long long qrho = 0;
  const double xi = sx[aa], yi = sy[aa], zi = sz[aa];
  const int beg = d_off[aa], end = d_off[aa + 1];
  for (int t = beg; t < end; ++t) {
    const int bb = d_idx[t];
    double ddx = xi - __ldg(&sx[bb]), ddy = yi - __ldg(&sy[bb]),
           ddz = zi - __ldg(&sz[bb]), r2;
    if (!geom.reduce(ddx, ddy, ddz, r2)) continue;  // EXACT rcut re-test
    double v, dv;
    eam.eval_rhoa(sqrt(r2), v, dv);
    qrho += quantize(v, dens_scale, overflow);
  }
  d_rho_s[aa] = qrho;
}

// pass 3 (SORTED + COALESCED): full-neighbour force for every sorted slot + pe.
// IDENTICAL math to eam_force_verlet_kernel / eam_force_kernel; the candidate
// loop is the flat CSR iterate over the sorted-slot list, with the neighbour
// gathers (sx[bb], sy[bb], sz[bb], d_fp_s[bb], s_key[bb]) issued through __ldg
// and now spatially clustered (coalesced). Kept verbatim: eval_F once per atom,
// quantize() calls, the f_over_r line, the φ-once gate `s_key[aa] < s_key[bb]`
// (ORIGINAL indices — same gated pair set as all-window), qfx/y/z registers,
// sred[] tree-reduce + one atomicAdd, atomicMin (d_min_r2) BEFORE the cutoff.
// d_rho_s / d_fp_s / d_fx_s / d_fy_s / d_fz_s are ALL in SORTED-slot space; the
// caller un-sorts d_fx_s/d_fy_s/d_fz_s back to original window order.
__global__ void eam_force_sorted_kernel(
    const double* sx, const double* sy, const double* sz, const long* s_key,
    int m, core::PairGeom geom, EamSetflView eam, double dens_scale,
    const long long* d_rho_s, const double* d_fp_s, const int* d_off,
    const int* d_idx, long long* d_fx_s, long long* d_fy_s, long long* d_fz_s,
    long long* d_pe, unsigned long long* d_min_r2, int* overflow) {
  __shared__ long long sred[kZoneBlock];
  const int aa = blockIdx.x * kZoneBlock + threadIdx.x;
  long long qpe = 0;
  if (aa < m) {
    long long qfx = 0, qfy = 0, qfz = 0;
    double mr2 = 1e300;
    double Fi, Fpi;
    eam.eval_F(double(d_rho_s[aa]) / dens_scale, Fi, Fpi);  // embedding energy once
    qpe += quantize(Fi, core::fixed::EnergyAccum::kScale, overflow);
    const double xi = sx[aa], yi = sy[aa], zi = sz[aa];
    const double fpi = d_fp_s[aa];
    const long ki = s_key[aa];  // ORIGINAL window index (φ-once gate)
    const int beg = d_off[aa], end = d_off[aa + 1];
    for (int t = beg; t < end; ++t) {
      const int bb = d_idx[t];
      double ddx = xi - __ldg(&sx[bb]), ddy = yi - __ldg(&sy[bb]),
             ddz = zi - __ldg(&sz[bb]), r2;
      const bool ok = geom.reduce(ddx, ddy, ddz, r2);
      mr2 = fmin(mr2, r2);  // BEFORE the cutoff (pass3 only) — overlap probe
      if (!ok) continue;
      const double r = sqrt(r2);
      double phi, dphi, ra, dra;
      eam.eval_phi(r, phi, dphi);
      eam.eval_rhoa(r, ra, dra);
      const double f_over_r = -(dphi + (fpi + __ldg(&d_fp_s[bb])) * dra) / r;  // FMA-killer site
      qfx += quantize(f_over_r * ddx, core::fixed::ForceAccum::kScale, overflow);
      qfy += quantize(f_over_r * ddy, core::fixed::ForceAccum::kScale, overflow);
      qfz += quantize(f_over_r * ddz, core::fixed::ForceAccum::kScale, overflow);
      if (ki < __ldg(&s_key[bb]))
        qpe += quantize(phi, core::fixed::EnergyAccum::kScale, overflow);
    }
    d_fx_s[aa] = qfx; d_fy_s[aa] = qfy; d_fz_s[aa] = qfz;  // SORTED slot aa
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

// --- host helper: cell-SORT the window + build the verlet list over it --------
//
// (1) Build the WHOLE-window cell grid (eam_build_window_grid — dropped-donor-safe
//     extent) to get the cell-sorted permutation order[] (sorted slot -> original
//     window slot). (2) Gather positions + key into sorted order (sx/sy/sz/s_key).
//     s_key carries the ORIGINAL window index per sorted slot (the φ-once gate).
// (3) Build the tight verlet list OVER THE SORTED positions — its CSR neighbour
//     indices are now spatially clustered ⇒ coalesced gathers in the kernels.
// All buffers device-side, caller-freed (eam_sorted_free). box_lo/box_len/periodic
// describe the window's box (the SAME box PairGeom uses); rcut the potential
// cutoff, skin the verlet skin (≥0). The grid sort uses rcut; the verlet build
// re-grids at (rcut+skin) internally.
inline EamSortedWindow eam_sort_window(const double* d_wx, const double* d_wy,
                                       const double* d_wz, const long* d_key,
                                       int m, const double box_lo[3],
                                       const double box_len[3],
                                       const bool periodic[3], double rcut,
                                       double skin) {
  EamSortedWindow s;
  s.m = m;
  // (1) cell-sort permutation order[] (reuse the EAM window grid + its scatter).
  EamCellGrid cg = eam_build_window_grid(d_wx, d_wy, d_wz, m, box_lo, box_len,
                                         periodic, rcut);
  cudaMalloc(&s.d_order, size_t(m) * sizeof(int));
  cudaMemcpy(s.d_order, cg.d_order, size_t(m) * sizeof(int),
             cudaMemcpyDeviceToDevice);

  // (2) gather positions + key into sorted order; s_key = ORIGINAL index.
  cudaMalloc(&s.sx, size_t(m) * sizeof(double));
  cudaMalloc(&s.sy, size_t(m) * sizeof(double));
  cudaMalloc(&s.sz, size_t(m) * sizeof(double));
  cudaMalloc(&s.s_key, size_t(m) * sizeof(long));
  const int blk = kZoneBlock;
  const int gd = (m + blk - 1) / blk;
  eam_sort_gather_kernel<<<gd, blk>>>(d_wx, d_wy, d_wz, d_key, m, cg.d_order, s.sx,
                                      s.sy, s.sz, s.s_key);
  eam_cells_free(cg);  // the sort grid is scratch — only order[] (copied) survives

  // (3) build the tight verlet list OVER the sorted positions (neighbours now
  //     spatially clustered ⇒ the gathers coalesce).
  s.vl = eam_build_verlet_list(s.sx, s.sy, s.sz, m, box_lo, box_len, periodic,
                               rcut, skin);
  return s;
}

}  // namespace tdmd::cuda
