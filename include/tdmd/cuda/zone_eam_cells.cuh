#pragma once
// M6 E5c (bake-off, first piece) — CELL-LIST culled EAM GPU kernels. A culled
// variant of zone_eam.cuh's O(m²) density + force sweeps, BITWISE-EQUAL to the
// all-window kernels by B1/INV-9, so the bench can measure the culling win on
// the heavy density+force passes (answering "is a culled neighbour backend
// justified for EAM" with DATA). CUDA-only header; --fmad=false MANDATORY
// (link tdmd_eam_cuda_flags — the f_over_r prefactor and the Horner steps fuse
// to fma otherwise and break 1 ULP). zone_eam.cuh / zone_cells.cuh are
// BYTE-UNTOUCHED — this header only adds new kernels and reuses theirs verbatim.
//
// BITWISE TRANSPARENCY (the zone_cells.cuh argument, applied to EAM): culling
// only restricts the CANDIDATE set. Acceptance is the SAME exact predicate
// (PairGeom::reduce, r2 < rc²); every accepted pair's contribution is quantized
// identically and integer-summed (order-free, B1). The 27-cell neighbourhood is
// a SUPERSET of the in-cutoff set (the ulp-skin/clamp/pad soundness in
// make_zone_grid), so the in-cutoff MULTISET is identical to the O(m²) sweep ⇒
// raw int64 ρ / fx / fy / fz / pe are bit-for-bit equal. min_r2 is the min over
// EXAMINED candidates only — it agrees on the Overlap-HALT boolean (any pair
// under rcut is in both candidate sets) but is NOT bitwise comparable across
// paths and must never be compared bit-for-bit.
//
// THE DROPPED-DONOR TRAP (M6 §3.5; THE risk this header must not commit): EAM
// density must be COMPLETE for every WINDOW atom that is a neighbour of an owned
// atom — the ≥2·rcut residence window. A grid that covers only an owned slab
// would truncate ρ on a non-owned-but-neighbour atom ⇒ wrong F'(ρ) ⇒ wrong
// force, and that truncation is DETERMINISTIC ⇒ invisible to cells-vs-itself
// (every path that culls the same way agrees). So the grid here spans the WHOLE
// gathered window extent (rcut-padded cells over ALL m atoms): we build it via
// make_zone_grid(.., n_zones=1, zone_id=0) over the window's box, which yields a
// single rcut-padded grid (periodic box → periodic grid; free box → AABB slab,
// clamp-into-edge). ONLY the test-B cross-check vs eam_direct_fp64 witnesses
// completeness; that test is non-negotiable.
#include <cuda_runtime.h>

#include "tdmd/cuda/zone_cells.cuh"        // CellGrid, make_zone_grid, cell_count/scatter, CUB scan
#include "tdmd/cuda/zone_eam.cuh"          // EamSetflView, quantize, pos_double_bits, kZoneBlock

namespace tdmd::cuda {

// pass 1 (CULLED): ρ_aa = Σ_{bb≠aa, bb in nbr-cells} ρ_a(r). Bit-for-bit ==
// eam_density_kernel: same int64 register, same self-skip (bb==aa, window-local
// index identity — order[] carries window-local indices), same geom.reduce cut,
// same quantize, same dens_scale. Single d_rho[aa] write (no cross-thread). The
// 27-cell (or degenerate) neighbourhood loop is copied verbatim from
// zone_pair_cells_kernel (same dxlo/dxhi clamp + wrap).
__global__ void eam_density_cells_kernel(const double* wx, const double* wy,
                                         const double* wz, int m,
                                         core::PairGeom geom, EamSetflView eam,
                                         double dens_scale, CellGrid g,
                                         const int* b_starts, const int* b_counts,
                                         const int* b_order, long long* d_rho,
                                         int* overflow) {
  const int aa = blockIdx.x * kZoneBlock + threadIdx.x;
  if (aa >= m) return;
  long long qrho = 0;
  const double xi = wx[aa], yi = wy[aa], zi = wz[aa];
  int cxi, cyi, czi;
  g.coords(xi, yi, zi, cxi, cyi, czi);
  // ±s_d stencil (sub-rcut: cells ~rcut/cell_div ⇒ s_d=cell_div; legacy s=1). s_d is
  // ceil((rcut+pad)/c) over the realized cell ⇒ spans >=rcut ⇒ superset ⇒ bitwise.
  const int dzlo = (g.nz == 1) ? 0 : -g.sz, dzhi = (g.nz == 1) ? 0 : g.sz;
  const int dylo = (g.ny == 1) ? 0 : -g.sy, dyhi = (g.ny == 1) ? 0 : g.sy;
  const int dxlo = (g.nx == 1) ? 0 : -g.sx, dxhi = (g.nx == 1) ? 0 : g.sx;
  for (int dz = dzlo; dz <= dzhi; ++dz) {
    int zc = czi + dz;
    if (g.wrapz) zc = ((zc % g.nz) + g.nz) % g.nz;  // |dz|>1-safe (g.nz>=2s+1 guaranteed)
    else if (zc < 0 || zc >= g.nz) continue;
    for (int dy = dylo; dy <= dyhi; ++dy) {
      int yc = cyi + dy;
      if (g.wrapy) yc = ((yc % g.ny) + g.ny) % g.ny;
      else if (yc < 0 || yc >= g.ny) continue;
      for (int dx = dxlo; dx <= dxhi; ++dx) {
        int xc = cxi + dx;
        if (g.wrapx) xc = ((xc % g.nx) + g.nx) % g.nx;
        else if (xc < 0 || xc >= g.nx) continue;
        const int c = g.idx(xc, yc, zc);
        const int beg = b_starts[c], cnt = b_counts[c];
        for (int t = beg; t < beg + cnt; ++t) {
          const int bb = b_order[t];
          if (bb == aa) continue;
          double ddx = xi - wx[bb], ddy = yi - wy[bb], ddz = zi - wz[bb], r2;
          if (!geom.reduce(ddx, ddy, ddz, r2)) continue;
          double v, dv;
          eam.eval_rhoa(sqrt(r2), v, dv);
          qrho += quantize(v, dens_scale, overflow);
        }
      }
    }
  }
  d_rho[aa] = qrho;
}

// pass 3 (CULLED): full-neighbour force for OWNED atoms + pe. IDENTICAL to
// eam_force_kernel except the inner `for(bb<m)` is the 27-cell loop. Kept
// verbatim: quantize() calls, the f_over_r line, the `ki<key[bb]` φ-once gate,
// qfx/y/z registers, sred[] tree-reduce + one atomicAdd, atomicMin(d_min_r2)
// BEFORE the cutoff.
__global__ void eam_force_cells_kernel(
    const double* wx, const double* wy, const double* wz, const long* key, int m,
    const int* owned, int n_owned, core::PairGeom geom, EamSetflView eam,
    double dens_scale, const long long* d_rho, const double* d_fp, CellGrid g,
    const int* b_starts, const int* b_counts, const int* b_order, long long* d_fx,
    long long* d_fy, long long* d_fz, long long* d_pe,
    unsigned long long* d_min_r2, int* overflow) {
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
    int cxi, cyi, czi;
    g.coords(xi, yi, zi, cxi, cyi, czi);
    // ±s_d stencil (sub-rcut; see eam_density_cells_kernel). Superset ⇒ bitwise.
    const int dzlo = (g.nz == 1) ? 0 : -g.sz, dzhi = (g.nz == 1) ? 0 : g.sz;
    const int dylo = (g.ny == 1) ? 0 : -g.sy, dyhi = (g.ny == 1) ? 0 : g.sy;
    const int dxlo = (g.nx == 1) ? 0 : -g.sx, dxhi = (g.nx == 1) ? 0 : g.sx;
    for (int dz = dzlo; dz <= dzhi; ++dz) {
      int zc = czi + dz;
      if (g.wrapz) zc = ((zc % g.nz) + g.nz) % g.nz;  // |dz|>1-safe
      else if (zc < 0 || zc >= g.nz) continue;
      for (int dy = dylo; dy <= dyhi; ++dy) {
        int yc = cyi + dy;
        if (g.wrapy) yc = ((yc % g.ny) + g.ny) % g.ny;
        else if (yc < 0 || yc >= g.ny) continue;
        for (int dx = dxlo; dx <= dxhi; ++dx) {
          int xc = cxi + dx;
          if (g.wrapx) xc = ((xc % g.nx) + g.nx) % g.nx;
          else if (xc < 0 || xc >= g.nx) continue;
          const int c = g.idx(xc, yc, zc);
          const int beg = b_starts[c], cnt = b_counts[c];
          for (int t = beg; t < beg + cnt; ++t) {
            const int bb = b_order[t];
            if (bb == ii) continue;
            double ddx = xi - wx[bb], ddy = yi - wy[bb], ddz = zi - wz[bb], r2;
            const bool ok = geom.reduce(ddx, ddy, ddz, r2);
            mr2 = fmin(mr2, r2);  // BEFORE the cutoff (pass3 only) — overlap probe
            if (!ok) continue;
            const double r = sqrt(r2);
            double phi, dphi, ra, dra;
            eam.eval_phi(r, phi, dphi);
            eam.eval_rhoa(r, ra, dra);
            const double f_over_r = -(dphi + (fpi + d_fp[bb]) * dra) / r;  // FMA-killer site
            qfx += quantize(f_over_r * ddx, core::fixed::ForceAccum::kScale, overflow);
            qfy += quantize(f_over_r * ddy, core::fixed::ForceAccum::kScale, overflow);
            qfz += quantize(f_over_r * ddz, core::fixed::ForceAccum::kScale, overflow);
            if (ki < key[bb]) qpe += quantize(phi, core::fixed::EnergyAccum::kScale, overflow);
          }
        }
      }
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

// --- host helper: build the WINDOW cell grid (the dropped-donor-safe extent) ---
//
// One grid over the WHOLE gathered window (ALL m atoms), rcut-padded, so ρ is
// complete for every window atom. Reuses make_zone_grid with n_zones=1 — that
// path produces a single rcut-padded grid covering the box (periodic box →
// periodic grid in each periodic dim; free/non-periodic dim → AABB slab padded
// by rcut, clamp-into-edge). The window's box IS the grid extent; all m atoms
// fall inside (or clamp into) it. The CUB scan turns per-cell counts into
// starts (exclusive prefix sum). All buffers are device-side and caller-freed.
struct EamCellGrid {
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

inline void eam_cells_free(EamCellGrid& c) {
  for (void* p : {(void*)c.d_cell_of, (void*)c.d_counts, (void*)c.d_starts,
                  (void*)c.d_cursor, (void*)c.d_order, c.d_cub})
    if (p) cudaFree(p);
  c = EamCellGrid{};
}

// Build the grid over the device window positions (d_wx/d_wy/d_wz, m atoms).
// box_lo/box_len/periodic describe the window's box (the SAME box PairGeom uses).
inline EamCellGrid eam_build_window_grid(const double* d_wx, const double* d_wy,
                                         const double* d_wz, int m,
                                         const double box_lo[3],
                                         const double box_len[3],
                                         const bool periodic[3], double rcut,
                                         int cell_div = 1) {
  EamCellGrid c;
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

  // exclusive prefix sum counts -> starts (CUB, ТЗ §2.2)
  cub::DeviceScan::ExclusiveSum(nullptr, c.cub_bytes, c.d_counts, c.d_starts,
                                c.ncells);
  cudaMalloc(&c.d_cub, c.cub_bytes);
  cub::DeviceScan::ExclusiveSum(c.d_cub, c.cub_bytes, c.d_counts, c.d_starts,
                                c.ncells);
  cudaMemcpy(c.d_cursor, c.d_starts, size_t(c.ncells) * sizeof(int),
             cudaMemcpyDeviceToDevice);
  cell_scatter_kernel<<<cg, blk>>>(c.d_cell_of, m, c.d_cursor, c.d_order);
  return c;
}

}  // namespace tdmd::cuda
