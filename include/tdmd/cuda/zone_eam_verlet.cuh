#pragma once
// M6 E5c (bake-off, TIGHT-LIST piece) — TIGHT per-atom VERLET neighbour-list EAM
// GPU kernels. A second culled variant of zone_eam.cuh's O(m²) density + force
// sweeps, BITWISE-EQUAL to the all-window kernels (and therefore to the cell-list
// kernels in zone_eam_cells.cuh) by B1/INV-9. Where zone_eam_cells.cuh iterates
// the 27-cell neighbourhood every pass (over-fetching ~8× — a full cell shell vs
// the in-cutoff sphere), this header walks a TIGHT per-atom CSR list of the
// ~(real + skin) neighbours, exactly like LAMMPS's Verlet half/full list, so the
// bench can quantify how much of the ~10× gap vs LAMMPS GPU EAM is the neighbour
// over-fetch. CUDA-only header; --fmad=false MANDATORY (link tdmd_eam_cuda_flags —
// the f_over_r prefactor and the Horner steps fuse to fma otherwise and break
// 1 ULP). zone_eam.cuh / zone_eam_cells.cuh / zone_cells.cuh are BYTE-UNTOUCHED —
// this header only ADDS new kernels and reuses theirs verbatim.
//
// BITWISE TRANSPARENCY (the zone_cells.cuh / zone_eam_cells.cuh argument, applied
// to a tight list): the list enumerates a DIFFERENT candidate set, nothing else.
// The list is built within (rcut + skin), a SUPERSET of the in-cutoff set; every
// kernel re-applies the EXACT acceptance predicate (PairGeom::reduce, r2 < rcut²)
// and quantizes + integer-sums identically (order-free, B1). So the in-cutoff
// MULTISET is identical to the O(m²) sweep ⇒ raw int64 ρ / fx / fy / fz / pe are
// bit-for-bit equal. min_r2 is the min over EXAMINED candidates only — it agrees
// on the Overlap-HALT boolean (any pair under rcut is in the list) but is NOT
// bitwise comparable across paths and must never be compared bit-for-bit.
//
// FULL-NEIGHBOUR REQUIREMENT (NON-NEGOTIABLE): the list must contain BOTH
// directions — aa's list has bb AND bb's list has aa. The density pass needs the
// full neighbour set (no Newton-3 sharing; aa sums its own neighbours) and the
// force pass's φ-once PE gate (ki < key[bb]) only counts each undirected pair
// once IF both endpoints see each other. A half list would drop ρ donors AND
// half the pair energy. The build below counts/fills bb≠aa symmetrically (the
// list is full by construction).
//
// DROPPED-DONOR SAFETY (M6 §3.5): same as zone_eam_cells.cuh — ρ must be COMPLETE
// for every WINDOW atom. The list is built from a grid spanning the WHOLE window
// (all m atoms), sized for (rcut + skin) cells so the 27-cell build-walk captures
// every (rcut+skin) neighbour. The Test-B' cross-check vs eam_direct_fp64 is the
// only witness of completeness and is non-negotiable.
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include "tdmd/cuda/zone_cells.cuh"       // CellGrid, make_zone_grid, cell_count/scatter, CUB scan
#include "tdmd/cuda/zone_eam.cuh"         // EamSetflView, quantize, pos_double_bits, kZoneBlock
#include "tdmd/cuda/zone_eam_cells.cuh"   // eam_build_window_grid / EamCellGrid (build the list FROM)

namespace tdmd::cuda {

// Device per-atom CSR neighbour list over the window within (rcut + skin).
//   d_off[m+1] : exclusive-scan offsets (d_off[aa]..d_off[aa+1] is aa's list)
//   d_idx[nnz] : window-local neighbour indices (FULL — both directions)
// nnz is the total neighbour count (sum over atoms). avg_neighbours = nnz/m is
// what the bench reports as "tight list size" against the cell-list's
// examined/atom, to make the over-fetch visible.
struct EamVerletList {
  int* d_off = nullptr;  // m+1
  int* d_idx = nullptr;  // nnz
  void* d_cub = nullptr; // CUB scan scratch
  size_t cub_bytes = 0;
  int m = 0;
  long long nnz = 0;
};

inline void eam_verlet_free(EamVerletList& v) {
  for (void* p : {(void*)v.d_off, (void*)v.d_idx, v.d_cub})
    if (p) cudaFree(p);
  v = EamVerletList{};
}

// COUNT kernel: per atom aa, walk the 27-cell neighbourhood of the (rcut+skin)
// grid, count bb≠aa with min-image r2 < (rcut+skin)². The min-image reduction is
// the SAME as PairGeom::reduce but against the EXTENDED cutoff rc2_ext — the list
// is a superset; the in-cutoff re-test happens in the iterate kernels. Writes the
// per-atom count into d_off[aa] (offsets are scanned to exclusive-sum after).
__global__ void eam_verlet_count_kernel(const double* wx, const double* wy,
                                        const double* wz, int m,
                                        core::PairGeom geom_ext, CellGrid g,
                                        const int* b_starts, const int* b_counts,
                                        const int* b_order, int* d_off) {
  const int aa = blockIdx.x * kZoneBlock + threadIdx.x;
  if (aa >= m) return;
  int cnt_nbr = 0;
  const double xi = wx[aa], yi = wy[aa], zi = wz[aa];
  int cxi, cyi, czi;
  g.coords(xi, yi, zi, cxi, cyi, czi);
  const int dzlo = (g.nz == 1) ? 0 : -1, dzhi = (g.nz == 1) ? 0 : 1;
  const int dylo = (g.ny == 1) ? 0 : -1, dyhi = (g.ny == 1) ? 0 : 1;
  const int dxlo = (g.nx == 1) ? 0 : -1, dxhi = (g.nx == 1) ? 0 : 1;
  for (int dz = dzlo; dz <= dzhi; ++dz) {
    int zc = czi + dz;
    if (g.wrapz) zc = (zc + g.nz) % g.nz;
    else if (zc < 0 || zc >= g.nz) continue;
    for (int dy = dylo; dy <= dyhi; ++dy) {
      int yc = cyi + dy;
      if (g.wrapy) yc = (yc + g.ny) % g.ny;
      else if (yc < 0 || yc >= g.ny) continue;
      for (int dx = dxlo; dx <= dxhi; ++dx) {
        int xc = cxi + dx;
        if (g.wrapx) xc = (xc + g.nx) % g.nx;
        else if (xc < 0 || xc >= g.nx) continue;
        const int c = g.idx(xc, yc, zc);
        const int beg = b_starts[c], cn = b_counts[c];
        for (int t = beg; t < beg + cn; ++t) {
          const int bb = b_order[t];
          if (bb == aa) continue;
          double ddx = xi - wx[bb], ddy = yi - wy[bb], ddz = zi - wz[bb], r2;
          if (!geom_ext.reduce(ddx, ddy, ddz, r2)) continue;  // r2 < (rcut+skin)²
          ++cnt_nbr;
        }
      }
    }
  }
  d_off[aa] = cnt_nbr;
}

// FILL kernel: same neighbourhood walk as COUNT; writes aa's neighbour
// window-local indices into d_idx[d_off[aa] .. d_off[aa+1]). The per-atom write
// region is private (no cross-thread races — the cursor is the atom's own
// offset), so the order WITHIN a list is deterministic by construction (the grid
// scatter order). Order-irrelevance for the accumulators is anyway guaranteed by
// int64 associativity (B1) — but a private cursor keeps the build race-free.
__global__ void eam_verlet_fill_kernel(const double* wx, const double* wy,
                                       const double* wz, int m,
                                       core::PairGeom geom_ext, CellGrid g,
                                       const int* b_starts, const int* b_counts,
                                       const int* b_order, const int* d_off,
                                       int* d_idx) {
  const int aa = blockIdx.x * kZoneBlock + threadIdx.x;
  if (aa >= m) return;
  int cur = d_off[aa];
  const double xi = wx[aa], yi = wy[aa], zi = wz[aa];
  int cxi, cyi, czi;
  g.coords(xi, yi, zi, cxi, cyi, czi);
  const int dzlo = (g.nz == 1) ? 0 : -1, dzhi = (g.nz == 1) ? 0 : 1;
  const int dylo = (g.ny == 1) ? 0 : -1, dyhi = (g.ny == 1) ? 0 : 1;
  const int dxlo = (g.nx == 1) ? 0 : -1, dxhi = (g.nx == 1) ? 0 : 1;
  for (int dz = dzlo; dz <= dzhi; ++dz) {
    int zc = czi + dz;
    if (g.wrapz) zc = (zc + g.nz) % g.nz;
    else if (zc < 0 || zc >= g.nz) continue;
    for (int dy = dylo; dy <= dyhi; ++dy) {
      int yc = cyi + dy;
      if (g.wrapy) yc = (yc + g.ny) % g.ny;
      else if (yc < 0 || yc >= g.ny) continue;
      for (int dx = dxlo; dx <= dxhi; ++dx) {
        int xc = cxi + dx;
        if (g.wrapx) xc = (xc + g.nx) % g.nx;
        else if (xc < 0 || xc >= g.nx) continue;
        const int c = g.idx(xc, yc, zc);
        const int beg = b_starts[c], cn = b_counts[c];
        for (int t = beg; t < beg + cn; ++t) {
          const int bb = b_order[t];
          if (bb == aa) continue;
          double ddx = xi - wx[bb], ddy = yi - wy[bb], ddz = zi - wz[bb], r2;
          if (!geom_ext.reduce(ddx, ddy, ddz, r2)) continue;  // r2 < (rcut+skin)²
          d_idx[cur++] = bb;
        }
      }
    }
  }
}

// pass 1 (TIGHT LIST): ρ_aa = Σ_{bb in aa's list} ρ_a(r). Bit-for-bit ==
// eam_density_kernel / eam_density_cells_kernel: same int64 register, same EXACT
// in-cutoff re-test (geom.reduce, r2 < rcut²), same quantize, same dens_scale,
// single d_rho[aa] write. The 27-cell loop is replaced by a flat CSR iterate.
__global__ void eam_density_verlet_kernel(const double* wx, const double* wy,
                                          const double* wz, int m,
                                          core::PairGeom geom, EamSetflView eam,
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
    // bb != aa by construction (the build skips self); no self-check needed, but
    // the in-cutoff predicate IS re-applied (the list is rcut+skin).
    double ddx = xi - wx[bb], ddy = yi - wy[bb], ddz = zi - wz[bb], r2;
    if (!geom.reduce(ddx, ddy, ddz, r2)) continue;  // EXACT rcut re-test
    double v, dv;
    eam.eval_rhoa(sqrt(r2), v, dv);
    qrho += quantize(v, dens_scale, overflow);
  }
  d_rho[aa] = qrho;
}

// pass 3 (TIGHT LIST): full-neighbour force for OWNED atoms + pe. IDENTICAL to
// eam_force_kernel / eam_force_cells_kernel except the inner candidate loop is
// the flat CSR iterate over aa's tight list. Kept verbatim: the eval_F embedding
// once per owned, quantize() calls, the f_over_r line, the `ki<key[bb]` φ-once
// gate, qfx/y/z registers, sred[] tree-reduce + one atomicAdd, atomicMin
// (d_min_r2) BEFORE the cutoff. The list is full-neighbour ⇒ the φ-once gate fires
// exactly as all-window. NB: owned[] here indexes the WINDOW; the list is keyed by
// window-local index, so d_off[ii] is the owned atom's own list.
__global__ void eam_force_verlet_kernel(
    const double* wx, const double* wy, const double* wz, const long* key, int m,
    const int* owned, int n_owned, core::PairGeom geom, EamSetflView eam,
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
    double Fi, Fpi;
    eam.eval_F(double(d_rho[ii]) / dens_scale, Fi, Fpi);  // embedding energy once per owned
    qpe += quantize(Fi, core::fixed::EnergyAccum::kScale, overflow);
    const double xi = wx[ii], yi = wy[ii], zi = wz[ii];
    const double fpi = d_fp[ii];
    const long ki = key[ii];
    const int beg = d_off[ii], end = d_off[ii + 1];
    for (int t = beg; t < end; ++t) {
      const int bb = d_idx[t];
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

// --- host helper: build the TIGHT per-atom CSR list over the window ----------
//
// Two passes from the (rcut+skin)-sized window grid (built here, separate from
// the cell-list's rcut grid — a 27-cell walk of an rcut grid only guarantees rcut
// reach, but we must capture every (rcut+skin) neighbour): COUNT → CUB
// ExclusiveSum → FILL. geom_ext is the EXTENDED-cutoff PairGeom ((rcut+skin)²);
// the iterate kernels take the ordinary geom (rcut²) for the exact re-test. All
// buffers are device-side and caller-freed (eam_verlet_free).
//
// box_lo/box_len/periodic describe the window's box (the SAME box PairGeom uses);
// rcut is the potential cutoff, skin the verlet skin (≥0). The grid built here is
// LOCAL (built + freed inside) — only the CSR list survives.
inline EamVerletList eam_build_verlet_list(const double* d_wx, const double* d_wy,
                                           const double* d_wz, int m,
                                           const double box_lo[3],
                                           const double box_len[3],
                                           const bool periodic[3], double rcut,
                                           double skin) {
  EamVerletList v;
  v.m = m;
  const double rc_ext = rcut + skin;
  // Build a window grid sized for (rcut+skin) so the 27-cell build-walk reaches
  // every (rcut+skin) neighbour (dropped-donor-safe — same WHOLE-window extent
  // argument as eam_build_window_grid, just a coarser cell so the wider shell
  // fits the ±1 neighbourhood).
  EamCellGrid g = eam_build_window_grid(d_wx, d_wy, d_wz, m, box_lo, box_len,
                                        periodic, rc_ext);

  // PairGeom for the EXTENDED cutoff (the list candidate predicate). Constructed
  // host-side from a Box mirror; passed by value into the build kernels.
  core::Box box;
  box.lo = {box_lo[0], box_lo[1], box_lo[2]};
  box.hi = {box_lo[0] + box_len[0], box_lo[1] + box_len[1], box_lo[2] + box_len[2]};
  box.periodic = {periodic[0], periodic[1], periodic[2]};
  const core::PairGeom geom_ext(box, rc_ext);

  cudaMalloc(&v.d_off, size_t(m + 1) * sizeof(int));
  cudaMemset(v.d_off, 0, size_t(m + 1) * sizeof(int));
  const int blk = kZoneBlock;
  const int gd = (m + blk - 1) / blk;

  // COUNT: write per-atom counts into d_off[0..m-1] (d_off[m] left 0 for the scan).
  eam_verlet_count_kernel<<<gd, blk>>>(d_wx, d_wy, d_wz, m, geom_ext, g.g,
                                       g.d_starts, g.d_counts, g.d_order, v.d_off);

  // exclusive prefix sum counts -> offsets (CUB, ТЗ §2.2). Scan over m+1 so
  // d_off[m] becomes the total nnz (ExclusiveSum of [c0..c_{m-1}, 0]).
  cub::DeviceScan::ExclusiveSum(nullptr, v.cub_bytes, v.d_off, v.d_off, m + 1);
  cudaMalloc(&v.d_cub, v.cub_bytes);
  cub::DeviceScan::ExclusiveSum(v.d_cub, v.cub_bytes, v.d_off, v.d_off, m + 1);

  int nnz = 0;
  cudaMemcpy(&nnz, v.d_off + m, sizeof(int), cudaMemcpyDeviceToHost);
  v.nnz = nnz;
  cudaMalloc(&v.d_idx, size_t(nnz > 0 ? nnz : 1) * sizeof(int));

  // FILL: each atom writes its neighbours into [d_off[aa], d_off[aa+1]).
  eam_verlet_fill_kernel<<<gd, blk>>>(d_wx, d_wy, d_wz, m, geom_ext, g.g,
                                      g.d_starts, g.d_counts, g.d_order, v.d_off,
                                      v.d_idx);

  eam_cells_free(g);  // the build grid is scratch — only the CSR list survives
  return v;
}

}  // namespace tdmd::cuda
