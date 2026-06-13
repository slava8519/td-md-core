#pragma once
// PR-1 (docs/_meta/verlet_skin) — persistent per-zone Verlet neighbour list for
// the GPU ring. A CSR list of partner LOCAL indices within (rcut + skin),
// materialised on a rebuild pass and reused for K>1 passes; the force kernel
// re-tests r2 < rcut2 every pass.
//
// BITWISE TRANSPARENCY BY CONSTRUCTION (B1, same argument as zone_cells.cuh):
// the list only RESTRICTS the candidate set. zone_pair_verlet_kernel applies
// the SAME acceptance predicate (PairGeom::reduce at rcut), the SAME per-pair
// quantization (quantize, Q24.40/Q34.30), the SAME energy single-count rule and
// order-free integer sums as zone_pair_kernel / zone_pair_cells_kernel.
// Therefore the raw int64 accumulators are bit-identical to the cell/tile path
// — PROVIDED the stored list is a SUPERSET of the within-rcut pairs at the
// reuse step. On a single pass (build and force at the same positions) the
// superset is exact and purely geometric: with identical min-image folding,
// r2 < rcut2 implies r2 < (rcut+skin)2, so every within-rcut pair was listed.
// Across reuse passes the superset is guaranteed by the deterministic skin
// budget (charge = 2*R_buf, NL-INV-1/2a; enforced in the conveyor, not here).
//
// Storage: the list is keyed by (zone-id, role) where role ∈ {self, next,
// prev} — a zone only ever pairs with itself and its slab neighbours. Indices
// are LOCAL to the partner zone and stable across passes (static membership:
// upload lays members[] in order, pack/unpack/D2D preserve i->i), so a list
// built when the partner sat in one slot resolves correctly when it later sits
// in another. The cursor-free CSR fill (one owner thread per A-atom) and the
// order-free integer force sum make the build itself bitwise-irrelevant.
#include <cuda_runtime.h>

#include "tdmd/cuda/zone_force.cuh"   // ZoneForceArgs, quantize, pos_double_bits, kZoneBlock
#include "tdmd/cuda/zone_cells.cuh"   // CellGrid

namespace tdmd::cuda {

inline constexpr int kVerletBlock = 128;

// 27-cell (or degenerate) neighbourhood sweep of B around A-atom i, invoking
// FN(j) once per candidate B-atom whose folded separation is < rlist2 (geom is
// a PairGeom built at rcut+skin). Mirrors zone_pair_cells_kernel's iteration so
// count and fill enumerate the identical candidate set in the identical order.
template <typename Fn>
__device__ inline void verlet_for_each_candidate(
    double xi, double yi, double zi, int i, bool same_zone,
    core::PairGeom geom_list, CellGrid bg, const int* b_starts,
    const int* b_counts, const int* b_order, Fn&& fn) {
  int cxi, cyi, czi;
  bg.coords(xi, yi, zi, cxi, cyi, czi);
  const int dzlo = (bg.nz == 1) ? 0 : -1, dzhi = (bg.nz == 1) ? 0 : 1;
  const int dylo = (bg.ny == 1) ? 0 : -1, dyhi = (bg.ny == 1) ? 0 : 1;
  const int dxlo = (bg.nx == 1) ? 0 : -1, dxhi = (bg.nx == 1) ? 0 : 1;
  for (int dz = dzlo; dz <= dzhi; ++dz) {
    int zc = czi + dz;
    if (bg.wrapz) zc = (zc + bg.nz) % bg.nz;
    else if (zc < 0 || zc >= bg.nz) continue;
    for (int dy = dylo; dy <= dyhi; ++dy) {
      int yc = cyi + dy;
      if (bg.wrapy) yc = (yc + bg.ny) % bg.ny;
      else if (yc < 0 || yc >= bg.ny) continue;
      for (int dx = dxlo; dx <= dxhi; ++dx) {
        int xc = cxi + dx;
        if (bg.wrapx) xc = (xc + bg.nx) % bg.nx;
        else if (xc < 0 || xc >= bg.nx) continue;
        const int c = bg.idx(xc, yc, zc);
        const int beg = b_starts[c], cnt = b_counts[c];
        for (int t = beg; t < beg + cnt; ++t) {
          const int j = b_order[t];
          if (same_zone && j == i) continue;
          fn(j);
        }
      }
    }
  }
}

// CSR offsets convention: counts[] has na+1 entries with counts[na]==0;
// ExclusiveSum over na+1 items yields offsets[0..na] with offsets[na]==total.
// (Standard trick — no separate "set total" kernel needed.)

// --- materialisation (rebuild pass) ---------------------------------------
// The B positions here are read through the partner zone's grid `bg`; ax/bx are
// the A/B position arrays of the co-resident slots.

// Step 1: per A-atom, count within-(rcut+skin) partners.
__global__ void verlet_count_kernel(const double* ax, const double* ay,
                                    const double* az, int na, const double* bx,
                                    const double* by, const double* bz,
                                    core::PairGeom geom_list, CellGrid bg,
                                    const int* b_starts, const int* b_counts,
                                    const int* b_order, bool same_zone,
                                    int* counts) {
  const int i = blockIdx.x * kVerletBlock + threadIdx.x;
  if (i >= na) return;
  const double xi = ax[i], yi = ay[i], zi = az[i];
  int c = 0;
  verlet_for_each_candidate(
      xi, yi, zi, i, same_zone, geom_list, bg, b_starts, b_counts, b_order,
      [&](int j) {
        double ddx = xi - bx[j], ddy = yi - by[j], ddz = zi - bz[j], r2;
        if (geom_list.reduce(ddx, ddy, ddz, r2)) ++c;
      });
  counts[i] = c;
}

// Step 2 (after exclusive-scan counts -> offsets[na+1]): fill partner indices.
// One owner thread per A-atom writes its CSR run in iteration order — no
// atomics, so the stored order is deterministic (irrelevant to the force sum,
// but keeps build bitwise-stable for tests/debug).
__global__ void verlet_fill_kernel(const double* ax, const double* ay,
                                   const double* az, int na, const double* bx,
                                   const double* by, const double* bz,
                                   core::PairGeom geom_list, CellGrid bg,
                                   const int* b_starts, const int* b_counts,
                                   const int* b_order, bool same_zone,
                                   const int* offsets, int* idx) {
  const int i = blockIdx.x * kVerletBlock + threadIdx.x;
  if (i >= na) return;
  const double xi = ax[i], yi = ay[i], zi = az[i];
  int w = offsets[i];
  verlet_for_each_candidate(
      xi, yi, zi, i, same_zone, geom_list, bg, b_starts, b_counts, b_order,
      [&](int j) {
        double ddx = xi - bx[j], ddy = yi - by[j], ddz = zi - bz[j], r2;
        if (geom_list.reduce(ddx, ddy, ddz, r2)) idx[w++] = j;
      });
}

// --- reuse force (every pass: rebuild AND reuse) --------------------------
// Same contract/epilogue as zone_pair_cells_kernel, candidates from the stored
// CSR (offsets/idx) instead of the cell neighbourhood. geom is at rcut.
template <typename PairF>
__global__ void zone_pair_verlet_kernel(ZoneForceArgs a, core::PairGeom geom,
                                        PairF pot, const int* offsets,
                                        const int* idx) {
  __shared__ long long sred[kVerletBlock];
  const int i = blockIdx.x * kVerletBlock + threadIdx.x;
  const bool active = i < a.na;
  const double xi = active ? a.ax[i] : 0.0;
  const double yi = active ? a.ay[i] : 0.0;
  const double zi = active ? a.az[i] : 0.0;
  long long qfx = 0, qfy = 0, qfz = 0, qpe = 0;
  double mr2 = 1e300;

  if (active) {
    const int beg = offsets[i], end = offsets[i + 1];
    for (int t = beg; t < end; ++t) {
      const int j = idx[t];
      double ddx = xi - a.bx[j];
      double ddy = yi - a.by[j];
      double ddz = zi - a.bz[j];
      double r2;
      const bool ok = geom.reduce(ddx, ddy, ddz, r2);
      mr2 = fmin(mr2, r2);  // see zone_cells.cuh: min over examined only —
                            // any sub-rcut pair is in the superset, so the
                            // Overlap halt still fires; not cross-path bitwise.
      if (!ok) continue;
      double u, f_over_r;
      pot(sqrt(r2), u, f_over_r);
      qfx += quantize(f_over_r * ddx, core::fixed::ForceAccum::kScale, a.overflow);
      qfy += quantize(f_over_r * ddy, core::fixed::ForceAccum::kScale, a.overflow);
      qfz += quantize(f_over_r * ddz, core::fixed::ForceAccum::kScale, a.overflow);
      if (a.count_energy && (!a.same_zone || i < j))
        qpe += quantize(u, core::fixed::EnergyAccum::kScale, a.overflow);
    }
  }

  if (active) {
    a.fx[i] += qfx;
    a.fy[i] += qfy;
    a.fz[i] += qfz;
    atomicMin(a.min_r2, pos_double_bits(mr2));
  }
  sred[threadIdx.x] = qpe;
  __syncthreads();
  for (int s = kVerletBlock / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicAdd(reinterpret_cast<unsigned long long*>(a.pe),
              (unsigned long long)sred[0]);
}

}  // namespace tdmd::cuda
