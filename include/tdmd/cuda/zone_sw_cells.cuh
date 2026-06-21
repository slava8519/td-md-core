#pragma once
// SW-ladder T5b — CELL-LIST culled Stillinger–Weber GPU kernel. A culled variant of
// zone_sw.cuh's three O(m) window re-scans (the φ₂ neighbour build, the φ₃-center pair
// enumeration over o's nbr[], and the φ₃-wing brute-force re-scan of each neighbour i's
// window neighbours), RAW-int64-BITWISE-EQUAL to the all-window kernel so the bench can
// measure the culling win AND enable the SW entry of the angular flagship (10⁶) table.
// CUDA-only header; --fmad=false MANDATORY (link tdmd_eam_cuda_flags) — NOT for CPU↔GPU
// bitwise (impossible: exp/pow diverge libm-vs-CUDA), but to keep GPU-INTERNAL determinism
// STRUCTURAL (no codegen-varying FMA fusion). zone_sw.cuh / zone_cells.cuh / sw.hpp are
// BYTE-UNTOUCHED — this header only ADDS one new kernel and reuses sw_phi2/sw_triplet (the
// HOST_DEVICE leaves from sw.hpp) verbatim.
//
// THE CULL STRUCTURE — SW IS int64-ORDER-FREE ⇒ NO CANONICAL SORT (the proof SW ≠ MEAM):
// UNLIKE MEAM (whose screening sij = Π_k fcut is an FP64 PRODUCT and dscrfcn an FP64 SUM ⇒
// FP-order-SENSITIVE ⇒ the cells walk must key-sort the stencil k-candidates to stay bitwise),
// SW has NO screening and NO FP reduction inside a triplet: every triplet's f_i/f_j/f_k is a
// SINGLE evaluation (sw_triplet), quantized INDEPENDENTLY into per-owner int64 registers. The
// candidate-set is third-atom-INDEPENDENT (a φ₃ triplet {i;j,k} survives or not purely by the
// two bond cutoffs r_ij,r_ik < a·σ — no k removes a j). So culling only RESTRICTS the candidate
// set; acceptance is the SAME exact predicate (PairGeom::reduce, r2 < rc²) and every accepted
// contribution is quantized identically and INTEGER-summed (order-free, B1). The ±s_d stencil is
// a SUPERSET of the in-cutoff set (the make_zone_grid ulp-skin/pad soundness, EXACTLY as EAM
// cells argue) ⇒ the in-cutoff MULTISET is identical to the all-window sweep ⇒ raw int64
// fx/fy/fz/pe/n_triplets are bit-for-bit equal FOR ANY cell_div, WITHOUT a canonical sort. This
// is the EAM-like proof (zone_eam_cells.cuh), not the MEAM canonical-k proof. G-A is the witness.
//
// THE M1 BUFFER GUARD (M5b acceptance): the kernel GATHERS o's in-rc neighbours into the
// per-thread nb[kMaxNbr] buffer (read by φ₂, the φ₃-center pair loop, and to drive the wing
// role). The all-window kernel already guards this write (atomicOr(overflow,4) past kMaxNbr); the
// cell-walk preserves it verbatim — a neighbour-cap overflow is a STICKY HALT thrown by compute()
// BEFORE writeback, NEVER a silent local-array OOB. G-OVERFLOW (a dense fixture) exercises it.
//
// THE REACH IS ONE WHOLE-WINDOW GRID: the wing role re-centers the stencil on a NEIGHBOUR i of o
// and re-scans i's window neighbours m' — the o-wing triplet {i; o, m'} reaches 2·rcut from o
// (the same graph distance as EAM's density donor; effective_range_for("sw")={2,true}). One
// rc-padded whole-window grid (make_zone_grid n_zones=1) over ALL m atoms covers it (reach is
// sub-cell — a single rc-padded grid is dropped-donor-safe). No second wider grid.
#include <cuda_runtime.h>

#include "tdmd/cuda/zone_cells.cuh"  // CellGrid, make_zone_grid, cell_count/scatter, CUB scan
#include "tdmd/cuda/zone_sw.cuh"     // sw_force_kernel, kMaxNbr, kZoneBlock
#include "tdmd/potentials/sw.hpp"    // SwParams, sw_phi2/sw_triplet (HOST_DEVICE)

namespace tdmd::cuda {

// A device __forceinline__ that walks the ±s_d cell-neighbourhood of an arbitrary CENTER cell
// (qcx,qcy,qcz) and INVOKES the supplied lambda body(b) on every window atom in it. The wing role
// re-centers the grid query on i (NOT o), so the stencil center coords change per role-iteration
// (one extra g.coords each — negligible). The body re-derives geometry itself. STRUCTURAL COPY of
// meam_cell_for_each (zone_meam_cells.cuh) / the eam_*_cells_kernel ±s_d loop.
template <typename Body>
__device__ inline void sw_cell_for_each(CellGrid g, const int* b_starts, const int* b_counts,
                                        const int* b_order, int qcx, int qcy, int qcz, Body body) {
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

// =============================================================================================
// CULLED SW force kernel — sw_force_cells_kernel. FORK of sw_force_kernel (zone_sw.cuh:38-136),
// THREE sub-culls: (1) o's nbr-build (the φ₂ + neighbour-list pass, `for(b<m)`) → ±s_d stencil
// centered on O; (2) φ₃ CENTER pair loop — ALREADY culled (iterates o's cached nb[], the buffer
// IS the cell-neighbourhood); (3) φ₃ WING re-scan (`for(b<m)` over i's window neighbours) → ±s_d
// stencil RE-CENTERED on i. Force output int64 register write-once ⇒ bitwise to all-window by B1
// (NO canonical sort — SW is order-free). Body of each role is byte-identical to the all-window
// kernel; only the neighbour enumeration is replaced by the cell walk.
// =============================================================================================
__global__ void sw_force_cells_kernel(
    const double* wx, const double* wy, const double* wz, const long* key,
    int m, const int* owned, int n_owned, core::PairGeom geom, potentials::SwParams sp,
    CellGrid g, const int* b_starts, const int* b_counts, const int* b_order,
    long long* d_fx, long long* d_fy, long long* d_fz,
    long long* d_pe, long long* d_ntri,
    unsigned long long* d_min_r2, int* overflow) {
  __shared__ long long sred[kZoneBlock];  // reused: PE reduce, then n_triplets reduce
  const int t = blockIdx.x * kZoneBlock + threadIdx.x;
  const double fscale = core::fixed::ForceAccum::kScale;
  const double escale = core::fixed::EnergyAccum::kScale;
  long long qpe = 0, qnt = 0;

  if (t < n_owned) {
    const int o = owned[t];
    long long qfx = 0, qfy = 0, qfz = 0;
    double mr2 = 1e300;
    const double xo = wx[o], yo = wy[o], zo = wz[o];
    const long ko = key[o];
    int ocx, ocy, ocz;
    g.coords(xo, yo, zo, ocx, ocy, ocz);

    // (1) o's own neighbour list, cached once — CULLED to o's cell-neighbourhood. dx = xo − wx[b]
    // (bond FROM o). The φ₂ pair force + energy-once are computed inline EXACTLY as the all-window
    // kernel; only the candidate set (`for b<m` → ±s_d stencil) changes.
    int nb[kMaxNbr];
    double nbx[kMaxNbr], nby[kMaxNbr], nbz[kMaxNbr], nbr_[kMaxNbr];
    int cnt = 0;
    sw_cell_for_each(g, b_starts, b_counts, b_order, ocx, ocy, ocz, [&](int b) {
      if (b == o) return;
      double dx = xo - wx[b], dy = yo - wy[b], dz = zo - wz[b], r2;
      if (!geom.reduce(dx, dy, dz, r2)) return;
      mr2 = fmin(mr2, r2);                 // B10 overlap probe (in-cutoff pairs)
      const double r = sqrt(r2);
      // (i) φ₂: o's pair force; the lower GLOBAL key owns the energy.
      double phi, dphi;
      potentials::sw_phi2(sp, r, phi, dphi);
      const double f_over_r = -dphi / r;
      qfx += quantize(f_over_r * dx, fscale, overflow);
      qfy += quantize(f_over_r * dy, fscale, overflow);
      qfz += quantize(f_over_r * dz, fscale, overflow);
      if (ko < key[b]) qpe += quantize(phi, escale, overflow);
      if (cnt < kMaxNbr) {
        nb[cnt] = b; nbx[cnt] = dx; nby[cnt] = dy; nbz[cnt] = dz; nbr_[cnt] = r; ++cnt;
      } else {
        atomicOr(overflow, 4);            // STICKY: neighbour cap exceeded → HALT (M1 guard)
      }
    });

    // (ii) φ₃ CENTER: every unordered wing pair (x<y); o owns f_i + E3 + count ONCE. Iterates o's
    // cached nb[] — ALREADY culled (the buffer IS the cell-neighbourhood). Byte-identical to the
    // all-window center loop.
    for (int x = 0; x < cnt; ++x)
      for (int y = x + 1; y < cnt; ++y) {
        potentials::SwVec3 fi, fj, fk;
        const double E3 = potentials::sw_triplet(sp, nbx[x], nby[x], nbz[x], nbr_[x],
                                                 nbx[y], nby[y], nbz[y], nbr_[y], fi, fj, fk);
        qfx += quantize(fi.x, fscale, overflow);
        qfy += quantize(fi.y, fscale, overflow);
        qfz += quantize(fi.z, fscale, overflow);
        qpe += quantize(E3, escale, overflow);
        ++qnt;
      }

    // (iii) φ₃ WING: for each neighbour i of o, replay every triplet {i; o, m'} (m' in i's window
    // nbr list), o in the j-slot. NO energy, NO count. THE NON-SYMMETRIC WRITE: o gets f_j of a
    // triplet centered at ANOTHER atom i. The all-window `for(b<m)` re-scan of i's neighbours →
    // ±s_d stencil RE-CENTERED on i. Body byte-identical to the all-window wing role.
    for (int e = 0; e < cnt; ++e) {
      const int i = nb[e];
      const double iox = -nbx[e], ioy = -nby[e], ioz = -nbz[e], rio = nbr_[e];  // x_i − x_o
      const double xi = wx[i], yi = wy[i], zi = wz[i];
      int icx, icy, icz;
      g.coords(xi, yi, zi, icx, icy, icz);
      sw_cell_for_each(g, b_starts, b_counts, b_order, icx, icy, icz, [&](int b) {
        if (b == i || b == o) return;   // m' != o AND m' != i
        double mdx = xi - wx[b], mdy = yi - wy[b], mdz = zi - wz[b], r2m;
        if (!geom.reduce(mdx, mdy, mdz, r2m)) return;
        potentials::SwVec3 fi, fj, fk;
        potentials::sw_triplet(sp, iox, ioy, ioz, rio, mdx, mdy, mdz, sqrt(r2m), fi, fj, fk);
        qfx += quantize(fj.x, fscale, overflow);
        qfy += quantize(fj.y, fscale, overflow);
        qfz += quantize(fj.z, fscale, overflow);
      });
    }

    d_fx[o] = qfx; d_fy[o] = qfy; d_fz[o] = qfz;  // window-local o, write-once
    atomicMin(d_min_r2, pos_double_bits(mr2));
  }

  // PE block tree-reduce → atomicAdd (zone_force epilogue).
  sred[threadIdx.x] = qpe; __syncthreads();
  for (int s = kZoneBlock / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicAdd(reinterpret_cast<unsigned long long*>(d_pe), (unsigned long long)sred[0]);
  __syncthreads();
  // n_triplets block tree-reduce → atomicAdd (second reduction, reusing sred).
  sred[threadIdx.x] = qnt; __syncthreads();
  for (int s = kZoneBlock / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicAdd(reinterpret_cast<unsigned long long*>(d_ntri), (unsigned long long)sred[0]);
}

// --- host helper: build the WINDOW cell grid (the dropped-donor-safe extent, one rc-padded grid
// over ALL m atoms — covers o's neighbours + the wing role's 2·rcut reach; reach is sub-cell).
// STRUCTURAL COPY of EamCellGrid / eam_build_window_grid (zone_eam_cells.cuh), renamed. ---
struct SwCellGrid {
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

inline void sw_cells_free(SwCellGrid& c) {
  for (void* p : {(void*)c.d_cell_of, (void*)c.d_counts, (void*)c.d_starts,
                  (void*)c.d_cursor, (void*)c.d_order, c.d_cub})
    if (p) cudaFree(p);
  c = SwCellGrid{};
}

inline SwCellGrid sw_build_window_grid(const double* d_wx, const double* d_wy, const double* d_wz,
                                       int m, const double box_lo[3], const double box_len[3],
                                       const bool periodic[3], double rcut, int cell_div = 1) {
  SwCellGrid c;
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
