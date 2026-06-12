#pragma once
// Post-M5a — per-zone cell-list culling for the GPU ring (the absolute
// atom-steps/s lever recorded in TD_MD_Core_Bench_v1_0.md). CUDA-only header.
//
// BITWISE TRANSPARENCY BY CONSTRUCTION (B1): culling only restricts the
// CANDIDATE set; acceptance is the same exact predicate (PairGeom::reduce,
// r2 < rc²) and every accepted pair's contribution is quantized identically
// and integer-summed (order-free). Therefore the cell path produces raw
// accumulators bit-identical to the tile path — and every existing bitwise
// claim (GPU == CPU ring, 1-vs-N streams, MPI == single-process, mixed
// 1-vs-z) survives untouched. The cell-build scatter order is even allowed
// to be nondeterministic (atomicAdd cursors): integer associativity makes
// the iteration order irrelevant.
//
// Grid design (one grid per zone, rebuilt every pass after the drift —
// zones are thin, the build is 3 tiny kernels + a CUB scan, ТЗ §2.2 allows
// CUB from the toolkit):
//   x/y: span the box, cell >= rcut; periodic dims FOLD coordinates into the
//        box; if a periodic dim has < 3 cells it degenerates to 1 cell (a
//        3-neighborhood would visit the same cell through two images).
//   z:   a slab grid [zone_lo - rcut, zone_hi + rcut], no wrap; queries from
//        a NEIGHBOUR zone fold z to the nearest image of the grid center
//        (PBC closure: the tail queries the head across the box seam).
//        Exception n_zones == 1 with periodic z: the grid is periodic in z
//        like x/y (the zone IS the box).
//   Out-of-range coordinates clamp into edge cells — sound: a pair within
//   rcut always lands in adjacent (or identical) cells because the clamp
//   only moves points from beyond a face onto it, and any in-range partner
//   of an outlier lies within rcut of that face.
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include "tdmd/cuda/zone_force.cuh"

namespace tdmd::cuda {

struct CellGrid {
  double lox, loy, loz;
  double cx, cy, cz;       // cell sizes, each >= rcut
  int nx, ny, nz;
  bool wrapx, wrapy, wrapz;
  double Lx, Ly, Lz;       // box lengths (coordinate folding)
  double zc;               // grid z-center (nearest-image query fold)
  bool perz_box;           // box periodic in z (fold queries across the seam)

  TDMD_HOST_DEVICE int ncells() const { return nx * ny * nz; }
  TDMD_HOST_DEVICE int idx(int ix, int iy, int iz) const {
    return (iz * ny + iy) * nx + ix;
  }
  TDMD_HOST_DEVICE void coords(double x, double y, double z, int& ix, int& iy,
                               int& iz) const {
    if (wrapx) x -= Lx * std::floor((x - lox) / Lx);
    if (wrapy) y -= Ly * std::floor((y - loy) / Ly);
    if (wrapz) z -= Lz * std::floor((z - loz) / Lz);
    else if (perz_box) z -= Lz * std::rint((z - zc) / Lz);
    auto cl = [](int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); };
    ix = cl(int(std::floor((x - lox) / cx)), nx);
    iy = cl(int(std::floor((y - loy) / cy)), ny);
    iz = cl(int(std::floor((z - loz) / cz)), nz);
  }
};

// Grid geometry for zone `zone_id` of an n_zones decomposition (host).
inline CellGrid make_zone_grid(const double box_lo[3], const double box_len[3],
                               const bool periodic[3], double rcut,
                               int n_zones, int zone_id) {
  auto ncell = [&](double L, bool per) {
    int n = int(L / rcut);
    if (n < 1) n = 1;
    if (per && n < 3) n = 1;  // avoid double-visiting images
    return n;
  };
  CellGrid g{};
  g.Lx = box_len[0];
  g.Ly = box_len[1];
  g.Lz = box_len[2];
  g.lox = box_lo[0];
  g.loy = box_lo[1];
  g.nx = ncell(g.Lx, periodic[0]);
  g.ny = ncell(g.Ly, periodic[1]);
  g.cx = g.Lx / g.nx;
  g.cy = g.Ly / g.ny;
  g.wrapx = periodic[0] && g.nx >= 3;
  g.wrapy = periodic[1] && g.ny >= 3;
  g.perz_box = periodic[2];
  if (periodic[2] && n_zones == 1) {  // the zone IS the box: periodic z grid
    g.loz = box_lo[2];
    g.nz = ncell(g.Lz, true);
    g.cz = g.Lz / g.nz;
    g.wrapz = g.nz >= 3;
    g.zc = box_lo[2] + 0.5 * g.Lz;
  } else {                            // slab grid, padded by rcut, no wrap
    const double width = g.Lz / n_zones;
    const double range = width + 2.0 * rcut;
    g.loz = box_lo[2] + zone_id * width - rcut;
    g.nz = int(range / rcut);
    if (g.nz < 1) g.nz = 1;
    g.cz = range / g.nz;
    g.wrapz = false;
    g.zc = g.loz + 0.5 * range;
  }
  return g;
}

// --- build kernels (per zone, per pass) ------------------------------------

static __global__ void cell_count_kernel(const double* x, const double* y,
                                         const double* z, int n, CellGrid g,
                                         int* cell_of, int* counts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int ix, iy, iz;
  g.coords(x[i], y[i], z[i], ix, iy, iz);
  const int c = g.idx(ix, iy, iz);
  cell_of[i] = c;
  atomicAdd(&counts[c], 1);
}

static __global__ void cell_scatter_kernel(const int* cell_of, int n,
                                           int* cursor, int* order) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  order[atomicAdd(&cursor[cell_of[i]], 1)] = i;
}

// --- culled force kernel ----------------------------------------------------
//
// Same contract and epilogue as zone_pair_kernel (zone_force.cuh): owner
// thread per zone-A atom, contributions quantized per pair into integer
// per-thread partials, energy counted once per pair (internal: lower
// ORIGINAL index; cross: A-side launch only), min_r2 before the cut, sticky
// overflow flag. Candidates come from the 3x3x3 (or degenerate) cell
// neighborhood of B's grid instead of all of B.
template <typename PairF>
static __global__ void zone_pair_cells_kernel(
    ZoneForceArgs a, core::PairGeom geom, PairF pot, CellGrid bg,
    const int* b_starts, const int* b_counts, const int* b_order) {
  __shared__ long long sred[kZoneBlock];
  const int i = blockIdx.x * kZoneBlock + threadIdx.x;
  const bool active = i < a.na;
  const double xi = active ? a.ax[i] : 0.0;
  const double yi = active ? a.ay[i] : 0.0;
  const double zi = active ? a.az[i] : 0.0;
  long long qfx = 0, qfy = 0, qfz = 0, qpe = 0;
  double mr2 = 1e300;

  if (active) {
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
            if (a.same_zone && j == i) continue;
            double ddx = xi - a.bx[j];
            double ddy = yi - a.by[j];
            double ddz = zi - a.bz[j];
            double r2;
            const bool ok = geom.reduce(ddx, ddy, ddz, r2);
            mr2 = fmin(mr2, r2);
            if (!ok) continue;
            double u, f_over_r;
            pot(sqrt(r2), u, f_over_r);
            qfx += quantize(f_over_r * ddx, core::fixed::ForceAccum::kScale,
                            a.overflow);
            qfy += quantize(f_over_r * ddy, core::fixed::ForceAccum::kScale,
                            a.overflow);
            qfz += quantize(f_over_r * ddz, core::fixed::ForceAccum::kScale,
                            a.overflow);
            if (a.count_energy && (!a.same_zone || i < j))
              qpe += quantize(u, core::fixed::EnergyAccum::kScale, a.overflow);
          }
        }
      }
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
  for (int s = kZoneBlock / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicAdd(reinterpret_cast<unsigned long long*>(a.pe),
              (unsigned long long)sred[0]);
}

}  // namespace tdmd::cuda
