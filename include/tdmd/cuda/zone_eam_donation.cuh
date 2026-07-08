#pragma once
// PR-3a (W-contract ladder — device-donation substrate; design of record:
// docs/_meta/PR3AB_GPU_DONATION_DESIGN_2026-07-07.md §5) — the CROSS donation
// kernel pair, the ONE new kernel pair of PR-3a/b. Everything else is verbatim
// reuse: the SELF batch runs eam_density_cells_kernel / eam_density_kernel
// (zone_eam.cuh / zone_eam_cells.cuh — BYTE-UNTOUCHED) over a zone's slab.
//
// Any .cu including this MUST link `tdmd_eam_cuda_flags` (--fmad=false) — the
// Horner steps fuse to fma otherwise and break the 1-ULP bitwise contract
// (rint ≡ std::rint ≡ cvt.rni; the E5 proof).
//
// CROSS BATCH SEMANTICS (bitwise vs the CPU executor eam_donate_cross, by
// W-1 + B1): the host concatenates two zone slabs [A|B] (na = |A|, m = na+nb);
// one thread per concat atom aa GATHERS over candidates restricted to the
// OTHER side and adds the quantized ρ_a(r) into its OWN zone-lane element.
// The CPU executor enumerates each unordered cross pair once and adds the SAME
// v into both ends; the gather form computes v from each side independently —
// bitwise-equal by lemma W-1 (dx negation is exact IEEE, min-image round is
// odd, operand order identical ⇒ r2 bitwise-equal from either direction ⇒
// sqrt/eval_rhoa/rint identical) ⇒ the per-lane-element quanta MULTISET is
// identical, and int64 addition is associative (B1) ⇒ lane sums are raw-equal
// for ANY enumeration/culling order. Culling (the ±s_d stencil walk, copied
// VERBATIM from eam_density_cells_kernel) only restricts CANDIDATES over a
// superset stencil (L-SUP) + the exact geom.reduce retest ⇒ the accepted-pair
// multiset is unchanged.
//
// Lane += (read-modify-write) is race-free: each lane element is owned by
// exactly ONE thread per launch, and launches are null-stream-ordered after
// the SELF kernel of both zones (the ring schedule guarantees CoRes(A,B) ⇒
// both selfs already executed; the stamp fence enforces it).
//
// NO min_r2, NO keys, NO PE in the signature — donations move ρ ONLY
// (WContract §6; the force/PE/min_r2/φ-once pass stays C-phase verbatim).
#include <cuda_runtime.h>

#include "tdmd/cuda/zone_cells.cuh"  // CellGrid (stencil geometry)
#include "tdmd/cuda/zone_eam.cuh"    // EamSetflView, quantize, kZoneBlock

namespace tdmd::cuda {

// CULLED cross batch: body = eam_density_cells_kernel's stencil walk VERBATIM
// with exactly two mechanical deltas: (i) the same-side skip
// `(bb < na) == (aa < na)` replaces the `bb == aa` self-skip (zones are
// disjoint ⇒ no self pair can occur); (ii) the epilogue routes the int64 sum
// into the OWN zone's lane element (+=, not overwrite — the lane already holds
// the SELF contribution).
__global__ void eam_donate_cross_cells_kernel(
    const double* cx, const double* cy, const double* cz, int na, int m,
    core::PairGeom geom, EamSetflView eam, double dens_scale, CellGrid g,
    const int* b_starts, const int* b_counts, const int* b_order,
    long long* rho_a, long long* rho_b, int* overflow) {
  const int aa = blockIdx.x * kZoneBlock + threadIdx.x;
  if (aa >= m) return;
  long long q = 0;
  const double xi = cx[aa], yi = cy[aa], zi = cz[aa];
  int cxi, cyi, czi;
  g.coords(xi, yi, zi, cxi, cyi, czi);
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
          if ((bb < na) == (aa < na)) continue;  // same-side skip (delta i)
          double ddx = xi - cx[bb], ddy = yi - cy[bb], ddz = zi - cz[bb], r2;
          if (!geom.reduce(ddx, ddy, ddz, r2)) continue;
          double v, dv;
          eam.eval_rhoa(sqrt(r2), v, dv);
          q += quantize(v, dens_scale, overflow);
        }
      }
    }
  }
  if (aa < na) rho_a[aa] += q;      // delta (ii): own-lane += (holds self already)
  else rho_b[aa - na] += q;
}

// PLAIN (cull=false leg) cross batch: the inner loop is over ALL m concat
// atoms — the O(m²) reference, mirroring eam_density_kernel's shape.
__global__ void eam_donate_cross_kernel(const double* cx, const double* cy,
                                        const double* cz, int na, int m,
                                        core::PairGeom geom, EamSetflView eam,
                                        double dens_scale, long long* rho_a,
                                        long long* rho_b, int* overflow) {
  const int aa = blockIdx.x * kZoneBlock + threadIdx.x;
  if (aa >= m) return;
  long long q = 0;
  const double xi = cx[aa], yi = cy[aa], zi = cz[aa];
  for (int bb = 0; bb < m; ++bb) {
    if ((bb < na) == (aa < na)) continue;  // same-side skip
    double ddx = xi - cx[bb], ddy = yi - cy[bb], ddz = zi - cz[bb], r2;
    if (!geom.reduce(ddx, ddy, ddz, r2)) continue;
    double v, dv;
    eam.eval_rhoa(sqrt(r2), v, dv);
    q += quantize(v, dens_scale, overflow);
  }
  if (aa < na) rho_a[aa] += q;
  else rho_b[aa - na] += q;
}

}  // namespace tdmd::cuda
