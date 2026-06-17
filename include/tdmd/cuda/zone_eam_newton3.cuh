#pragma once
// M6 E5c (bake-off, HALF-LIST + NEWTON-3 piece) — measures how much of the
// ~2.45× GPU-EAM gap vs LAMMPS is the FULL-NEIGHBOUR determinism choice. A third
// culled variant of zone_eam.cuh's density + force sweeps that walks each
// undirected in-list pair ONCE (bb>aa) and Newton-3 shares the contribution to
// BOTH partners via int64 atomicAdd. The eval is halved (each pair's spline math
// runs once, not twice); the cost is cross-thread atomics instead of a private
// register. CUDA-only header; --fmad=false MANDATORY (link tdmd_eam_cuda_flags —
// the f_over_r prefactor and the Horner steps fuse to fma otherwise and break
// 1 ULP). zone_eam.cuh / zone_eam_cells.cuh / zone_eam_verlet.cuh / zone_cells.cuh
// are BYTE-UNTOUCHED — this header only ADDS new kernels and reuses theirs (the
// EamVerletList CSR build, EamSetflView, quantize) verbatim.
//
// BITWISE ≡ FULL-NEIGHBOUR (the load-bearing claim — this is NOT a determinism
// tradeoff; Test A'' in test_cuda_eam_cells.cu is the gate). The half-list raw
// int64 ρ / fx / fy / fz / pe are bit-for-bit equal to the full-neighbour
// all-window kernels, by:
//   1. int64 atomicAdd is ASSOCIATIVE ⇒ d_fx[aa] is the same int64 regardless of
//      thread order (B1/INV-9). atomicAdd is the ONLY writer of d_fx/fy/fz/d_rho
//      here (no private register), so the per-atom int64 is order-free.
//   2. quantize() = rint(x·scale); rint is ODD ⇒ quantize(-x) = -quantize(x). For
//      pair (aa,bb) the full-neighbour term aa accumulates is q=quantize(f_over_r·
//      dx_ab); bb accumulates quantize(f_over_r·dx_ba)=quantize(f_over_r·(−dx_ab))
//      =−q. So atomicAdd(f_aa,+q)+atomicAdd(f_bb,−q) reproduces EXACTLY the
//      full-neighbour per-atom multiset (the same ±q the all-window kernel sums
//      into its two private registers). f_over_r is symmetric in (aa,bb): it
//      depends on r, fp[aa]+fp[bb] (commutative), dra(r) — identical from either
//      endpoint, so the SAME q is shared (no 1-ULP asymmetry).
//   3. ρ_a(r) is symmetric ⇒ pair (aa,bb) adds the SAME q=quantize(ρ_a(r)) to
//      d_rho[aa] and d_rho[bb] — same multiset as the full-neighbour density
//      (which sums ρ_a(r) once into aa's register and once into bb's).
//   4. φ-once PE: the full-neighbour kernel processes each undirected pair from
//      BOTH owners and counts φ on the lower-key side (`ki<key[bb]`) ⇒ exactly
//      ONCE per undirected pair total (keys distinct). The half-list processes
//      each undirected pair ONCE (bb>aa) and counts φ unconditionally ⇒ also once
//      per pair. The two φ COUNTS are identical pair-for-pair, KEY-INDEPENDENT
//      (no reliance on key being monotonic in index). Embedding energy F(ρ_aa) is
//      per-ATOM (not per-pair) ⇒ added once per OWNED atom in a separate pass.
//
// CANDIDATE-SET TRANSPARENCY (the verlet argument): the list is built within
// (rcut+skin), a SUPERSET of the in-cutoff set; every kernel re-applies the EXACT
// acceptance predicate (PairGeom::reduce, r2<rcut²). The bb>aa gate removes the
// duplicate direction (the FULL EamVerletList has both (aa,bb) and (bb,aa)) so
// each undirected in-cutoff pair is hit ONCE; Newton-3 restores the symmetric
// contribution to both partners. So the in-cutoff MULTISET per atom is identical
// to the O(m²) sweep ⇒ raw int64 bit-for-bit equal.
//
// min_r2: NOT computed here as a bitwise-comparable quantity (the half traversal
// examines a different candidate multiset than the full kernels' per-owned walk).
// The Overlap-HALT boolean still holds (any pair under rcut is in the half list
// once), but min_r2 must never be compared bit-for-bit across paths — the test
// only checks it is >0 / finite.
//
// FULL EamVerletList REUSED: gating bb>aa on the full list makes it a half
// traversal — each undirected pair is visited exactly once (the (aa,bb) entry in
// aa's list with bb>aa fires; the (bb,aa) entry in bb's list with aa<bb is gated
// out). A genuine lower-triangle CSR would halve the WALK too, but the EVAL is the
// dominant cost and the gate already halves it; reusing the full list keeps this
// header strictly additive (no new build). d_fx/fy/fz/d_rho MUST be zeroed before
// the atomicAdd passes (atomicAdd accumulates onto whatever is there).
#include <cuda_runtime.h>

#include "tdmd/cuda/zone_eam.cuh"         // EamSetflView, quantize, pos_double_bits, kZoneBlock
#include "tdmd/cuda/zone_eam_verlet.cuh"  // EamVerletList CSR build (full list) — reused verbatim

namespace tdmd::cuda {

// pass 1 (HALF-LIST + NEWTON-3): ρ. Each thread owns atom aa and walks aa's FULL
// verlet list, but only acts on bb>aa (each undirected in-list pair once). For an
// accepted pair: q=quantize(ρ_a(r), dens_scale); atomicAdd(d_rho[aa],q) AND
// atomicAdd(d_rho[bb],q). Because ρ_a is symmetric the SAME q lands on both — the
// exact ±0 (here +,+) pair the full-neighbour density sums into aa's and bb's
// private registers. d_rho MUST be pre-zeroed (atomicAdd accumulates).
__global__ void eam_density_n3_kernel(const double* wx, const double* wy,
                                      const double* wz, int m, core::PairGeom geom,
                                      EamSetflView eam, double dens_scale,
                                      const int* d_off, const int* d_idx,
                                      long long* d_rho, int* overflow) {
  const int aa = blockIdx.x * kZoneBlock + threadIdx.x;
  if (aa >= m) return;
  const double xi = wx[aa], yi = wy[aa], zi = wz[aa];
  const int beg = d_off[aa], end = d_off[aa + 1];
  for (int t = beg; t < end; ++t) {
    const int bb = d_idx[t];
    if (bb <= aa) continue;  // half traversal: each undirected pair once (bb>aa)
    double ddx = xi - wx[bb], ddy = yi - wy[bb], ddz = zi - wz[bb], r2;
    if (!geom.reduce(ddx, ddy, ddz, r2)) continue;  // EXACT rcut re-test
    double v, dv;
    eam.eval_rhoa(sqrt(r2), v, dv);
    const long long q = quantize(v, dens_scale, overflow);  // symmetric ⇒ same q
    atomicAdd(reinterpret_cast<unsigned long long*>(&d_rho[aa]),
              static_cast<unsigned long long>(q));
    atomicAdd(reinterpret_cast<unsigned long long*>(&d_rho[bb]),
              static_cast<unsigned long long>(q));
  }
}

// embedding-energy-only per-atom kernel (HALF-LIST companion). The full-neighbour
// force kernel folds the embedding energy F(ρ_ii) into pass 3 (qpe += quantize(Fi)
// once per OWNED atom). The half-list force kernel below is pair-driven (it cannot
// cheaply emit a once-per-owned term without re-deriving ownership), so the
// embedding ENERGY is emitted here, once per OWNED atom, into d_pe — bit-for-bit
// the same quantize(Fi, EnergyAccum::kScale) the full-neighbour kernel adds. (The
// embedding DERIVATIVE F'(ρ) for the force still comes from eam_embedding_kernel,
// which writes d_fp — unchanged, reused as-is.)
__global__ void eam_embed_energy_n3_kernel(int m, const int* owned, int n_owned,
                                           EamSetflView eam, double dens_scale,
                                           const long long* d_rho, long long* d_pe,
                                           int* overflow) {
  __shared__ long long sred[kZoneBlock];
  const int o = blockIdx.x * kZoneBlock + threadIdx.x;
  long long qpe = 0;
  if (o < n_owned) {
    const int ii = owned[o];
    double Fi, Fpi;
    eam.eval_F(double(d_rho[ii]) / dens_scale, Fi, Fpi);  // embedding energy once per owned
    qpe = quantize(Fi, core::fixed::EnergyAccum::kScale, overflow);
  }
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

// pass 3 (HALF-LIST + NEWTON-3): pair force + φ-once PE. Each thread owns atom aa,
// walks aa's FULL verlet list, acts on bb>aa once. For an accepted pair:
//   f_over_r = -(dphi + (fp[aa]+fp[bb])*dra)/r  (the SAME line, --fmad=false)
//   q_fx = quantize(f_over_r·dx_ab, ForceAccum::kScale)  (dx_ab = x_aa - x_bb)
//   atomicAdd(d_fx[aa], +q_fx);  atomicAdd(d_fx[bb], -q_fx)   (Newton-3; rint odd)
//   φ once: atomicAdd(d_pe, quantize(phi))                    (bb>aa ⇒ once)
// f_over_r is symmetric in (aa,bb) [r, fp[aa]+fp[bb] commute, dra(r)] so the SAME
// q_fx is shared; rint-odd makes the bb-side contribution exactly -q_fx — the
// ±q_fx pair the full-neighbour kernels sum into their two registers. The
// EMBEDDING energy F(ρ) is NOT here (it is per-atom — eam_embed_energy_n3_kernel).
// d_fx/fy/fz MUST be pre-zeroed (atomicAdd accumulates). The phi gate is bb>aa
// (each undirected pair once, counted unconditionally) — key-independent, matches
// the full-neighbour total of one phi per pair (header note 4); no key/owned here.
__global__ void eam_force_n3_kernel(const double* wx, const double* wy,
                                    const double* wz, int m, core::PairGeom geom,
                                    EamSetflView eam, const double* d_fp,
                                    const int* d_off, const int* d_idx,
                                    long long* d_fx, long long* d_fy,
                                    long long* d_fz, long long* d_pe,
                                    unsigned long long* d_min_r2, int* overflow) {
  __shared__ long long sred[kZoneBlock];
  const int aa = blockIdx.x * kZoneBlock + threadIdx.x;
  long long qpe = 0;
  double mr2 = 1e300;
  if (aa < m) {
    const double xi = wx[aa], yi = wy[aa], zi = wz[aa];
    const double fpi = d_fp[aa];
    const int beg = d_off[aa], end = d_off[aa + 1];
    for (int t = beg; t < end; ++t) {
      const int bb = d_idx[t];
      if (bb <= aa) continue;  // half traversal: each undirected pair once (bb>aa)
      double ddx = xi - wx[bb], ddy = yi - wy[bb], ddz = zi - wz[bb], r2;
      const bool ok = geom.reduce(ddx, ddy, ddz, r2);
      mr2 = fmin(mr2, r2);  // BEFORE the cutoff — overlap probe (NOT bitwise-comparable)
      if (!ok) continue;
      const double r = sqrt(r2);
      double phi, dphi, ra, dra;
      eam.eval_phi(r, phi, dphi);
      eam.eval_rhoa(r, ra, dra);
      const double f_over_r = -(dphi + (fpi + d_fp[bb]) * dra) / r;  // FMA-killer site
      const long long qx = quantize(f_over_r * ddx, core::fixed::ForceAccum::kScale, overflow);
      const long long qy = quantize(f_over_r * ddy, core::fixed::ForceAccum::kScale, overflow);
      const long long qz = quantize(f_over_r * ddz, core::fixed::ForceAccum::kScale, overflow);
      // Newton-3: aa gets +q, bb gets -q (== quantize(f_over_r·dx_ba), rint odd).
      atomicAdd(reinterpret_cast<unsigned long long*>(&d_fx[aa]), static_cast<unsigned long long>(qx));
      atomicAdd(reinterpret_cast<unsigned long long*>(&d_fy[aa]), static_cast<unsigned long long>(qy));
      atomicAdd(reinterpret_cast<unsigned long long*>(&d_fz[aa]), static_cast<unsigned long long>(qz));
      atomicAdd(reinterpret_cast<unsigned long long*>(&d_fx[bb]), static_cast<unsigned long long>(-qx));
      atomicAdd(reinterpret_cast<unsigned long long*>(&d_fy[bb]), static_cast<unsigned long long>(-qy));
      atomicAdd(reinterpret_cast<unsigned long long*>(&d_fz[bb]), static_cast<unsigned long long>(-qz));
      qpe += quantize(phi, core::fixed::EnergyAccum::kScale, overflow);  // φ once (bb>aa)
    }
    atomicMin(d_min_r2, pos_double_bits(mr2));
  }
  // per-block int64 tree-reduce of φ → one atomicAdd (zone_eam.cuh epilogue)
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

}  // namespace tdmd::cuda
