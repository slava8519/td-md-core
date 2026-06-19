#pragma once
#include <cuda_runtime.h>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/zones.hpp"        // PairGeom (HOST_DEVICE reduce)
#include "tdmd/cuda/zone_force.cuh"   // quantize, pos_double_bits, kZoneBlock
#include "tdmd/potentials/sw.hpp"     // SwParams, sw_phi2/sw_g/sw_triplet (HOST_DEVICE)

// M6 / SW-ladder T5 — GPU Stillinger–Weber force: the φ₃ TRANSPOSE accumulator on the
// device (force to a third atom k, q(k)≠−q(i)) the symmetric EAM GPU int64 accumulator
// cannot run. Design of record: wf_f13d5221-b7e. [ENG].
//
// HONEST CONTRACT (the INVERSE of EAM's transcendental-free bitwise spline):
//   CPU↔GPU is TOLERANCE (~1e-9 force / ~1e-6 PE), NOT bitwise — sw_phi2/sw_g use exp/pow,
//   which differ libm-vs-CUDA by ~1 ulp (the Morse situation, zone_force.cuh banner; the
//   inverse of eam_window_force_gpu.cuh's bitwise headline). GPU-INTERNAL determinism
//   (run-to-run, 1-vs-z) IS bitwise SOLELY by B1 int64 register order-freedom (write-once,
//   no cross-atom force write, no force atomics). sw_triplet's wing-canonical operand order
//   is INHERITED/DEFENSIVE here, NOT load-bearing on the GPU: each owned atom always sits in
//   its OWN j-slot, so no two GPU threads ever compute the same triplet with swapped j/k
//   slots (it IS load-bearing on the CPU relabeling path, TripletWingSymmetryIsBitwise).
//   --fmad=false is STILL REQUIRED — NOT for CPU↔GPU bitwise (impossible here) but to keep
//   GPU-internal determinism STRUCTURAL (no codegen-varying FMA fusion). Do NOT add a
//   CPU↔GPU-bitwise gate; the independent FP64 oracle (sw_direct_fp64, O(N²)) <1e-9 is the
//   only MB2 witness. Do NOT use --use_fast_math / --ftz / --prec-div=false.
namespace tdmd::cuda {

// per-thread cap on o's OWN in-rcut neighbour list (the φ₃-center pair loop + wing driver;
// NOT the window size m). Diamond-Si: only the ~4 first-shell atoms (2.35 Å) are within
// rcut=a·σ≈3.77 Å — the 12 second-shell atoms (3.84 Å) sit just OUTSIDE; ≤~16 under thermal
// perturbation ⇒ 64 is ~4× headroom. A cap OVERFLOW is a guarded HALT (sticky bit 4, thrown
// by compute() BEFORE any writeback), never a silent truncation (the dropped-donor class).
inline constexpr int kMaxNbr = 64;

// One thread per OWNED atom o: the whole transpose-replay over the gathered window into
// int64 REGISTERS, write-once (no cross-atom write, no force atomics — only PE/n_triplets
// cross threads, via a block tree-reduce + atomicAdd, like eam_force_kernel).
__global__ void sw_force_kernel(
    const double* wx, const double* wy, const double* wz, const long* key,
    int m, const int* owned, int n_owned, core::PairGeom geom, potentials::SwParams sp,
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

    // o's own neighbour list, cached once (local buffer) — read by φ₂, the φ₃-center
    // pair loop, and to drive the φ₃-wing role. dx = xo − wx[b] (bond FROM o).
    int nb[kMaxNbr];
    double nbx[kMaxNbr], nby[kMaxNbr], nbz[kMaxNbr], nbr_[kMaxNbr];
    int cnt = 0;
    for (int b = 0; b < m; ++b) {
      if (b == o) continue;
      double dx = xo - wx[b], dy = yo - wy[b], dz = zo - wz[b], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
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
        atomicOr(overflow, 4);            // STICKY: neighbour cap exceeded → HALT
      }
    }

    // (ii) φ₃ CENTER: every unordered wing pair (x<y); o owns f_i + E3 + count ONCE.
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

    // (iii) φ₃ WING: for each neighbour i of o, replay every triplet {i; o, m'} (m' in i's
    // FULL window nbr list, brute-force re-scan), o in the j-slot. NO energy, NO count.
    // THE NON-SYMMETRIC WRITE: o gets f_j of a triplet centered at ANOTHER atom i.
    for (int e = 0; e < cnt; ++e) {
      const int i = nb[e];
      const double iox = -nbx[e], ioy = -nby[e], ioz = -nbz[e], rio = nbr_[e];  // x_i − x_o
      const double xi = wx[i], yi = wy[i], zi = wz[i];
      for (int b = 0; b < m; ++b) {
        if (b == i || b == o) continue;   // m' != o AND m' != i
        double mdx = xi - wx[b], mdy = yi - wy[b], mdz = zi - wz[b], r2m;
        if (!geom.reduce(mdx, mdy, mdz, r2m)) continue;
        potentials::SwVec3 fi, fj, fk;
        potentials::sw_triplet(sp, iox, ioy, ioz, rio, mdx, mdy, mdz, sqrt(r2m), fi, fj, fk);
        qfx += quantize(fj.x, fscale, overflow);
        qfy += quantize(fj.y, fscale, overflow);
        qfz += quantize(fj.z, fscale, overflow);
      }
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

}  // namespace tdmd::cuda
