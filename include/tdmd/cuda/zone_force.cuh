#pragma once
// M4 — GPU zone-force kernels with B1 fixed-point accumulation (CUDA-only
// header: include from .cu translation units).
//
// Bitwise contract (INV-9, deterministic_fp64). The CPU zone path quantizes
// EVERY pair contribution to int64 (Q24.40 forces / Q34.30 energy, rint =
// round-to-nearest-even) and integer-sums — order-free. The kernel reproduces
// the exact same multiset of quantized integers per accumulator:
//   * full-neighbour evaluation: the thread owning atom i computes the pair
//     (i,j) from dx = xi - xj; the j-owner computes -dx. IEEE negation is
//     exact and rint is symmetric, so q(j-side) == -q(i-side) — bitwise the
//     same ±q pair the CPU's half-list Newton-3 writes.
//   * per-pair quantization, per-thread INTEGER partial sums, one += per
//     atom-component at flush — integer addition is associative, so tile
//     order, block order, grid shape and stream count cannot change the sum.
//   * energy: counted once per pair (internal: owner with the lower index;
//     cross: the A-side launch only), quantized per pair like the CPU.
// Therefore, GPU raw int64 accumulators == CPU raw accumulators bit-for-bit
// PROVIDED the per-pair FP64 contributions are identical doubles. That holds
// when (a) the TU is compiled with --fmad=false (the CPU is built with
// -ffp-contract=off) and (b) the pair functor uses no transcendentals whose
// CPU/GPU implementations differ in ulps: LJ qualifies (+,*,/,sqrt — all
// correctly rounded), Morse does NOT (exp differs between libm and CUDA), so
// Morse is cross-checked by tolerance and by GPU-internal bitwise tests.
// PairGeom::reduce and CutoffScheme::apply are the SAME host+device code the
// CPU path runs (single source — zones.hpp / cutoff.hpp).
//
// Overflow (B1): the device cannot throw like FixedAccum::add — out-of-range
// contributions set a sticky flag (atomicOr) and contribute 0; the host must
// treat a set flag as Halt::Internal (ZoneFSM §9). NaN forces also trip it
// (the range predicate is written NaN-false).
#include <cuda_runtime.h>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/potentials/cutoff.hpp"
#include "tdmd/potentials/pair_lj.hpp"
#include "tdmd/potentials/pair_morse.hpp"

namespace tdmd::cuda {

inline constexpr int kZoneBlock = 128;

// Quantize one FP64 contribution — textually mirrors FixedAccum::add
// (std::rint under FE_TONEAREST == device rint == cvt.rni); the in-range
// double->ll conversion of an integer-valued double is exact.
__device__ inline long long quantize(double x, double scale, int* overflow) {
  const double q = rint(x * scale);
  if (!(fabs(q) < 9.2e18)) {
    atomicOr(overflow, 1);
    return 0;
  }
  return (long long)(q);
}

// Order-preserving bit map for non-negative doubles (r2 >= 0): bit patterns
// compare like the values, so atomicMin over bits == min over doubles.
__device__ inline unsigned long long pos_double_bits(double v) {
  return __double_as_longlong(v);
}

struct ZoneForceArgs {
  const double* ax;  // zone A positions (the OWNER side: forces land here)
  const double* ay;
  const double* az;
  int na;
  const double* bx;  // zone B positions (tile side; may alias A)
  const double* by;
  const double* bz;
  int nb;
  long long* fx;     // zone A raw Q24.40 accumulators, += by the owner thread
  long long* fy;
  long long* fz;
  long long* pe;                  // pass energy, raw Q34.30 (atomicAdd)
  unsigned long long* min_r2;     // pos_double_bits, atomicMin (B10 overlap)
  int* overflow;                  // sticky B1 range flag
  bool same_zone;                 // B aliases A: skip j==i, energy via i<j
  bool count_energy;              // internal pass or the A-side cross launch
};

// One launch = contributions to zone A from pairs (A x B) [or internal when
// same_zone]. Owner thread per A atom, B staged in shared tiles. Plain += on
// the A accumulators: exactly one owner thread per atom per launch, and the
// internal/cross launches of one zone are stream-ordered.
template <typename PairF>
__global__ void zone_pair_kernel(ZoneForceArgs a, core::PairGeom geom,
                                 PairF pot) {
  __shared__ double sx[kZoneBlock], sy[kZoneBlock], sz[kZoneBlock];
  __shared__ long long sred[kZoneBlock];
  const int i = blockIdx.x * kZoneBlock + threadIdx.x;
  const bool active = i < a.na;
  const double xi = active ? a.ax[i] : 0.0;
  const double yi = active ? a.ay[i] : 0.0;
  const double zi = active ? a.az[i] : 0.0;
  long long qfx = 0, qfy = 0, qfz = 0, qpe = 0;
  double mr2 = 1e300;

  for (int base = 0; base < a.nb; base += kZoneBlock) {
    const int t = base + threadIdx.x;
    if (t < a.nb) {
      sx[threadIdx.x] = a.bx[t];
      sy[threadIdx.x] = a.by[t];
      sz[threadIdx.x] = a.bz[t];
    }
    __syncthreads();
    const int tile = min(kZoneBlock, a.nb - base);
    if (active) {
      for (int k = 0; k < tile; ++k) {
        const int j = base + k;
        if (a.same_zone && j == i) continue;
        double dx = xi - sx[k];
        double dy = yi - sy[k];
        double dz = zi - sz[k];
        double r2;
        const bool ok = geom.reduce(dx, dy, dz, r2);
        mr2 = fmin(mr2, r2);  // before the cut — coincident pairs must HALT
        if (!ok) continue;
        double u, f_over_r;
        pot(sqrt(r2), u, f_over_r);
        qfx += quantize(f_over_r * dx, core::fixed::ForceAccum::kScale,
                        a.overflow);
        qfy += quantize(f_over_r * dy, core::fixed::ForceAccum::kScale,
                        a.overflow);
        qfz += quantize(f_over_r * dz, core::fixed::ForceAccum::kScale,
                        a.overflow);
        if (a.count_energy && (!a.same_zone || i < j))
          qpe += quantize(u, core::fixed::EnergyAccum::kScale, a.overflow);
      }
    }
    __syncthreads();
  }

  if (active) {
    a.fx[i] += qfx;
    a.fy[i] += qfy;
    a.fz[i] += qfz;
    atomicMin(a.min_r2, pos_double_bits(mr2));
  }
  // pass-energy block reduction (integer tree -> one atomicAdd per block)
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

// Device-ready pair functors over the SAME single-source math the CPU
// drivers instantiate (pair_lj.hpp / pair_morse.hpp + cutoff.hpp). Built on
// the host (CutoffScheme::make), passed to kernels by value (POD copies).
struct LJDev {
  potentials::LJParams<double> prm;
  potentials::CutoffScheme cs;
  TDMD_HOST_DEVICE void operator()(double r, double& u,
                                   double& f_over_r) const {
    potentials::pair_lj(r, prm, u, f_over_r);
    cs.apply(r, u, f_over_r);
  }
};

struct MorseDev {
  potentials::MorseParams<double> prm;
  potentials::CutoffScheme cs;
  TDMD_HOST_DEVICE void operator()(double r, double& u,
                                   double& f_over_r) const {
    potentials::pair_morse(r, prm, u, f_over_r);
    cs.apply(r, u, f_over_r);
  }
};

}  // namespace tdmd::cuda
