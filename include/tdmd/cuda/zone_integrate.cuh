#pragma once
// M4 — velocity-Verlet halves and zone-local reductions on the GPU (CUDA-only
// header). Bitwise contract with the CPU conveyor (conveyor.hpp):
//
//   * drift / kick are per-atom FP64 with the EXACT expressions of
//     integrator.hpp / conveyor.hpp (one mul-add chain per component) —
//     bitwise CPU<->GPU under --fmad=false;
//   * force conversion from raw Q24.40 is double(raw)/kScale — the same
//     single expression FixedAccum::value() uses;
//   * v_max²/a_max² (max) and k2cap (min) reductions use the SHARED per-atom
//     kernels of buffer.hpp (host+device) — max/min are exact, so the zone
//     locals feeding INV-4 and the Λ-chain are bitwise CPU<->GPU;
//   * KE is accumulated in fixed point (Q34.30) — deterministic and
//     stream-count-independent on the GPU. The CPU conveyor sums KE in FP64
//     member order, so GPU-vs-CPU KE agrees only to the quantization scale
//     (~1e-9 eV) — diagnostics-only (PassStats), never trajectory data.
//
// Reduction layout: block-level integer/bit tricks — atomicMax/atomicMin on
// the ordered bit patterns of non-negative doubles, atomicAdd(ull) for the
// fixed-point KE. All order-free => bitwise run-to-run and across stream
// counts (INV-9).
#include <cuda_runtime.h>

#include "tdmd/core/buffer.hpp"
#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/units.hpp"

namespace tdmd::cuda {

inline constexpr int kIntBlock = 128;

// velocity-Verlet first half (drift to x_h with dt) — conveyor ensure_drift.
static __global__ void zone_drift_kernel(double* x, double* y, double* z,
                                         double* vx, double* vy, double* vz,
                                         const double* fx, const double* fy,
                                         const double* fz, const double* mass,
                                         int n, double dt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double inv_m = units::ftm2v / mass[i];
  vx[i] += 0.5 * dt * inv_m * fx[i];
  vy[i] += 0.5 * dt * inv_m * fy[i];
  vz[i] += 0.5 * dt * inv_m * fz[i];
  x[i] += dt * vx[i];
  y[i] += dt * vy[i];
  z[i] += dt * vz[i];
}

// Zone END (conveyor end_zone, T4): convert raw force accumulators to FP64,
// second-half kick, zone-local reductions. Outputs (device scalars):
//   v2max, a2max — ordered double bits, atomicMax;
//   k2cap        — ordered double bits, atomicMin;
//   ke_raw       — Q34.30 fixed point, atomicAdd.
static __global__ void zone_end_kernel(
    double* vx, double* vy, double* vz, double* fx, double* fy, double* fz,
    const long long* rfx, const long long* rfy, const long long* rfz,
    const double* mass, int n, double dt, double K2,
    unsigned long long* v2max_bits, unsigned long long* a2max_bits,
    unsigned long long* k2cap_bits, long long* ke_raw, int* overflow) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // FixedAccum::value() — the identical conversion expression
  fx[i] = double(rfx[i]) / core::fixed::ForceAccum::kScale;
  fy[i] = double(rfy[i]) / core::fixed::ForceAccum::kScale;
  fz[i] = double(rfz[i]) / core::fixed::ForceAccum::kScale;
  const double inv_m = units::ftm2v / mass[i];
  vx[i] += 0.5 * dt * inv_m * fx[i];
  vy[i] += 0.5 * dt * inv_m * fy[i];
  vz[i] += 0.5 * dt * inv_m * fz[i];
  const double vi2 = core::buffer::speed2(vx[i], vy[i], vz[i]);
  const double ai2 = core::buffer::accel2(fx[i], fy[i], fz[i], mass[i]);
  const double kc = core::buffer::k2_limited_dt_atom(
      fx[i], fy[i], fz[i], vx[i], vy[i], vz[i], mass[i], K2);
  atomicMax(v2max_bits, __double_as_longlong(vi2));
  atomicMax(a2max_bits, __double_as_longlong(ai2));
  atomicMin(k2cap_bits, __double_as_longlong(kc));
  // KE quantized per atom (0.5·mvv2e·m·v² is one FP64 expression — identical
  // on CPU; the SUM differs from the CPU's FP64 ordered sum by quantization)
  const double kei = 0.5 * units::mvv2e * mass[i] * vi2;
  const double q = rint(kei * core::fixed::EnergyAccum::kScale);
  if (!(fabs(q) < 9.2e18)) {
    atomicOr(overflow, 1);
  } else {
    atomicAdd(reinterpret_cast<unsigned long long*>(ke_raw),
              (unsigned long long)(long long)(q));
  }
}

}  // namespace tdmd::cuda
