#pragma once
#include <cassert>
#include <cmath>
#include <cstdint>

// B1 / INV-9 — fixed-point force accumulation (M3.5 first PR).
// Numerically ≡ AMBER SPFP (Le Grand 2013; MIXED_PRECISION doc §2, ZoneFSM
// §6/INV-9): contributions are quantized once to int64 and summed in integer
// arithmetic. Integer addition is ASSOCIATIVE, so the sum is bit-identical
// for ANY accumulation order — zone passes, threads, GPU blocks. This is the
// project's bitwise-determinism mechanism for cross-pass force assembly;
// hardware atomicAdd(double) has an unfixed order and stays reference-only.
//
// Formats (per ZoneFSM §6): forces Q24.40 (quantum 2⁻⁴⁰ ≈ 9.1e-13 eV/Å,
// range ±2²³ ≈ 8.4e6 — far beyond any physical force in metal units);
// energy/virial Q34.30 (sign-alternating global sums need more headroom).
//
// Rounding: rint() under the default FE_TONEAREST mode = round-to-nearest-
// even — chosen to MATCH CUDA's cvt.rni (the M4 GPU path must quantize
// identically, or CPU↔GPU bitwise comparison dies on ties).
namespace tdmd::core::fixed {

template <int FracBits>
struct FixedAccum {
  static_assert(FracBits > 0 && FracBits < 63);
  static constexpr double kScale = double(int64_t(1) << FracBits);

  int64_t raw = 0;

  // Quantize one FP64 contribution and add. The quantization is a pure
  // function of x — independent of accumulation history/order.
  void add(double x) {
    const double q = std::rint(x * kScale);
    assert(std::fabs(q) < 9.2e18 && "FixedAccum overflow: contribution too large");
    raw += int64_t(q);
  }

  double value() const { return double(raw) / kScale; }
  void reset() { raw = 0; }
};

using ForceAccum = FixedAccum<40>;   // Q24.40 — per-atom force components
using EnergyAccum = FixedAccum<30>;  // Q34.30 — PE / virial global sums

}  // namespace tdmd::core::fixed
