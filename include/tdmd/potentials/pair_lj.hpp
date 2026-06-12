#pragma once
#include <cmath>
#include <numbers>

#include "tdmd/hal/hal.hpp"

// Pair Lennard-Jones math — the SINGLE source of truth (same rule as
// pair_morse.hpp): every driver (host O(N²), M3 clustered, M4 CUDA kernel)
// instantiates THESE functions, never re-derives the formulas.
//   U(r) = 4ε[(σ/r)¹² − (σ/r)⁶]
//   F(r) = −dU/dr = (24ε/r)[2(σ/r)¹² − (σ/r)⁶]
// LJ is the project's external-validation potential (NIST SRSW frozen
// configurations, M3 acceptance) and the flagship 10⁶–10⁷ benchmark species
// (validation ladder §3.5; needed by the M5b replicas).
// template<Real>: FP64 reference path, FP32 production_mixed pair math.
namespace tdmd::potentials {

template <typename Real>
struct LJParams {
  Real epsilon, sigma;
};

// Energy and force/r for one pair at distance r (0 < r < r_cut; cutoff test
// and truncation scheme stay in the driver — cutoff.hpp).
template <typename Real>
TDMD_HOST_DEVICE inline void pair_lj(Real r, const LJParams<Real>& p,
                                     Real& u, Real& f_over_r) {
  const Real inv_r2 = Real(1) / (r * r);
  const Real sr2 = p.sigma * p.sigma * inv_r2;
  const Real sr6 = sr2 * sr2 * sr2;
  const Real sr12 = sr6 * sr6;
  u = Real(4) * p.epsilon * (sr12 - sr6);
  f_over_r = Real(24) * p.epsilon * (Real(2) * sr12 - sr6) * inv_r2;
}

// Standard long-range (tail) corrections for the truncated LJ potential,
// assuming g(r)=1 beyond r_cut. EXTENSIVE values (×N), matching the NIST SRSW
// "+LRC" tables: U_LRC = N·(8/3)πρεσ³[⅓(σ/rc)⁹ − (σ/rc)³].
// Not added by the drivers (they report the pure pair sum) — callers add it
// where the comparison scheme requires (EOS work, NIST cross-checks).
inline double lj_lrc_energy(double epsilon, double sigma, double rcut,
                            long n, double volume) {
  const double rho = double(n) / volume;
  const double sr3 = (sigma / rcut) * (sigma / rcut) * (sigma / rcut);
  const double sr9 = sr3 * sr3 * sr3;
  return double(n) * (8.0 / 3.0) * std::numbers::pi * rho * epsilon *
         sigma * sigma * sigma * (sr9 / 3.0 - sr3);
}

}  // namespace tdmd::potentials
