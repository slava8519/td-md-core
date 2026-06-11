#pragma once
#include <cstdint>
#include <cmath>
#include <array>

#include "tdmd/core/soa.hpp"
#include "tdmd/units.hpp"

// M2.6 (B8): thermal initialization — Maxwell-Boltzmann velocities from the
// global `run.seed`, COM-momentum removal, exact rescale to the target T
// (LAMMPS `velocity create` semantics). Дисс. §3.5–3.6 тестирует тепловые
// режимы; до M2.6 все прогоны стартовали при T≈0, где K2/буфер не нагружены.
namespace tdmd::core::thermal {

// SplitMix64 — fixed PRNG algorithm. std::mt19937/std::normal_distribution are
// implementation-defined across stdlibs; the reproducibility invariant
// ("same seed => bitwise identical runs", INV-9) requires our own generator.
struct SplitMix64 {
  uint64_t s;
  explicit SplitMix64(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  // uniform in (0,1] — never 0, so log() in Box-Muller stays finite
  double uniform() { return double((next() >> 11) + 1) * 0x1.0p-53; }
};

// Standard-normal generator (Box-Muller, fixed evaluation order).
struct Gaussian {
  SplitMix64 rng;
  double spare = 0.0;
  bool   has_spare = false;
  explicit Gaussian(uint64_t seed) : rng(seed) {}
  double operator()() {
    if (has_spare) { has_spare = false; return spare; }
    const double u1 = rng.uniform();
    const double u2 = rng.uniform();
    const double r  = std::sqrt(-2.0 * std::log(u1));
    const double a  = 6.283185307179586476925286766559 * u2;  // 2π
    spare = r * std::sin(a);
    has_spare = true;
    return r * std::cos(a);
  }
};

// Total momentum, amu·Å/ps. FP64 accumulation in fixed index order (INV-9).
template <typename Real>
std::array<double, 3> momentum(const AtomSoA<Real>& a) {
  std::array<double, 3> p{0.0, 0.0, 0.0};
  for (int i = 0; i < a.n; ++i) {
    p[0] += a.mass[i] * double(a.vx[i]);
    p[1] += a.mass[i] * double(a.vy[i]);
    p[2] += a.mass[i] * double(a.vz[i]);
  }
  return p;
}

// Thermal degrees of freedom: 3N minus 3 for the removed COM momentum.
inline int dof_thermal(int n) { return (n > 1) ? 3 * n - 3 : 0; }

// Instantaneous temperature, K: T = 2·KE / (n_dof·kB).
template <typename Real>
double temperature(const AtomSoA<Real>& a, int n_dof) {
  if (n_dof <= 0) return 0.0;
  double ke = 0.0;
  for (int i = 0; i < a.n; ++i) {
    const double v2 = double(a.vx[i]) * a.vx[i] +
                      double(a.vy[i]) * a.vy[i] +
                      double(a.vz[i]) * a.vz[i];
    ke += 0.5 * units::mvv2e * a.mass[i] * v2;
  }
  return 2.0 * ke / (n_dof * units::kB);
}

// Subtract the mass-weighted mean velocity so total momentum is ~0
// (residual is FP rounding only).
template <typename Real>
void zero_momentum(AtomSoA<Real>& a) {
  if (a.n < 2) return;
  double m_tot = 0.0;
  for (int i = 0; i < a.n; ++i) m_tot += a.mass[i];
  if (m_tot <= 0.0) return;
  const auto p = momentum(a);
  const double ux = p[0] / m_tot, uy = p[1] / m_tot, uz = p[2] / m_tot;
  for (int i = 0; i < a.n; ++i) {
    a.vx[i] -= Real(ux);
    a.vy[i] -= Real(uy);
    a.vz[i] -= Real(uz);
  }
}

// Uniform rescale to the exact target temperature. A uniform factor preserves
// zero total momentum, so this is safe after zero_momentum().
template <typename Real>
void scale_to_temperature(AtomSoA<Real>& a, double T_target, int n_dof) {
  const double T_now = temperature(a, n_dof);
  if (T_now <= 0.0 || T_target <= 0.0) return;
  const Real f = Real(std::sqrt(T_target / T_now));
  for (int i = 0; i < a.n; ++i) { a.vx[i] *= f; a.vy[i] *= f; a.vz[i] *= f; }
}

// Maxwell-Boltzmann initialization at temperature T (K) from `seed`.
// Per-component sigma = sqrt(kB·T / (mvv2e·m)) in Å/ps; draw order is fixed
// (atom index ascending, then x,y,z) for reproducibility. After COM removal
// the velocities are rescaled so T(0) == T exactly on n_dof = 3N−3.
template <typename Real>
void maxwell_init(AtomSoA<Real>& a, double T, uint64_t seed) {
  if (a.n == 0 || T <= 0.0) return;
  Gaussian g(seed);
  for (int i = 0; i < a.n; ++i) {
    const double sigma = std::sqrt(units::kB * T / (units::mvv2e * a.mass[i]));
    a.vx[i] = Real(sigma * g());
    a.vy[i] = Real(sigma * g());
    a.vz[i] = Real(sigma * g());
  }
  zero_momentum(a);
  scale_to_temperature(a, T, dof_thermal(a.n));
  // Second pass kills the FP-rounding residual of the first subtraction
  // (~1e-12 amu·Å/ps for unlucky seeds) down to ~1e-13; the temperature
  // change is O((u_res/v_thermal)²) ~ 1e-30 — far below any tolerance.
  zero_momentum(a);
}

} // namespace tdmd::core::thermal
