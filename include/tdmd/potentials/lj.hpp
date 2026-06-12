#pragma once
#include "tdmd/core/soa.hpp"
#include "tdmd/potentials/pair_driver.hpp"
#include "tdmd/potentials/pair_lj.hpp"

// Lennard-Jones drivers (M3): the pair_lj functor over the SAME shared loops
// as Morse (pair_driver.hpp) — direct O(N²) reference and clustered. Same
// contract: compute() ACCUMULATES forces (caller zeroes), exposes last_min_r2
// and last_virial, drops into core::run_simulation unchanged.
//
// External validation: NIST SRSW frozen LJ configurations (M3 acceptance) —
// reference_data/nist_lj/ + Test_LJ_NIST. Defaults are reduced units
// (ε=σ=1); metal-unit parameters come from config potential.lj.
namespace tdmd::potentials {

template <typename Real>
struct LJPotential {
  double epsilon = 1.0, sigma = 1.0, rcut = 3.0;
  Truncation truncation = Truncation::Shift;
  double last_min_r2 = 1e300;
  double last_virial = 0.0;

  double compute(AtomSoA<Real>& a, const Box& box) {
    const LJParams<double> prm{epsilon, sigma};
    const CutoffScheme cs = CutoffScheme::make(
        truncation, rcut,
        [&](double r, double& u, double& f) { pair_lj<double>(r, prm, u, f); });
    const PairAccum acc = direct_pair_loop(
        a, box, rcut, [&](double r, double& u, double& f_over_r) {
          pair_lj<double>(r, prm, u, f_over_r);
          cs.apply(r, u, f_over_r);
        });
    last_min_r2 = acc.min_r2;
    last_virial = acc.virial;
    return acc.pe;
  }
};

template <typename Real>
struct ClusteredLJ : ClusteredPairEngine<Real> {
  double epsilon = 1.0, sigma = 1.0, rcut = 3.0;
  Truncation truncation = Truncation::Shift;
  double last_min_r2 = 1e300;
  double last_virial = 0.0;

  double compute(AtomSoA<Real>& a, const Box& box) {
    const LJParams<Real> prm{Real(epsilon), Real(sigma)};
    const LJParams<double> prmd{epsilon, sigma};
    const CutoffScheme cs = CutoffScheme::make(
        truncation, rcut,
        [&](double r, double& u, double& f) { pair_lj<double>(r, prmd, u, f); });
    const PairAccum acc = this->run_pairs(
        a, box, rcut, [&](double r, double& u, double& f_over_r) {
          Real ur, fr;
          pair_lj<Real>(Real(r), prm, ur, fr);
          u = double(ur);
          f_over_r = double(fr);
          cs.apply(r, u, f_over_r);
        });
    last_min_r2 = acc.min_r2;
    last_virial = acc.virial;
    return acc.pe;
  }
};

}  // namespace tdmd::potentials
