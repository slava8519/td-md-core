#pragma once
#include "tdmd/core/soa.hpp"
#include "tdmd/potentials/pair_driver.hpp"
#include "tdmd/potentials/pair_morse.hpp"

// M3 clustered Morse driver: the pair_morse functor over the shared
// ClusteredPairEngine loop (pair_driver.hpp — single source of the clustered
// loop, the rebuild policy and the determinism notes). Same contract as the
// reference driver (compute ACCUMULATES, caller zeroes, exposes last_min_r2),
// so it drops into core::run_simulation unchanged.
//
// Precision split (production_mixed contract, MIXED_PRECISION doc §3.3):
// coordinates and pair differences (min-image) stay FP64; the PAIR MATH runs
// in Real (FP32 on the mixed path); accumulation is FP64 with a fixed
// deterministic order (int64 fixed-point replaces it on the GPU at M4 — B1).
// The cutoff decision is made in FP64, so the pair SET is identical to the
// O(N^2) reference for any Real — the cross test compares apples to apples.
namespace tdmd::potentials {

template <typename Real>
struct ClusteredMorse : ClusteredPairEngine<Real> {
  // pair parameters (defaults = dissertation Al/Morse, как у MorsePotential)
  double D = 0.29614, alpha = 1.11892, r0 = 3.29692, rcut = 4.0;
  Truncation truncation = Truncation::Shift;

  double last_min_r2 = 1e300;
  double last_virial = 0.0;

  double compute(AtomSoA<Real>& a, const Box& box) {
    const MorseParams<Real> prm{Real(D), Real(alpha), Real(r0)};
    const MorseParams<double> prmd{D, alpha, r0};
    const CutoffScheme cs = CutoffScheme::make(
        truncation, rcut,
        [&](double r, double& u, double& f) { pair_morse<double>(r, prmd, u, f); });
    const PairAccum acc = this->run_pairs(
        a, box, rcut, [&](double r, double& u, double& f_over_r) {
          Real ur, fr;
          pair_morse<Real>(Real(r), prm, ur, fr);
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
