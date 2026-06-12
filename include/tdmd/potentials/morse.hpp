#pragma once
#include "tdmd/core/soa.hpp"
#include "tdmd/potentials/pair_driver.hpp"
#include "tdmd/potentials/pair_morse.hpp"

// Reference O(N²) Morse driver over the pair_morse functor (single source of
// pair math) and the shared direct_pair_loop (single source of the loop —
// pair_driver.hpp). This driver is the FP64 oracle for the M3 clustered path
// (cross test: clustered ≡ O(N²) to <=1e-12).
// Reproduces reference_data/generate_reference.py exactly (truncation=Shift).
//
// CONTRACT (M3 split): compute() ACCUMULATES into a.f{x,y,z} — the caller
// zeroes via core::zero_forces() (zone w-mechanism stacks several partial
// passes into one buffer, INV-3/INV-8). Driver math stays FP64 regardless of
// Real — it is the deterministic reference; the Real-typed fast path is the
// clustered driver.
namespace tdmd::potentials {

template <typename Real>
struct MorsePotential {
  double D = 0.29614, alpha = 1.11892, r0 = 3.29692, rcut = 4.0;
  Truncation truncation = Truncation::Shift;
  // min pair distance^2 seen by the last compute() — overlap-HALT probe (B10).
  // Tracked over ALL pairs (also beyond rcut), so it is the true system minimum.
  double last_min_r2 = 1e300;
  double last_virial = 0.0;  // Σ_{i<j} r·F (NIST W_pair convention)

  // Accumulates into a.f{x,y,z}; returns total potential energy (eV).
  double compute(AtomSoA<Real>& a, const Box& box) {
    const MorseParams<double> prm{D, alpha, r0};
    const CutoffScheme cs = CutoffScheme::make(
        truncation, rcut,
        [&](double r, double& u, double& f) { pair_morse<double>(r, prm, u, f); });
    const PairAccum acc = direct_pair_loop(
        a, box, rcut, [&](double r, double& u, double& f_over_r) {
          pair_morse<double>(r, prm, u, f_over_r);
          cs.apply(r, u, f_over_r);
        });
    last_min_r2 = acc.min_r2;
    last_virial = acc.virial;
    return acc.pe;
  }
};

}  // namespace tdmd::potentials
