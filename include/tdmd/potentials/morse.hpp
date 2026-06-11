#pragma once
#include <cmath>
#include <algorithm>

#include "tdmd/core/soa.hpp"
#include "tdmd/potentials/pair_morse.hpp"

// Reference O(N²) driver over the pair_morse functor (single source of pair
// math — see pair_morse.hpp), with minimum-image PBC and optional energy shift
// at rcut. This driver is the FP64 oracle for the M3 clustered path
// (cross test: clustered ≡ O(N²) to <=1e-12).
// Reproduces reference_data/generate_reference.py exactly.
//
// CONTRACT (M3 split): compute() ACCUMULATES into a.f{x,y,z} — the caller
// zeroes via core::zero_forces() (zone w-mechanism stacks several partial
// passes into one buffer, INV-3/INV-8). Driver math stays FP64 regardless of
// Real — it is the deterministic reference; the Real-typed fast path arrives
// with the clustered driver.
namespace tdmd::potentials {

using tdmd::core::AtomSoA;
using tdmd::core::Box;

template <typename Real>
struct MorsePotential {
  double D = 0.29614, alpha = 1.11892, r0 = 3.29692, rcut = 4.0;
  bool   shift = true;
  // min pair distance^2 seen by the last compute() — overlap-HALT probe (B10).
  // Tracked over ALL pairs (also beyond rcut), so it is the true system minimum.
  double last_min_r2 = 1e300;

  double energy_shift() const {
    if (!shift) return 0.0;
    double u, f_over_r;
    pair_morse<double>(rcut, {D, alpha, r0}, u, f_over_r);
    return u;
  }

  // Accumulates into a.f{x,y,z}; returns total potential energy (eV).
  // Deterministic accumulation order: outer i, inner j, locals -> store.
  double compute(AtomSoA<Real>& a, const Box& box) {
    const MorseParams<double> prm{D, alpha, r0};
    const double ush = energy_shift();
    const double rc2 = rcut * rcut;
    const double L[3] = {box.len(0), box.len(1), box.len(2)};

    double pe = 0.0;
    double min_r2 = 1e300;
    for (int i = 0; i < a.n; ++i) {
      double fxi = 0.0, fyi = 0.0, fzi = 0.0;
      for (int j = 0; j < a.n; ++j) {
        if (j == i) continue;
        double dx = a.x[i] - a.x[j];
        double dy = a.y[i] - a.y[j];
        double dz = a.z[i] - a.z[j];
        if (box.periodic[0]) dx -= L[0] * std::round(dx / L[0]);
        if (box.periodic[1]) dy -= L[1] * std::round(dy / L[1]);
        if (box.periodic[2]) dz -= L[2] * std::round(dz / L[2]);
        const double r2 = dx * dx + dy * dy + dz * dz;
        min_r2 = std::min(min_r2, r2);
        if (r2 >= rc2 || r2 < 1e-18) continue;
        double u, f_over_r;
        pair_morse<double>(std::sqrt(r2), prm, u, f_over_r);
        pe += 0.5 * (u - ush);  // pair counted twice over i,j
        fxi += f_over_r * dx;
        fyi += f_over_r * dy;
        fzi += f_over_r * dz;
      }
      a.fx[i] += fxi;
      a.fy[i] += fyi;
      a.fz[i] += fzi;
    }
    last_min_r2 = min_r2;
    return pe;
  }
};

} // namespace tdmd::potentials
