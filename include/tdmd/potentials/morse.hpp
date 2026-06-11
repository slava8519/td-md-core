#pragma once
#include <cmath>
#include <algorithm>
#include "tdmd/core/soa.hpp"

// Pair Morse potential with minimum-image PBC and optional energy shift at rcut.
// Naive O(N^2) over the whole system (M0 scope — no neighbour lists, no zones).
// Reproduces reference_data/generate_reference.py exactly:
//   U(r) = D*(e^{-2a(r-r0)} - 2 e^{-a(r-r0)}) - Ushift,   r < rcut
//   F_i  = sum_j (-dU/dr) * (r_i - r_j)/r
// Params from dissertation Гл.3.5 (см. reference_data/README.md).
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
    return D * (std::exp(-2 * alpha * (rcut - r0)) -
                2 * std::exp(-alpha * (rcut - r0)));
  }

  // Fills a.f{x,y,z}; returns total potential energy (eV).
  // Deterministic accumulation order: outer i, inner j, locals -> store.
  double compute(AtomSoA<Real>& a, const Box& box) {
    const double ush = energy_shift();
    const double rc2 = rcut * rcut;
    const double L[3] = {box.len(0), box.len(1), box.len(2)};
    std::fill(a.fx.begin(), a.fx.end(), 0.0);
    std::fill(a.fy.begin(), a.fy.end(), 0.0);
    std::fill(a.fz.begin(), a.fz.end(), 0.0);

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
        const double r   = std::sqrt(r2);
        const double ea  = std::exp(-alpha * (r - r0));
        const double e2a = ea * ea;
        pe += 0.5 * (D * (e2a - 2 * ea) - ush);  // pair counted twice over i,j
        const double fmag  = 2 * alpha * D * (e2a - ea);  // = -dU/dr
        const double inv_r = 1.0 / r;
        fxi += fmag * inv_r * dx;
        fyi += fmag * inv_r * dy;
        fzi += fmag * inv_r * dz;
      }
      a.fx[i] = fxi;
      a.fy[i] = fyi;
      a.fz[i] = fzi;
    }
    last_min_r2 = min_r2;
    return pe;
  }
};

} // namespace tdmd::potentials
