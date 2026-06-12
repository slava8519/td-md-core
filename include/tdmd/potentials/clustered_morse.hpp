#pragma once
#include <cmath>
#include <vector>
#include <algorithm>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/cluster.hpp"
#include "tdmd/potentials/pair_morse.hpp"

// M3 clustered driver over the pair_morse functor. Same contract as the
// reference driver (compute ACCUMULATES, caller zeroes, exposes last_min_r2),
// so it drops into core::run_simulation unchanged.
//
// Precision split (production_mixed contract, MIXED_PRECISION doc §3.3):
// coordinates and pair differences (min-image) stay FP64; the PAIR MATH runs
// in Real (FP32 on the mixed path); accumulation is FP64 with a fixed
// deterministic order (int64 fixed-point replaces it on the GPU at M4 — B1).
// The cutoff decision is made in FP64, so the pair SET is identical to the
// O(N^2) reference for any Real — the cross test compares apples to apples.
//
// Rebuild policy (Verlet-table analogue, ур. 32-контекст): the pair list is
// valid while no atom has moved more than skin/2 since the last build; the
// auto-step C1 bound makes the rebuild cadence predictable.
namespace tdmd::potentials {

using tdmd::core::AtomSoA;
using tdmd::core::Box;

template <typename Real>
struct ClusteredMorse {
  // pair parameters (defaults = dissertation Al/Morse, как у MorsePotential)
  double D = 0.29614, alpha = 1.11892, r0 = 3.29692, rcut = 4.0;
  bool   shift = true;
  double skin = 1.0;        // Å (neighbor.skin)
  double cell = 2.33;       // Å binning cell (decomposition.cell_size)

  double last_min_r2 = 1e300;
  long   rebuild_count = 0;   // M3 acceptance: cadence must match the C1 estimate

  double energy_shift() const {
    if (!shift) return 0.0;
    double u, f_over_r;
    pair_morse<double>(rcut, {D, alpha, r0}, u, f_over_r);
    return u;
  }

  // NOTE: rebuild SORTS the atoms in place (Z-order) — atom identity lives in
  // a.id. The permutation is deterministic (key: morton, id), so run-to-run
  // bitwise reproducibility is preserved.
  double compute(AtomSoA<Real>& a, const Box& box) {
    if (need_rebuild(a)) rebuild(a, box);

    const MorseParams<Real> prm{Real(D), Real(alpha), Real(r0)};
    const double ush = energy_shift();
    const double rc2 = rcut * rcut;
    const double L[3] = {box.len(0), box.len(1), box.len(2)};

    double pe = 0.0;
    double min_r2 = 1e300;
    const auto& cl = set_.clusters;
    for (size_t A = 0; A < cl.size(); ++A) {
      for (int i = cl[A].begin; i < cl[A].end; ++i) {
        double fxi = 0.0, fyi = 0.0, fzi = 0.0;
        for (int B : set_.nbr[A]) {
          for (int j = cl[B].begin; j < cl[B].end; ++j) {
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
            Real u, f_over_r;
            pair_morse<Real>(Real(std::sqrt(r2)), prm, u, f_over_r);
            pe += 0.5 * (double(u) - ush);
            fxi += double(f_over_r) * dx;
            fyi += double(f_over_r) * dy;
            fzi += double(f_over_r) * dz;
          }
        }
        a.fx[i] += fxi;
        a.fy[i] += fyi;
        a.fz[i] += fzi;
      }
    }
    last_min_r2 = min_r2;
    return pe;
  }

  const core::ClusterSet& cluster_set() const { return set_; }

 private:
  bool need_rebuild(const AtomSoA<Real>& a) const {
    if (x0_.empty() || int(x0_.size()) != a.n) return true;
    const double lim2 = 0.25 * skin * skin;  // (skin/2)^2
    for (int i = 0; i < a.n; ++i) {
      const double dx = a.x[i] - x0_[i];
      const double dy = a.y[i] - y0_[i];
      const double dz = a.z[i] - z0_[i];
      if (dx * dx + dy * dy + dz * dz > lim2) return true;
    }
    return false;
  }

  void rebuild(AtomSoA<Real>& a, const Box& box) {
    set_.build(a, box, cell, rcut + skin);
    x0_.assign(a.x.begin(), a.x.end());
    y0_.assign(a.y.begin(), a.y.end());
    z0_.assign(a.z.begin(), a.z.end());
    ++rebuild_count;
  }

  core::ClusterSet set_;
  std::vector<double> x0_, y0_, z0_;  // positions at last rebuild
};

} // namespace tdmd::potentials
