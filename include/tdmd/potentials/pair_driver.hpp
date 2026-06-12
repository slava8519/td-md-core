#pragma once
#include <cmath>
#include <vector>
#include <algorithm>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/cluster.hpp"
#include "tdmd/potentials/cutoff.hpp"

// Shared pair-driver LOOPS (M3 LJ generalization). The O(N²) reference loop
// and the clustered loop are potential-agnostic: each potential wrapper
// (Morse/LJ × direct/clustered) supplies a pair callback and keeps the public
// contract — compute() ACCUMULATES forces (caller zeroes, core::zero_forces),
// exposes last_min_r2 (overlap HALT, B10) and last_virial.
//
// PairFn contract: void(double r, double& u, double& f_over_r). Pair math may
// run in Real inside (production_mixed §3.3); the returned FP64 contributions
// have the truncation scheme ALREADY applied (cutoff policy belongs to the
// driver side — pair_morse.hpp note). The cutoff test itself is HERE, in FP64,
// so the pair SET is identical on every path and precision.
//
// Numerics are op-for-op those of the pre-split M3 drivers (deterministic
// accumulation order: outer i, inner j, locals -> store) — verified bitwise
// against the pre-refactor binary on config_m0/config_auto/cluster runs.
namespace tdmd::potentials {

using tdmd::core::AtomSoA;
using tdmd::core::Box;

// Per-compute() bookkeeping shared by both loops.
struct PairAccum {
  double pe = 0.0;        // Σ pair energies (truncation applied)
  double virial = 0.0;    // Σ_{i<j} r·F(r) = −Σ_{i<j} r·dU/dr (NIST W_pair)
  double min_r2 = 1e300;  // min over VISITED pairs (true system min for the
                          // direct loop; pairs within rcut+skin for clustered)
};

// Reference O(N²) loop with minimum-image PBC — the FP64 oracle.
template <typename Real, typename PairFn>
PairAccum direct_pair_loop(AtomSoA<Real>& a, const Box& box, double rcut,
                           PairFn&& pair) {
  const double rc2 = rcut * rcut;
  const double L[3] = {box.len(0), box.len(1), box.len(2)};
  PairAccum acc;
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
      acc.min_r2 = std::min(acc.min_r2, r2);
      if (r2 >= rc2 || r2 < 1e-18) continue;
      double u, f_over_r;
      pair(std::sqrt(r2), u, f_over_r);
      acc.pe += 0.5 * u;                       // pair counted twice over i,j
      acc.virial += 0.5 * f_over_r * r2;       // r·F = f_over_r·r²
      fxi += f_over_r * dx;
      fyi += f_over_r * dy;
      fzi += f_over_r * dz;
    }
    a.fx[i] += fxi;
    a.fy[i] += fyi;
    a.fz[i] += fzi;
  }
  return acc;
}

// Clustered driver engine (M3): Z-order clusters + cluster-pair list with a
// skin layer + the Verlet-table rebuild policy (valid while no atom moved
// more than skin/2 since the last build; the auto-step C1 bound makes the
// cadence predictable — see the Cluster.RebuildCadence test).
//
// NOTE: rebuild SORTS the atoms in place (Z-order) — atom identity lives in
// a.id. The permutation is deterministic (key: morton, id), so run-to-run
// bitwise reproducibility is preserved.
template <typename Real>
class ClusteredPairEngine {
 public:
  double skin = 1.0;   // Å (neighbor.skin)
  double cell = 2.33;  // Å binning cell (decomposition.cell_size)
  long rebuild_count = 0;

  const core::ClusterSet& cluster_set() const { return set_; }

 protected:
  template <typename PairFn>
  PairAccum run_pairs(AtomSoA<Real>& a, const Box& box, double rcut,
                      PairFn&& pair) {
    if (need_rebuild(a)) rebuild(a, box, rcut);

    const double rc2 = rcut * rcut;
    const double L[3] = {box.len(0), box.len(1), box.len(2)};
    PairAccum acc;
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
            acc.min_r2 = std::min(acc.min_r2, r2);
            if (r2 >= rc2 || r2 < 1e-18) continue;
            double u, f_over_r;
            pair(std::sqrt(r2), u, f_over_r);
            acc.pe += 0.5 * u;
            acc.virial += 0.5 * f_over_r * r2;
            fxi += f_over_r * dx;
            fyi += f_over_r * dy;
            fzi += f_over_r * dz;
          }
        }
        a.fx[i] += fxi;
        a.fy[i] += fyi;
        a.fz[i] += fzi;
      }
    }
    return acc;
  }

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

  void rebuild(AtomSoA<Real>& a, const Box& box, double rcut) {
    set_.build(a, box, cell, rcut + skin);
    x0_.assign(a.x.begin(), a.x.end());
    y0_.assign(a.y.begin(), a.y.end());
    z0_.assign(a.z.begin(), a.z.end());
    ++rebuild_count;
  }

  core::ClusterSet set_;
  std::vector<double> x0_, y0_, z0_;  // positions at last rebuild
};

}  // namespace tdmd::potentials
