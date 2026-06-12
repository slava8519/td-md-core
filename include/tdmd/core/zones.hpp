#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"

// M3.5 (first PR) — zone decomposition along z + the w-mechanism partial-force
// assembly, SERIAL: the force logic of the TD conveyor (дисс. Гл. 2.1,
// ZoneFSM §2/§7.2) without threads/transport, so it can be cross-checked
// against the monolithic O(N²) oracle before the ring lands on top.
//
// Per dissertation INV-8 (Newton-3, single counting): computing zone S_i
// contributes (a) the pairs INTERNAL to S_i and (b) the cross pairs
// S_i ↔ S_{i+1}, each evaluated ONCE with the force applied to both atoms —
// the S_{i+1} side of (b) is exactly the partial w-contribution the conveyor
// later stacks across passes (d→w SPHERE, T2). PBC closure (§7.2/B3):
// S_{n-1}'s partner wraps to S_0.
//
// Preconditions enforced by build(): zone width >= rcut (a zone only reaches
// its immediate neighbour — ConfigSchema validation rule), and >= 3 zones
// when z is periodic (with 2 zones the direct interface and the PBC closure
// connect the SAME zone pair, double-counting every min-image pair — the
// same reason cell lists need >= 3 cells per periodic dim).
namespace tdmd::core {

struct ZoneDecomposition {
  int n_zones = 0;
  double width = 0.0;
  std::vector<std::vector<int>> members;  // atom indices per zone, ascending

  // Bins atoms into n_zones equal slabs along z (wrapped into the box on a
  // periodic z). Throws std::invalid_argument on precondition violations.
  template <typename Real>
  static ZoneDecomposition build(const AtomSoA<Real>& a, const Box& box,
                                 int n_zones, double rcut) {
    const double L = box.len(2);
    ZoneDecomposition zd;
    zd.n_zones = n_zones;
    zd.width = L / n_zones;
    if (n_zones < 1) throw std::invalid_argument("zones: n_zones must be >= 1");
    if (n_zones > 1 && zd.width < rcut)
      throw std::invalid_argument(
          "zones: zone width < r_cut — a zone would reach beyond its "
          "neighbour (ConfigSchema: decomposition.zone_width >= potential.r_cut)");
    if (box.periodic[2] && n_zones == 2)
      throw std::invalid_argument(
          "zones: 2 zones with periodic z double-count every pair (direct "
          "interface == PBC closure); use 1 or >= 3 zones");
    zd.members.assign(n_zones, {});
    for (int i = 0; i < a.n; ++i) {
      double z = a.z[i];
      if (box.periodic[2]) z -= L * std::floor((z - box.lo[2]) / L);
      int zi = int((z - box.lo[2]) / zd.width);
      zd.members[std::clamp(zi, 0, n_zones - 1)].push_back(i);
    }
    return zd;
  }
};

// Shared per-pair geometry core — ONE source of the min-image/cutoff FP
// expressions for the serial w-pass below AND the ring conveyor
// (core/conveyor.hpp), so their per-pair math is bit-identical (INV-9).
struct PairGeom {
  double L[3];
  bool per[3];
  double rc2;
  PairGeom(const Box& box, double rcut)
      : L{box.len(0), box.len(1), box.len(2)},
        per{box.periodic[0], box.periodic[1], box.periodic[2]},
        rc2(rcut * rcut) {}
  // Min-image reduction + acceptance — exactly the pre-refactor inline
  // predicate: accept iff 1e-18 <= r2 < rc2.
  bool reduce(double& dx, double& dy, double& dz, double& r2) const {
    if (per[0]) dx -= L[0] * std::round(dx / L[0]);
    if (per[1]) dy -= L[1] * std::round(dy / L[1]);
    if (per[2]) dz -= L[2] * std::round(dz / L[2]);
    r2 = dx * dx + dy * dy + dz * dz;
    return r2 < rc2 && r2 >= 1e-18;
  }
};

// One full force assembly over zone passes. PairFn: the drivers' contract —
// void(double r, double& u, double& f_over_r), FP64 contributions with the
// truncation scheme already applied (potentials/cutoff.hpp policy).
// PairHook: void(int i, int j) — invoked once per EVALUATED pair (the INV-8
// single-counting probe of the acceptance test); pass a no-op otherwise.
//
// Accumulation is FIXED-POINT (B1): per-atom Q24.40 force accumulators and a
// Q34.30 PE accumulator, so the result is bit-identical for ANY zone
// processing order (`order` — a permutation of 0..n_zones-1; the conveyor
// shuffles effective ordering across nodes). Quantized results are written
// (ACCUMULATED, like the drivers) into a.f and returned as PE.
template <typename Real, typename PairFn, typename PairHook>
double zone_force_pass(AtomSoA<Real>& a, const Box& box,
                       const ZoneDecomposition& zd, double rcut, PairFn&& pair,
                       const std::vector<int>& order, PairHook&& on_pair) {
  const PairGeom geom(box, rcut);

  std::vector<fixed::ForceAccum> fx(a.n), fy(a.n), fz(a.n);
  fixed::EnergyAccum pe;

  // one evaluated pair: Newton-3, force on BOTH atoms, energy once
  auto do_pair = [&](int i, int j) {
    double dx = a.x[i] - a.x[j];
    double dy = a.y[i] - a.y[j];
    double dz = a.z[i] - a.z[j];
    double r2;
    if (!geom.reduce(dx, dy, dz, r2)) return;
    double u, f_over_r;
    pair(std::sqrt(r2), u, f_over_r);
    on_pair(i, j);
    pe.add(u);
    fx[i].add(f_over_r * dx);
    fy[i].add(f_over_r * dy);
    fz[i].add(f_over_r * dz);
    fx[j].add(-f_over_r * dx);
    fy[j].add(-f_over_r * dy);
    fz[j].add(-f_over_r * dz);
  };

  for (int zi : order) {
    const auto& zone = zd.members[zi];
    // (a) internal pairs of S_zi (i < j — once)
    for (size_t s = 0; s < zone.size(); ++s)
      for (size_t t = s + 1; t < zone.size(); ++t) do_pair(zone[s], zone[t]);
    // (b) cross pairs S_zi <-> S_{zi+1}, counted HERE (INV-8); PBC closure
    // wraps the last zone's partner to S_0 (§7.2)
    if (zd.n_zones > 1) {
      const int znext = (zi + 1) % zd.n_zones;
      if (znext > zi || box.periodic[2])
        for (int i : zone)
          for (int j : zd.members[znext]) do_pair(i, j);
    }
  }

  for (int i = 0; i < a.n; ++i) {
    a.fx[i] += Real(fx[i].value());
    a.fy[i] += Real(fy[i].value());
    a.fz[i] += Real(fz[i].value());
  }
  return pe.value();
}

template <typename Real, typename PairFn>
double zone_force_pass(AtomSoA<Real>& a, const Box& box,
                       const ZoneDecomposition& zd, double rcut, PairFn&& pair) {
  std::vector<int> order(zd.n_zones);
  std::iota(order.begin(), order.end(), 0);
  return zone_force_pass(a, box, zd, rcut, pair, order, [](int, int) {});
}

}  // namespace tdmd::core
