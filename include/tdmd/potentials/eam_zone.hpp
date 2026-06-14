#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"  // ZoneDecomposition, PairGeom
#include "tdmd/potentials/eam.hpp"  // EamPotential

// M6 PR-E3 — serial multi-pass zone EAM with SYMMETRIC THREE-ZONE residence.
// The bitwise reference for the EAM ring (analog of zones.hpp::zone_force_pass
// for pairs); the threaded TimeConveyor wiring lands in PR-E3b.
//
// Per owned zone S_zi the node holds the window {S_{zi-1}, S_zi, S_{zi+1}}
// (width ≥ 2·rcut ⇒ every density donor of any neighbour of an owned atom is
// resident — M6_EAM_MANYBODY_DESIGN §1.2) and runs three passes:
//   1 density   ρ_j (fixed-point) for every WINDOW atom — complete for any j
//               that is a neighbour of an owned atom;
//   2 embedding F'_j = F'(ρ_j) for window atoms;
//   3 force     FULL-NEIGHBOUR per OWNED atom: f_i = Σ_j (...). No Newton-3
//               sharing across zones — each owned atom's force is written ONCE
//               by its own zone. Bitwise ≡ the Newton-3 monolith because the
//               quantizer rint is ODD: rint(−x) = −rint(x), so the i-side and
//               j-side contributions of a pair quantize to negatives of each
//               other on either accumulation scheme.
//
// Determinism (B1/INV-9): fixed-point ρ and force ⇒ the per-atom result is
// bit-identical for ANY zone count / processing order. DUAL-FORMAT density
// dispatch on density_fracbits() (Q19.44 / Q23.40) — closes the PR-E2 P0 stopgap
// (eam_run_fixed only handled Q19.44).
namespace tdmd::potentials {

// Window atom indices for zone zi. symmetric=true ⇒ {S_{zi-1}, S_zi, S_{zi+1}}
// (the correct EAM residence). symmetric=false ⇒ {S_zi, S_{zi+1}} — the
// forward-only window of the current pair conveyor; EAM-incorrect, kept ONLY so
// the test can demonstrate that it diverges from the monolith (the adversarial
// halo finding, M6 §9). PBC wraps zone indices cyclically.
inline std::vector<int> zone_eam_window(const core::ZoneDecomposition& zd, int zi,
                                        bool pbc_z, bool symmetric) {
  std::vector<int> w;
  const int nz = zd.n_zones;
  auto add_zone = [&](int z) {
    if (pbc_z) z = (z % nz + nz) % nz;
    if (z < 0 || z >= nz) return;
    for (int at : zd.members[z]) w.push_back(at);
  };
  if (symmetric) add_zone(zi - 1);
  add_zone(zi);
  add_zone(zi + 1);
  std::sort(w.begin(), w.end());
  w.erase(std::unique(w.begin(), w.end()), w.end());
  return w;
}

// M6 PR-E3b-1 — the 3-pass EAM force on ONE owned zone over a CONTIGUOUS window,
// addressed by local index. Extracted from the per-zone body so BOTH the serial
// oracle (global AtomSoA) and the threaded EamRing (fragmented per-zone payloads
// gathered into a window) call the SAME code (the ring has no global AtomSoA
// mid-pass). Bit-exact to the pre-refactor oracle by construction (same multiset
// of int64 contributions). `key` is the canonical φ-once order (serial: global
// atom index; ring: atom id) — either gives φ exactly once per undirected pair,
// PE order-invariant. Forces written into wF{x,y,z}[owned] (local); pe/min_r2
// accumulated. Caller zeroes the accumulators.
template <typename Math, typename DensAccum>
void eam_window_force(const double* wx, const double* wy, const double* wz,
                      const long* key, int m, const int* owned, int n_owned,
                      const Math& math, const core::PairGeom& geom, double rho_cap,
                      std::vector<core::fixed::ForceAccum>& wFx,
                      std::vector<core::fixed::ForceAccum>& wFy,
                      std::vector<core::fixed::ForceAccum>& wFz,
                      core::fixed::EnergyAccum& pe, double& min_r2) {
  // pass 1: density for every window atom
  std::vector<DensAccum> rho(m);
  for (int aa = 0; aa < m; ++aa)
    for (int bb = 0; bb < m; ++bb) {
      if (bb == aa) continue;
      double dx = wx[aa] - wx[bb], dy = wy[aa] - wy[bb], dz = wz[aa] - wz[bb], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
      double v, dv;
      math.eval_rhoa(std::sqrt(r2), v, dv);
      rho[aa].add(v);
    }
  // pass 2: embedding F'(ρ) for window atoms (+ density-range HALT, P1)
  std::vector<double> fp(m);
  for (int aa = 0; aa < m; ++aa) {
    const double r = rho[aa].value();
    if (r > rho_cap)
      throw std::runtime_error("eam_window_force: ρ exceeds the F(ρ) grid");
    double F, Fpv;
    math.eval_F(r, F, Fpv);
    fp[aa] = Fpv;
  }
  // pass 3: FULL-NEIGHBOUR force for OWNED atoms only
  for (int o = 0; o < n_owned; ++o) {
    const int ii = owned[o];
    double Fi, Fpi;
    math.eval_F(rho[ii].value(), Fi, Fpi);
    pe.add(Fi);  // embedding energy, once per owned atom
    for (int bb = 0; bb < m; ++bb) {
      if (bb == ii) continue;
      double dx = wx[ii] - wx[bb], dy = wy[ii] - wy[bb], dz = wz[ii] - wz[bb], r2;
      const bool ok = geom.reduce(dx, dy, dz, r2);
      min_r2 = std::min(min_r2, r2);  // before cutoff — captures overlaps too
      if (!ok) continue;
      const double r = std::sqrt(r2);
      double phi, dphi, ra, dra;
      math.eval_phi(r, phi, dphi);
      math.eval_rhoa(r, ra, dra);
      const double f_over_r = -(dphi + (fp[ii] + fp[bb]) * dra) / r;
      wFx[ii].add(f_over_r * dx);
      wFy[ii].add(f_over_r * dy);
      wFz[ii].add(f_over_r * dz);
      if (key[ii] < key[bb]) pe.add(phi);  // φ once per undirected pair (PE bitwise)
    }
  }
}

template <typename Real, typename Math, typename DensAccum>
EamAccum zone_eam_pass_impl(core::AtomSoA<Real>& a, const core::Box& box,
                            const core::ZoneDecomposition& zd, const Math& math,
                            const std::vector<int>& order, bool symmetric) {
  const core::PairGeom geom(box, math.rcut);
  const int n = a.n;
  core::fixed::EnergyAccum pe;
  double min_r2 = 1e300;  // overlap probe (B10) — system min over visited pairs
  const double rho_cap = math.density_grid_max();
  std::vector<int> pos(n, -1);  // global atom -> window-local index

  for (int zi : order) {
    const auto win = zone_eam_window(zd, zi, box.periodic[2], symmetric);
    const int m = int(win.size());
    // gather the window into contiguous arrays (key = global atom index)
    std::vector<double> wx(m), wy(m), wz(m);
    std::vector<long> key(m);
    for (int aa = 0; aa < m; ++aa) {
      const int g = win[aa];
      wx[aa] = a.x[g]; wy[aa] = a.y[g]; wz[aa] = a.z[g]; key[aa] = g;
      pos[g] = aa;
    }
    std::vector<int> owned;
    owned.reserve(zd.members[zi].size());
    for (int g : zd.members[zi]) owned.push_back(pos[g]);

    std::vector<core::fixed::ForceAccum> wFx(m), wFy(m), wFz(m);
    eam_window_force<Math, DensAccum>(wx.data(), wy.data(), wz.data(), key.data(),
                                      m, owned.data(), int(owned.size()), math,
                                      geom, rho_cap, wFx, wFy, wFz, pe, min_r2);
    // scatter owned forces back (each atom owned by exactly one zone ⇒ once)
    for (int o = 0; o < int(owned.size()); ++o) {
      const int loc = owned[o], g = win[loc];
      a.fx[g] += Real(wFx[loc].value());
      a.fy[g] += Real(wFy[loc].value());
      a.fz[g] += Real(wFz[loc].value());
    }
    for (int aa = 0; aa < m; ++aa) pos[win[aa]] = -1;  // reset for next zone
  }

  EamAccum acc;
  acc.pe = pe.value();
  acc.min_r2 = min_r2;
  return acc;
}

// Dual-format dispatch on the load-time density guard. Writes forces into a.f
// (ACCUMULATED, caller zeroes), returns PE. order = zone processing order
// (permutation of 0..n_zones-1). symmetric=true is the correct EAM residence.
template <typename Real, typename Math>
EamAccum zone_eam_pass(core::AtomSoA<Real>& a, const core::Box& box,
                       const core::ZoneDecomposition& zd,
                       const EamPotential<Real, Math>& pot,
                       const std::vector<int>& order, bool symmetric = true) {
  core::validate_zone_order(order, zd.n_zones);  // permutation guard (UB / silent corruption)
  // Residence guard: the symmetric three-zone window covers ±2·rcut around an
  // owned atom only if the zone is at least 2·rcut wide. Catches a potential/
  // decomposition rcut mismatch (the window would otherwise silently miss
  // donors — deterministically, so 1-vs-z would NOT flag it).
  if (symmetric && zd.n_zones > 1 && zd.width < 2.0 * pot.math.rcut)
    throw std::runtime_error(
        "zone_eam_pass: zone width < 2·rcut — three-zone window insufficient for "
        "the EAM force range (potential.rcut and the zone decomposition must "
        "use the same rcut)");
  const int fb = pot.math.density_fracbits();  // 44 (Q19.44) or 40 (Q23.40)
  if (fb == 44)
    return zone_eam_pass_impl<Real, Math, core::fixed::FixedAccum<44>>(
        a, box, zd, pot.math, order, symmetric);
  if (fb == 40)
    return zone_eam_pass_impl<Real, Math, core::fixed::FixedAccum<40>>(
        a, box, zd, pot.math, order, symmetric);
  throw std::runtime_error("zone_eam_pass: unexpected density_fracbits (not 44/40)");
}

// Convenience overload: identity zone order.
template <typename Real, typename Math>
EamAccum zone_eam_pass(core::AtomSoA<Real>& a, const core::Box& box,
                       const core::ZoneDecomposition& zd,
                       const EamPotential<Real, Math>& pot, bool symmetric = true) {
  std::vector<int> order(zd.n_zones);
  std::iota(order.begin(), order.end(), 0);
  return zone_eam_pass(a, box, zd, pot, order, symmetric);
}

}  // namespace tdmd::potentials
