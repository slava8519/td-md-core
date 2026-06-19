#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/potentials/eam_zone.hpp"  // REUSE zone_eam_window verbatim (potential-agnostic)
#include "tdmd/potentials/tersoff.hpp"

// M6 / Tersoff-ladder Te3 — serial ZONE-Tersoff: the bond-order angular force on the 3-zone TD
// residence {S_{j-1},S_j,S_{j+1}} (width ≥ 2·rcut). Design of record: wf_fc46259f-f57. Mirrors
// the SW T3 driver (sw_zone.hpp) — the gather/scatter skeleton, zone_eam_window, the residence
// THROW, the two-bond min-image guard all transfer. THE GENUINE NEW MACHINERY: Te1's
// tersoff_run_fixed is a SCATTER (center i writes f_i,f_j,f_k to three atoms); on the ring the
// zetaterm f_k write lands in a halo atom another zone OWNS — forbidden. So each owned atom o
// GATHERS only its own force via a TRANSPOSE-REPLAY. UNLIKE SW (wing-canonical collapsed the
// wing into ONE role), Tersoff's directed bond b_ij≠b_ji has no j/k symmetry ⇒ FOUR receiver
// roles: (1) PAIR/repulsive, (2) CENTER-i, (3) ENDPOINT-j, (4) THIRD-k.
//
// BITWISE to tersoff_run_fixed by construction (same int64 multiset — §3 bijection). ζ_ij is an
// FP64 sum (order-sensitive in ℝ — no SW analog). zone_eam_window sorts the window by global id
// ⇒ window-local order == ascending-global == Te1's tersoff_build_nbr order; the single
// zeta_replay helper makes "no role reorders nbr[c]" a structural invariant ⇒ ζ matches Te1.
// HONEST status (Te3 acceptance, the SW T3b pattern — MEASURE-FIRST, do not overstate): the
// global-id sort is DEFENSIVE/robustness, NOT load-bearing for the integrated int64 force on the
// Te3 fixtures — reversing the whole window is BITWISE-identical to run_fixed at perturb ≤ 0.6
// (the ~1e-11 ζ-order delta sits below the Q24.40 quantum after b_ij/prefactor scaling), becoming
// load-bearing only at perturb ≥ 0.75 (G5 measures both regimes). The sort+helper are kept
// defensively (correct for ANY perturb + the Te3b/Te5 carry-forward depends on the invariant).
// Serial here (Te3); the threaded TersoffRing is Te3b; the GPU is Te5.
namespace tdmd::potentials {

// Tersoff force over a CONTIGUOUS gathered window, addressed by LOCAL index. OWNED atoms get
// forces; key = GLOBAL atom id (the φ-once / energy ownership gate). Bit-exact to
// tersoff_run_fixed (same int64 multiset). PRECONDITION (caller asserts via the build): every
// periodic box dim > 2·rcut ⇒ unambiguous min-image for BOTH the (i,j) and i-k bonds. Caller
// zeroes wF*/pe; min_r2/n_bonds/n_triplets accumulated.
template <typename Real>
void tersoff_window_force(const double* wx, const double* wy, const double* wz, const long* key,
                          int m, const int* owned, int n_owned, const TersoffParams& p,
                          const core::PairGeom& geom, std::vector<core::fixed::ForceAccum>& wFx,
                          std::vector<core::fixed::ForceAccum>& wFy,
                          std::vector<core::fixed::ForceAccum>& wFz, core::fixed::EnergyAccum& pe,
                          double& min_r2, long& n_bonds, long& n_triplets) {
  // (0) ALL-WINDOW full neighbour lists (owned AND halo — roles 3/4 read nbr[i] for a HALO
  //     center i, complete because w ≥ 2·rcut). LOCAL index, ascending-global slot order (the
  //     window is global-id-sorted ⇒ matches Te1's tersoff_build_nbr — THE ζ-replay anchor).
  struct WNbr { int b; double dx, dy, dz, r; };
  std::vector<std::vector<WNbr>> nbr(m);
  for (int a = 0; a < m; ++a)
    for (int b = 0; b < m; ++b) {
      if (b == a) continue;
      double dx = wx[a] - wx[b], dy = wy[a] - wy[b], dz = wz[a] - wz[b], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
      min_r2 = std::min(min_r2, r2);  // B10 overlap probe (live path)
      nbr[a].push_back({b, dx, dy, dz, std::sqrt(r2)});
    }

  // The MANDATORY single ζ-replay: ζ_{c,x} = Σ_{k∈nbr[c], k≠x} tersoff_zeta. Walks nbr[c]
  // front-to-back (ascending-global) — the SAME order as Te1's center role ⇒ bitwise ζ. NO
  // role may sort/filter/reorder nbr[c] (structural bitwise invariant).
  auto zeta_replay = [&](int c, int x_local, double rcx, const double* rcxh) -> double {
    double zeta = 0.0;
    for (const auto& k : nbr[c]) {
      if (k.b == x_local) continue;
      const double rikinv = 1.0 / k.r;
      const double rckh[3] = {-k.dx * rikinv, -k.dy * rikinv, -k.dz * rikinv};  // (x_k−x_c)/r
      zeta += tersoff_zeta(rcx, k.r, rcxh, rckh, p);  // FP64 — NEVER a FixedAccum
    }
    return zeta;
  };

  for (int oi = 0; oi < n_owned; ++oi) {
    const int o = owned[oi];

    // Role 1 — PAIR/repulsive: o's symmetric force; energy fc·fR + n_bonds owned by the LOWER
    // GLOBAL KEY (= Te1's i<j on global ids). DEFENSIVE on the sorted serial path.
    for (const auto& e : nbr[o]) {
      const double r = e.r, tmp_exp = std::exp(-p.lam1 * r);
      const double fforce = -p.biga * tmp_exp * (ters_fc_d(r, p) - ters_fc(r, p) * p.lam1) / r;
      wFx[o].add(e.dx * fforce); wFy[o].add(e.dy * fforce); wFz[o].add(e.dz * fforce);
      if (key[o] < key[e.b]) { pe.add(ters_fc(r, p) * p.biga * tmp_exp); ++n_bonds; }
    }

    // Role 2 — CENTER-i of directed bond (o,j): attractive-radial f_i + angular f_i; OWNS the
    // attractive energy + triplet count (ONCE, here only).
    for (const auto& e : nbr[o]) {
      const double rij = e.r, rijinv = 1.0 / rij;
      const double rijh[3] = {-e.dx * rijinv, -e.dy * rijinv, -e.dz * rijinv};  // (x_j−x_o)/r
      const double zeta = zeta_replay(o, e.b, rij, rijh);
      const double fa = ters_fa(rij, p), fa_d = ters_fa_d(rij, p), bij = ters_bij(zeta, p);
      const double fpair_z = 0.5 * bij * fa_d * rijinv;
      const double prefactor = -0.5 * fa * ters_bij_d(zeta, p);
      pe.add(0.5 * bij * fa);  // ATTRACTIVE energy — center-o, ONCE
      wFx[o].add(-e.dx * fpair_z); wFy[o].add(-e.dy * fpair_z); wFz[o].add(-e.dz * fpair_z);
      for (const auto& k : nbr[o]) {
        if (k.b == e.b) continue;
        const double rikinv = 1.0 / k.r;
        const double rikh[3] = {-k.dx * rikinv, -k.dy * rikinv, -k.dz * rikinv};
        double fi[3], fj[3], fk[3];
        tersoff_zetaterm_d(prefactor, rijh, rij, rijinv, rikh, k.r, rikinv, fi, fj, fk, p);
        wFx[o].add(fi[0]); wFy[o].add(fi[1]); wFz[o].add(fi[2]);  // o is CENTER ⇒ the dri slot
        ++n_triplets;                                            // count HERE only
      }
    }

    // Role 3 — ENDPOINT-j of someone else's bond (i,o): attractive-radial f_j + angular f_j,
    // NO energy/count (center i owns both).
    for (const auto& e : nbr[o]) {
      const int i = e.b;
      const double rij = e.r, rijinv = 1.0 / rij;
      const double rijh[3] = {e.dx * rijinv, e.dy * rijinv, e.dz * rijinv};  // (x_o−x_i)/r (opp. of role 2)
      const double zeta = zeta_replay(i, o, rij, rijh);  // ζ_{i,o} over I's nbrs
      const double fa = ters_fa(rij, p), fa_d = ters_fa_d(rij, p), bij = ters_bij(zeta, p);
      const double fpair_z = 0.5 * bij * fa_d * rijinv;
      const double prefactor = -0.5 * fa * ters_bij_d(zeta, p);
      wFx[o].add(-e.dx * fpair_z); wFy[o].add(-e.dy * fpair_z); wFz[o].add(-e.dz * fpair_z);  // f_j radial
      for (const auto& k : nbr[i]) {
        if (k.b == o) continue;
        const double rikinv = 1.0 / k.r;
        const double rikh[3] = {-k.dx * rikinv, -k.dy * rikinv, -k.dz * rikinv};
        double fi[3], fj[3], fk[3];
        tersoff_zetaterm_d(prefactor, rijh, rij, rijinv, rikh, k.r, rikinv, fi, fj, fk, p);
        wFx[o].add(fj[0]); wFy[o].add(fj[1]); wFz[o].add(fj[2]);  // o is ENDPOINT ⇒ the drj slot
      }
    }

    // Role 4 — THIRD-k of someone else's triplet (i;j,o): angular f_k only, NO energy/count.
    for (const auto& e : nbr[o]) {
      const int i = e.b;
      const double rio = e.r, rioinv = 1.0 / rio;
      const double riohat[3] = {e.dx * rioinv, e.dy * rioinv, e.dz * rioinv};  // (x_o−x_i)/r = k-hat
      for (const auto& jb : nbr[i]) {
        if (jb.b == o) continue;  // (i,o) is role 3, not a k-triplet
        const double rij = jb.r, rijinv = 1.0 / rij;
        const double rijh[3] = {-jb.dx * rijinv, -jb.dy * rijinv, -jb.dz * rijinv};  // (x_j−x_i)/r
        const double zeta = zeta_replay(i, jb.b, rij, rijh);  // ζ_{i,j} over I's nbrs
        const double prefactor = -0.5 * ters_fa(rij, p) * ters_bij_d(zeta, p);
        double fi[3], fj[3], fk[3];
        tersoff_zetaterm_d(prefactor, rijh, rij, rijinv, riohat, rio, rioinv, fi, fj, fk, p);
        wFx[o].add(fk[0]); wFy[o].add(fk[1]); wFz[o].add(fk[2]);  // o is THIRD ⇒ the drk slot
      }
    }
  }
}

template <typename Real>
TersoffAccum tersoff_zone_pass_impl(core::AtomSoA<Real>& a, const core::Box& box,
                                    const core::ZoneDecomposition& zd, const TersoffParams& p,
                                    const std::vector<int>& order, bool symmetric) {
  const core::PairGeom geom(box, p.rcut());
  const int n = a.n;
  core::fixed::EnergyAccum pe;
  double min_r2 = 1e300;
  long n_bonds = 0, n_triplets = 0;
  std::vector<int> pos(n, -1);  // global atom -> window-local index

  for (int zi : order) {
    const auto win = zone_eam_window(zd, zi, box.periodic[2], symmetric);  // REUSE verbatim
    const int m = int(win.size());
    std::vector<double> wx(m), wy(m), wz(m);
    std::vector<long> key(m);
    for (int aa = 0; aa < m; ++aa) {
      const int g = win[aa];
      wx[aa] = a.x[g]; wy[aa] = a.y[g]; wz[aa] = a.z[g]; key[aa] = g; pos[g] = aa;
    }
    std::vector<int> owned;
    owned.reserve(zd.members[zi].size());
    for (int g : zd.members[zi]) owned.push_back(pos[g]);

    std::vector<core::fixed::ForceAccum> wFx(m), wFy(m), wFz(m);
    tersoff_window_force<Real>(wx.data(), wy.data(), wz.data(), key.data(), m, owned.data(),
                               int(owned.size()), p, geom, wFx, wFy, wFz, pe, min_r2, n_bonds,
                               n_triplets);
    for (int o = 0; o < int(owned.size()); ++o) {  // owned-written-once scatter
      const int loc = owned[o], g = win[loc];
      a.fx[g] += Real(wFx[loc].value());
      a.fy[g] += Real(wFy[loc].value());
      a.fz[g] += Real(wFz[loc].value());
    }
    for (int aa = 0; aa < m; ++aa) pos[win[aa]] = -1;  // reset for next zone
  }

  TersoffAccum acc;
  acc.pe = pe.value();
  acc.min_r2 = min_r2;
  acc.n_bonds = n_bonds;
  acc.n_triplets = n_triplets;
  return acc;
}

// Serial zone-Tersoff (forces ACCUMULATED into a.f, caller zeroes; returns PE/min_r2/counts).
// order = zone processing order (a permutation of 0..n_zones-1). symmetric=true is the correct
// 3-zone residence; symmetric=false is the forward-only {S_j,S_{j+1}} POISON (drops S_{j-1}) —
// kept only for the residence-loss gate (it diverges from the FP64 oracle while 1-vs-z stays green).
template <typename Real>
TersoffAccum tersoff_zone_pass(core::AtomSoA<Real>& a, const core::Box& box,
                               const core::ZoneDecomposition& zd, const TersoffParams& p,
                               const std::vector<int>& order, bool symmetric = true) {
  core::validate_zone_order(order, zd.n_zones);  // permutation guard
  // Two-bond min-image guard: Tersoff has the (i,j) bond AND the i-k bonds, each min-imaging
  // independently (like SW's two wings) ⇒ a thin transverse box would silently pick a wrong
  // image (corrupt seam angles, deterministically — invisible to 1-vs-z).
  for (int d = 0; d < 3; ++d)
    if (box.periodic[d] && box.len(d) < 2.0 * p.rcut())
      throw std::runtime_error(
          "tersoff_zone_pass: periodic box dim < 2·rcut — min-image is ambiguous (a bond could "
          "wrap to a wrong image); enlarge the box or shrink rcut");
  // Residence guard: the symmetric three-zone window covers ±2·rcut around an owned atom only
  // if each zone is ≥ 2·rcut wide (else the far donor of a ζ-bond escapes — deterministically,
  // so 1-vs-z would NOT flag it; only the independent oracle would).
  if (symmetric && zd.n_zones > 1 && zd.width < 2.0 * p.rcut())
    throw std::runtime_error(
        "tersoff_zone_pass: zone width < 2·rcut — three-zone window insufficient for the "
        "Tersoff bond-order range (potential.rcut and the zone decomposition must match)");
  return tersoff_zone_pass_impl<Real>(a, box, zd, p, order, symmetric);
}

template <typename Real>
TersoffAccum tersoff_zone_pass(core::AtomSoA<Real>& a, const core::Box& box,
                               const core::ZoneDecomposition& zd, const TersoffParams& p,
                               bool symmetric = true) {
  std::vector<int> order(zd.n_zones);
  std::iota(order.begin(), order.end(), 0);
  return tersoff_zone_pass(a, box, zd, p, order, symmetric);
}

}  // namespace tdmd::potentials
