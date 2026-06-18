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
#include "tdmd/potentials/sw.hpp"

// M6 / SW-ladder T3 — serial ZONE-SW: the Stillinger–Weber angular force on the
// 3-zone TD residence {S_{j-1}, S_j, S_{j+1}} (width ≥ 2·rcut). Design of record:
// wf_c90b13cf-202. Mirrors the EAM zone driver (eam_zone.hpp) — the gather/scatter
// skeleton, zone_eam_window, the width<2·rcut residence THROW, the per-bond min-image
// all transfer. The genuine SW deltas (3-body non-symmetric): (A) ALL-WINDOW neighbour
// lists (the wing role reads nbr(i) for a possibly-HALO center i); (B) triplet energy +
// count owned by the CENTER's zone, ONCE (never in the wing role); (C) φ₂ energy gated on
// the GLOBAL key (not a local index); (D) two wing bonds that can independently PBC-wrap.
//
// 1-vs-z BITWISE by construction: the window holds every triplet an owned atom is in
// (every role), and sw_triplet is wing-canonical, so the per-owner gather emits the same
// order-free int64 multiset as the whole-system sw_run_fixed. Serial here (T3); the
// threaded SwRing = EamRing<…,SwWinForce> is T3b; the GPU kernel is T5.
namespace tdmd::potentials {

// SW force over a CONTIGUOUS gathered window, addressed by LOCAL index. OWNED atoms get
// forces; key = GLOBAL atom id (the φ-once / triplet-energy ownership gate). Bit-exact to
// sw_run_fixed by construction (same int64 multiset). PRECONDITION (caller asserts via
// the build): every periodic box dim > 2·rcut ⇒ unambiguous min-image for BOTH wing
// bonds. Caller zeroes wF*/pe; min_r2/n_triplets accumulated.
template <typename Real>
void sw_window_force(const double* wx, const double* wy, const double* wz,
                     const long* key, int m, const int* owned, int n_owned,
                     const SwParams& sp, const core::PairGeom& geom,
                     std::vector<core::fixed::ForceAccum>& wFx,
                     std::vector<core::fixed::ForceAccum>& wFy,
                     std::vector<core::fixed::ForceAccum>& wFz,
                     core::fixed::EnergyAccum& pe, double& min_r2, long& n_triplets) {
  // (0) ALL-WINDOW full neighbour lists (owned AND halo — risk #3: the wing role reads
  //     nbr[i] for a halo center i, which must be complete; the window reaches rcut past
  //     any neighbour i of an owned atom because w ≥ 2·rcut). LOCAL index, per-bond
  //     min-image via the SAME PairGeom as the monolith. dx = wx[a]−wx[b] (bond from a).
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

  // (1) per OWNED atom o (local index), TRANSPOSE-REPLAY — byte-identical to sw_run_fixed.
  for (int oi = 0; oi < n_owned; ++oi) {
    const int o = owned[oi];
    const auto& no = nbr[o];

    // (i) φ₂: o sums its own pair force; the LOWER GLOBAL KEY owns the energy. The key is a
    //     DEFENSIVE choice (correct for ANY window order). On the serial path the window is
    //     sorted ⇒ a local index would also work; and — MEASURED in T3b (test_sw_ring) — even
    //     the UNSORTED ring gather is count-once-invariant (a φ₂ pair spans only adjacent
    //     zones, the slot order is zone-consistent) ⇒ the global key is ROBUST, not load-bearing.
    for (const auto& e : no) {
      double phi, dphi;
      sw_phi2(sp, e.r, phi, dphi);
      const double f_over_r = -dphi / e.r;
      wFx[o].add(f_over_r * e.dx); wFy[o].add(f_over_r * e.dy); wFz[o].add(f_over_r * e.dz);
      if (key[o] < key[e.b]) pe.add(phi);
    }

    // (ii) φ₃ with o as CENTER: o owns f_i AND the energy + count, over unordered wing
    //      pairs. Energy/count are emitted ONCE, HERE ONLY (risk #4 — adding them in the
    //      wing role double-counts, still z-invariant ⇒ invisible to 1-vs-z, caught only
    //      by the FP64-oracle PE compare).
    for (std::size_t x = 0; x < no.size(); ++x)
      for (std::size_t y = x + 1; y < no.size(); ++y) {
        SwVec3 fi, fj, fk;
        const double E3 = sw_triplet(sp, no[x].dx, no[x].dy, no[x].dz, no[x].r,
                                     no[y].dx, no[y].dy, no[y].dz, no[y].r, fi, fj, fk);
        wFx[o].add(fi.x); wFy[o].add(fi.y); wFz[o].add(fi.z);
        pe.add(E3); ++n_triplets;
      }

    // (iii) φ₃ with o as a WING: for each neighbour i (a center), replay every triplet
    //       {i; o, m} (m in i's FULL window nbr list) putting o in the j-slot. NO energy,
    //       NO count. sw_triplet wing-canonical ⇒ f_j(i;o,m) == f_k(i;m,o) bitwise.
    for (const auto& ie : no) {
      const int i = ie.b;
      const double iox = -ie.dx, ioy = -ie.dy, ioz = -ie.dz, rio = ie.r;  // x_i − x_o
      for (const auto& me : nbr[i]) {
        if (me.b == o) continue;  // m ≠ o (the i-o bond is the o-wing, not a wing pair)
        SwVec3 fi, fj, fk;
        sw_triplet(sp, iox, ioy, ioz, rio, me.dx, me.dy, me.dz, me.r, fi, fj, fk);
        wFx[o].add(fj.x); wFy[o].add(fj.y); wFz[o].add(fj.z);
      }
    }
  }
}

template <typename Real>
SwAccum sw_zone_pass_impl(core::AtomSoA<Real>& a, const core::Box& box,
                          const core::ZoneDecomposition& zd, const SwParams& sp,
                          const std::vector<int>& order, bool symmetric) {
  const core::PairGeom geom(box, sp.rcut());
  const int n = a.n;
  core::fixed::EnergyAccum pe;
  double min_r2 = 1e300;
  long n_triplets = 0;
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
    sw_window_force<Real>(wx.data(), wy.data(), wz.data(), key.data(), m, owned.data(),
                          int(owned.size()), sp, geom, wFx, wFy, wFz, pe, min_r2, n_triplets);
    // scatter owned forces back (each atom owned by exactly one zone ⇒ written once)
    for (int o = 0; o < int(owned.size()); ++o) {
      const int loc = owned[o], g = win[loc];
      a.fx[g] += Real(wFx[loc].value());
      a.fy[g] += Real(wFy[loc].value());
      a.fz[g] += Real(wFz[loc].value());
    }
    for (int aa = 0; aa < m; ++aa) pos[win[aa]] = -1;  // reset for next zone
  }

  SwAccum acc;
  acc.pe = pe.value();
  acc.min_r2 = min_r2;
  acc.n_triplets = n_triplets;
  return acc;
}

// Serial zone-SW (writes forces into a.f ACCUMULATED, caller zeroes; returns PE/min_r2/
// n_triplets). order = zone processing order (a permutation of 0..n_zones-1).
// symmetric=true is the correct 3-zone residence; symmetric=false is the forward-only
// {S_j, S_{j+1}} POISON (drops S_{j-1}) — EAM-incorrect, kept only for the residence-loss
// gate (it diverges from the FP64 oracle while 1-vs-z stays green).
template <typename Real>
SwAccum sw_zone_pass(core::AtomSoA<Real>& a, const core::Box& box,
                     const core::ZoneDecomposition& zd, const SwParams& sp,
                     const std::vector<int>& order, bool symmetric = true) {
  core::validate_zone_order(order, zd.n_zones);  // permutation guard (UB / silent corruption)
  // Min-image guard (SW two-wing PBC, risk #5b): a triplet has TWO wing bonds that can
  // independently wrap; each PairGeom::reduce picks the UNIQUE nearest image only if every
  // periodic box dim > 2·rcut. A thin transverse box would silently pick a wrong image ⇒
  // corrupt seam angles, deterministically (invisible to 1-vs-z). EAM (one donor bond) is
  // laxer; SW makes this explicit.
  for (int d = 0; d < 3; ++d)
    if (box.periodic[d] && box.len(d) < 2.0 * sp.rcut())
      throw std::runtime_error(
          "sw_zone_pass: periodic box dim < 2·rcut — min-image is ambiguous (a wing bond "
          "could wrap to a wrong image); enlarge the box or shrink rcut");
  // Residence guard: the symmetric three-zone window covers ±2·rcut around an owned atom
  // only if each zone is ≥ 2·rcut wide. Catches a potential/decomposition rcut mismatch
  // (the window would otherwise silently miss the far wing — deterministically, so 1-vs-z
  // would NOT flag it; only the independent oracle would).
  if (symmetric && zd.n_zones > 1 && zd.width < 2.0 * sp.rcut())
    throw std::runtime_error(
        "sw_zone_pass: zone width < 2·rcut — three-zone window insufficient for the SW "
        "force range (potential.rcut and the zone decomposition must use the same rcut)");
  return sw_zone_pass_impl<Real>(a, box, zd, sp, order, symmetric);
}

template <typename Real>
SwAccum sw_zone_pass(core::AtomSoA<Real>& a, const core::Box& box,
                     const core::ZoneDecomposition& zd, const SwParams& sp,
                     bool symmetric = true) {
  std::vector<int> order(zd.n_zones);
  std::iota(order.begin(), order.end(), 0);
  return sw_zone_pass(a, box, zd, sp, order, symmetric);
}

}  // namespace tdmd::potentials
