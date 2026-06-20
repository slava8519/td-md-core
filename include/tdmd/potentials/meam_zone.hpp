#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/potentials/eam_zone.hpp"  // REUSE zone_eam_window verbatim (potential-agnostic gather)
#include "tdmd/potentials/meam.hpp"

// M6 / MEAM-ladder Me3 — serial ZONE-MEAM: the screened many-body force on the 3-zone TD
// residence {S_{j-1},S_j,S_{j+1}} (width ≥ 2·rc). Design of record: /tmp/meam3_design.txt.
// Mirrors the Te3 driver (tersoff_zone.hpp) — the gather/scatter skeleton, zone_eam_window, the
// residence THROW, the min-image guard all transfer.
//
// THE GENUINE NEW MACHINERY vs EAM/Tersoff. Me2's meam_run_fixed_force is a SCATTER: each directed
// bond (i,j), owner i<j, writes the symmetric pair/embedding force to TWO atoms (ffx[i]+=fm,
// ffx[j]-=fm) AND — for a PARTIALLY-screened bond (0<S<1) — the screening k-loop writes to THREE
// atoms (ffx[i]+=force1·d_ik, ffx[j]+=force2·d_jk, ffx[k]-=...). On the zone path the cross-zone
// writes to j and k are forbidden; each owned atom o GATHERS its own force by replaying every
// bond/triplet where o is a receiver.
//
// THE BITWISE-TO-Me2 STRUCTURE (the load-bearing discipline, design §1.2). Instead of three
// hand-inverted role bodies (error-prone — each would re-derive the 470-line bond chain and could
// silently diverge), meam_window_force REPLAYS THE Me2 SCATTER VERBATIM over every directed bond
// (i,j), i<j, whose CENTER i is resident in the window (owned OR halo — the EAM redundant-halo
// discipline lifted to the bond level), computing fm0/fm1/fm2 and force1/force2 in I'S FRAME
// exactly as Me2 — then scatters each contribution ONLY when its target (i, j, or k) is an OWNED
// window atom. So an owned o receives the EXACT int64 multiset Me2 writes to ffx[o]:
//   • Role A (o is the lower endpoint i): the +fm side + the f_i screening term;
//   • Role B (o is the higher endpoint j): the −fm side + the f_j screening term — computed from
//     i's frame (i may be halo), NOT recomputed from o's — the LOAD-BEARING line: reading S_ij
//     from o's frame would diverge it UNIFORMLY across z (1-vs-z blind, only the oracle catches);
//   • Role C (o is the screening-k): the −(force1·d_ik+force2·d_jk) third-atom write (q(k)≠−q(i),
//     the needs_transpose write).
// core::fixed::ForceAccum integer addition is associative (B1/INV-9) ⇒ each owned force is BITWISE
// == meam_run_fixed_force at z=1 and BITWISE across z (1-vs-z), summed in a different order.
//
// CANDIDATE-COMPLETENESS — the {2,true} reach (design §2.1, PROVEN ≡ EAM). Me2's screening k-loop
// draws k STRICTLY from i's rc neighbour list (the LAMMPS getscreen approximation), so the worst
// reach is i∈nbr(o) (<rc), then k∈nbr(i) (<rc of i) = 2·rc — identical to EAM. (NB: the
// rbound=ebound·rij² ellipse filter is an ADDITIONAL rejection layered on top of the rc cutoff —
// rik<rc ALWAYS, NOT √ebound·rc; the reach is rc+rc=2·rc, not rc·(1+√ebound).) The genuine
// escalation is the redundant HALO DENSITY: a halo center i's density is itself a SCREENED product
// Π_k S(C_ijk) over k∈nbr[i], so every screening-k of every halo bond must be resident — the reach
// proves it is; the independent meam_direct_fp64 oracle (NOT 1-vs-z, which is blind to a uniformly
// dropped k) is the sole completeness witness. A future MEAM variant with a SEPARATE screening
// neighbour list (genuine non-neighbour screening) would break the {2,true} reach silently — gate
// at that rung. Serial here (Me3); the threaded MeamRing is Me3b; the GPU is Me5.
namespace tdmd::potentials {

// MEAM force over a CONTIGUOUS gathered window, addressed by LOCAL index. OWNED atoms get forces;
// key = GLOBAL atom id (the φ-once / energy ownership gate AND the directed-bond i<j owner gate).
// Bit-exact to meam_run_fixed_force (same int64 multiset). Caller zeroes wF*/pe; min_r2 +
// n_screened_* accumulated. drop_class>0 = POISON (the screening k-scatter dropped — the MB2
// teeth; on the zone path this is the in-zone analog of the Me2 poison).
template <typename Real>
void meam_window_force(const double* wx, const double* wy, const double* wz, const long* key,
                       int m, const int* owned, int n_owned, const MeamParams& p,
                       const core::PairGeom& geom, std::vector<core::fixed::ForceAccum>& wFx,
                       std::vector<core::fixed::ForceAccum>& wFy,
                       std::vector<core::fixed::ForceAccum>& wFz,
                       core::fixed::EnergyAccum& pe_embed, core::fixed::EnergyAccum& pe_pair,
                       double& min_r2, long& n_screened_partial, long& n_screened_zero,
                       int drop_class = 0) {
  // owned[] is a list of LOCAL window indices; mark them for the scatter-to-owned-only gate.
  std::vector<char> is_owned(m, 0);
  for (int oi = 0; oi < n_owned; ++oi) is_owned[owned[oi]] = 1;

  // (0) ALL-WINDOW full neighbour lists nbr[a] for EVERY window atom a (owned AND halo), built
  //     with dx = wx[b] − wx[a] to MATCH MeamNbr exactly (meam_build_nbr stores x_j − x_i for
  //     nbr[i]) ⇒ every Me2 helper takes a byte-identical MeamNbr. The window is global-id-sorted
  //     by zone_eam_window ⇒ window-local order == ascending-global == Me2's meam_build_nbr order
  //     ⇒ the screening product Π_k fcut(C_ijk) and the bond-body arithmetic are bit-identical.
  std::vector<std::vector<MeamNbr>> nbr(m);
  for (int a = 0; a < m; ++a)
    for (int b = 0; b < m; ++b) {
      if (b == a) continue;
      double dx = wx[b] - wx[a], dy = wy[b] - wy[a], dz = wz[b] - wz[a], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
      min_r2 = std::min(min_r2, r2);
      nbr[a].push_back({b, dx, dy, dz, r2, std::sqrt(r2)});
    }

  // (1) screening (scrfcn,fcpair,dscrfcn) + int64 partial densities for EVERY window atom. The
  //     single meam_getscreen_d helper — NEVER recomputed per role. Counts increment ONLY when a
  //     is OWNED AND key[a] < key[ej.j] (count-once, == Me2's i<j on global ids).
  struct FixedDens {
    core::fixed::ForceAccum rho0, arho2b;
    core::fixed::ForceAccum arho1[3], arho2[6], arho3[10], arho3b[3], t_ave[3];
  };
  std::vector<FixedDens> fd(m);
  std::vector<std::vector<MeamScreenD>> scr(m);
  for (int c = 0; c < m; ++c) {
    scr[c].resize(nbr[c].size());
    for (size_t jn = 0; jn < nbr[c].size(); ++jn) {
      const auto& ej = nbr[c][jn];
      const MeamScreenD s = meam_getscreen_d(ej, nbr[c], p);
      scr[c][jn] = s;
      const double sij = s.scrfcn * s.fcpair;
      if (is_owned[c] && key[c] < key[ej.j]) {
        if (s.scrfcn == 0.0) ++n_screened_zero;
        else if (s.scrfcn > 0.0 && s.scrfcn < 1.0) ++n_screened_partial;
      }
      if (std::fabs(sij) < 1e-20) continue;
      const double rij2 = ej.r2, rij = ej.r;
      const double aj = rij / p.re - 1.0, ro0 = p.rho0;
      const double rhoa0j = ro0 * std::exp(-p.beta0 * aj) * sij;
      const double rhoa1j = ro0 * std::exp(-p.beta1 * aj) * sij;
      const double rhoa2j = ro0 * std::exp(-p.beta2 * aj) * sij;
      const double rhoa3j = ro0 * std::exp(-p.beta3 * aj) * sij;
      fd[c].rho0.add(rhoa0j);
      fd[c].t_ave[0].add(p.t1_eff * rhoa0j);
      fd[c].t_ave[1].add(p.t2 * rhoa0j);
      fd[c].t_ave[2].add(p.t3 * rhoa0j);
      fd[c].arho2b.add(rhoa2j);
      const double A1j = rhoa1j / rij, A2j = rhoa2j / rij2, A3j = rhoa3j / (rij2 * rij);
      const double del[3] = {ej.dx, ej.dy, ej.dz};
      int nv2 = 0, nv3 = 0;
      for (int mm = 0; mm < 3; ++mm) {
        fd[c].arho1[mm].add(A1j * del[mm]);
        fd[c].arho3b[mm].add(rhoa3j * del[mm] / rij);
        for (int n = mm; n < 3; ++n) {
          fd[c].arho2[nv2].add(A2j * del[mm] * del[n]);
          ++nv2;
          for (int pp = n; pp < 3; ++pp) {
            fd[c].arho3[nv3].add(A3j * del[mm] * del[n] * del[pp]);
            ++nv3;
          }
        }
      }
    }
  }

  // (2) decode int64 densities + meam_dens_final_deriv for EVERY window atom (halo atoms get the
  //     full MeamEmbedDeriv because Role B/C read ed[i] of a HALO center i). pe_embed only for
  //     OWNED atoms (once each, partitioned by zone membership).
  std::vector<MeamDensity> dens(m);
  std::vector<MeamEmbedDeriv> ed(m);
  for (int c = 0; c < m; ++c) {
    MeamDensity& d = dens[c];
    d.rho0 = fd[c].rho0.value();
    d.arho2b = fd[c].arho2b.value();
    for (int mm = 0; mm < 3; ++mm) {
      d.arho1[mm] = fd[c].arho1[mm].value();
      d.arho3b[mm] = fd[c].arho3b[mm].value();
      d.t_ave[mm] = fd[c].t_ave[mm].value();
    }
    for (int mm = 0; mm < 6; ++mm) d.arho2[mm] = fd[c].arho2[mm].value();
    for (int mm = 0; mm < 10; ++mm) d.arho3[mm] = fd[c].arho3[mm].value();
    ed[c] = meam_dens_final_deriv(d, p);
    if (is_owned[c]) pe_embed.add(ed[c].F);
  }

  // (3) the FORCE transpose-replay: walk every directed bond (i,j), i<j by GLOBAL KEY, whose
  //     CENTER i is resident (every window atom — owned and halo); compute the full Me2 bond body
  //     in i's frame; scatter each f_i/f_j/f_k contribution ONLY to OWNED targets. This is the
  //     EXACT inversion of meam_run_fixed_force's scatter — same int64 contributions, gathered by
  //     owner. (A bond owned by a halo center i still writes its −fm to an owned endpoint j and
  //     its f_k to an owned screening atom — that is exactly why the halo's nbr list + ed[i] must
  //     be resident.)
  for (int i = 0; i < m; ++i) {
    for (size_t jn = 0; jn < nbr[i].size(); ++jn) {
      const auto& ej = nbr[i][jn];
      const int j = ej.j;
      if (key[i] >= key[j]) continue;  // each undirected bond ONCE, owner = lower global key
      const double scrfcn_ij = scr[i][jn].scrfcn, dscrfcn_ij = scr[i][jn].dscrfcn;
      if (std::fabs(scrfcn_ij) < 1e-20) continue;
      const double sij0 = scrfcn_ij * scr[i][jn].fcpair;
      const double rij2 = ej.r2, rij = ej.r;
      const double phi = p.phi_spline(rij), phip = p.phip_spline(rij);
      // pair energy: owner i<j, but pe_pair only counts when i is OWNED (its zone attributes it).
      if (is_owned[i]) pe_pair.add(phi * sij0);

      // SINGLE-SOURCE bond body (shared with meam_run_fixed_force + the Me5 GPU K3 kernel). The
      // transpose-replay scatters each contribution ONLY to OWNED targets. F-NOOP.
      const MeamForceParams fp = meam_force_params(p);
      const MeamBondForce bf =
          meam_bond_force_device(dens[i], dens[j], ed[i], ed[j], ej.dx, ej.dy, ej.dz, rij2, rij,
                                 scrfcn_ij, scr[i][jn].fcpair, dscrfcn_ij, phi, phip, fp);
      // Role A (o == i, the lower endpoint): +fm. Role B (o == j, the higher endpoint): −fm.
      if (is_owned[i]) { wFx[i].add(bf.fm[0]); wFy[i].add(bf.fm[1]); wFz[i].add(bf.fm[2]); }
      if (is_owned[j]) { wFx[j].add(-bf.fm[0]); wFy[j].add(-bf.fm[1]); wFz[j].add(-bf.fm[2]); }

      // The screening 3rd-atom k-loop (fires only for partial 0<sij<1 — dead on binary S). POISON
      // (drop_class>0): drop it entirely — the MB2 enumeration teeth (atom-k's force collapses ⇒
      // diverges from the oracle).
      if (std::fabs(sij0) < 1e-20 || std::fabs(sij0 - 1.0) < 1e-20) continue;
      if (drop_class > 0) continue;
      for (size_t kn = 0; kn < nbr[i].size(); ++kn) {
        const auto& ek = nbr[i][kn];
        const int k = ek.j;
        if (k == j) continue;
        const double rik2 = ek.dx * ek.dx + ek.dy * ek.dy + ek.dz * ek.dz;
        const MeamScreenK sk = meam_screen_k_device(ek.dx, ek.dy, ek.dz, rik2, ej.dx, ej.dy, ej.dz,
                                                    rij2, sij0, bf.dUdsij, fp);
        if (!sk.active) continue;
        // Role A (o == i): +force1·d_ik. Role B (o == j): +force2·d_jk. Role C (o == k): the
        // −(force1·d_ik + force2·d_jk) third-atom write — the needs_transpose non-symmetric write.
        if (is_owned[i]) { wFx[i].add(sk.force1 * sk.dik[0]); wFy[i].add(sk.force1 * sk.dik[1]); wFz[i].add(sk.force1 * sk.dik[2]); }
        if (is_owned[j]) { wFx[j].add(sk.force2 * sk.djk[0]); wFy[j].add(sk.force2 * sk.djk[1]); wFz[j].add(sk.force2 * sk.djk[2]); }
        if (is_owned[k]) {
          wFx[k].add(-(sk.force1 * sk.dik[0] + sk.force2 * sk.djk[0]));
          wFy[k].add(-(sk.force1 * sk.dik[1] + sk.force2 * sk.djk[1]));
          wFz[k].add(-(sk.force1 * sk.dik[2] + sk.force2 * sk.djk[2]));
        }
      }
    }
  }
}

template <typename Real>
MeamAccum meam_zone_pass_impl(core::AtomSoA<Real>& a, const core::Box& box,
                              const core::ZoneDecomposition& zd, const MeamParams& p,
                              const std::vector<int>& order, bool symmetric, int drop_class) {
  const core::PairGeom geom(box, p.rc);
  const int n = a.n;
  core::fixed::EnergyAccum pe_embed, pe_pair;
  double min_r2 = 1e300;
  long n_partial = 0, n_zero = 0;
  std::vector<int> pos(n, -1);  // global atom -> window-local index

  for (int zi : order) {
    const auto win = zone_eam_window(zd, zi, box.periodic[2], symmetric);  // REUSE verbatim (sorts)
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
    meam_window_force<Real>(wx.data(), wy.data(), wz.data(), key.data(), m, owned.data(),
                            int(owned.size()), p, geom, wFx, wFy, wFz, pe_embed, pe_pair, min_r2,
                            n_partial, n_zero, drop_class);
    for (int o = 0; o < int(owned.size()); ++o) {  // owned-written-once scatter
      const int loc = owned[o], g = win[loc];
      a.fx[g] += Real(wFx[loc].value());
      a.fy[g] += Real(wFy[loc].value());
      a.fz[g] += Real(wFz[loc].value());
    }
    for (int aa = 0; aa < m; ++aa) pos[win[aa]] = -1;  // reset for next zone
  }

  MeamAccum acc;
  acc.pe_embed = pe_embed.value();
  acc.pe_pair = pe_pair.value();
  acc.pe = acc.pe_embed + acc.pe_pair;
  acc.min_r2 = min_r2;
  acc.n_screened_partial = n_partial;
  acc.n_screened_zero = n_zero;
  return acc;
}

// Serial zone-MEAM (forces ACCUMULATED into a.f, caller zeroes; returns PE/min_r2/screening counts).
// order = zone processing order (a permutation of 0..n_zones-1). symmetric=true is the correct
// 3-zone residence; symmetric=false is the forward-only {S_j,S_{j+1}} POISON (drops S_{j-1}) — kept
// only for the residence-loss gate (it diverges from the FP64 oracle while 1-vs-z stays green).
template <typename Real>
MeamAccum meam_zone_pass(core::AtomSoA<Real>& a, const core::Box& box,
                         const core::ZoneDecomposition& zd, const MeamParams& p,
                         const std::vector<int>& order, bool symmetric = true, int drop_class = 0) {
  core::validate_zone_order(order, zd.n_zones);  // permutation guard
  // Three-leg min-image guard: MEAM is WORSE than Tersoff — the screening triple has THREE
  // independent inter-atom vectors: i–j (ej), i–k (ek), and j–k = ek.d − ej.d (a DIFFERENCE of two
  // min-imaged vectors, meam.hpp:1438). A thin periodic box could pick a wrong image for rjk2 even
  // when i–j and i–k are correct. 2·rc < L ⇒ all three legs < rc < L/2 are unambiguous.
  for (int d = 0; d < 3; ++d)
    if (box.periodic[d] && box.len(d) < 2.0 * p.rc)
      throw std::runtime_error(
          "meam_zone_pass: periodic box dim < 2·rc — min-image ambiguous (the screening j–k leg, "
          "a difference of two min-imaged vectors, could wrap to a wrong image); enlarge the box "
          "or shrink rc");
  // Residence guard: the symmetric three-zone window covers ±2·rc around an owned atom only if
  // each zone is ≥ 2·rc wide (else the screening-k of an owned bond — drawn from i∈nbr(o), then
  // k∈nbr(i) up to 2·rc away — escapes the window, deterministically; 1-vs-z would NOT flag it,
  // only the independent oracle would).
  if (symmetric && zd.n_zones > 1 && zd.width < 2.0 * p.rc)
    throw std::runtime_error(
        "meam_zone_pass: zone width < 2·rc — three-zone window insufficient for the MEAM screening "
        "range (potential.rc and the zone decomposition must use the same rc)");
  return meam_zone_pass_impl<Real>(a, box, zd, p, order, symmetric, drop_class);
}

// Convenience overload: identity zone order.
template <typename Real>
MeamAccum meam_zone_pass(core::AtomSoA<Real>& a, const core::Box& box,
                         const core::ZoneDecomposition& zd, const MeamParams& p,
                         bool symmetric = true, int drop_class = 0) {
  std::vector<int> order(zd.n_zones);
  std::iota(order.begin(), order.end(), 0);
  return meam_zone_pass(a, box, zd, p, order, symmetric, drop_class);
}

}  // namespace tdmd::potentials
