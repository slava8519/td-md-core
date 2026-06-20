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
      const double rij2 = ej.r2, rij = ej.r, recip = 1.0 / rij, rij3 = rij * rij2;
      const double delij[3] = {ej.dx, ej.dy, ej.dz};
      const double phi = p.phi_spline(rij), phip = p.phip_spline(rij);
      // pair energy: owner i<j, but pe_pair only counts when i is OWNED (its zone attributes it).
      if (is_owned[i]) pe_pair.add(phi * sij0);

      const double invre = 1.0 / p.re, ai = rij * invre - 1.0, ro0 = p.rho0;
      const double rhoa0j = ro0 * std::exp(-p.beta0 * ai), drhoa0j = -p.beta0 * invre * rhoa0j;
      const double rhoa1j = ro0 * std::exp(-p.beta1 * ai), drhoa1j = -p.beta1 * invre * rhoa1j;
      const double rhoa2j = ro0 * std::exp(-p.beta2 * ai), drhoa2j = -p.beta2 * invre * rhoa2j;
      const double rhoa3j = ro0 * std::exp(-p.beta3 * ai), drhoa3j = -p.beta3 * invre * rhoa3j;
      const double rhoa0i = rhoa0j, drhoa0i = drhoa0j, rhoa1i = rhoa1j, drhoa1i = drhoa1j;
      const double rhoa2i = rhoa2j, drhoa2i = drhoa2j, rhoa3i = rhoa3j, drhoa3i = drhoa3j;
      const double t1mi = p.t1_eff, t2mi = p.t2, t3mi = p.t3;
      const double t1mj = p.t1_eff, t2mj = p.t2, t3mj = p.t3;
      const MeamDensity &di = dens[i], &dj = dens[j];

      double arg1i1 = 0, arg1j1 = 0, arg1i2 = 0, arg1j2 = 0;
      double arg1i3 = 0, arg1j3 = 0, arg3i3 = 0, arg3j3 = 0;
      {
        int nv2 = 0, nv3 = 0;
        for (int n = 0; n < 3; ++n) {
          for (int pp = n; pp < 3; ++pp) {
            for (int q = pp; q < 3; ++q) {
              const double arg = delij[n] * delij[pp] * delij[q] * p.v3D[nv3];
              arg1i3 += di.arho3[nv3] * arg; arg1j3 -= dj.arho3[nv3] * arg; ++nv3;
            }
            const double arg = delij[n] * delij[pp] * p.v2D[nv2];
            arg1i2 += di.arho2[nv2] * arg; arg1j2 += dj.arho2[nv2] * arg; ++nv2;
          }
          arg1i1 += di.arho1[n] * delij[n]; arg1j1 -= dj.arho1[n] * delij[n];
          arg3i3 += di.arho3b[n] * delij[n]; arg3j3 -= dj.arho3b[n] * delij[n];
        }
      }

      const double drho0dr1 = drhoa0j * sij0, drho0dr2 = drhoa0i * sij0;
      double a1 = 2.0 * sij0 / rij;
      const double drho1dr1 = a1 * (drhoa1j - rhoa1j / rij) * arg1i1;
      const double drho1dr2 = a1 * (drhoa1i - rhoa1i / rij) * arg1j1;
      double drho1drm1[3], drho1drm2[3];
      for (int mm = 0; mm < 3; ++mm) {
        drho1drm1[mm] = a1 * rhoa1j * di.arho1[mm];
        drho1drm2[mm] = -a1 * rhoa1i * dj.arho1[mm];
      }
      double a2 = 2.0 * sij0 / rij2;
      const double drho2dr1 =
          a2 * (drhoa2j - 2.0 * rhoa2j / rij) * arg1i2 - 2.0 / 3.0 * di.arho2b * drhoa2j * sij0;
      const double drho2dr2 =
          a2 * (drhoa2i - 2.0 * rhoa2i / rij) * arg1j2 - 2.0 / 3.0 * dj.arho2b * drhoa2i * sij0;
      a2 = 4.0 * sij0 / rij2;
      double drho2drm1[3], drho2drm2[3];
      for (int mm = 0; mm < 3; ++mm) {
        double s1 = 0.0, s2 = 0.0;
        for (int n = 0; n < 3; ++n) {
          s1 += di.arho2[p.vind2D[mm][n]] * delij[n];
          s2 -= dj.arho2[p.vind2D[mm][n]] * delij[n];
        }
        drho2drm1[mm] = a2 * rhoa2j * s1;
        drho2drm2[mm] = -a2 * rhoa2i * s2;
      }
      double a3 = 2.0 * sij0 / rij3, a3a = 6.0 / 5.0 * sij0 / rij;
      const double drho3dr1 =
          a3 * (drhoa3j - 3.0 * rhoa3j / rij) * arg1i3 - a3a * (drhoa3j - rhoa3j / rij) * arg3i3;
      const double drho3dr2 =
          a3 * (drhoa3i - 3.0 * rhoa3i / rij) * arg1j3 - a3a * (drhoa3i - rhoa3i / rij) * arg3j3;
      a3 = 6.0 * sij0 / rij3;
      a3a = 6.0 * sij0 / (5.0 * rij);
      double drho3drm1[3], drho3drm2[3];
      for (int mm = 0; mm < 3; ++mm) {
        double s1 = 0.0, s2 = 0.0;
        int nv2 = 0;
        for (int n = 0; n < 3; ++n)
          for (int pp = n; pp < 3; ++pp) {
            const double arg = delij[n] * delij[pp] * p.v2D[nv2];
            s1 += di.arho3[p.vind3D[mm][n][pp]] * arg;
            s2 += dj.arho3[p.vind3D[mm][n][pp]] * arg;
            ++nv2;
          }
        drho3drm1[mm] = (a3 * s1 - a3a * di.arho3b[mm]) * rhoa3j;
        drho3drm2[mm] = (-a3 * s2 + a3a * dj.arho3b[mm]) * rhoa3i;
      }

      const double t1i = ed[i].t_ave[0], t2i = ed[i].t_ave[1], t3i = ed[i].t_ave[2];
      const double t1j = ed[j].t_ave[0], t2j = ed[j].t_ave[1], t3j = ed[j].t_ave[2];
      const double aif = (ed[i].rho0 != 0.0) ? drhoa0j * sij0 / ed[i].rho0 : 0.0;
      const double ajf = (ed[j].rho0 != 0.0) ? drhoa0i * sij0 / ed[j].rho0 : 0.0;
      const double dt1dr1 = aif * (t1mj - t1i), dt1dr2 = ajf * (t1mi - t1j);
      const double dt2dr1 = aif * (t2mj - t2i), dt2dr2 = ajf * (t2mi - t2j);
      const double dt3dr1 = aif * (t3mj - t3i), dt3dr2 = ajf * (t3mi - t3j);
      const double* shp = p.shp;

      const double drhodr1 =
          ed[i].dgamma1 * drho0dr1 +
          ed[i].dgamma2 * (dt1dr1 * ed[i].rho1 + t1i * drho1dr1 + dt2dr1 * ed[i].rho2 +
                           t2i * drho2dr1 + dt3dr1 * ed[i].rho3 + t3i * drho3dr1) -
          ed[i].dgamma3 * (shp[0] * dt1dr1 + shp[1] * dt2dr1 + shp[2] * dt3dr1);
      const double drhodr2 =
          ed[j].dgamma1 * drho0dr2 +
          ed[j].dgamma2 * (dt1dr2 * ed[j].rho1 + t1j * drho1dr2 + dt2dr2 * ed[j].rho2 +
                           t2j * drho2dr2 + dt3dr2 * ed[j].rho3 + t3j * drho3dr2) -
          ed[j].dgamma3 * (shp[0] * dt1dr2 + shp[1] * dt2dr2 + shp[2] * dt3dr2);
      double drhodrm1[3], drhodrm2[3];
      for (int mm = 0; mm < 3; ++mm) {
        drhodrm1[mm] = ed[i].dgamma2 *
                       (t1i * drho1drm1[mm] + t2i * drho2drm1[mm] + t3i * drho3drm1[mm]);
        drhodrm2[mm] = ed[j].dgamma2 *
                       (t1j * drho1drm2[mm] + t2j * drho2drm2[mm] + t3j * drho3drm2[mm]);
      }

      double drhods1 = 0.0, drhods2 = 0.0;
      const bool screen_active = std::fabs(dscrfcn_ij) > 1e-20;
      if (screen_active) {
        const double drho0ds1 = rhoa0j, drho0ds2 = rhoa0i;
        const double b1 = 2.0 / rij, b2 = 2.0 / rij2, b3 = 2.0 / rij3, b3a = 6.0 / (5.0 * rij);
        const double drho1ds1 = b1 * rhoa1j * arg1i1, drho1ds2 = b1 * rhoa1i * arg1j1;
        const double drho2ds1 = b2 * rhoa2j * arg1i2 - 2.0 / 3.0 * di.arho2b * rhoa2j;
        const double drho2ds2 = b2 * rhoa2i * arg1j2 - 2.0 / 3.0 * dj.arho2b * rhoa2i;
        const double drho3ds1 = b3 * rhoa3j * arg1i3 - b3a * rhoa3j * arg3i3;
        const double drho3ds2 = b3 * rhoa3i * arg1j3 - b3a * rhoa3i * arg3j3;
        const double ais = (ed[i].rho0 != 0.0) ? rhoa0j / ed[i].rho0 : 0.0;
        const double ajs = (ed[j].rho0 != 0.0) ? rhoa0i / ed[j].rho0 : 0.0;
        const double dt1ds1b = ais * (t1mj - t1i), dt1ds2b = ajs * (t1mi - t1j);
        const double dt2ds1b = ais * (t2mj - t2i), dt2ds2b = ajs * (t2mi - t2j);
        const double dt3ds1b = ais * (t3mj - t3i), dt3ds2b = ajs * (t3mi - t3j);
        drhods1 =
            ed[i].dgamma1 * drho0ds1 +
            ed[i].dgamma2 * (dt1ds1b * ed[i].rho1 + t1i * drho1ds1 + dt2ds1b * ed[i].rho2 +
                             t2i * drho2ds1 + dt3ds1b * ed[i].rho3 + t3i * drho3ds1) -
            ed[i].dgamma3 * (shp[0] * dt1ds1b + shp[1] * dt2ds1b + shp[2] * dt3ds1b);
        drhods2 =
            ed[j].dgamma1 * drho0ds2 +
            ed[j].dgamma2 * (dt1ds2b * ed[j].rho1 + t1j * drho1ds2 + dt2ds2b * ed[j].rho2 +
                             t2j * drho2ds2 + dt3ds2b * ed[j].rho3 + t3j * drho3ds2) -
            ed[j].dgamma3 * (shp[0] * dt1ds2b + shp[1] * dt2ds2b + shp[2] * dt3ds2b);
      }

      const double dUdrij = phip * sij0 + ed[i].frhop * drhodr1 + ed[j].frhop * drhodr2;
      double dUdsij = 0.0;
      if (screen_active) dUdsij = phi + ed[i].frhop * drhods1 + ed[j].frhop * drhods2;
      double dUdrijm[3];
      for (int mm = 0; mm < 3; ++mm)
        dUdrijm[mm] = ed[i].frhop * drhodrm1[mm] + ed[j].frhop * drhodrm2[mm];

      const double force = dUdrij * recip + dUdsij * dscrfcn_ij;
      const double fm0 = delij[0] * force + dUdrijm[0];
      const double fm1 = delij[1] * force + dUdrijm[1];
      const double fm2 = delij[2] * force + dUdrijm[2];
      // Role A (o == i, the lower endpoint): +fm. Role B (o == j, the higher endpoint): −fm.
      if (is_owned[i]) { wFx[i].add(fm0); wFy[i].add(fm1); wFz[i].add(fm2); }
      if (is_owned[j]) { wFx[j].add(-fm0); wFy[j].add(-fm1); wFz[j].add(-fm2); }

      // The screening 3rd-atom k-loop (fires only for partial 0<sij<1 — dead on binary S). POISON
      // (drop_class>0): drop it entirely — the MB2 enumeration teeth (atom-k's force collapses ⇒
      // diverges from the oracle).
      if (std::fabs(sij0) < 1e-20 || std::fabs(sij0 - 1.0) < 1e-20) continue;
      if (drop_class > 0) continue;
      const double delc = p.Cmax - p.Cmin, rbound = rij2 * p.ebound;
      for (size_t kn = 0; kn < nbr[i].size(); ++kn) {
        const auto& ek = nbr[i][kn];
        const int k = ek.j;
        if (k == j) continue;
        const double dxik = ek.dx, dyik = ek.dy, dzik = ek.dz;
        const double dxjk = ek.dx - ej.dx, dyjk = ek.dy - ej.dy, dzjk = ek.dz - ej.dz;
        const double rjk2 = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
        if (rjk2 > rbound) continue;
        const double rik2 = dxik * dxik + dyik * dyik + dzik * dzik;
        if (rik2 > rbound) continue;
        const double xik = rik2 / rij2, xjk = rjk2 / rij2;
        const double aa = 1.0 - (xik - xjk) * (xik - xjk);
        if (std::fabs(aa) < 1e-20) continue;
        double cikj = (2.0 * (xik + xjk) + aa - 2.0) / aa;
        if (!(cikj >= p.Cmin && cikj <= p.Cmax)) continue;
        cikj = (cikj - p.Cmin) / delc;
        double dfc;
        const double sikj = meam_detail::dfcut(cikj, dfc);
        double dCikj1, dCikj2;
        meam_detail::dCfunc2(rij2, rik2, rjk2, dCikj1, dCikj2);
        const double aw = sij0 / delc * dfc / sikj;
        const double dsij1 = aw * dCikj1, dsij2 = aw * dCikj2;
        if (std::fabs(dsij1) < 1e-20 && std::fabs(dsij2) < 1e-20) continue;
        const double force1 = dUdsij * dsij1, force2 = dUdsij * dsij2;
        // Role A (o == i): +force1·d_ik. Role B (o == j): +force2·d_jk. Role C (o == k): the
        // −(force1·d_ik + force2·d_jk) third-atom write — the needs_transpose non-symmetric write.
        if (is_owned[i]) { wFx[i].add(force1 * dxik); wFy[i].add(force1 * dyik); wFz[i].add(force1 * dzik); }
        if (is_owned[j]) { wFx[j].add(force2 * dxjk); wFy[j].add(force2 * dyjk); wFz[j].add(force2 * dzjk); }
        if (is_owned[k]) {
          wFx[k].add(-(force1 * dxik + force2 * dxjk));
          wFy[k].add(-(force1 * dyik + force2 * dyjk));
          wFz[k].add(-(force1 * dzik + force2 * dzjk));
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
