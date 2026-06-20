#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"  // PairGeom (single source of min-image/cutoff)
#include "tdmd/potentials/many_body.hpp"

// M6 / MEAM-ladder Me1 — the FIRST Modified-Embedded-Atom-Method rung: single-element Si
// ENERGY (the ~470-line analytic force is Me2). Design of record: /tmp/meam_design.txt.
// [ENG] (not in the 2007 dissertation, like EAM/SW/Tersoff). Arc: SW→Tersoff→MEAM.
//
// THE ESCALATION (the load-bearing point): SW's all-triplets oracle shares only the
// ENUMERATION with the checked path; Tersoff's FD-of-energy still used only value helpers.
// MEAM's energy ITSELF runs the screening + 4-density value algebra the future force will
// differentiate ⇒ the LAMMPS golden is the ONLY value-algebra witness in Me1. Energy-first
// locks that algebra before any derivative is written.
//
// All leaf math (fcut/dfcut/G_gam/embedding/erose/get_shpfcn DIA) + getscreen (VALUE path)
// + calc_rho1 + dens_final + the φ-from-spline (interpolate_meam) are replicated CHARACTER-
// FOR-CHARACTER from LAMMPS MEAM/* (stable 22Jul2025), keeping ONLY the single-element,
// msmeam=0, ialloy=0, mix_ref_t=0, bkgd_dyn=0, nn2=0, ibar=1, lat=DIA branch. Two silent
// traps are replicated exactly (or the energy is wrong): (1) augt1 t1_eff = t1 + 0.6·t3 =
// 2.82; (2) ebound = Cmax²/(4(Cmax−1)), NOT a constant. The colinear near-negative-C
// non-rejection (only a≤0 rejects k) is replicated verbatim — see meam_getscreen.
namespace tdmd::potentials {

// ----------------------------------------------------------------------------------------
// Leaf math — TDMD_HOST_DEVICE for the future GPU rung (Me5), mirroring SW/Tersoff.
// ----------------------------------------------------------------------------------------
namespace meam_detail {

// LAMMPS meam.h::fcut — the screening smoothstep (1 − (1−ξ)⁴)².
TDMD_HOST_DEVICE inline double fcut(double xi) {
  if (xi >= 1.0) return 1.0;
  if (xi <= 0.0) return 0.0;
  double a = 1.0 - xi;
  a *= a;
  a *= a;
  a = 1.0 - a;
  return a * a;
}

// LAMMPS meam.h::dfcut — value path (returns fcut; dfc the derivative, used for the radial
// taper fcpair = dfcut(rnorm) in getscreen, where we only need the VALUE).
TDMD_HOST_DEVICE inline double dfcut(double xi, double& dfc) {
  if (xi >= 1.0) { dfc = 0.0; return 1.0; }
  if (xi <= 0.0) { dfc = 0.0; return 0.0; }
  double a = 1.0 - xi;
  double a3 = a * a * a;
  double a4 = a * a3;
  double a1m4 = 1.0 - a4;
  dfc = 8.0 * a1m4 * a3;
  return a1m4 * a1m4;
}

// LAMMPS meam_funcs.cpp::G_gam — full switch replicated; ibar=1 reachable (G = exp(γ/2)).
// (A Si fixture cannot distinguish a wrong default branch from a right one — replicate all.)
// fm_exp → std::exp (the SW/Morse CPU↔LAMMPS-is-tolerance precedent; ~1 ulp/op).
TDMD_HOST_DEVICE inline double G_gam(double gamma, int ibar, double gsmooth_factor,
                                     int& errorflag) {
  switch (ibar) {
    case 0:
    case 4: {
      double sp = -gsmooth_factor / (gsmooth_factor + 1.0);
      if (gamma < sp) {
        if (gamma == 0.0) return 0.0;
        double G = 1.0 / (gsmooth_factor + 1.0) * std::pow(sp / gamma, gsmooth_factor);
        return std::sqrt(G);
      }
      return std::sqrt(1.0 + gamma);
    }
    case 1:
      return std::exp(gamma / 2.0);
    case 3:
      return 2.0 / (1.0 + std::exp(-gamma));
    case -5:
      if ((1.0 + gamma) >= 0.0) return std::sqrt(1.0 + gamma);
      return -std::sqrt(-1.0 - gamma);
  }
  errorflag = 1;
  return 0.0;
}

// LAMMPS meam_funcs.cpp::dG_gam — G(γ) AND dG/dγ; ibar=1 reachable (G = exp(γ/2), dG = G/2).
// Me2 force needs the derivative; Me1 G_gam (value-only) is left untouched. Replicate the full
// switch (a Si fixture cannot distinguish a wrong default branch — the deferred-branch discipline).
TDMD_HOST_DEVICE inline double dG_gam(double gamma, int ibar, double gsmooth_factor, double& dG) {
  switch (ibar) {
    case 0:
    case 4: {
      double sp = -gsmooth_factor / (gsmooth_factor + 1.0);
      if (gamma < sp) {
        double G = 1.0 / (gsmooth_factor + 1.0) * std::pow(sp / gamma, gsmooth_factor);
        G = std::sqrt(G);
        dG = -gsmooth_factor * G / (2.0 * gamma);
        return G;
      }
      double G = std::sqrt(1.0 + gamma);
      dG = 1.0 / (2.0 * G);
      return G;
    }
    case 1: {
      double G = std::exp(gamma / 2.0);
      dG = G / 2.0;
      return G;
    }
    case 3: {
      double G = 2.0 / (1.0 + std::exp(-gamma));
      dG = G * (2.0 - G) / 2.0;
      return G;
    }
    case -5:
      if ((1.0 + gamma) >= 0.0) {
        double G = std::sqrt(1.0 + gamma);
        dG = 1.0 / (2.0 * G);
        return G;
      } else {
        double G = -std::sqrt(-1.0 - gamma);
        dG = -1.0 / (2.0 * G);
        return G;
      }
  }
  dG = 1.0;
  return 0.0;
}

// LAMMPS meam.h::dCfunc — ∂C_ikj/∂(rij²). Used by getscreen's dscrfcn (the RADIAL screening
// derivative). DISTINCT from dCfunc2 (the rik²/rjk² derivatives in the force k-loop).
TDMD_HOST_DEVICE inline double dCfunc(double rij2, double rik2, double rjk2) {
  const double rij4 = rij2 * rij2;
  const double a = rik2 - rjk2;
  const double b = rik2 + rjk2;
  const double asq = a * a;
  double denom = rij4 - asq;
  denom = denom * denom;
  return -4.0 * (-2.0 * rij2 * asq + rij4 * b + asq * b) / denom;
}

// LAMMPS meam.h::dCfunc2 — ∂C_ikj/∂(rik²) and ∂C_ikj/∂(rjk²). The force k-loop's screening
// 3rd-atom derivative (the dsij1/dsij2 weights that scatter to f_i/f_j/f_k).
TDMD_HOST_DEVICE inline void dCfunc2(double rij2, double rik2, double rjk2, double& dCikj1,
                                     double& dCikj2) {
  const double rij4 = rij2 * rij2;
  const double rik4 = rik2 * rik2;
  const double rjk4 = rjk2 * rjk2;
  const double a = rik2 - rjk2;
  double denom = rij4 - a * a;
  denom = denom * denom;
  dCikj1 = 4.0 * rij2 * (rij4 + rik4 + 2.0 * rik2 * rjk2 - 3.0 * rjk4 - 2.0 * rij2 * a) / denom;
  dCikj2 = 4.0 * rij2 * (rij4 - 3.0 * rik4 + 2.0 * rik2 * rjk2 + rjk4 + 2.0 * rij2 * a) / denom;
}

// LAMMPS meam_funcs.cpp::embedding — F = A·Ec·ρ̄·ln(ρ̄); emb_lin_neg=0 ⇒ ρ̄≤0 returns 0.
TDMD_HOST_DEVICE inline double embedding(double A, double Ec, double rhobar, int emb_lin_neg,
                                         double& dF) {
  const double AEc = A * Ec;
  if (rhobar > 0.0) {
    const double lrb = std::log(rhobar);
    dF = AEc * (1.0 + lrb);
    return AEc * rhobar * lrb;
  }
  if (emb_lin_neg == 0) { dF = 0.0; return 0.0; }
  dF = -AEc;
  return -AEc * rhobar;
}

// LAMMPS meam_funcs.cpp::zbl — universal screened-Coulomb, blended into φ at small r. On
// near-equilibrium diamond Si NO real pair samples the blend region (r < re(1−1/α) ≈ 1.87 Å);
// it is replicated only so the spuline knots match LAMMPS character-for-character. A separate
// min_r guard in meam_energy asserts the blended knots stay unread (the Tersoff ters_fc_d
// structurally-untouched-branch methodology).
TDMD_HOST_DEVICE inline double zbl(double r, int z1, int z2) {
  const double c[4] = {0.028171, 0.28022, 0.50986, 0.18175};
  const double d[4] = {0.20162, 0.40290, 0.94229, 3.1998};
  const double azero = 0.4685;
  const double cc = 14.3997;
  const double aa = azero / (std::pow(double(z1), 0.23) + std::pow(double(z2), 0.23));
  double result = 0.0;
  const double x = r / aa;
  for (int i = 0; i <= 3; ++i) result += c[i] * std::exp(-d[i] * x);
  if (r > 0.0) result = result * z1 * z2 / r * cc;
  return result;
}

// LAMMPS meam_funcs.cpp::erose — form=0 else branch (used only in φ setup).
TDMD_HOST_DEVICE inline double erose(double r, double re, double alpha, double Ec, double repuls,
                                     double attrac, int form) {
  double astar, a3;
  double result = 0.0;
  if (r > 0.0) {
    astar = alpha * (r / re - 1.0);
    a3 = 0.0;
    if (astar >= 0.0) a3 = attrac;
    else if (astar < 0.0) a3 = repuls;
    const double as3 = astar * astar * astar;
    if (form == 1)
      result = -Ec * (1.0 + astar + (-attrac + repuls / r) * as3) * std::exp(-astar);
    else if (form == 2)
      result = -Ec * (1.0 + astar + a3 * as3) * std::exp(-astar);
    else
      result = -Ec * (1.0 + astar + a3 * as3 / (r / re)) * std::exp(-astar);
  }
  return result;
}

}  // namespace meam_detail

// ----------------------------------------------------------------------------------------
// MeamParams — hardcoded single-element Si (library.meam 'Si' entry + setup_global defaults)
// + recompute() deriving re, t1_eff (augt1), ebound, rho_ref, and the φ spline (phirar*).
// ----------------------------------------------------------------------------------------
struct MeamParams {
  // --- library.meam 'Si' 'dia' 4 14 28.086 -----------------------------------------------
  double alpha = 4.87;
  double beta0 = 4.8, beta1 = 4.8, beta2 = 4.8, beta3 = 4.8;
  double alat = 5.431;
  double Ec = 4.63;     // esub
  double A = 1.0;       // asub
  double t0 = 1.0, t1 = 3.30, t2 = 5.105, t3 = -0.80;
  double rho0 = 1.0;    // rozero
  int ibar = 1;
  int Z = 4;            // get_Zij(DIA)
  int ielement = 14;    // Si atomic number (for the ZBL blend at small r)
  // --- setup_global defaults (Ni.meam sets only rc, delr; rest are these) -----------------
  double rc = 4.0, delr = 0.1;
  double Cmin = 2.0, Cmax = 2.8;
  double gsmooth = 99.0;
  int augt1 = 1, ialloy = 0, mix_ref_t = 0, bkgd_dyn = 0, emb_lin_neg = 0, erose_form = 0;
  int nn2 = 0, zbl = 1;  // zbl inactive on near-equilibrium diamond (astar > −1)
  // --- shape factors get_shpfcn(DIA) → {0, 0, 32/9} --------------------------------------
  double shp[3] = {0.0, 0.0, 32.0 / 9.0};

  // --- DERIVED in recompute() ------------------------------------------------------------
  double re = 0.0;       // alat·√3/4
  double t1_eff = 0.0;   // augt1: t1 + 0.6·t3 = 2.82
  double ebound = 0.0;   // Cmax²/(4(Cmax−1))
  double rho_ref = 0.0;  // compute_reference_density(DIA) = rho0·Z·Gbar
  // v2D/v3D multiplicities (LAMMPS meam_setup_done.cpp)
  std::array<int, 6> v2D{1, 2, 2, 1, 2, 1};
  std::array<int, 10> v3D{1, 3, 3, 3, 6, 3, 1, 3, 3, 1};
  // vind2D[3][3] / vind3D[3][3][3] — symmetric Voigt index tables (the angular-direction force
  // derivatives drho{2,3}drm read them). Built character-for-character in recompute() from
  // meam_setup_done.cpp:43-60 (the same nested m≤n≤p loop that emits v2D/v3D).
  int vind2D[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  int vind3D[3][3][3] = {};
  // φ spline (interpolate_meam over phi_meam grid) — MUST-FIX-1
  int nr = 1000;
  double dr = 0.0, rdrar = 0.0;
  std::vector<double> phir;                                         // raw φ(r) on the grid
  std::vector<double> phirar, phirar1, phirar2, phirar3, phirar4, phirar5, phirar6;

  MeamParams() { recompute(); }

  // get_densref(DIA) single-element a==b: rho01 = 4·rhoa0, rho31 = (32/9)·rhoa3² (rho11/21 = 0).
  // Returns the reference partial densities of the central atom at pair distance r.
  void densref(double r, double& rho0_, double& rho1_, double& rho2_, double& rho3_) const {
    const double a1 = r / re - 1.0;
    const double rhoa0 = rho0 * std::exp(-beta0 * a1);
    const double rhoa3 = rho0 * std::exp(-beta3 * a1);
    rho0_ = 4.0 * rhoa0;       // DIA: 4 first neighbours
    rho1_ = 0.0;
    rho2_ = 0.0;
    rho3_ = 32.0 / 9.0 * rhoa3 * rhoa3;
  }

  // LAMMPS phi_meam(r, Si, Si) — DIA single-element else branch (φ = (2·Eu − F1 − F2)/Z12).
  double phi_meam(double r) const {
    int errorflag = 0;
    double dF;
    double rho01, rho11, rho21, rho31;
    densref(r, rho01, rho11, rho21, rho31);
    if (rho01 <= 1e-14) return 0.0;
    // get_tavref(DIA): all neighbours of opposite type ⇒ t*1av = t*2av = t*_meam (a==b ⇒ same).
    const double t1av = t1_eff, t2av = t2, t3av = t3;
    // Gam = (t1·rho1 + t2·rho2 + t3·rho3) / rho0² ; rho1=rho2=0 here, rho3 nonzero for DIA.
    double Gam1 = t1av * rho11 + t2av * rho21 + t3av * rho31;
    if (rho01 < 1.0e-14) Gam1 = 0.0; else Gam1 = Gam1 / (rho01 * rho01);
    const double G1 = meam_detail::G_gam(Gam1, ibar, gsmooth, errorflag);
    const double rho_bkgd1 = rho_ref;  // bkgd_dyn=0, mix_ref_t=0
    const double rhobar1 = rho01 / rho_bkgd1 * G1;
    const double rhobar2 = rhobar1;  // a==b ⇒ identical reference background
    const double F1 = meam_detail::embedding(A, Ec, rhobar1, emb_lin_neg, dF);
    const double F2 = meam_detail::embedding(A, Ec, rhobar2, emb_lin_neg, dF);
    const double Eu = meam_detail::erose(r, re, alpha, Ec, /*repuls*/ 0.0, /*attrac*/ 0.0,
                                         erose_form);
    const int Z12 = Z;  // get_Zij(DIA) = 4
    double phi_m = (2.0 * Eu - F1 - F2) / Z12;
    if (std::fabs(r) < 1e-20) phi_m = 0.0;
    return phi_m;
  }

  // LAMMPS interpolate_meam(ind) — the cubic φ spline (phirar1..6) over the phir grid.
  void interpolate() {
    const int nrar = nr;
    const double drar = dr;
    rdrar = 1.0 / drar;
    phirar.assign(nrar, 0.0);
    phirar1.assign(nrar, 0.0);
    phirar2.assign(nrar, 0.0);
    phirar3.assign(nrar, 0.0);
    phirar4.assign(nrar, 0.0);
    phirar5.assign(nrar, 0.0);
    phirar6.assign(nrar, 0.0);
    for (int j = 0; j < nrar; ++j) phirar[j] = phir[j];
    phirar1[0] = phirar[1] - phirar[0];
    phirar1[1] = 0.5 * (phirar[2] - phirar[0]);
    phirar1[nrar - 2] = 0.5 * (phirar[nrar - 1] - phirar[nrar - 3]);
    phirar1[nrar - 1] = 0.0;
    for (int j = 2; j < nrar - 2; ++j)
      phirar1[j] = ((phirar[j - 2] - phirar[j + 2]) + 8.0 * (phirar[j + 1] - phirar[j - 1])) / 12.0;
    for (int j = 0; j < nrar - 1; ++j) {
      phirar2[j] = 3.0 * (phirar[j + 1] - phirar[j]) - 2.0 * phirar1[j] - phirar1[j + 1];
      phirar3[j] = phirar1[j] + phirar1[j + 1] - 2.0 * (phirar[j + 1] - phirar[j]);
    }
    phirar2[nrar - 1] = 0.0;
    phirar3[nrar - 1] = 0.0;
    for (int j = 0; j < nrar; ++j) {
      phirar4[j] = phirar1[j] / drar;
      phirar5[j] = 2.0 * phirar2[j] / drar;
      phirar6[j] = 3.0 * phirar3[j] / drar;
    }
  }

  // LAMMPS meam_force.cpp:106-111 — read φ(rij) from the cubic spline.
  double phi_spline(double rij) const {
    const int nrar = nr;
    double pp = rij * rdrar;
    int kk = static_cast<int>(pp);
    kk = std::min(kk, nrar - 2);
    pp = pp - kk;
    pp = std::min(pp, 1.0);
    return ((phirar3[kk] * pp + phirar2[kk]) * pp + phirar1[kk]) * pp + phirar[kk];
  }

  // LAMMPS meam_force.cpp:112 — read φ'(rij) from the SAME spline grid. phirar4/5/6 are the
  // /drar-scaled coefficients (interpolate() already built them; Me1 never read them). Me2 force.
  double phip_spline(double rij) const {
    const int nrar = nr;
    double pp = rij * rdrar;
    int kk = static_cast<int>(pp);
    kk = std::min(kk, nrar - 2);
    pp = pp - kk;
    pp = std::min(pp, 1.0);
    return (phirar6[kk] * pp + phirar5[kk]) * pp + phirar4[kk];
  }

  void recompute() {
    if (Cmax <= 1.0) throw std::runtime_error("MeamParams: Cmax must be > 1");
    if (nn2 != 0) throw std::runtime_error("MeamParams: nn2!=0 deferred to a later rung");
    if (ialloy != 0) throw std::runtime_error("MeamParams: ialloy!=0 deferred");
    if (mix_ref_t != 0) throw std::runtime_error("MeamParams: mix_ref_t!=0 deferred");
    if (bkgd_dyn != 0) throw std::runtime_error("MeamParams: bkgd_dyn!=0 deferred");
    // the value branches actually exercised + golden-validated for Si are ibar=1, emb_lin_neg=0,
    // erose_form=0; guard them so a careless flip cannot silently select an UNVALIDATED branch of
    // G_gam/embedding/erose (the deferred-branch discipline; acceptance SHOULD-improve).
    if (ibar != 1) throw std::runtime_error("MeamParams: only ibar=1 (Si) is validated");
    if (emb_lin_neg != 0) throw std::runtime_error("MeamParams: emb_lin_neg!=0 branch unvalidated");
    if (erose_form != 0) throw std::runtime_error("MeamParams: erose_form!=0 branch unvalidated");

    // Voigt symmetric index tables (meam_setup_done.cpp:43-60) — used by the Me2 force only.
    {
      int nv2 = 0, nv3 = 0;
      for (int m = 0; m < 3; ++m)
        for (int n = m; n < 3; ++n) {
          vind2D[m][n] = nv2;
          vind2D[n][m] = nv2;
          ++nv2;
          for (int pp = n; pp < 3; ++pp) {
            vind3D[m][n][pp] = nv3;
            vind3D[m][pp][n] = nv3;
            vind3D[n][m][pp] = nv3;
            vind3D[n][pp][m] = nv3;
            vind3D[pp][m][n] = nv3;
            vind3D[pp][n][m] = nv3;
            ++nv3;
          }
        }
    }

    re = alat * std::sqrt(3.0) / 4.0;                  // DIA
    t1_eff = t1 + augt1 * (3.0 / 5.0) * t3;            // augt1 trap (= 2.82 for Si)
    ebound = (Cmax * Cmax) / (4.0 * (Cmax - 1.0));     // MUST-FIX-2 (NOT a constant)
    // residence/min-image sanity: √ebound·rc must stay inside 2·rc (the {2,true} reach).
    if (std::sqrt(ebound) * rc >= 2.0 * rc)
      throw std::runtime_error("MeamParams: √ebound·rc >= 2·rc — screening reach exceeds window");

    // compute_reference_density(DIA): rho_ref = rho0·Z·Gbar, Gbar = G_gam(γ_bar, ibar).
    int errorflag = 0;
    double Gbar;
    if (ibar <= 0) {
      Gbar = 1.0;
    } else {
      const double gam = (t1_eff * shp[0] + t2 * shp[1] + t3 * shp[2]) / (double(Z) * double(Z));
      Gbar = meam_detail::G_gam(gam, ibar, gsmooth, errorflag);
    }
    rho_ref = rho0 * Z * Gbar;

    // Build the φ spline (phi_meam on a 1000-pt grid → interpolate_meam). MUST-FIX-1.
    dr = 1.1 * rc / nr;
    phir.assign(nr, 0.0);
    for (int j = 0; j < nr; ++j) {
      const double r = j * dr;
      phir[j] = phi_meam(r);
      // ZBL blend (compute_pair_meam.cpp:295-304): astar<=−3 ⇒ pure ZBL; −3<astar<−1 ⇒ linear
      // blend. Engages only for r < re(1−1/α) ≈ 1.87 Å. Replicated character-for-character so
      // the spline knots match LAMMPS; the meam_energy min_r guard asserts they stay unread.
      if (zbl == 1) {
        const double astar = alpha * (r / re - 1.0);
        if (astar <= -3.0) {
          phir[j] = meam_detail::zbl(r, ielement, ielement);
        } else if (astar > -3.0 && astar < -1.0) {
          const double frac = meam_detail::fcut(1.0 - (astar + 1.0) / (-3.0 + 1.0));
          const double phizbl = meam_detail::zbl(r, ielement, ielement);
          phir[j] = frac * phir[j] + (1.0 - frac) * phizbl;
        }
      }
    }
    interpolate();
  }
};

// ----------------------------------------------------------------------------------------
// Me5 — the POD φ-SPLINE VIEW (device pointers, no std::vector ⇒ NEVER slices). The Me5 kernel
// reads φ(rij)/φ'(rij) via this view (the spline is transcendental-free — pure Horner + an int
// knot — so phi_spline/phip_spline are even BITWISE under --fmad=false; the K3 tolerance is from
// the exp/log/pow in the density/embedding chain, not here). The host uploads the 7 phirar arrays
// to device pointers; this view carries those + nr + rdrar. Mirrors EamSetflView.
struct MeamParamsView {
  const double *phirar = nullptr, *phirar1 = nullptr, *phirar2 = nullptr, *phirar3 = nullptr;
  const double *phirar4 = nullptr, *phirar5 = nullptr, *phirar6 = nullptr;
  int nr = 0;
  double rdrar = 0.0;

  TDMD_HOST_DEVICE double phi_spline(double rij) const {
    const int nrar = nr;
    double pp = rij * rdrar;
    int kk = static_cast<int>(pp);
    kk = kk < nrar - 2 ? kk : nrar - 2;
    pp = pp - kk;
    pp = pp < 1.0 ? pp : 1.0;
    return ((phirar3[kk] * pp + phirar2[kk]) * pp + phirar1[kk]) * pp + phirar[kk];
  }
  TDMD_HOST_DEVICE double phip_spline(double rij) const {
    const int nrar = nr;
    double pp = rij * rdrar;
    int kk = static_cast<int>(pp);
    kk = kk < nrar - 2 ? kk : nrar - 2;
    pp = pp - kk;
    pp = pp < 1.0 ? pp : 1.0;
    return (phirar6[kk] * pp + phirar5[kk]) * pp + phirar4[kk];
  }
};
static_assert(std::is_trivially_copyable_v<MeamParamsView>,
              "MeamParamsView must be POD — passed by value into the Me5 kernel (no spline slice)");

// ----------------------------------------------------------------------------------------
// Neighbour list (per atom i): min-imaged dx = x_j − x_i (LAMMPS convention) + r. Both the
// half-list (pair energy) and the full k-enumeration (screening) draw from this.
// ----------------------------------------------------------------------------------------
struct MeamNbr { int j; double dx, dy, dz, r2, r; };
template <typename Real>
inline std::vector<std::vector<MeamNbr>> meam_build_nbr(const AtomSoA<Real>& a,
                                                        const PairGeom& geom) {
  std::vector<std::vector<MeamNbr>> nbr(a.n);
  for (int i = 0; i < a.n; ++i)
    for (int j = 0; j < a.n; ++j) {
      if (j == i) continue;
      double dx = a.x[j] - a.x[i], dy = a.y[j] - a.y[i], dz = a.z[j] - a.z[i], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;  // r2 < rc²
      nbr[i].push_back({j, dx, dy, dz, r2, std::sqrt(r2)});
    }
  return nbr;
}

// ----------------------------------------------------------------------------------------
// getscreen (VALUE path) — for neighbour jn of i, S_ij = Π_k S(C_ijk) over all third atoms k
// in i's full neighbour list (k != j), times the radial taper fcpair = dfcut(rnorm). Returns
// sij = scrfcn·fcpair (the screened weight that feeds BOTH the densities and the pair energy).
// CHARACTER-FOR-CHARACTER from meam_dens_init.cpp:154-290 (dscrfcn k-derivative dropped).
// ----------------------------------------------------------------------------------------
struct MeamScreen { double scrfcn, fcpair; };

// Me2 — getscreen WITH the dscrfcn radial screening derivative. Returns scrfcn/fcpair (same as
// the value path) PLUS dscrfcn = ∂(sij)/∂(rij) (partial-only — zero on the diamond's binary S).
// CHARACTER-FOR-CHARACTER from meam_dens_init.cpp:154-289 (the second k-loop + the −coef2 taper).
struct MeamScreenD { double scrfcn, fcpair, dscrfcn; };
inline MeamScreenD meam_getscreen_d(const MeamNbr& ej, const std::vector<MeamNbr>& nbr_i,
                                    const MeamParams& p) {
  const double cutforce = p.rc;
  const double drinv = 1.0 / p.delr;
  const double rij2 = ej.r2;
  const double rij = ej.r;
  const double rbound = p.ebound * rij2;
  const double rnorm = (cutforce - rij) * drinv;
  double sij = 1.0;

  // First pass — the screening product itself (== meam_getscreen).
  for (const auto& ek : nbr_i) {
    if (ek.j == ej.j) continue;
    const double dxjk = ek.dx - ej.dx, dyjk = ek.dy - ej.dy, dzjk = ek.dz - ej.dz;
    const double rjk2 = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
    if (rjk2 > rbound) continue;
    const double rik2 = ek.r2;
    if (rik2 > rbound) continue;
    const double xik = rik2 / rij2;
    const double xjk = rjk2 / rij2;
    const double a = 1.0 - (xik - xjk) * (xik - xjk);
    if (a <= 0.0) continue;
    double cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
    if (cikj >= p.Cmax) continue;
    else if (cikj <= p.Cmin) { sij = 0.0; break; }
    else {
      const double delc = p.Cmax - p.Cmin;
      cikj = (cikj - p.Cmin) / delc;
      sij *= meam_detail::fcut(cikj);
    }
  }

  double dfc;
  const double fc = meam_detail::dfcut(rnorm, dfc);
  const double fcij = fc;
  const double dfcij = dfc * drinv;

  // Second pass — dscrfcn (the radial screening derivative; partial-only).
  double dscrfcn = 0.0;
  const double sfcij = sij * fcij;
  if (std::fabs(sfcij) > 1e-20 && std::fabs(sfcij - 1.0) > 1e-20) {  // !iszero && !isone
    for (const auto& ek : nbr_i) {
      if (ek.j == ej.j) continue;
      const double dxjk = ek.dx - ej.dx, dyjk = ek.dy - ej.dy, dzjk = ek.dz - ej.dz;
      const double rjk2 = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
      if (rjk2 > rbound) continue;
      const double rik2 = ek.r2;
      if (rik2 > rbound) continue;
      const double xik = rik2 / rij2;
      const double xjk = rjk2 / rij2;
      const double a = 1.0 - (xik - xjk) * (xik - xjk);
      if (a <= 0.0) continue;
      double cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
      if (cikj >= p.Cmax) {
        continue;  // (0<cikj<Cmin impossible here — sij would already be 0)
      } else {
        const double delc = p.Cmax - p.Cmin;
        cikj = (cikj - p.Cmin) / delc;
        double dfikj;
        const double sikj = meam_detail::dfcut(cikj, dfikj);
        const double coef1 = dfikj / (delc * sikj);
        const double dCikj = meam_detail::dCfunc(rij2, rik2, rjk2);
        dscrfcn += coef1 * dCikj;
      }
    }
    const double coef1 = sfcij;
    const double coef2 = sij * dfcij / rij;
    dscrfcn = dscrfcn * coef1 - coef2;  // ⭐ the MINUS on coef2 (radial taper); a + is silent on diamond
  }

  return {sij, fcij, dscrfcn};
}

// ----------------------------------------------------------------------------------------
// Me5 — the DEVICE screening core (window arrays, no std::vector). Replicates meam_getscreen_d
// CHARACTER-FOR-CHARACTER over the gathered window: the inner `for(ek : nbr_i)` becomes a scan
// `for(k=0;k<m;++k)` with an in-rc geom.reduce test (k is a screening candidate iff rik2<rc²),
// in ASCENDING WINDOW-SLOT order (= ascending global on the host-sorted window) ⇒ the FP product
// Π_k fcut(C) is summed in the SAME order the CPU vector path walks nbr[c] (also ascending). The
// bond (c,jl) geometry is passed by value (delij = wx[jl]−wx[c], already min-imaged by the caller).
// TWO TRAPS replicated verbatim (Risk #1): (1) `if (a <= 0.0) continue;` is the ONLY negative-C
// reject (a cikj<0 early-reject would silently drop screening-k); (2) `dscrfcn = dscrfcn*coef1 −
// coef2` — the MINUS on the radial taper (a + is silent on the diamond's binary S).
struct MeamScreenCParams {
  double rc, delr, ebound, Cmin, Cmax;
};
TDMD_HOST_DEVICE inline MeamScreenD meam_getscreen_d_device(
    const double* wx, const double* wy, const double* wz, int m, int c, int jl, double djx,
    double djy, double djz, double rij2, double rij, const core::PairGeom& geom,
    const MeamScreenCParams& p) {
  const double cutforce = p.rc;
  const double drinv = 1.0 / p.delr;
  const double rbound = p.ebound * rij2;
  const double rnorm = (cutforce - rij) * drinv;
  const double xc = wx[c], yc = wy[c], zc = wz[c];
  double sij = 1.0;

  // First pass — the screening product itself.
  for (int k = 0; k < m; ++k) {
    if (k == c || k == jl) continue;  // k != c (self) and k != j (the bond's other endpoint)
    double dxik = wx[k] - xc, dyik = wy[k] - yc, dzik = wz[k] - zc, rik2;
    if (!geom.reduce(dxik, dyik, dzik, rik2)) continue;  // k a window neighbour of c (rik2<rc²)
    const double dxjk = dxik - djx, dyjk = dyik - djy, dzjk = dzik - djz;
    const double rjk2 = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
    if (rjk2 > rbound) continue;
    if (rik2 > rbound) continue;
    const double xik = rik2 / rij2;
    const double xjk = rjk2 / rij2;
    const double a = 1.0 - (xik - xjk) * (xik - xjk);
    if (a <= 0.0) continue;  // ⭐ TRAP 1: the ONLY negative-C rejection (ellipse)
    double cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
    if (cikj >= p.Cmax) continue;
    else if (cikj <= p.Cmin) { sij = 0.0; break; }
    else {
      const double delc = p.Cmax - p.Cmin;
      cikj = (cikj - p.Cmin) / delc;
      sij *= meam_detail::fcut(cikj);
    }
  }

  double dfc;
  const double fc = meam_detail::dfcut(rnorm, dfc);
  const double fcij = fc;
  const double dfcij = dfc * drinv;

  // Second pass — dscrfcn (the radial screening derivative; partial-only).
  double dscrfcn = 0.0;
  const double sfcij = sij * fcij;
  if (fabs(sfcij) > 1e-20 && fabs(sfcij - 1.0) > 1e-20) {  // !iszero && !isone
    for (int k = 0; k < m; ++k) {
      if (k == c || k == jl) continue;
      double dxik = wx[k] - xc, dyik = wy[k] - yc, dzik = wz[k] - zc, rik2;
      if (!geom.reduce(dxik, dyik, dzik, rik2)) continue;
      const double dxjk = dxik - djx, dyjk = dyik - djy, dzjk = dzik - djz;
      const double rjk2 = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
      if (rjk2 > rbound) continue;
      if (rik2 > rbound) continue;
      const double xik = rik2 / rij2;
      const double xjk = rjk2 / rij2;
      const double a = 1.0 - (xik - xjk) * (xik - xjk);
      if (a <= 0.0) continue;
      double cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
      if (cikj >= p.Cmax) {
        continue;
      } else {
        const double delc = p.Cmax - p.Cmin;
        cikj = (cikj - p.Cmin) / delc;
        double dfikj;
        const double sikj = meam_detail::dfcut(cikj, dfikj);
        const double coef1 = dfikj / (delc * sikj);
        const double dCikj = meam_detail::dCfunc(rij2, rik2, rjk2);
        dscrfcn += coef1 * dCikj;
      }
    }
    const double coef1 = sfcij;
    const double coef2 = sij * dfcij / rij;
    dscrfcn = dscrfcn * coef1 - coef2;  // ⭐ TRAP 2: the MINUS on coef2 (radial taper)
  }

  return {sij, fcij, dscrfcn};
}

inline MeamScreen meam_getscreen(const MeamNbr& ej, const std::vector<MeamNbr>& nbr_i,
                                 const MeamParams& p) {
  const double cutforce = p.rc;
  const double drinv = 1.0 / p.delr;
  const double rij2 = ej.r2;
  const double rij = ej.r;
  const double rbound = p.ebound * rij2;
  const double rnorm = (cutforce - rij) * drinv;
  double sij = 1.0;

  for (const auto& ek : nbr_i) {
    if (ek.j == ej.j) continue;           // k != j
    // rjk2 from k and j positions (both relative to i): delr_jk = delr_ik − delr_ij.
    const double dxjk = ek.dx - ej.dx, dyjk = ek.dy - ej.dy, dzjk = ek.dz - ej.dz;
    const double rjk2 = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
    if (rjk2 > rbound) continue;
    const double rik2 = ek.r2;
    if (rik2 > rbound) continue;

    const double xik = rik2 / rij2;
    const double xjk = rjk2 / rij2;
    const double a = 1.0 - (xik - xjk) * (xik - xjk);
    if (a <= 0.0) continue;               // the ONLY negative-C rejection (ellipse)

    double cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
    if (cikj >= p.Cmax) continue;         // k doesn't screen
    // NOTE: cikj may be slightly negative for colinear atoms — do NOT reject here
    // (the a>0 test already handled the genuine outside-ellipse cases). SCREENING TRAP.
    else if (cikj <= p.Cmin) { sij = 0.0; break; }  // i–j bond fully screened
    else {
      const double delc = p.Cmax - p.Cmin;
      cikj = (cikj - p.Cmin) / delc;
      sij *= meam_detail::fcut(cikj);
    }
  }

  double dfc;
  const double fcij = meam_detail::dfcut(rnorm, dfc);
  return {sij, fcij};
}

// ----------------------------------------------------------------------------------------
// Per-atom partial-density accumulator (the MEAM analogue of EAM's DensityAccum). For Me1
// these are plain doubles; meam_run_fixed (below) swaps them for FixedAccum for B1/INV-9.
// ----------------------------------------------------------------------------------------
struct MeamDensity {
  double rho0 = 0.0;
  double arho1[3] = {0, 0, 0};
  double arho2[6] = {0, 0, 0, 0, 0, 0};
  double arho2b = 0.0;
  double arho3[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  double arho3b[3] = {0, 0, 0};
  double t_ave[3] = {0, 0, 0};
};

// calc_rho1 contribution of neighbour j (screened weight sij) into atom i's density.
// CHARACTER-FOR-CHARACTER from meam_dens_init.cpp:313-427, single-element / ialloy=0 branch.
// (LAMMPS scatters i→j and j→i in one pass over the half-list; here we GATHER per-owner i
//  over the full neighbour list, so only the i-side accumulation is applied — identical sum.)
inline void meam_calc_rho1(MeamDensity& di, const MeamNbr& ej, double sij, const MeamParams& p) {
  const double rij2 = ej.r2;
  const double rij = ej.r;
  const double aj = rij / p.re - 1.0;
  const double ro0j = p.rho0;
  const double rhoa0j = ro0j * std::exp(-p.beta0 * aj) * sij;
  const double rhoa1j = ro0j * std::exp(-p.beta1 * aj) * sij;
  const double rhoa2j = ro0j * std::exp(-p.beta2 * aj) * sij;
  const double rhoa3j = ro0j * std::exp(-p.beta3 * aj) * sij;

  di.rho0 += rhoa0j;
  di.t_ave[0] += p.t1_eff * rhoa0j;
  di.t_ave[1] += p.t2 * rhoa0j;
  di.t_ave[2] += p.t3 * rhoa0j;
  di.arho2b += rhoa2j;

  const double A1j = rhoa1j / rij;
  const double A2j = rhoa2j / rij2;
  const double A3j = rhoa3j / (rij2 * rij);
  // delij[m] = x_j − x_i (== ej.dx/dy/dz, project min-image already applied).
  const double del[3] = {ej.dx, ej.dy, ej.dz};
  int nv2 = 0, nv3 = 0;
  for (int m = 0; m < 3; ++m) {
    di.arho1[m] += A1j * del[m];
    di.arho3b[m] += rhoa3j * del[m] / rij;
    for (int n = m; n < 3; ++n) {
      di.arho2[nv2] += A2j * del[m] * del[n];
      ++nv2;
      for (int pp = n; pp < 3; ++pp) {
        di.arho3[nv3] += A3j * del[m] * del[n] * del[pp];
        ++nv3;
      }
    }
  }
}

// dens_final — background density + embedding F(ρ̄) for atom i (single-element, ialloy=0,
// mix_ref_t=0, bkgd_dyn=0 branch of meam_dens_final.cpp:135-247).
inline double meam_dens_final(const MeamDensity& d, const MeamParams& p) {
  int errorflag = 0;
  double rho1 = 0.0, rho2 = -1.0 / 3.0 * d.arho2b * d.arho2b, rho3 = 0.0;
  double t_ave[3] = {d.t_ave[0], d.t_ave[1], d.t_ave[2]};
  for (int m = 0; m < 3; ++m) {
    rho1 += d.arho1[m] * d.arho1[m];
    rho3 -= 3.0 / 5.0 * d.arho3b[m] * d.arho3b[m];
  }
  for (int m = 0; m < 6; ++m) rho2 += p.v2D[m] * d.arho2[m] * d.arho2[m];
  for (int m = 0; m < 10; ++m) rho3 += p.v3D[m] * d.arho3[m] * d.arho3[m];

  if (d.rho0 > 0.0) {
    t_ave[0] /= d.rho0;
    t_ave[1] /= d.rho0;
    t_ave[2] /= d.rho0;
  }
  double gamma = t_ave[0] * rho1 + t_ave[1] * rho2 + t_ave[2] * rho3;
  if (d.rho0 > 0.0) gamma /= (d.rho0 * d.rho0);

  const double G = meam_detail::G_gam(gamma, p.ibar, p.gsmooth, errorflag);
  const double rho = d.rho0 * G;
  const double rho_bkgd = p.rho_ref;  // bkgd_dyn=0, mix_ref_t=0
  const double rhob = rho / rho_bkgd;
  double dF;
  return meam_detail::embedding(p.A, p.Ec, rhob, p.emb_lin_neg, dF);
}

// ----------------------------------------------------------------------------------------
// Me2 — the per-atom prerequisites the FORCE pass reads (meam_dens_final.cpp:135-247 stores
// these into rho0/1/2/3/frhop/gamma/dgamma1/2/3/t_ave; Me1's meam_dens_final discarded them).
// F (the embedding energy) is identical to meam_dens_final's return ⇒ meam_energy stays exact.
// ----------------------------------------------------------------------------------------
struct MeamEmbedDeriv {
  double F = 0.0;         // embedding energy F(ρ̄)  (== meam_dens_final return)
  double frhop = 0.0;     // dF/dρ̄
  double gamma = 0.0;
  double dgamma1 = 0.0;
  double dgamma2 = 0.0;
  double dgamma3 = 0.0;   // mix_ref_t=0 ⇒ always 0 for Si (G-DGAMMA3-DEAD)
  double rho0 = 0.0;      // bare ρ⁰ accumulator
  double rho1 = 0.0;      // Σ arho1²
  double rho2 = 0.0;      // −⅓arho2b² + Σ v2D·arho2²
  double rho3 = 0.0;      // Σ v3D·arho3² − ⅗Σ arho3b²
  double t_ave[3] = {0, 0, 0};
};

// POD-only body (reads d.* + p scalars/Voigt tables, calls only HOST_DEVICE leaf helpers) ⇒
// TDMD_HOST_DEVICE for the Me5 GPU embedding kernel (K2 calls it directly with a stack MeamDensity
// decoded from the int64 lanes). MeamParams is passed by const-ref on the host; on the device the
// kernel passes a POD MeamParamsView-equivalent of the scalars/Voigt (the spline vectors are NOT
// read here). Replicating CHARACTER-FOR-CHARACTER the original — F-NOOP on the CPU.
TDMD_HOST_DEVICE inline MeamEmbedDeriv meam_dens_final_deriv_pod(
    const MeamDensity& d, const double* v2D, const double* v3D, double A, double Ec, int ibar,
    double gsmooth, int emb_lin_neg, double rho_ref) {
  int errorflag = 0;
  MeamEmbedDeriv e;
  double rho1 = 0.0, rho2 = -1.0 / 3.0 * d.arho2b * d.arho2b, rho3 = 0.0;
  double t_ave[3] = {d.t_ave[0], d.t_ave[1], d.t_ave[2]};
  for (int m = 0; m < 3; ++m) {
    rho1 += d.arho1[m] * d.arho1[m];
    rho3 -= 3.0 / 5.0 * d.arho3b[m] * d.arho3b[m];
  }
  for (int m = 0; m < 6; ++m) rho2 += v2D[m] * d.arho2[m] * d.arho2[m];
  for (int m = 0; m < 10; ++m) rho3 += v3D[m] * d.arho3[m] * d.arho3[m];

  if (d.rho0 > 0.0) {  // ialloy=0 branch
    t_ave[0] /= d.rho0;
    t_ave[1] /= d.rho0;
    t_ave[2] /= d.rho0;
  }
  double gamma = t_ave[0] * rho1 + t_ave[1] * rho2 + t_ave[2] * rho3;
  if (d.rho0 > 0.0) gamma /= (d.rho0 * d.rho0);

  const double rho = d.rho0 * meam_detail::G_gam(gamma, ibar, gsmooth, errorflag);
  const double rho_bkgd = rho_ref;  // bkgd_dyn=0, mix_ref_t=0
  const double rhob = rho / rho_bkgd;
  const double denom = 1.0 / rho_bkgd;

  double dG;
  const double G = meam_detail::dG_gam(gamma, ibar, gsmooth, dG);
  e.dgamma1 = (G - 2.0 * dG * gamma) * denom;
  e.dgamma2 = (d.rho0 != 0.0) ? (dG / d.rho0) * denom : 0.0;
  e.dgamma3 = 0.0;  // mix_ref_t=0

  e.F = meam_detail::embedding(A, Ec, rhob, emb_lin_neg, e.frhop);
  e.gamma = gamma;
  e.rho0 = d.rho0;
  e.rho1 = rho1;
  e.rho2 = rho2;
  e.rho3 = rho3;
  e.t_ave[0] = t_ave[0];
  e.t_ave[1] = t_ave[1];
  e.t_ave[2] = t_ave[2];
  return e;
}

inline MeamEmbedDeriv meam_dens_final_deriv(const MeamDensity& d, const MeamParams& p) {
  // Delegate to the HOST_DEVICE POD body (F-NOOP: int v2D[m] → double is exact, the IEEE
  // multiply v2D_d[m]·arho2[m]·arho2[m] is bit-identical to the original p.v2D[m]·… promotion).
  double v2D[6], v3D[10];
  for (int m = 0; m < 6; ++m) v2D[m] = double(p.v2D[m]);
  for (int m = 0; m < 10; ++m) v3D[m] = double(p.v3D[m]);
  return meam_dens_final_deriv_pod(d, v2D, v3D, p.A, p.Ec, p.ibar, p.gsmooth, p.emb_lin_neg,
                                   p.rho_ref);
}
static_assert(std::is_trivially_copyable_v<MeamDensity>,
              "MeamDensity must be POD — the Me5 GPU K2 decodes it from int64 lanes on a stack");
static_assert(std::is_trivially_copyable_v<MeamEmbedDeriv>,
              "MeamEmbedDeriv must be POD — the Me5 GPU writes/reads it per window atom");

// ----------------------------------------------------------------------------------------
// Me5 — the DEVICE bond-force core (POD-only, HOST_DEVICE). Factors the ~200-line Me2 bond body
// (meam_direct_fp64 / meam_run_fixed_force / meam_window_force, all CHARACTER-FOR-CHARACTER the
// same) into ONE single-source helper shared by the CPU force paths and the Me5 GPU K3 kernel.
// Given a directed bond (i,j) with center-i densities/embedding (di,ed_i) and endpoint-j (dj,ed_j)
// + the bond's screening (scrfcn_ij,fcpair,dscrfcn_ij), it returns fm[3] (the symmetric radial +
// angular force the lower-key center writes to i and −1× to j) plus the screening prerequisites
// (sij0, dUdsij, screen_active) the k-loop consumes. The screening 3rd-atom k-loop is a SEPARATE
// helper (meam_screen_k_device) so each receiver role (i / j / k) can scatter its own piece.
// ----------------------------------------------------------------------------------------
struct MeamForceParams {
  double re, rho0, beta0, beta1, beta2, beta3, Cmin, Cmax, ebound;
  double t1_eff, t2, t3, shp[3];
  double v2D[6], v3D[10];
  int vind2D[3][3];
  int vind3D[3][3][3];
};

struct MeamBondForce {
  double fm[3];         // symmetric radial+angular force (write +fm to i, −fm to j)
  double sij0;          // scrfcn·fcpair
  double dUdsij;        // screening force scale (0 unless screen_active)
  bool screen_active;   // |dscrfcn| > 1e-20  ⇒ the k-loop fires
};

// Reads di/dj/ed_i/ed_j + the bond geometry (delij = x_j − x_i, min-imaged) + the cached
// screening (scrfcn_ij,fcpair,dscrfcn_ij). phi/phip are read from the deterministic φ spline by the
// caller (the spline lives in MeamParams on the host / a MeamParamsView on the device). Returns the
// symmetric fm[3] + the screening prerequisites. CHARACTER-FOR-CHARACTER from meam_direct_fp64.
TDMD_HOST_DEVICE inline MeamBondForce meam_bond_force_device(
    const MeamDensity& di, const MeamDensity& dj, const MeamEmbedDeriv& ed_i,
    const MeamEmbedDeriv& ed_j, double dijx, double dijy, double dijz, double rij2, double rij,
    double scrfcn_ij, double fcpair_ij, double dscrfcn_ij, double phi, double phip,
    const MeamForceParams& p) {
  MeamBondForce out;
  const double recip = 1.0 / rij, rij3 = rij * rij2;
  const double delij[3] = {dijx, dijy, dijz};
  const double sij0 = scrfcn_ij * fcpair_ij;
  out.sij0 = sij0;

  const double invre = 1.0 / p.re, ai = rij * invre - 1.0, ro0 = p.rho0;
  const double rhoa0j = ro0 * exp(-p.beta0 * ai), drhoa0j = -p.beta0 * invre * rhoa0j;
  const double rhoa1j = ro0 * exp(-p.beta1 * ai), drhoa1j = -p.beta1 * invre * rhoa1j;
  const double rhoa2j = ro0 * exp(-p.beta2 * ai), drhoa2j = -p.beta2 * invre * rhoa2j;
  const double rhoa3j = ro0 * exp(-p.beta3 * ai), drhoa3j = -p.beta3 * invre * rhoa3j;
  const double rhoa0i = rhoa0j, drhoa0i = drhoa0j, rhoa1i = rhoa1j, drhoa1i = drhoa1j;
  const double rhoa2i = rhoa2j, drhoa2i = drhoa2j, rhoa3i = rhoa3j, drhoa3i = drhoa3j;
  const double t1mi = p.t1_eff, t2mi = p.t2, t3mi = p.t3;
  const double t1mj = p.t1_eff, t2mj = p.t2, t3mj = p.t3;

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
  for (int m = 0; m < 3; ++m) {
    drho1drm1[m] = a1 * rhoa1j * di.arho1[m];
    drho1drm2[m] = -a1 * rhoa1i * dj.arho1[m];
  }
  double a2 = 2.0 * sij0 / rij2;
  const double drho2dr1 =
      a2 * (drhoa2j - 2.0 * rhoa2j / rij) * arg1i2 - 2.0 / 3.0 * di.arho2b * drhoa2j * sij0;
  const double drho2dr2 =
      a2 * (drhoa2i - 2.0 * rhoa2i / rij) * arg1j2 - 2.0 / 3.0 * dj.arho2b * drhoa2i * sij0;
  a2 = 4.0 * sij0 / rij2;
  double drho2drm1[3], drho2drm2[3];
  for (int m = 0; m < 3; ++m) {
    double s1 = 0.0, s2 = 0.0;
    for (int n = 0; n < 3; ++n) {
      s1 += di.arho2[p.vind2D[m][n]] * delij[n];
      s2 -= dj.arho2[p.vind2D[m][n]] * delij[n];
    }
    drho2drm1[m] = a2 * rhoa2j * s1;
    drho2drm2[m] = -a2 * rhoa2i * s2;
  }
  double a3 = 2.0 * sij0 / rij3, a3a = 6.0 / 5.0 * sij0 / rij;
  const double drho3dr1 =
      a3 * (drhoa3j - 3.0 * rhoa3j / rij) * arg1i3 - a3a * (drhoa3j - rhoa3j / rij) * arg3i3;
  const double drho3dr2 =
      a3 * (drhoa3i - 3.0 * rhoa3i / rij) * arg1j3 - a3a * (drhoa3i - rhoa3i / rij) * arg3j3;
  a3 = 6.0 * sij0 / rij3;
  a3a = 6.0 * sij0 / (5.0 * rij);
  double drho3drm1[3], drho3drm2[3];
  for (int m = 0; m < 3; ++m) {
    double s1 = 0.0, s2 = 0.0;
    int nv2 = 0;
    for (int n = 0; n < 3; ++n)
      for (int pp = n; pp < 3; ++pp) {
        const double arg = delij[n] * delij[pp] * p.v2D[nv2];
        s1 += di.arho3[p.vind3D[m][n][pp]] * arg;
        s2 += dj.arho3[p.vind3D[m][n][pp]] * arg;
        ++nv2;
      }
    drho3drm1[m] = (a3 * s1 - a3a * di.arho3b[m]) * rhoa3j;
    drho3drm2[m] = (-a3 * s2 + a3a * dj.arho3b[m]) * rhoa3i;
  }

  const double t1i = ed_i.t_ave[0], t2i = ed_i.t_ave[1], t3i = ed_i.t_ave[2];
  const double t1j = ed_j.t_ave[0], t2j = ed_j.t_ave[1], t3j = ed_j.t_ave[2];
  const double aif = (ed_i.rho0 != 0.0) ? drhoa0j * sij0 / ed_i.rho0 : 0.0;
  const double ajf = (ed_j.rho0 != 0.0) ? drhoa0i * sij0 / ed_j.rho0 : 0.0;
  const double dt1dr1 = aif * (t1mj - t1i), dt1dr2 = ajf * (t1mi - t1j);
  const double dt2dr1 = aif * (t2mj - t2i), dt2dr2 = ajf * (t2mi - t2j);
  const double dt3dr1 = aif * (t3mj - t3i), dt3dr2 = ajf * (t3mi - t3j);
  const double* shp = p.shp;

  const double drhodr1 =
      ed_i.dgamma1 * drho0dr1 +
      ed_i.dgamma2 * (dt1dr1 * ed_i.rho1 + t1i * drho1dr1 + dt2dr1 * ed_i.rho2 +
                      t2i * drho2dr1 + dt3dr1 * ed_i.rho3 + t3i * drho3dr1) -
      ed_i.dgamma3 * (shp[0] * dt1dr1 + shp[1] * dt2dr1 + shp[2] * dt3dr1);
  const double drhodr2 =
      ed_j.dgamma1 * drho0dr2 +
      ed_j.dgamma2 * (dt1dr2 * ed_j.rho1 + t1j * drho1dr2 + dt2dr2 * ed_j.rho2 +
                      t2j * drho2dr2 + dt3dr2 * ed_j.rho3 + t3j * drho3dr2) -
      ed_j.dgamma3 * (shp[0] * dt1dr2 + shp[1] * dt2dr2 + shp[2] * dt3dr2);
  double drhodrm1[3], drhodrm2[3];
  for (int m = 0; m < 3; ++m) {
    drhodrm1[m] = ed_i.dgamma2 *
                  (t1i * drho1drm1[m] + t2i * drho2drm1[m] + t3i * drho3drm1[m]);
    drhodrm2[m] = ed_j.dgamma2 *
                  (t1j * drho1drm2[m] + t2j * drho2drm2[m] + t3j * drho3drm2[m]);
  }

  double drhods1 = 0.0, drhods2 = 0.0;
  const bool screen_active = fabs(dscrfcn_ij) > 1e-20;
  if (screen_active) {
    const double drho0ds1 = rhoa0j, drho0ds2 = rhoa0i;
    const double b1 = 2.0 / rij, b2 = 2.0 / rij2, b3 = 2.0 / rij3, b3a = 6.0 / (5.0 * rij);
    const double drho1ds1 = b1 * rhoa1j * arg1i1, drho1ds2 = b1 * rhoa1i * arg1j1;
    const double drho2ds1 = b2 * rhoa2j * arg1i2 - 2.0 / 3.0 * di.arho2b * rhoa2j;
    const double drho2ds2 = b2 * rhoa2i * arg1j2 - 2.0 / 3.0 * dj.arho2b * rhoa2i;
    const double drho3ds1 = b3 * rhoa3j * arg1i3 - b3a * rhoa3j * arg3i3;
    const double drho3ds2 = b3 * rhoa3i * arg1j3 - b3a * rhoa3i * arg3j3;
    const double ais = (ed_i.rho0 != 0.0) ? rhoa0j / ed_i.rho0 : 0.0;
    const double ajs = (ed_j.rho0 != 0.0) ? rhoa0i / ed_j.rho0 : 0.0;
    const double dt1ds1b = ais * (t1mj - t1i), dt1ds2b = ajs * (t1mi - t1j);
    const double dt2ds1b = ais * (t2mj - t2i), dt2ds2b = ajs * (t2mi - t2j);
    const double dt3ds1b = ais * (t3mj - t3i), dt3ds2b = ajs * (t3mi - t3j);
    drhods1 =
        ed_i.dgamma1 * drho0ds1 +
        ed_i.dgamma2 * (dt1ds1b * ed_i.rho1 + t1i * drho1ds1 + dt2ds1b * ed_i.rho2 +
                        t2i * drho2ds1 + dt3ds1b * ed_i.rho3 + t3i * drho3ds1) -
        ed_i.dgamma3 * (shp[0] * dt1ds1b + shp[1] * dt2ds1b + shp[2] * dt3ds1b);
    drhods2 =
        ed_j.dgamma1 * drho0ds2 +
        ed_j.dgamma2 * (dt1ds2b * ed_j.rho1 + t1j * drho1ds2 + dt2ds2b * ed_j.rho2 +
                        t2j * drho2ds2 + dt3ds2b * ed_j.rho3 + t3j * drho3ds2) -
        ed_j.dgamma3 * (shp[0] * dt1ds2b + shp[1] * dt2ds2b + shp[2] * dt3ds2b);
  }

  const double dUdrij = phip * sij0 + ed_i.frhop * drhodr1 + ed_j.frhop * drhodr2;
  double dUdsij = 0.0;
  if (screen_active) dUdsij = phi + ed_i.frhop * drhods1 + ed_j.frhop * drhods2;
  double dUdrijm[3];
  for (int m = 0; m < 3; ++m)
    dUdrijm[m] = ed_i.frhop * drhodrm1[m] + ed_j.frhop * drhodrm2[m];

  const double force = dUdrij * recip + dUdsij * dscrfcn_ij;
  out.fm[0] = delij[0] * force + dUdrijm[0];
  out.fm[1] = delij[1] * force + dUdrijm[1];
  out.fm[2] = delij[2] * force + dUdrijm[2];
  out.dUdsij = dUdsij;
  out.screen_active = screen_active;
  return out;
}

// The screening 3rd-atom k-contribution for ONE candidate k (relative to center i): given the
// bond (i,j) context (sij0, dUdsij) + the i-frame vectors d_ik (= x_k − x_i) and d_ij, returns
// whether k screens this bond (active) and, if so, force1·d_ik (→ i / −k) and force2·d_jk (→ j /
// −k). CHARACTER-FOR-CHARACTER from meam_run_fixed_force's k-loop body. The caller supplies d_ik
// (the i-frame k vector) + rik2; d_jk = d_ik − d_ij. drop_class>0 is the POISON (caller drops the
// whole k-loop).
struct MeamScreenK {
  bool active;
  double force1;        // → f_i += force1·d_ik ; → f_k −= force1·d_ik
  double force2;        // → f_j += force2·d_jk ; → f_k −= force2·d_jk
  double dik[3], djk[3];
};
TDMD_HOST_DEVICE inline MeamScreenK meam_screen_k_device(
    double dikx, double diky, double dikz, double rik2, double dijx, double dijy, double dijz,
    double rij2, double sij0, double dUdsij, const MeamForceParams& p) {
  MeamScreenK r;
  r.active = false;
  const double delc = p.Cmax - p.Cmin, rbound = rij2 * p.ebound;
  const double dxjk = dikx - dijx, dyjk = diky - dijy, dzjk = dikz - dijz;
  const double rjk2 = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
  if (rjk2 > rbound) return r;
  if (rik2 > rbound) return r;
  const double xik = rik2 / rij2, xjk = rjk2 / rij2;
  const double aa = 1.0 - (xik - xjk) * (xik - xjk);
  if (fabs(aa) < 1e-20) return r;  // iszero(a)
  double cikj = (2.0 * (xik + xjk) + aa - 2.0) / aa;
  if (!(cikj >= p.Cmin && cikj <= p.Cmax)) return r;
  cikj = (cikj - p.Cmin) / delc;
  double dfc;
  const double sikj = meam_detail::dfcut(cikj, dfc);
  double dCikj1, dCikj2;
  meam_detail::dCfunc2(rij2, rik2, rjk2, dCikj1, dCikj2);
  const double aw = sij0 / delc * dfc / sikj;
  const double dsij1 = aw * dCikj1, dsij2 = aw * dCikj2;
  if (fabs(dsij1) < 1e-20 && fabs(dsij2) < 1e-20) return r;
  r.active = true;
  r.force1 = dUdsij * dsij1;
  r.force2 = dUdsij * dsij2;
  r.dik[0] = dikx; r.dik[1] = diky; r.dik[2] = dikz;
  r.djk[0] = dxjk; r.djk[1] = dyjk; r.djk[2] = dzjk;
  return r;
}

// Build the POD MeamForceParams from a MeamParams (the int Voigt mults → double, exact).
inline MeamForceParams meam_force_params(const MeamParams& p) {
  MeamForceParams fp;
  fp.re = p.re; fp.rho0 = p.rho0;
  fp.beta0 = p.beta0; fp.beta1 = p.beta1; fp.beta2 = p.beta2; fp.beta3 = p.beta3;
  fp.Cmin = p.Cmin; fp.Cmax = p.Cmax; fp.ebound = p.ebound;
  fp.t1_eff = p.t1_eff; fp.t2 = p.t2; fp.t3 = p.t3;
  for (int m = 0; m < 3; ++m) fp.shp[m] = p.shp[m];
  for (int m = 0; m < 6; ++m) fp.v2D[m] = double(p.v2D[m]);
  for (int m = 0; m < 10; ++m) fp.v3D[m] = double(p.v3D[m]);
  for (int m = 0; m < 3; ++m)
    for (int n = 0; n < 3; ++n) fp.vind2D[m][n] = p.vind2D[m][n];
  for (int m = 0; m < 3; ++m)
    for (int n = 0; n < 3; ++n)
      for (int q = 0; q < 3; ++q) fp.vind3D[m][n][q] = p.vind3D[m][n][q];
  return fp;
}
static_assert(std::is_trivially_copyable_v<MeamForceParams>,
              "MeamForceParams is passed by value into the Me5 kernel");

// ----------------------------------------------------------------------------------------
// The energy driver: E = Σ_i F(ρ̄_i) + Σ_{pairs} φ(r_ij)·S_ij  (LAMMPS adds φ·sij once per
// directed half-list neighbour ⇒ once per undirected pair; the per-atom ½ is internal).
// ----------------------------------------------------------------------------------------
struct MeamAccum {
  double pe = 0.0;
  double pe_embed = 0.0;
  double pe_pair = 0.0;
  double min_r2 = 1e300;
  long n_screened_partial = 0;  // pairs with 0 < S_ij < 1 (the G-VACUITY counter)
  long n_screened_zero = 0;     // pairs killed by screening (S_ij == 0)
};

// poison: if true, the screening k-product is SKIPPED (S ≡ 1) — the G-SCREEN-POISON
// discriminator. On the active fixture this must DIVERGE from the screened energy and
// disagree with the golden, proving the golden exercises screening.
template <typename Real>
inline MeamAccum meam_energy(const AtomSoA<Real>& a, const PairGeom& geom, const MeamParams& p,
                             bool poison = false) {
  const auto nbr = meam_build_nbr(a, geom);
  std::vector<MeamDensity> dens(a.n);
  MeamAccum acc;

  // Pass 1 — screening + partial densities (per-owner gather over the full neighbour list).
  for (int i = 0; i < a.n; ++i) {
    for (const auto& ej : nbr[i]) {
      acc.min_r2 = std::min(acc.min_r2, ej.r2);
      double sij;
      if (poison) {
        double dfc;
        const double rnorm = (p.rc - ej.r) / p.delr;
        sij = meam_detail::dfcut(rnorm, dfc);  // S≡1, radial taper only
      } else {
        const MeamScreen s = meam_getscreen(ej, nbr[i], p);
        sij = s.scrfcn * s.fcpair;
        if (i < ej.j) {  // count each undirected pair once
          if (s.scrfcn == 0.0) ++acc.n_screened_zero;
          else if (s.scrfcn > 0.0 && s.scrfcn < 1.0) ++acc.n_screened_partial;
        }
      }
      if (std::fabs(sij) < 1e-20) continue;  // iszero(scrfcn) ⇒ no density contribution
      meam_calc_rho1(dens[i], ej, sij, p);
    }
  }

  // Pass 2 — embedding F(ρ̄_i).
  for (int i = 0; i < a.n; ++i) {
    const double F = meam_dens_final(dens[i], p);
    acc.pe_embed += F;
  }

  // Pass 3 — pair energy Σ_{i<j} φ(r_ij)·S_ij. Recompute the screened weight for each
  // undirected pair (LAMMPS uses the i-owned scrfcn[jn]; screening is symmetric in i↔j so
  // the i<j owner gives the same S_ij).
  for (int i = 0; i < a.n; ++i) {
    for (const auto& ej : nbr[i]) {
      if (i >= ej.j) continue;  // each pair once
      double sij;
      if (poison) {
        double dfc;
        const double rnorm = (p.rc - ej.r) / p.delr;
        sij = meam_detail::dfcut(rnorm, dfc);
      } else {
        const MeamScreen s = meam_getscreen(ej, nbr[i], p);
        sij = s.scrfcn * s.fcpair;
      }
      if (std::fabs(sij) < 1e-20) continue;
      const double phi = p.phi_spline(ej.r);
      acc.pe_pair += phi * sij;
    }
  }

  // ZBL structural-untouched guard: assert the shortest real bond is above the blend boundary
  // r_zbl = re(1 − 1/α) (astar = −1). If a pair sampled the blend region the spline read would
  // hit ZBL-polluted knots and this energy would be a different (un-validated) potential.
  const double r_zbl = p.re * (1.0 - 1.0 / p.alpha);
  if (std::sqrt(acc.min_r2) <= r_zbl)
    throw std::runtime_error("meam_energy: a pair is inside the ZBL blend region — deferred");

  acc.pe = acc.pe_embed + acc.pe_pair;
  return acc;
}

// ----------------------------------------------------------------------------------------
// Me2 — THE ANALYTIC FORCE (FP64 SCATTER oracle). Mirrors meam_force.cpp over all centers i:
//   • per directed half-list bond i→j (owner i<j): the embedding+pair force, symmetric write
//     f[i]+=forcem, f[j]-=forcem (q(j)=−q(i)).
//   • the screening k-loop (FULL list, fires only for partial 0<sij<1 — dead on the diamond):
//     the dscrfcn/dCfunc2/k-scatter to f[i],f[j],f[k] (the needs_transpose non-symmetric write).
// Reads per-atom prereqs from a preceding dens pass (densities + MeamEmbedDeriv). The returned
// MeamAccum.pe == meam_energy (the FD-of-energy comparand). drop_class>0 = POISON (skip the
// screening k-term whose center-straddle matches — the MB2 teeth). CONTRACT: tolerance <1e-9.
// ----------------------------------------------------------------------------------------
template <typename Real>
inline MeamAccum meam_direct_fp64(AtomSoA<Real>& a, const PairGeom& geom, const MeamParams& p,
                                  bool with_forces = true, int drop_class = 0) {
  const auto nbr = meam_build_nbr(a, geom);
  std::vector<MeamDensity> dens(a.n);
  MeamAccum acc;

  // -------- Pass 1: screening + partial densities (gather per owner over the full list) -------
  // Cache the i-owned screening (scrfcn,fcpair,dscrfcn) for every directed bond i→j; the force
  // pass + the k-loop read scrfcn[i][jn] = the i-centred radial screening derivative.
  std::vector<std::vector<MeamScreenD>> scr(a.n);
  for (int i = 0; i < a.n; ++i) {
    scr[i].resize(nbr[i].size());
    for (size_t jn = 0; jn < nbr[i].size(); ++jn) {
      const auto& ej = nbr[i][jn];
      acc.min_r2 = std::min(acc.min_r2, ej.r2);
      const MeamScreenD s = meam_getscreen_d(ej, nbr[i], p);
      scr[i][jn] = s;
      const double sij = s.scrfcn * s.fcpair;
      if (i < ej.j) {
        if (s.scrfcn == 0.0) ++acc.n_screened_zero;
        else if (s.scrfcn > 0.0 && s.scrfcn < 1.0) ++acc.n_screened_partial;
      }
      if (std::fabs(sij) < 1e-20) continue;
      meam_calc_rho1(dens[i], ej, sij, p);
    }
  }

  // -------- Pass 2: per-atom embedding F + the stored derivatives ------------------------------
  std::vector<MeamEmbedDeriv> ed(a.n);
  for (int i = 0; i < a.n; ++i) {
    ed[i] = meam_dens_final_deriv(dens[i], p);
    acc.pe_embed += ed[i].F;
  }

  // -------- Pass 3: the FORCE (per directed half-list bond, owner i<j) -------------------------
  for (int i = 0; i < a.n; ++i) {
    for (size_t jn = 0; jn < nbr[i].size(); ++jn) {
      const auto& ej = nbr[i][jn];
      const int j = ej.j;
      const double scrfcn_ij = scr[i][jn].scrfcn;
      const double dscrfcn_ij = scr[i][jn].dscrfcn;
      if (std::fabs(scrfcn_ij) < 1e-20) continue;          // iszero(scrfcn) ⇒ skip bond
      const double sij0 = scrfcn_ij * scr[i][jn].fcpair;   // sij == scrfcn·fcpair
      const double rij2 = ej.r2, rij = ej.r;
      const double recip = 1.0 / rij;
      const double rij3 = rij * rij2;
      // delij[m] = x_j − x_i (LAMMPS convention; == ej.dx/dy/dz min-imaged)
      const double delij[3] = {ej.dx, ej.dy, ej.dz};

      // The pair energy contribution belongs to this bond once (LAMMPS adds φ·sij once per
      // directed neighbour ⇒ once per undirected pair; we own it at i<j).
      const double phi = p.phi_spline(rij);
      const double phip = p.phip_spline(rij);
      if (i < j) acc.pe_pair += phi * sij0;

      // Only the owner (i<j) writes the symmetric pair force + runs the k-loop (each pair once).
      if (i >= j) continue;
      if (!with_forces) continue;

      // ---- pair densities + radial derivatives (single Si: elti==eltj ⇒ j-copy = i-form) ----
      const double invre = 1.0 / p.re;
      const double ai = rij * invre - 1.0;
      const double ro0 = p.rho0;
      const double rhoa0j = ro0 * std::exp(-p.beta0 * ai), drhoa0j = -p.beta0 * invre * rhoa0j;
      const double rhoa1j = ro0 * std::exp(-p.beta1 * ai), drhoa1j = -p.beta1 * invre * rhoa1j;
      const double rhoa2j = ro0 * std::exp(-p.beta2 * ai), drhoa2j = -p.beta2 * invre * rhoa2j;
      const double rhoa3j = ro0 * std::exp(-p.beta3 * ai), drhoa3j = -p.beta3 * invre * rhoa3j;
      const double rhoa0i = rhoa0j, drhoa0i = drhoa0j;     // elti==eltj
      const double rhoa1i = rhoa1j, drhoa1i = drhoa1j;
      const double rhoa2i = rhoa2j, drhoa2i = drhoa2j;
      const double rhoa3i = rhoa3j, drhoa3i = drhoa3j;
      // (ialloy=0 ⇒ NO rhoa*·=t* multiply)

      const double t1mi = p.t1_eff, t2mi = p.t2, t3mi = p.t3;  // augmented t1 (=2.82), NOT 3.30
      const double t1mj = p.t1_eff, t2mj = p.t2, t3mj = p.t3;

      const MeamDensity& di = dens[i];
      const MeamDensity& dj = dens[j];

      // directional contractions (meam_force.cpp:227-244)
      double arg1i1 = 0, arg1j1 = 0, arg1i2 = 0, arg1j2 = 0;
      double arg1i3 = 0, arg1j3 = 0, arg3i3 = 0, arg3j3 = 0;
      {
        int nv2 = 0, nv3 = 0;
        for (int n = 0; n < 3; ++n) {
          for (int pp = n; pp < 3; ++pp) {
            for (int q = pp; q < 3; ++q) {
              const double arg = delij[n] * delij[pp] * delij[q] * p.v3D[nv3];
              arg1i3 += di.arho3[nv3] * arg;
              arg1j3 -= dj.arho3[nv3] * arg;
              ++nv3;
            }
            const double arg = delij[n] * delij[pp] * p.v2D[nv2];
            arg1i2 += di.arho2[nv2] * arg;
            arg1j2 += dj.arho2[nv2] * arg;
            ++nv2;
          }
          arg1i1 += di.arho1[n] * delij[n];
          arg1j1 -= dj.arho1[n] * delij[n];
          arg3i3 += di.arho3b[n] * delij[n];
          arg3j3 -= dj.arho3b[n] * delij[n];
        }
      }

      // rho0 (meam_force.cpp:280)
      const double drho0dr1 = drhoa0j * sij0;
      const double drho0dr2 = drhoa0i * sij0;

      // rho1 (:284-291)
      double a1 = 2.0 * sij0 / rij;
      const double drho1dr1 = a1 * (drhoa1j - rhoa1j / rij) * arg1i1;
      const double drho1dr2 = a1 * (drhoa1i - rhoa1i / rij) * arg1j1;
      a1 = 2.0 * sij0 / rij;
      double drho1drm1[3], drho1drm2[3];
      for (int m = 0; m < 3; ++m) {
        drho1drm1[m] = a1 * rhoa1j * di.arho1[m];
        drho1drm2[m] = -a1 * rhoa1i * dj.arho1[m];
      }

      // rho2 (:294-307)
      double a2 = 2.0 * sij0 / rij2;
      const double drho2dr1 =
          a2 * (drhoa2j - 2.0 * rhoa2j / rij) * arg1i2 - 2.0 / 3.0 * di.arho2b * drhoa2j * sij0;
      const double drho2dr2 =
          a2 * (drhoa2i - 2.0 * rhoa2i / rij) * arg1j2 - 2.0 / 3.0 * dj.arho2b * drhoa2i * sij0;
      a2 = 4.0 * sij0 / rij2;
      double drho2drm1[3], drho2drm2[3];
      for (int m = 0; m < 3; ++m) {
        double s1 = 0.0, s2 = 0.0;
        for (int n = 0; n < 3; ++n) {
          s1 += di.arho2[p.vind2D[m][n]] * delij[n];
          s2 -= dj.arho2[p.vind2D[m][n]] * delij[n];
        }
        drho2drm1[m] = a2 * rhoa2j * s1;
        drho2drm2[m] = -a2 * rhoa2i * s2;
      }

      // rho3 (:310-331)
      double a3 = 2.0 * sij0 / rij3;
      double a3a = 6.0 / 5.0 * sij0 / rij;
      const double drho3dr1 =
          a3 * (drhoa3j - 3.0 * rhoa3j / rij) * arg1i3 - a3a * (drhoa3j - rhoa3j / rij) * arg3i3;
      const double drho3dr2 =
          a3 * (drhoa3i - 3.0 * rhoa3i / rij) * arg1j3 - a3a * (drhoa3i - rhoa3i / rij) * arg3j3;
      a3 = 6.0 * sij0 / rij3;
      a3a = 6.0 * sij0 / (5.0 * rij);
      double drho3drm1[3], drho3drm2[3];
      for (int m = 0; m < 3; ++m) {
        double s1 = 0.0, s2 = 0.0;
        int nv2 = 0;
        for (int n = 0; n < 3; ++n)
          for (int pp = n; pp < 3; ++pp) {
            const double arg = delij[n] * delij[pp] * p.v2D[nv2];
            s1 += di.arho3[p.vind3D[m][n][pp]] * arg;
            s2 += dj.arho3[p.vind3D[m][n][pp]] * arg;
            ++nv2;
          }
        drho3drm1[m] = (a3 * s1 - a3a * di.arho3b[m]) * rhoa3j;
        drho3drm2[m] = (-a3 * s2 + a3a * dj.arho3b[m]) * rhoa3i;
      }

      // t-average derivatives (ialloy=0 else branch, :451-466) — BARE subtraction
      const double t1i = ed[i].t_ave[0], t2i = ed[i].t_ave[1], t3i = ed[i].t_ave[2];
      const double t1j = ed[j].t_ave[0], t2j = ed[j].t_ave[1], t3j = ed[j].t_ave[2];
      const double aif = (ed[i].rho0 != 0.0) ? drhoa0j * sij0 / ed[i].rho0 : 0.0;
      const double ajf = (ed[j].rho0 != 0.0) ? drhoa0i * sij0 / ed[j].rho0 : 0.0;
      const double dt1dr1 = aif * (t1mj - t1i), dt1dr2 = ajf * (t1mi - t1j);
      const double dt2dr1 = aif * (t2mj - t2i), dt2dr2 = ajf * (t2mi - t2j);
      const double dt3dr1 = aif * (t3mj - t3i), dt3dr2 = ajf * (t3mi - t3j);

      // shape factors (single Si DIA) — shpi==shpj==p.shp
      const double* shpi = p.shp;
      const double* shpj = p.shp;

      // total density radial derivative (:497-510) — dgamma3 term DEAD (=0), replicated verbatim
      const double drhodr1 =
          ed[i].dgamma1 * drho0dr1 +
          ed[i].dgamma2 * (dt1dr1 * ed[i].rho1 + t1i * drho1dr1 + dt2dr1 * ed[i].rho2 +
                           t2i * drho2dr1 + dt3dr1 * ed[i].rho3 + t3i * drho3dr1) -
          ed[i].dgamma3 * (shpi[0] * dt1dr1 + shpi[1] * dt2dr1 + shpi[2] * dt3dr1);
      const double drhodr2 =
          ed[j].dgamma1 * drho0dr2 +
          ed[j].dgamma2 * (dt1dr2 * ed[j].rho1 + t1j * drho1dr2 + dt2dr2 * ed[j].rho2 +
                           t2j * drho2dr2 + dt3dr2 * ed[j].rho3 + t3j * drho3dr2) -
          ed[j].dgamma3 * (shpj[0] * dt1dr2 + shpj[1] * dt2dr2 + shpj[2] * dt3dr2);
      double drhodrm1[3], drhodrm2[3];
      for (int m = 0; m < 3; ++m) {
        drhodrm1[m] = ed[i].dgamma2 *
                      (t1i * drho1drm1[m] + t2i * drho2drm1[m] + t3i * drho3drm1[m]);
        drhodrm2[m] = ed[j].dgamma2 *
                      (t1j * drho1drm2[m] + t2j * drho2drm2[m] + t3j * drho3drm2[m]);
      }

      // sij derivatives (drhods1/2) — only if dscrfcn != 0 (partial; dead on the diamond)
      double drhods1 = 0.0, drhods2 = 0.0;
      const bool screen_active = std::fabs(dscrfcn_ij) > 1e-20;
      if (screen_active) {
        const double drho0ds1 = rhoa0j, drho0ds2 = rhoa0i;
        const double b1 = 2.0 / rij, b2 = 2.0 / rij2, b3 = 2.0 / rij3, b3a = 6.0 / (5.0 * rij);
        const double drho1ds1 = b1 * rhoa1j * arg1i1;
        const double drho1ds2 = b1 * rhoa1i * arg1j1;
        const double drho2ds1 = b2 * rhoa2j * arg1i2 - 2.0 / 3.0 * di.arho2b * rhoa2j;
        const double drho2ds2 = b2 * rhoa2i * arg1j2 - 2.0 / 3.0 * dj.arho2b * rhoa2i;
        const double drho3ds1 = b3 * rhoa3j * arg1i3 - b3a * rhoa3j * arg3i3;
        const double drho3ds2 = b3 * rhoa3i * arg1j3 - b3a * rhoa3i * arg3j3;
        // ai/aj for sij derivatives use rhoa0j/rho0[i] (NO sij factor, :586-589)
        const double ais = (ed[i].rho0 != 0.0) ? rhoa0j / ed[i].rho0 : 0.0;
        const double ajs = (ed[j].rho0 != 0.0) ? rhoa0i / ed[j].rho0 : 0.0;
        const double dt1ds1b = ais * (t1mj - t1i), dt1ds2b = ajs * (t1mi - t1j);
        const double dt2ds1b = ais * (t2mj - t2i), dt2ds2b = ajs * (t2mi - t2j);
        const double dt3ds1b = ais * (t3mj - t3i), dt3ds2b = ajs * (t3mi - t3j);
        drhods1 =
            ed[i].dgamma1 * drho0ds1 +
            ed[i].dgamma2 * (dt1ds1b * ed[i].rho1 + t1i * drho1ds1 + dt2ds1b * ed[i].rho2 +
                             t2i * drho2ds1 + dt3ds1b * ed[i].rho3 + t3i * drho3ds1) -
            ed[i].dgamma3 * (shpi[0] * dt1ds1b + shpi[1] * dt2ds1b + shpi[2] * dt3ds1b);
        drhods2 =
            ed[j].dgamma1 * drho0ds2 +
            ed[j].dgamma2 * (dt1ds2b * ed[j].rho1 + t1j * drho1ds2 + dt2ds2b * ed[j].rho2 +
                             t2j * drho2ds2 + dt3ds2b * ed[j].rho3 + t3j * drho3ds2) -
            ed[j].dgamma3 * (shpj[0] * dt1ds2b + shpj[1] * dt2ds2b + shpj[2] * dt3ds2b);
      }

      // energy derivatives (:613-637)
      const double dUdrij = phip * sij0 + ed[i].frhop * drhodr1 + ed[j].frhop * drhodr2;
      double dUdsij = 0.0;
      if (screen_active) dUdsij = phi + ed[i].frhop * drhods1 + ed[j].frhop * drhods2;
      double dUdrijm[3];
      for (int m = 0; m < 3; ++m)
        dUdrijm[m] = ed[i].frhop * drhodrm1[m] + ed[j].frhop * drhodrm2[m];
      // scaleij == 1 (single element) ⇒ no scale multiply

      const double force = dUdrij * recip + dUdsij * dscrfcn_ij;
      for (int m = 0; m < 3; ++m) {
        const double forcem = delij[m] * force + dUdrijm[m];
        if (m == 0) { a.fx[i] += forcem; a.fx[j] -= forcem; }
        else if (m == 1) { a.fy[i] += forcem; a.fy[j] -= forcem; }
        else { a.fz[i] += forcem; a.fz[j] -= forcem; }
      }

      // ---- the screening 3rd-atom k-loop (FULL list; fires only for partial 0<sij<1) ----
      if (std::fabs(sij0) < 1e-20 || std::fabs(sij0 - 1.0) < 1e-20) continue;  // binary-S re-gate
      // POISON (drop_class>0): drop the screening k-scatter entirely — the MB2 enumeration teeth.
      // atom-k's force collapses to ~0 ⇒ diverges from the golden, proving this loop is load-bearing.
      if (drop_class > 0) continue;
      const double delc = p.Cmax - p.Cmin;
      const double rbound = rij2 * p.ebound;
      for (size_t kn = 0; kn < nbr[i].size(); ++kn) {
        const auto& ek = nbr[i][kn];
        const int k = ek.j;
        if (k == j) continue;
        // k-relative vectors (meam_force.cpp:689-696): d_jk = x_k − x_j, d_ik = x_k − x_i.
        // From i's frame: ek.d* = x_k − x_i (== d_ik); d_jk = (x_k−x_i) − (x_j−x_i) = ek.d − ej.d.
        const double dxik = ek.dx, dyik = ek.dy, dzik = ek.dz;
        const double dxjk = ek.dx - ej.dx, dyjk = ek.dy - ej.dy, dzjk = ek.dz - ej.dz;
        const double rjk2 = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
        if (rjk2 > rbound) continue;
        const double rik2 = dxik * dxik + dyik * dyik + dzik * dzik;
        if (rik2 > rbound) continue;
        const double xik = rik2 / rij2, xjk = rjk2 / rij2;
        const double aa = 1.0 - (xik - xjk) * (xik - xjk);
        if (std::fabs(aa) < 1e-20) continue;  // iszero(a)
        double cikj = (2.0 * (xik + xjk) + aa - 2.0) / aa;
        if (!(cikj >= p.Cmin && cikj <= p.Cmax)) continue;
        cikj = (cikj - p.Cmin) / delc;
        double dfc;
        const double sikj = meam_detail::dfcut(cikj, dfc);
        double dCikj1, dCikj2;
        meam_detail::dCfunc2(rij2, rik2, rjk2, dCikj1, dCikj2);
        const double aw = sij0 / delc * dfc / sikj;
        const double dsij1 = aw * dCikj1;
        const double dsij2 = aw * dCikj2;
        if (std::fabs(dsij1) < 1e-20 && std::fabs(dsij2) < 1e-20) continue;
        const double force1 = dUdsij * dsij1;
        const double force2 = dUdsij * dsij2;
        a.fx[i] += force1 * dxik; a.fy[i] += force1 * dyik; a.fz[i] += force1 * dzik;
        a.fx[j] += force2 * dxjk; a.fy[j] += force2 * dyjk; a.fz[j] += force2 * dzjk;
        a.fx[k] -= force1 * dxik + force2 * dxjk;
        a.fy[k] -= force1 * dyik + force2 * dyjk;
        a.fz[k] -= force1 * dzik + force2 * dzjk;
      }
    }
  }

  const double r_zbl = p.re * (1.0 - 1.0 / p.alpha);
  if (std::sqrt(acc.min_r2) <= r_zbl)
    throw std::runtime_error("meam_direct_fp64: a pair is inside the ZBL blend region — deferred");
  acc.pe = acc.pe_embed + acc.pe_pair;
  return acc;
}

// ----------------------------------------------------------------------------------------
// int64 fixed-point density accumulation (G-B1 witness) — order-free (bitwise under atom
// relabeling). The signed angular arho1/2/3 sums are quantized + integer-summed; the final
// nonlinear F(ρ̄) (exp/log) is then a function of the order-free accumulators. Q24.40.
// (The φ pair term is FP64 — it reads the deterministic spline; its order-freedom is from
//  the i<j once-per-pair iteration, not int64.)
// ----------------------------------------------------------------------------------------
template <typename Real>
inline MeamAccum meam_run_fixed(const AtomSoA<Real>& a, const PairGeom& geom, const MeamParams& p) {
  const auto nbr = meam_build_nbr(a, geom);
  // 14 signed accumulators per atom: rho0, arho1[3], arho2[6], arho2b, arho3 (we keep the
  // 23-component vector); for compactness use core::fixed::ForceAccum (Q24.40 signed).
  struct FixedDens {
    core::fixed::ForceAccum rho0, arho2b;
    core::fixed::ForceAccum arho1[3], arho2[6], arho3[10], arho3b[3], t_ave[3];
  };
  std::vector<FixedDens> fd(a.n);

  for (int i = 0; i < a.n; ++i) {
    for (const auto& ej : nbr[i]) {
      const MeamScreen s = meam_getscreen(ej, nbr[i], p);
      const double sij = s.scrfcn * s.fcpair;
      if (std::fabs(sij) < 1e-20) continue;
      const double rij2 = ej.r2, rij = ej.r;
      const double aj = rij / p.re - 1.0;
      const double ro0j = p.rho0;
      const double rhoa0j = ro0j * std::exp(-p.beta0 * aj) * sij;
      const double rhoa1j = ro0j * std::exp(-p.beta1 * aj) * sij;
      const double rhoa2j = ro0j * std::exp(-p.beta2 * aj) * sij;
      const double rhoa3j = ro0j * std::exp(-p.beta3 * aj) * sij;
      fd[i].rho0.add(rhoa0j);
      fd[i].t_ave[0].add(p.t1_eff * rhoa0j);
      fd[i].t_ave[1].add(p.t2 * rhoa0j);
      fd[i].t_ave[2].add(p.t3 * rhoa0j);
      fd[i].arho2b.add(rhoa2j);
      const double A1j = rhoa1j / rij, A2j = rhoa2j / rij2, A3j = rhoa3j / (rij2 * rij);
      const double del[3] = {ej.dx, ej.dy, ej.dz};
      int nv2 = 0, nv3 = 0;
      for (int m = 0; m < 3; ++m) {
        fd[i].arho1[m].add(A1j * del[m]);
        fd[i].arho3b[m].add(rhoa3j * del[m] / rij);
        for (int n = m; n < 3; ++n) {
          fd[i].arho2[nv2].add(A2j * del[m] * del[n]);
          ++nv2;
          for (int pp = n; pp < 3; ++pp) {
            fd[i].arho3[nv3].add(A3j * del[m] * del[n] * del[pp]);
            ++nv3;
          }
        }
      }
    }
  }

  MeamAccum acc;
  // The PER-ATOM densities are order-free (int64 above) ⇒ each F(ρ̄_i) is deterministic; but the
  // TOTAL energy sum must ALSO be int64 (EnergyAccum) or Σ over relabeled atoms/pairs is FP64-
  // order-sensitive. Quantizing the per-atom F + per-pair φ·S and summing in int64 ⇒ the total
  // is bitwise under relabeling (B1/INV-9). (φ·S reads the deterministic spline; sij the
  // deterministic screening — the order-freedom is the int64 sum + the once-per-pair iteration.)
  core::fixed::EnergyAccum pe_embed, pe_pair;
  for (int i = 0; i < a.n; ++i) {
    MeamDensity d;
    d.rho0 = fd[i].rho0.value();
    d.arho2b = fd[i].arho2b.value();
    for (int m = 0; m < 3; ++m) {
      d.arho1[m] = fd[i].arho1[m].value();
      d.arho3b[m] = fd[i].arho3b[m].value();
      d.t_ave[m] = fd[i].t_ave[m].value();
    }
    for (int m = 0; m < 6; ++m) d.arho2[m] = fd[i].arho2[m].value();
    for (int m = 0; m < 10; ++m) d.arho3[m] = fd[i].arho3[m].value();
    pe_embed.add(meam_dens_final(d, p));
  }
  for (int i = 0; i < a.n; ++i)
    for (const auto& ej : nbr[i]) {
      if (i >= ej.j) continue;
      const MeamScreen s = meam_getscreen(ej, nbr[i], p);
      const double sij = s.scrfcn * s.fcpair;
      if (std::fabs(sij) < 1e-20) continue;
      pe_pair.add(p.phi_spline(ej.r) * sij);
    }
  acc.pe_embed = pe_embed.value();
  acc.pe_pair = pe_pair.value();
  acc.pe = acc.pe_embed + acc.pe_pair;
  return acc;
}

// ----------------------------------------------------------------------------------------
// Me2 — the int64 Q24.40 FORCE (G-B1 witness, order-free under atom relabeling). Same analytic
// chain as meam_direct_fp64, but every f_i/f_j/f_k component is QUANTIZED + integer-summed
// (core::fixed::ForceAccum) ⇒ the per-atom force is bitwise-invariant to the j/k accumulation
// order (B1/INV-9). The densities + embedding derivatives are FP64 functions of the order-free
// int64 densities (exactly as meam_run_fixed's energy). This is the LAMMPS-golden comparand
// (the value-algebra witness). drop_class>0 = POISON (the screening k-loop dropped).
// ----------------------------------------------------------------------------------------
template <typename Real>
inline MeamAccum meam_run_fixed_force(AtomSoA<Real>& a, const PairGeom& geom, const MeamParams& p,
                                      int drop_class = 0) {
  const auto nbr = meam_build_nbr(a, geom);
  std::vector<MeamDensity> dens(a.n);
  std::vector<std::vector<MeamScreenD>> scr(a.n);
  MeamAccum acc;

  // Pass 1 — int64 densities (order-free) + cached i-owned screening.
  struct FixedDens {
    core::fixed::ForceAccum rho0, arho2b;
    core::fixed::ForceAccum arho1[3], arho2[6], arho3[10], arho3b[3], t_ave[3];
  };
  std::vector<FixedDens> fd(a.n);
  for (int i = 0; i < a.n; ++i) {
    scr[i].resize(nbr[i].size());
    for (size_t jn = 0; jn < nbr[i].size(); ++jn) {
      const auto& ej = nbr[i][jn];
      acc.min_r2 = std::min(acc.min_r2, ej.r2);
      const MeamScreenD s = meam_getscreen_d(ej, nbr[i], p);
      scr[i][jn] = s;
      const double sij = s.scrfcn * s.fcpair;
      if (i < ej.j) {
        if (s.scrfcn == 0.0) ++acc.n_screened_zero;
        else if (s.scrfcn > 0.0 && s.scrfcn < 1.0) ++acc.n_screened_partial;
      }
      if (std::fabs(sij) < 1e-20) continue;
      const double rij2 = ej.r2, rij = ej.r;
      const double aj = rij / p.re - 1.0, ro0 = p.rho0;
      const double rhoa0j = ro0 * std::exp(-p.beta0 * aj) * sij;
      const double rhoa1j = ro0 * std::exp(-p.beta1 * aj) * sij;
      const double rhoa2j = ro0 * std::exp(-p.beta2 * aj) * sij;
      const double rhoa3j = ro0 * std::exp(-p.beta3 * aj) * sij;
      fd[i].rho0.add(rhoa0j);
      fd[i].t_ave[0].add(p.t1_eff * rhoa0j);
      fd[i].t_ave[1].add(p.t2 * rhoa0j);
      fd[i].t_ave[2].add(p.t3 * rhoa0j);
      fd[i].arho2b.add(rhoa2j);
      const double A1j = rhoa1j / rij, A2j = rhoa2j / rij2, A3j = rhoa3j / (rij2 * rij);
      const double del[3] = {ej.dx, ej.dy, ej.dz};
      int nv2 = 0, nv3 = 0;
      for (int m = 0; m < 3; ++m) {
        fd[i].arho1[m].add(A1j * del[m]);
        fd[i].arho3b[m].add(rhoa3j * del[m] / rij);
        for (int n = m; n < 3; ++n) {
          fd[i].arho2[nv2].add(A2j * del[m] * del[n]);
          ++nv2;
          for (int pp = n; pp < 3; ++pp) {
            fd[i].arho3[nv3].add(A3j * del[m] * del[n] * del[pp]);
            ++nv3;
          }
        }
      }
    }
  }

  // Decode order-free densities + embedding derivatives.
  std::vector<MeamEmbedDeriv> ed(a.n);
  core::fixed::EnergyAccum pe_embed, pe_pair;
  for (int i = 0; i < a.n; ++i) {
    MeamDensity& d = dens[i];
    d.rho0 = fd[i].rho0.value();
    d.arho2b = fd[i].arho2b.value();
    for (int m = 0; m < 3; ++m) {
      d.arho1[m] = fd[i].arho1[m].value();
      d.arho3b[m] = fd[i].arho3b[m].value();
      d.t_ave[m] = fd[i].t_ave[m].value();
    }
    for (int m = 0; m < 6; ++m) d.arho2[m] = fd[i].arho2[m].value();
    for (int m = 0; m < 10; ++m) d.arho3[m] = fd[i].arho3[m].value();
    ed[i] = meam_dens_final_deriv(d, p);
    pe_embed.add(ed[i].F);
  }

  // Pass 2 — the FORCE scatter into per-atom int64 accumulators (order-free, B1).
  std::vector<core::fixed::ForceAccum> ffx(a.n), ffy(a.n), ffz(a.n);
  for (int i = 0; i < a.n; ++i) {
    for (size_t jn = 0; jn < nbr[i].size(); ++jn) {
      const auto& ej = nbr[i][jn];
      const int j = ej.j;
      const double scrfcn_ij = scr[i][jn].scrfcn, dscrfcn_ij = scr[i][jn].dscrfcn;
      if (std::fabs(scrfcn_ij) < 1e-20) continue;
      const double sij0 = scrfcn_ij * scr[i][jn].fcpair;
      const double rij2 = ej.r2, rij = ej.r;
      const double phi = p.phi_spline(rij), phip = p.phip_spline(rij);
      if (i < j) pe_pair.add(phi * sij0);
      if (i >= j) continue;

      // SINGLE-SOURCE bond body (shared with the Me5 GPU K3 kernel). F-NOOP: meam_bond_force_device
      // is the verbatim Me2 bond chain; meam_screen_k_device is the verbatim k-loop body.
      const MeamForceParams fp = meam_force_params(p);
      const MeamBondForce bf =
          meam_bond_force_device(dens[i], dens[j], ed[i], ed[j], ej.dx, ej.dy, ej.dz, rij2, rij,
                                 scrfcn_ij, scr[i][jn].fcpair, dscrfcn_ij, phi, phip, fp);
      ffx[i].add(bf.fm[0]); ffy[i].add(bf.fm[1]); ffz[i].add(bf.fm[2]);
      ffx[j].add(-bf.fm[0]); ffy[j].add(-bf.fm[1]); ffz[j].add(-bf.fm[2]);

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
        ffx[i].add(sk.force1 * sk.dik[0]); ffy[i].add(sk.force1 * sk.dik[1]); ffz[i].add(sk.force1 * sk.dik[2]);
        ffx[j].add(sk.force2 * sk.djk[0]); ffy[j].add(sk.force2 * sk.djk[1]); ffz[j].add(sk.force2 * sk.djk[2]);
        ffx[k].add(-(sk.force1 * sk.dik[0] + sk.force2 * sk.djk[0]));
        ffy[k].add(-(sk.force1 * sk.dik[1] + sk.force2 * sk.djk[1]));
        ffz[k].add(-(sk.force1 * sk.dik[2] + sk.force2 * sk.djk[2]));
      }
    }
  }

  for (int i = 0; i < a.n; ++i) {
    a.fx[i] = ffx[i].value();
    a.fy[i] = ffy[i].value();
    a.fz[i] = ffz[i].value();
  }
  acc.pe_embed = pe_embed.value();
  acc.pe_pair = pe_pair.value();
  acc.pe = acc.pe_embed + acc.pe_pair;
  const double r_zbl = p.re * (1.0 - 1.0 / p.alpha);
  if (std::sqrt(acc.min_r2) <= r_zbl)
    throw std::runtime_error("meam_run_fixed_force: a pair is inside the ZBL blend region — deferred");
  return acc;
}

// ----------------------------------------------------------------------------------------
// Descriptor (the plug-in seam) + the inverted firewall — ACCEPTS the future Force pass's
// needs_transpose (the MEAM screening 3rd-atom write is the legitimate non-symmetric write,
// like SW/Tersoff). Energy-only here: run_pass throws "lands in Me3".
// ----------------------------------------------------------------------------------------
template <typename Real>
struct MeamPotential final : IManyBodyPotential<Real> {
  MeamParams meam;
  explicit MeamPotential(MeamParams m = {}) : meam(std::move(m)) {}
  // Density pass STATE is the 23-component MeamDensity (owned per-atom in the driver, NOT
  // bloated into the shared ManyBodyState which carries only scalar rho — the E5b firewall
  // discipline). accum_fracbits=44 is INERT for Me1 energy (Te1 inert-fracbits precedent).
  static constexpr PassDecl kPasses[3] = {
      {PassKind::Density, /*reuse*/ true, /*needs_transpose*/ false, false, 44},
      {PassKind::Embedding, /*reuse*/ true, /*needs_transpose*/ false, false, 0},
      {PassKind::Force, /*reuse*/ true, /*needs_transpose*/ true, false, 40},
  };
  std::span<const PassDecl> passes() const override { return {kPasses, 3}; }
  EffectiveRange effective_range() const override { return {2, /*symmetric_reach*/ true}; }
  void run_pass(int, const AtomSoA<Real>&, const PairGeom&, ManyBodyState<Real>&,
                const VisitPairs<Real>&) override {
    throw std::runtime_error("MeamPotential::run_pass: zone/ring dispatch lands in Me3");
  }
};
template <typename Real>
constexpr PassDecl MeamPotential<Real>::kPasses[3];

}  // namespace tdmd::potentials
