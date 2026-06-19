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
