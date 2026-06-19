#pragma once
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"  // PairGeom (single source of min-image/cutoff)
#include "tdmd/potentials/many_body.hpp"

// M6 / Tersoff-ladder Te1 — the FIRST bond-order angular potential. Design of record:
// wf_1816b773-9f2. [ENG] (not in the 2007 dissertation, like EAM/SW). Arc: SW→Tersoff→MEAM.
//
// THE ESCALATION (the load-bearing point): SW's angular force is an EXPLICIT 3-body term
// and its all-triplets oracle shares only the ENUMERATION with the checked path ⇒ FD-of-
// energy was an INDEPENDENT algebra witness. Tersoff wraps the angle INSIDE the bond-order
// b_ij(ζ_ij), so a force-recomputation oracle SHARES the b_ij chain-rule algebra and goes
// BLIND to a sign/algebra bug. Independence is therefore carried by (1) FD-of-ENERGY (needs
// only the far-simpler energy) and (2) the external LAMMPS `pair_style tersoff run 0` golden.
// The all-bonds oracle is demoted to ENUMERATION/MB2 only. Both load-bearing gates ship Te1.
//
// All radial/bond-order helpers + the third-atom force (tersoff_zetaterm_d) are replicated
// CHARACTER-FOR-CHARACTER from LAMMPS pair_tersoff.cpp (stable 22Jul2025), including the
// 5-branch small-ζ b_ij guard, the ±69.0776 exp clamp, and BOTH sign conventions: the
// two-body uses delx = x_i−x_j (project); the three-body uses delr1 = x_j−x_i (= −dx) for
// the hats and the radial. Do NOT "simplify" the signs — FD-of-energy pins them.
namespace tdmd::potentials {

struct TersoffParams {  // canonical Tersoff-1988 Si (LAMMPS Si.tersoff)
  double powermint = 3.0;  // m
  double gamma = 1.0, lam3 = 1.3258, c = 4.8381, d = 2.0417, h = 0.0;  // h ≡ costheta0 (inside g)
  double powern = 22.956, beta = 0.33675, lam2 = 1.3258, bigb = 95.373;
  double bigr = 3.0, bigd = 0.2, lam1 = 3.2394, biga = 3264.7;
  double c1 = 0, c2 = 0, c3 = 0, c4 = 0;  // small-ζ branch thresholds (precomputed)

  TersoffParams() { recompute(); }
  void recompute() {
    if (!(powermint == 1.0 || powermint == 3.0) || bigd > bigr)
      throw std::runtime_error("TersoffParams: m must be 1 or 3, and D <= R");
    c1 = std::pow(2.0 * powern * 1.0e-16, -1.0 / powern);
    c2 = std::pow(2.0 * powern * 1.0e-8, -1.0 / powern);
    c3 = 1.0 / c2;
    c4 = 1.0 / c1;
  }
  double rcut() const { return bigr + bigd; }  // = cutmax = 3.2 Å (single-element)
};

namespace ters_detail {
inline constexpr double kPi2 = std::numbers::pi / 2.0;
inline constexpr double kPi4 = std::numbers::pi / 4.0;
TDMD_HOST_DEVICE inline double cube(double x) { return x * x * x; }
TDMD_HOST_DEVICE inline double sq(double x) { return x * x; }
}  // namespace ters_detail

// --- radial functions + cutoff (LAMMPS ters_fc/fc_d/fa/fa_d; fA FOLDS fc) -------------
TDMD_HOST_DEVICE inline double ters_fc(double r, const TersoffParams& p) {
  if (r < p.bigr - p.bigd) return 1.0;
  if (r > p.bigr + p.bigd) return 0.0;
  return 0.5 * (1.0 - std::sin(ters_detail::kPi2 * (r - p.bigr) / p.bigd));
}
TDMD_HOST_DEVICE inline double ters_fc_d(double r, const TersoffParams& p) {
  if (r < p.bigr - p.bigd || r > p.bigr + p.bigd) return 0.0;
  return -(ters_detail::kPi4 / p.bigd) * std::cos(ters_detail::kPi2 * (r - p.bigr) / p.bigd);
}
TDMD_HOST_DEVICE inline double ters_fa(double r, const TersoffParams& p) {
  if (r > p.bigr + p.bigd) return 0.0;
  return -p.bigb * std::exp(-p.lam2 * r) * ters_fc(r, p);
}
TDMD_HOST_DEVICE inline double ters_fa_d(double r, const TersoffParams& p) {
  if (r > p.bigr + p.bigd) return 0.0;
  return p.bigb * std::exp(-p.lam2 * r) * (p.lam2 * ters_fc(r, p) - ters_fc_d(r, p));
}
// --- angular g(θ) + its derivative (LAMMPS ters_gijk/gijk_d; h ≡ costheta0) ------------
TDMD_HOST_DEVICE inline double ters_gijk(double costheta, const TersoffParams& p) {
  const double cc = p.c * p.c, dd = p.d * p.d, hcth = p.h - costheta;
  return p.gamma * (1.0 + cc / dd - cc / (dd + hcth * hcth));
}
TDMD_HOST_DEVICE inline double ters_gijk_d(double costheta, const TersoffParams& p) {
  const double cc = p.c * p.c, dd = p.d * p.d, hcth = p.h - costheta;
  const double num = -2.0 * cc * hcth, den = 1.0 / (dd + hcth * hcth);
  return p.gamma * num * den * den;
}
// --- the exp(λ3^m (r_ij − r_ik)^m) envelope (LAMMPS ex_delr; clamp ±69.0776) -----------
TDMD_HOST_DEVICE inline double ters_ex_delr(double rij, double rik, const TersoffParams& p) {
  double arg = (p.powermint == 3.0) ? ters_detail::cube(p.lam3 * (rij - rik)) : p.lam3 * (rij - rik);
  if (arg > 69.0776) return 1.0e30;
  if (arg < -69.0776) return 0.0;
  return std::exp(arg);
}
// --- bond order b_ij + db_ij/dζ (LAMMPS ters_bij/bij_d — the 5-branch small-ζ guard) ---
TDMD_HOST_DEVICE inline double ters_bij(double zeta, const TersoffParams& p) {
  const double tmp = p.beta * zeta;
  if (tmp > p.c1) return 1.0 / std::sqrt(tmp);
  if (tmp > p.c2) return (1.0 - std::pow(tmp, -p.powern) / (2.0 * p.powern)) / std::sqrt(tmp);
  if (tmp < p.c4) return 1.0;
  if (tmp < p.c3) return 1.0 - std::pow(tmp, p.powern) / (2.0 * p.powern);
  return std::pow(1.0 + std::pow(tmp, p.powern), -1.0 / (2.0 * p.powern));
}
TDMD_HOST_DEVICE inline double ters_bij_d(double zeta, const TersoffParams& p) {
  const double tmp = p.beta * zeta;
  if (tmp > p.c1) return p.beta * -0.5 * std::pow(tmp, -1.5);
  if (tmp > p.c2)
    return p.beta * (-0.5 * std::pow(tmp, -1.5) *
                     (1.0 - (1.0 + 1.0 / (2.0 * p.powern)) * std::pow(tmp, -p.powern)));
  if (tmp < p.c4) return 0.0;
  if (tmp < p.c3) return -0.5 * p.beta * std::pow(tmp, p.powern - 1.0);
  const double tmp_n = std::pow(tmp, p.powern);
  return -0.5 * std::pow(1.0 + tmp_n, -1.0 - (1.0 / (2.0 * p.powern))) * tmp_n / zeta;
}

// --- ζ term for one (i;j,k) (LAMMPS zeta()) — hats are LAMMPS convention (x_j−x_i)/r ---
TDMD_HOST_DEVICE inline double tersoff_zeta(double rij, double rik, const double* rij_hat,
                                            const double* rik_hat, const TersoffParams& p) {
  const double costheta = rij_hat[0] * rik_hat[0] + rij_hat[1] * rik_hat[1] + rij_hat[2] * rik_hat[2];
  return ters_fc(rik, p) * ters_gijk(costheta, p) * ters_ex_delr(rij, rik, p);
}

// --- ∂cosθ/∂{x_i,x_j,x_k} (LAMMPS costheta_d) -----------------------------------------
TDMD_HOST_DEVICE inline void tersoff_costheta_d(const double* rij_hat, double rijinv,
                                                const double* rik_hat, double rikinv,
                                                double* dri, double* drj, double* drk) {
  const double ct = rij_hat[0] * rik_hat[0] + rij_hat[1] * rik_hat[1] + rij_hat[2] * rik_hat[2];
  for (int a = 0; a < 3; ++a) {
    drj[a] = (rik_hat[a] - ct * rij_hat[a]) * rijinv;
    drk[a] = (rij_hat[a] - ct * rik_hat[a]) * rikinv;
    dri[a] = -(drj[a] + drk[a]);
  }
}
// --- the third-atom force: ∂(ζ term)/∂{i,j,k} × prefactor (LAMMPS ters_zetaterm_d) -----
TDMD_HOST_DEVICE inline void tersoff_zetaterm_d(double prefactor, const double* rij_hat, double rij,
                                                double rijinv, const double* rik_hat, double rik,
                                                double rikinv, double* dri, double* drj, double* drk,
                                                const TersoffParams& p) {
  const double fc = ters_fc(rik, p), dfc = ters_fc_d(rik, p);
  const double ex = ters_ex_delr(rij, rik, p);
  const double ex_d = (p.powermint == 3.0)
                          ? 3.0 * ters_detail::cube(p.lam3) * ters_detail::sq(rij - rik) * ex
                          : p.lam3 * ex;
  const double ct = rij_hat[0] * rik_hat[0] + rij_hat[1] * rik_hat[1] + rij_hat[2] * rik_hat[2];
  const double g = ters_gijk(ct, p), g_d = ters_gijk_d(ct, p);
  double dcdi[3], dcdj[3], dcdk[3];
  tersoff_costheta_d(rij_hat, rijinv, rik_hat, rikinv, dcdi, dcdj, dcdk);
  for (int a = 0; a < 3; ++a) {
    dri[a] = prefactor * (-dfc * g * ex * rik_hat[a] + fc * g_d * ex * dcdi[a] +
                          fc * g * ex_d * (rik_hat[a] - rij_hat[a]));
    drj[a] = prefactor * (fc * g_d * ex * dcdj[a] + fc * g * ex_d * rij_hat[a]);
    drk[a] = prefactor * (dfc * g * ex * rik_hat[a] + fc * g_d * ex * dcdk[a] - fc * g * ex_d * rik_hat[a]);
  }
}

// --- neighbour list (per atom i): min-imaged dx = x_i − x_j (PROJECT convention) + r ---
struct TersNbr { int j; double dx, dy, dz, r; };
template <typename Real>
inline std::vector<std::vector<TersNbr>> tersoff_build_nbr(const AtomSoA<Real>& a,
                                                           const PairGeom& geom) {
  std::vector<std::vector<TersNbr>> nbr(a.n);
  for (int i = 0; i < a.n; ++i)
    for (int j = 0; j < a.n; ++j) {
      if (j == i) continue;
      double dx = a.x[i] - a.x[j], dy = a.y[i] - a.y[j], dz = a.z[i] - a.z[j], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
      nbr[i].push_back({j, dx, dy, dz, std::sqrt(r2)});
    }
  return nbr;
}

struct TersoffAccum { double pe = 0.0; double min_r2 = 1e300; long n_bonds = 0; long n_triplets = 0; };

// --- energy-only (the FD driver): E = Σ_{i<j} fc·fR + ½ Σ_directed fc·b_ij·fA. The TWO-TERM
// LAMMPS split (repulsive once-per-pair NO ½; attractive directed ½, center-i owned). FP64 ζ.
template <typename Real>
double tersoff_energy(const AtomSoA<Real>& a, const PairGeom& geom, const TersoffParams& p) {
  const auto nbr = tersoff_build_nbr(a, geom);
  double E = 0.0;
  for (int i = 0; i < a.n; ++i) {
    for (const auto& e : nbr[i]) {
      if (i < e.j) E += ters_fc(e.r, p) * p.biga * std::exp(-p.lam1 * e.r);  // repulsive once/pair
      // attractive directed bond (i, e.j): ζ_ij over k, then ½·fc·b·fA (fA folds fc).
      double zeta = 0.0;
      const double rijh[3] = {-e.dx / e.r, -e.dy / e.r, -e.dz / e.r};  // (x_j−x_i)/r
      for (const auto& k : nbr[i]) {
        if (k.j == e.j) continue;
        const double rikh[3] = {-k.dx / k.r, -k.dy / k.r, -k.dz / k.r};
        zeta += tersoff_zeta(e.r, k.r, rijh, rikh, p);
      }
      E += 0.5 * ters_bij(zeta, p) * ters_fa(e.r, p);  // ½·b_ij·fA  (fA = −B e^{−λ2 r} fc)
    }
  }
  return E;
}

// --- (1) FP64 ORACLE (scatter, 2-pass, LAMMPS structure). drop_class>0 = POISON (skip the
// k-term whose (k<i) straddles i — MB2 enumeration teeth). Forces ACCUMULATED into a.f.
template <typename Real>
TersoffAccum tersoff_direct_fp64(AtomSoA<Real>& a, const PairGeom& geom, const TersoffParams& p,
                                 bool with_forces, int drop_class = 0) {
  const auto nbr = tersoff_build_nbr(a, geom);
  TersoffAccum acc;
  for (int i = 0; i < a.n; ++i) {
    for (const auto& e : nbr[i]) {
      acc.min_r2 = std::min(acc.min_r2, e.r * e.r);
      // (A) repulsive two-body, once per undirected pair (i<j): delx = dx (project).
      if (i < e.j) {
        double fforce, eng;
        const double r = e.r, tmp_fc = ters_fc(r, p), tmp_fc_d = ters_fc_d(r, p);
        const double tmp_exp = std::exp(-p.lam1 * r);
        fforce = -p.biga * tmp_exp * (tmp_fc_d - tmp_fc * p.lam1) / r;
        eng = tmp_fc * p.biga * tmp_exp;
        acc.pe += eng; ++acc.n_bonds;
        if (with_forces) {
          a.fx[i] += e.dx * fforce; a.fy[i] += e.dy * fforce; a.fz[i] += e.dz * fforce;
          a.fx[e.j] -= e.dx * fforce; a.fy[e.j] -= e.dy * fforce; a.fz[e.j] -= e.dz * fforce;
        }
      }
      // (B,C) attractive directed bond (i, j): ζ_ij over k, force_zeta, then the k-loop.
      const double rij = e.r, rijinv = 1.0 / rij;
      const double rijh[3] = {-e.dx * rijinv, -e.dy * rijinv, -e.dz * rijinv};  // delr1/rij
      double zeta = 0.0;
      for (const auto& k : nbr[i]) {
        if (k.j == e.j) continue;
        if (drop_class && ((k.j < i) != (e.j < i))) continue;  // POISON (enumeration teeth)
        const double rikinv = 1.0 / k.r;
        const double rikh[3] = {-k.dx * rikinv, -k.dy * rikinv, -k.dz * rikinv};
        zeta += tersoff_zeta(rij, k.r, rijh, rikh, p);
      }
      const double fa = ters_fa(rij, p), fa_d = ters_fa_d(rij, p), bij = ters_bij(zeta, p);
      const double fforce_z = 0.5 * bij * fa_d, prefactor = -0.5 * fa * ters_bij_d(zeta, p);
      acc.pe += 0.5 * bij * fa;  // attractive energy, center-i owned, ½
      if (!with_forces) continue;
      // (B) attractive radial: fpair = fforce·(1/r) (LAMMPS folds /r here, NOT in force_zeta);
      // f[i] += delr1·fpair, f[j] −= delr1·fpair; delr1 = −dx.
      const double fpair_z = fforce_z * rijinv;
      a.fx[i] += -e.dx * fpair_z; a.fy[i] += -e.dy * fpair_z; a.fz[i] += -e.dz * fpair_z;
      a.fx[e.j] -= -e.dx * fpair_z; a.fy[e.j] -= -e.dy * fpair_z; a.fz[e.j] -= -e.dz * fpair_z;
      // (C) the angular third-atom write, scatter f_i,f_j,f_k.
      for (const auto& k : nbr[i]) {
        if (k.j == e.j) continue;
        if (drop_class && ((k.j < i) != (e.j < i))) continue;
        const double rikinv = 1.0 / k.r;
        const double rikh[3] = {-k.dx * rikinv, -k.dy * rikinv, -k.dz * rikinv};
        double fi[3], fj[3], fk[3];
        tersoff_zetaterm_d(prefactor, rijh, rij, rijinv, rikh, k.r, rikinv, fi, fj, fk, p);
        a.fx[i] += fi[0]; a.fy[i] += fi[1]; a.fz[i] += fi[2];
        a.fx[e.j] += fj[0]; a.fy[e.j] += fj[1]; a.fz[e.j] += fj[2];
        a.fx[k.j] += fk[0]; a.fy[k.j] += fk[1]; a.fz[k.j] += fk[2];
        ++acc.n_triplets;
      }
    }
  }
  return acc;
}

// --- (2) int64 SCATTER path (Q24.40 ForceAccum) — the determinism witness (G-B1). Same
// LAMMPS structure as the oracle; FP64 ζ, every force contribution quantized + integer-summed
// ⇒ order-free (bitwise under relabeling). (The owned-written-once transpose-replay is Te3.)
template <typename Real>
TersoffAccum tersoff_run_fixed(AtomSoA<Real>& a, const PairGeom& geom, const TersoffParams& p) {
  const auto nbr = tersoff_build_nbr(a, geom);
  std::vector<core::fixed::ForceAccum> fx(a.n), fy(a.n), fz(a.n);
  core::fixed::EnergyAccum pe;
  TersoffAccum acc;
  for (int i = 0; i < a.n; ++i)
    for (const auto& e : nbr[i]) {
      acc.min_r2 = std::min(acc.min_r2, e.r * e.r);
      if (i < e.j) {
        const double r = e.r, tmp_exp = std::exp(-p.lam1 * r);
        const double fforce = -p.biga * tmp_exp * (ters_fc_d(r, p) - ters_fc(r, p) * p.lam1) / r;
        pe.add(ters_fc(r, p) * p.biga * tmp_exp);
        fx[i].add(e.dx * fforce); fy[i].add(e.dy * fforce); fz[i].add(e.dz * fforce);
        fx[e.j].add(-e.dx * fforce); fy[e.j].add(-e.dy * fforce); fz[e.j].add(-e.dz * fforce);
      }
      const double rij = e.r, rijinv = 1.0 / rij;
      const double rijh[3] = {-e.dx * rijinv, -e.dy * rijinv, -e.dz * rijinv};
      double zeta = 0.0;
      for (const auto& k : nbr[i]) {
        if (k.j == e.j) continue;
        const double rikinv = 1.0 / k.r;
        const double rikh[3] = {-k.dx * rikinv, -k.dy * rikinv, -k.dz * rikinv};
        zeta += tersoff_zeta(rij, k.r, rijh, rikh, p);
      }
      const double fa = ters_fa(rij, p), bij = ters_bij(zeta, p);
      const double fforce_z = 0.5 * bij * ters_fa_d(rij, p), prefactor = -0.5 * fa * ters_bij_d(zeta, p);
      pe.add(0.5 * bij * fa);
      const double fpair_z = fforce_z * rijinv;  // LAMMPS folds /r here (see oracle)
      fx[i].add(-e.dx * fpair_z); fy[i].add(-e.dy * fpair_z); fz[i].add(-e.dz * fpair_z);
      fx[e.j].add(e.dx * fpair_z); fy[e.j].add(e.dy * fpair_z); fz[e.j].add(e.dz * fpair_z);
      for (const auto& k : nbr[i]) {
        if (k.j == e.j) continue;
        const double rikinv = 1.0 / k.r;
        const double rikh[3] = {-k.dx * rikinv, -k.dy * rikinv, -k.dz * rikinv};
        double fi[3], fj[3], fk[3];
        tersoff_zetaterm_d(prefactor, rijh, rij, rijinv, rikh, k.r, rikinv, fi, fj, fk, p);
        fx[i].add(fi[0]); fy[i].add(fi[1]); fz[i].add(fi[2]);
        fx[e.j].add(fj[0]); fy[e.j].add(fj[1]); fz[e.j].add(fj[2]);
        fx[k.j].add(fk[0]); fy[k.j].add(fk[1]); fz[k.j].add(fk[2]);
      }
    }
  for (int i = 0; i < a.n; ++i) { a.fx[i] += Real(fx[i].value()); a.fy[i] += Real(fy[i].value()); a.fz[i] += Real(fz[i].value()); }
  acc.pe = pe.value();
  return acc;
}

// --- descriptor (the plug-in seam) + the inverted firewall (accepts the Force needs_transpose).
template <typename Real>
struct TersoffPotential final : IManyBodyPotential<Real> {
  TersoffParams ters;
  explicit TersoffPotential(TersoffParams t = {}) : ters(std::move(t)) {}
  // NOTE: ζ_ij is FP64 (NOT a FixedAccum — owned by center i, single fixed window order ⇒
  // order-free; a FixedAccum ζ would make b_ij(ζ) piecewise-constant and break FD-of-energy).
  // The BondOrder pass's accum_fracbits is therefore INERT — Te3's driver must NOT allocate a
  // FixedAccum off it. Only the Force pass writes int64 (Q24.40, the third-atom transpose).
  static constexpr PassDecl kPasses[2] = {
      {PassKind::BondOrder, /*reuse*/ true, /*needs_transpose*/ false, false, 0},   // ζ_ij FP64 scalar
      {PassKind::Force, /*reuse*/ true, /*needs_transpose*/ true, false, 40},        // 3rd-atom write
  };
  std::span<const PassDecl> passes() const override { return {kPasses, 2}; }
  EffectiveRange effective_range() const override { return {2, /*symmetric_reach*/ true}; }
  void run_pass(int, const AtomSoA<Real>&, const PairGeom&, ManyBodyState<Real>&,
                const VisitPairs<Real>&) override {
    throw std::runtime_error("TersoffPotential::run_pass: zone/ring dispatch lands in Te3");
  }
};
template <typename Real>
constexpr PassDecl TersoffPotential<Real>::kPasses[2];

}  // namespace tdmd::potentials
