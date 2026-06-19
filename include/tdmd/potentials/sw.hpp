#pragma once
#include <cmath>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"  // PairGeom (single source of min-image/cutoff)
#include "tdmd/potentials/many_body.hpp"

// M6 / SW-ladder PR-1 (T1) — Stillinger–Weber, the FIRST ANGULAR potential.
// Design of record: the angular-potential design workflow (wf_5294df4d-65c).
//
// [ENG] SW is not in the 2007 dissertation (like EAM — Glossary §EAM). It is the
// simplest 3-body form (explicit φ₃(r_ij,r_ik,θ_jik), no bond-order / no screening)
// and its all-triplets oracle shares NO derivative algebra with the path it checks
// (only the ENUMERATION differs) — the cleanest place to establish the NON-SYMMETRIC
// third-atom-k force write (PassDecl.needs_transpose) the symmetric EAM accumulator
// (q(j)=−q(i)) has no slot for. Arc: SW → Tersoff → MEAM.
//
// This PR is CPU-only: the analytic φ₂+φ₃, the FP64 all-triplets ORACLE (scatter per
// CENTER — sw_direct_fp64), and the int64 deterministic path (TRANSPOSE-REPLAY, gather
// per OWNER — sw_run_fixed). The two use genuinely different enumeration, so the oracle
// catches a dropped/wrong triplet (MB2) that 1-vs-z and run-to-run are blind to. The
// zone / ring / GPU rungs are T3 / T3b / T5; the LAMMPS Si golden is T7.
namespace tdmd::potentials {

using core::AtomSoA;
using core::Box;
using core::PairGeom;

// Stillinger–Weber Si parameters (LAMMPS Si.sw — single-element). rcut = a·σ is the
// single cutoff for BOTH φ₂ and φ₃; the exp envelope makes value+derivative vanish
// smoothly at a·σ (a force-shift is intrinsic — no explicit subtraction, unlike EAM).
struct SwParams {
  double eps = 2.1683;       // eV
  double sigma = 2.0951;     // Å
  double a = 1.80;           // cutoff in units of σ
  double lambda = 21.0;      // 3-body strength
  double gamma = 1.20;       // 3-body envelope decay
  double A = 7.049556277;    // 2-body strength
  double B = 0.6022245584;   // 2-body well
  double cos0 = -1.0 / 3.0;  // cos(109.47°) — tetrahedral equilibrium angle
  int p = 4, q = 0;          // 2-body radial exponents

  double rcut() const { return a * sigma; }
};

struct SwAccum {
  double pe = 0.0;
  double min_r2 = 1e300;
  long n_triplets = 0;  // MB2 multiset-size witness (oracle vs zone count must match)
};

// --- two-body φ₂(r): value + dφ₂/dr (r < a·σ; caller gates on the cutoff). -------
TDMD_HOST_DEVICE inline void sw_phi2(const SwParams& sp, double r, double& phi, double& dphi) {
  const double sr = sp.sigma / r;
  const double d = r - sp.a * sp.sigma;          // < 0 inside the cutoff
  const double env = std::exp(sp.sigma / d);     // → 0 as r → (a·σ)⁻
  const double srp = std::pow(sr, sp.p), srq = std::pow(sr, sp.q);
  const double term = sp.A * sp.eps * (sp.B * srp - srq);
  phi = term * env;
  const double dsr = -sp.sigma / (r * r);        // d(σ/r)/dr
  const double dterm = sp.A * sp.eps * dsr *
      (sp.B * sp.p * std::pow(sr, sp.p - 1) - sp.q * std::pow(sr, sp.q - 1));
  const double denv = env * (-sp.sigma / (d * d));
  dphi = dterm * env + term * denv;
}

// --- three-body envelope g(r)=exp(γσ/(r−aσ)) + g'(r). ---------------------------
TDMD_HOST_DEVICE inline void sw_g(const SwParams& sp, double r, double& g, double& gp) {
  const double d = r - sp.a * sp.sigma;
  g = std::exp(sp.gamma * sp.sigma / d);
  gp = -sp.gamma * sp.sigma / (d * d) * g;
}

struct SwVec3 { double x, y, z; };

// THE three-atom force write (the new machinery). Triplet centered at i with wings
// j,k. Inputs are the two bond vectors FROM the center: rij = x_i − x_j (the project
// sign convention, dx=xi−xj at zone_force.cuh / eam_zone.hpp), rik = x_i − x_k, with
// magnitudes rij, rik (both < a·σ; caller gates). Returns φ₃ and writes f_i,f_j,f_k.
//   f_j = +2λεΔ g_ij g_ik (1/r_ij)(ê_ik − c ê_ij) + λεΔ² g'_ij g_ik ê_ij
//   f_k = +2λεΔ g_ij g_ik (1/r_ik)(ê_ij − c ê_ik) + λεΔ² g_ij g'_ik ê_ik
//   f_i = −(f_j + f_k)   ← triplet translational invariance (Σf=0 per triplet; G-TI).
// (The +2 angular sign uses ∂cosθ/∂x_j = −(1/r_ij)(ê_ik − c ê_ij) — the POSITION
//  gradient, the negative of the bond-vector gradient; this is the SW sign trap that
//  G-FD exists to catch — the oracle shares this algebra and is blind to it.)
// f_k is the contribution the symmetric (q(j)=−q(i)) accumulator has no slot for. No
// acos anywhere (cosθ from the dot product) — accuracy + determinism asset.
TDMD_HOST_DEVICE inline double sw_triplet(const SwParams& sp,
                         double rijx, double rijy, double rijz, double rij,
                         double rikx, double riky, double rikz, double rik,
                         SwVec3& fi, SwVec3& fj, SwVec3& fk) {
  double gij, gpij, gik, gpik;
  sw_g(sp, rij, gij, gpij);
  sw_g(sp, rik, gik, gpik);
  const double eijx = rijx / rij, eijy = rijy / rij, eijz = rijz / rij;
  const double eikx = rikx / rik, eiky = riky / rik, eikz = rikz / rik;
  const double c = eijx * eikx + eijy * eiky + eijz * eikz;  // cosθ_jik
  const double dlt = c - sp.cos0;
  const double le = sp.lambda * sp.eps;
  // WING-CANONICAL form (B1/INV-9 — load-bearing). The transpose-replay relies on
  // f_j(i;o,m) == f_k(i;m,o) BITWISE. The two slots are equal in ℝ, but FP multiply is
  // non-ASSOCIATIVE: writing the j-slot as ((2le·dlt)·g_ij)·g_ik and the k-slot as
  // ((2le·dlt)·g_ij)·g_ik with the wings swapped grates ~1 ULP, which straddles the
  // Q24.40 tie ⇒ a different int64 under atom relabeling (non-determinism). Fix: gg is a
  // 2-operand product (IEEE multiply IS commutative ⇒ g_ij·g_ik == g_ik·g_ij bitwise);
  // angcoef is shared; the radial B-terms mirror the operand order (gp·g on BOTH slots).
  // Then the k-slot is the exact bitwise image of the j-slot with the wings exchanged.
  const double gg = gij * gik;
  const double ledd = le * dlt;
  const double ledd2 = ledd * dlt;
  const double E3 = ledd2 * gg;
  const double angcoef = 2.0 * ledd * gg;

  const double Aj = angcoef / rij;        // coeff of (ê_ik − c ê_ij)
  const double Bj = ledd2 * gpij * gik;   // (ledd2·gp_ij)·g_ik
  fj.x = Aj * (eikx - c * eijx) + Bj * eijx;
  fj.y = Aj * (eiky - c * eijy) + Bj * eijy;
  fj.z = Aj * (eikz - c * eijz) + Bj * eijz;

  const double Ak = angcoef / rik;        // coeff of (ê_ij − c ê_ik)
  const double Bk = ledd2 * gpik * gij;   // mirror of Bj: (ledd2·gp_ik)·g_ij
  fk.x = Ak * (eijx - c * eikx) + Bk * eikx;
  fk.y = Ak * (eijy - c * eiky) + Bk * eiky;
  fk.z = Ak * (eijz - c * eikz) + Bk * eikz;

  fi.x = -(fj.x + fk.x);  fi.y = -(fj.y + fk.y);  fi.z = -(fj.z + fk.z);
  return E3;
}

// Full neighbour list at a's positions (i has j AND j has i), min-imaged bonds
// dx = x_i − x_j. Single source of min-image/cutoff (PairGeom). Built once, shared
// by the oracle (scatter) and the transpose-replay (gather) — their enumerations
// differ, the candidate topology does not.
struct SwNbr { int j; double dx, dy, dz, r; };
template <typename Real>
inline std::vector<std::vector<SwNbr>> sw_build_nbr(const AtomSoA<Real>& a,
                                                    const PairGeom& geom) {
  std::vector<std::vector<SwNbr>> nbr(a.n);
  for (int i = 0; i < a.n; ++i)
    for (int j = 0; j < a.n; ++j) {
      if (j == i) continue;
      double dx = a.x[i] - a.x[j], dy = a.y[i] - a.y[j], dz = a.z[i] - a.z[j], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
      nbr[i].push_back({j, dx, dy, dz, std::sqrt(r2)});
    }
  return nbr;
}

// --- (1) FP64 ORACLE: scatter per CENTER. The structural analog of eam_direct_fp64 —
// FP64 throughout (a fixed-point force is piecewise-constant in x ⇒ breaks FD), no
// zone/window/cell-list (sees the whole box ⇒ cannot drop a triplet by a residence
// bug), O(N·nbr²). `drop_self_class`>0 deliberately narrows the φ₃ enumeration (a
// POISON: skip triplets whose two wings straddle the center index) so the acceptance
// can prove the oracle has TEETH (a dropped class DIVERGES while 1-vs-z stays green).
template <typename Real>
SwAccum sw_direct_fp64(AtomSoA<Real>& a, const Box& box, const SwParams& sp,
                       bool with_forces, int drop_self_class = 0) {
  const PairGeom geom(box, sp.rcut());
  const auto nbr = sw_build_nbr(a, geom);
  SwAccum acc;

  // φ₂ — symmetric pair, each undirected pair once (lower index owns the energy).
  for (int i = 0; i < a.n; ++i)
    for (const auto& e : nbr[i]) {
      acc.min_r2 = std::min(acc.min_r2, e.r * e.r);
      if (e.j < i) continue;  // count {i,j} once
      double phi, dphi;
      sw_phi2(sp, e.r, phi, dphi);
      acc.pe += phi;
      if (!with_forces) continue;
      const double f_over_r = -dphi / e.r;  // f_i = −φ₂'(r) ê_ij
      a.fx[i] += f_over_r * e.dx;  a.fy[i] += f_over_r * e.dy;  a.fz[i] += f_over_r * e.dz;
      a.fx[e.j] -= f_over_r * e.dx; a.fy[e.j] -= f_over_r * e.dy; a.fz[e.j] -= f_over_r * e.dz;
    }

  // φ₃ — per CENTER i, every unordered wing pair (j<k by index). Scatter f_i,f_j,f_k.
  for (int i = 0; i < a.n; ++i) {
    const auto& nb = nbr[i];
    for (std::size_t x = 0; x < nb.size(); ++x)
      for (std::size_t y = x + 1; y < nb.size(); ++y) {
        if (drop_self_class && (nb[x].j < i) != (nb[y].j < i)) continue;  // POISON
        SwVec3 fi, fj, fk;
        const double E3 = sw_triplet(sp, nb[x].dx, nb[x].dy, nb[x].dz, nb[x].r,
                                     nb[y].dx, nb[y].dy, nb[y].dz, nb[y].r, fi, fj, fk);
        acc.pe += E3;
        ++acc.n_triplets;
        if (!with_forces) continue;
        a.fx[i] += fi.x; a.fy[i] += fi.y; a.fz[i] += fi.z;
        a.fx[nb[x].j] += fj.x; a.fy[nb[x].j] += fj.y; a.fz[nb[x].j] += fj.z;
        a.fx[nb[y].j] += fk.x; a.fy[nb[y].j] += fk.y; a.fz[nb[y].j] += fk.z;
      }
  }
  return acc;
}

// --- (2) int64 deterministic path: TRANSPOSE-REPLAY, gather per OWNER. Each owned
// atom o sums ONLY its own role-gradient over every triplet it participates in (center,
// or either wing) — zero cross-atom writes ⇒ a drop-in WinForce policy for the zone
// ring (T3, owned-written-once). B1/INV-9 needs only that each atom receives an
// order-free MULTISET of quantized int64 contributions (integer add is associative);
// it does NOT need q(j)=−q(i) (a pair convenience). Per-triplet Σf≠0 after quantization
// is fine — global Σf=0 is bitwise because every triplet is enumerated once across the
// system. Q24.40 ForceAccum suffices (SW forces are O(1–10) eV/Å); the add() guard
// HALTs on a near-cutoff blowup. Bitwise == itself under any triplet visit order (G-B1).
template <typename Real>
SwAccum sw_run_fixed(AtomSoA<Real>& a, const Box& box, const SwParams& sp) {
  const PairGeom geom(box, sp.rcut());
  const auto nbr = sw_build_nbr(a, geom);
  std::vector<core::fixed::ForceAccum> fx(a.n), fy(a.n), fz(a.n);
  core::fixed::EnergyAccum pe;
  SwAccum acc;

  for (int o = 0; o < a.n; ++o) {
    const auto& no = nbr[o];
    // φ₂: owner o sums its own pair force; lower index owns the energy.
    for (const auto& e : no) {
      acc.min_r2 = std::min(acc.min_r2, e.r * e.r);  // B10 overlap probe (live path)
      double phi, dphi;
      sw_phi2(sp, e.r, phi, dphi);
      const double f_over_r = -dphi / e.r;  // force on o from pair {o, e.j}
      fx[o].add(f_over_r * e.dx);  fy[o].add(f_over_r * e.dy);  fz[o].add(f_over_r * e.dz);
      if (o < e.j) pe.add(phi);
    }
    // φ₃ with o as CENTER: every unordered wing pair; o owns f_i AND the energy.
    for (std::size_t x = 0; x < no.size(); ++x)
      for (std::size_t y = x + 1; y < no.size(); ++y) {
        SwVec3 fi, fj, fk;
        const double E3 = sw_triplet(sp, no[x].dx, no[x].dy, no[x].dz, no[x].r,
                                     no[y].dx, no[y].dy, no[y].dz, no[y].r, fi, fj, fk);
        fx[o].add(fi.x);  fy[o].add(fi.y);  fz[o].add(fi.z);
        pe.add(E3);  ++acc.n_triplets;
      }
    // φ₃ with o as a WING: for each neighbour i (a center candidate), replay every
    // triplet {i; o, m}. o sits in the j-slot; sw_triplet's wing symmetry guarantees
    // f_j(i;o,m) == f_k(i;m,o), so this matches the oracle's index-ordered scatter.
    for (const auto& ie : no) {
      const int i = ie.j;
      const double iox = -ie.dx, ioy = -ie.dy, ioz = -ie.dz, rio = ie.r;  // x_i − x_o
      for (const auto& me : nbr[i]) {
        if (me.j == o) continue;  // m ≠ o (the i-o bond is the o-wing, not a wing pair)
        SwVec3 fi, fj, fk;
        sw_triplet(sp, iox, ioy, ioz, rio, me.dx, me.dy, me.dz, me.r, fi, fj, fk);
        fx[o].add(fj.x);  fy[o].add(fj.y);  fz[o].add(fj.z);  // o is the j-wing
      }
    }
  }

  for (int i = 0; i < a.n; ++i) {
    a.fx[i] += Real(fx[i].value());
    a.fy[i] += Real(fy[i].value());
    a.fz[i] += Real(fz[i].value());
  }
  acc.pe = pe.value();
  return acc;
}

// --- descriptor (the plug-in seam). φ₂ is a SYMMETRIC pair pass; φ₃ is the 3-atom
// needs_transpose pass. effective_range = {2, true} — IDENTICAL to EAM (proven: a
// wing-atom's force reaches the other wing at 2·rcut, the same graph distance as EAM's
// density-donor). The virtual run_pass (zone dispatch) lands in T3; here it throws so
// the descriptor is live without faking the zone path.
template <typename Real>
struct SwPotential final : IManyBodyPotential<Real> {
  SwParams sw;
  explicit SwPotential(SwParams s = {}) : sw(std::move(s)) {}

  static constexpr PassDecl kPasses[2] = {
      {PassKind::Force, /*reuse*/ true, /*needs_transpose*/ false, false, 40},  // φ₂ pair
      {PassKind::Force, /*reuse*/ true, /*needs_transpose*/ true, false, 40},   // φ₃ triplet
  };
  std::span<const PassDecl> passes() const override { return {kPasses, 2}; }
  EffectiveRange effective_range() const override { return {2, /*symmetric_reach*/ true}; }

  void run_pass(int, const AtomSoA<Real>&, const PairGeom&, ManyBodyState<Real>&,
                const VisitPairs<Real>&) override {
    throw std::runtime_error("SwPotential::run_pass: the SW zone/ring dispatch lands "
                             "in T3 — PR-1 uses sw_run_fixed (transpose-replay)");
  }
};
template <typename Real>
constexpr PassDecl SwPotential<Real>::kPasses[2];

}  // namespace tdmd::potentials
