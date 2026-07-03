#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"  // PairGeom (single source of min-image/cutoff)
#include "tdmd/potentials/many_body.hpp"

// M6 PR-E1 — EAM drivers over a direct O(N²) candidate walk.
//   (1) eam_direct_fp64  — the FP64 ORACLE (density → embedding → force in FP64).
//       Used by the FD self-consistency test and the density sanity test. The
//       oracle must be FP64: a fixed-point ρ is piecewise-constant in x, which
//       would break finite differences (M6_EAM_MANYBODY_DESIGN §1.3).
//   (2) EamPotential<Real,Math> : IManyBodyPotential — the first concrete
//       multi-pass instance; run_pass writes the fixed-point accumulators (B1).
//       eam_run_fixed() drives it over the same O(N²) walk and is checked AGAINST
//       the oracle within the quantization bound — the interface + the
//       fixed-point density path, validated before PR-E3 wires them to the ring.
namespace tdmd::potentials {

using core::AtomSoA;
using core::Box;
using core::PairGeom;

struct EamAccum {
  double pe = 0.0;        // Σ_{i<j} φ + Σ_i F(ρ_i)
  double virial = 0.0;    // Σ_{i<j} r·F (pair-style virial of the total force)
  double min_r2 = 1e300;  // min over visited pairs (overlap probe, B10)
};

// --- (1) FP64 oracle. Fills a.f (ACCUMULATED — caller zeroes, core::zero_forces).
// with_forces=false computes the energy only (FD path). rho_out, if non-null,
// receives the per-atom density (Test_EAM_Density_Sanity).
template <typename Real, typename Math>
EamAccum eam_direct_fp64(AtomSoA<Real>& a, const Box& box, const Math& m,
                         bool with_forces, std::vector<double>* rho_out = nullptr) {
  const PairGeom geom(box, m.rcut);
  EamAccum acc;
  std::vector<double> rho(a.n, 0.0);

  // pass 1: density ρ_i = Σ_j ρ_a(r_ij) — each pair contributes to BOTH atoms.
  for (int i = 0; i < a.n; ++i)
    for (int j = i + 1; j < a.n; ++j) {
      double dx = a.x[i] - a.x[j], dy = a.y[i] - a.y[j], dz = a.z[i] - a.z[j], r2;
      const bool ok = geom.reduce(dx, dy, dz, r2);
      acc.min_r2 = std::min(acc.min_r2, r2);
      if (!ok) continue;
      double ra, dra;
      m.eval_rhoa(std::sqrt(r2), ra, dra);
      rho[i] += ra;
      rho[j] += ra;
    }

  // density-range guard (adversarial finding P1): a tabulated embedding F(ρ)
  // has a finite grid; ρ past it would be silently clamped to the last knot
  // (wrong force/energy under compression/melt, fully deterministic ⇒ invisible
  // to 1-vs-z). Closed-form Math returns +inf here (never trips). HALT instead.
  const double rho_cap = m.density_grid_max();
  for (int i = 0; i < a.n; ++i)
    if (rho[i] > rho_cap)
      throw std::runtime_error(
          "eam_direct_fp64: ρ exceeds the F(ρ) tabulation grid (compression "
          "beyond table) — extend rho_max or this would be silently clamped");

  // pass 2: embedding F(ρ_i) — local per-atom; F'(ρ_i) feeds the force pass.
  std::vector<double> fp(a.n);
  for (int i = 0; i < a.n; ++i) {
    double F, Fp;
    m.eval_F(rho[i], F, Fp);
    acc.pe += F;
    fp[i] = Fp;
  }

  // pass 3: pair energy φ + total force (pair + embedding), Newton-3.
  for (int i = 0; i < a.n; ++i)
    for (int j = i + 1; j < a.n; ++j) {
      double dx = a.x[i] - a.x[j], dy = a.y[i] - a.y[j], dz = a.z[i] - a.z[j], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
      const double r = std::sqrt(r2);
      double phi, dphi, ra, dra;
      m.eval_phi(r, phi, dphi);
      m.eval_rhoa(r, ra, dra);
      acc.pe += phi;
      if (!with_forces) continue;
      const double f_over_r = -(dphi + (fp[i] + fp[j]) * dra) / r;
      acc.virial += f_over_r * r2;
      a.fx[i] += f_over_r * dx;  a.fy[i] += f_over_r * dy;  a.fz[i] += f_over_r * dz;
      a.fx[j] -= f_over_r * dx;  a.fy[j] -= f_over_r * dy;  a.fz[j] -= f_over_r * dz;
    }

  if (rho_out) *rho_out = std::move(rho);
  return acc;
}

// PR-0a: the SINGLE source of the EAM descriptor (needed by eam_gpu_run_singlenode,
// which has no potential object). EamPotential::passes() returns THIS same span —
// no two copies of truth (tooth T-SRC). Was EamPotential<Real,Math>::kPasses, a
// per-instantiation member unreachable without the Math type; hoisted to a
// namespace-scope constant (kPasses had no consumer outside eam.hpp — verified).
// PR-1 (D5, К1): Density is FLIPPED to kAccumB1 + all three donor roles — the first
// shipped W-descriptor. Legal by the PR-0a sanction ("flipping EAM Density → kAccumB1
// in PR-1 breaks no consumer" — w_class declares a CAPABILITY; every ring keeps its
// monolithic path). The first runtime consumer is the donation driver's ledger assert
// (eam_donation.hpp) — WITHOUT the flip want_closure_mask(EAM)≡0 and that assert would
// be structurally dead (the 5-recidive class). Embedding stays kCOnly in the enum
// (SEMANTICALLY kWrapup — see WContract §10); Force stays kCOnly (C-phase).
// Rollback tooth: assert_eam_donation_descriptor (Т-13, test_eam_donation.cpp).
inline constexpr PassDecl kEamPassDecls[3] = {
    {PassKind::Density, /*reuse*/ true, false, false, 44, WClass::kAccumB1,
     uint8_t(donor_bit(DonorRole::kSelf) | donor_bit(DonorRole::kCrossLo) |
             donor_bit(DonorRole::kCrossHi))},
    {PassKind::Embedding, /*reuse*/ false, false, false, 30},  // local map
    {PassKind::Force, /*reuse*/ true, false, false, 40},
};
inline constexpr std::span<const PassDecl> eam_pass_decls() { return {kEamPassDecls, 3}; }

// PR-0a: the shared "symmetric EAM [Density,Embedding,Force]" gate — the accept/reject
// SEMANTICS of GpuEamWindowForce::assert_supported (eam_window_force_gpu.cuh) preserved;
// message texts normalized + `who`-prefixed (teeth check throw/no-throw, not strings —
// design R4). Consumers: GpuEamWindowForce (DELEGATES, К3 —
// one truth instead of parity-policing), CpuEamWindowForce (new), eam_gpu_run_
// singlenode. Calls validate_pass_decls (К9) ⇒ every EAM entry point also rejects
// wclass-illegal descriptors.
inline void assert_eam_symmetric_passes(std::span<const PassDecl> passes, const char* who) {
  validate_pass_decls(passes);
  auto fail = [&](std::string m) {
    throw std::runtime_error(std::string(who) + ": " + std::move(m));
  };
  if (passes.size() != 3)
    fail("only the EAM 3-pass (Density,Embedding,Force) sequence is supported — got "
         + std::to_string(passes.size()) + " passes");
  const PassKind want[3] = {PassKind::Density, PassKind::Embedding, PassKind::Force};
  for (std::size_t p = 0; p < 3; ++p) {
    if (passes[p].kind != want[p]) fail("unexpected pass kind at " + std::to_string(p));
    if (passes[p].needs_transpose)
      fail("needs_transpose UNSUPPORTED — the symmetric int64 accumulator q(j)=-q(i) "
           "cannot run a non-symmetric angular/bond-order term (force to a third atom k)");
    if (passes[p].iterative) fail("iterative pass (QEq/CG) UNSUPPORTED");
  }
}

// --- (2) IManyBodyPotential instance. Math supplies eval_phi/eval_rhoa/eval_F
// + rcut + density_fracbits() (AnalyticEam now; EamSetfl spline in PR-E2).
template <typename Real, typename Math>
struct EamPotential final : IManyBodyPotential<Real> {
  Math math;
  explicit EamPotential(Math m) : math(std::move(m)) {}

  std::span<const PassDecl> passes() const override { return {kEamPassDecls, 3}; }
  EffectiveRange effective_range() const override { return {2, /*symmetric_reach*/ true}; }

  void run_pass(int p, const AtomSoA<Real>& a, const PairGeom& /*geom*/,
                ManyBodyState<Real>& st, const VisitPairs<Real>& visit) override {
    if (p == 0) {  // density: per-atom int64 reduction (B1, order-free)
      visit([&](int i, int j, const PairGeomResult& g) {
        double v, dv;
        math.eval_rhoa(g.r, v, dv);
        (*st.rho)[i].add(v);  // no Newton-3 sharing — i sums its own neighbours,
        (*st.rho)[j].add(v);  // symmetric because ρ_a(r) is symmetric
      });
    } else if (p == 1) {  // embedding: local, no neighbours
      for (int i = 0; i < a.n; ++i) {
        const double rho = (*st.rho)[i].value();  // bit-identical double CPU↔GPU
        double F, Fp;
        math.eval_F(rho, F, Fp);
        (*st.fp)[i] = Fp;
        st.pe->add(F);
      }
    } else {  // force: symmetric bracket ⇒ q(j) = −q(i) (Newton-3 preserved)
      visit([&](int i, int j, const PairGeomResult& g) {
        double phi, dphi, ra, dra;
        math.eval_phi(g.r, phi, dphi);
        math.eval_rhoa(g.r, ra, dra);
        const double f_over_r = -(dphi + ((*st.fp)[i] + (*st.fp)[j]) * dra) / g.r;
        (*st.fx)[i].add(f_over_r * g.dx);  (*st.fy)[i].add(f_over_r * g.dy);  (*st.fz)[i].add(f_over_r * g.dz);
        (*st.fx)[j].add(-f_over_r * g.dx); (*st.fy)[j].add(-f_over_r * g.dy); (*st.fz)[j].add(-f_over_r * g.dz);
        st.pe->add(phi);
      });
    }
  }
};

// Drives EamPotential through its 3 passes over a direct O(N²) candidate walk,
// writing the fixed-point accumulators back into a.f. Mirrors PR-E3's engine
// with the trivial all-pairs topology — proves the interface + fixed-point ρ.
template <typename Real, typename Math>
EamAccum eam_run_fixed(AtomSoA<Real>& a, const Box& box,
                       EamPotential<Real, Math>& pot) {
  // P0 stopgap (adversarial finding): the fixed-point density buffer below is
  // hardwired to DensityAccum = FixedAccum<44>. A potential whose load-time
  // guard demands Q23.40 (density_fracbits()==40) would silently wrap int64 in
  // <44> (the per-contribution add() guard ≈ coincides with the <44> ceiling).
  // The dual-format dispatch lands with the ring in PR-E3; until then, HALT
  // loudly rather than corrupt ρ. Normal Al returns 44 (no-op).
  if (pot.math.density_fracbits() != 44)
    throw std::runtime_error(
        "eam_run_fixed: potential needs Q23.40 density, but this path is "
        "hardwired to Q19.44 — dual-format dispatch lands in PR-E3");
  const PairGeom geom(box, pot.math.rcut);
  std::vector<core::fixed::DensityAccum> rho(a.n);
  std::vector<double> fp(a.n, 0.0);
  std::vector<core::fixed::ForceAccum> fx(a.n), fy(a.n), fz(a.n);
  core::fixed::EnergyAccum pe;
  ManyBodyState<Real> st;
  st.rho = &rho; st.fp = &fp; st.fx = &fx; st.fy = &fy; st.fz = &fz; st.pe = &pe;

  const VisitPairs<Real> visit =
      [&](const std::function<void(int, int, const PairGeomResult&)>& fn) {
        for (int i = 0; i < a.n; ++i)
          for (int j = i + 1; j < a.n; ++j) {
            double dx = a.x[i] - a.x[j], dy = a.y[i] - a.y[j], dz = a.z[i] - a.z[j], r2;
            if (!geom.reduce(dx, dy, dz, r2)) continue;
            fn(i, j, PairGeomResult{dx, dy, dz, r2, std::sqrt(r2)});
          }
      };
  pot.run_pass(0, a, geom, st, visit);  // density
  const double rho_cap = pot.math.density_grid_max();  // P1 guard (see eam_direct_fp64)
  for (int i = 0; i < a.n; ++i)
    if (rho[i].value() > rho_cap)
      throw std::runtime_error("eam_run_fixed: ρ exceeds the F(ρ) tabulation grid");
  pot.run_pass(1, a, geom, st, visit);  // embedding
  pot.run_pass(2, a, geom, st, visit);  // force

  for (int i = 0; i < a.n; ++i) {
    a.fx[i] += Real(fx[i].value());
    a.fy[i] += Real(fy[i].value());
    a.fz[i] += Real(fz[i].value());
  }
  EamAccum acc;
  acc.pe = pe.value();
  return acc;
}

}  // namespace tdmd::potentials
