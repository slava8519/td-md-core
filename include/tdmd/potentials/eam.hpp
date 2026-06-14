#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
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

// --- (2) IManyBodyPotential instance. Math supplies eval_phi/eval_rhoa/eval_F
// + rcut + density_fracbits() (AnalyticEam now; EamSetfl spline in PR-E2).
template <typename Real, typename Math>
struct EamPotential final : IManyBodyPotential<Real> {
  Math math;
  explicit EamPotential(Math m) : math(std::move(m)) {}

  static constexpr PassDecl kPasses[3] = {
      {PassKind::Density, /*reuse*/ true, false, false, 44},
      {PassKind::Embedding, /*reuse*/ false, false, false, 30},  // local map
      {PassKind::Force, /*reuse*/ true, false, false, 40},
  };
  std::span<const PassDecl> passes() const override { return {kPasses, 3}; }
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

template <typename Real, typename Math>
constexpr PassDecl EamPotential<Real, Math>::kPasses[3];

// Drives EamPotential through its 3 passes over a direct O(N²) candidate walk,
// writing the fixed-point accumulators back into a.f. Mirrors PR-E3's engine
// with the trivial all-pairs topology — proves the interface + fixed-point ρ.
template <typename Real, typename Math>
EamAccum eam_run_fixed(AtomSoA<Real>& a, const Box& box,
                       EamPotential<Real, Math>& pot) {
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
  for (int p = 0; p < 3; ++p) pot.run_pass(p, a, geom, st, visit);

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
