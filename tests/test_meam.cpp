// M6 / MEAM-ladder Me1: single-element Si MEAM ENERGY (screening + partial densities ρ⁰⁻³ +
// embedding + pair φ). Design wf_63aa980a-eab. Energy-only (the ~470-line analytic force is Me2).
// Gates (⭐ = the MEAM-signature teeth):
//   G-SCREEN-POISON ⭐ — S≡1 (no k-product) DIVERGES from the screened energy + the golden
//   G-VACUITY          — screening is non-vacuous (n_screened_zero > 0; the Mask_triplet bites)
//   G-B1               — the int64 density-path energy is bitwise-invariant under atom relabeling
//   G-AUGT1            — the augt1 t1-normalization (t1_eff=2.82) is load-bearing
//   G-DENSITY          — the embed/pair split is well-formed (both nonzero, the right signs)
//   G-ZBL-GUARD        — a compressed config (a bond in the ZBL blend region) THROWS (deferred)
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/potentials/meam.hpp"

namespace core = tdmd::core;
namespace pot = tdmd::potentials;
using tdmd::gen::make_diamond_si;

namespace {
core::AtomSoA<double> diamond(double pert, core::Box& box, double a0 = 5.431) {
  auto a = make_diamond_si(2, 2, 2, a0, pert, box);
  box.periodic = {true, true, true};
  return a;
}
// the 3-atom partial-screening cluster (i-j screened by k, S≈0.50) — the SOLE screening-force witness.
core::AtomSoA<double> cluster(core::Box& box) {
  box.lo = {0, 0, 0}; box.hi = {24, 24, 24}; box.periodic = {false, false, false};
  core::AtomSoA<double> a; a.resize(3);
  const double pos[3][3] = {{10, 10, 10}, {13.2, 10, 10}, {11.6, 12.38, 10}};
  for (int i = 0; i < 3; ++i) { a.x[i] = pos[i][0]; a.y[i] = pos[i][1]; a.z[i] = pos[i][2]; a.type[i] = 1; a.mass[i] = 28.0855; a.id[i] = i + 1; }
  return a;
}
double max_fdiff(const core::AtomSoA<double>& a, const core::AtomSoA<double>& b) {
  double m = 0; for (int i = 0; i < a.n; ++i) m = std::max({m, std::fabs(a.fx[i]-b.fx[i]), std::fabs(a.fy[i]-b.fy[i]), std::fabs(a.fz[i]-b.fz[i])}); return m;
}
double sum_force_mag(const core::AtomSoA<double>& a) {
  double x=0,y=0,z=0; for (int i=0;i<a.n;++i){x+=a.fx[i];y+=a.fy[i];z+=a.fz[i];} return std::sqrt(x*x+y*y+z*z);
}
// FD-of-energy: the analytic force vs central-difference of meam_energy (independent of the force
// chain-rule — uses only the value helpers). Catches a wrong dGamma/drhodr/dt/dCfunc term.
double fd_of_energy_maxerr(core::AtomSoA<double>& a, const core::Box& box, const pot::MeamParams& p) {
  const core::PairGeom geom(box, p.rc);
  core::AtomSoA<double> g = a; core::zero_forces(g);
  pot::meam_direct_fp64(g, geom, p, true);
  const double h = 1e-6; double maxerr = 0;
  for (int i = 0; i < std::min(a.n, 16); ++i) {
    double* c[3] = {&a.x[i], &a.y[i], &a.z[i]};
    const double fan[3] = {g.fx[i], g.fy[i], g.fz[i]};
    for (int d = 0; d < 3; ++d) {
      const double x0 = *c[d];
      *c[d] = x0 + h; const double ep = pot::meam_energy(a, geom, p).pe;
      *c[d] = x0 - h; const double em = pot::meam_energy(a, geom, p).pe;
      *c[d] = x0;
      maxerr = std::max(maxerr, std::fabs(fan[d] - (-(ep - em) / (2 * h))));
    }
  }
  return maxerr;
}
}  // namespace

// G-SCREEN-POISON ⭐ — the Mask_triplet teeth: dropping the screening k-product (S≡1) materially
// changes the energy (the screened-out pairs re-enter). The agent measured ~17.9 eV divergence.
TEST(Meam, ScreeningPoisonDiverges) {
  core::Box box; auto a = diamond(0.20, box);
  pot::MeamParams p;
  const core::PairGeom geom(box, p.rc);
  const auto correct = pot::meam_energy(a, geom, p, /*poison=*/false);
  const auto poison = pot::meam_energy(a, geom, p, /*poison=*/true);
  EXPECT_GT(std::fabs(poison.pe - correct.pe), 1.0)
      << "S≡1 did not diverge — screening is vacuous or unwired";
  EXPECT_GT(correct.n_screened_zero, 0) << "no screened pairs on the active fixture";
}

// G-VACUITY — the screening's S=0 KILL-band is active (a third atom k DELETES an i-j bond). On
// the near-perfect diamond the partial band (0<S<1) is empty, so this gate covers ONLY the kill-
// band; the partial (smoothstep) multiply is validated separately by the cluster golden
// (Test_MEAM_LAMMPS.PartialScreeningClusterMatchesLammps — the acceptance MUST-FIX).
TEST(Meam, ScreeningKillBandIsNonVacuous) {
  core::Box box; auto a = diamond(0.20, box);
  pot::MeamParams p;
  const auto acc = pot::meam_energy(a, core::PairGeom(box, p.rc), p);
  EXPECT_GT(acc.n_screened_zero, 50) << "few/no screened pairs — Mask_triplet not biting";
}

// G-B1 — the int64 density-accumulation path gives an energy bitwise-invariant under atom
// relabeling (the angular arho1/2/3 sums are order-free in Q24.40; F(ρ̄) is then deterministic).
TEST(Meam, FixedPointBitwiseUnderRelabeling) {
  core::Box box; auto a = diamond(0.20, box);
  pot::MeamParams p;
  const core::PairGeom geom(box, p.rc);
  const double pe1 = pot::meam_run_fixed(a, geom, p).pe;

  std::vector<int> perm(a.n); std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), std::mt19937(2024));
  core::AtomSoA<double> b; b.resize(a.n);
  for (int i = 0; i < a.n; ++i) { b.x[i]=a.x[perm[i]]; b.y[i]=a.y[perm[i]]; b.z[i]=a.z[perm[i]]; b.type[i]=1; b.mass[i]=28.0855; }
  const double pe2 = pot::meam_run_fixed(b, geom, p).pe;
  EXPECT_EQ(pe1, pe2) << "int64 MEAM energy not bitwise order-invariant";
}

// G-AUGT1 — the augt1 t1-augmentation (t1_eff = t1 + 0.6·t3 = 2.82, not 3.30) is load-bearing:
// disabling it shifts the energy materially (the silent-normalization trap the design flagged).
TEST(Meam, Augt1IsLoadBearing) {
  core::Box box; auto a = diamond(0.20, box);
  pot::MeamParams on;  // augt1 = 1 (default)
  pot::MeamParams off = on; off.augt1 = 0; off.recompute();
  ASSERT_NEAR(on.t1_eff, 2.82, 1e-9);
  ASSERT_NEAR(off.t1_eff, 3.30, 1e-9);
  const core::PairGeom geom(box, on.rc);
  EXPECT_GT(std::fabs(pot::meam_energy(a, geom, off, false).pe - pot::meam_energy(a, geom, on, false).pe), 0.5)
      << "augt1 had no effect — the t1-normalization is not wired";
}

// G-DENSITY — the embed/pair split is well-formed (both materially nonzero; the embedding F>0
// for ρ̄·ln ρ̄ above 1, the pair φ·S net-attractive ⇒ a bound solid).
TEST(Meam, EmbedPairSplitWellFormed) {
  core::Box box; auto a = diamond(0.20, box);
  pot::MeamParams p;
  const auto acc = pot::meam_energy(a, core::PairGeom(box, p.rc), p);
  EXPECT_GT(std::fabs(acc.pe_embed), 1.0);
  EXPECT_GT(std::fabs(acc.pe_pair), 1.0);
  EXPECT_NEAR(acc.pe, acc.pe_embed + acc.pe_pair, 1e-9);
  EXPECT_LT(acc.pe, 0.0) << "diamond Si not bound";
}

// G-ZBL-GUARD — a compressed config (a bond inside the ZBL blend region r < re(1−1/α)) THROWS:
// the spline read would hit ZBL-polluted knots = a different, un-validated potential (deferred).
TEST(Meam, ZblBlendRegionThrows) {
  core::Box box; auto a = diamond(0.0, box, /*a0=*/3.0);  // heavily compressed ⇒ nn ≈ 1.30 < r_zbl
  pot::MeamParams p;
  EXPECT_THROW(pot::meam_energy(a, core::PairGeom(box, p.rc), p), std::runtime_error);
}

// ============================ Me2 — FORCE gates =============================================

// G-FD-FORCE ⭐ — the analytic force == −dU/dx (FD of meam_energy, the independent algebra
// witness) on BOTH the diamond (embedding+pair+density chains) AND the cluster (+ the screening
// derivative). The Te1 FD precedent; catches a wrong dGamma/drhodr/dt/dCfunc chain-rule term.
TEST(Meam, ForceMatchesFiniteDifferenceOfEnergy) {
  pot::MeamParams p;
  core::Box bd; auto d = diamond(0.20, bd);
  EXPECT_LT(fd_of_energy_maxerr(d, bd, p), 1e-5) << "diamond force ≠ −dU/dx";
  core::Box bc; auto c = cluster(bc);
  EXPECT_LT(fd_of_energy_maxerr(c, bc, p), 1e-5) << "cluster force ≠ −dU/dx (screening derivative)";
}

// G-SCREEN-FORCE-DEAD ⭐ — on the diamond, S is BINARY (∂S=0) ⇒ the screening-derivative k-loop
// is STRUCTURALLY DEAD: the POISON (drop the screening k-scatter) changes NOTHING. So the diamond
// golden is BLIND to a dCfunc/dscrfcn sign bug — the cluster gate below is the only witness.
TEST(Meam, ScreeningForceDeadOnDiamond) {
  pot::MeamParams p; core::Box box; auto a = diamond(0.20, box);
  const core::PairGeom geom(box, p.rc);
  core::AtomSoA<double> n = a, q = a; core::zero_forces(n); core::zero_forces(q);
  const auto rn = pot::meam_direct_fp64(n, geom, p, true, /*drop_class=*/0);
  pot::meam_direct_fp64(q, geom, p, true, /*drop_class=*/1);
  EXPECT_EQ(rn.n_screened_partial, 0) << "diamond has partial pairs — finding changed";
  EXPECT_EQ(max_fdiff(n, q), 0.0) << "screening k-loop NOT dead on diamond — finding wrong";
}

// G-SCREEN-FORCE-CLUSTER ⭐⭐ — the SOLE witness of the screening 3rd-atom force ∂S/∂x_k. On the
// partial-screening cluster the POISON (drop the k-loop) DIVERGES materially; atom-3's force is
// pure ∂S/∂x_k. (The LAMMPS-golden match of this force is in Test_MEAM_LAMMPS.)
TEST(Meam, ScreeningForceClusterHasTeeth) {
  pot::MeamParams p; core::Box box; auto a = cluster(box);
  const core::PairGeom geom(box, p.rc);
  core::AtomSoA<double> n = a, q = a; core::zero_forces(n); core::zero_forces(q);
  const auto rn = pot::meam_direct_fp64(n, geom, p, true, 0);
  pot::meam_direct_fp64(q, geom, p, true, 1);
  EXPECT_GT(rn.n_screened_partial, 0) << "cluster has no partial pair — fixture wrong";
  EXPECT_GT(max_fdiff(n, q), 1.0) << "screening force POISON did not diverge — k-loop unwired";
}

// G-B1-FORCE — the int64 force scatter is bitwise-invariant under atom relabeling (B1/INV-9).
TEST(Meam, ForceFixedPointBitwiseUnderRelabeling) {
  pot::MeamParams p; core::Box box; auto a = diamond(0.20, box);
  const core::PairGeom geom(box, p.rc);
  core::AtomSoA<double> a1 = a; core::zero_forces(a1); pot::meam_run_fixed_force(a1, geom, p);
  std::vector<int> perm(a.n); std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), std::mt19937(2024));
  core::AtomSoA<double> a2; a2.resize(a.n);
  for (int i = 0; i < a.n; ++i) { a2.x[i]=a.x[perm[i]]; a2.y[i]=a.y[perm[i]]; a2.z[i]=a.z[perm[i]]; a2.type[i]=1; a2.mass[i]=28.0855; a2.id[i]=a.id[perm[i]]; }
  core::zero_forces(a2); pot::meam_run_fixed_force(a2, geom, p);
  for (int i = 0; i < a.n; ++i) {
    ASSERT_EQ(a1.fx[perm[i]], a2.fx[i]) << "int64 force not order-invariant at " << i;
    ASSERT_EQ(a1.fy[perm[i]], a2.fy[i]); ASSERT_EQ(a1.fz[perm[i]], a2.fz[i]);
  }
}

// G-MOMENTUM — the int64 non-symmetric momentum floor (SW-T4/Tersoff): on the cluster the
// screening k-scatter quantizes f_i,f_j,f_k independently ⇒ int64 Σf≠0 at the quantum, while the
// FP64 oracle Σf is round-off. (On the diamond the writes are symmetric ⇒ int64 Σf is exact.)
TEST(Meam, MomentumFloorIsInt64Quantization) {
  pot::MeamParams p; core::Box box; auto a = cluster(box);
  const core::PairGeom geom(box, p.rc);
  core::AtomSoA<double> o = a, q = a; core::zero_forces(o); core::zero_forces(q);
  pot::meam_direct_fp64(o, geom, p, true);
  pot::meam_run_fixed_force(q, geom, p);
  EXPECT_LT(sum_force_mag(o), 1e-12) << "FP64 oracle Σf not round-off — a force-asymmetry bug";
  EXPECT_GT(sum_force_mag(q), 0.0) << "int64 Σf exactly zero — the finding would be vacuous";
  EXPECT_LT(sum_force_mag(q), 1e-9) << "int64 Σf far above the quantum — a bug";
}
