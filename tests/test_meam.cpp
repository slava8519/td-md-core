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
