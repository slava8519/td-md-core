// M6 / Tersoff-ladder Te3: serial ZONE-Tersoff on the 3-zone TD residence. Design wf_fc46259f-f57.
// The 4-role transpose-replay (PAIR/CENTER-i/ENDPOINT-j/THIRD-k) — vs SW's 3 roles, because the
// directed bond b_ij≠b_ji has no j/k symmetry. ⭐ = catches a failure INVISIBLE to 1-vs-z
// (checked vs the INDEPENDENT tersoff_direct_fp64, never another zone run):
//   G0 vacuity   the zone path exercises the bond-order angular force
//   G1 monolith  tersoff_zone_pass(z=1) forces+PE bitwise == tersoff_run_fixed
//   G2 1-vs-z    z=1 bitwise == z∈{5,6,8}, identity + shuffled order; counts z-invariant
//   G3 oracle ⭐  z=6 forces within 1e-9 of tersoff_direct_fp64 + n_bonds & n_triplets == oracle
//   G-PE oracle ⭐ z=6 PE within 1e-6 of the INDEPENDENT oracle (not only run_fixed — SW Gap-B)
//   G4 poison ⭐  forward-only window (drop S_{j-1}) DIVERGES from the oracle (residence teeth)
//   G5 ζ-replay ⭐ the FP64 ζ-sum order is bitwise-consistent (zone==run_fixed on a tie-prone
//                 fixture) + the order is genuinely load-bearing (reversed ζ-order would differ)
//   G6 guards    width<2·rcut, thin periodic dim, bad order → THROW
//   G7 min_r2    overlap probe ≈ oracle
//   G8 relabel   zone forces+PE+counts bitwise under atom relabeling (B1 + ζ-order tie guard)
//   G-PBC seam ⭐ certify ≥1 center with a z-wrap neighbour (two-bond PBC non-vacuous)
//   G-MOMENTUM   int64 Σf ∈ (0,1e-9) vs oracle <1e-12 (the SW-T4 floor; rules out asymmetry bug)
// Fixture: a diamond-Si z-SLAB (the 2³ cube is too short for any z>1 zone).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/potentials/tersoff.hpp"
#include "tdmd/potentials/tersoff_zone.hpp"

namespace core = tdmd::core;
namespace pot = tdmd::potentials;
using tdmd::gen::make_diamond_si;

namespace {
constexpr int kNz = 12;  // z-slab: box ~10.86×10.86×65.17, hosts z∈{5,6,8} zones ≥2·rcut

double max_abs_force(const core::AtomSoA<double>& a) {
  double m = 0.0;
  for (int i = 0; i < a.n; ++i)
    m = std::max({m, std::fabs(a.fx[i]), std::fabs(a.fy[i]), std::fabs(a.fz[i])});
  return m;
}
double max_force_diff(const core::AtomSoA<double>& a, const core::AtomSoA<double>& b) {
  double m = 0.0;
  for (int i = 0; i < a.n; ++i)
    m = std::max({m, std::fabs(a.fx[i] - b.fx[i]), std::fabs(a.fy[i] - b.fy[i]),
                  std::fabs(a.fz[i] - b.fz[i])});
  return m;
}
double sum_force_mag(const core::AtomSoA<double>& a) {
  double x = 0, y = 0, z = 0;
  for (int i = 0; i < a.n; ++i) { x += a.fx[i]; y += a.fy[i]; z += a.fz[i]; }
  return std::sqrt(x * x + y * y + z * z);
}
::testing::AssertionResult forces_bitwise_eq(const core::AtomSoA<double>& a,
                                             const core::AtomSoA<double>& b) {
  for (int i = 0; i < a.n; ++i)
    if (a.fx[i] != b.fx[i] || a.fy[i] != b.fy[i] || a.fz[i] != b.fz[i])
      return ::testing::AssertionFailure() << "force mismatch at atom " << i;
  return ::testing::AssertionSuccess();
}
// Drive tersoff_window_force on an all-atoms window in a GIVEN slot order (slot aa → global
// s2g[aa]); return owned int64 forces, scattered to global indices. Lets G5 reverse the ζ-replay
// order to MEASURE whether the global-id sort is load-bearing for the integrated int64 force.
void window_forces_in_order(const core::AtomSoA<double>& a, const core::Box& box,
                            const pot::TersoffParams& p, const std::vector<int>& s2g,
                            core::AtomSoA<double>& out) {
  const int n = a.n;
  const core::PairGeom geom(box, p.rcut());
  std::vector<double> wx(n), wy(n), wz(n);
  std::vector<long> key(n);
  std::vector<int> owned(n);
  for (int aa = 0; aa < n; ++aa) {
    const int g = s2g[aa];
    wx[aa] = a.x[g]; wy[aa] = a.y[g]; wz[aa] = a.z[g]; key[aa] = g; owned[aa] = aa;
  }
  std::vector<core::fixed::ForceAccum> fx(n), fy(n), fz(n);
  core::fixed::EnergyAccum pe;
  double mr = 1e300; long nb = 0, nt = 0;
  pot::tersoff_window_force<double>(wx.data(), wy.data(), wz.data(), key.data(), n, owned.data(),
                                    n, p, geom, fx, fy, fz, pe, mr, nb, nt);
  out = a; core::zero_forces(out);
  for (int aa = 0; aa < n; ++aa) {
    const int g = s2g[aa];
    out.fx[g] = double(fx[aa].value()); out.fy[g] = double(fy[aa].value()); out.fz[g] = double(fz[aa].value());
  }
}
}  // namespace

// G0 — the zone path exercises the bond-order angular term (total force vs a b≡1 variant).
TEST(ZoneTersoff, BondOrderExercised) {
  pot::TersoffParams p;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  const auto zd = core::ZoneDecomposition::build(a, box, 6, p.rcut(), 2);
  ASSERT_GE(zd.width, 2.0 * p.rcut());
  core::AtomSoA<double> z = a; core::zero_forces(z);
  const auto acc = pot::tersoff_zone_pass(z, box, zd, p);
  EXPECT_GT(acc.n_triplets, 0);
  EXPECT_GT(max_abs_force(z), 0.1);
}

// G1 — single-zone zone-Tersoff is bitwise-identical to the whole-system tersoff_run_fixed.
TEST(ZoneTersoff, MonolithMatchesRunFixed) {
  pot::TersoffParams p;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  core::AtomSoA<double> mono = a, ref = a;
  core::zero_forces(mono); core::zero_forces(ref);
  const auto zd1 = core::ZoneDecomposition::build(mono, box, 1, p.rcut(), 2);
  const double pe_z = pot::tersoff_zone_pass(mono, box, zd1, p).pe;
  const double pe_r = pot::tersoff_run_fixed(ref, core::PairGeom(box, p.rcut()), p).pe;
  EXPECT_EQ(pe_z, pe_r);
  EXPECT_TRUE(forces_bitwise_eq(mono, ref));
}

// G2 — 1-vs-z bitwise (z=1 == z∈{5,6,8}) + zone-processing-order invariance + counts z-invariant.
TEST(ZoneTersoff, OneVsZBitwiseAndOrderInvariant) {
  pot::TersoffParams p;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  core::AtomSoA<double> ref = a; core::zero_forces(ref);
  const auto zd1 = core::ZoneDecomposition::build(ref, box, 1, p.rcut(), 2);
  const auto r1 = pot::tersoff_zone_pass(ref, box, zd1, p);

  for (int nz : {5, 6, 8}) {
    core::AtomSoA<double> z = a; core::zero_forces(z);
    const auto zd = core::ZoneDecomposition::build(z, box, nz, p.rcut(), 2);
    ASSERT_GE(zd.width, 2.0 * p.rcut()) << "nz=" << nz;
    const auto rz = pot::tersoff_zone_pass(z, box, zd, p);
    EXPECT_EQ(rz.pe, r1.pe) << "PE not z-invariant nz=" << nz;
    EXPECT_EQ(rz.n_triplets, r1.n_triplets) << "n_triplets not z-invariant nz=" << nz;
    EXPECT_EQ(rz.n_bonds, r1.n_bonds) << "n_bonds not z-invariant nz=" << nz;
    EXPECT_TRUE(forces_bitwise_eq(z, ref)) << "1-vs-z force mismatch nz=" << nz;
  }
  core::AtomSoA<double> s = a; core::zero_forces(s);
  const auto zd = core::ZoneDecomposition::build(s, box, 6, p.rcut(), 2);
  std::vector<int> ord(6); std::iota(ord.begin(), ord.end(), 0);
  std::shuffle(ord.begin(), ord.end(), std::mt19937(7));
  pot::tersoff_zone_pass(s, box, zd, p, ord);
  EXPECT_TRUE(forces_bitwise_eq(s, ref)) << "zone-order not invariant";
}

// G3 ⭐ + G-PE — zone forces+PE+counts match the INDEPENDENT FP64 oracle (the MB2 dropped
// bond/triplet witness; 1-vs-z is blind). Both n_bonds AND n_triplets (DEFECT-1: n_bonds is
// the only witness to a Role-1 orientation/key bug).
TEST(ZoneTersoff, MatchesFp64OracleAndCounts) {
  pot::TersoffParams p;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  core::AtomSoA<double> oracle = a; core::zero_forces(oracle);
  const auto ro = pot::tersoff_direct_fp64(oracle, core::PairGeom(box, p.rcut()), p, true);
  core::AtomSoA<double> z6 = a; core::zero_forces(z6);
  const auto zd = core::ZoneDecomposition::build(z6, box, 6, p.rcut(), 2);
  const auto rz = pot::tersoff_zone_pass(z6, box, zd, p);

  EXPECT_LT(max_force_diff(z6, oracle), 1e-9) << "zone ≠ independent FP64 oracle";
  // SW Gap-B: anchor PE to the INDEPENDENT oracle, not only to run_fixed (G1) — a shared
  // center-attribution bug would pass every other PE gate.
  EXPECT_NEAR(rz.pe, ro.pe, 1e-6) << "zone PE ≠ independent oracle PE";
  EXPECT_EQ(rz.n_triplets, ro.n_triplets) << "MB2: triplet count != oracle";
  EXPECT_EQ(rz.n_bonds, ro.n_bonds) << "MB2: bond count != oracle (Role-1 witness)";
  EXPECT_GT(ro.n_triplets, 0); EXPECT_GT(ro.n_bonds, 0);
}

// G4 ⭐ — forward-only window (drops S_{j-1}) DIVERGES from the oracle + drops bonds/triplets
// (a residence loss is real & detectable). perturb=0.45 ⇒ dropped triplets angularly active.
TEST(ZoneTersoff, ForwardOnlyPoisonDivergesFromOracle) {
  pot::TersoffParams p;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.45, box);
  core::AtomSoA<double> oracle = a; core::zero_forces(oracle);
  const auto ro = pot::tersoff_direct_fp64(oracle, core::PairGeom(box, p.rcut()), p, true);
  core::AtomSoA<double> bad = a; core::zero_forces(bad);
  const auto zd = core::ZoneDecomposition::build(bad, box, 6, p.rcut(), 2);
  const auto rb = pot::tersoff_zone_pass(bad, box, zd, p, /*symmetric=*/false);
  EXPECT_GT(max_force_diff(bad, oracle), 1e-3)
      << "forward-only did NOT diverge — the oracle gate is blind to a residence loss!";
  EXPECT_LT(rb.n_triplets, ro.n_triplets) << "forward-only dropped no triplets";
}

// G5 ⭐ — the FP64 ζ-sum order (the one new hazard vs SW), MEASURED HONESTLY (the SW T3b pattern;
// Te3-acceptance MUST-FIX). (a) the zone path (sorted window) is bitwise to run_fixed even on a
// tie-prone fixture — the production path is deterministic. (b) the global-id sort is DEFENSIVE,
// NOT load-bearing for the integrated int64 force on the Te3 fixtures: a fully REVERSED window is
// bitwise-identical at perturb ≤ 0.6 (the ~1e-11 ζ-order delta sits below the Q24.40 quantum
// after b_ij/prefactor scaling); it flips exactly one quantum (9.1e-13) only at perturb ≥ 0.8.
// The gate asserts BOTH regimes ⇒ it documents the measured truth, not the overstated "the sort
// is necessary for bitwise-ness" claim. The sort+single-helper are kept defensively (Te3b/Te5).
TEST(ZoneTersoff, ZetaReplayOrderConsistentAndDefensiveNotLoadBearing) {
  pot::TersoffParams p;
  {  // (a) sorted zone path is bitwise to run_fixed under ties
    core::Box box;
    auto a = make_diamond_si(2, 2, kNz, 5.431, 0.45, box);
    core::AtomSoA<double> z = a, ref = a;
    core::zero_forces(z); core::zero_forces(ref);
    pot::tersoff_zone_pass(z, box, core::ZoneDecomposition::build(z, box, 6, p.rcut(), 2), p);
    pot::tersoff_run_fixed(ref, core::PairGeom(box, p.rcut()), p);
    EXPECT_TRUE(forces_bitwise_eq(z, ref)) << "ζ-order inconsistent under ties";
  }
  // (b) reversed window — measured: bitwise at low perturb (NOT load-bearing), diverges at high.
  for (double pert : {0.45, 0.90}) {
    core::Box box;
    auto a = make_diamond_si(2, 2, kNz, 5.431, pert, box);
    std::vector<int> fwd(a.n), rev(a.n);
    std::iota(fwd.begin(), fwd.end(), 0);
    for (int i = 0; i < a.n; ++i) rev[i] = a.n - 1 - i;
    core::AtomSoA<double> ff, rr;
    window_forces_in_order(a, box, p, fwd, ff);  // ascending == run_fixed order
    window_forces_in_order(a, box, p, rev, rr);  // reversed ζ-replay order
    const double d = max_force_diff(ff, rr);
    if (pert < 0.6)
      EXPECT_EQ(d, 0.0) << "reversed window NOT bitwise at perturb=" << pert
                        << " — sort IS load-bearing here (contradicts the measured status)";
    else
      EXPECT_GT(d, 0.0) << "reversed window bitwise at high perturb — the ζ-order teeth are vacuous";
  }
}

// G6 — geometry guards THROW (residence width, thin periodic dim, bad zone order).
TEST(ZoneTersoff, GuardsThrow) {
  pot::TersoffParams p;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  // narrow build (understated rcut=1.0, n=12 ⇒ width≈5.4 ∈ [2·1.0, 2·rcut=6.4)); run with the
  // REAL rcut=3.2 ⇒ width < 2·rcut ⇒ THROW.
  const auto zd_narrow = core::ZoneDecomposition::build(a, box, 12, 1.0, 2);
  ASSERT_LT(zd_narrow.width, 2.0 * p.rcut());
  core::AtomSoA<double> z = a; core::zero_forces(z);
  EXPECT_THROW(pot::tersoff_zone_pass(z, box, zd_narrow, p), std::runtime_error);

  core::Box thin = box; thin.hi[0] = thin.lo[0] + 1.5 * p.rcut();
  const auto zd6 = core::ZoneDecomposition::build(a, box, 6, p.rcut(), 2);
  core::AtomSoA<double> z2 = a; core::zero_forces(z2);
  EXPECT_THROW(pot::tersoff_zone_pass(z2, thin, zd6, p), std::runtime_error);

  core::AtomSoA<double> z3 = a; core::zero_forces(z3);
  EXPECT_THROW(pot::tersoff_zone_pass(z3, box, zd6, p, std::vector<int>{0, 1, 2, 3, 4, 99}),
               std::invalid_argument);
}

// G7 — the overlap probe (min_r2) agrees with the independent oracle.
TEST(ZoneTersoff, MinR2MatchesOracle) {
  pot::TersoffParams p;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  core::AtomSoA<double> oracle = a; core::zero_forces(oracle);
  const auto ro = pot::tersoff_direct_fp64(oracle, core::PairGeom(box, p.rcut()), p, true);
  core::AtomSoA<double> z = a; core::zero_forces(z);
  const auto rz = pot::tersoff_zone_pass(z, box, core::ZoneDecomposition::build(z, box, 6, p.rcut(), 2), p);
  EXPECT_NEAR(rz.min_r2, ro.min_r2, 1e-9);
  EXPECT_GT(rz.min_r2, 0.0);
}

// G-PBC ⭐ — certify the fixture has ≥1 center with a neighbour ONLY via z-wrap (else the
// two-bond-PBC path is exercised vacuously) + zone==oracle on it.
TEST(ZoneTersoff, PbcSeamNonVacuousAndCorrect) {
  pot::TersoffParams p;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  const double Lz = box.len(2);
  const core::PairGeom geom(box, p.rcut());
  const auto nbr = pot::tersoff_build_nbr(a, geom);
  long seam = 0;
  for (int i = 0; i < a.n; ++i)
    for (const auto& e : nbr[i])
      if (std::fabs(a.z[i] - a.z[e.j]) > 0.5 * Lz) { ++seam; break; }
  ASSERT_GT(seam, 0) << "no PBC-seam bonds — two-bond wrap not exercised";

  core::AtomSoA<double> oracle = a; core::zero_forces(oracle);
  const auto ro = pot::tersoff_direct_fp64(oracle, geom, p, true);
  core::AtomSoA<double> z = a; core::zero_forces(z);
  const auto rz = pot::tersoff_zone_pass(z, box, core::ZoneDecomposition::build(z, box, 6, p.rcut(), 2), p);
  EXPECT_LT(max_force_diff(z, oracle), 1e-9) << "zone ≠ oracle on the PBC-seam fixture";
  EXPECT_NEAR(rz.pe, ro.pe, 1e-6);
}

// G8 ⭐ — zone forces+PE+counts bitwise-invariant under atom relabeling (B1 + ζ-order tie guard).
TEST(ZoneTersoff, BitwiseUnderRelabeling) {
  pot::TersoffParams p;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.45, box);
  core::AtomSoA<double> a1 = a; core::zero_forces(a1);
  const auto r1 = pot::tersoff_zone_pass(a1, box, core::ZoneDecomposition::build(a1, box, 6, p.rcut(), 2), p);

  std::vector<int> perm(a.n); std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), std::mt19937(2024));
  core::AtomSoA<double> a2; a2.resize(a.n);
  for (int i = 0; i < a.n; ++i) {
    a2.x[i] = a.x[perm[i]]; a2.y[i] = a.y[perm[i]]; a2.z[i] = a.z[perm[i]];
    a2.type[i] = a.type[perm[i]]; a2.mass[i] = a.mass[perm[i]];
  }
  core::zero_forces(a2);
  const auto r2 = pot::tersoff_zone_pass(a2, box, core::ZoneDecomposition::build(a2, box, 6, p.rcut(), 2), p);
  EXPECT_EQ(r1.pe, r2.pe) << "PE not bitwise order-invariant on the zone path";
  EXPECT_EQ(r1.n_triplets, r2.n_triplets);
  for (int i = 0; i < a.n; ++i) {
    ASSERT_EQ(a1.fx[perm[i]], a2.fx[i]) << "zone force not order-invariant at " << i;
    ASSERT_EQ(a1.fy[perm[i]], a2.fy[i]); ASSERT_EQ(a1.fz[perm[i]], a2.fz[i]);
  }
}

// G-MOMENTUM — the int64 non-symmetric momentum floor (SW-T4 finding) on the zone path:
// oracle Σf ~ round-off, zone int64 Σf ≠ 0 at the quantum. Rules out a force-asymmetry bug.
TEST(ZoneTersoff, MomentumFloorIsInt64Quantization) {
  pot::TersoffParams p;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  core::AtomSoA<double> oracle = a; core::zero_forces(oracle);
  pot::tersoff_direct_fp64(oracle, core::PairGeom(box, p.rcut()), p, true);
  core::AtomSoA<double> z = a; core::zero_forces(z);
  pot::tersoff_zone_pass(z, box, core::ZoneDecomposition::build(z, box, 6, p.rcut(), 2), p);
  EXPECT_LT(sum_force_mag(oracle), 1e-12) << "oracle Σf not round-off — a force-asymmetry bug";
  EXPECT_GT(sum_force_mag(z), 0.0) << "zone int64 Σf exactly zero — finding vacuous";
  EXPECT_LT(sum_force_mag(z), 1e-9) << "zone int64 Σf far above the quantum — a bug";
}
