// M6 / SW-ladder T3: serial ZONE-SW on the 3-zone TD residence. Design wf_c90b13cf-202.
// Pre-registered gates (⭐ = catches an SW-specific failure INVISIBLE to 1-vs-z — every
// completeness/attribution claim is checked against the INDEPENDENT sw_direct_fp64, never
// another zone run):
//   G0 vacuity   the zone path exercises φ₃ (angular force material)
//   G1 monolith  sw_zone_pass(z=1) forces+PE bitwise == sw_run_fixed
//   G2 1-vs-z    z=1 bitwise == z∈{5,6,8}, identity + shuffled zone orders
//   G3 oracle ⭐  sw_zone_pass(z=6) forces within 1e-9 of sw_direct_fp64 + count equal & >0
//   G4 poison ⭐  forward-only window (drop S_{j-1}) DIVERGES from the oracle (residence teeth)
//   G5b PE ⭐     PE bitwise z-invariant {1,5,6,8} (only gate catching a local-key φ₂ bug)
//   G6 guards    width<2·rcut, thin periodic dim, bad order → THROW
//   G7 min_r2    overlap probe ≈ oracle
//   G8 relabel   zone forces+PE bitwise under atom relabeling (wing-canonical tie guard)
// Fixture: a diamond-Si z-SLAB (the 2³ cube is too short for any z>1 zone).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/potentials/sw.hpp"
#include "tdmd/potentials/sw_zone.hpp"

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
::testing::AssertionResult forces_bitwise_eq(const core::AtomSoA<double>& a,
                                             const core::AtomSoA<double>& b) {
  for (int i = 0; i < a.n; ++i)
    if (a.fx[i] != b.fx[i] || a.fy[i] != b.fy[i] || a.fz[i] != b.fz[i])
      return ::testing::AssertionFailure() << "force mismatch at atom " << i;
  return ::testing::AssertionSuccess();
}
}  // namespace

// G0 — the zone path exercises the angular term (total force vs φ₂-only, on a z=6 run).
TEST(ZoneSW, AngularTermExercised) {
  pot::SwParams sp;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  const auto zd = core::ZoneDecomposition::build(a, box, 6, sp.rcut(), 2);
  ASSERT_GE(zd.width, 2.0 * sp.rcut());
  core::AtomSoA<double> z = a;
  core::zero_forces(z);
  const auto acc = pot::sw_zone_pass(z, box, zd, sp);
  EXPECT_GT(acc.n_triplets, 0);
  EXPECT_GT(max_abs_force(z), 0.05);
}

// G1 — single-zone zone-SW is bitwise-identical to the whole-system sw_run_fixed.
TEST(ZoneSW, MonolithMatchesRunFixed) {
  pot::SwParams sp;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  core::AtomSoA<double> mono = a, ref = a;
  core::zero_forces(mono); core::zero_forces(ref);
  const auto zd1 = core::ZoneDecomposition::build(mono, box, 1, sp.rcut(), 2);
  const double pe_z = pot::sw_zone_pass(mono, box, zd1, sp).pe;
  const double pe_r = pot::sw_run_fixed(ref, box, sp).pe;
  EXPECT_EQ(pe_z, pe_r);
  EXPECT_TRUE(forces_bitwise_eq(mono, ref));
}

// G2 — 1-vs-z bitwise (z=1 == z∈{5,6,8}) + zone-processing-order invariance.
TEST(ZoneSW, OneVsZBitwiseAndOrderInvariant) {
  pot::SwParams sp;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  core::AtomSoA<double> ref = a;
  core::zero_forces(ref);
  const auto zd1 = core::ZoneDecomposition::build(ref, box, 1, sp.rcut(), 2);
  const double pe1 = pot::sw_zone_pass(ref, box, zd1, sp).pe;

  for (int nz : {5, 6, 8}) {
    core::AtomSoA<double> z = a;
    core::zero_forces(z);
    const auto zd = core::ZoneDecomposition::build(z, box, nz, sp.rcut(), 2);
    ASSERT_GE(zd.width, 2.0 * sp.rcut()) << "nz=" << nz;
    const double pe = pot::sw_zone_pass(z, box, zd, sp).pe;
    EXPECT_EQ(pe, pe1) << "PE not z-invariant nz=" << nz;
    EXPECT_TRUE(forces_bitwise_eq(z, ref)) << "1-vs-z force mismatch nz=" << nz;
  }
  // shuffled zone order — same result.
  core::AtomSoA<double> s = a;
  core::zero_forces(s);
  const auto zd = core::ZoneDecomposition::build(s, box, 6, sp.rcut(), 2);
  std::vector<int> ord(6);
  std::iota(ord.begin(), ord.end(), 0);
  std::shuffle(ord.begin(), ord.end(), std::mt19937(7));
  pot::sw_zone_pass(s, box, zd, sp, ord);
  EXPECT_TRUE(forces_bitwise_eq(s, ref)) << "zone-order not invariant";
}

// G3 ⭐ + G5 — zone forces match the INDEPENDENT FP64 oracle (the MB2 dropped-triplet
// witness; 1-vs-z is blind to it), and n_triplets is z-invariant AND equals the oracle.
// Also asserts the PBC fixture actually has SEAM triplets (wing bonds that wrap in z).
TEST(ZoneSW, MatchesFp64OracleAndCountZInvariant) {
  pot::SwParams sp;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);

  // seam existence: certify SEAM TRIPLETS (a center with ≥2 neighbours, ≥1 of which is a
  // neighbour ONLY via z-wrap), not just seam bonds — else the two-wing-PBC claim is
  // exercised vacuously (acceptance SHOULD-improve).
  const double Lz = box.len(2);
  const core::PairGeom geom(box, sp.rcut());
  const auto nbr = pot::sw_build_nbr(a, geom);
  long seam_triplets = 0;
  for (int i = 0; i < a.n; ++i) {
    if (nbr[i].size() < 2) continue;
    bool wrap = false;
    for (const auto& e : nbr[i])
      if (std::fabs(a.z[i] - a.z[e.j]) > 0.5 * Lz) wrap = true;
    if (wrap) ++seam_triplets;
  }
  ASSERT_GT(seam_triplets, 0) << "no PBC-seam triplets — two-wing wrap not exercised";

  core::AtomSoA<double> oracle = a;
  core::zero_forces(oracle);
  const auto ro = pot::sw_direct_fp64(oracle, box, sp, true);

  core::AtomSoA<double> z6 = a;
  core::zero_forces(z6);
  const auto zd = core::ZoneDecomposition::build(z6, box, 6, sp.rcut(), 2);
  const auto rz = pot::sw_zone_pass(z6, box, zd, sp);

  EXPECT_LT(max_force_diff(z6, oracle), 1e-9) << "zone ≠ independent FP64 oracle";
  // Gap-B (acceptance MUST-FIX): anchor PE to the INDEPENDENT oracle, not only to
  // sw_run_fixed (G1) — that is a structural sibling sharing the φ₃ center-owner
  // attribution, so a shared attribution bug would pass every other PE gate.
  EXPECT_NEAR(rz.pe, ro.pe, 1e-6) << "zone PE ≠ independent oracle PE (φ₃ attribution)";
  EXPECT_EQ(rz.n_triplets, ro.n_triplets) << "MB2: triplet count != oracle";
  EXPECT_GT(ro.n_triplets, 0);
}

// G4 ⭐ — the GEOMETRIC poison: forward-only window (drops S_{j-1}) DROPS triplets the
// independent oracle catches (a residence loss is real and detectable). Mirrors EAM
// ForwardOnlyWindowDiverges. perturb=0.45 ⇒ dropped triplets are angularly active.
TEST(ZoneSW, ForwardOnlyPoisonDivergesFromOracle) {
  pot::SwParams sp;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.45, box);

  core::AtomSoA<double> oracle = a;
  core::zero_forces(oracle);
  const auto ro = pot::sw_direct_fp64(oracle, box, sp, true);

  core::AtomSoA<double> bad = a;
  core::zero_forces(bad);
  const auto zd = core::ZoneDecomposition::build(bad, box, 6, sp.rcut(), 2);
  const auto rb = pot::sw_zone_pass(bad, box, zd, sp, /*symmetric=*/false);  // forward-only

  EXPECT_GT(max_force_diff(bad, oracle), 1e-3)
      << "forward-only did NOT diverge — the oracle gate is blind to a residence loss!";
  EXPECT_LT(rb.n_triplets, ro.n_triplets) << "forward-only dropped no triplets";
}

// G5b — PE bitwise z-invariant. NOTE (acceptance Gap-A): on the SERIAL path the φ₂
// global-key gate has NO teeth — zone_eam_window sorts the window by global id, so
// key[o]<key[e.b] ≡ local o<e.b, and a local-index bug would still pass here. The global
// key is DEFENSIVE for T3b, where the ring gathers the window UNSORTED (slot order) and
// the key becomes load-bearing; the witness there is the T3b ring-vs-serial PE bitwise gate.
TEST(ZoneSW, EnergyBitwiseZInvariant) {
  pot::SwParams sp;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  core::AtomSoA<double> r1 = a;
  core::zero_forces(r1);
  const double pe1 = pot::sw_zone_pass(
      r1, box, core::ZoneDecomposition::build(r1, box, 1, sp.rcut(), 2), sp).pe;
  for (int nz : {5, 6, 8}) {
    core::AtomSoA<double> z = a;
    core::zero_forces(z);
    const double pe = pot::sw_zone_pass(
        z, box, core::ZoneDecomposition::build(z, box, nz, sp.rcut(), 2), sp).pe;
    EXPECT_EQ(pe, pe1) << "PE not bitwise z-invariant nz=" << nz;
  }
}

// G6 — geometry guards THROW (residence width, thin periodic dim, bad zone order).
TEST(ZoneSW, GuardsThrow) {
  pot::SwParams sp;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);

  // residence: build a decomposition with an UNDERSTATED rcut (narrow zones; n=10 ⇒
  // width≈6.5 ∈ [2·1.0, 2·rcut)), then run sw_zone_pass with the REAL rcut ⇒ width <
  // 2·rcut ⇒ THROW.
  const auto zd_narrow = core::ZoneDecomposition::build(a, box, 10, 1.0, 2);
  core::AtomSoA<double> z = a; core::zero_forces(z);
  EXPECT_THROW(pot::sw_zone_pass(z, box, zd_narrow, sp), std::runtime_error);

  // thin transverse periodic dim (Lx < 2·rcut) — min-image ambiguous ⇒ THROW.
  core::Box thin = box; thin.hi[0] = thin.lo[0] + 1.5 * sp.rcut();  // Lx = 1.5·rcut < 2·rcut
  const auto zd6 = core::ZoneDecomposition::build(a, box, 6, sp.rcut(), 2);
  core::AtomSoA<double> z2 = a; core::zero_forces(z2);
  EXPECT_THROW(pot::sw_zone_pass(z2, thin, zd6, sp), std::runtime_error);

  // bad zone order (out of range) — validate_zone_order throws std::invalid_argument.
  core::AtomSoA<double> z3 = a; core::zero_forces(z3);
  EXPECT_THROW(pot::sw_zone_pass(z3, box, zd6, sp, std::vector<int>{0, 1, 2, 3, 4, 99}),
               std::invalid_argument);
}

// G7 — the overlap probe (min_r2) agrees with the independent oracle.
TEST(ZoneSW, MinR2MatchesOracle) {
  pot::SwParams sp;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  core::AtomSoA<double> oracle = a; core::zero_forces(oracle);
  const auto ro = pot::sw_direct_fp64(oracle, box, sp, true);
  core::AtomSoA<double> z = a; core::zero_forces(z);
  const auto rz = pot::sw_zone_pass(z, box, core::ZoneDecomposition::build(z, box, 6, sp.rcut(), 2), sp);
  EXPECT_NEAR(rz.min_r2, ro.min_r2, 1e-9);
  EXPECT_GT(rz.min_r2, 0.0);
}

// G8 — zone forces+PE bitwise-invariant under atom relabeling (the wing-canonical tie
// guard + the global-key attribution, exercised over the zone path on a tie-prone slab).
TEST(ZoneSW, BitwiseUnderRelabeling) {
  pot::SwParams sp;
  core::Box box;
  auto a = make_diamond_si(2, 2, kNz, 5.431, 0.45, box);
  core::AtomSoA<double> a1 = a; core::zero_forces(a1);
  const double pe1 = pot::sw_zone_pass(
      a1, box, core::ZoneDecomposition::build(a1, box, 6, sp.rcut(), 2), sp).pe;

  std::vector<int> perm(a.n);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), std::mt19937(2024));
  core::AtomSoA<double> a2; a2.resize(a.n);
  for (int i = 0; i < a.n; ++i) {
    a2.x[i] = a.x[perm[i]]; a2.y[i] = a.y[perm[i]]; a2.z[i] = a.z[perm[i]];
    a2.type[i] = a.type[perm[i]]; a2.mass[i] = a.mass[perm[i]];
  }
  core::zero_forces(a2);
  const double pe2 = pot::sw_zone_pass(
      a2, box, core::ZoneDecomposition::build(a2, box, 6, sp.rcut(), 2), sp).pe;

  EXPECT_EQ(pe1, pe2) << "PE not bitwise order-invariant on the zone path";
  for (int i = 0; i < a.n; ++i) {
    ASSERT_EQ(a1.fx[perm[i]], a2.fx[i]) << "zone force not order-invariant at " << i;
    ASSERT_EQ(a1.fy[perm[i]], a2.fy[i]);
    ASSERT_EQ(a1.fz[perm[i]], a2.fz[i]);
  }
}
