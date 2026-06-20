// M6 / MEAM-ladder Me3: serial zone-MEAM on the 3-zone TD residence. Design wf_0b517d66-4b2.
// The transpose-replay (meam_window_force replays the Me2 scatter over every directed bond whose
// center i is resident, scattering ONLY to OWNED targets ⇒ bitwise to meam_run_fixed_force). The
// GENUINE MEAM escalation: SCREENING candidate-completeness — every screening-k of every owned bond
// must reside in {S_{j-1},S_j,S_{j+1}} (√ebound·rc=1.04·rc<2·rc ⇒ {2,true} suffices). The diamond
// is BINARY-S (screening force DEAD) ⇒ the screening machinery is exercised ONLY by the constructed
// partial-screening SLAB (8 partial-S triples) — the FP64 oracle is the sole completeness witness.
//   G0 vacuity   the zone exercises the screening force (slab n_partial>0, force material)
//   G1 monolith  zone(z=1) forces+PE+counts bitwise == meam_run_fixed_force (diamond + slab)
//   G2 1-vs-z    z=1 bitwise == z∈{5,6,8}, identity+shuffled, free+PBC (diamond + slab)
//   G3 oracle ⭐  zone(z=6) within 1e-9 of meam_direct_fp64 + counts == (slab) — completeness witness
//   G-POISON ⭐   forward-only window + screening-drop DIVERGE from the oracle (residence/screening loss)
//   G-RESIDENCE  width<2·rc, thin periodic dim → THROW
//   G-B1         zone forces+PE bitwise under atom relabeling (slab)
//   G-MOMENTUM   the int64 screening-transpose momentum floor (slab int64 Σf vs FP64 round-off)
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/gen/partial_screen_slab.hpp"
#include "tdmd/potentials/meam.hpp"
#include "tdmd/potentials/meam_zone.hpp"

namespace core = tdmd::core;
namespace pot = tdmd::potentials;
using tdmd::gen::make_diamond_si;
using tdmd::gen::make_partial_screen_slab;

namespace {
double max_fdiff(const core::AtomSoA<double>& a, const core::AtomSoA<double>& b) {
  double m = 0; for (int i = 0; i < a.n; ++i) m = std::max({m, std::fabs(a.fx[i]-b.fx[i]), std::fabs(a.fy[i]-b.fy[i]), std::fabs(a.fz[i]-b.fz[i])}); return m;
}
double max_abs_force(const core::AtomSoA<double>& a) {
  double m = 0; for (int i = 0; i < a.n; ++i) m = std::max({m, std::fabs(a.fx[i]), std::fabs(a.fy[i]), std::fabs(a.fz[i])}); return m;
}
double sum_force_mag(const core::AtomSoA<double>& a) {
  double x=0,y=0,z=0; for (int i=0;i<a.n;++i){x+=a.fx[i];y+=a.fy[i];z+=a.fz[i];} return std::sqrt(x*x+y*y+z*z);
}
::testing::AssertionResult forces_bitwise(const core::AtomSoA<double>& a, const core::AtomSoA<double>& b) {
  for (int i = 0; i < a.n; ++i)
    if (a.fx[i] != b.fx[i] || a.fy[i] != b.fy[i] || a.fz[i] != b.fz[i])
      return ::testing::AssertionFailure() << "force mismatch at atom " << i;
  return ::testing::AssertionSuccess();
}
// diamond z-slab: pert=0.15 (at 0.20 the elongated diamond samples a sub-ZBL pair), Lz=12·5.431.
core::AtomSoA<double> diamond(core::Box& box, bool pbc_z) {
  auto a = make_diamond_si(2, 2, 12, 5.431, 0.15, box);
  box.periodic = {true, true, pbc_z};
  return a;
}
}  // namespace

// G0 — the zone exercises the bond order + the screening force (the slab's partial-S triples).
TEST(ZoneMeam, ScreeningForceExercised) {
  pot::MeamParams p; auto s = make_partial_screen_slab(8);
  const auto zd = core::ZoneDecomposition::build(s.atoms, s.box, 6, p.rc, 2);
  ASSERT_GE(zd.width, 2.0 * p.rc);
  core::AtomSoA<double> z = s.atoms; core::zero_forces(z);
  const auto acc = pot::meam_zone_pass(z, s.box, zd, p);
  EXPECT_GT(acc.n_screened_partial, 0) << "no partial-S pairs — the screening machinery is untested";
  EXPECT_GT(max_abs_force(z), 1.0);
}

// G1 — zone(z=1) is bitwise-identical to the whole-system meam_run_fixed_force (Me2 scatter), on
// BOTH the diamond (density+pair) AND the slab (incl. the Role-C screening 3rd-atom force).
TEST(ZoneMeam, MonolithMatchesRunFixedForce) {
  pot::MeamParams p;
  for (int which = 0; which < 2; ++which) {
    core::Box box; core::AtomSoA<double> a;
    if (which == 0) { a = diamond(box, true); } else { auto s = make_partial_screen_slab(8); a = s.atoms; box = s.box; }
    core::AtomSoA<double> ref = a, z = a; core::zero_forces(ref); core::zero_forces(z);
    const auto rr = pot::meam_run_fixed_force(ref, core::PairGeom(box, p.rc), p);
    const auto zd1 = core::ZoneDecomposition::build(z, box, 1, p.rc, 2);
    const auto rz = pot::meam_zone_pass(z, box, zd1, p);
    EXPECT_TRUE(forces_bitwise(z, ref)) << "which=" << which;
    EXPECT_EQ(rz.pe_embed, rr.pe_embed) << "which=" << which;
    EXPECT_EQ(rz.pe_pair, rr.pe_pair);
    EXPECT_EQ(rz.n_screened_partial, rr.n_screened_partial);
    EXPECT_EQ(rz.n_screened_zero, rr.n_screened_zero);
  }
}

// G2 — 1-vs-z bitwise (z=1 == z∈{5,6,8}) + zone-order invariance, free + PBC, diamond + slab.
TEST(ZoneMeam, OneVsZBitwise) {
  pot::MeamParams p;
  auto run = [&](core::AtomSoA<double> init, const core::Box& box) {
    core::AtomSoA<double> ref = init; core::zero_forces(ref);
    pot::meam_zone_pass(ref, box, core::ZoneDecomposition::build(ref, box, 1, p.rc, 2), p);
    for (int nz : {5, 6, 8}) {
      core::AtomSoA<double> z = init; core::zero_forces(z);
      const auto zd = core::ZoneDecomposition::build(z, box, nz, p.rc, 2);
      ASSERT_GE(zd.width, 2.0 * p.rc) << "nz=" << nz;
      pot::meam_zone_pass(z, box, zd, p);
      EXPECT_TRUE(forces_bitwise(z, ref)) << "1-vs-z nz=" << nz;
    }
    core::AtomSoA<double> sh = init; core::zero_forces(sh);
    const auto zd = core::ZoneDecomposition::build(sh, box, 6, p.rc, 2);
    std::vector<int> ord(6); std::iota(ord.begin(), ord.end(), 0);
    std::shuffle(ord.begin(), ord.end(), std::mt19937(7));
    pot::meam_zone_pass(sh, box, zd, p, ord);
    EXPECT_TRUE(forces_bitwise(sh, ref)) << "zone-order not invariant";
  };
  core::Box bp; auto dp = diamond(bp, true); run(dp, bp);
  core::Box bf; auto df = diamond(bf, false); run(df, bf);
  auto s = make_partial_screen_slab(8); run(s.atoms, s.box);
}

// G3 ⭐ — zone(z=6) forces+counts match the INDEPENDENT FP64 oracle (meam_direct_fp64) on the SLAB:
// the screening candidate-completeness witness (a dropped screening-k is invisible to 1-vs-z).
TEST(ZoneMeam, MatchesFp64OracleOnSlab) {
  pot::MeamParams p; auto s = make_partial_screen_slab(8);
  core::AtomSoA<double> oracle = s.atoms; core::zero_forces(oracle);
  const auto ro = pot::meam_direct_fp64(oracle, core::PairGeom(s.box, p.rc), p, true);
  EXPECT_GT(ro.n_screened_partial, 0);
  // z=6 AND z=8 (the tightest width==2·rc, ALL triples straddle a zone boundary — the worst
  // candidate-completeness case is oracle-checked, not only bitwise-to-z1).
  for (int nz : {6, 8}) {
    core::AtomSoA<double> z = s.atoms; core::zero_forces(z);
    const auto rz = pot::meam_zone_pass(z, s.box, core::ZoneDecomposition::build(z, s.box, nz, p.rc, 2), p);
    EXPECT_LT(max_fdiff(z, oracle), 1e-9) << "zone(nz=" << nz << ") ≠ FP64 oracle (dropped screening-k?)";
    EXPECT_NEAR(rz.pe_embed + rz.pe_pair, ro.pe_embed + ro.pe_pair, 1e-6) << "nz=" << nz;
    EXPECT_EQ(rz.n_screened_partial, ro.n_screened_partial) << "nz=" << nz;
  }
}

// G-POISON ⭐ — forward-only window (drops S_{j-1}) AND the screening-drop DIVERGE from the oracle:
// a residence/screening loss is real & detectable (1-vs-z is blind to it).
TEST(ZoneMeam, ForwardOnlyAndScreeningPoisonDiverge) {
  pot::MeamParams p; auto s = make_partial_screen_slab(8);
  core::AtomSoA<double> oracle = s.atoms; core::zero_forces(oracle);
  pot::meam_direct_fp64(oracle, core::PairGeom(s.box, p.rc), p, true);
  const auto zd = core::ZoneDecomposition::build(s.atoms, s.box, 6, p.rc, 2);
  core::AtomSoA<double> fwd = s.atoms; core::zero_forces(fwd);
  pot::meam_zone_pass(fwd, s.box, zd, p, /*symmetric=*/false);
  EXPECT_GT(max_fdiff(fwd, oracle), 1.0) << "forward-only did not diverge — residence gate blind";
  core::AtomSoA<double> drop = s.atoms; core::zero_forces(drop);
  pot::meam_zone_pass(drop, s.box, zd, p, /*symmetric=*/true, /*drop_class=*/1);
  EXPECT_GT(max_fdiff(drop, oracle), 1.0) << "screening-k drop did not diverge — k-loop unwired";
}

// G-RESIDENCE — geometry guards THROW (residence width, thin periodic dim).
TEST(ZoneMeam, GuardsThrow) {
  pot::MeamParams p; auto s = make_partial_screen_slab(8);
  // narrow build (understated rc=1.0, many zones ⇒ width < 2·rc=8) run with the real rc ⇒ THROW.
  const int nz_narrow = std::max(8, int(s.box.len(2) / 7.0));
  const auto zd_narrow = core::ZoneDecomposition::build(s.atoms, s.box, nz_narrow, 1.0, 2);
  if (zd_narrow.width < 2.0 * p.rc) {
    core::AtomSoA<double> z = s.atoms; core::zero_forces(z);
    EXPECT_THROW(pot::meam_zone_pass(z, s.box, zd_narrow, p), std::runtime_error);
  }
  core::Box thin = s.box; thin.hi[0] = thin.lo[0] + 1.5 * p.rc;  // Lx < 2·rc
  const auto zd6 = core::ZoneDecomposition::build(s.atoms, s.box, 6, p.rc, 2);
  core::AtomSoA<double> z2 = s.atoms; core::zero_forces(z2);
  EXPECT_THROW(pot::meam_zone_pass(z2, thin, zd6, p), std::runtime_error);
}

// G-B1 — zone forces+PE bitwise-invariant under atom relabeling (B1/INV-9), on the slab.
TEST(ZoneMeam, BitwiseUnderRelabeling) {
  pot::MeamParams p; auto s = make_partial_screen_slab(8);
  core::AtomSoA<double> a = s.atoms;
  core::AtomSoA<double> a1 = a; core::zero_forces(a1);
  const auto r1 = pot::meam_zone_pass(a1, s.box, core::ZoneDecomposition::build(a1, s.box, 6, p.rc, 2), p);
  std::vector<int> perm(a.n); std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), std::mt19937(2024));
  core::AtomSoA<double> a2; a2.resize(a.n);
  for (int i = 0; i < a.n; ++i) { a2.x[i]=a.x[perm[i]]; a2.y[i]=a.y[perm[i]]; a2.z[i]=a.z[perm[i]]; a2.type[i]=1; a2.mass[i]=28.0855; }
  core::zero_forces(a2);
  const auto r2 = pot::meam_zone_pass(a2, s.box, core::ZoneDecomposition::build(a2, s.box, 6, p.rc, 2), p);
  EXPECT_EQ(r1.pe_embed, r2.pe_embed); EXPECT_EQ(r1.pe_pair, r2.pe_pair);
  for (int i = 0; i < a.n; ++i) {
    ASSERT_EQ(a1.fx[perm[i]], a2.fx[i]) << "zone force not order-invariant at " << i;
    ASSERT_EQ(a1.fy[perm[i]], a2.fy[i]); ASSERT_EQ(a1.fz[perm[i]], a2.fz[i]);
  }
}

// G-MOMENTUM — the int64 screening-transpose momentum floor (the SW-T4/Tersoff finding): on the
// slab the screening k-scatter quantizes f_i,f_j,f_k independently ⇒ int64 Σf≠0 at the quantum,
// while the FP64 oracle Σf is round-off.
TEST(ZoneMeam, MomentumFloorIsInt64Quantization) {
  pot::MeamParams p; auto s = make_partial_screen_slab(8);
  core::AtomSoA<double> o = s.atoms, z = s.atoms; core::zero_forces(o); core::zero_forces(z);
  pot::meam_direct_fp64(o, core::PairGeom(s.box, p.rc), p, true);
  pot::meam_zone_pass(z, s.box, core::ZoneDecomposition::build(z, s.box, 6, p.rc, 2), p);
  EXPECT_LT(sum_force_mag(o), 1e-12) << "FP64 oracle Σf not round-off — a force-asymmetry bug";
  EXPECT_GT(sum_force_mag(z), 0.0) << "int64 Σf exactly zero — the finding would be vacuous";
  EXPECT_LT(sum_force_mag(z), 1e-9) << "int64 Σf far above the quantum — a bug";
}
