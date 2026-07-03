// PR-1 (W-contract ladder) — the SERIAL DONATION ORACLE for EAM.
// Design of record: docs/_meta/PR1_DONATION_ORACLE_DESIGN_2026-07-02.md §9.
// Every gate carries its kill-mutation in the comment. The G-A domain is EXPLICIT:
// forces (all atoms; written owned-only) + PE + min_r2 + OWNED-ρ bitwise vs
// zone_eam_pass; a FULL-ρ memcmp across halo is FALSE-RED and FORBIDDEN as a gate
// (the persistent accumulators legitimately hold edges the 3-zone window never saw).
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/potentials/eam.hpp"
#include "tdmd/potentials/eam_analytic.hpp"
#include "tdmd/potentials/eam_donation.hpp"
#include "tdmd/potentials/eam_spline.hpp"
#include "tdmd/potentials/eam_zone.hpp"

using namespace tdmd;
using potentials::AnalyticEam;
using potentials::DonationPoison;
using potentials::EamPotential;
using potentials::EamSetfl;

namespace {

std::string project_root() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT);
#else
  return ".";
#endif
}

// Slab FCC supercell (tall z) — the straddle fixture: nn=2.86 < rcut=3.0, atoms on
// every zone boundary for n in {5,6,8} (anti-vacuity is asserted via hook counters).
core::AtomSoA<double> make_fcc(int nx, int ny, int nz, double alat, core::Box& box,
                               bool periodic_z) {
  box.lo = {0, 0, 0};
  box.hi = {nx * alat, ny * alat, nz * alat};
  box.periodic = {true, true, periodic_z};
  const double b[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  std::vector<std::array<double, 3>> pos;
  for (int ix = 0; ix < nx; ++ix)
    for (int iy = 0; iy < ny; ++iy)
      for (int iz = 0; iz < nz; ++iz)
        for (auto& bb : b)
          pos.push_back({(ix + bb[0]) * alat, (iy + bb[1]) * alat, (iz + bb[2]) * alat});
  core::AtomSoA<double> a;
  a.resize(int(pos.size()));
  for (int i = 0; i < a.n; ++i) {
    a.x[i] = pos[i][0]; a.y[i] = pos[i][1]; a.z[i] = pos[i][2];
    a.type[i] = 1; a.mass[i] = 26.98;
  }
  return a;
}

// jitter ±0.06 keeps every atom inside its slab (g_eam = 0.0375 at n=8 is TIGHTER
// than the jitter... so the jitter is z-clamped: xy jitter free, z jitter small).
// NOTE (design §9): n=8 gives width 6.075 vs 2·rcut=6.0 — a deliberately TIGHT
// legal fixture (g_eam=0.0375); the z-jitter must stay below it.
void jitter(core::AtomSoA<double>& a, unsigned seed, double zj = 0.03) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> j(-0.06, 0.06);
  std::uniform_real_distribution<double> jz(-zj, zj);
  for (int i = 0; i < a.n; ++i) { a.x[i] += j(rng); a.y[i] += j(rng); a.z[i] += jz(rng); }
}

bool forces_bitwise_equal(const core::AtomSoA<double>& a, const core::AtomSoA<double>& b) {
  if (a.n != b.n) return false;
  for (int i = 0; i < a.n; ++i)
    if (a.fx[i] != b.fx[i] || a.fy[i] != b.fy[i] || a.fz[i] != b.fz[i]) return false;
  return true;
}
double max_abs_force(const core::AtomSoA<double>& a) {
  double m = 0;
  for (int i = 0; i < a.n; ++i)
    m = std::max({m, std::fabs(a.fx[i]), std::fabs(a.fy[i]), std::fabs(a.fz[i])});
  return m;
}
double max_force_dev(const core::AtomSoA<double>& a, const core::AtomSoA<double>& b) {
  double m = 0;
  for (int i = 0; i < a.n; ++i)
    m = std::max({m, std::fabs(a.fx[i] - b.fx[i]), std::fabs(a.fy[i] - b.fy[i]),
                  std::fabs(a.fz[i] - b.fz[i])});
  return m;
}

constexpr double kRcut = 3.0;
AnalyticEam<double> test_eam(double beta = 1.5) {
  AnalyticEam<double> m;
  m.rcut = kRcut;
  m.beta = beta;
  m.finalize();
  return m;
}

// the canonical scan, step-wise (mirrors zone_eam_pass_donated) — used by the
// hook/trace/poisoned tests that need the class API.
template <typename Real, typename Math, typename DensAccum, typename Trace,
          typename SelfHook, typename CrossHook>
potentials::EamAccum run_donated_scan(
    potentials::EamDonationPass<Real, Math, DensAccum, Trace>& pass, int n, bool pbc,
    SelfHook&& sh, CrossHook&& ch) {
  for (int j = 0; j < n; ++j) {
    const potentials::DonationBatches b = potentials::donation_layout(j, n, pbc);
    for (int i = 0; i < b.n_self; ++i) pass.donate_self(b.self[i], sh);
    for (int i = 0; i < b.n_cross; ++i) pass.donate_cross(b.cross[i][0], b.cross[i][1], ch);
    if (!(pbc && n > 1 && j == 0)) pass.finalize(j);
  }
  if (pbc && n > 1) pass.finalize(0);
  return pass.finish();
}

}  // namespace

// ===========================================================================
// Т-1 — G-A free-z: donated ≡ zone_eam_pass BITWISE (forces/PE/min_r2) + owned-ρ
// raw int64 ≡ a test-local O(N²) int64 recompute (the window-independent witness).
// Kill: Т-5/Т-7/Т-8 prove non-vacuity of this gate.
// ===========================================================================
TEST(EamDonation, GAFreeBitwise) {
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  for (unsigned seed : {1u, 2u}) {
    core::Box box;
    auto base = make_fcc(3, 3, 12, 4.05, box, /*periodic_z=*/false);
    jitter(base, seed);
    for (int nz : {1, 2, 3, 5, 6, 8}) {
      core::AtomSoA<double> ref = base, don = base;
      core::zero_forces(ref);
      core::zero_forces(don);
      const auto zd = core::ZoneDecomposition::build(ref, box, nz, kRcut, 2);
      const auto ao = potentials::zone_eam_pass(ref, box, zd, pot);
      const auto ad = potentials::zone_eam_pass_donated(don, box, zd, pot);
      EXPECT_GT(max_abs_force(ref), 0.05);
      EXPECT_TRUE(forces_bitwise_equal(ref, don)) << "nz=" << nz << " seed=" << seed;
      EXPECT_EQ(ao.pe, ad.pe) << "nz=" << nz;
      EXPECT_EQ(ao.min_r2, ad.min_r2) << "nz=" << nz;
    }
    // owned-ρ ≡ the full O(N²) int64 recompute (window-independent; the extra
    // non-adjacent-zone pairs it visits are all beyond rcut ⇒ literal zeros).
    // FULL-ρ vs the WINDOW's local ρ is NOT compared — false-red (design §3.6).
    {
      core::AtomSoA<double> don = base;
      core::zero_forces(don);
      const auto zd = core::ZoneDecomposition::build(don, box, 6, kRcut, 2);
      using DA = core::fixed::FixedAccum<44>;
      potentials::EamDonationPass<double, AnalyticEam<double>, DA> pass(don, box, zd, pot);
      run_donated_scan(pass, zd.n_zones, false, potentials::donation_detail::NullPairHook{},
                       potentials::donation_detail::NullPairHook{});
      std::vector<DA> full(std::size_t(don.n));
      const core::PairGeom geom(box, kRcut);
      for (int i = 0; i < don.n; ++i)
        for (int j = i + 1; j < don.n; ++j) {
          double dx = don.x[i] - don.x[j], dy = don.y[i] - don.y[j],
                 dz = don.z[i] - don.z[j], r2;
          if (!geom.reduce(dx, dy, dz, r2)) continue;
          double v, dv;
          m.eval_rhoa(std::sqrt(r2), v, dv);
          full[std::size_t(i)].add(v);
          full[std::size_t(j)].add(v);
        }
      for (int z = 0; z < zd.n_zones; ++z)
        for (std::size_t t = 0; t < zd.members[std::size_t(z)].size(); ++t)
          EXPECT_EQ(pass.state().rho[std::size_t(z)][t].raw,
                    full[std::size_t(zd.members[std::size_t(z)][t])].raw)
              << "owned-rho zone " << z << " member " << t;
    }
  }
}

// Т-2 — G-A PBC-z (12 cells = exact lattice period): n∈{1,5,6,8}; the n=1 case is
// the SLOT-DEDUP gate (eam_window_layout emits {0,0,0} under pbc — К7); the seam
// hook counter proves the (n-1,0) batch is non-vacuous. pbc n∈{2..4} is rejected
// one layer earlier by ZoneDecomposition::build (reach_mult=2) — documented here.
// Kill: drop the dedup ⇒ n=1 red; drop the seam ⇒ Т-6.
TEST(EamDonation, GAPbcBitwiseAndDedup) {
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box, /*periodic_z=*/true);
  jitter(base, 3);
  for (int nz : {1, 5, 6, 8}) {
    core::AtomSoA<double> ref = base, don = base;
    core::zero_forces(ref);
    core::zero_forces(don);
    const auto zd = core::ZoneDecomposition::build(ref, box, nz, kRcut, 2);
    const auto ao = potentials::zone_eam_pass(ref, box, zd, pot);
    const auto ad = potentials::zone_eam_pass_donated(don, box, zd, pot);
    EXPECT_TRUE(forces_bitwise_equal(ref, don)) << "pbc nz=" << nz;
    EXPECT_EQ(ao.pe, ad.pe);
    EXPECT_EQ(ao.min_r2, ad.min_r2);
  }
  // seam non-vacuity (n=6): count accepted pairs in the cross(5,0) batch
  {
    core::AtomSoA<double> don = base;
    core::zero_forces(don);
    const auto zd = core::ZoneDecomposition::build(don, box, 6, kRcut, 2);
    using DA = core::fixed::FixedAccum<44>;
    potentials::EamDonationPass<double, AnalyticEam<double>, DA> pass(don, box, zd, pot);
    long seam_pairs = 0;
    for (int j = 0; j < 6; ++j) {
      const auto b = potentials::donation_layout(j, 6, true);
      for (int i = 0; i < b.n_self; ++i) pass.donate_self(b.self[i]);
      for (int i = 0; i < b.n_cross; ++i) {
        const bool seam = (b.cross[i][0] == 5 && b.cross[i][1] == 0);
        if (seam)
          pass.donate_cross(5, 0, [&](long, long) { ++seam_pairs; });
        else
          pass.donate_cross(b.cross[i][0], b.cross[i][1]);
      }
      if (j != 0) pass.finalize(j);
    }
    pass.finalize(0);
    EXPECT_GT(seam_pairs, 0) << "the seam batch is vacuous — fixture broken";
  }
  // layered guard: periodic n∈{2..4} never reaches the driver
  for (int nz : {2, 3, 4})
    EXPECT_THROW(core::ZoneDecomposition::build(base, box, nz, kRcut, 2),
                 std::invalid_argument);
}

// Т-3 — G-EDGE (INV-8 bijection): the hook-collected multiset of accepted donated
// pairs ≡ the brute multiset of ALL min-image pairs within rcut, each EXACTLY once;
// anti-vacuity: self>0 and EVERY cross boundary>0. Runtime exactly-once: a repeated
// batch THROWS; with ledger_off a duplicated batch shows cnt==2.
TEST(EamDonation, GEdgeBijection) {
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box, /*periodic_z=*/false);
  jitter(base, 4);
  const auto zd = core::ZoneDecomposition::build(base, box, 6, kRcut, 2);
  using DA = core::fixed::FixedAccum<44>;

  std::map<std::pair<long, long>, int> seen;
  std::vector<long> self_cnt(6, 0), cross_cnt(6, 0);
  {
    core::AtomSoA<double> don = base;
    core::zero_forces(don);
    potentials::EamDonationPass<double, AnalyticEam<double>, DA> pass(don, box, zd, pot);
    int cur_zone = -1, cur_edge = -1;
    auto sh = [&](long a, long b) {
      ++seen[{std::min(a, b), std::max(a, b)}];
      ++self_cnt[std::size_t(cur_zone)];
    };
    auto ch = [&](long a, long b) {
      ++seen[{std::min(a, b), std::max(a, b)}];
      ++cross_cnt[std::size_t(cur_edge)];
    };
    for (int j = 0; j < 6; ++j) {
      const auto b = potentials::donation_layout(j, 6, false);
      for (int i = 0; i < b.n_self; ++i) { cur_zone = b.self[i]; pass.donate_self(b.self[i], sh); }
      for (int i = 0; i < b.n_cross; ++i) {
        cur_edge = b.cross[i][0];
        pass.donate_cross(b.cross[i][0], b.cross[i][1], ch);
      }
      pass.finalize(j);
    }
    pass.finish();
  }
  // brute oracle of ALL accepted pairs
  std::map<std::pair<long, long>, int> brute;
  const core::PairGeom geom(box, kRcut);
  for (int i = 0; i < base.n; ++i)
    for (int j = i + 1; j < base.n; ++j) {
      double dx = base.x[i] - base.x[j], dy = base.y[i] - base.y[j],
             dz = base.z[i] - base.z[j], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
      ++brute[{long(i), long(j)}];
    }
  EXPECT_EQ(seen.size(), brute.size()) << "pair-set mismatch";
  for (const auto& [k, c] : brute) {
    auto it = seen.find(k);
    ASSERT_NE(it, seen.end()) << "missing pair " << k.first << "," << k.second;
    EXPECT_EQ(it->second, 1) << "pair visited " << it->second << " times";
  }
  for (int z = 0; z < 6; ++z) EXPECT_GT(self_cnt[std::size_t(z)], 0) << "self " << z;
  for (int e = 0; e < 5; ++e) EXPECT_GT(cross_cnt[std::size_t(e)], 0) << "edge " << e;

  // exactly-once runtime: repeating a batch throws (INV-8)
  {
    core::AtomSoA<double> don = base;
    core::zero_forces(don);
    potentials::EamDonationPass<double, AnalyticEam<double>, DA> pass(don, box, zd, pot);
    pass.donate_self(0);
    EXPECT_THROW(pass.donate_self(0), std::runtime_error);
    pass.donate_cross(0, 1);
    EXPECT_THROW(pass.donate_cross(0, 1), std::runtime_error);
    // and cross(a,b) demands the cyclic successor
    EXPECT_THROW(pass.donate_cross(2, 4), std::runtime_error);
  }
}

// Т-4 — G-ORACLE: donated forces vs the INDEPENDENT full-system FP64 EAM (shares
// NO window/zone/donation logic) < 1e-10, free + PBC. The non-negotiable witness
// of a pair lost INSIDE a batch (the ledger, 1-vs-z and run-to-run are blind).
// Kill: Т-7 drives this above 1e-3.
TEST(EamDonation, FP64OracleAgrees) {
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  for (bool pbc : {false, true}) {
    core::Box box;
    auto base = make_fcc(3, 3, 12, 4.05, box, pbc);
    jitter(base, 5);
    core::AtomSoA<double> don = base, fp = base;
    core::zero_forces(don);
    core::zero_forces(fp);
    const auto zd = core::ZoneDecomposition::build(don, box, pbc ? 5 : 6, kRcut, 2);
    potentials::zone_eam_pass_donated(don, box, zd, pot);
    potentials::eam_direct_fp64(fp, box, m, /*with_forces=*/true);
    EXPECT_LT(max_force_dev(don, fp), 1e-10) << "pbc=" << pbc;
  }
}

// Т-5 — P-DROP: a skipped cross batch (2,3) ⇒ (а) finalize(2) LEDGER-THROW;
// (б) with ledger_off the physics itself is red vs both oracles (two layers:
// bookkeeping AND physics). This is the drop-a-batch tooth of G-A/G-EDGE.
TEST(EamDonation, PoisonDroppedCrossBatch) {
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box, /*periodic_z=*/false);
  jitter(base, 6);
  const auto zd = core::ZoneDecomposition::build(base, box, 6, kRcut, 2);
  using DA = core::fixed::FixedAccum<44>;

  auto scan_dropping_23 = [&](core::AtomSoA<double>& don, const DonationPoison& p) {
    potentials::EamDonationPass<double, AnalyticEam<double>, DA> pass(
        don, box, zd, pot, nullptr, -1.0, p);
    for (int j = 0; j < 6; ++j) {
      const auto b = potentials::donation_layout(j, 6, false);
      for (int i = 0; i < b.n_self; ++i) pass.donate_self(b.self[i]);
      for (int i = 0; i < b.n_cross; ++i)
        if (!(b.cross[i][0] == 2 && b.cross[i][1] == 3))
          pass.donate_cross(b.cross[i][0], b.cross[i][1]);
      pass.finalize(j);  // throws at j=2 unless ledger_off
    }
    return pass.finish();
  };
  {
    core::AtomSoA<double> don = base;
    core::zero_forces(don);
    EXPECT_THROW(scan_dropping_23(don, {}), std::runtime_error);
  }
  {
    core::AtomSoA<double> don = base, ref = base, fp = base;
    core::zero_forces(don);
    core::zero_forces(ref);
    core::zero_forces(fp);
    DonationPoison p;
    p.ledger_off = true;
    scan_dropping_23(don, p);
    potentials::zone_eam_pass(ref, box, zd, pot);
    potentials::eam_direct_fp64(fp, box, m, true);
    EXPECT_FALSE(forces_bitwise_equal(don, ref)) << "dropped batch invisible to G-A?!";
    EXPECT_GT(max_force_dev(don, fp), 1e-3) << "dropped batch invisible to FP64?!";
  }
}

// Т-6 — P-LATE-SEAM (pbc): executing the seam AFTER finalize(n-1) is the refuted
// "pass tail" variant. (а) the ledger THROWS at finalize(n-1) (CrossHi missing);
// (б) with ledger_off the owned(n-1) forces diverge BITWISE ⇒ the deadline is
// load-bearing physics, not bookkeeping.
TEST(EamDonation, PoisonLateSeam) {
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box, /*periodic_z=*/true);
  jitter(base, 7);
  const auto zd = core::ZoneDecomposition::build(base, box, 6, kRcut, 2);
  using DA = core::fixed::FixedAccum<44>;

  auto scan_late_seam = [&](core::AtomSoA<double>& don, const DonationPoison& p) {
    potentials::EamDonationPass<double, AnalyticEam<double>, DA> pass(
        don, box, zd, pot, nullptr, -1.0, p);
    for (int j = 0; j < 6; ++j) {
      const auto b = potentials::donation_layout(j, 6, true);
      for (int i = 0; i < b.n_self; ++i) pass.donate_self(b.self[i]);
      for (int i = 0; i < b.n_cross; ++i)
        if (!(b.cross[i][0] == 5 && b.cross[i][1] == 0))
          pass.donate_cross(b.cross[i][0], b.cross[i][1]);
      if (j != 0) pass.finalize(j);  // finalize(5) BEFORE the seam — the poison
    }
    pass.donate_cross(5, 0);  // late: after finalize(5)
    pass.finalize(0);
    return pass.finish();
  };
  {
    core::AtomSoA<double> don = base;
    core::zero_forces(don);
    EXPECT_THROW(scan_late_seam(don, {}), std::runtime_error);
  }
  {
    core::AtomSoA<double> don = base, ref = base;
    core::zero_forces(don);
    core::zero_forces(ref);
    DonationPoison p;
    p.ledger_off = true;
    scan_late_seam(don, p);
    potentials::zone_eam_pass(ref, box, zd, pot);
    bool zone5_diverged = false;
    for (int g : zd.members[5])
      zone5_diverged = zone5_diverged || don.fx[g] != ref.fx[g] || don.fy[g] != ref.fy[g] ||
                       don.fz[g] != ref.fz[g];
    EXPECT_TRUE(zone5_diverged) << "late seam invisible on owned(n-1) — deadline vacuous?!";
  }
}

// Т-7 — skip_first_pair: the ledger stays CLEAN (whole batches executed) yet the
// physics is red vs G-A and FP64 ⇒ the ledger's honest boundary is PINNED (it sees
// batches, not pairs; the FP64 oracle is the non-negotiable intra-batch witness).
TEST(EamDonation, PoisonSkipFirstPairLedgerBlind) {
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box, /*periodic_z=*/false);
  jitter(base, 8);
  const auto zd = core::ZoneDecomposition::build(base, box, 6, kRcut, 2);
  core::AtomSoA<double> don = base, ref = base, fp = base;
  core::zero_forces(don);
  core::zero_forces(ref);
  core::zero_forces(fp);
  DonationPoison p;
  p.skip_first_pair = true;
  EXPECT_NO_THROW(potentials::zone_eam_pass_donated(don, box, zd, pot, p));  // ledger blind
  potentials::zone_eam_pass(ref, box, zd, pot);
  potentials::eam_direct_fp64(fp, box, m, true);
  EXPECT_FALSE(forces_bitwise_equal(don, ref));
  EXPECT_GT(max_force_dev(don, fp), 1e-6);
}

// Т-8 — one_sided: the KNOB-driven half-bug (donating into one end only) breaks the
// two-end lemma ⇒ G-A red. NOTE (acceptance finding): this is a PROPERTY tooth for
// "one-sidedness diverges", NOT a regression tooth for "the default is two-sided" — a
// PERMANENTLY one-sided eam_donate_cross is caught by Т-1 (positive G-A), not here
// (a permanently-broken default makes the poisoned run indistinguishable, so this
// EXPECT_FALSE still holds). Т-1 is the regression guard; Т-8 pins the lemma.
TEST(EamDonation, PoisonOneSided) {
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box, /*periodic_z=*/false);
  jitter(base, 9);
  const auto zd = core::ZoneDecomposition::build(base, box, 6, kRcut, 2);
  core::AtomSoA<double> don = base, ref = base;
  core::zero_forces(don);
  core::zero_forces(ref);
  DonationPoison p;
  p.one_sided = true;
  potentials::zone_eam_pass_donated(don, box, zd, pot, p);
  potentials::zone_eam_pass(ref, box, zd, pot);
  EXPECT_FALSE(forces_bitwise_equal(don, ref)) << "one-sided donation invisible?!";
}

// Т-9 — G-ORDER (the executable W-1 witness): a WILDLY reordered batch schedule
// (all batches up-front, positions reversed, cross before self) is BITWISE
// identical to the canonical scan — the §5.2(iv) freedom is real. Kill: an FP
// (non-int64) accumulator in the executors.
TEST(EamDonation, GOrderBatchScheduleFreedom) {
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box, /*periodic_z=*/false);
  jitter(base, 10);
  const auto zd = core::ZoneDecomposition::build(base, box, 6, kRcut, 2);
  using DA = core::fixed::FixedAccum<44>;

  core::AtomSoA<double> canon = base, wild = base;
  core::zero_forces(canon);
  core::zero_forces(wild);
  const auto ac = potentials::zone_eam_pass_donated(canon, box, zd, pot);
  potentials::EamDonationPass<double, AnalyticEam<double>, DA> pass(wild, box, zd, pot);
  for (int e = 4; e >= 0; --e) pass.donate_cross(e, e + 1);  // cross first, reversed
  for (int k = 5; k >= 0; --k) pass.donate_self(k);          // then selfs, reversed
  for (int j = 0; j < 6; ++j) pass.finalize(j);
  const auto aw = pass.finish();
  EXPECT_TRUE(forces_bitwise_equal(canon, wild)) << "batch order leaked into the result";
  EXPECT_EQ(ac.pe, aw.pe);
  EXPECT_EQ(ac.min_r2, aw.min_r2);
}

// Т-10 — G-HALT (rho_cap): (а) pins the ORACLE's cap-throw first (zone_eam_pass had
// no cap-tripping test — the eam_direct_fp64 one exists, EamSpline.DensityBeyondGrid-
// Halts); (б) the donated path throws on the SAME config with the owner-zone message
// (К6 — the deterministic attribution PR-2 inherits); (в) near-cap-below: NEITHER
// throws; (г) old-vs-new attribution DIFFERENCE is legitimate and NOT gated.
// Kill: disable the owned-cap check in eam_window_force_from_rho ⇒ (б) silent.
TEST(EamDonation, GHaltRhoCapTriggerSetEquivalence) {
  const auto m = test_eam();
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box, /*periodic_z=*/false);
  jitter(base, 11);
  // measure the actual max full density, then build tight/loose setfl grids
  double rho_max_actual = 0.0;
  {
    core::AtomSoA<double> probe = base;
    core::zero_forces(probe);
    std::vector<double> rho;
    potentials::eam_direct_fp64(probe, box, m, false, &rho);
    for (double r : rho) rho_max_actual = std::max(rho_max_actual, r);
  }
  ASSERT_GT(rho_max_actual, 0.1);
  const auto tight = EamSetfl<double>::from_analytic(m, 2000, 2000, rho_max_actual * 0.5);
  const auto loose = EamSetfl<double>::from_analytic(m, 2000, 2000, rho_max_actual * 1.5);
  const EamPotential<double, EamSetfl<double>> pot_tight(tight), pot_loose(loose);
  const auto zd = core::ZoneDecomposition::build(base, box, 6, tight.rcut, 2);

  // (а) the oracle throws — pinned HERE for zone_eam_pass itself
  {
    core::AtomSoA<double> a = base;
    core::zero_forces(a);
    EXPECT_THROW(potentials::zone_eam_pass(a, box, zd, pot_tight), std::runtime_error);
  }
  // (б) the donated path throws on the same config, attributed to an owner ZONE
  {
    core::AtomSoA<double> a = base;
    core::zero_forces(a);
    try {
      potentials::zone_eam_pass_donated(a, box, zd, pot_tight);
      FAIL() << "donated path missed the cap";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find("zone"), std::string::npos)
          << "owner-zone attribution missing: " << e.what();
    }
  }
  // (в) near-cap-below: neither path throws (the trigger set is {full ρ > cap})
  {
    core::AtomSoA<double> a = base, b = base;
    core::zero_forces(a);
    core::zero_forces(b);
    EXPECT_NO_THROW(potentials::zone_eam_pass(a, box, zd, pot_loose));
    EXPECT_NO_THROW(potentials::zone_eam_pass_donated(b, box, zd, pot_loose));
    EXPECT_TRUE(forces_bitwise_equal(a, b));
  }
}

// Т-11 — G-LCLOSE + D7 (the five measurements, design §8; the band semantics are
// the MEASURED truth of the verification: G-A goes RED on (g_eam, g_pair] via
// divergence class 3 — the persistent ρ of a read halo atom is STRICTLY FULLER
// than the window's). Hand-built 3-atom chain, membership pre-drift by hand.
namespace {
struct ReadTrace {  // records every accepted pass-3 neighbour read: (zone, key, raw)
  struct Rec { int zone; long key; long long raw; };
  std::vector<Rec> recs;
  void on_neighbor_read(int z, long k, long long r) { recs.push_back({z, k, r}); }
};

// o(z=12.3)∈zone1, b(z=14.3)∈zone2, d — a MEMBER of zone3 drifted DOWN to z_d.
// width=6.2, rcut=3.0 ⇒ g_eam=0.1, g_pair=1.6. |o-b|=2.0, |b-d|=|z_d-14.3|.
core::AtomSoA<double> chain3(core::Box& box, double z_d) {
  box.lo = {0, 0, 0};
  box.hi = {10, 10, 31};
  box.periodic = {false, false, false};
  core::AtomSoA<double> a;
  a.resize(3);
  const double zs[3] = {12.3, 14.3, z_d};
  for (int i = 0; i < 3; ++i) {
    a.x[i] = 5.0; a.y[i] = 5.0; a.z[i] = zs[i];
    a.type[i] = 1; a.mass[i] = 26.98;
  }
  return a;
}
core::ZoneDecomposition chain_zd() {
  core::ZoneDecomposition zd;
  zd.n_zones = 5;
  zd.width = 6.2;
  zd.members = {{}, {0}, {1}, {2}, {}};  // zones 0 and 4 empty (also the Т-14 shape)
  return zd;
}
}  // namespace

TEST(EamDonation, GLCloseAndD7DriftBand) {
  const auto m = test_eam();  // rcut = 3.0
  EamPotential<double, AnalyticEam<double>> pot(m);
  const auto zd = chain_zd();
  const double g_pair = 0.5 * (6.2 - 3.0);  // 1.6 — the PAIR formula (the copy-paste mutant)

  // (1) the default guard THROWS on the band fixture (D=1.5 > g_eam=0.1) — honest HALT
  {
    core::Box box;
    auto a = chain3(box, 17.1);  // zone3 lo=18.6 ⇒ excess 1.5
    core::zero_forces(a);
    EXPECT_THROW(potentials::zone_eam_pass_donated(a, box, zd, pot), std::runtime_error);
  }
  // (2) mutant g_override=g_pair: the guard is silent; G-LCLOSE goes RED — ρ(b) as
  // read at finalize(1) != final ρ(b) (the (b,d) edge donates later) ⇒ the guard is
  // load-bearing for the read contract.
  {
    core::Box box;
    auto a = chain3(box, 17.1);
    core::zero_forces(a);
    using DA = core::fixed::FixedAccum<44>;
    ReadTrace tr;
    potentials::EamDonationPass<double, AnalyticEam<double>, DA, ReadTrace> pass(
        a, box, zd, pot, &tr, /*g_override=*/g_pair);
    run_donated_scan(pass, 5, false, potentials::donation_detail::NullPairHook{},
                     potentials::donation_detail::NullPairHook{});
    bool stale_read = false;
    for (const auto& r : tr.recs) {
      // final rho of atom key: find its zone/member slot
      for (int z = 0; z < 5; ++z)
        for (std::size_t t = 0; t < zd.members[std::size_t(z)].size(); ++t)
          if (long(zd.members[std::size_t(z)][t]) == r.key)
            stale_read = stale_read || (pass.state().rho[std::size_t(z)][t].raw != r.raw);
    }
    EXPECT_TRUE(stale_read) << "G-LCLOSE blind to the band — the guard is NOT load-bearing?!";
  }
  // (3) the same mutant: G-A vs zone_eam_pass — RED (MEASURED, divergence class 3:
  // the donated ρ(b) read at finalize(3) holds (b,o) via cross(1,2); the window
  // {2,3,4} cannot see it).
  {
    core::Box box;
    auto don = chain3(box, 17.1), ref = chain3(box, 17.1);
    core::zero_forces(don);
    core::zero_forces(ref);
    potentials::zone_eam_pass_donated(don, box, zd, pot, {}, /*g_override=*/g_pair);
    potentials::zone_eam_pass(ref, box, zd, pot);
    EXPECT_FALSE(forces_bitwise_equal(don, ref))
        << "band G-A stayed green — contradicts the measured class-3 divergence";
  }
  // (4) mutant g=inf + D > g_pair (d at 15.2, D=3.4): the WINDOW counts the (o,d)
  // pair (both inside {1,2,3}), donations have no (1,3) batch ⇒ G-A red — donations
  // without the guard diverge from the oracle outright.
  {
    core::Box box;
    auto don = chain3(box, 15.2), ref = chain3(box, 15.2);
    core::zero_forces(don);
    core::zero_forces(ref);
    potentials::zone_eam_pass_donated(don, box, zd, pot, {},
                                      /*g_override=*/std::numeric_limits<double>::infinity());
    potentials::zone_eam_pass(ref, box, zd, pot);
    EXPECT_FALSE(forces_bitwise_equal(don, ref)) << "no-guard divergence missing";
  }
  // (5) D7c: δ=0.08 <= 0.9*g_eam ⇒ the guard is silent AND G-A holds bitwise
  // (the guard is not over-tightened).
  {
    core::Box box;
    auto don = chain3(box, 18.52), ref = chain3(box, 18.52);
    core::zero_forces(don);
    core::zero_forces(ref);
    EXPECT_NO_THROW(potentials::zone_eam_pass_donated(don, box, zd, pot));
    potentials::zone_eam_pass(ref, box, zd, pot);
    EXPECT_TRUE(forces_bitwise_equal(don, ref));
  }
  // G-LCLOSE positive: on the staple straddle fixture every read is bit-final and
  // non-owned reads exist (>0) — the schedule freedom witness.
  {
    const auto ms = test_eam();
    EamPotential<double, AnalyticEam<double>> pots(ms);
    core::Box box;
    auto a = make_fcc(3, 3, 12, 4.05, box, false);
    jitter(a, 12);
    const auto zds = core::ZoneDecomposition::build(a, box, 6, kRcut, 2);
    core::zero_forces(a);
    using DA = core::fixed::FixedAccum<44>;
    ReadTrace tr;
    potentials::EamDonationPass<double, AnalyticEam<double>, DA, ReadTrace> pass(
        a, box, zds, pots, &tr);
    run_donated_scan(pass, zds.n_zones, false, potentials::donation_detail::NullPairHook{},
                     potentials::donation_detail::NullPairHook{});
    // global->(zone,member) map once
    std::vector<std::pair<int, int>> where(std::size_t(a.n), {-1, -1});
    for (int z = 0; z < zds.n_zones; ++z)
      for (std::size_t t = 0; t < zds.members[std::size_t(z)].size(); ++t)
        where[std::size_t(zds.members[std::size_t(z)][t])] = {z, int(t)};
    long nonowned_reads = 0;
    for (const auto& r : tr.recs) {
      const auto [z, t] = where[std::size_t(r.key)];
      EXPECT_EQ(pass.state().rho[std::size_t(z)][std::size_t(t)].raw, r.raw)
          << "read not bit-final at read time (L-CLOSE broken)";
      if (z != r.zone) ++nonowned_reads;
    }
    EXPECT_GT(nonowned_reads, 0) << "no halo reads — the L-CLOSE gate is vacuous";
  }
}

// Т-12 — G-FB40 (the fb trap): a Q23.40 potential (beta=3.3, the in-tree precedent)
// runs the donated path bitwise vs the oracle. Anti-vacuity: the fixture REALLY is
// fb=40 and the 44-vs-40 quantum differs. Kill: key the format off the static
// descriptor (44) instead of density_fracbits().
TEST(EamDonation, GFb40DualFormat) {
  const auto m = test_eam(/*beta=*/3.3);
  ASSERT_EQ(m.density_fracbits(), 40) << "fixture no longer reaches Q23.40";
  {  // the 44-vs-40 quantum genuinely differs for a representative value
    core::fixed::FixedAccum<44> a44;
    core::fixed::FixedAccum<40> a40;
    a44.add(0.1234567890123);
    a40.add(0.1234567890123);
    EXPECT_NE(a44.raw, a40.raw * 16) << "quantum indistinct — anti-vacuity broken";
  }
  EamPotential<double, AnalyticEam<double>> pot(m);
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box, /*periodic_z=*/false);
  jitter(base, 13);
  const auto zd = core::ZoneDecomposition::build(base, box, 6, kRcut, 2);
  core::AtomSoA<double> ref = base, don = base;
  core::zero_forces(ref);
  core::zero_forces(don);
  const auto ao = potentials::zone_eam_pass(ref, box, zd, pot);
  const auto ad = potentials::zone_eam_pass_donated(don, box, zd, pot);
  EXPECT_TRUE(forces_bitwise_equal(ref, don));
  EXPECT_EQ(ao.pe, ad.pe);
}

// Т-13 — the descriptor rollback guard (К1): the FREE function (the Т-13 seam —
// EamPotential is final, no injection through the ctor) rejects a kCOnly copy and
// accepts the shipped flipped descriptor. Kill: revert the eam.hpp flip ⇒ the
// driver ctor path throws everywhere.
TEST(EamDonation, DescriptorRollbackGuard) {
  EXPECT_NO_THROW(potentials::assert_eam_donation_descriptor(potentials::eam_pass_decls()));
  std::array<potentials::PassDecl, 3> conly{potentials::kEamPassDecls[0],
                                            potentials::kEamPassDecls[1],
                                            potentials::kEamPassDecls[2]};
  conly[0].w_class = potentials::WClass::kCOnly;
  conly[0].donor_roles = 0;
  EXPECT_THROW(potentials::assert_eam_donation_descriptor(conly), std::runtime_error);
  // partial roles are also a rollback (the ledger would starve on an edge)
  std::array<potentials::PassDecl, 3> partial{potentials::kEamPassDecls[0],
                                              potentials::kEamPassDecls[1],
                                              potentials::kEamPassDecls[2]};
  partial[0].donor_roles = potentials::donor_bit(potentials::DonorRole::kSelf);
  EXPECT_THROW(potentials::assert_eam_donation_descriptor(partial), std::runtime_error);
}

// Т-14 — G-VACUUM: empty zones (the chain fixture leaves zones 0 and 4 empty):
// batches are no-ops but the ledger bits are set VACUOUSLY (a bit means "the batch
// executed, possibly vacuously") — finalize never starves; result ≡ the oracle.
// Kill: drop the vacuous bit-setting ⇒ ledger-THROW on the empty zone's finalize.
TEST(EamDonation, GVacuumEmptyZones) {
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  const auto zd = chain_zd();
  core::Box box;
  auto don = chain3(box, 18.59), ref = chain3(box, 18.59);  // d ~at its slab edge (no drift)
  core::zero_forces(don);
  core::zero_forces(ref);
  EXPECT_NO_THROW(potentials::zone_eam_pass_donated(don, box, zd, pot));
  potentials::zone_eam_pass(ref, box, zd, pot);
  EXPECT_TRUE(forces_bitwise_equal(don, ref));
}

// Т-15 — the D3 teeth (ρ_a >= 0 load guards):
//  (а) a negative KNOT ⇒ throw; (б) knots >= 0 but an INTER-knot undershoot ⇒
//  throw — kills the "knot-only check" mutant (the D3 mandate was the spline
//  overshoot BETWEEN knots); (в) the shipped data loads (the exact-0 edge of
//  from_analytic is accepted — strict `< 0`); (г) AnalyticEam beta<=0 ⇒ throw.
TEST(EamDonation, D3RhoaNonNegativeGuards) {
  // (а) negative knot
  {
    potentials::EamSetfl<double> s;
    s.Nrho = 8; s.Nr = 8; s.drho = 1.0; s.dr = 0.5; s.rcut = 3.5;
    std::vector<double> F(8, 0.0), rphi(8, 0.0);
    std::vector<double> rhoa = {1.0, 0.8, 0.6, -0.05, 0.2, 0.1, 0.0, 0.0};
    EXPECT_THROW(s.build_(F, rhoa, rphi), std::runtime_error);
  }
  // (б) knots >= 0, inter-knot undershoot: a flat-zero run entered with a negative
  // slope rings below zero inside the next interval (the classic overshoot)
  {
    potentials::EamSetfl<double> s;
    s.Nrho = 8; s.Nr = 8; s.drho = 1.0; s.dr = 0.5; s.rcut = 3.5;
    std::vector<double> F(8, 0.0), rphi(8, 0.0);
    std::vector<double> rhoa = {1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    EXPECT_THROW(s.build_(F, rhoa, rphi), std::runtime_error)
        << "knot-only guard mutant: the undershoot between knots was missed";
  }
  // (в) shipped data loads clean (measured: Al_zhou min +2.9e-13; analytic min == 0)
  EXPECT_NO_THROW(potentials::EamSetfl<double>::from_setfl(
      project_root() + "/reference_data/eam_al/Al_zhou.eam.alloy"));
  EXPECT_NO_THROW(potentials::EamSetfl<double>::from_setfl(
      project_root() + "/reference_data/eam_al/Al_small.eam.alloy"));
  EXPECT_NO_THROW(potentials::EamSetfl<double>::from_analytic(test_eam(), 500, 500, 20.0));
  // (г) the analytic guard: beta<=0 / rho_amp<=0 break the closed-form proof
  {
    AnalyticEam<double> bad;
    bad.rcut = kRcut;
    bad.beta = -0.5;
    EXPECT_THROW(bad.finalize(), std::runtime_error);
    AnalyticEam<double> bad2;
    bad2.rcut = kRcut;
    bad2.rho_amp = 0.0;
    EXPECT_THROW(bad2.finalize(), std::runtime_error);
  }
}
