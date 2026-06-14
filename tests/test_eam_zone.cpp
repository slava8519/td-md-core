// M6 PR-E3: serial multi-pass zone EAM with symmetric three-zone residence —
// the bitwise reference for the EAM ring (threaded TimeConveyor wiring = E3b).
// Acceptance (M6_EAM_MANYBODY_DESIGN §7, PR-E3 row):
//   - zone(z=1) ≡ eam_run_fixed monolith bitwise (full-neighbour ≡ Newton-3);
//   - zone(z=1) ≡ z∈{5,6,8} bitwise (determinism AND static-residence correctness
//     — z=1 is the complete monolith, so equality proves the window is complete);
//   - zone order invariance (B1);
//   - dual-format density dispatch (Q19.44 / Q23.40) — closes the E2 P0 stopgap;
//   - the FORWARD-ONLY two-zone window DIVERGES from the monolith (empirically
//     reproduces the adversarial halo finding: three-zone residence is necessary).
#include <gtest/gtest.h>

#include <array>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/potentials/eam.hpp"
#include "tdmd/potentials/eam_analytic.hpp"
#include "tdmd/potentials/eam_zone.hpp"

using namespace tdmd;
using potentials::AnalyticEam;
using potentials::EamPotential;

namespace {

// Slab FCC supercell (small x,y; tall z) so z=8 EAM zones (width ≥ 2·rcut) fit
// with few atoms. nx=ny=3, nz=12, a=4.05 ⇒ box 12.15×12.15×48.6, 432 atoms.
core::AtomSoA<double> make_fcc(int nx, int ny, int nz, double alat, core::Box& box) {
  box.lo = {0, 0, 0};
  box.hi = {nx * alat, ny * alat, nz * alat};
  box.periodic = {true, true, true};
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

// small jitter so forces are nonzero (perfect FCC ⇒ zero net force ⇒ vacuous)
void jitter(core::AtomSoA<double>& a, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> j(-0.06, 0.06);
  for (int i = 0; i < a.n; ++i) { a.x[i] += j(rng); a.y[i] += j(rng); a.z[i] += j(rng); }
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

constexpr double kRcut = 3.0;  // > nn 2.86, < 2nd shell 4.05 ⇒ 12 nn; 2·rcut = 6.0

// Analytic EAM whose rcut MATCHES the zone decomposition (kRcut). The potential
// cutoff and the zone-width guard MUST use the SAME rcut — otherwise the
// three-zone window is sized for the wrong range (the real conveyor passes one
// rcut, so no mismatch there; this just keeps the serial test self-consistent).
AnalyticEam<double> test_eam(double beta = 1.5) {
  AnalyticEam<double> m;
  m.rcut = kRcut;
  m.beta = beta;
  m.finalize();
  return m;
}

}  // namespace

// --- zone(z=1) ≡ eam_run_fixed monolith (full-neighbour ≡ Newton-3, bitwise) ---
TEST(EamZone, MonolithMatchesEamRunFixed) {
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  jitter(base, 1);
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);

  core::AtomSoA<double> a = base, b = base;
  core::zero_forces(a);
  potentials::eam_run_fixed(a, box, pot);  // Newton-3 all-pairs monolith

  core::zero_forces(b);
  const auto zd1 = core::ZoneDecomposition::build(b, box, 1, kRcut, 2);
  potentials::zone_eam_pass(b, box, zd1, pot);  // full-neighbour, single zone

  EXPECT_GT(max_abs_force(a), 0.05);
  EXPECT_TRUE(forces_bitwise_equal(a, b)) << "full-neighbour zone ≠ Newton-3 monolith";
}

// --- 1-vs-z bitwise: z=1 (monolith) ≡ z∈{5,6,8} ⇒ determinism + complete window ---
TEST(EamZone, OneVsZBitwise) {
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  jitter(base, 2);
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);

  auto run = [&](int nz) {
    core::AtomSoA<double> a = base;
    core::zero_forces(a);
    const auto zd = core::ZoneDecomposition::build(a, box, nz, kRcut, 2);
    potentials::zone_eam_pass(a, box, zd, pot);
    return a;
  };
  const auto ref = run(1);
  EXPECT_GT(max_abs_force(ref), 0.05);
  for (int nz : {5, 6, 8}) {
    const auto z = run(nz);
    EXPECT_TRUE(forces_bitwise_equal(ref, z)) << "z=" << nz << " ≠ monolith bitwise";
  }
}

// --- zone processing order invariance (integer-associative accumulation) ---
TEST(EamZone, ZoneOrderInvariant) {
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  jitter(base, 3);
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);

  core::AtomSoA<double> a = base, b = base;
  const auto zd = core::ZoneDecomposition::build(a, box, 6, kRcut, 2);
  std::vector<int> ord(6); std::iota(ord.begin(), ord.end(), 0);
  std::vector<int> shuf = {3, 0, 5, 1, 4, 2};

  core::zero_forces(a); potentials::zone_eam_pass(a, box, zd, pot, ord);
  core::zero_forces(b); potentials::zone_eam_pass(b, box, zd, pot, shuf);
  EXPECT_TRUE(forces_bitwise_equal(a, b)) << "zone order changed the result";
}

// --- forward-only TWO-zone window diverges from the monolith (adversarial finding) ---
TEST(EamZone, ForwardOnlyWindowDiverges) {
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  jitter(base, 4);
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);

  core::AtomSoA<double> ref = base, bad = base;
  core::zero_forces(ref);
  const auto zd1 = core::ZoneDecomposition::build(ref, box, 1, kRcut, 2);
  potentials::zone_eam_pass(ref, box, zd1, pot);  // monolith (complete)

  core::zero_forces(bad);
  const auto zd = core::ZoneDecomposition::build(bad, box, 6, kRcut, 2);
  potentials::zone_eam_pass(bad, box, zd, pot, /*symmetric=*/false);  // forward-only

  // The two-zone window misses lower-face density donors ⇒ wrong EAM force.
  EXPECT_FALSE(forces_bitwise_equal(ref, bad))
      << "forward-only window matched the monolith — three-zone residence would "
         "then be unnecessary (contradicts the adversarial halo finding)";
}

// --- cyclic (PBC) window vs a fully INDEPENDENT FP64 O(N²) min-image oracle.
// The 1-vs-z test shares eam_window_force, so it cannot catch a shared cyclic-
// window/kernel error; eam_direct_fp64 is a separate code path (adversarial
// acceptance blind-spot closure). ---
TEST(EamZone, PbcCyclicWindowVsFp64Oracle) {
  core::Box box;
  auto a = make_fcc(3, 3, 12, 4.05, box);  // periodic in all axes
  ASSERT_TRUE(box.periodic[2]);
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> jit(-0.05, 0.05);
  for (int i = 0; i < a.n; ++i) { a.x[i] += jit(rng); a.y[i] += jit(rng); a.z[i] += jit(rng); }
  const auto m = test_eam();  // rcut = kRcut = 3.0
  potentials::EamPotential<double, AnalyticEam<double>> pot(m);

  core::AtomSoA<double> zone = a;  // cyclic-window zone force (fixed-point), z=6
  core::zero_forces(zone);
  const auto zd = core::ZoneDecomposition::build(zone, box, 6, kRcut, 2);
  potentials::zone_eam_pass(zone, box, zd, pot);

  core::AtomSoA<double> oracle = a;  // independent FP64 O(N²) min-image EAM
  core::zero_forces(oracle);
  potentials::eam_direct_fp64<double, AnalyticEam<double>>(oracle, box, m, true);

  double maxdev = 0;
  for (int i = 0; i < a.n; ++i) {
    maxdev = std::max(maxdev, std::fabs(zone.fx[i] - oracle.fx[i]));
    maxdev = std::max(maxdev, std::fabs(zone.fy[i] - oracle.fy[i]));
    maxdev = std::max(maxdev, std::fabs(zone.fz[i] - oracle.fz[i]));
  }
  EXPECT_LT(maxdev, 1e-10) << "cyclic window ≠ FP64 oracle: " << maxdev;
}

// --- residence guard: zone width < 2·rcut (rcut mismatch) HALTs, not silently
// computes a wrong force (the determinism gates would not catch it) ---
TEST(EamZone, NarrowZoneHalts) {
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);  // Lz=48.6
  AnalyticEam<double> m;
  m.rcut = 4.0;  // 2·rcut = 8.0, but zones below are decomposed as if rcut=2.0
  m.finalize();
  EamPotential<double, AnalyticEam<double>> pot(m);
  core::zero_forces(base);
  // build passes its own (smaller) rcut=2.0 guard (width 6.075 ≥ 2·2.0=4.0)...
  const auto zd = core::ZoneDecomposition::build(base, box, 8, 2.0, 2);
  // ...but zone_eam_pass sees the real 2·rcut=8.0 > width 6.075 ⇒ HALT.
  EXPECT_THROW(potentials::zone_eam_pass(base, box, zd, pot), std::runtime_error);
}

// --- dual-format density dispatch: a Q23.40-fallback potential runs and stays
// bitwise z-independent (the path eam_run_fixed could not take — E2 P0) ---
TEST(EamZone, DualFormatQ23Fallback) {
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  jitter(base, 5);
  AnalyticEam<double> steep;
  steep.rcut = kRcut;  // match the zone decomposition (see test_eam())
  steep.beta = 3.3;    // density_fracbits()==40 ⇒ Q23.40
  steep.finalize();
  ASSERT_EQ(steep.density_fracbits(), 40);
  EamPotential<double, AnalyticEam<double>> pot(steep);

  auto run = [&](int nz) {
    core::AtomSoA<double> a = base;
    core::zero_forces(a);
    const auto zd = core::ZoneDecomposition::build(a, box, nz, kRcut, 2);
    potentials::zone_eam_pass(a, box, zd, pot);  // dispatches to FixedAccum<40>
    return a;
  };
  const auto ref = run(1);
  EXPECT_GT(max_abs_force(ref), 0.05);
  const auto z5 = run(5);
  EXPECT_TRUE(forces_bitwise_equal(ref, z5)) << "Q23.40 path not z-independent";
}

// --- order argument must be a permutation (UB / silent force corruption) ---
TEST(EamZone, OrderValidation) {
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  const auto zd = core::ZoneDecomposition::build(base, box, 6, kRcut, 2);
  core::zero_forces(base);
  EXPECT_THROW(potentials::zone_eam_pass(base, box, zd, pot, {0, 1, 2, 3, 4}),        // short
               std::invalid_argument);
  EXPECT_THROW(potentials::zone_eam_pass(base, box, zd, pot, {0, 1, 2, 3, 4, 4}),     // dup
               std::invalid_argument);
  EXPECT_THROW(potentials::zone_eam_pass(base, box, zd, pot, {0, 1, 2, 3, 4, 9}),     // out of range
               std::invalid_argument);
  EXPECT_NO_THROW(potentials::zone_eam_pass(base, box, zd, pot, {5, 0, 4, 1, 3, 2}));  // valid perm
}

// --- PE is bitwise-equal to the monolith and z-independent (the oracle's
// energy channel, previously unverified — adversarial finding) ---
TEST(EamZone, EnergyBitwiseAndZIndependent) {
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  jitter(base, 7);
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);

  core::AtomSoA<double> mono = base;
  core::zero_forces(mono);
  const double pe_mono = potentials::eam_run_fixed(mono, box, pot).pe;

  auto pe_at = [&](int nz) {
    core::AtomSoA<double> a = base;
    core::zero_forces(a);
    const auto zd = core::ZoneDecomposition::build(a, box, nz, kRcut, 2);
    return potentials::zone_eam_pass(a, box, zd, pot).pe;
  };
  EXPECT_EQ(pe_at(1), pe_mono) << "zone PE ≠ Newton-3 monolith PE bitwise";
  EXPECT_EQ(pe_at(5), pe_mono) << "zone PE not z-independent (z=5)";
  EXPECT_EQ(pe_at(6), pe_mono) << "zone PE not z-independent (z=6)";
}

// --- min_r2 overlap probe (B10) matches the FP64 oracle (needed by PR-E4 NVE) ---
TEST(EamZone, MinR2MatchesOracle) {
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  jitter(base, 8);
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);

  core::AtomSoA<double> a = base;
  core::zero_forces(a);
  const double r2_oracle =
      potentials::eam_direct_fp64<double, AnalyticEam<double>>(a, box, m, false).min_r2;

  core::AtomSoA<double> b = base;
  core::zero_forces(b);
  const auto zd = core::ZoneDecomposition::build(b, box, 6, kRcut, 2);
  const double r2_zone = potentials::zone_eam_pass(b, box, zd, pot).min_r2;

  EXPECT_GT(r2_oracle, 0.0);
  EXPECT_EQ(r2_zone, r2_oracle) << "zone min_r2 ≠ FP64 oracle min_r2";
}
