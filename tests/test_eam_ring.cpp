// M6 PR-E3b — EamRing (threaded EAM TD ring, free-z) ≡ serial velocity-Verlet
// driven by the zone_eam_pass oracle, bitwise. Acceptance (design §8):
//   - z=1 ring ≡ serial VV bitwise (the send-delay/center-j schedule is correct);
//   - 1-vs-z bitwise (n_nodes>1; determinism under the Λ-chain dt handoff);
//   - anti-deadlock z=1..N; HALT cases.
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/units.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/potentials/eam.hpp"
#include "tdmd/potentials/eam_analytic.hpp"
#include "tdmd/potentials/eam_ring.hpp"
#include "tdmd/potentials/eam_zone.hpp"

using namespace tdmd;
using potentials::AnalyticEam;
using potentials::EamPotential;

namespace {

constexpr double kRcut = 3.0;

core::AtomSoA<double> make_fcc(int nx, int ny, int nz, double alat, core::Box& box) {
  box.lo = {0, 0, 0};
  box.hi = {nx * alat, ny * alat, nz * alat};
  box.periodic = {true, true, false};  // FREE z (EamRing free-z PR)
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
    a.vx[i] = 0.001 * ((i * 7) % 5 - 2);  // small deterministic velocities
    a.vy[i] = 0.001 * ((i * 3) % 5 - 2);
    a.vz[i] = 0.001 * ((i * 11) % 5 - 2);
  }
  return a;
}

AnalyticEam<double> test_eam() {
  AnalyticEam<double> m; m.rcut = kRcut; m.finalize(); return m;
}

// Serial velocity-Verlet using the zone_eam_pass oracle for the force — the
// SAME integration arithmetic the ring uses (ftm2v/mass, half-kicks).
template <typename PotT>
void serial_vv(core::AtomSoA<double>& a, const core::Box& box, const PotT& pot,
               const core::ZoneDecomposition& zd, long steps, double dt) {
  core::zero_forces(a);
  potentials::zone_eam_pass(a, box, zd, pot);
  for (long h = 0; h < steps; ++h) {
    for (int i = 0; i < a.n; ++i) {
      const double inv_m = units::ftm2v / a.mass[i];
      a.vx[i] += 0.5 * dt * inv_m * a.fx[i]; a.vy[i] += 0.5 * dt * inv_m * a.fy[i]; a.vz[i] += 0.5 * dt * inv_m * a.fz[i];
      a.x[i] += dt * a.vx[i]; a.y[i] += dt * a.vy[i]; a.z[i] += dt * a.vz[i];
    }
    core::zero_forces(a);
    potentials::zone_eam_pass(a, box, zd, pot);
    for (int i = 0; i < a.n; ++i) {
      const double inv_m = units::ftm2v / a.mass[i];
      a.vx[i] += 0.5 * dt * inv_m * a.fx[i]; a.vy[i] += 0.5 * dt * inv_m * a.fy[i]; a.vz[i] += 0.5 * dt * inv_m * a.fz[i];
    }
  }
}

bool state_bitwise_equal(const core::AtomSoA<double>& a, const core::AtomSoA<double>& b) {
  if (a.n != b.n) return false;
  for (int i = 0; i < a.n; ++i)
    if (a.x[i] != b.x[i] || a.y[i] != b.y[i] || a.z[i] != b.z[i] ||
        a.vx[i] != b.vx[i] || a.vy[i] != b.vy[i] || a.vz[i] != b.vz[i])
      return false;
  return true;
}

core::ConveyorOptions ring_opts(long steps, int n_zones, int n_nodes, double dt) {
  core::ConveyorOptions o;
  o.steps = steps; o.n_zones = n_zones; o.n_nodes = n_nodes;
  o.auto_step = false; o.dt_initial = dt;
  o.reach_mult = 2; o.symmetric_reach = true;
  return o;
}

}  // namespace

// --- z=1 ring ≡ serial velocity-Verlet (the schedule is correct) ---
TEST(EamRing, SingleNodeMatchesSerialVV) {
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  const double dt = 0.002;
  const long steps = 8;

  core::AtomSoA<double> ref = base;
  const auto zd_mono = core::ZoneDecomposition::build(ref, box, 1, kRcut, 2);
  serial_vv(ref, box, pot, zd_mono, steps, dt);

  core::AtomSoA<double> ring = base;
  const auto res = potentials::run_eam_ring(ring, box, pot, ring_opts(steps, 6, 1, dt));
  EXPECT_EQ(int(res.halt), int(core::Halt::None)) << res.halt_msg;
  EXPECT_TRUE(state_bitwise_equal(ring, ref)) << "z=1 ring ≠ serial VV bitwise";
}

// --- 1-vs-z bitwise (threaded; Λ-chain dt handoff z-independent) ---
TEST(EamRing, OneVsZBitwise) {
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  const double dt = 0.002;
  const long steps = 10;

  core::AtomSoA<double> ref = base;
  const auto r1 = potentials::run_eam_ring(ref, box, pot, ring_opts(steps, 6, 1, dt));
  ASSERT_EQ(int(r1.halt), int(core::Halt::None)) << r1.halt_msg;

  for (int z : {2, 3, 5}) {
    core::AtomSoA<double> a = base;
    const auto rz = potentials::run_eam_ring(a, box, pot, ring_opts(steps, 6, z, dt));
    ASSERT_EQ(int(rz.halt), int(core::Halt::None)) << "z=" << z << " " << rz.halt_msg;
    EXPECT_TRUE(state_bitwise_equal(a, ref)) << "z=" << z << " ring ≠ z=1 bitwise";
  }
}

// --- auto-dt 1-vs-z bitwise (Λ-chain dt SEQUENCE z-independent, no Allreduce) ---
TEST(EamRing, AutoDtOneVsZ) {
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  auto opt = [&](int z) {
    auto o = ring_opts(12, 6, z, 0.0005);
    o.auto_step = true; o.ts.C1 = 0.002; o.ts.C3 = 1.0; o.ts.C_buf = 2.5;  // conservative
    return o;
  };
  core::AtomSoA<double> ref = base;
  const auto r1 = potentials::run_eam_ring(ref, box, pot, opt(1));
  ASSERT_EQ(int(r1.halt), int(core::Halt::None)) << r1.halt_msg;
  for (int z : {2, 3}) {
    core::AtomSoA<double> a = base;
    const auto rz = potentials::run_eam_ring(a, box, pot, opt(z));
    ASSERT_EQ(int(rz.halt), int(core::Halt::None)) << "z=" << z << " " << rz.halt_msg;
    EXPECT_TRUE(state_bitwise_equal(a, ref)) << "auto-dt z=" << z << " ≠ z=1 bitwise";
  }
}

// --- dual-format (Q23.40) ring stays z-independent (E2 P0 path) ---
TEST(EamRing, DualFormatQ23OneVsZ) {
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  AnalyticEam<double> steep; steep.rcut = kRcut; steep.beta = 3.3; steep.finalize();
  ASSERT_EQ(steep.density_fracbits(), 40);
  EamPotential<double, AnalyticEam<double>> pot(steep);
  core::AtomSoA<double> ref = base;
  const auto r1 = potentials::run_eam_ring(ref, box, pot, ring_opts(8, 6, 1, 0.002));
  ASSERT_EQ(int(r1.halt), int(core::Halt::None)) << r1.halt_msg;
  core::AtomSoA<double> a = base;
  const auto r3 = potentials::run_eam_ring(a, box, pot, ring_opts(8, 6, 3, 0.002));
  ASSERT_EQ(int(r3.halt), int(core::Halt::None)) << r3.halt_msg;
  EXPECT_TRUE(state_bitwise_equal(a, ref)) << "Q23.40 ring not z-independent";
}

// --- §3.6-style long-run determinism: 1 node vs N nodes, zero deviation ---
TEST(EamRing, LongRunReplicaBitwise) {
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  const long steps = 150;  // wider zones (n=5, g=1.86 Å) keep atoms in their slabs
  core::AtomSoA<double> ref = base;
  const auto r1 = potentials::run_eam_ring(ref, box, pot, ring_opts(steps, 5, 1, 0.001));
  ASSERT_EQ(int(r1.halt), int(core::Halt::None)) << r1.halt_msg;
  core::AtomSoA<double> a = base;
  const auto r4 = potentials::run_eam_ring(a, box, pot, ring_opts(steps, 5, 4, 0.001));
  ASSERT_EQ(int(r4.halt), int(core::Halt::None)) << r4.halt_msg;
  EXPECT_TRUE(state_bitwise_equal(a, ref)) << steps << "-step 1-vs-4 deviation";
}

// --- honest HALT: overlap probe trips (B10), longest valid stats prefix kept ---
TEST(EamRing, OverlapHalt) {
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  auto o = ring_opts(5, 6, 2, 0.002);
  o.r_min_halt = 2.9;  // nn ≈ 2.86 < 2.9 ⇒ overlap halt on the very first pass
  core::AtomSoA<double> a = base;
  const auto r = potentials::run_eam_ring(a, box, pot, o);
  EXPECT_EQ(int(r.halt), int(core::Halt::Overlap));
  EXPECT_LT(r.steps_done, 5);
}

// --- EMPTY zones (vacuum gap) are legitimate: ring ≡ serial oracle bitwise.
// Regression for the adversarial finding (empty slab → false Halt::Internal). ---
TEST(EamRing, VacuumGapEmptyZones) {
  core::Box box;
  auto full = make_fcc(3, 3, 12, 4.05, box);  // free-z, box z = 48.6
  // keep only zones 0 (z<8.1) and 5 (z>=40.5); zones 1..4 are EMPTY.
  core::AtomSoA<double> base;
  std::vector<int> keep;
  for (int i = 0; i < full.n; ++i)
    if (full.z[i] < 8.1 || full.z[i] >= 40.5) keep.push_back(i);
  base.resize(int(keep.size()));
  for (int k = 0; k < base.n; ++k) {
    const int i = keep[k];
    base.x[k] = full.x[i]; base.y[k] = full.y[i]; base.z[k] = full.z[i];
    base.vx[k] = full.vx[i]; base.vy[k] = full.vy[i]; base.vz[k] = full.vz[i];
    base.type[k] = 1; base.mass[k] = 26.98;
  }
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  const double dt = 0.002; const long steps = 12;

  core::AtomSoA<double> ref = base;
  const auto zd_mono = core::ZoneDecomposition::build(ref, box, 1, kRcut, 2);
  serial_vv(ref, box, pot, zd_mono, steps, dt);  // oracle handles empty zones fine

  for (int z : {1, 3, 6}) {  // 6 zones ⇒ zones 1..4 empty
    core::AtomSoA<double> a = base;
    const auto r = potentials::run_eam_ring(a, box, pot, ring_opts(steps, 6, z, dt));
    ASSERT_EQ(int(r.halt), int(core::Halt::None)) << "z=" << z << " " << r.halt_msg;
    EXPECT_TRUE(state_bitwise_equal(a, ref)) << "vacuum-gap z=" << z << " ≠ oracle";
  }
}

// ===================== PR-E3b-PBC: periodic-z ring =====================
namespace {
// FCC slab made FULLY periodic in z (12 cells = exact lattice period ⇒ seamless
// PBC nn via min-image). reach_mult=2 ⇒ periodic n_zones must be 1 or >=5.
core::AtomSoA<double> make_fcc_pbc(core::Box& box) {
  auto a = make_fcc(3, 3, 12, 4.05, box);
  box.periodic[2] = true;
  return a;
}
std::array<double, 3> total_momentum(const core::AtomSoA<double>& a) {
  std::array<double, 3> p{0, 0, 0};
  for (int i = 0; i < a.n; ++i) {
    p[0] += a.mass[i] * a.vx[i]; p[1] += a.mass[i] * a.vy[i]; p[2] += a.mass[i] * a.vz[i];
  }
  return p;
}
}  // namespace

// --- PBC: z=1 ring ≡ serial VV over the CYCLIC-window oracle, bitwise (the only
// test that catches a wrong/clamped cyclic window — a deterministic miss) ---
TEST(EamRingPBC, SingleNodeMatchesSerialVV) {
  core::Box box;
  auto base = make_fcc_pbc(box);
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  const double dt = 0.002; const long steps = 8;
  core::AtomSoA<double> ref = base;
  const auto zd = core::ZoneDecomposition::build(ref, box, 6, kRcut, 2);  // cyclic window
  serial_vv(ref, box, pot, zd, steps, dt);
  core::AtomSoA<double> ring = base;
  const auto res = potentials::run_eam_ring(ring, box, pot, ring_opts(steps, 6, 1, dt));
  ASSERT_EQ(int(res.halt), int(core::Halt::None)) << res.halt_msg;
  EXPECT_TRUE(state_bitwise_equal(ring, ref)) << "PBC z=1 ring ≠ serial cyclic oracle";
}

// --- PBC 1-vs-z bitwise (rotation + defer_head + tail-batched sends) ---
TEST(EamRingPBC, OneVsZBitwise) {
  core::Box box;
  auto base = make_fcc_pbc(box);
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  core::AtomSoA<double> ref = base;
  const auto r1 = potentials::run_eam_ring(ref, box, pot, ring_opts(10, 6, 1, 0.002));
  ASSERT_EQ(int(r1.halt), int(core::Halt::None)) << r1.halt_msg;
  for (int z : {2, 3, 6}) {
    core::AtomSoA<double> a = base;
    const auto rz = potentials::run_eam_ring(a, box, pot, ring_opts(10, 6, z, 0.002));
    ASSERT_EQ(int(rz.halt), int(core::Halt::None)) << "z=" << z << " " << rz.halt_msg;
    EXPECT_TRUE(state_bitwise_equal(a, ref)) << "PBC z=" << z << " ≠ z=1 bitwise";
  }
}

// --- PBC momentum conservation (fully periodic ⇒ no boundary force leak) ---
TEST(EamRingPBC, MomentumConservation) {
  core::Box box;
  auto base = make_fcc_pbc(box);
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  const auto p0 = total_momentum(base);
  core::AtomSoA<double> a = base;
  const auto r = potentials::run_eam_ring(a, box, pot, ring_opts(60, 6, 3, 0.002));
  ASSERT_EQ(int(r.halt), int(core::Halt::None)) << r.halt_msg;
  const auto p1 = total_momentum(a);
  for (int d = 0; d < 3; ++d)
    EXPECT_LT(std::fabs(p1[d] - p0[d]), 1e-9) << "momentum drift, dim " << d;
}

// --- PBC §3.6-style long run: 1 node vs 4 nodes, zero deviation ---
TEST(EamRingPBC, LongRunReplicaBitwise) {
  core::Box box;
  auto base = make_fcc_pbc(box);
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  const long steps = 150;
  core::AtomSoA<double> ref = base;
  const auto r1 = potentials::run_eam_ring(ref, box, pot, ring_opts(steps, 5, 1, 0.001));
  ASSERT_EQ(int(r1.halt), int(core::Halt::None)) << r1.halt_msg;
  core::AtomSoA<double> a = base;
  const auto r4 = potentials::run_eam_ring(a, box, pot, ring_opts(steps, 5, 4, 0.001));
  ASSERT_EQ(int(r4.halt), int(core::Halt::None)) << r4.halt_msg;
  EXPECT_TRUE(state_bitwise_equal(a, ref)) << "PBC " << steps << "-step 1-vs-4 deviation";
}

// --- PBC anti-deadlock (n=1 free path; n>=5 cyclic; n in 2..4 throws at build) ---
TEST(EamRingPBC, AntiDeadlock) {
  core::Box box;
  auto base = make_fcc_pbc(box);
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  for (int z : {1, 2, 3, 4, 5}) {
    core::AtomSoA<double> a = base;
    const auto r = potentials::run_eam_ring(a, box, pot, ring_opts(2 * z + 3, 5, z, 0.002));
    EXPECT_EQ(int(r.halt), int(core::Halt::None)) << "z=" << z << " " << r.halt_msg;
    EXPECT_EQ(r.steps_done, 2 * z + 3);
  }
}

// --- anti-deadlock across node counts ---
TEST(EamRing, AntiDeadlock) {
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  for (int z : {1, 2, 3, 4, 5, 6}) {
    core::AtomSoA<double> a = base;
    const auto r = potentials::run_eam_ring(a, box, pot, ring_opts(2 * z + 3, 6, z, 0.002));
    EXPECT_EQ(int(r.halt), int(core::Halt::None)) << "z=" << z << " deadlock/halt: " << r.halt_msg;
    EXPECT_EQ(r.steps_done, 2 * z + 3);
  }
}
