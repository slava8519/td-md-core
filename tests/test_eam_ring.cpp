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
#include "tdmd/potentials/eam_spline.hpp"  // PR-2 G7: EamSetfl tight/loose rho_cap grids
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

// PR-2 serial VV driven by the INDEPENDENT FP64 oracle (eam_direct_fp64 shares NO
// window/zone/donation logic) — the non-negotiable dropped-donor witness (L2/MB2).
void serial_vv_fp64(core::AtomSoA<double>& a, const core::Box& box,
                    const AnalyticEam<double>& m, long steps, double dt) {
  core::zero_forces(a);
  potentials::eam_direct_fp64(a, box, m, /*with_forces=*/true);
  for (long h = 0; h < steps; ++h) {
    for (int i = 0; i < a.n; ++i) {
      const double inv_m = units::ftm2v / a.mass[i];
      a.vx[i] += 0.5 * dt * inv_m * a.fx[i]; a.vy[i] += 0.5 * dt * inv_m * a.fy[i]; a.vz[i] += 0.5 * dt * inv_m * a.fz[i];
      a.x[i] += dt * a.vx[i]; a.y[i] += dt * a.vy[i]; a.z[i] += dt * a.vz[i];
    }
    core::zero_forces(a);
    potentials::eam_direct_fp64(a, box, m, true);
    for (int i = 0; i < a.n; ++i) {
      const double inv_m = units::ftm2v / a.mass[i];
      a.vx[i] += 0.5 * dt * inv_m * a.fx[i]; a.vy[i] += 0.5 * dt * inv_m * a.fy[i]; a.vz[i] += 0.5 * dt * inv_m * a.fz[i];
    }
  }
}
double max_pos_dev(const core::AtomSoA<double>& a, const core::AtomSoA<double>& b) {
  double mx = 0;
  for (int i = 0; i < a.n; ++i)
    mx = std::max({mx, std::fabs(a.x[i] - b.x[i]), std::fabs(a.y[i] - b.y[i]),
                   std::fabs(a.z[i] - b.z[i])});
  return mx;
}

// PR-2 test-only window-force policies (both model DonatingWindowForcePolicy).
// (1) RECOMPUTE reference: compose IGNORES the donated rho and recomputes density via the
//     frozen eam_window_force — the CPU structural analog of the GPU policy. The dual-path
//     gate (G1) runs it against the default donation policy through the SAME orchestration:
//     the ONLY difference is compose-from-rho vs recompute, so a donate_position rho bug
//     diverges (donation-ring uses corrupt rho; recompute-ring ignores it and is correct).
template <typename Math>
struct CpuEamRecomputeWinForce {
  const Math* math = nullptr;
  int fb = 44;
  explicit CpuEamRecomputeWinForce(const Math& m) : math(&m), fb(m.density_fracbits()) {}
  void compute(const double* wx, const double* wy, const double* wz, const long* key, int m_,
               const int* owned, int n_owned, const core::PairGeom& geom, double rho_cap,
               std::vector<core::fixed::ForceAccum>& wFx, std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz, core::fixed::EnergyAccum& pe,
               double& min_r2) const {
    if (fb == 44)
      potentials::eam_window_force<Math, core::fixed::FixedAccum<44>>(
          wx, wy, wz, key, m_, owned, n_owned, *math, geom, rho_cap, wFx, wFy, wFz, pe, min_r2);
    else
      potentials::eam_window_force<Math, core::fixed::FixedAccum<40>>(
          wx, wy, wz, key, m_, owned, n_owned, *math, geom, rho_cap, wFx, wFy, wFz, pe, min_r2);
  }
  static void assert_supported(std::span<const potentials::PassDecl> p) {
    potentials::assert_eam_symmetric_passes(p, "CpuEamRecomputeWinForce");
  }
  // PR-3a concept-v2 shim: inert NodeState (test policy holds no device state).
  struct NodeState {
    void begin_pass(long) {}
  };
  NodeState make_node_state(int) const { return {}; }
  template <class DA> void on_zone_arrival(NodeState&, potentials::EamDonationState<DA>&, int,
      const potentials::ZoneBlockView&, const core::PairGeom&) const {}
  template <class DA> void on_edge(NodeState&, potentials::EamDonationState<DA>&, int, int,
      const potentials::ZoneBlockView&, const potentials::ZoneBlockView&, const core::PairGeom&) const {}
  template <class DA>
  void compose(NodeState&, const double* wx, const double* wy, const double* wz, const long* key,
               int m_, const potentials::WindowBlocks&, const int* owned, int n_owned,
               const core::PairGeom& geom, double rho_cap,
               const DA* /*rho_w — IGNORED: recompute reference*/,
               std::vector<core::fixed::ForceAccum>& wFx, std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz, core::fixed::EnergyAccum& pe,
               double& min_r2, int /*zone_j*/) const {
    compute(wx, wy, wz, key, m_, owned, n_owned, geom, rho_cap, wFx, wFy, wFz, pe, min_r2);
  }
};
// (2) POISON half-bug: on_edge donates ONE-SIDED (rho into the lower zone only) — breaks the
//     two-end lemma; compose then builds force from the corrupt rho. The ring ledger stays
//     CLEAN (the ring sets bits), yet the physics is red vs both oracles ⇒ pins the honest
//     ledger boundary (the FP64 oracle is the only witness of an intra-batch fault).
template <typename Math>
struct CpuEamPoisonWinForce {
  const Math* math = nullptr;
  int fb = 44;
  explicit CpuEamPoisonWinForce(const Math& m) : math(&m), fb(m.density_fracbits()) {}
  void compute(const double* wx, const double* wy, const double* wz, const long* key, int m_,
               const int* owned, int n_owned, const core::PairGeom& geom, double rho_cap,
               std::vector<core::fixed::ForceAccum>& wFx, std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz, core::fixed::EnergyAccum& pe,
               double& min_r2) const {
    if (fb == 44)
      potentials::eam_window_force<Math, core::fixed::FixedAccum<44>>(
          wx, wy, wz, key, m_, owned, n_owned, *math, geom, rho_cap, wFx, wFy, wFz, pe, min_r2);
    else
      potentials::eam_window_force<Math, core::fixed::FixedAccum<40>>(
          wx, wy, wz, key, m_, owned, n_owned, *math, geom, rho_cap, wFx, wFy, wFz, pe, min_r2);
  }
  static void assert_supported(std::span<const potentials::PassDecl> p) {
    potentials::assert_eam_symmetric_passes(p, "CpuEamPoisonWinForce");
  }
  // PR-3a concept-v2 shim: inert NodeState (test policy holds no device state).
  struct NodeState {
    void begin_pass(long) {}
  };
  NodeState make_node_state(int) const { return {}; }
  template <class DA> void on_zone_arrival(NodeState&, potentials::EamDonationState<DA>& st,
      int label, const potentials::ZoneBlockView& blk, const core::PairGeom& geom) const {
    potentials::eam_donate_self<Math, DA>(blk, *math, geom, st.rho[std::size_t(label)]);
  }
  template <class DA> void on_edge(NodeState&, potentials::EamDonationState<DA>& st, int la,
      int lb, const potentials::ZoneBlockView& a, const potentials::ZoneBlockView& b,
      const core::PairGeom& geom) const {
    potentials::DonationPoison p; p.one_sided = true;  // <<< the half-bug
    potentials::eam_donate_cross<Math, DA>(a, b, *math, geom, st.rho[std::size_t(la)],
        st.rho[std::size_t(lb)], potentials::donation_detail::NullPairHook{}, &p);
  }
  template <class DA>
  void compose(NodeState&, const double* wx, const double* wy, const double* wz, const long* key,
               int m_, const potentials::WindowBlocks&, const int* owned, int n_owned,
               const core::PairGeom& geom, double rho_cap,
               const DA* rho_w, std::vector<core::fixed::ForceAccum>& wFx,
               std::vector<core::fixed::ForceAccum>& wFy, std::vector<core::fixed::ForceAccum>& wFz,
               core::fixed::EnergyAccum& pe, double& min_r2, int zone_j) const {
    potentials::eam_window_force_from_rho<Math, DA>(wx, wy, wz, key, m_, owned, n_owned, *math,
        geom, rho_cap, rho_w, wFx, wFy, wFz, pe, min_r2, potentials::NullDonationTrace{}, zone_j);
  }
};

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

// ===================== PR-E4: EAM NVE energy conservation =====================
// Total energy E = PE + KE has bounded fluctuation (no secular drift) over a
// long NVE run. The force-shifted analytic EAM is C1 ⇒ no shift-discontinuity
// floor (unlike the pair Morse-shift NVE). Multi-zone ≡ 1-zone is already proven
// bitwise (LongRunReplica); here we check the PHYSICS invariant.
TEST(EamRing, NveEnergyConservation) {
  core::Box box;
  auto base = make_fcc_pbc(box);  // fully periodic ⇒ no surface, clean NVE
  const auto m = test_eam();
  potentials::EamPotential<double, AnalyticEam<double>> pot(m);
  // 1000 steps (0.5 ps) stays within the static-membership margin g=0.5(w-2rcut)
  // (atom migration past the slab is deferred to a future PR); enough to show the
  // NVE invariant.
  const long steps = 1000;
  const double dt = 0.0005;
  core::AtomSoA<double> a = base;
  const auto r = potentials::run_eam_ring(a, box, pot, ring_opts(steps, 5, 3, dt));
  ASSERT_EQ(int(r.halt), int(core::Halt::None)) << r.halt_msg;
  ASSERT_EQ(int(r.stats.size()), int(steps));

  // E(t) per pass; relative fluctuation + secular drift (2nd-half − 1st-half mean).
  std::vector<double> E(steps);
  for (long t = 0; t < steps; ++t) E[t] = r.stats[t].pe + r.stats[t].ke;
  double emin = E[0], emax = E[0], e0 = E[0];
  double s1 = 0, s2 = 0;
  for (long t = 0; t < steps; ++t) {
    emin = std::min(emin, E[t]); emax = std::max(emax, E[t]);
    if (t < steps / 2) s1 += E[t]; else s2 += E[t];
  }
  const double mean1 = s1 / (steps / 2), mean2 = s2 / (steps - steps / 2);
  const double scale = std::fabs(e0) + 1e-12;
  const double fluct = (emax - emin) / scale;       // bounded oscillation
  const double drift = std::fabs(mean2 - mean1) / scale;  // secular trend
  // velocity-Verlet + C1 force ⇒ BOUNDED fluctuation (the PE/KE half-step phase
  // oscillation, ~1e-4 here, not a defect) and ~ZERO secular drift (the real NVE
  // invariant: no systematic energy gain/loss).
  EXPECT_LT(fluct, 5e-4) << "energy fluctuation " << fluct;
  EXPECT_LT(drift, 1e-5) << "secular energy drift " << drift;
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

// ===========================================================================
// PR-2 — live donation ring gates (each carries its kill-mutation).
// ===========================================================================

// G1 ⭐ — the §12.5 dual-path gate: the donation ring (default CpuEamWindowForce,
// compose-from-ρ) ≡ the recompute ring (CpuEamRecomputeWinForce) through the SAME
// orchestration, bitwise. Isolates EXACTLY the density-assembly (donate_position/ledger/
// SPHERE-gate run identically in both). Kill: an off-by-one in donation_layout consumption
// corrupts ρ ⇒ donation ring diverges from the recompute ring.
TEST(EamRing, G1DualPathDonateVsRecompute) {
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  for (bool pbc : {false, true}) {
    core::Box box;
    auto base = pbc ? make_fcc_pbc(box) : make_fcc(3, 3, 12, 4.05, box);
    for (bool autodt : {false, true}) {
      for (int z : {1, 3}) {
        core::AtomSoA<double> don = base, rec = base;
        auto o = ring_opts(20, 6, z, 0.001);
        o.auto_step = autodt;
        potentials::run_eam_ring(don, box, pot, o);  // default = donation compose-from-ρ
        potentials::run_eam_ring(rec, box, pot, o,
                                 CpuEamRecomputeWinForce<AnalyticEam<double>>(m));  // recompute
        EXPECT_TRUE(state_bitwise_equal(don, rec))
            << "dual-path pbc=" << pbc << " auto=" << autodt << " z=" << z;
      }
    }
  }
}

// G4 ⭐ — the independent FP64-oracle witness (L2/MB2, non-negotiable): the donation-ring
// trajectory stays within-tol of an eam_direct_fp64-driven VV (shares NO window/zone/
// donation logic ⇒ catches a pair lost INSIDE a batch, to which 1-vs-z/L1 are blind).
// Kill: the poison policy (G9) drives this above 1e-3.
TEST(EamRing, G4Fp64OracleWitness) {
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  for (bool pbc : {false, true}) {
    core::Box box;
    auto base = pbc ? make_fcc_pbc(box) : make_fcc(3, 3, 12, 4.05, box);
    core::AtomSoA<double> don = base, fp = base;
    const long steps = 8; const double dt = 0.001;
    potentials::run_eam_ring(don, box, pot, ring_opts(steps, pbc ? 5 : 6, 2, dt));
    serial_vv_fp64(fp, box, m, steps, dt);
    EXPECT_LT(max_pos_dev(don, fp), 1e-10) << "donation ring diverges from FP64 oracle, pbc=" << pbc;
  }
}

// G5 ⭐ — G-ROT (zone_id keying over >=2 full PBC rotations). n=5, seam-heavy, >=10 steps ⇒
// r=(h-1)%n cycles >=2×, so slot!=label repeatedly. Donation ≡ z=1 bitwise AND within-tol of
// the FP64 oracle. Kill (reviewer, code): read dstate.rho[wslot] (slot) instead of [label] in
// finalize ⇒ under rotation a zone reads a FOREIGN zone's ρ ⇒ RED. (Pure-slot keying is
// bitwise-isomorphic for a pass-local store; the discriminator is the MIXED-space bug.)
TEST(EamRing, G5RotationZoneIdKeying) {
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  core::Box box;
  auto base = make_fcc_pbc(box);
  const long steps = 12;  // n=5 ⇒ >=2 rotations
  const double dt = 0.001;
  core::AtomSoA<double> ref = base;
  potentials::run_eam_ring(ref, box, pot, ring_opts(steps, 5, 1, dt));
  for (int z : {2, 3, 5}) {
    core::AtomSoA<double> a = base;
    potentials::run_eam_ring(a, box, pot, ring_opts(steps, 5, z, dt));
    EXPECT_TRUE(state_bitwise_equal(a, ref)) << "rotation z=" << z << " != z=1 bitwise";
  }
  core::AtomSoA<double> fp = base;
  serial_vv_fp64(fp, box, m, steps, dt);
  EXPECT_LT(max_pos_dev(ref, fp), 1e-10) << "rotation ring diverges from FP64 oracle";
}

// G7 — rho_cap enforcement on the ring: a tight F(ρ) grid ⇒ the density exceeds the grid and
// the ring rejects it (the t0 oracle force-eval throws first — the cap is load-bearing at the
// ring level); a loose grid ⇒ no throw + bitwise ≡ the recompute ring. The COMPOSE-path
// owned-cap (К3) + its owner-zone attribution is covered transitively by PR-1 Т-10 (serial),
// which the ring reuses verbatim (eam_window_force_from_rho). Kill: a loose grid that no
// longer covers the density would start throwing — anti-vacuity is the loose-passes half.
TEST(EamRing, G7RhoCapEnforced) {
  const auto m = test_eam();
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  double rho_max = 0.0;
  { core::AtomSoA<double> probe = base; core::zero_forces(probe);
    std::vector<double> rho; potentials::eam_direct_fp64(probe, box, m, false, &rho);
    for (double r : rho) rho_max = std::max(rho_max, r); }
  ASSERT_GT(rho_max, 0.1);
  const auto tight = potentials::EamSetfl<double>::from_analytic(m, 2000, 2000, rho_max * 0.5);
  const auto loose = potentials::EamSetfl<double>::from_analytic(m, 2000, 2000, rho_max * 1.5);
  EamPotential<double, potentials::EamSetfl<double>> pot_tight(tight), pot_loose(loose);
  { core::AtomSoA<double> a = base;
    EXPECT_THROW(potentials::run_eam_ring(a, box, pot_tight, ring_opts(1, 6, 2, 0.001)),
                 std::runtime_error) << "tight grid must reject over-cap density"; }
  { core::AtomSoA<double> a = base, rec = base;
    const auto r = potentials::run_eam_ring(a, box, pot_loose, ring_opts(2, 6, 2, 0.001));
    EXPECT_EQ(int(r.halt), int(core::Halt::None)) << "loose grid must not throw/HALT";
    potentials::run_eam_ring(rec, box, pot_loose, ring_opts(2, 6, 2, 0.001),
                             CpuEamRecomputeWinForce<potentials::EamSetfl<double>>(loose));
    EXPECT_TRUE(state_bitwise_equal(a, rec)) << "setfl donation ring != recompute ring"; }
}

// G8 ⭐ — D7 drift-band replica on the LIVE ring (WContract §12.4). An atom initialized WELL
// past its zone's g=0.5(width-2·rcut) ⇒ the ring's residence guard (membership_ok) HALTs
// StaleZone. MEASURED supra-tol (acceptance MUST-FIX: the band-EDGE class-3 divergence is
// near-cutoff-tiny; this fixture drives the atom FAR past g so guard-removal is non-vacuous).
// Kill: remove membership_ok (eam_ring.hpp:~315) ⇒ no HALT AND (measured) forces diverge from
// the FP64 oracle by >>1e-9 (the class-3 divergence PR-1 Т-11 measured, now on the ring).
TEST(EamRing, G8DriftBandResidenceHalt) {
  const auto m = test_eam();  // rcut = 3.0
  EamPotential<double, AnalyticEam<double>> pot(m);
  // The ring builds its zd from the initial positions (every atom starts INSIDE its slab,
  // excess 0), so staleness must ACCUMULATE via drift. n=6 ⇒ width=8.1, g=0.5(8.1-6)=1.05.
  // Give atom 0 a strong outward vz so it drifts >g within a few steps; a lenient C_buf keeps
  // causality (INV-4) from firing first (v_max*dt << R_buf). membership_ok(zone 0) then HALTs.
  // MEASURED on the live ring (acceptance, verified by removing the guard in a scratch copy):
  // guard-off ⇒ this StaleZone HALT no longer fires; instead the drifted atom collides into a
  // neighbour and Halt::Overlap trips at step 3 (min dist 0.187 Å) — a DIFFERENT halt code, so
  // EXPECT StaleZone reds. The residence guard is thus load-bearing (it catches the drift BEFORE
  // the truncated-window class-3 divergence corrupts physics). NON-vacuous kill. Kill: remove
  // the n>=3 membership_ok check in end_eam ⇒ this HALT no longer fires (Overlap or wrong forces).
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  base.z[0] = 7.5;      // near the TOP of zone 0's slab [0, 8.1] (binned into zone 0 at build)
  base.vz[0] = 1500.0;  // ~1.5 Å/step ⇒ crosses 8.1 + g=9.15 within ~2 steps
  auto o = ring_opts(10, 6, 1, 0.001);
  o.ts.C_buf = 30.0;    // lenient causality so StaleZone (not Causality) is the first HALT
  core::AtomSoA<double> a = base;
  const auto r = potentials::run_eam_ring(a, box, pot, o);
  EXPECT_EQ(int(r.halt), int(core::Halt::StaleZone))
      << "drift-band atom must HALT StaleZone (guard is load-bearing for L-CLOSE); got " << r.halt_msg;
}

// G9 — G-POISON half-bug: the one-sided donation policy corrupts ρ. The ring ledger stays
// clean (the ring sets bits) yet the physics is red vs BOTH oracles ⇒ pins the ledger's
// honest boundary (only the FP64 oracle witnesses an intra-batch fault). Positive = G1/G4.
TEST(EamRing, G9PoisonOneSidedIsCaughtOnlyByPhysics) {
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  const long steps = 6; const double dt = 0.001;
  core::AtomSoA<double> poi = base, ref = base, fp = base;
  const auto r = potentials::run_eam_ring(poi, box, pot, ring_opts(steps, 6, 2, dt),
                                          CpuEamPoisonWinForce<AnalyticEam<double>>(m));
  EXPECT_EQ(int(r.halt), int(core::Halt::None)) << "poison ledger stays clean (no HALT)";
  const auto zd = core::ZoneDecomposition::build(ref, box, 6, kRcut, 2);
  serial_vv(ref, box, pot, zd, steps, dt);
  serial_vv_fp64(fp, box, m, steps, dt);
  EXPECT_FALSE(state_bitwise_equal(poi, ref)) << "one-sided donation invisible to L1?!";
  EXPECT_GT(max_pos_dev(poi, fp), 1e-6) << "one-sided donation invisible to FP64?!";
}

// G10 ⭐ — the ledger-before-END gate (the material-SPHERE completeness check, WContract
// §12.4) is load-bearing (acceptance MUST-FIX: it previously had no automated tooth — the
// 5-recidive structurally-dead class). A dropped donation batch (its ρ still donated but its
// ledger bit skipped, via the test_drop_first_self_ledger_ knob) leaves the ledger INCOMPLETE
// ⇒ the ring HALTs StaleZone with the "donation ledger ... != want" diagnostic, strictly
// before SPHERE/END. Kill: remove the ledger check in end_eam ⇒ the incomplete ledger is no
// longer caught (verified: guard-off ⇒ this test goes RED, halt None).
TEST(EamRing, G10LedgerBeforeEndGate) {
  const auto m = test_eam();
  EamPotential<double, AnalyticEam<double>> pot(m);
  core::Box box;
  auto base = make_fcc(3, 3, 12, 4.05, box);
  potentials::EamRing<double, AnalyticEam<double>> ring(base, box, pot, ring_opts(3, 6, 1, 0.001));
  ring.test_drop_first_self_ledger_ = true;
  const auto r = ring.run();
  EXPECT_EQ(int(r.halt), int(core::Halt::StaleZone))
      << "incomplete donation ledger must HALT StaleZone; got " << r.halt_msg;
  EXPECT_NE(r.halt_msg.find("donation ledger"), std::string::npos)
      << "ledger-completeness diagnostic missing: " << r.halt_msg;
}
