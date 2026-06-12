// M3.5 — TimeConveyor-CPU acceptance (Roadmap M3.5, ZoneFSM §10):
//   * bitwise determinism "1 node vs z nodes" (coords AND velocities, the
//     §3.6 methodology), fixed and auto timestep modes;
//   * the conveyor reproduces the serial zone-pass stepper BITWISE (fixed dt)
//     — including across zone counts (B1: same pair set => same quantized
//     sums regardless of assembly grouping);
//   * PBC closure along the decomposition axis (rotation variant of §7.2)
//     against the serial oracle + momentum conservation;
//   * anti-deadlock: rings of odd/even z run >= 2z steps (§7.4);
//   * INV-4 and the static-membership guard HALT honestly;
//   * §3.6 replica: Al-72, free boundaries, auto-step, 25 900 steps,
//     bitwise 1 vs 4 nodes. Historical determinism replica, NOT a melting
//     measurement (см. _meta/VALIDATION_EXPERIMENT_2026-06-12.md §4).
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "tdmd/core/conveyor.hpp"
#include "tdmd/core/thermal.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/potentials/cutoff.hpp"
#include "tdmd/potentials/morse.hpp"

using namespace tdmd;

static std::string project_root() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT);
#else
  return ".";
#endif
}

namespace {

// The drivers' Morse pair callback (functor + truncation policy), dissertation
// Al parameters — same as Test_Zones; the conveyor is potential-agnostic.
struct MorsePair {
  potentials::MorseParams<double> prm{0.29614, 1.11892, 3.29692};
  double rcut = 4.0;
  potentials::CutoffScheme cs = potentials::CutoffScheme::make(
      potentials::Truncation::Shift, rcut,
      [&](double r, double& u, double& f) { potentials::pair_morse(r, prm, u, f); });

  void operator()(double r, double& u, double& f_over_r) const {
    potentials::pair_morse(r, prm, u, f_over_r);
    cs.apply(r, u, f_over_r);
  }
};

constexpr double kRcut = 4.0;

// FCC lattice shifted by a0/4 along z so that atomic planes (spacing a0/2)
// sit MID-SLAB between zone boundaries — otherwise boundary-sitting atoms
// start with zero slack against the static-membership guard and thermal
// excursions trip StaleZone on perfectly healthy runs.
core::AtomSoA<double> make_fcc(core::Box& box, int cx, int cy, int cz,
                               bool pz, double a0 = 4.05) {
  core::AtomSoA<double> at;
  at.resize(4 * cx * cy * cz);
  box.lo = {0.0, 0.0, 0.0};
  box.hi = {cx * a0, cy * a0, cz * a0};
  box.periodic = {true, true, pz};
  static const double basis[4][3] = {
      {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  int k = 0;
  for (int ix = 0; ix < cx; ++ix)
    for (int iy = 0; iy < cy; ++iy)
      for (int iz = 0; iz < cz; ++iz)
        for (int b = 0; b < 4; ++b, ++k) {
          at.x[k] = (ix + basis[b][0]) * a0;
          at.y[k] = (iy + basis[b][1]) * a0;
          at.z[k] = (iz + basis[b][2]) * a0 + 0.25 * a0;
          at.type[k] = 1;
          at.mass[k] = 26.9815;
        }
  return at;
}

core::AtomSoA<double> load72(core::Box& box) {
  core::AtomSoA<double> atoms;
  EXPECT_TRUE(io::read_lammps_data(
      project_root() + "/reference_data/al_fcc_72.data", atoms, box));
  return atoms;
}

// Serial reference stepper: monolithic velocity-Verlet over the SAME static
// zone decomposition, forces assembled by the serial w-pass (Test_Zones'
// validated path). The conveyor must reproduce it BITWISE at fixed dt.
void serial_stepper(core::AtomSoA<double>& a, const core::Box& box,
                    int n_zones, long steps, double dt) {
  MorsePair pair;
  const auto zd = core::ZoneDecomposition::build(a, box, n_zones, kRcut);
  core::zero_forces(a);
  core::zone_force_pass(a, box, zd, kRcut, pair);
  for (long h = 1; h <= steps; ++h) {
    core::VelocityVerlet<double>::first_half(a, dt);
    core::zero_forces(a);
    core::zone_force_pass(a, box, zd, kRcut, pair);
    core::VelocityVerlet<double>::second_half(a, dt);
  }
}

::testing::AssertionResult bitwise_eq(const core::AtomSoA<double>& a,
                                      const core::AtomSoA<double>& b) {
  if (a.n != b.n) return ::testing::AssertionFailure() << "size mismatch";
  auto cmp = [&](const std::vector<double>& u, const std::vector<double>& v,
                 const char* name) {
    return std::memcmp(u.data(), v.data(), u.size() * sizeof(double)) == 0
               ? ""
               : name;
  };
  std::string bad;
  bad += cmp(a.x, b.x, "x ");   bad += cmp(a.y, b.y, "y ");
  bad += cmp(a.z, b.z, "z ");   bad += cmp(a.vx, b.vx, "vx ");
  bad += cmp(a.vy, b.vy, "vy "); bad += cmp(a.vz, b.vz, "vz ");
  if (bad.empty()) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << "bitwise mismatch in: " << bad;
}

core::ConveyorOptions opts_fixed(long steps, int n_zones, int n_nodes,
                                 double dt) {
  core::ConveyorOptions o;
  o.steps = steps;
  o.n_zones = n_zones;
  o.n_nodes = n_nodes;
  o.auto_step = false;
  o.dt_initial = dt;
  return o;
}

core::AtomSoA<double> run_case(const core::AtomSoA<double>& init,
                               const core::Box& box,
                               const core::ConveyorOptions& o,
                               core::ConveyorResult* out = nullptr) {
  core::AtomSoA<double> a = init;
  auto r = core::run_conveyor(a, box, kRcut, MorsePair{}, o);
  EXPECT_EQ(r.halt, core::Halt::None) << r.halt_msg;
  EXPECT_EQ(r.steps_done, o.steps);
  if (out) *out = std::move(r);
  return a;
}

}  // namespace

// --- conveyor == serial zone-pass stepper, bitwise (fixed dt) ---

TEST(Conveyor, FixedMatchesSerialStepperBitwise) {
  core::Box box;
  auto init = make_fcc(box, 2, 2, 6, /*pz=*/false);  // Lz=24.3, 4 zones of 6.075
  core::thermal::maxwell_init(init, 300.0, 7);

  core::AtomSoA<double> ref = init;
  serial_stepper(ref, box, 4, 60, 0.002);

  auto got = run_case(init, box, opts_fixed(60, 4, 1, 0.002));
  EXPECT_TRUE(bitwise_eq(ref, got));

  // B1 corollary: the same pair set quantized per contribution gives the same
  // integer sums for ANY zone grouping — 1 zone == 4 zones, bitwise.
  auto got1 = run_case(init, box, opts_fixed(60, 1, 1, 0.002));
  EXPECT_TRUE(bitwise_eq(got, got1));
}

// --- §3.6 methodology: 1 node vs z nodes, bitwise, fixed dt ---

TEST(Conveyor, Determinism1vsZFixed) {
  core::Box box;
  auto init = make_fcc(box, 2, 2, 6, /*pz=*/false);
  core::thermal::maxwell_init(init, 300.0, 11);

  const auto o1 = opts_fixed(80, 4, 1, 0.002);
  auto ref = run_case(init, box, o1);
  for (int z : {2, 3, 4, 5}) {
    auto o = o1;
    o.n_nodes = z;
    auto got = run_case(init, box, o);
    EXPECT_TRUE(bitwise_eq(ref, got)) << "z=" << z;
  }
}

// --- 1 vs z with timestep.mode=auto: the Δt-handoff chain (B2) must make the
// dt sequence independent of z; C3=1 retargets dt every pass. ---

TEST(Conveyor, Determinism1vsZAuto) {
  core::Box box;
  auto init = make_fcc(box, 2, 2, 6, /*pz=*/false);
  core::thermal::maxwell_init(init, 300.0, 13);

  core::ConveyorOptions o;
  o.steps = 200;
  o.n_zones = 4;
  o.n_nodes = 1;
  o.auto_step = true;
  o.dt_initial = 0.001;
  o.ts.C1 = 0.01;   // small displacement target -> dt tracks v_max(h-n+1)
  o.ts.C3 = 1.0;    // retarget every pass — the handoff chain is really used
  o.ts.K2 = 50.0;
  o.ts.C_buf = 1.5;
  o.ts.cell_size = 2.33;
  o.ts.dt_max = 0.02;

  core::ConveyorResult r1;
  auto ref = run_case(init, box, o, &r1);
  std::set<double> dts;
  for (const auto& s : r1.stats) dts.insert(s.dt);
  EXPECT_GE(dts.size(), 5u) << "auto-dt did not evolve — chain not exercised";

  // Λ-chain ORACLE (review M3.5): 1-vs-z equality alone holds by construction
  // for ANY chain-indexing bug (the message stream is z-independent), so the
  // documented recurrence dt(h+1) = auto_dt(v_max(h−n+1), dt(h), k2cap(h−n+1))
  // is replayed here from the recorded per-pass aggregates.
  for (std::size_t i = 4; i < r1.stats.size(); ++i) {
    const double want = core::buffer::auto_dt(
        r1.stats[i - 4].v_max, r1.stats[i - 1].dt, o.ts, r1.stats[i - 4].k2cap);
    ASSERT_EQ(want, r1.stats[i].dt) << "Λ-chain recurrence broken at pass "
                                    << i + 1;
  }

  for (int z : {2, 3, 4}) {
    auto oz = o;
    oz.n_nodes = z;
    core::ConveyorResult rz;
    auto got = run_case(init, box, oz, &rz);
    EXPECT_TRUE(bitwise_eq(ref, got)) << "z=" << z;
    ASSERT_EQ(rz.stats.size(), r1.stats.size());
    for (std::size_t i = 0; i < r1.stats.size(); ++i)
      ASSERT_EQ(r1.stats[i].dt, rz.stats[i].dt) << "dt diverged at pass " << i + 1;
  }

  // PBC + auto: the combination no other test covers (rotation closure and
  // the Δt-handoff chain together).
  core::Box pbox;
  auto pinit = make_fcc(pbox, 2, 2, 8, /*pz=*/true);
  core::thermal::maxwell_init(pinit, 300.0, 19);
  auto po = o;
  po.steps = 120;
  core::ConveyorResult p1, p3;
  auto pref = run_case(pinit, pbox, po, &p1);
  po.n_nodes = 3;
  auto pgot = run_case(pinit, pbox, po, &p3);
  EXPECT_TRUE(bitwise_eq(pref, pgot)) << "PBC auto z=3";
  ASSERT_EQ(p1.stats.size(), p3.stats.size());
  for (std::size_t i = 0; i < p1.stats.size(); ++i)
    ASSERT_EQ(p1.stats[i].dt, p3.stats[i].dt) << "PBC dt diverged, pass " << i + 1;
}

// --- the n=1 base case must reproduce the SERIAL auto-step rule
// dt(h+1) = auto_dt(v_max(h), dt(h), k2cap(h)) — trajectory and dt sequence
// bitwise (review M3.5: the implementation used to lag one extra pass). ---

TEST(Conveyor, SerialAutoRuleOracleN1) {
  core::Box box;
  auto init = make_fcc(box, 2, 2, 6, /*pz=*/false);
  core::thermal::maxwell_init(init, 300.0, 23);

  core::ConveyorOptions o;
  o.steps = 150;
  o.n_zones = 1;
  o.n_nodes = 1;
  o.auto_step = true;
  o.dt_initial = 0.001;
  o.ts.C1 = 0.01;
  o.ts.C3 = 1.0;

  core::ConveyorResult r;
  auto got = run_case(init, box, o, &r);

  // Hand-rolled serial reference: same force path (zone_force_pass), the
  // run_simulation dt rule evaluated from post-step atom state.
  MorsePair pair;
  core::AtomSoA<double> a = init;
  const auto zd = core::ZoneDecomposition::build(a, box, 1, kRcut);
  core::zero_forces(a);
  core::zone_force_pass(a, box, zd, kRcut, pair);
  double dt = o.dt_initial;
  for (long h = 1; h <= o.steps; ++h) {
    core::VelocityVerlet<double>::first_half(a, dt);
    core::zero_forces(a);
    core::zone_force_pass(a, box, zd, kRcut, pair);
    core::VelocityVerlet<double>::second_half(a, dt);
    ASSERT_EQ(dt, r.stats[std::size_t(h - 1)].dt) << "dt diverged at pass " << h;
    dt = core::buffer::auto_dt(core::buffer::max_speed(a), dt, o.ts,
                               core::buffer::temperature_limited_dt(a, o.ts.K2));
  }
  EXPECT_TRUE(bitwise_eq(a, got));
}

// --- PBC closure along z (§7.2, rotation variant): bitwise vs the serial
// oracle (whose closure Test_Zones validated against the monolith) and
// momentum conservation through the ring. ---

TEST(Conveyor, PbcClosureMatchesSerialAndConservesMomentum) {
  core::Box box;
  auto init = make_fcc(box, 2, 2, 8, /*pz=*/true);  // Lz=32.4, 4 zones of 8.1
  core::thermal::maxwell_init(init, 300.0, 17);
  const auto p0 = core::thermal::momentum(init);

  core::AtomSoA<double> ref = init;
  serial_stepper(ref, box, 4, 50, 0.002);

  for (int z : {1, 3}) {
    auto got = run_case(init, box, opts_fixed(50, 4, z, 0.002));
    EXPECT_TRUE(bitwise_eq(ref, got)) << "z=" << z;
    const auto p = core::thermal::momentum(got);
    for (int d = 0; d < 3; ++d)
      EXPECT_NEAR(p[d], p0[d], 1e-9) << "momentum drift, z=" << z;
  }
}

// --- anti-deadlock (§7.4): odd and even rings, >= 2z steps each.
// NB (review M3.5): with BUFFERED channels of capacity n+2 a send never
// blocks (measured ring occupancy <= n), so this verifies liveness of the
// shipped configuration; the §7.4 parity order becomes load-bearing only
// with rendezvous transport (MPI ssend — M5a), where this test gains teeth. ---

TEST(Conveyor, AntiDeadlockOddEvenRings) {
  core::Box box;
  auto init = make_fcc(box, 2, 2, 6, /*pz=*/false);
  for (int z = 1; z <= 6; ++z) {
    const long steps = 2 * z + 3;
    auto o = opts_fixed(steps, 4, z, 0.002);
    core::AtomSoA<double> a = init;
    auto r = core::run_conveyor(a, box, kRcut, MorsePair{}, o);
    EXPECT_EQ(r.halt, core::Halt::None) << "z=" << z << ": " << r.halt_msg;
    EXPECT_EQ(r.steps_done, steps) << "z=" << z;
  }
}

// --- INV-4 fires in the ring exactly like in the serial driver: force
// appearing FASTER than the entry-state forecast (flight into the wall). ---

TEST(Conveyor, CausalityHaltFires) {
  core::Box box;
  box.lo = {0.0, 0.0, 0.0};
  box.hi = {40.0, 12.0, 12.0};
  box.periodic = {false, false, false};
  core::AtomSoA<double> a;
  a.resize(2);
  a.x[0] = 17.9; a.x[1] = 22.1;       // r = 4.2 — just beyond rcut 4.0
  a.y[0] = a.y[1] = 6.0;
  a.z[0] = a.z[1] = 6.0;
  a.vx[0] = 20.0; a.vx[1] = -20.0;    // closing fast: force pops mid-step
  a.type = {1, 1};
  a.mass = {1.0, 1.0};                // light -> huge acceleration

  auto o = opts_fixed(10, 1, 1, 0.02);
  auto r = core::run_conveyor(a, box, kRcut, MorsePair{}, o);
  EXPECT_EQ(r.halt, core::Halt::Causality) << r.halt_msg;
}

// --- static-membership guard (M3.5 [ENG]): an atom wandering deeper than
// (width - rcut)/2 into a non-adjacent slab must HALT, not silently lose
// pairs. Migration is an M4 work item. ---

TEST(Conveyor, StaleZoneGuardFires) {
  core::Box box;
  box.lo = {0.0, 0.0, 0.0};
  box.hi = {12.0, 12.0, 15.0};        // 3 zones of width 5 >= rcut 4
  box.periodic = {false, false, false};
  core::AtomSoA<double> a;
  a.resize(2);                         // zones 0 and 1; zone 2 stays empty
  for (int i = 0; i < 2; ++i) {
    a.x[i] = 6.0; a.y[i] = 6.0;
    a.z[i] = 2.5 + 5.0 * i;           // far beyond rcut — no forces, ever
    a.type[i] = 1;
    a.mass[i] = 26.9815;
  }
  a.vz[1] = 1.0;                       // middle-zone atom marches up the axis

  auto o = opts_fixed(200, 3, 1, 0.05);
  auto r = core::run_conveyor(a, box, kRcut, MorsePair{}, o);
  EXPECT_EQ(r.halt, core::Halt::StaleZone) << r.halt_msg;
}

// --- degenerate inputs (review M3.5): empty zones must ride the ring —
// including the FIRST-SENT position under PBC rotation, which carries the
// dt_next header; steps==1; more nodes than steps; the n=2+PBC ban; and a
// coincident pair must HALT Overlap, not silently skip the interaction. ---

TEST(Conveyor, DegenerateInputsAndGuards) {
  // (1) all atoms in zone 0 of a 3-zone periodic box: zones 1, 2 stay empty
  // and each becomes the pass head (and the dt-carrying first-sent zone) as
  // the rotation cycles.
  core::Box box;
  box.lo = {0.0, 0.0, 0.0};
  box.hi = {12.0, 12.0, 15.0};
  box.periodic = {true, true, true};
  core::AtomSoA<double> a;
  a.resize(4);
  const double pos[4][3] = {
      {3.0, 3.0, 2.0}, {6.0, 6.0, 2.5}, {9.0, 9.0, 3.0}, {4.0, 8.0, 2.2}};
  for (int i = 0; i < 4; ++i) {
    a.x[i] = pos[i][0]; a.y[i] = pos[i][1]; a.z[i] = pos[i][2];
    a.type[i] = 1;
    a.mass[i] = 26.9815;
  }
  core::AtomSoA<double> ref = a;
  serial_stepper(ref, box, 3, 7, 0.001);
  core::AtomSoA<double> got = a;
  auto r = core::run_conveyor(got, box, kRcut, MorsePair{},
                              opts_fixed(7, 3, 2, 0.001));
  EXPECT_EQ(r.halt, core::Halt::None) << r.halt_msg;
  EXPECT_TRUE(bitwise_eq(ref, got));

  // (2) steps == 1 with more nodes than steps (idle nodes must exit cleanly)
  core::Box fbox;
  auto init = make_fcc(fbox, 2, 2, 6, /*pz=*/false);
  core::AtomSoA<double> ref1 = init;
  serial_stepper(ref1, fbox, 4, 1, 0.002);
  auto got1 = run_case(init, fbox, opts_fixed(1, 4, 6, 0.002));
  EXPECT_TRUE(bitwise_eq(ref1, got1));

  // (3) 2 zones + periodic z double-counts every pair — must throw (build())
  core::AtomSoA<double> b = init;
  core::Box pbox = fbox;
  pbox.periodic = {true, true, true};
  EXPECT_THROW(core::run_conveyor(b, pbox, kRcut, MorsePair{},
                                  opts_fixed(2, 2, 1, 0.001)),
               std::invalid_argument);

  // (4) coincident atoms: PairGeom skips the degenerate pair from evaluation,
  // but min_r2 must still see it -> Halt::Overlap (B10), not a silent no-force run
  core::AtomSoA<double> c;
  c.resize(2);
  c.x = {6.0, 6.0}; c.y = {6.0, 6.0}; c.z = {2.5, 2.5};
  c.type = {1, 1};
  c.mass = {26.9815, 26.9815};
  core::Box cbox;
  cbox.lo = {0.0, 0.0, 0.0};
  cbox.hi = {12.0, 12.0, 12.0};
  cbox.periodic = {false, false, false};
  auto rc = core::run_conveyor(c, cbox, kRcut, MorsePair{},
                               opts_fixed(2, 1, 1, 0.001));
  EXPECT_EQ(rc.halt, core::Halt::Overlap) << rc.halt_msg;
}

// --- §3.6 replica: Al-72 (golden), free boundaries, auto-step (дисс. C1=10
// == cfg.C1=0.1), >= 25 900 steps, coordinates AND velocities bitwise,
// 1 node vs 4 nodes (the dissertation's processor count). Free z + 2 zones
// is a bipartition — membership-complete for any trajectory. Historical
// determinism replica of Гл. 3.6 (рис. 43–44, нулевое отклонение), NOT a
// melting measurement. ---

TEST(Conveyor, Replica36BitwiseDeterminism) {
  core::Box box;
  auto init = load72(box);
  ASSERT_EQ(init.n, 72);
  box.periodic = {false, false, false};
  core::thermal::maxwell_init(init, 300.0, 1);

  core::ConveyorOptions o;
  o.steps = 25900;
  o.n_zones = 2;
  o.n_nodes = 1;
  o.auto_step = true;
  o.dt_initial = 0.001;
  o.ts = {};            // C1=0.1 (дисс. С1=10), K2=50, C3=0.5, C_buf=1.5
  o.ts.cell_size = 2.33;

  core::ConveyorResult r1, r4;
  auto ref = run_case(init, box, o, &r1);
  o.n_nodes = 4;
  auto got = run_case(init, box, o, &r4);

  EXPECT_TRUE(bitwise_eq(ref, got));  // нулевое отклонение — как в Гл. 3.6
  ASSERT_EQ(r1.stats.size(), r4.stats.size());
  for (std::size_t i = 0; i < r1.stats.size(); ++i)
    ASSERT_EQ(r1.stats[i].dt, r4.stats[i].dt) << "dt diverged at pass " << i + 1;
}
