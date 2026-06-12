// M3 safety net (audit решение 7): run-to-run BITWISE determinism of the
// extracted run_simulation() — established BEFORE clustering changes the pair
// order, so any ASLR-dependent tie-break or uninitialized read introduced by
// M3 shows up here instead of exploding later in M3.5. Plus the INV-4
// fixed-mode HALT (B10/M12: causality is checked in both timestep modes).
#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "tdmd/core/simulation.hpp"
#include "tdmd/core/thermal.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/io/reader_lammps.hpp"
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

struct Snapshot {
  std::vector<double> x, y, z, vx, vy, vz;
};

// One full thermal run on the golden system; returns the final state.
Snapshot run_once(bool auto_step) {
  core::AtomSoA<double> atoms;
  core::Box box;
  box.periodic = {true, true, true};
  EXPECT_TRUE(io::read_lammps_data(
      project_root() + "/reference_data/al_fcc_72.data", atoms, box));
  core::thermal::maxwell_init(atoms, 300.0, 20070101u);

  potentials::MorsePotential<double> morse;
  core::SimOptions o;
  o.steps = 200;
  o.dt = auto_step ? 0.005 : 0.001;
  o.auto_step = auto_step;
  o.ts = {0.1, 50.0, 0.5, 1.5, 2.33, 0.02, 1e-6};  // = config_auto.yaml
  auto res = core::run_simulation(atoms, box, morse, o, [](long) {});
  EXPECT_EQ(res.halt, core::Halt::None) << res.halt_msg;
  EXPECT_EQ(res.steps_done, 200);

  return {atoms.x, atoms.y, atoms.z, atoms.vx, atoms.vy, atoms.vz};
}

// Bitwise: memcmp on the raw doubles, not EXPECT_NEAR.
void expect_bitwise_equal(const Snapshot& a, const Snapshot& b) {
  auto eq = [](const std::vector<double>& u, const std::vector<double>& v) {
    return u.size() == v.size() &&
           std::memcmp(u.data(), v.data(), u.size() * sizeof(double)) == 0;
  };
  EXPECT_TRUE(eq(a.x, b.x));
  EXPECT_TRUE(eq(a.y, b.y));
  EXPECT_TRUE(eq(a.z, b.z));
  EXPECT_TRUE(eq(a.vx, b.vx));
  EXPECT_TRUE(eq(a.vy, b.vy));
  EXPECT_TRUE(eq(a.vz, b.vz));
}

}  // namespace

TEST(Determinism, RunToRunBitwiseFixedDt) {
  expect_bitwise_equal(run_once(false), run_once(false));
}

TEST(Determinism, RunToRunBitwiseAutoDt) {
  expect_bitwise_equal(run_once(true), run_once(true));
}

// B10/M12: INV-4 fires in FIXED mode too (was gated on timestep.mode=auto).
// Two light atoms fly at each other from OUTSIDE the cutoff (zero force, so
// the entry-state forecast v_pred = v + a·dt sees nothing) and punch into the
// repulsive wall within one step — force appears faster than the forecast,
// v_max outruns the buffer. This is the true causality hazard INV-4 guards.
TEST(Determinism, CausalityHaltFiresInFixedMode) {
  core::AtomSoA<double> atoms;
  atoms.resize(2);
  core::Box box;
  box.lo = {0, 0, 0};
  box.hi = {20, 20, 20};
  box.periodic = {true, true, true};
  atoms.x[0] = 7.9; atoms.x[1] = 12.1;        // r = 4.2 Å — beyond r_cut = 4.0
  atoms.y[0] = atoms.y[1] = 10.0;
  atoms.z[0] = atoms.z[1] = 10.0;
  atoms.mass[0] = atoms.mass[1] = 1.0;        // light => huge acceleration
  atoms.vx[0] = 20.0; atoms.vx[1] = -20.0;    // 0.8 Å closing per step

  potentials::MorsePotential<double> morse;
  core::SimOptions o;
  o.steps = 50;
  o.dt = 0.02;
  o.auto_step = false;                        // the whole point
  o.ts = {0.1, 50.0, 0.5, 1.5, 2.33, 0.02, 1e-6};
  auto res = core::run_simulation(atoms, box, morse, o, [](long) {});
  EXPECT_EQ(res.halt, core::Halt::Causality) << res.halt_msg;
  EXPECT_NE(res.halt_msg.find("INV-4"), std::string::npos);
}

// Regression (M3): a cold start (v=0, fixed dt) must NOT trip INV-4 — the
// ballistic ramp doubles v_max per step, which no constant C_buf can absorb;
// the v_pred = v + a·dt buffer sizing covers it (this is config_m0.yaml).
TEST(Determinism, ColdStartFixedDtRunsClean) {
  core::AtomSoA<double> atoms;
  core::Box box;
  box.periodic = {true, true, true};
  ASSERT_TRUE(io::read_lammps_data(
      project_root() + "/reference_data/al_fcc_72.data", atoms, box));
  potentials::MorsePotential<double> morse;
  core::SimOptions o;
  o.steps = 100;
  o.dt = 0.005;                                // = config_m0.yaml
  o.auto_step = false;
  o.ts = {0.1, 50.0, 0.5, 1.5, 2.33, 0.02, 1e-6};
  auto res = core::run_simulation(atoms, box, morse, o, [](long) {});
  EXPECT_EQ(res.halt, core::Halt::None) << res.halt_msg;
  EXPECT_EQ(res.steps_done, 100);
}

// Overlap HALT at the simulation-driver level (engine smoke covered exit code).
TEST(Determinism, OverlapHaltAtStepZero) {
  core::AtomSoA<double> atoms;
  atoms.resize(2);
  core::Box box;
  box.lo = {0, 0, 0};
  box.hi = {10, 10, 10};
  atoms.x[0] = 5.0; atoms.x[1] = 5.2;         // 0.2 Å < r_min_halt
  atoms.y[0] = atoms.y[1] = 5.0;
  atoms.z[0] = atoms.z[1] = 5.0;
  atoms.mass[0] = atoms.mass[1] = 26.98;

  potentials::MorsePotential<double> morse;
  core::SimOptions o;
  o.steps = 10;
  o.dt = 0.001;
  auto res = core::run_simulation(atoms, box, morse, o, [](long) {});
  EXPECT_EQ(res.halt, core::Halt::Overlap);
  EXPECT_EQ(res.steps_done, 0);
}
