#include <gtest/gtest.h>

#include <string>
#include <cmath>
#include <cstdio>

#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/potentials/morse.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/integrator.hpp"
#include "tdmd/core/thermal.hpp"

using namespace tdmd;

static std::string project_root() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT);
#else
  return ".";
#endif
}

// M0 DoD: 100 NVE steps without NaN and small total-energy drift (recorded).
TEST(NveDrift, EnergyConservedNoNaN) {
  core::AtomSoA<double> atoms;
  core::Box box;
  box.periodic = {true, true, true};
  ASSERT_TRUE(io::read_lammps_data(
      project_root() + "/reference_data/al_fcc_72.data", atoms, box));

  potentials::MorsePotential<double> morse;
  core::zero_forces(atoms);
  double pe = morse.compute(atoms, box);
  const double e0 = pe + core::kinetic_energy(atoms);

  // 1 fs is a physically sane step for Al; the M0 config's 5 fs is an aggressive
  // smoke-run default. Energy conservation is asserted at this finer step.
  const double dt = 0.001;  // ps
  const int steps = 100;
  double emin = e0, emax = e0;
  for (int s = 0; s < steps; ++s) {
    core::VelocityVerlet<double>::first_half(atoms, dt);
    core::zero_forces(atoms);
    pe = morse.compute(atoms, box);
    core::VelocityVerlet<double>::second_half(atoms, dt);
    const double e = pe + core::kinetic_energy(atoms);
    ASSERT_FALSE(std::isnan(e) || std::isinf(e));
    emin = std::min(emin, e);
    emax = std::max(emax, e);
  }
  const double rel = (emax - emin) / std::fabs(e0);
  std::printf("Test_NVE_Drift: E0=%.8f eV  drift(max-min)=%.3e eV  rel=%.3e over %d steps\n",
              e0, emax - emin, rel, steps);
  EXPECT_LT(rel, 3e-3);  // bounded oscillation; ~5e-4 measured at dt=1fs/100 steps
}

// M2.6 (B8) acceptance: NVE drift with a thermal (Maxwell, 300 K) start is
// recorded numerically, and total momentum is conserved by the integrator
// (pair forces obey Newton's third law => p stays at FP-rounding level).
TEST(NveDrift, EnergyAndMomentumConservedAt300K) {
  core::AtomSoA<double> atoms;
  core::Box box;
  box.periodic = {true, true, true};
  ASSERT_TRUE(io::read_lammps_data(
      project_root() + "/reference_data/al_fcc_72.data", atoms, box));

  const double T = 300.0;
  core::thermal::maxwell_init(atoms, T, 20070101u);
  ASSERT_NEAR(core::thermal::temperature(
                  atoms, core::thermal::dof_thermal(atoms.n)),
              T, 1e-9);

  potentials::MorsePotential<double> morse;
  core::zero_forces(atoms);
  double pe = morse.compute(atoms, box);
  const double e0 = pe + core::kinetic_energy(atoms);

  const double dt = 0.001;  // ps (1 fs — дисс. §3.5 starts thermal runs at ~1 fs)
  const int steps = 100;
  double emin = e0, emax = e0;
  for (int s = 0; s < steps; ++s) {
    core::VelocityVerlet<double>::first_half(atoms, dt);
    core::zero_forces(atoms);
    pe = morse.compute(atoms, box);
    core::VelocityVerlet<double>::second_half(atoms, dt);
    const double e = pe + core::kinetic_energy(atoms);
    ASSERT_FALSE(std::isnan(e) || std::isinf(e));
    emin = std::min(emin, e);
    emax = std::max(emax, e);
  }

  const double rel = (emax - emin) / std::fabs(e0);
  const auto p = core::thermal::momentum(atoms);
  std::printf("Test_NVE_Drift(300K): E0=%.8f eV  drift=%.3e eV  rel=%.3e  "
              "|p|=(%.3e %.3e %.3e) over %d steps\n",
              e0, emax - emin, rel, p[0], p[1], p[2], steps);
  EXPECT_LT(rel, 3e-3);
  // momentum conservation: starts <=1e-12, per-step force-sum rounding can
  // accumulate ~1e-13/step at most over 100 steps
  for (int d = 0; d < 3; ++d) EXPECT_LE(std::fabs(p[d]), 1e-10);
}
