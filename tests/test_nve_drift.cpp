#include <gtest/gtest.h>

#include <string>
#include <cmath>
#include <cstdio>

#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/potentials/morse.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/integrator.hpp"

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
  double pe = morse.compute(atoms, box);
  const double e0 = pe + core::kinetic_energy(atoms);

  // 1 fs is a physically sane step for Al; the M0 config's 5 fs is an aggressive
  // smoke-run default. Energy conservation is asserted at this finer step.
  const double dt = 0.001;  // ps
  const int steps = 100;
  double emin = e0, emax = e0;
  for (int s = 0; s < steps; ++s) {
    core::VelocityVerlet<double>::first_half(atoms, dt);
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
