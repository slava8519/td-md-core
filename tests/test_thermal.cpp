#include <gtest/gtest.h>

#include <string>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <filesystem>

#include "tdmd/core/thermal.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/units.hpp"

using namespace tdmd;

static std::string project_root() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT);
#else
  return ".";
#endif
}

namespace {

core::AtomSoA<double> golden_atoms() {
  core::AtomSoA<double> atoms;
  core::Box box;
  if (!io::read_lammps_data(project_root() + "/reference_data/al_fcc_72.data",
                            atoms, box))
    throw std::runtime_error("cannot read golden al_fcc_72.data");
  return atoms;
}

}  // namespace

// M2.6 acceptance: T(0) equals the target (criterion ±5%; rescale makes it
// exact up to FP rounding, asserted much tighter).
TEST(Thermal, InitialTemperatureMatchesTarget) {
  auto atoms = golden_atoms();
  const double T_target = 300.0;
  core::thermal::maxwell_init(atoms, T_target, 20070101u);

  const double T0 =
      core::thermal::temperature(atoms, core::thermal::dof_thermal(atoms.n));
  std::printf("Test_Thermal: T(0)=%.10f K (target %.1f)\n", T0, T_target);
  EXPECT_NEAR(T0, T_target, 1e-9);              // exact rescale
  EXPECT_NEAR(T0, T_target, 0.05 * T_target);   // the formal ±5% criterion
}

// M2.6 acceptance: |p_COM| <= 1e-12 after init (amu·Å/ps, per component).
TEST(Thermal, MomentumZeroAfterInit) {
  auto atoms = golden_atoms();
  core::thermal::maxwell_init(atoms, 300.0, 20070101u);
  const auto p = core::thermal::momentum(atoms);
  std::printf("Test_Thermal: |p|=(%.3e %.3e %.3e) amu·Å/ps\n", p[0], p[1], p[2]);
  for (int d = 0; d < 3; ++d) EXPECT_LE(std::fabs(p[d]), 1e-12);
}

// M2.6 acceptance: same seed => bitwise-identical velocities; different seed
// => different velocities.
TEST(Thermal, SeedReproducibilityBitwise) {
  auto a = golden_atoms();
  auto b = golden_atoms();
  auto c = golden_atoms();
  core::thermal::maxwell_init(a, 300.0, 20070101u);
  core::thermal::maxwell_init(b, 300.0, 20070101u);
  core::thermal::maxwell_init(c, 300.0, 20070102u);

  bool differs_from_c = false;
  for (int i = 0; i < a.n; ++i) {
    // bitwise: exact FP equality, not EXPECT_NEAR
    ASSERT_EQ(a.vx[i], b.vx[i]);
    ASSERT_EQ(a.vy[i], b.vy[i]);
    ASSERT_EQ(a.vz[i], b.vz[i]);
    differs_from_c |= (a.vx[i] != c.vx[i]);
  }
  EXPECT_TRUE(differs_from_c);
}

// Gaussian shape sanity: per-component variance of the scaled velocities is
// kB·T/(mvv2e·m) within a statistical tolerance (single species => one sigma).
TEST(Thermal, VelocityVarianceMatchesMaxwell) {
  auto atoms = golden_atoms();
  const double T = 300.0;
  core::thermal::maxwell_init(atoms, T, 20070101u);

  double sum2 = 0.0;
  for (int i = 0; i < atoms.n; ++i)
    sum2 += atoms.vx[i] * atoms.vx[i] + atoms.vy[i] * atoms.vy[i] +
            atoms.vz[i] * atoms.vz[i];
  const double var = sum2 / (3.0 * atoms.n);
  const double sigma2 = units::kB * T / (units::mvv2e * atoms.mass[0]);
  // 3N=216 samples => relative s.e. of variance ~ sqrt(2/216) ≈ 10%
  EXPECT_NEAR(var, sigma2, 0.35 * sigma2);
}

TEST(Thermal, ZeroTemperatureIsNoop) {
  auto atoms = golden_atoms();
  core::thermal::maxwell_init(atoms, 0.0, 20070101u);
  for (int i = 0; i < atoms.n; ++i) {
    EXPECT_EQ(atoms.vx[i], 0.0);
    EXPECT_EQ(atoms.vy[i], 0.0);
    EXPECT_EQ(atoms.vz[i], 0.0);
  }
}

// M2.6: reader_lammps fills velocities from a Velocities section, matched by
// atom id (rows may come in any order).
TEST(Thermal, ReaderParsesVelocitiesSection) {
  namespace fs = std::filesystem;
  const fs::path path = fs::temp_directory_path() / "tdmd_vel_test.data";
  {
    std::ofstream f(path);
    f << "test cell with velocities\n\n"
      << "2 atoms\n"
      << "1 atom types\n\n"
      << "0.0 10.0 xlo xhi\n"
      << "0.0 10.0 ylo yhi\n"
      << "0.0 10.0 zlo zhi\n\n"
      << "Masses\n\n"
      << "1 26.9815385\n\n"
      << "Atoms # atomic\n\n"
      << "2 1 5.0 5.0 5.0\n"
      << "1 1 1.0 2.0 3.0\n\n"
      << "Velocities\n\n"
      << "2 -0.5 0.25 4.0\n"
      << "1 1.5 -2.5 3.5\n";
  }
  core::AtomSoA<double> atoms;
  core::Box box;
  ASSERT_TRUE(io::read_lammps_data(path.string(), atoms, box));
  fs::remove(path);

  ASSERT_EQ(atoms.n, 2);
  // index 0 == id 1, index 1 == id 2
  EXPECT_DOUBLE_EQ(atoms.vx[0], 1.5);
  EXPECT_DOUBLE_EQ(atoms.vy[0], -2.5);
  EXPECT_DOUBLE_EQ(atoms.vz[0], 3.5);
  EXPECT_DOUBLE_EQ(atoms.vx[1], -0.5);
  EXPECT_DOUBLE_EQ(atoms.vy[1], 0.25);
  EXPECT_DOUBLE_EQ(atoms.vz[1], 4.0);
}

// Golden data file has no Velocities section — velocities must stay zero.
TEST(Thermal, GoldenFileWithoutVelocitiesIsZero) {
  auto atoms = golden_atoms();
  for (int i = 0; i < atoms.n; ++i) {
    EXPECT_EQ(atoms.vx[i], 0.0);
    EXPECT_EQ(atoms.vy[i], 0.0);
    EXPECT_EQ(atoms.vz[i], 0.0);
  }
}
