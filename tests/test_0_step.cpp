#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <cstdio>

#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/potentials/morse.hpp"
#include "tdmd/core/soa.hpp"

using namespace tdmd;

static std::string project_root() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT);
#else
  return ".";
#endif
}

// M0 Definition of Done: forces and total PE match the golden Al/Morse dataset
// to <= 1e-6 (FP64). Reference: reference_data/README.md.
TEST(Test0Step, MorseForcesMatchGolden) {
  core::AtomSoA<double> atoms;
  core::Box box;
  box.periodic = {true, true, true};
  ASSERT_TRUE(io::read_lammps_data(
      project_root() + "/reference_data/al_fcc_72.data", atoms, box));
  ASSERT_EQ(atoms.n, 72);

  potentials::MorsePotential<double> morse;  // defaults = dissertation params
  core::zero_forces(atoms);
  const double pe = morse.compute(atoms, box);
  EXPECT_NEAR(pe, 14.7286803884, 1e-6);  // golden PE_total (eV)

  std::ifstream f(project_root() + "/reference_data/reference_forces.csv");
  ASSERT_TRUE(f.good());
  std::string line;
  std::getline(f, line);  // header: id,fx,fy,fz

  double max_err = 0.0;
  int rows = 0;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string cell;
    std::getline(ss, cell, ','); const int id = std::stoi(cell);
    std::getline(ss, cell, ','); const double fx = std::stod(cell);
    std::getline(ss, cell, ','); const double fy = std::stod(cell);
    std::getline(ss, cell, ','); const double fz = std::stod(cell);
    const int i = id - 1;
    ASSERT_GE(i, 0);
    ASSERT_LT(i, atoms.n);
    max_err = std::max(max_err, std::fabs(atoms.fx[i] - fx));
    max_err = std::max(max_err, std::fabs(atoms.fy[i] - fy));
    max_err = std::max(max_err, std::fabs(atoms.fz[i] - fz));
    ++rows;
  }
  EXPECT_EQ(rows, 72);
  std::printf("Test_0_Step: max|F_engine - F_golden| = %.3e eV/A\n", max_err);
  EXPECT_LE(max_err, 1e-6);
}
