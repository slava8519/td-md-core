// M6 / SW-ladder T7: external validation — our Stillinger-Weber forces+PE vs a FROZEN
// LAMMPS `pair_style sw run 0` golden (reference_data/sw_si/). CI is LAMMPS-free. NOT a
// bitwise-to-LAMMPS claim — exp/pow differ libm-vs-LAMMPS (~1e-12 floor); the gate is
// ×1000 looser for cross-build margin. Mirrors EamSpline.LammpsCrossCheck.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/potentials/sw.hpp"

namespace core = tdmd::core;
namespace io = tdmd::io;
namespace pot = tdmd::potentials;

namespace {
std::string sw_dir() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT) + "/reference_data/sw_si/";
#else
  return "reference_data/sw_si/";
#endif
}
struct Golden { double pe = 0; std::vector<std::array<double, 3>> f; };  // f indexed by id
Golden load_golden(const std::string& path, int n) {
  std::ifstream in(path);
  Golden g; g.f.assign(n + 1, {0, 0, 0});
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string tok; ss >> tok;
    if (tok == "PE") { ss >> g.pe; continue; }
    const int id = std::stoi(tok);
    double fx, fy, fz; ss >> fx >> fy >> fz;
    if (id >= 1 && id <= n) g.f[id] = {fx, fy, fz};
  }
  return g;
}
}  // namespace

TEST(SwLammps, RunZeroCrossCheck) {
  core::Box box; box.periodic = {true, true, true};
  core::AtomSoA<double> a;
  ASSERT_TRUE(io::read_lammps_data(sw_dir() + "sw_si_64.data", a, box));
  ASSERT_EQ(a.n, 64);

  pot::SwParams sp;
  core::zero_forces(a);
  // sw_direct_fp64 is the FP64 oracle; the int64 production path (sw_run_fixed) ALSO matches
  // this golden directly (~7e-12) and is bitwise-tied to the oracle (T1 G-ORACLE/MB2), so the
  // zone/ring/GPU paths are transitively LAMMPS-validated.
  const auto acc = pot::sw_direct_fp64(a, box, sp, /*with_forces=*/true);

  const auto g = load_golden(sw_dir() + "sw_si_64.forces", a.n);
  EXPECT_NEAR(acc.pe, g.pe, 1e-6) << "PE ours=" << acc.pe << " lammps=" << g.pe;

  double maxerr = 0, maxf = 0;
  for (int i = 0; i < a.n; ++i) {
    const int id = a.id[i];  // reader sorts by id ⇒ id == i+1
    maxerr = std::max({maxerr, std::fabs(a.fx[i] - g.f[id][0]),
                       std::fabs(a.fy[i] - g.f[id][1]), std::fabs(a.fz[i] - g.f[id][2])});
    for (double c : g.f[id]) maxf = std::max(maxf, std::fabs(c));
  }
  EXPECT_GT(maxf, 0.1) << "reference forces must be nonzero";
  EXPECT_LT(maxerr, 1e-9) << "max|F_ours − F_lammps| = " << maxerr << " eV/Å";
}
