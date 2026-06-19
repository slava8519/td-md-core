// M6 / Tersoff-ladder Te1: the LOAD-BEARING external witness — our Tersoff forces+PE vs a
// FROZEN LAMMPS `pair_style tersoff run 0` golden. For Tersoff this is NOT just a nice check
// (as for SW): a force-recomputation oracle shares the b_ij derivative algebra, so this golden
// + G-FD are the ONLY independent witnesses of the algebra, and the only witness of the
// parameter traps (h-in-g, m=3-odd λ₃³). CI is LAMMPS-free. Mirrors Test_SW_LAMMPS.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/potentials/tersoff.hpp"

namespace core = tdmd::core;
namespace io = tdmd::io;
namespace pot = tdmd::potentials;

namespace {
std::string ters_dir() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT) + "/reference_data/tersoff_si/";
#else
  return "reference_data/tersoff_si/";
#endif
}
struct Golden { double pe = 0; std::vector<std::array<double, 3>> f; };
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

TEST(TersoffLammps, RunZeroCrossCheck) {
  core::Box box; box.periodic = {true, true, true};
  core::AtomSoA<double> a;
  ASSERT_TRUE(io::read_lammps_data(ters_dir() + "tersoff_si_64.data", a, box));
  ASSERT_EQ(a.n, 64);

  pot::TersoffParams p;
  const core::PairGeom geom(box, p.rcut());
  core::zero_forces(a);
  // the int64 production path is bitwise-tied to the FP64 oracle (G-ORACLE) ⇒ validating the
  // oracle against LAMMPS transitively validates the int64/zone/ring paths.
  const auto acc = pot::tersoff_direct_fp64(a, geom, p, /*with_forces=*/true);

  const auto g = load_golden(ters_dir() + "tersoff_si_64.forces", a.n);
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
