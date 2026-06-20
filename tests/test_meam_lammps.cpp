// M6 / MEAM-ladder Me1: the LOAD-BEARING external witness — our MEAM total ENERGY vs a FROZEN
// LAMMPS `pair_style meam run 0` golden. For MEAM this golden is MORE load-bearing than for
// Tersoff: MEAM's energy itself runs the screening + partial-density VALUE algebra (a self-check
// oracle shares even the candidate set), so FD-of-energy (Me2) is independent of force-assembly
// bugs but NOT of a shared-value bug ⇒ the LAMMPS golden is the ONLY value-algebra witness in
// Me1. CI is LAMMPS-free (frozen reference_data/meam_si/). CONTRACT: tolerance (std::exp vs
// LAMMPS fm_exp ~1 ulp/op), NOT bitwise — MEASURED here ~3e-13 (std::exp happened to agree).
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
#include "tdmd/potentials/meam.hpp"

namespace core = tdmd::core;
namespace io = tdmd::io;
namespace pot = tdmd::potentials;

namespace {
std::string meam_dir() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT) + "/reference_data/meam_si/";
#else
  return "reference_data/meam_si/";
#endif
}
double load_golden_pe(const std::string& path) {
  std::ifstream in(path); std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line); std::string tok; ss >> tok;
    if (tok == "PE") { double pe; ss >> pe; return pe; }
  }
  return 0.0;
}
std::vector<std::array<double, 3>> load_golden_forces(const std::string& path, int n) {
  std::ifstream in(path); std::string line;
  std::vector<std::array<double, 3>> f(n + 1, {0, 0, 0});
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line); int id; double fx, fy, fz;
    if (ss >> id >> fx >> fy >> fz && id >= 1 && id <= n) f[id] = {fx, fy, fz};
  }
  return f;
}
// the int64 production force path (transitively golden-validated) vs the frozen LAMMPS forces.
double max_force_err(core::AtomSoA<double>& a, const core::Box& box, const pot::MeamParams& p,
                     const std::string& fpath) {
  core::zero_forces(a);
  pot::meam_run_fixed_force(a, core::PairGeom(box, p.rc), p);
  const auto g = load_golden_forces(fpath, a.n);
  double m = 0;
  for (int i = 0; i < a.n; ++i) {
    const int id = a.id[i];
    m = std::max({m, std::fabs(a.fx[i] - g[id][0]), std::fabs(a.fy[i] - g[id][1]), std::fabs(a.fz[i] - g[id][2])});
  }
  return m;
}
}  // namespace

TEST(MeamLammps, RunZeroEnergyCrossCheck) {
  core::Box box; box.periodic = {true, true, true};
  core::AtomSoA<double> a;
  ASSERT_TRUE(io::read_lammps_data(meam_dir() + "meam_si_64.data", a, box));
  ASSERT_EQ(a.n, 64);

  pot::MeamParams p;
  const core::PairGeom geom(box, p.rc);
  const auto acc = pot::meam_energy(a, geom, p);
  const double golden = load_golden_pe(meam_dir() + "meam_si_64.energy");
  ASSERT_LT(golden, -1.0) << "golden not loaded";

  EXPECT_NEAR(acc.pe, golden, 1e-4) << "MEAM total energy ours=" << acc.pe << " lammps=" << golden;
  // the golden genuinely exercises screening (non-vacuous) — the screened-zero pairs are real.
  EXPECT_GT(acc.n_screened_zero, 0) << "no screened pairs — the golden would not test screening";
  // Me2 FORCE: the int64 production path vs the frozen LAMMPS forces (the embedding + pair +
  // density-derivative chains; the screening DERIVATIVE is structurally dead here — binary S).
  EXPECT_LT(max_force_err(a, box, p, meam_dir() + "meam_si_64.forces"), 1e-9)
      << "MEAM forces (64-atom diamond) ≠ LAMMPS golden";
}

// G-SCREEN-PARTIAL ⭐ (acceptance MUST-FIX) — the PARTIAL screening band (0<S<1), the literal MEAM
// signature, is STRUCTURALLY DEAD on the diamond fixture (310 S=0, 0 partial; the reviewer proved
// by mutation that corrupting fcut(C) in the partial branch changes nothing there — the Te1 fc_d
// finding). A free 3-atom cluster reaches the band cleanly (i-j screened by k, S≈0.50, all bonds
// above the ZBL region) and its energy is anchored to an INDEPENDENT LAMMPS golden ⇒ a wrong fcut
// argument / (C−Cmin)/delc normalization is now caught.
TEST(MeamLammps, PartialScreeningClusterMatchesLammps) {
  core::Box box; box.periodic = {false, false, false};
  core::AtomSoA<double> a;
  ASSERT_TRUE(io::read_lammps_data(meam_dir() + "meam_tri3.data", a, box));
  ASSERT_EQ(a.n, 3);

  pot::MeamParams p;
  const auto acc = pot::meam_energy(a, core::PairGeom(box, p.rc), p);
  const double golden = load_golden_pe(meam_dir() + "meam_tri3.energy");
  ASSERT_LT(golden, 0.0) << "cluster golden not loaded";

  EXPECT_GT(acc.n_screened_partial, 0) << "the 0<S<1 partial band never fired — gate vacuous";
  EXPECT_NEAR(acc.pe, golden, 1e-5) << "partial-screening energy ours=" << acc.pe << " lammps=" << golden;
  // Me2 FORCE — the SOLE witness of the screening DERIVATIVE (dscrfcn/dCfunc/dCfunc2/the k-loop
  // 3rd-atom write ∂S/∂x_k). The diamond golden is BLIND to it (binary S ⇒ ∂S=0); here atom-3's
  // force is PURE ∂S/∂x_k (fy≈+56.16). A dCfunc/dscrfcn sign bug fails here but passes the diamond.
  EXPECT_LT(max_force_err(a, box, p, meam_dir() + "meam_tri3.forces"), 1e-9)
      << "MEAM screening-force (3-atom cluster) ≠ LAMMPS golden";
}

// Me3 — the partial-screening SLAB (8 tiled partial-S triples, the zone-MEAM screening-force
// fixture): the int64 scatter force vs the frozen LAMMPS golden. Transitively validates the zone
// path (Test_MEAM_Zone proves zone ≡ this scatter bitwise, and ≡ the FP64 oracle).
TEST(MeamLammps, PartialScreeningSlabForcesMatchLammps) {
  core::Box box; box.periodic = {true, true, true};
  core::AtomSoA<double> a;
  ASSERT_TRUE(io::read_lammps_data(meam_dir() + "partial_slab.data", a, box));
  pot::MeamParams p;
  const double pe = pot::meam_energy(a, core::PairGeom(box, p.rc), p).pe;
  EXPECT_NEAR(pe, load_golden_pe(meam_dir() + "partial_slab.energy"), 1e-4) << "slab PE";
  EXPECT_LT(max_force_err(a, box, p, meam_dir() + "partial_slab.forces"), 1e-9)
      << "MEAM partial-screening slab forces ≠ LAMMPS golden";
}
