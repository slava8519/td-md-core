// M3 acceptance — EXTERNAL correctness reference for LJ (validation ladder
// §2/§6): pair energy U_pair* and virial W_pair* on the four frozen NIST SRSW
// configurations, matching each NIST cutoff scheme exactly:
//   - r_c*=3.0 truncated (+LRC reported separately)  -> Truncation::Cut
//   - r_c*=4.0 truncated (+LRC)                      -> Truncation::Cut
//   - r_c*=3.0 linear-force-shifted (LFS)            -> Truncation::ForceShift
// Source data + full reference table: reference_data/nist_lj/README.md
// (NIST page "LJ Fluid Reference Calculations: Cuboid Cell", upd. 2026-04-08).
//
// NIST conventions (verbatim from the page):
//   W_pair = -Σ_{i<j} r_ij · dV/dr|r_ij      == drivers' last_virial
//   V_LFS(r) = V(r) - V(rc) - dV/dr|rc (r-rc) == CutoffScheme::ForceShift
// Everything runs in reduced LJ units (ε=σ=1) — the pair math is unit-blind.
// NIST prints 5 significant digits => relative tolerance 1e-4.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>

#include "tdmd/core/soa.hpp"
#include "tdmd/potentials/lj.hpp"

using namespace tdmd;

static std::string project_root() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT);
#else
  return ".";
#endif
}

namespace {

// NIST sample-configuration format (metadata.README): line 1 — box x y z;
// line 2 — N; then "id x y z" per atom, reduced units, coords in [-L/2, L/2].
bool read_nist_config(const std::string& path, core::AtomSoA<double>& a,
                      core::Box& box) {
  std::ifstream in(path);
  if (!in) return false;
  double L[3];
  long n;
  if (!(in >> L[0] >> L[1] >> L[2] >> n)) return false;
  box.lo = {-0.5 * L[0], -0.5 * L[1], -0.5 * L[2]};
  box.hi = {0.5 * L[0], 0.5 * L[1], 0.5 * L[2]};
  box.periodic = {true, true, true};
  a.resize(int(n));
  for (long k = 0; k < n; ++k) {
    int id;
    if (!(in >> id >> a.x[k] >> a.y[k] >> a.z[k])) return false;
    a.id[k] = id;
    a.mass[k] = 1.0;
    a.type[k] = 1;
  }
  return true;
}

struct NistRef {
  const char* file;
  long n;
  double volume;
  // truncated (no shift) pair sums + analytic tail corrections
  double u3, w3, ulrc3;  // r_c* = 3.0
  double u4, w4, ulrc4;  // r_c* = 4.0
  double ufs, wfs;       // r_c* = 3.0 linear-force shift
};

// reference_data/nist_lj/README.md — verbatim NIST table (5 sig. digits).
const NistRef kRefs[] = {
    {"lj_sample_config_periodic1.txt", 800, 1000.0,
     -4.3515e+03, -5.6867e+02, -1.9849e+02,
     -4.4675e+03, -1.2639e+03, -8.3769e+01,
     -3.8709e+03, 3.1754e+02},
    {"lj_sample_config_periodic2.txt", 200, 512.0,
     -6.9000e+02, -5.6846e+02, -2.4230e+01,
     -7.0460e+02, -6.5599e+02, -1.0226e+01,
     -6.2012e+02, -4.4533e+02},
    {"lj_sample_config_periodic3.txt", 400, 1000.0,
     -1.1467e+03, -1.1649e+03, -4.9622e+01,
     -1.1754e+03, -1.3371e+03, -2.0942e+01,
     -1.0210e+03, -9.3578e+02},
    {"lj_sample_config_periodic4.txt", 30, 512.0,
     -1.6790e+01, -4.6249e+01, -5.4517e-01,
     -1.7060e+01, -4.7869e+01, -2.3008e-01,
     -1.5001e+01, -4.3096e+01},
};

constexpr double kRelTol = 1e-4;  // NIST rounds to 5 significant digits

void expect_rel(double got, double want, const char* what, int cfg) {
  EXPECT_NEAR(got, want, std::fabs(want) * kRelTol)
      << what << " mismatch on NIST config " << cfg;
}

// One scheme on one configuration via the direct O(N²) driver.
void check_scheme(const core::AtomSoA<double>& atoms, const core::Box& box,
                  double rcut, potentials::Truncation tr, double u_want,
                  double w_want, const char* what, int cfg) {
  core::AtomSoA<double> a = atoms;
  potentials::LJPotential<double> lj;
  lj.rcut = rcut;
  lj.truncation = tr;
  core::zero_forces(a);
  const double pe = lj.compute(a, box);
  expect_rel(pe, u_want, what, cfg);
  expect_rel(lj.last_virial, w_want, what, cfg);
}

}  // namespace

TEST(LJNist, FrozenConfigEnergiesAndVirials) {
  const std::string dir = project_root() + "/reference_data/nist_lj/";
  int cfg = 0;
  for (const NistRef& ref : kRefs) {
    ++cfg;
    core::AtomSoA<double> atoms;
    core::Box box;
    ASSERT_TRUE(read_nist_config(dir + ref.file, atoms, box)) << ref.file;
    ASSERT_EQ(atoms.n, ref.n);

    check_scheme(atoms, box, 3.0, potentials::Truncation::Cut,
                 ref.u3, ref.w3, "rc=3 cut", cfg);
    check_scheme(atoms, box, 4.0, potentials::Truncation::Cut,
                 ref.u4, ref.w4, "rc=4 cut", cfg);
    check_scheme(atoms, box, 3.0, potentials::Truncation::ForceShift,
                 ref.ufs, ref.wfs, "rc=3 force-shift", cfg);

    // analytic tail corrections (lj_lrc_energy) vs NIST U_LRC*
    expect_rel(potentials::lj_lrc_energy(1.0, 1.0, 3.0, ref.n, ref.volume),
               ref.ulrc3, "U_LRC rc=3", cfg);
    expect_rel(potentials::lj_lrc_energy(1.0, 1.0, 4.0, ref.n, ref.volume),
               ref.ulrc4, "U_LRC rc=4", cfg);
  }
}

// The clustered path must reproduce the direct path on LJ exactly (the M3
// cross-test oracle is potential-agnostic): N=800 liquid, rc=3 + skin.
TEST(LJNist, ClusteredMatchesDirectOnConfig1) {
  core::AtomSoA<double> atoms;
  core::Box box;
  ASSERT_TRUE(read_nist_config(
      project_root() + "/reference_data/nist_lj/lj_sample_config_periodic1.txt",
      atoms, box));

  core::AtomSoA<double> ref = atoms;
  potentials::LJPotential<double> direct;
  direct.rcut = 3.0;
  direct.truncation = potentials::Truncation::ForceShift;
  core::zero_forces(ref);
  const double pe_ref = direct.compute(ref, box);
  std::map<int, int> ref_of_id;
  for (int i = 0; i < ref.n; ++i) ref_of_id[ref.id[i]] = i;

  core::AtomSoA<double> clu = atoms;
  potentials::ClusteredLJ<double> clustered;
  clustered.rcut = 3.0;
  clustered.truncation = potentials::Truncation::ForceShift;
  clustered.skin = 0.3;  // reduced units: rc+skin=3.3 < L/2=5
  clustered.cell = 1.0;
  core::zero_forces(clu);
  const double pe_clu = clustered.compute(clu, box);

  double maxd = 0.0;
  for (int i = 0; i < clu.n; ++i) {
    const int r = ref_of_id.at(clu.id[i]);
    maxd = std::max({maxd, std::fabs(clu.fx[i] - ref.fx[r]),
                     std::fabs(clu.fy[i] - ref.fy[r]),
                     std::fabs(clu.fz[i] - ref.fz[r])});
  }
  std::printf("Test_LJ_NIST: clustered vs direct max|dF|=%.3e (reduced)\n", maxd);
  EXPECT_LT(maxd, 1e-12);
  EXPECT_LT(std::fabs(pe_clu - pe_ref), 1e-9);
  EXPECT_LT(std::fabs(clustered.last_virial - direct.last_virial), 1e-9);
}

// Real=float instantiation of the LJ clustered driver (production_mixed
// contract): pair math fp32, coordinates/cutoff/accumulation fp64.
TEST(LJNist, Fp32PairMathTolerance) {
  core::AtomSoA<double> atoms;
  core::Box box;
  ASSERT_TRUE(read_nist_config(
      project_root() + "/reference_data/nist_lj/lj_sample_config_periodic4.txt",
      atoms, box));

  core::AtomSoA<double> ref = atoms;
  potentials::LJPotential<double> direct;
  direct.rcut = 3.0;
  core::zero_forces(ref);
  direct.compute(ref, box);
  std::map<int, int> ref_of_id;
  for (int i = 0; i < ref.n; ++i) ref_of_id[ref.id[i]] = i;

  core::AtomSoA<float> cf;
  cf.resize(atoms.n);
  cf.x = atoms.x; cf.y = atoms.y; cf.z = atoms.z;
  cf.mass = atoms.mass; cf.type = atoms.type; cf.id = atoms.id;
  potentials::ClusteredLJ<float> clustered;
  clustered.rcut = 3.0;
  clustered.skin = 0.3;
  clustered.cell = 1.0;
  core::zero_forces(cf);
  clustered.compute(cf, box);

  double maxd = 0.0;
  for (int i = 0; i < cf.n; ++i) {
    const int r = ref_of_id.at(cf.id[i]);
    maxd = std::max({maxd, std::fabs(cf.fx[i] - ref.fx[r]),
                     std::fabs(cf.fy[i] - ref.fy[r]),
                     std::fabs(cf.fz[i] - ref.fz[r])});
  }
  std::printf("Test_LJ_NIST: fp32 pair math max|dF|=%.3e (reduced)\n", maxd);
  EXPECT_LT(maxd, 1e-3);
}
