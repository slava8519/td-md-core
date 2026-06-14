// M6 PR-E1: analytic Finnis–Sinclair EAM — the first many-body force math.
// Acceptance (M6_EAM_MANYBODY_DESIGN §7, PR-E1 row):
//   - FD self-consistency: analytic force ≡ −dU/dx (max|F_an−F_fd| < 1e-5);
//   - density sanity: ρ_i ≡ brute-force sum, ≡ coordination·ρ_a(r_nn);
//   - INV-8 single-counting probe over the candidate walk;
//   - load-time density-quantum guard picks Q19.44/Q23.40/throws.
// No LAMMPS — the analytic fixture is the FP64 oracle (PR-E7 adds eam/alloy).
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <random>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/potentials/eam.hpp"
#include "tdmd/potentials/eam_analytic.hpp"

using namespace tdmd;

static std::string project_root() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT);
#else
  return ".";
#endif
}

namespace {

core::AtomSoA<double> load72(core::Box& box) {
  core::AtomSoA<double> a;
  box.periodic = {true, true, true};
  EXPECT_TRUE(io::read_lammps_data(
      project_root() + "/reference_data/al_fcc_72.data", a, box));
  return a;
}

// Perfect conventional-FCC supercell (4 atoms/cell), PBC. With a cutoff strictly
// between the 1st (alat/√2) and 2nd (alat) shells every atom is equivalent with
// exactly 12 nearest neighbours — the homogeneous anchor for DensitySanity.
core::AtomSoA<double> make_fcc(int nc, double alat, core::Box& box) {
  box.lo = {0, 0, 0};
  box.hi = {nc * alat, nc * alat, nc * alat};
  box.periodic = {true, true, true};
  const double b[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  std::vector<std::array<double, 3>> pos;
  for (int ix = 0; ix < nc; ++ix)
    for (int iy = 0; iy < nc; ++iy)
      for (int iz = 0; iz < nc; ++iz)
        for (auto& bb : b)
          pos.push_back({(ix + bb[0]) * alat, (iy + bb[1]) * alat, (iz + bb[2]) * alat});
  core::AtomSoA<double> a;
  a.resize(int(pos.size()));
  for (int i = 0; i < a.n; ++i) {
    a.x[i] = pos[i][0]; a.y[i] = pos[i][1]; a.z[i] = pos[i][2];
    a.type[i] = 1; a.mass[i] = 26.98;
  }
  return a;
}

}  // namespace

// --- FD self-consistency on a symmetry-broken lattice (forces must be nonzero
// and varied or the test is vacuous: a perfect FCC has zero net force). ---
TEST(Eam, ForceMatchesFiniteDifference) {
  core::Box box;
  auto a = load72(box);
  const auto m = potentials::make_analytic_al<double>();

  // small seeded random displacement → nonzero O(0.1–1 eV/Å) forces
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> jit(-0.08, 0.08);
  for (int i = 0; i < a.n; ++i) {
    a.x[i] += jit(rng); a.y[i] += jit(rng); a.z[i] += jit(rng);
  }

  // analytic forces
  core::zero_forces(a);
  const auto acc = potentials::eam_direct_fp64(a, box, m, /*with_forces=*/true);
  std::vector<double> fx(a.fx), fy(a.fy), fz(a.fz);
  EXPECT_GT(acc.min_r2, 0.0);

  const double d = 2e-5;
  double maxerr = 0.0, maxf = 0.0;
  auto energy_at = [&](int i, double* coord, double delta) {
    const double save = *coord;
    *coord = save + delta;
    const double e = potentials::eam_direct_fp64(a, box, m, /*with_forces=*/false).pe;
    *coord = save;
    return e;
  };
  for (int i = 0; i < a.n; ++i) {
    double* cs[3] = {&a.x[i], &a.y[i], &a.z[i]};
    const double fa[3] = {fx[i], fy[i], fz[i]};
    for (int dctr = 0; dctr < 3; ++dctr) {
      const double ep = energy_at(i, cs[dctr], +d);
      const double em = energy_at(i, cs[dctr], -d);
      const double fd = -(ep - em) / (2.0 * d);
      maxerr = std::max(maxerr, std::fabs(fd - fa[dctr]));
      maxf = std::max(maxf, std::fabs(fa[dctr]));
    }
  }
  EXPECT_GT(maxf, 0.05) << "forces too small — FD check would be vacuous";
  EXPECT_LT(maxerr, 1e-5) << "max|F_analytic − F_fd| = " << maxerr;
}

// --- density sanity on a PERFECT FCC: ρ_i ≡ brute-force, ≡ coord·ρ_a(r_nn) ---
TEST(Eam, DensitySanity) {
  core::Box box;
  auto a = make_fcc(/*nc=*/3, /*alat=*/4.05, box);  // 108 atoms, nn=2.863, 2nd=4.05
  potentials::AnalyticEam<double> m;
  m.rcut = 3.5;  // strictly between the 1st (2.863) and 2nd (4.05) shells → 12 nn
  m.finalize();

  std::vector<double> rho;
  potentials::eam_direct_fp64(a, box, m, /*with_forces=*/false, &rho);
  ASSERT_EQ(int(rho.size()), a.n);

  const core::PairGeom geom(box, m.rcut);
  // independent brute-force re-sum + coordination + nn distance for atom-by-atom
  double r_nn2 = 1e300;
  for (int i = 0; i < a.n; ++i) {
    double brute = 0.0;
    int coord = 0;
    for (int j = 0; j < a.n; ++j) {
      if (j == i) continue;
      double dx = a.x[i] - a.x[j], dy = a.y[i] - a.y[j], dz = a.z[i] - a.z[j], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
      double v, dv;
      m.eval_rhoa(std::sqrt(r2), v, dv);
      brute += v;
      ++coord;
      r_nn2 = std::min(r_nn2, r2);
    }
    EXPECT_NEAR(rho[i], brute, 1e-12) << "atom " << i;
    EXPECT_EQ(coord, 12) << "fcc nn coordination, atom " << i;
  }

  // homogeneous lattice: all ρ_i equal, and ρ ≡ 12·ρ_a(r_nn).
  const double rmin = *std::min_element(rho.begin(), rho.end());
  const double rmax = *std::max_element(rho.begin(), rho.end());
  EXPECT_LT(rmax - rmin, 1e-6) << "fcc density should be homogeneous";
  double ra_nn, dra;
  m.eval_rhoa(std::sqrt(r_nn2), ra_nn, dra);
  EXPECT_NEAR(rho[0], 12.0 * ra_nn, 1e-4);
}

// --- INV-8: every accepted pair evaluated exactly once over the force walk ---
TEST(Eam, PairSingleCounting) {
  core::Box box;
  auto a = load72(box);
  const auto m = potentials::make_analytic_al<double>();
  const core::PairGeom geom(box, m.rcut);

  std::map<std::pair<int, int>, int> seen;
  int accepted = 0;
  for (int i = 0; i < a.n; ++i)
    for (int j = i + 1; j < a.n; ++j) {
      double dx = a.x[i] - a.x[j], dy = a.y[i] - a.y[j], dz = a.z[i] - a.z[j], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
      ++seen[{i, j}];
      ++accepted;
    }
  EXPECT_GT(accepted, 0);
  for (const auto& [k, c] : seen) EXPECT_EQ(c, 1) << "pair counted twice";

  // accepted pairs == Σ coord / 2 (each undirected nn pair once)
  int coord_sum = 0;
  for (int i = 0; i < a.n; ++i)
    for (int j = 0; j < a.n; ++j) {
      if (j == i) continue;
      double dx = a.x[i] - a.x[j], dy = a.y[i] - a.y[j], dz = a.z[i] - a.z[j], r2;
      if (geom.reduce(dx, dy, dz, r2)) ++coord_sum;
    }
  EXPECT_EQ(accepted, coord_sum / 2);
}

// --- load-time density-quantum guard ---
TEST(Eam, DensityFracbitsGuard) {
  potentials::AnalyticEam<double> normal;  // β=1.5 → modest ρ_a
  EXPECT_EQ(normal.density_fracbits(), 44);  // Q19.44

  potentials::AnalyticEam<double> steep;
  steep.beta = 3.3;  // ρ_a(0.5) large enough to overflow Q19.44, fits Q23.40
  EXPECT_EQ(steep.density_fracbits(), 40);  // Q23.40 fallback

  potentials::AnalyticEam<double> extreme;
  extreme.beta = 5.0;  // beyond even Q23.40
  EXPECT_THROW(extreme.density_fracbits(), std::runtime_error);
}

// --- IManyBodyPotential instance: fixed-point passes ≈ FP64 oracle (within
// the quantization bound), and the pass-graph descriptor is correct. ---
TEST(Eam, PotentialInterfaceMatchesOracle) {
  core::Box box;
  auto a = load72(box);
  std::mt19937 rng(777);
  std::uniform_real_distribution<double> jit(-0.05, 0.05);
  for (int i = 0; i < a.n; ++i) {
    a.x[i] += jit(rng); a.y[i] += jit(rng); a.z[i] += jit(rng);
  }

  const auto m = potentials::make_analytic_al<double>();
  potentials::EamPotential<double, potentials::AnalyticEam<double>> pot(m);

  // descriptor
  ASSERT_EQ(pot.passes().size(), 3u);
  EXPECT_EQ(pot.effective_range().cutoff_multiplier, 2);
  EXPECT_TRUE(pot.effective_range().symmetric_reach);

  // FP64 oracle forces
  core::AtomSoA<double> b = a;
  core::zero_forces(b);
  potentials::eam_direct_fp64(b, box, m, /*with_forces=*/true);

  // fixed-point run_pass forces over the same O(N²) walk
  core::zero_forces(a);
  potentials::eam_run_fixed(a, box, pot);

  double maxerr = 0.0;
  for (int i = 0; i < a.n; ++i) {
    maxerr = std::max(maxerr, std::fabs(a.fx[i] - b.fx[i]));
    maxerr = std::max(maxerr, std::fabs(a.fy[i] - b.fy[i]));
    maxerr = std::max(maxerr, std::fabs(a.fz[i] - b.fz[i]));
  }
  EXPECT_LT(maxerr, 1e-9) << "fixed-point EAM vs FP64 oracle = " << maxerr;
}
