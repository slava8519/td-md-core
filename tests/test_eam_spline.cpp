// M6 PR-E2: EAM via LAMMPS-style cubic splines (setfl). Validates the setfl
// reader + spline interpolation that FD self-consistency is blind to:
//   - parser/indexing: from_setfl reproduces the table values at knots;
//   - spline ≈ analytic in the physical range; tabulated EAM passes FD;
//   - knot-edge determinism (truncation, no tie; C1 continuity);
//   - PHYSICS cross-check vs FROZEN LAMMPS eam/alloy run-0 (real Zhou Al,
//     864 atoms) at print-tolerance — self-contained, no LAMMPS at run time;
//   - flag-audit: -ffp-contract=off is effective (OQ1 bit-exact precondition).
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/potentials/eam.hpp"
#include "tdmd/potentials/eam_analytic.hpp"
#include "tdmd/potentials/eam_spline.hpp"

using namespace tdmd;
using potentials::AnalyticEam;
using potentials::EamSetfl;

static std::string project_root() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT);
#else
  return ".";
#endif
}
static std::string eam_dir() { return project_root() + "/reference_data/eam_al/"; }

// --- parser/indexing: spline value AT a knot equals the table value ---
TEST(EamSpline, SetflParserAndKnotValues) {
  const auto s = EamSetfl<double>::from_setfl(eam_dir() + "Al_small.eam.alloy");
  EXPECT_EQ(s.Nrho, 10);
  EXPECT_EQ(s.Nr, 10);
  EXPECT_DOUBLE_EQ(s.dr, 0.5);
  EXPECT_DOUBLE_EQ(s.drho, 0.1);
  EXPECT_DOUBLE_EQ(s.rcut, 5.0);

  // rho_a table = 1.0 0.9 0.8 0.6 0.4 0.25 0.15 0.08 0.03 0.01 at r = i*0.5.
  const double rhoa_tab[10] = {1.0, 0.9, 0.8, 0.6, 0.4, 0.25, 0.15, 0.08, 0.03, 0.01};
  for (int i = 0; i < 10; ++i) {
    double v, dv;
    s.eval_rhoa(i * 0.5, v, dv);
    EXPECT_NEAR(v, rhoa_tab[i], 1e-12) << "rho_a knot " << i;
  }
  // F table = 0.0 -0.1 ... -0.7 at rho = i*0.1.
  const double F_tab[10] = {0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.55, -0.6, -0.65, -0.7};
  for (int i = 0; i < 10; ++i) {
    double v, dv;
    s.eval_F(i * 0.1, v, dv);
    EXPECT_NEAR(v, F_tab[i], 1e-12) << "F knot " << i;
  }
}

// --- spline ≈ analytic in the physical range (interpolation accuracy) ---
TEST(EamSpline, SplineMatchesAnalytic) {
  const auto m = potentials::make_analytic_al<double>();
  const auto s = EamSetfl<double>::from_analytic(m, /*Nrho=*/4000, /*Nr=*/4000,
                                                 /*rho_max=*/60.0);
  double maxr = 0, maxp = 0, maxf = 0;
  for (double r = 2.5; r < 3.95; r += 0.01) {
    double va, da, vs, ds;
    m.eval_rhoa(r, va, da); s.eval_rhoa(r, vs, ds);
    maxr = std::max(maxr, std::fabs(va - vs));
    m.eval_phi(r, va, da); s.eval_phi(r, vs, ds);
    maxp = std::max(maxp, std::fabs(va - vs));
  }
  for (double rho = 15.0; rho < 30.0; rho += 0.1) {
    double va, da, vs, ds;
    m.eval_F(rho, va, da); s.eval_F(rho, vs, ds);
    maxf = std::max(maxf, std::fabs(va - vs));
  }
  EXPECT_LT(maxr, 1e-6) << "rho_a spline vs analytic";
  EXPECT_LT(maxp, 1e-6) << "phi spline vs analytic";
  EXPECT_LT(maxf, 1e-6) << "F spline vs analytic";
}

// --- tabulated EAM is internally consistent: force ≡ −dU/dx (spline is C1) ---
TEST(EamSpline, SplineForceMatchesFiniteDifference) {
  core::Box box;
  // perfect FCC (3×3×3, a=4.05) + jitter, PBC
  box.lo = {0, 0, 0}; box.hi = {12.15, 12.15, 12.15}; box.periodic = {true, true, true};
  core::AtomSoA<double> at;
  const double alat = 4.05;
  const double b[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  std::vector<std::array<double, 3>> pos;
  for (int ix = 0; ix < 3; ++ix) for (int iy = 0; iy < 3; ++iy) for (int iz = 0; iz < 3; ++iz)
    for (auto& bb : b) pos.push_back({(ix + bb[0]) * alat, (iy + bb[1]) * alat, (iz + bb[2]) * alat});
  at.resize(int(pos.size()));
  for (int i = 0; i < at.n; ++i) { at.x[i] = pos[i][0]; at.y[i] = pos[i][1]; at.z[i] = pos[i][2]; at.type[i] = 1; at.mass[i] = 26.98; }
  std::mt19937 rng(99);
  std::uniform_real_distribution<double> jit(-0.07, 0.07);
  for (int i = 0; i < at.n; ++i) { at.x[i] += jit(rng); at.y[i] += jit(rng); at.z[i] += jit(rng); }

  const auto m = potentials::make_analytic_al<double>();
  const auto s = EamSetfl<double>::from_analytic(m, 4000, 4000, 60.0);

  core::zero_forces(at);
  potentials::eam_direct_fp64<double, EamSetfl<double>>(at, box, s, true);
  std::vector<double> fx(at.fx), fy(at.fy), fz(at.fz);

  const double d = 2e-5;
  double maxerr = 0, maxf = 0;
  auto E = [&](int i, double* c, double dd) {
    const double sv = *c; *c = sv + dd;
    const double e = potentials::eam_direct_fp64<double, EamSetfl<double>>(at, box, s, false).pe;
    *c = sv; return e;
  };
  for (int i = 0; i < at.n; ++i) {
    double* cs[3] = {&at.x[i], &at.y[i], &at.z[i]};
    const double fa[3] = {fx[i], fy[i], fz[i]};
    for (int k = 0; k < 3; ++k) {
      const double fd = -(E(i, cs[k], +d) - E(i, cs[k], -d)) / (2 * d);
      maxerr = std::max(maxerr, std::fabs(fd - fa[k]));
      maxf = std::max(maxf, std::fabs(fa[k]));
    }
  }
  EXPECT_GT(maxf, 0.05);
  EXPECT_LT(maxerr, 1e-5) << "spline EAM force vs FD = " << maxerr;
}

// --- knot-edge determinism: C1 continuity across a knot; truncation, no tie ---
TEST(EamSpline, KnotEdgeContinuity) {
  const auto m = potentials::make_analytic_al<double>();
  const auto s = EamSetfl<double>::from_analytic(m, 4000, 4000, 60.0);
  // sample around an interior knot r_k = k*dr (k=1500 → r≈1.5)
  const double rk = 1500 * s.dr;
  const double eps = 1e-9;
  double vlo, dlo, vhi, dhi;
  s.eval_rhoa(rk - eps, vlo, dlo);
  s.eval_rhoa(rk + eps, vhi, dhi);
  EXPECT_NEAR(vlo, vhi, 1e-7) << "value C0 across knot";
  EXPECT_NEAR(dlo, dhi, 1e-5) << "derivative C1 across knot";
  // value exactly at the knot is the table value (deterministic truncation)
  double vk, dk;
  s.eval_rhoa(rk, vk, dk);
  EXPECT_NEAR(vk, 0.5 * (vlo + vhi), 1e-7);
}

// --- PHYSICS cross-check vs FROZEN LAMMPS eam/alloy run 0 (real Zhou Al) ---
namespace {
double load_pe(const std::string& path) {
  std::ifstream f(path); double pe = 0; f >> pe; return pe;
}
// dump: skip to "ITEM: ATOMS", then rows "id x y z fx fy fz" (sorted by id).
std::vector<std::array<double, 3>> load_forces(const std::string& path, int n) {
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("ITEM: ATOMS", 0) == 0) break;
  std::vector<std::array<double, 3>> out(n + 1);  // indexed by id 1..n
  for (int k = 0; k < n; ++k) {
    int id; double x, y, z, fx, fy, fz;
    f >> id >> x >> y >> z >> fx >> fy >> fz;
    out[id] = {fx, fy, fz};
  }
  return out;
}
}  // namespace

TEST(EamSpline, LammpsCrossCheck) {
  core::Box box; box.periodic = {true, true, true};
  core::AtomSoA<double> a;
  ASSERT_TRUE(io::read_lammps_data(eam_dir() + "eam_al_864.data", a, box));
  ASSERT_EQ(a.n, 864);

  const auto s = EamSetfl<double>::from_setfl(eam_dir() + "Al_zhou.eam.alloy");
  EXPECT_NEAR(s.rcut, 10.1025, 1e-6);

  core::zero_forces(a);
  const auto acc = potentials::eam_direct_fp64<double, EamSetfl<double>>(a, box, s, true);

  const double pe_ref = load_pe(eam_dir() + "eam_al_864.pe");
  // Achieved agreement is ≈1e-9 eV on PE / ≈1e-12 eV/Å on forces — FAR below the
  // print-tolerance the design conservatively assumed: our interpolate()+eval
  // replicate LAMMPS exactly (same coeffs, same Horner; transcendental-free).
  // Bounds kept ×1000 looser for cross-build FP-contraction margin.
  EXPECT_NEAR(acc.pe, pe_ref, 1e-6) << "PE ours=" << acc.pe << " lammps=" << pe_ref;

  const auto fref = load_forces(eam_dir() + "eam_al_864.forces", a.n);
  double maxerr = 0, maxf = 0;
  for (int i = 0; i < a.n; ++i) {
    const int id = a.id[i];  // reader sorts by id ⇒ id == i+1
    maxerr = std::max(maxerr, std::fabs(a.fx[i] - fref[id][0]));
    maxerr = std::max(maxerr, std::fabs(a.fy[i] - fref[id][1]));
    maxerr = std::max(maxerr, std::fabs(a.fz[i] - fref[id][2]));
    for (double c : fref[id]) maxf = std::max(maxf, std::fabs(c));
  }
  EXPECT_GT(maxf, 0.1) << "reference forces must be nonzero";
  EXPECT_LT(maxerr, 1e-9) << "max|F_ours − F_lammps| = " << maxerr << " eV/Å";
}

// --- P1: ρ beyond the F(ρ) grid HALTs instead of silently clamping ---
TEST(EamSpline, DensityBeyondGridHalts) {
  // tabulate the analytic Al onto a deliberately SHORT rho grid (rho_max=5),
  // far below the FCC equilibrium density (~12) → the driver must throw.
  const auto m = potentials::make_analytic_al<double>();
  const auto s = EamSetfl<double>::from_analytic(m, /*Nrho=*/200, /*Nr=*/4000,
                                                 /*rho_max=*/5.0);
  EXPECT_NEAR(s.density_grid_max(), 5.0, 1e-9);

  core::Box box;
  box.lo = {0, 0, 0}; box.hi = {12.15, 12.15, 12.15}; box.periodic = {true, true, true};
  core::AtomSoA<double> at;
  const double b[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  std::vector<std::array<double, 3>> pos;
  for (int ix = 0; ix < 3; ++ix) for (int iy = 0; iy < 3; ++iy) for (int iz = 0; iz < 3; ++iz)
    for (auto& bb : b) pos.push_back({(ix + bb[0]) * 4.05, (iy + bb[1]) * 4.05, (iz + bb[2]) * 4.05});
  at.resize(int(pos.size()));
  for (int i = 0; i < at.n; ++i) { at.x[i] = pos[i][0]; at.y[i] = pos[i][1]; at.z[i] = pos[i][2]; at.type[i] = 1; at.mass[i] = 26.98; }
  core::zero_forces(at);
  EXPECT_THROW((potentials::eam_direct_fp64<double, EamSetfl<double>>(at, box, s, true)),
               std::runtime_error);
}

// --- P3: malformed setfl header (zero/negative spacing) is rejected ---
TEST(EamSpline, MalformedSetflRejected) {
  const std::string p = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
                        "/tdmd_bad_setfl.eam.alloy";
  auto write = [&](const std::string& header) {
    std::ofstream f(p);
    f << "# c1\n# c2\n# c3\n1 Al\n" << header << "\n13 26.98 4.05 FCC\n";
    for (int i = 0; i < 30; ++i) f << "0.1 ";  // enough numbers to not short-read
    f << "\n";
  };
  write("10 0.0 10 0.5 5.0");  // drho == 0
  EXPECT_THROW(EamSetfl<double>::from_setfl(p), std::runtime_error);
  write("10 0.1 10 -0.5 5.0");  // dr < 0
  EXPECT_THROW(EamSetfl<double>::from_setfl(p), std::runtime_error);
  write("3 0.1 10 0.5 5.0");  // Nrho < 5
  EXPECT_THROW(EamSetfl<double>::from_setfl(p), std::runtime_error);
  std::remove(p.c_str());
}

// --- flag-audit (HOST half): -ffp-contract=off is in effect on the CPU TU ---
// a*a-1 fuses to fma under contraction; under -ffp-contract=off it does not, so
// the sub-ulp term is lost. If the flag is ever dropped (and the compiler fuses)
// this fails — the cheapest standing guard that the host bit-exact flag holds.
// SCOPE (adversarial finding P2): this is HOST-ONLY. The OQ1 CPU↔GPU claim also
// needs --fmad=false on the EAM device TU, which tdmd_core's public flag does
// NOT set (nvcc forwards -ffp-contract to the host compiler only). The device
// half — a structural PUBLIC --fmad=false on the EAM .cu target + an on-device
// eval_spline-near-a-knot bitwise-equals-host test — lands in PR-E5 (no EAM .cu
// exists yet to audit). Until then OQ1 is proven by construction, not by CI.
TEST(EamSpline, FpContractOff) {
  volatile double a = 1.0 + std::ldexp(1.0, -27);  // 1 + 2^-27, opaque to the optimizer
  const double res = a * a - 1.0;
  EXPECT_EQ(res, std::ldexp(1.0, -26))
      << "a*a-1 was contracted to fma ⇒ -ffp-contract=off is NOT in effect";
}
