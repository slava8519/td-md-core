// M6 / Tersoff-ladder Te1: bond-order angular potential. Design wf_1816b773-9f2.
// LOAD-BEARING (the b_ij algebra escalation): G-FD (FD-of-ENERGY, the only in-CI independent
// witness of the derivative algebra — a force-recomputation oracle shares the b_ij chain rule
// and goes blind) + G-LAMMPS (test_tersoff_lammps.cpp). Supporting: G-TI, G-B1, G-BIJ-LIMITS,
// G-ORACLE/MB2 (enumeration only), G-VACUITY.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/potentials/tersoff.hpp"

namespace core = tdmd::core;
namespace pot = tdmd::potentials;
using tdmd::gen::make_diamond_si;

namespace {
double max_force_diff(const core::AtomSoA<double>& a, const core::AtomSoA<double>& b) {
  double m = 0;
  for (int i = 0; i < a.n; ++i)
    m = std::max({m, std::fabs(a.fx[i] - b.fx[i]), std::fabs(a.fy[i] - b.fy[i]), std::fabs(a.fz[i] - b.fz[i])});
  return m;
}
double max_abs_force(const core::AtomSoA<double>& a) {
  double m = 0;
  for (int i = 0; i < a.n; ++i)
    m = std::max({m, std::fabs(a.fx[i]), std::fabs(a.fy[i]), std::fabs(a.fz[i])});
  return m;
}
}  // namespace

// G-FD ⭐⭐ — the analytic bond-order force equals −dU/dx by central difference (independent
// of the force code — tersoff_energy calls only VALUE helpers, never *_d). THE load-bearing
// algebra witness (caught the attractive-radial /r bug). Run on TWO fixtures: a0=5.431 (the
// standard, all bonds in the flat fc≡1 region) AND a0=6.70 STRETCHED (nn≈2.90 Å lands in the
// taper [2.8,3.2] ⇒ exercises ters_fc_d + the dfc taper terms in zetaterm_d/fa_d/repulsive —
// the acceptance MUST-FIX: on the standard fixture max|fc_d|=0, so a cutoff-derivative bug was
// invisible to all gates). h=1e-6 ≪ the 0.4-Å taper width ⇒ no C¹-knot straddle.
TEST(Tersoff, ForceMatchesFiniteDifferenceOfEnergy) {
  pot::TersoffParams p;
  for (auto fx : {std::pair{5.431, 0.20}, std::pair{6.70, 0.10}}) {  // standard + taper-shell
    core::Box box;
    auto a = make_diamond_si(2, 2, 2, fx.first, fx.second, box);
    box.periodic = {true, true, true};
    const core::PairGeom geom(box, p.rcut());
    core::AtomSoA<double> g = a;
    core::zero_forces(g);
    pot::tersoff_direct_fp64(g, geom, p, true);
    const double h = 1e-6;
    double maxerr = 0;
    for (int i = 0; i < std::min(a.n, 16); ++i) {
      double* c[3] = {&a.x[i], &a.y[i], &a.z[i]};
      const double fan[3] = {g.fx[i], g.fy[i], g.fz[i]};
      for (int d = 0; d < 3; ++d) {
        const double x0 = *c[d];
        *c[d] = x0 + h; const double ep = pot::tersoff_energy(a, geom, p);
        *c[d] = x0 - h; const double em = pot::tersoff_energy(a, geom, p);
        *c[d] = x0;
        maxerr = std::max(maxerr, std::fabs(fan[d] - (-(ep - em) / (2 * h))));
      }
    }
    EXPECT_LT(maxerr, 1e-5) << "bond-order force ≠ −dU/dx (algebra bug), a0=" << fx.first;
  }
}

// G-FC ⭐ — direct FD of the C¹ cosine cutoff: ters_fc_d == d(ters_fc)/dr across the taper
// [R−D, R+D]. The acceptance MUST-FIX: fc_d is otherwise structurally unexercised on the
// standard fixture, so a sign/constant bug in it stays green. Skip the exact knot (C¹ kink).
TEST(Tersoff, CutoffDerivativeMatchesFD) {
  pot::TersoffParams p;
  const double h = 1e-7;
  double maxerr = 0, maxd = 0;
  for (double r = p.bigr - p.bigd + 0.01; r < p.bigr + p.bigd - 0.01; r += 0.005) {
    const double fd = (pot::ters_fc(r + h, p) - pot::ters_fc(r - h, p)) / (2 * h);
    maxerr = std::max(maxerr, std::fabs(pot::ters_fc_d(r, p) - fd));
    maxd = std::max(maxd, std::fabs(pot::ters_fc_d(r, p)));
  }
  EXPECT_GT(maxd, 0.5) << "the taper-shell sweep never exercised fc_d";
  EXPECT_LT(maxerr, 1e-6) << "ters_fc_d ≠ d(ters_fc)/dr";
}

// G-TI — the angular zetaterm is translationally invariant (∂ζ/∂x_i+∂x_j+∂x_k = 0).
TEST(Tersoff, ZetaTermForceSumsToZero) {
  pot::TersoffParams p;
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> dir(-1, 1), mag(2.0, 3.0);
  double maxsum = 0;
  for (int t = 0; t < 10000; ++t) {
    auto bond = [&](double* hat, double& r) {
      double x = dir(rng), y = dir(rng), z = dir(rng), n = std::sqrt(x*x+y*y+z*z);
      if (n < 1e-6) { x = 1; y = z = 0; n = 1; }
      r = mag(rng); hat[0] = x/n; hat[1] = y/n; hat[2] = z/n;
    };
    double rijh[3], rikh[3], rij, rik;
    bond(rijh, rij); bond(rikh, rik);
    double dri[3], drj[3], drk[3];
    pot::tersoff_zetaterm_d(-0.37, rijh, rij, 1/rij, rikh, rik, 1/rik, dri, drj, drk, p);
    for (int d = 0; d < 3; ++d) maxsum = std::max(maxsum, std::fabs(dri[d]+drj[d]+drk[d]));
  }
  EXPECT_LT(maxsum, 1e-12) << "zetaterm not translationally invariant";
}

// G-B1 — int64 forces+PE bitwise-invariant under atom relabeling (order-free B1/INV-9).
TEST(Tersoff, FixedPointBitwiseUnderRelabeling) {
  pot::TersoffParams p;
  core::Box box;
  auto a = make_diamond_si(2, 2, 2, 5.431, 0.20, box);
  box.periodic = {true, true, true};
  const core::PairGeom geom(box, p.rcut());

  core::AtomSoA<double> a1 = a; core::zero_forces(a1);
  const double pe1 = pot::tersoff_run_fixed(a1, geom, p).pe;
  std::vector<int> perm(a.n); std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), std::mt19937(2024));
  core::AtomSoA<double> a2; a2.resize(a.n);
  for (int i = 0; i < a.n; ++i) { a2.x[i]=a.x[perm[i]]; a2.y[i]=a.y[perm[i]]; a2.z[i]=a.z[perm[i]]; a2.type[i]=1; a2.mass[i]=28.0855; }
  core::zero_forces(a2);
  const double pe2 = pot::tersoff_run_fixed(a2, geom, p).pe;
  EXPECT_EQ(pe1, pe2) << "PE not bitwise order-invariant";
  for (int i = 0; i < a.n; ++i) {
    ASSERT_EQ(a1.fx[perm[i]], a2.fx[i]) << "force not bitwise order-invariant at " << i;
    ASSERT_EQ(a1.fy[perm[i]], a2.fy[i]); ASSERT_EQ(a1.fz[perm[i]], a2.fz[i]);
  }
}

// G-BIJ-LIMITS — the 5-branch small-ζ guard: b(0)=1 (isolated atom), b large-ζ ≈ 1/√(βζ),
// all branches finite + continuous (no NaN — the naïve closed form NaNs at ζ=0).
TEST(Tersoff, BondOrderLimitsAreSafe) {
  pot::TersoffParams p;
  EXPECT_EQ(pot::ters_bij(0.0, p), 1.0) << "b(ζ=0) must be 1 (no neighbours)";
  const double big = 1e6;
  EXPECT_NEAR(pot::ters_bij(big, p), 1.0 / std::sqrt(p.beta * big), 1e-6) << "large-ζ asymptote";
  double prev = 1.0;
  for (double z = 1e-12; z < 1e8; z *= 3.0) {
    const double b = pot::ters_bij(z, p), bd = pot::ters_bij_d(z, p);
    ASSERT_TRUE(std::isfinite(b) && std::isfinite(bd)) << "NaN/inf at ζ=" << z;
    ASSERT_LE(b, prev + 1e-9) << "b not monotonically non-increasing at ζ=" << z;
    prev = b;
  }
}

// G-ORACLE/MB2 (enumeration only) — int64 path ≡ FP64 oracle within quantum; the POISON
// (dropped bond/triplet class) DIVERGES; counts agree. (Does NOT witness the algebra — shared.)
TEST(Tersoff, FixedMatchesOracleAndPoisonHasTeeth) {
  pot::TersoffParams p;
  core::Box box;
  auto a = make_diamond_si(2, 2, 2, 5.431, 0.20, box);
  box.periodic = {true, true, true};
  const core::PairGeom geom(box, p.rcut());

  core::AtomSoA<double> fixed = a, oracle = a, poison = a;
  core::zero_forces(fixed); core::zero_forces(oracle); core::zero_forces(poison);
  const auto rf = pot::tersoff_run_fixed(fixed, geom, p);
  const auto ro = pot::tersoff_direct_fp64(oracle, geom, p, true, 0);
  const auto rp = pot::tersoff_direct_fp64(poison, geom, p, true, 1);

  EXPECT_LT(max_force_diff(fixed, oracle), 1e-9) << "int64 ≠ FP64 oracle";
  EXPECT_NEAR(rf.pe, ro.pe, 1e-7);
  EXPECT_GT(ro.n_triplets, 0);
  EXPECT_LT(rp.n_triplets, ro.n_triplets) << "poison dropped no triplets";
  EXPECT_GT(max_force_diff(oracle, poison), 1e-3) << "POISON did not diverge — G-ORACLE blind";
}

// G-VACUITY — the bond order is genuinely active (b_ij < 1: the angular term reduces it).
TEST(Tersoff, BondOrderIsActive) {
  pot::TersoffParams p;
  core::Box box;
  auto a = make_diamond_si(2, 2, 2, 5.431, 0.20, box);
  box.periodic = {true, true, true};
  const core::PairGeom geom(box, p.rcut());
  core::AtomSoA<double> g = a; core::zero_forces(g);
  pot::tersoff_direct_fp64(g, geom, p, true);
  EXPECT_GT(max_abs_force(g), 0.1) << "forces too quiet";
  // a pure-pair (b≡1) reference would differ materially — confirm the angular term bites by
  // checking the attractive force is reduced vs a no-bond-order (β=0 ⇒ b=1) variant.
  pot::TersoffParams nob = p; nob.beta = 0.0;  // b_ij = 1 for all bonds
  core::AtomSoA<double> g2 = a; core::zero_forces(g2);
  pot::tersoff_direct_fp64(g2, geom, nob, true);
  EXPECT_GT(max_force_diff(g, g2), 0.1) << "bond-order (b<1) not exercised";
}

// G-MOMENTUM ⭐ — the int64 non-symmetric-triplet momentum floor (the SW T4 finding replicates
// for Tersoff): the FP64 oracle conserves Σf to round-off; the int64 zetaterm quantizes
// f_i,f_j,f_k INDEPENDENTLY (rint not additive) ⇒ Σf≠0 at the quantum (random-walks ∝√steps
// in NVE — Te4 inherits this). A force-asymmetry bug would break BOTH; the gap rules it out.
TEST(Tersoff, MomentumFloorIsInt64Quantization) {
  pot::TersoffParams p;
  core::Box box;
  auto a = make_diamond_si(2, 2, 2, 5.431, 0.20, box);
  box.periodic = {true, true, true};
  const core::PairGeom geom(box, p.rcut());
  auto sf = [](const core::AtomSoA<double>& s) {
    double x = 0, y = 0, z = 0;
    for (int i = 0; i < s.n; ++i) { x += s.fx[i]; y += s.fy[i]; z += s.fz[i]; }
    return std::sqrt(x * x + y * y + z * z);
  };
  core::AtomSoA<double> o = a, q = a; core::zero_forces(o); core::zero_forces(q);
  pot::tersoff_direct_fp64(o, geom, p, true);
  pot::tersoff_run_fixed(q, geom, p);
  EXPECT_LT(sf(o), 1e-12) << "FP64 oracle Σf not round-off — a force-asymmetry bug";
  EXPECT_GT(sf(q), 0.0) << "int64 Σf exactly zero — the finding would be vacuous";
  EXPECT_LT(sf(q), 1e-9) << "int64 Σf far above the quantum — a bug";
}
