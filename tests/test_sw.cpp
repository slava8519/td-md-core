// M6 / SW-ladder PR-1 (T1): Stillinger–Weber — the first ANGULAR potential.
// Pre-registered acceptance (angular-potential design wf_5294df4d-65c §3g):
//   G-FD      analytic force ≡ −dU/dx central-difference (validates the φ₃ f_k write)
//   G-TI/MB1  per-triplet |f_i+f_j+f_k| < 1e-12 (the three-atom partition)
//   G-B1      sw_run_fixed int64 forces+PE bitwise-invariant under atom relabeling
//   G-ORACLE  sw_run_fixed (gather) ≡ sw_direct_fp64 (scatter) within quantum; A2 count
//             equal & >0; POISON (dropped triplet class) DIVERGES — the oracle has teeth
//   G-VACUITY the diamond fixture actually exercises φ₃ (angular force non-negligible)
// No LAMMPS — the FP64 all-triplets oracle is the reference; the Si golden is T7.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/gen/diamond_si.hpp"  // shared fixture (also used by tools/sw_lammps_check)
#include "tdmd/potentials/sw.hpp"

namespace core = tdmd::core;
namespace pot = tdmd::potentials;
using tdmd::gen::make_diamond_si;

namespace {

double max_abs_force(const core::AtomSoA<double>& a) {
  double m = 0.0;
  for (int i = 0; i < a.n; ++i) {
    m = std::max(m, std::fabs(a.fx[i]));
    m = std::max(m, std::fabs(a.fy[i]));
    m = std::max(m, std::fabs(a.fz[i]));
  }
  return m;
}

double max_force_diff(const core::AtomSoA<double>& a, const core::AtomSoA<double>& b) {
  double m = 0.0;
  for (int i = 0; i < a.n; ++i) {
    m = std::max(m, std::fabs(a.fx[i] - b.fx[i]));
    m = std::max(m, std::fabs(a.fy[i] - b.fy[i]));
    m = std::max(m, std::fabs(a.fz[i] - b.fz[i]));
  }
  return m;
}

double energy_of(core::AtomSoA<double> a, const core::Box& box, const pot::SwParams& sp) {
  return pot::sw_direct_fp64(a, box, sp, /*with_forces=*/false).pe;  // by value (FD probe)
}

}  // namespace

// G-FD: the analytic force (FP64 oracle, φ₂+φ₃) equals −dU/dx by central difference,
// per SIGNED component — a flipped f_k (still summing to 0 per triplet) is caught here.
TEST(StillingerWeber, ForceMatchesFiniteDifference) {
  pot::SwParams sp;
  core::Box box;
  auto a = make_diamond_si(2, 2, 2, 5.431, 0.20, box);
  core::AtomSoA<double> g = a;
  core::zero_forces(g);
  pot::sw_direct_fp64(g, box, sp, /*with_forces=*/true);

  const double h = 1e-5;
  double maxerr = 0.0;
  for (int i = 0; i < std::min(a.n, 12); ++i) {  // a representative subset
    double* coord[3] = {&a.x[i], &a.y[i], &a.z[i]};
    const double fan[3] = {g.fx[i], g.fy[i], g.fz[i]};
    for (int d = 0; d < 3; ++d) {
      const double x0 = *coord[d];
      *coord[d] = x0 + h; const double ep = energy_of(a, box, sp);
      *coord[d] = x0 - h; const double em = energy_of(a, box, sp);
      *coord[d] = x0;
      const double ffd = -(ep - em) / (2 * h);
      maxerr = std::max(maxerr, std::fabs(fan[d] - ffd));
    }
  }
  EXPECT_LT(maxerr, 1e-5) << "analytic SW force disagrees with central-difference";
}

// G-TI / MB1: every triplet's three-atom force sums to zero (translational invariance);
// guards the third-atom-k partition (f_i = −(f_j+f_k)).
TEST(StillingerWeber, TripletForcesSumToZero) {
  pot::SwParams sp;
  core::Box box;
  auto a = make_diamond_si(2, 2, 2, 5.431, 0.20, box);
  const core::PairGeom geom(box, sp.rcut());
  const auto nbr = pot::sw_build_nbr(a, geom);
  double maxsum = 0.0;
  long checked = 0;
  for (int i = 0; i < a.n; ++i)
    for (std::size_t x = 0; x < nbr[i].size(); ++x)
      for (std::size_t y = x + 1; y < nbr[i].size(); ++y) {
        pot::SwVec3 fi, fj, fk;
        pot::sw_triplet(sp, nbr[i][x].dx, nbr[i][x].dy, nbr[i][x].dz, nbr[i][x].r,
                        nbr[i][y].dx, nbr[i][y].dy, nbr[i][y].dz, nbr[i][y].r, fi, fj, fk);
        maxsum = std::max({maxsum, std::fabs(fi.x + fj.x + fk.x),
                           std::fabs(fi.y + fj.y + fk.y), std::fabs(fi.z + fj.z + fk.z)});
        ++checked;
      }
  EXPECT_GT(checked, 0) << "no triplets — fixture vacuous";
  EXPECT_LT(maxsum, 1e-12) << "triplet force does not sum to zero (bad k-partition)";
}

// G-B1: the int64 transpose-replay is order-free. Relabel the atoms by a random
// permutation: each atom's force (and the global PE) must be BITWISE identical (a
// different index order ⇒ a different add() sequence ⇒ must give the same int64 sum).
// TEETH: this runs an ENSEMBLE over a TIE-PRONE fixture (perturb=0.45 — near-tetrahedral
// triplets put f_j/f_k ~1 ULP apart, straddling the Q24.40 tie). The wing-canonical
// sw_triplet (gg 2-operand + mirrored radial order) is what keeps this green; without it
// ~1-2/200 relabelings flip an int64. A single quiet seed (the perfect-ish lattice) is
// tie-free by luck and would NOT witness a regression — hence the ensemble.
TEST(StillingerWeber, FixedPointBitwiseUnderRelabeling) {
  pot::SwParams sp;
  for (double pert : {0.20, 0.45}) {
    for (unsigned fseed : {1u, 7u, 12345u}) {
      core::Box box;
      auto a = make_diamond_si(2, 2, 2, 5.431, pert, box, fseed);
      core::AtomSoA<double> a1 = a;
      core::zero_forces(a1);
      const double pe1 = pot::sw_run_fixed(a1, box, sp).pe;

      for (unsigned pseed : {3u, 17u, 101u, 999u}) {  // many relabelings → catch any tie
        std::vector<int> perm(a.n);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), std::mt19937(pseed));
        core::AtomSoA<double> a2;
        a2.resize(a.n);
        for (int i = 0; i < a.n; ++i) {
          a2.x[i] = a.x[perm[i]]; a2.y[i] = a.y[perm[i]]; a2.z[i] = a.z[perm[i]];
          a2.type[i] = a.type[perm[i]]; a2.mass[i] = a.mass[perm[i]];
        }
        core::zero_forces(a2);
        const double pe2 = pot::sw_run_fixed(a2, box, sp).pe;

        EXPECT_EQ(pe1, pe2) << "PE not bitwise order-invariant (pert=" << pert << ")";
        for (int i = 0; i < a.n; ++i) {
          ASSERT_EQ(a1.fx[perm[i]], a2.fx[i])
              << "force not bitwise order-invariant — wing/center canonicalization broke"
              << " (pert=" << pert << " fseed=" << fseed << " pseed=" << pseed << " atom=" << i << ")";
          ASSERT_EQ(a1.fy[perm[i]], a2.fy[i]);
          ASSERT_EQ(a1.fz[perm[i]], a2.fz[i]);
        }
      }
    }
  }
}

// G-B1 teeth (deterministic): the transpose-replay's load-bearing invariant is that
// sw_triplet is WING-CANONICAL — f_j(i;o,m) == f_k(i;m,o) BITWISE (not merely in ℝ).
// This tests it directly over 100k random in-cutoff triplets; the un-canonicalized form
// (separate j/k expression trees) fails by ~1 ULP on ~1/3 of triplets immediately — no
// reliance on a Q24.40 tie, so it is a reliable regression guard for the canonicalization.
TEST(StillingerWeber, TripletWingSymmetryIsBitwise) {
  pot::SwParams sp;
  std::mt19937 rng(2024);
  std::uniform_real_distribution<double> dir(-1.0, 1.0), mag(2.0, 3.6);  // < rcut=3.771
  auto bond = [&](double& bx, double& by, double& bz, double& r) {
    double x = dir(rng), y = dir(rng), z = dir(rng);
    double n = std::sqrt(x * x + y * y + z * z);
    if (n < 1e-6) { x = 1; y = z = 0; n = 1; }
    r = mag(rng);
    bx = x / n * r; by = y / n * r; bz = z / n * r;
  };
  long checked = 0;
  for (int t = 0; t < 100000; ++t) {
    double ox, oy, oz, ro, mx, my, mz, rm;
    bond(ox, oy, oz, ro); bond(mx, my, mz, rm);
    pot::SwVec3 fi1, fj1, fk1, fi2, fj2, fk2;
    pot::sw_triplet(sp, ox, oy, oz, ro, mx, my, mz, rm, fi1, fj1, fk1);  // o in j-slot
    pot::sw_triplet(sp, mx, my, mz, rm, ox, oy, oz, ro, fi2, fj2, fk2);  // o in k-slot
    ASSERT_EQ(fj1.x, fk2.x) << "wing symmetry broken bitwise at triplet " << t;
    ASSERT_EQ(fj1.y, fk2.y); ASSERT_EQ(fj1.z, fk2.z);
    ASSERT_EQ(fi1.x, fi2.x) << "center not invariant under wing swap at " << t;  // f_i = −(f_j+f_k), + commutes
    ASSERT_EQ(fi1.y, fi2.y); ASSERT_EQ(fi1.z, fi2.z);
    ++checked;
  }
  EXPECT_EQ(checked, 100000);
}

// G-ORACLE (MB2): the int64 gather-per-owner path ≡ the FP64 scatter-per-center oracle
// within the quantization bound, and the triplet COUNTS match and are >0 (A2). The
// enumerations are genuinely different, so a dropped/mis-resident triplet would diverge.
// POISON teeth: a deliberately narrowed oracle (drop the straddling-wing class) must
// DIVERGE from the correct path — proving the comparison is not vacuous.
TEST(StillingerWeber, FixedMatchesOracleAndPoisonHasTeeth) {
  pot::SwParams sp;
  core::Box box;
  auto a = make_diamond_si(2, 2, 2, 5.431, 0.20, box);

  core::AtomSoA<double> fixed = a, oracle = a, poison = a;
  core::zero_forces(fixed); core::zero_forces(oracle); core::zero_forces(poison);
  const auto rf = pot::sw_run_fixed(fixed, box, sp);
  const auto ro = pot::sw_direct_fp64(oracle, box, sp, true, /*drop_self_class=*/0);
  const auto rp = pot::sw_direct_fp64(poison, box, sp, true, /*drop_self_class=*/1);

  EXPECT_EQ(rf.n_triplets, ro.n_triplets) << "A2: triplet multiset size differs";
  EXPECT_GT(ro.n_triplets, 0) << "no triplets enumerated";
  EXPECT_LT(max_force_diff(fixed, oracle), 1e-9) << "int64 gather ≠ FP64 scatter oracle";
  EXPECT_NEAR(rf.pe, ro.pe, 1e-7) << "PE mismatch fixed vs oracle";

  EXPECT_LT(rp.n_triplets, ro.n_triplets) << "poison did not drop any triplet";
  EXPECT_GT(max_force_diff(fixed, poison), 1e-3)
      << "POISON did NOT diverge — the oracle comparison is blind to a dropped triplet!";
}

// G-VACUITY: the fixture exercises the ANGULAR term — triplets exist, forces are real,
// and the φ₃ contribution is non-negligible (total force differs materially from a
// φ₂-only computation). A perfect (or FCC) lattice would make this fail.
TEST(StillingerWeber, DiamondFixtureExercisesAngularTerm) {
  pot::SwParams sp;
  core::Box box;
  auto a = make_diamond_si(2, 2, 2, 5.431, 0.20, box);

  core::AtomSoA<double> full = a;
  core::zero_forces(full);
  const auto rf = pot::sw_direct_fp64(full, box, sp, true);
  EXPECT_GT(rf.n_triplets, 0);
  EXPECT_GT(max_abs_force(full), 0.05) << "fixture too quiet";

  // φ₂-only forces (the symmetric pair term alone) — the difference is the angular force.
  core::AtomSoA<double> pair_only = a;
  core::zero_forces(pair_only);
  const core::PairGeom geom(box, sp.rcut());
  const auto nbr = pot::sw_build_nbr(a, geom);
  for (int i = 0; i < a.n; ++i)
    for (const auto& e : nbr[i]) {
      if (e.j < i) continue;
      double phi, dphi;
      pot::sw_phi2(sp, e.r, phi, dphi);
      const double f_over_r = -dphi / e.r;
      pair_only.fx[i] += f_over_r * e.dx; pair_only.fy[i] += f_over_r * e.dy; pair_only.fz[i] += f_over_r * e.dz;
      pair_only.fx[e.j] -= f_over_r * e.dx; pair_only.fy[e.j] -= f_over_r * e.dy; pair_only.fz[e.j] -= f_over_r * e.dz;
    }
  EXPECT_GT(max_force_diff(full, pair_only), 0.01)
      << "angular (φ₃) force is negligible — the fixture is vacuous for the 3-body term";
}
