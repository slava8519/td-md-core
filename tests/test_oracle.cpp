// M4-N Gate-02a — Physical Oracle (CPU layer), NON-NEGOTIABLE correctness gate
// (design §F.3, adversarially designed 2026-06-17).
//
// WHY IT EXISTS: a too-coarse neighbour-list rebuild trigger lets a pair drift
// from outside rcut+skin to inside rcut BETWEEN rebuilds undetected → its force
// term is silently dropped. 1-vs-z identity and run-to-run determinism BOTH stay
// GREEN (every decomposition drops the SAME pair at the SAME step). Only a
// comparison to an independent ground truth catches it — this gate.
//
// CONSTRUCTION: two FP64 force paths over the SAME unsorted AtomSoA, identical
// original-index ascending order, one shared pair body:
//   REFERENCE  = direct_pair_loop (the blessed O(N²) oracle; cannot miss).
//   PERSISTENT = the same inner body, j iterated over a persistent neighbour list
//                nbr[i] (ascending original indices, radius rcut+skin), rebuilt on
//                a trigger. Superset (P_t ⊆ nbr) ⇒ identical terms in identical
//                order ⇒ BITWISE-equal forces. A dropped in-cutoff pair injects a
//                nonzero missing term (Shift truncation ⇒ force discontinuous at
//                rcut, |F|~0.16 eV/Å ≫ ULP) ⇒ a flipped bit ⇒ caught.
//
// SCOPE (Layer A): proves the superset⇒bitwise lemma and that a too-coarse trigger
// is caught, on a single-node CPU narrow phase. It does NOT certify the conveyor's
// 2·R_buf skin budget, the n−1 light-cone lag-ramp, or the C_buf gap — those live
// in conveyor_gpu.cuh / zone_verlet.cuh and need Gate-02b (GPU, gpu_gate, cells-vs-
// verlet kernels on RAW int64 accumulators, K>1), REQUIRED before any
// PersistentVerlet backend ships as default; NOT implemented here.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "tdmd/core/integrator.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/thermal.hpp"
#include "tdmd/potentials/cutoff.hpp"
#include "tdmd/potentials/pair_driver.hpp"
#include "tdmd/potentials/pair_morse.hpp"

using namespace tdmd;

namespace {
constexpr double kAlMass = 26.9815, kA0 = 4.05, kRcut = 4.0, kDt = 0.001;
constexpr double kD = 0.29614, kAlpha = 1.11892, kR0 = 3.29692;

core::AtomSoA<double> make_fcc(int nc, core::Box& box) {
  core::AtomSoA<double> a;
  a.resize(4 * nc * nc * nc);
  box.lo = {0, 0, 0}; box.hi = {nc * kA0, nc * kA0, nc * kA0};
  box.periodic = {true, true, true};
  const double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  int k = 0;
  for (int i = 0; i < nc; ++i)
    for (int j = 0; j < nc; ++j)
      for (int l = 0; l < nc; ++l)
        for (auto& b : basis) {
          a.x[k] = (i + b[0]) * kA0; a.y[k] = (j + b[1]) * kA0; a.z[k] = (l + b[2]) * kA0;
          a.mass[k] = kAlMass; a.type[k] = 1; ++k;
        }
  return a;
}

// Persistent neighbour list: nbr[i] = ascending original indices j (j≠i) within
// rlist (= rcut+skin) under the SAME minimum-image fold as direct_pair_loop.
std::vector<std::vector<int>> build_nbr(const core::AtomSoA<double>& a,
                                        const core::Box& box, double rlist) {
  const double rl2 = rlist * rlist;
  const double L[3] = {box.len(0), box.len(1), box.len(2)};
  std::vector<std::vector<int>> nbr(a.n);
  for (int i = 0; i < a.n; ++i)
    for (int j = 0; j < a.n; ++j) {       // ascending j ⇒ nbr[i] sorted
      if (j == i) continue;
      double dx = a.x[i] - a.x[j], dy = a.y[i] - a.y[j], dz = a.z[i] - a.z[j];
      if (box.periodic[0]) dx -= L[0] * std::round(dx / L[0]);
      if (box.periodic[1]) dy -= L[1] * std::round(dy / L[1]);
      if (box.periodic[2]) dz -= L[2] * std::round(dz / L[2]);
      if (dx * dx + dy * dy + dz * dz < rl2) nbr[i].push_back(j);
    }
  return nbr;
}

// PERSISTENT force path — op-for-op identical to potentials::direct_pair_loop
// (pair_driver.hpp:39) EXCEPT j iterates nbr[i] instead of all atoms. Same fold,
// same r²<rc² predicate, same pair callback, same fxi locals→store, same order.
// Any divergence from that body is caught by the static-superset lock test below.
template <typename PairFn>
potentials::PairAccum persistent_force(core::AtomSoA<double>& a, const core::Box& box,
                                       double rcut, const std::vector<std::vector<int>>& nbr,
                                       PairFn&& pair) {
  const double rc2 = rcut * rcut;
  const double L[3] = {box.len(0), box.len(1), box.len(2)};
  potentials::PairAccum acc;
  for (int i = 0; i < a.n; ++i) {
    double fxi = 0.0, fyi = 0.0, fzi = 0.0;
    for (int j : nbr[i]) {
      if (j == i) continue;
      double dx = a.x[i] - a.x[j], dy = a.y[i] - a.y[j], dz = a.z[i] - a.z[j];
      if (box.periodic[0]) dx -= L[0] * std::round(dx / L[0]);
      if (box.periodic[1]) dy -= L[1] * std::round(dy / L[1]);
      if (box.periodic[2]) dz -= L[2] * std::round(dz / L[2]);
      const double r2 = dx * dx + dy * dy + dz * dz;
      acc.min_r2 = std::min(acc.min_r2, r2);
      if (r2 >= rc2 || r2 < 1e-18) continue;
      double u, f_over_r;
      pair(std::sqrt(r2), u, f_over_r);
      acc.pe += 0.5 * u;
      acc.virial += 0.5 * f_over_r * r2;
      fxi += f_over_r * dx; fyi += f_over_r * dy; fzi += f_over_r * dz;
    }
    a.fx[i] += fxi; a.fy[i] += fyi; a.fz[i] += fzi;
  }
  return acc;
}

// A2 backstop: count in-cutoff pairs present in REFERENCE but absent from nbr.
// Force-magnitude-independent (load-bearing if anyone swaps the truncation).
long count_missed(const core::AtomSoA<double>& a, const core::Box& box, double rcut,
                  const std::vector<std::vector<int>>& nbr) {
  const double rc2 = rcut * rcut;
  const double L[3] = {box.len(0), box.len(1), box.len(2)};
  long missed = 0;
  for (int i = 0; i < a.n; ++i)
    for (int j = 0; j < a.n; ++j) {
      if (j == i) continue;
      double dx = a.x[i] - a.x[j], dy = a.y[i] - a.y[j], dz = a.z[i] - a.z[j];
      if (box.periodic[0]) dx -= L[0] * std::round(dx / L[0]);
      if (box.periodic[1]) dy -= L[1] * std::round(dy / L[1]);
      if (box.periodic[2]) dz -= L[2] * std::round(dz / L[2]);
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 >= rc2 || r2 < 1e-18) continue;
      if (!std::binary_search(nbr[i].begin(), nbr[i].end(), j)) ++missed;
    }
  return missed;
}

enum class Trig { HalfSkin, FixedCadence };

struct OracleResult {
  bool bitwise_all = true;   // A1 held every step
  long max_missed = 0;       // A2 worst-case dropped in-cutoff pairs
  long first_div_step = 0;   // first step A1 failed (0 = never)
  long rebuilds = 0;
  double min_r2 = 1e300;
  double max_abs_df = 0.0;   // worst |F_persistent − F_reference| (teeth magnitude)
};

// Trajectory integrates with REFERENCE forces (ground truth) so it never drifts
// due to the persistent path's bug — every step is a fair force-array compare on
// the SAME correct trajectory. The trajectory is independent of the trigger.
OracleResult run(double skin, Trig trig, long cadence, double T, long steps) {
  core::Box box;
  auto a = make_fcc(4, box);  // 256 atoms
  core::thermal::maxwell_init(a, T, /*seed=*/99ull);
  const double rlist = kRcut + skin;

  const potentials::MorseParams<double> prm{kD, kAlpha, kR0};
  auto raw = [&](double r, double& u, double& f) { potentials::pair_morse<double>(r, prm, u, f); };
  const auto cs = potentials::CutoffScheme::make(potentials::Truncation::Shift, kRcut, raw);
  auto pairfn = [&](double r, double& u, double& f) { raw(r, u, f); cs.apply(r, u, f); };

  auto nbr = build_nbr(a, box, rlist);
  std::vector<double> rx(a.x), ry(a.y), rz(a.z);  // ref positions for half-skin
  core::zero_forces(a);
  potentials::direct_pair_loop(a, box, kRcut, pairfn);  // prime a.f at x0

  OracleResult R;
  std::vector<double> pfx, pfy, pfz;
  for (long s = 1; s <= steps; ++s) {
    core::VelocityVerlet<double>::first_half(a, kDt);  // advances x with a.f=a(t)
    bool rebuild = false;
    if (trig == Trig::HalfSkin) {
      double md2 = 0.0;
      for (int i = 0; i < a.n; ++i) {
        const double dx = a.x[i] - rx[i], dy = a.y[i] - ry[i], dz = a.z[i] - rz[i];
        md2 = std::max(md2, dx * dx + dy * dy + dz * dz);
      }
      if (2.0 * std::sqrt(md2) >= skin) rebuild = true;
    } else {
      if (s % cadence == 0) rebuild = true;
    }
    if (rebuild) { nbr = build_nbr(a, box, rlist); rx = a.x; ry = a.y; rz = a.z; ++R.rebuilds; }

    core::zero_forces(a);
    persistent_force(a, box, kRcut, nbr, pairfn);     // PERSISTENT → a.f
    pfx = a.fx; pfy = a.fy; pfz = a.fz;
    core::zero_forces(a);
    const auto accR = potentials::direct_pair_loop(a, box, kRcut, pairfn);  // REFERENCE → a.f
    R.min_r2 = std::min(R.min_r2, accR.min_r2);

    R.max_missed = std::max(R.max_missed, count_missed(a, box, kRcut, nbr));
    const std::size_t nb = std::size_t(a.n) * sizeof(double);
    const bool eq = std::memcmp(pfx.data(), a.fx.data(), nb) == 0 &&
                    std::memcmp(pfy.data(), a.fy.data(), nb) == 0 &&
                    std::memcmp(pfz.data(), a.fz.data(), nb) == 0;
    if (!eq) {
      double mdf = 0.0;
      for (int i = 0; i < a.n; ++i) {
        mdf = std::max(mdf, std::fabs(pfx[i] - a.fx[i]));
        mdf = std::max(mdf, std::fabs(pfy[i] - a.fy[i]));
        mdf = std::max(mdf, std::fabs(pfz[i] - a.fz[i]));
      }
      R.max_abs_df = std::max(R.max_abs_df, mdf);
      if (R.bitwise_all) { R.bitwise_all = false; R.first_div_step = s; }
    }

    core::VelocityVerlet<double>::second_half(a, kDt);  // a.f = reference = a(t+dt)
  }
  return R;
}
}  // namespace

// (i) POSITIVE / sufficiency: a correct half-skin trigger keeps the superset by
// construction ⇒ forces bitwise-equal EVERY step over a hot 5000-step trajectory.
// The list is genuinely exercised (rebuilds>0) and no overlap-HALT red-herring.
TEST(PhysicalOracle, SupersetGivesBitwiseForces) {
  const OracleResult r = run(/*skin=*/1.0, Trig::HalfSkin, /*cadence=*/0, /*T=*/2500.0, 5000);
  EXPECT_TRUE(r.bitwise_all) << "first divergence at step " << r.first_div_step;
  EXPECT_EQ(r.max_missed, 0);
  EXPECT_GT(r.rebuilds, 0) << "list never rebuilt — gate would be vacuous";
  // non-vacuity ceiling: the list must be REUSED, not rebuilt ~every step, else
  // the superset⇒bitwise lemma is never exercised over a stale window (~23 here).
  EXPECT_LT(r.rebuilds, 5000 / 2) << "list rebuilt too often — gate near-vacuous";
  EXPECT_GT(r.min_r2, 1.0) << "atoms overlapped — lower T to avoid B10 red-herring";
}

// (ii) TEETH / non-vacuity (the PRIMARY proof the gate works): a too-coarse trigger
// (rebuild only every 50 steps, far longer than K_eff~15-20 in this hot regime)
// lets a pair cross rcut+skin→rcut undetected. The gate MUST trip on BOTH channels.
TEST(PhysicalOracle, TooCoarseTriggerIsCaught) {
  const OracleResult r = run(/*skin=*/0.2, Trig::FixedCadence, /*cadence=*/25, /*T=*/2500.0, 500);
  EXPECT_GT(r.max_missed, 0) << "A2: no pair was dropped — regime not hot enough";
  EXPECT_FALSE(r.bitwise_all) << "A1: a dropped pair did not flip a force bit";
  EXPECT_GE(r.first_div_step, 1);
  // the divergence must be PHYSICALLY large (Shift cutoff-edge force ~0.16 eV/Å),
  // not a marginal ULP flip — certifies the gate caught a real dropped pair.
  EXPECT_GT(r.max_abs_df, 1e-6);
}

// (iii) STATIC lock: with a full-superset list the PERSISTENT body must be bitwise-
// identical to direct_pair_loop — pins that both share the strict r²<rc² predicate
// and the op-for-op inner arithmetic (guards against the persistent body drifting).
TEST(PhysicalOracle, PersistentEqualsReferenceOnFullSuperset) {
  core::Box box;
  auto a = make_fcc(4, box);
  core::thermal::maxwell_init(a, 1500.0, /*seed=*/7ull);
  const potentials::MorseParams<double> prm{kD, kAlpha, kR0};
  auto raw = [&](double r, double& u, double& f) { potentials::pair_morse<double>(r, prm, u, f); };
  const auto cs = potentials::CutoffScheme::make(potentials::Truncation::Shift, kRcut, raw);
  auto pairfn = [&](double r, double& u, double& f) { raw(r, u, f); cs.apply(r, u, f); };
  // a few steps off the lattice so the config is generic
  core::zero_forces(a); potentials::direct_pair_loop(a, box, kRcut, pairfn);
  for (int s = 0; s < 5; ++s) {
    core::VelocityVerlet<double>::first_half(a, kDt);
    core::zero_forces(a); potentials::direct_pair_loop(a, box, kRcut, pairfn);
    core::VelocityVerlet<double>::second_half(a, kDt);
  }
  const auto nbr = build_nbr(a, box, kRcut + 3.0);  // full superset (< L/2 = 8.1)
  // N2: count_missed's binary_search relies on nbr[i] being sorted — build_nbr is
  // the sole producer and emits ascending j; pin that invariant here so a future
  // refactor cannot silently break A2.
  for (const auto& l : nbr) ASSERT_TRUE(std::is_sorted(l.begin(), l.end()));
  core::zero_forces(a); persistent_force(a, box, kRcut, nbr, pairfn);
  std::vector<double> px = a.fx, py = a.fy, pz = a.fz;
  core::zero_forces(a); potentials::direct_pair_loop(a, box, kRcut, pairfn);
  const std::size_t nb = std::size_t(a.n) * sizeof(double);
  EXPECT_EQ(std::memcmp(px.data(), a.fx.data(), nb), 0);
  EXPECT_EQ(std::memcmp(py.data(), a.fy.data(), nb), 0);
  EXPECT_EQ(std::memcmp(pz.data(), a.fz.data(), nb), 0);
}
