// M3.5 (first PR): fixed-point accumulation (B1) + serial zone decomposition
// with the w-mechanism partial-force assembly.
// Acceptance slice (Roadmap M3.5): "сумма w-вкладов ≡ монолитный расчёт" —
// BITWISE here (fixed-point accumulation makes the assembly order-free,
// stronger than the ≤1e-12 wording) plus a quantization-bound check against
// the FP64 oracle; "однократность пары через границу зон" (INV-8) — every
// min-image pair inside r_cut is evaluated exactly once across zone passes.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/potentials/cutoff.hpp"
#include "tdmd/potentials/morse.hpp"

using namespace tdmd;

static std::string project_root() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT);
#else
  return ".";
#endif
}

namespace {

// The drivers' Morse pair callback (functor + truncation policy), dissertation
// Al parameters — zones.hpp is potential-agnostic, the test supplies the math.
struct MorsePair {
  potentials::MorseParams<double> prm{0.29614, 1.11892, 3.29692};
  double rcut = 4.0;
  potentials::CutoffScheme cs = potentials::CutoffScheme::make(
      potentials::Truncation::Shift, rcut,
      [&](double r, double& u, double& f) { potentials::pair_morse(r, prm, u, f); });

  void operator()(double r, double& u, double& f_over_r) const {
    potentials::pair_morse(r, prm, u, f_over_r);
    cs.apply(r, u, f_over_r);
  }
};

core::AtomSoA<double> load144(core::Box& box) {
  core::AtomSoA<double> atoms;
  box.periodic = {true, true, true};
  EXPECT_TRUE(io::read_lammps_data(
      project_root() + "/reference_data/al_fcc_144.data", atoms, box));
  return atoms;  // box z = 16.2 Å -> 4 zones of 4.05 >= rcut 4.0
}

// All unordered min-image pairs within rcut — the INV-8 oracle.
std::set<std::pair<int, int>> brute_pairs(const core::AtomSoA<double>& a,
                                          const core::Box& box, double rcut) {
  std::set<std::pair<int, int>> out;
  const double L[3] = {box.len(0), box.len(1), box.len(2)};
  for (int i = 0; i < a.n; ++i)
    for (int j = i + 1; j < a.n; ++j) {
      double dx = a.x[i] - a.x[j], dy = a.y[i] - a.y[j], dz = a.z[i] - a.z[j];
      if (box.periodic[0]) dx -= L[0] * std::round(dx / L[0]);
      if (box.periodic[1]) dy -= L[1] * std::round(dy / L[1]);
      if (box.periodic[2]) dz -= L[2] * std::round(dz / L[2]);
      if (dx * dx + dy * dy + dz * dz < rcut * rcut) out.insert({i, j});
    }
  return out;
}

}  // namespace

// --- B1: fixed-point accumulator ---

TEST(FixedAccum, OrderIndependentBitwise) {
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> uni(-50.0, 50.0);
  std::vector<double> xs(10000);
  for (auto& x : xs) x = uni(rng);

  core::fixed::ForceAccum fwd, shuf;
  for (double x : xs) fwd.add(x);
  std::shuffle(xs.begin(), xs.end(), rng);
  for (double x : xs) shuf.add(x);
  EXPECT_EQ(fwd.raw, shuf.raw);  // integer addition is associative — bitwise

  // FP64 naive sum agrees within the quantization bound: N halves of a quantum
  double naive = 0.0;
  for (double x : xs) naive += x;
  EXPECT_NEAR(fwd.value(), naive, xs.size() * 0.5 / core::fixed::ForceAccum::kScale);
}

TEST(FixedAccum, QuantumAndRange) {
  core::fixed::ForceAccum f;
  f.add(1.0);
  EXPECT_DOUBLE_EQ(f.value(), 1.0);  // 1.0 is exactly representable in Q24.40
  f.add(-1.0);
  EXPECT_EQ(f.raw, 0);
  // ties round to even (rint, FE_TONEAREST) — the CUDA cvt.rni convention
  f.reset();
  f.add(0.5 / core::fixed::ForceAccum::kScale);   // exactly half a quantum
  EXPECT_EQ(f.raw, 0);                            // ties-to-even: 0.5 -> 0
  f.reset();
  f.add(1.5 / core::fixed::ForceAccum::kScale);   // 1.5 quanta
  EXPECT_EQ(f.raw, 2);                            // ties-to-even: 1.5 -> 2
}

// Review M3.5: contributions beyond the int64 range (r^-13-class potentials
// at sub-Å distances) and non-finite forces must THROW, never hit the UB of
// an unrepresentable double->int64 conversion.
TEST(FixedAccum, OutOfRangeContributionThrows) {
  core::fixed::ForceAccum f;
  EXPECT_THROW(f.add(9.3e18 / core::fixed::ForceAccum::kScale),
               std::overflow_error);
  EXPECT_THROW(f.add(std::numeric_limits<double>::infinity()),
               std::overflow_error);
  EXPECT_THROW(f.add(std::nan("")), std::overflow_error);
  EXPECT_EQ(f.raw, 0);  // failed adds must not corrupt the accumulator
  f.add(9.1e18 / core::fixed::ForceAccum::kScale);  // just inside the range
  EXPECT_NE(f.raw, 0);
}

// --- zone decomposition ---

TEST(Zones, PartitionAndPreconditions) {
  core::Box box;
  auto atoms = load144(box);
  const auto zd = core::ZoneDecomposition::build(atoms, box, 4, 4.0);
  EXPECT_DOUBLE_EQ(zd.width, 4.05);
  int total = 0;
  for (int z = 0; z < 4; ++z) {
    for (int i : zd.members[z]) {
      EXPECT_GE(atoms.z[i], box.lo[2] + z * zd.width - 1e-12);
      EXPECT_LT(atoms.z[i], box.lo[2] + (z + 1) * zd.width + 1e-12);
    }
    total += int(zd.members[z].size());
  }
  EXPECT_EQ(total, atoms.n);

  // width < rcut — fatal (ConfigSchema rule)
  EXPECT_THROW(core::ZoneDecomposition::build(atoms, box, 5, 4.0),
               std::invalid_argument);
  // 2 zones with periodic z — fatal (closure == direct interface)
  EXPECT_THROW(core::ZoneDecomposition::build(atoms, box, 2, 4.0),
               std::invalid_argument);
}

// Acceptance: sum of w-contributions ≡ monolithic. Bitwise vs the SAME
// fixed-point math in a single zone (assembly split changes nothing), and
// within the Q24.40/Q34.30 quantization bound vs the FP64 oracle.
TEST(Zones, WContributionsMatchMonolithic) {
  core::Box box;
  auto atoms = load144(box);
  MorsePair pair;

  // single-zone fixed-point reference (same quantization, no decomposition)
  auto mono = atoms;
  core::zero_forces(mono);
  const auto zd1 = core::ZoneDecomposition::build(mono, box, 1, 4.0);
  const double pe1 = core::zone_force_pass(mono, box, zd1, 4.0, pair);

  // 4-zone assembly with PBC closure
  auto zoned = atoms;
  core::zero_forces(zoned);
  const auto zd4 = core::ZoneDecomposition::build(zoned, box, 4, 4.0);
  const double pe4 = core::zone_force_pass(zoned, box, zd4, 4.0, pair);

  EXPECT_EQ(std::memcmp(mono.fx.data(), zoned.fx.data(), mono.n * sizeof(double)), 0);
  EXPECT_EQ(std::memcmp(mono.fy.data(), zoned.fy.data(), mono.n * sizeof(double)), 0);
  EXPECT_EQ(std::memcmp(mono.fz.data(), zoned.fz.data(), mono.n * sizeof(double)), 0);
  EXPECT_EQ(pe1, pe4);  // Q34.30 — bit-identical sums

  // FP64 oracle (MorsePotential, same truncation): difference is bounded by
  // the per-contribution quantization (~n_neighbours · 2⁻⁴¹ ≈ 1e-11)
  auto ref = atoms;
  potentials::MorsePotential<double> oracle;
  core::zero_forces(ref);
  const double pe_ref = oracle.compute(ref, box);
  double maxd = 0.0;
  for (int i = 0; i < ref.n; ++i)
    maxd = std::max({maxd, std::fabs(ref.fx[i] - zoned.fx[i]),
                     std::fabs(ref.fy[i] - zoned.fy[i]),
                     std::fabs(ref.fz[i] - zoned.fz[i])});
  std::printf("Test_Zones: zoned vs FP64 oracle max|dF|=%.3e eV/Å |dPE|=%.3e eV\n",
              maxd, std::fabs(pe4 - pe_ref));
  EXPECT_LT(maxd, 1e-10);
  EXPECT_LT(std::fabs(pe4 - pe_ref), 1e-6);  // Q34.30 quantum × ~2000 pairs
}

// Acceptance: INV-8 — every min-image pair inside r_cut is evaluated exactly
// once across all zone passes (guards against a "cheating" full recompute
// per zone), including the cross-boundary and PBC-closure pairs.
TEST(Zones, EveryPairCountedExactlyOnce) {
  core::Box box;
  auto atoms = load144(box);
  MorsePair pair;
  const auto zd = core::ZoneDecomposition::build(atoms, box, 4, 4.0);

  std::map<std::pair<int, int>, int> counted;
  std::vector<int> order = {0, 1, 2, 3};
  core::zero_forces(atoms);
  core::zone_force_pass(atoms, box, zd, 4.0, pair, order, [&](int i, int j) {
    ++counted[{std::min(i, j), std::max(i, j)}];
  });

  const auto expected = brute_pairs(atoms, box, 4.0);
  ASSERT_EQ(counted.size(), expected.size());
  int cross_zone = 0;
  std::vector<int> zone_of(atoms.n);
  for (int z = 0; z < zd.n_zones; ++z)
    for (int i : zd.members[z]) zone_of[i] = z;
  for (const auto& [pr, cnt] : counted) {
    EXPECT_EQ(cnt, 1) << "pair (" << pr.first << "," << pr.second
                      << ") counted " << cnt << " times";
    EXPECT_TRUE(expected.count(pr));
    if (zone_of[pr.first] != zone_of[pr.second]) ++cross_zone;
  }
  EXPECT_GT(cross_zone, 0);  // boundary pairs do exist and go through T2 logic
}

// B1 end-to-end: the assembly is bit-identical for ANY zone processing order
// (the conveyor shuffles effective ordering across nodes — INV-9).
TEST(Zones, AssemblyIsOrderIndependentBitwise) {
  core::Box box;
  auto atoms = load144(box);
  MorsePair pair;

  auto run = [&](std::vector<int> order) {
    auto a = atoms;
    core::zero_forces(a);
    const auto zd = core::ZoneDecomposition::build(a, box, 4, 4.0);
    const double pe = core::zone_force_pass(a, box, zd, 4.0, pair, order,
                                            [](int, int) {});
    return std::make_pair(a, pe);
  };
  auto [a1, pe1] = run({0, 1, 2, 3});
  auto [a2, pe2] = run({3, 1, 0, 2});
  EXPECT_EQ(pe1, pe2);
  EXPECT_EQ(std::memcmp(a1.fx.data(), a2.fx.data(), a1.n * sizeof(double)), 0);
  EXPECT_EQ(std::memcmp(a1.fy.data(), a2.fy.data(), a1.n * sizeof(double)), 0);
  EXPECT_EQ(std::memcmp(a1.fz.data(), a2.fz.data(), a1.n * sizeof(double)), 0);
}

// Free z boundary: no closure — last zone has no forward partner; the
// assembly still matches its own single-zone reference bitwise.
TEST(Zones, FreeBoundaryNoClosure) {
  core::Box box;
  auto atoms = load144(box);
  box.periodic = {true, true, false};
  MorsePair pair;

  auto mono = atoms;
  core::zero_forces(mono);
  const auto zd1 = core::ZoneDecomposition::build(mono, box, 1, 4.0);
  const double pe1 = core::zone_force_pass(mono, box, zd1, 4.0, pair);

  auto zoned = atoms;
  core::zero_forces(zoned);
  const auto zd4 = core::ZoneDecomposition::build(zoned, box, 4, 4.0);
  const double pe4 = core::zone_force_pass(zoned, box, zd4, 4.0, pair);

  EXPECT_EQ(pe1, pe4);
  EXPECT_EQ(std::memcmp(mono.fx.data(), zoned.fx.data(), mono.n * sizeof(double)), 0);
  EXPECT_EQ(std::memcmp(mono.fz.data(), zoned.fz.data(), mono.n * sizeof(double)), 0);

  // and 2 zones are LEGAL on a free z (no closure to collide with)
  EXPECT_NO_THROW(core::ZoneDecomposition::build(atoms, box, 2, 4.0));
}
