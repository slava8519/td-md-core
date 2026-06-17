// M4-N — the perf-measurement harness math (EamPhaseBreakdown). A wrong derived
// metric (ms/step, atom-steps/s, hit-rate, phase fractions) would mislead the
// neighbour-backend bake-off, so the formulas are unit-tested.
#include <gtest/gtest.h>

#include "tdmd/metrics/eam_breakdown.hpp"
#include "tdmd/metrics/k_cadence.hpp"

using tdmd::metrics::EamPhaseBreakdown;
using tdmd::metrics::KCadenceMeter;

TEST(EamBreakdown, DerivedMetrics) {
  EamPhaseBreakdown b;
  b.density_ms = 200.0; b.embedding_ms = 0.0; b.force_ms = 200.0;  // total 400 ms
  b.n_atoms = 1000; b.steps = 100;
  b.examined_pairs = 1'000'000; b.real_pairs = 10'000;

  EXPECT_DOUBLE_EQ(b.total_ms(), 400.0);
  EXPECT_DOUBLE_EQ(b.ms_per_step(), 4.0);                 // 400/100
  // atom-steps/s = atoms·steps / total_s = 1000·100 / 0.4
  EXPECT_DOUBLE_EQ(b.atom_steps_per_s(), 1000.0 * 100.0 / 0.4);
  EXPECT_DOUBLE_EQ(b.hit_rate(), 0.01);                  // 10k/1M
  EXPECT_DOUBLE_EQ(b.density_frac(), 0.5);
  EXPECT_DOUBLE_EQ(b.embedding_frac(), 0.0);
  EXPECT_DOUBLE_EQ(b.force_frac(), 0.5);
}

TEST(EamBreakdown, ZeroSafe) {
  EamPhaseBreakdown b;  // all zero
  EXPECT_DOUBLE_EQ(b.ms_per_step(), 0.0);
  EXPECT_DOUBLE_EQ(b.atom_steps_per_s(), 0.0);
  EXPECT_DOUBLE_EQ(b.hit_rate(), 0.0);
  EXPECT_DOUBLE_EQ(b.density_frac(), 0.0);
}

// K-cadence meter: one atom moving +0.3 Å/step along x with skin=1.0 must fire a
// rebuild every time the displacement reaches skin/2=0.5 (i.e. 2·disp≥1.0). disp
// after step k from the reference = 0.3k ⇒ rebuild first fires at step 2 (0.6),
// reference resets, then again at step 4, 6, ... ⇒ 3 rebuilds over 6 steps,
// K_eff=2.0. Feeding R_buf=0.5 each step ⇒ K_pred=skin/(2·0.5)=1.0, conservatism=2.0.
TEST(KCadence, RebuildCriterionAndDerived) {
  std::vector<double> x{0.0}, y{0.0}, z{0.0};
  KCadenceMeter m;
  m.begin(x, y, z, /*skin=*/1.0);
  const bool expect_rebuild[6] = {false, true, false, true, false, true};
  for (int k = 0; k < 6; ++k) {
    x[0] += 0.3;
    const bool fired = m.step(x, y, z, /*r_buf=*/0.5);
    EXPECT_EQ(fired, expect_rebuild[k]) << "step " << (k + 1);
  }
  EXPECT_EQ(m.acc.steps, 6);
  EXPECT_EQ(m.acc.rebuilds, 3);
  EXPECT_DOUBLE_EQ(m.acc.K_eff(), 2.0);
  EXPECT_DOUBLE_EQ(m.acc.K_pred(), 1.0);          // skin / (2·mean(0.5))
  EXPECT_DOUBLE_EQ(m.acc.conservatism(), 2.0);
}

// K_pred is the HARMONIC mean of per-step skin/(2·R_buf), i.e. skin/(2·mean(R_buf)),
// NOT the arithmetic mean of K_pred (BUG-1 fix). Feed alternating R_buf={0.25,0.05}
// (per-step K_pred {2,10}). The budget-consistent answer is skin/(2·mean(R_buf)) =
// 1/(2·0.15) = 3.333…, far from the arithmetic mean 6.0. No motion ⇒ no rebuild.
TEST(KCadence, KPredIsBudgetHarmonicMean) {
  std::vector<double> x{0.0}, y{0.0}, z{0.0};
  KCadenceMeter m;
  m.begin(x, y, z, /*skin=*/1.0);
  const double rbuf[4] = {0.25, 0.05, 0.25, 0.05};
  for (double rb : rbuf) m.step(x, y, z, rb);  // atom frozen
  EXPECT_EQ(m.acc.rebuilds, 0);
  const double mean_R = (0.25 + 0.05 + 0.25 + 0.05) / 4.0;  // 0.15
  EXPECT_NEAR(m.acc.K_pred(), 1.0 / (2.0 * mean_R), 1e-12);  // ≈3.333, not 6.0
  EXPECT_GT(m.acc.K_pred(), 3.0);
  EXPECT_LT(m.acc.K_pred(), 4.0);
}

// The rebuild uses the MAX displacement over atoms (the standard half-skin rule):
// a single fast atom forces a rebuild even if every other atom is frozen.
TEST(KCadence, MaxOverAtomsTriggers) {
  std::vector<double> x{0.0, 0.0, 0.0}, y{0, 0, 0}, z{0, 0, 0};
  KCadenceMeter m;
  m.begin(x, y, z, /*skin=*/1.0);
  x[2] = 0.4;  // 2·0.4=0.8 < 1.0 — no rebuild
  EXPECT_FALSE(m.step(x, y, z, 0.1));
  x[2] = 0.6;  // 2·0.6=1.2 ≥ 1.0 — rebuild
  EXPECT_TRUE(m.step(x, y, z, 0.1));
  EXPECT_EQ(m.acc.rebuilds, 1);
}

// Zero rebuilds ⇒ K_eff is a lower bound (= steps), conservatism still defined.
// R_buf=0.125 ⇒ K_pred = 1/(2·0.125) = 4.0 ⇒ conservatism = 10/4 = 2.5.
TEST(KCadence, NeverRebuilds) {
  std::vector<double> x{0.0}, y{0.0}, z{0.0};
  KCadenceMeter m;
  m.begin(x, y, z, /*skin=*/1.0);
  for (int k = 0; k < 10; ++k) { x[0] += 0.01; m.step(x, y, z, 0.125); }  // disp 0.1 < 0.5
  EXPECT_EQ(m.acc.rebuilds, 0);
  EXPECT_DOUBLE_EQ(m.acc.K_eff(), 10.0);        // lower bound
  EXPECT_DOUBLE_EQ(m.acc.K_pred(), 4.0);
  EXPECT_DOUBLE_EQ(m.acc.conservatism(), 2.5);
}
