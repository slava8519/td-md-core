// M4-N — the perf-measurement harness math (EamPhaseBreakdown). A wrong derived
// metric (ms/step, atom-steps/s, hit-rate, phase fractions) would mislead the
// neighbour-backend bake-off, so the formulas are unit-tested.
#include <gtest/gtest.h>

#include "tdmd/metrics/eam_breakdown.hpp"

using tdmd::metrics::EamPhaseBreakdown;

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
