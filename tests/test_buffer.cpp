#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include <string>
#include "tdmd/core/soa.hpp"
#include "tdmd/core/buffer.hpp"
#include "tdmd/io/rescue.hpp"

using namespace tdmd;
using namespace tdmd::core;

// M2 acceptance (Roadmap M2 / ZoneFSM §6 INV-4 / dissertation Гл.2.1, 3.3, 3.5).

TEST(Buffer, RBufAndCausality) {
  // R_buf = v_max · dt · C (eq. 33)
  EXPECT_DOUBLE_EQ(buffer::compute_R_buf(10.0, 0.001, 1.5), 0.015);
  // INV-4: v_max·dt <= R_buf
  EXPECT_TRUE(buffer::causality_ok(10.0, 0.001, 0.015));
  EXPECT_TRUE(buffer::causality_ok(10.0, 0.001, 0.010));  // exactly v_max·dt
  EXPECT_FALSE(buffer::causality_ok(10.0, 0.002, 0.015)); // 0.02 > 0.015 -> violation
}

TEST(Buffer, MaxSpeedReduction) {
  AtomSoA<double> a; a.resize(3);
  a.vx = {3.0, 0.0, 1.0}; a.vy = {4.0, 0.0, 1.0}; a.vz = {0.0, 5.0, 1.0};
  EXPECT_DOUBLE_EQ(buffer::max_speed(a), 5.0);  // atom0: |(3,4,0)|=5; atom1: 5
}

// C1 (highest priority): fastest atom travels exactly C1·cell_size per step.
TEST(Buffer, AutoDtC1Displacement) {
  buffer::TimeStepCfg cfg;
  cfg.C1 = 0.1; cfg.cell_size = 2.33; cfg.C3 = 1.0;  // always update
  cfg.dt_max = 1.0;
  const double v_max = 10.0;
  const double dt = buffer::auto_dt(v_max, 0.5, cfg);
  EXPECT_NEAR(dt, 0.1 * 2.33 / 10.0, 1e-12);     // 0.0233 ps
  EXPECT_NEAR(v_max * dt, 0.1 * 2.33, 1e-12);    // displacement = 0.233 Å
}

// C3 hysteresis (ур.62): 0 = fixed, 1 = always, 0.5 = switch on >=50% change.
TEST(Buffer, AutoDtC3Hysteresis) {
  buffer::TimeStepCfg cfg;
  cfg.C1 = 0.1; cfg.cell_size = 2.33; cfg.dt_max = 1.0;

  cfg.C3 = 0.0;  // fixed step — never changes
  EXPECT_DOUBLE_EQ(buffer::auto_dt(10.0, 0.02, cfg), 0.02);

  cfg.C3 = 1.0;  // update every step
  EXPECT_NEAR(buffer::auto_dt(10.0, 0.02, cfg), 0.0233, 1e-9);

  cfg.C3 = 0.5;  // switch only on >= 50% relative change
  // small change: target 0.0233 vs current 0.02 -> 16% -> keep
  EXPECT_DOUBLE_EQ(buffer::auto_dt(10.0, 0.02, cfg), 0.02);
  // big change: target 0.00233 vs current 0.02 -> 88% -> switch
  EXPECT_NEAR(buffer::auto_dt(100.0, 0.02, cfg), 0.00233, 1e-9);
}

TEST(Buffer, TemperatureLimitedDtFinitePositive) {
  AtomSoA<double> a; a.resize(2);
  a.mass = {26.9815, 26.9815};
  a.vx = {5.0, 0.0}; a.vy = {0.0, 0.0}; a.vz = {0.0, 0.0};
  a.fx = {1.0, 0.5}; a.fy = {0.0, 0.0}; a.fz = {0.0, 0.0};
  const double dt = buffer::temperature_limited_dt(a, 50.0);
  EXPECT_TRUE(std::isfinite(dt));
  // sane physical range for Al at K2=50 — also guards the mvv2e units factor
  // (a missing mvv2e would collapse dt by ~1e4).
  EXPECT_GT(dt, 1e-4);
  EXPECT_LT(dt, 1e-1);
}

// INV-4 violation -> HALT path writes a rescue dump (ZoneFSM §9).
TEST(Buffer, RescueDumpOnViolation) {
  AtomSoA<double> a; a.resize(2);
  a.type = {1, 1};
  a.x = {0.1, 1.2}; a.y = {0.2, 1.3}; a.z = {0.3, 1.4};
  Box box; box.lo = {0, 0, 0}; box.hi = {10, 10, 10};

  // construct a violation: v_max·dt > R_buf
  const double v_max = 50.0, dt = 0.01;
  const double R_buf = buffer::compute_R_buf(5.0 /*stale*/, dt, 1.5);
  ASSERT_FALSE(buffer::causality_ok(v_max, dt, R_buf));

  const std::string path = "test_rescue.xyz";
  ASSERT_TRUE(io::write_rescue_xyz(path, a, box, "INV-4 test"));
  std::ifstream f(path);
  ASSERT_TRUE(f.good());
  std::string first; std::getline(f, first);
  EXPECT_EQ(first, "2");          // atom count
  std::string second; std::getline(f, second);
  EXPECT_NE(second.find("INV-4 test"), std::string::npos);
  f.close();
  std::remove(path.c_str());
}
