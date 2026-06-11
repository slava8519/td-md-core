// M3/B10: strict config validation (ConfigSchema §2) — bad enums/ranges are
// fatal, unknown keys warn with the key name, the shipped configs stay valid.
#include <gtest/gtest.h>

#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "tdmd/io/config.hpp"

using namespace tdmd;
namespace fs = std::filesystem;

static std::string project_root() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT);
#else
  return ".";
#endif
}

namespace {

// Writes a config with one section overridden and returns its path.
class TempConfig {
 public:
  explicit TempConfig(const std::string& yaml) {
    static int n = 0;
    path_ = fs::temp_directory_path() /
            ("tdmd_cfg_test_" + std::to_string(n++) + ".yaml");
    std::ofstream f(path_);
    f << yaml;
  }
  ~TempConfig() { std::error_code ec; fs::remove(path_, ec); }
  std::string path() const { return path_.string(); }

 private:
  fs::path path_;
};

const char* kValid = R"(
run: { steps: 10, ensemble: nve, seed: 1, init_temperature: 300.0 }
units: metal
precision: { mode: deterministic_fp64 }
geometry: { file: reference_data/al_fcc_72.data, format: lammps_data }
boundary: { x: periodic, y: periodic, z: free }
potential: { type: morse, r_cut: 4.0, shift: true, morse: { D: 0.29614, alpha: 1.11892, r0: 3.29692 } }
timestep: { mode: auto, dt_initial: 0.001, dt_max: 0.02, C1: 0.1, K2: 50.0, C3: 0.5, C_buf: 1.5 }
)";

void expect_throws_with(const std::string& yaml, const std::string& needle) {
  TempConfig cfg(yaml);
  try {
    io::load_config(cfg.path());
    FAIL() << "expected validation failure mentioning '" << needle << "'";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find(needle), std::string::npos)
        << "actual message: " << e.what();
  }
}

}  // namespace

TEST(Config, ValidConfigParses) {
  TempConfig cfg(kValid);
  io::Config c = io::load_config(cfg.path());
  EXPECT_EQ(c.steps, 10);
  EXPECT_EQ(c.ensemble, "nve");
  EXPECT_DOUBLE_EQ(c.init_temperature, 300.0);
  EXPECT_EQ(c.periodic[0], true);
  EXPECT_EQ(c.periodic[2], false);
  EXPECT_EQ(c.ts_mode, "auto");
  EXPECT_DOUBLE_EQ(c.C_buf, 1.5);
}

TEST(Config, ShippedConfigsAreValid) {
  EXPECT_NO_THROW(io::load_config(project_root() + "/config/config_m0.yaml"));
  EXPECT_NO_THROW(io::load_config(project_root() + "/config/config_auto.yaml"));
}

TEST(Config, ShippedConfigsEmitNoWarnings) {
  testing::internal::CaptureStderr();
  io::load_config(project_root() + "/config/config_m0.yaml");
  io::load_config(project_root() + "/config/config_auto.yaml");
  const std::string err = testing::internal::GetCapturedStderr();
  EXPECT_EQ(err.find("[config] warning"), std::string::npos) << err;
}

TEST(Config, UnknownKeyWarnsWithName) {
  TempConfig cfg(std::string(kValid) + "\nrunn: { steps: 5 }\n");
  testing::internal::CaptureStderr();
  EXPECT_NO_THROW(io::load_config(cfg.path()));  // warning, not fatal
  const std::string err = testing::internal::GetCapturedStderr();
  EXPECT_NE(err.find("unknown key 'runn'"), std::string::npos) << err;
}

TEST(Config, UnknownNestedKeyWarnsWithSection) {
  TempConfig cfg(std::string(kValid) +
                 "\nio: { trajectory: { file: t.lammpstrj, evry: 5 } }\n");
  testing::internal::CaptureStderr();
  EXPECT_NO_THROW(io::load_config(cfg.path()));
  const std::string err = testing::internal::GetCapturedStderr();
  EXPECT_NE(err.find("'io.trajectory.evry'"), std::string::npos) << err;
}

// ConfigSchema §2: unknown timestep.mode is FATAL, not a silent 'fixed'
TEST(Config, BadTimestepModeIsFatal) {
  std::string y(kValid);
  y.replace(y.find("mode: auto"), 10, "mode: warp");
  expect_throws_with(y, "timestep.mode");
}

TEST(Config, BadBoundaryIsFatal) {  // was: 'fre' silently meant periodic
  std::string y(kValid);
  y.replace(y.find("z: free"), 7, "z: fre");
  expect_throws_with(y, "boundary.z");
}

TEST(Config, BadEnsembleIsFatal) {
  std::string y(kValid);
  y.replace(y.find("ensemble: nve"), 13, "ensemble: nvt");
  expect_throws_with(y, "run.ensemble");
}

TEST(Config, BadPrecisionModeIsFatal) {
  std::string y(kValid);
  y.replace(y.find("mode: deterministic_fp64"), 24, "mode: fp64");
  expect_throws_with(y, "precision.mode");
}

TEST(Config, BadUnitsIsFatal) {
  std::string y(kValid);
  y.replace(y.find("units: metal"), 12, "units: real");
  expect_throws_with(y, "units");
}

TEST(Config, CbufBelowOneIsFatal) {  // eq.33: buffer cannot guarantee causality
  std::string y(kValid);
  y.replace(y.find("C_buf: 1.5"), 10, "C_buf: 0.5");
  expect_throws_with(y, "C_buf");
}

TEST(Config, C1OutOfRangeIsFatal) {  // ConfigSchema: 0 < C1 <= 1
  std::string y(kValid);
  y.replace(y.find("C1: 0.1"), 7, "C1: 1.5");
  expect_throws_with(y, "C1");
}

TEST(Config, NegativeDtIsFatal) {
  std::string y(kValid);
  y.replace(y.find("dt_initial: 0.001"), 17, "dt_initial: -0.001");
  expect_throws_with(y, "dt_initial");
}

TEST(Config, DtMaxBelowDtInitialIsFatal) {
  std::string y(kValid);
  y.replace(y.find("dt_max: 0.02"), 12, "dt_max: 1e-5");
  expect_throws_with(y, "dt_max");
}

TEST(Config, MissingGeometryFileIsFatal) {
  expect_throws_with("run: { steps: 1 }\n", "geometry.file");
}

TEST(Config, AllErrorsAreCollected) {  // one pass reports every problem
  std::string y(kValid);
  y.replace(y.find("ensemble: nve"), 13, "ensemble: nvt");
  y.replace(y.find("C_buf: 1.5"), 10, "C_buf: 0.5");
  TempConfig cfg(y);
  try {
    io::load_config(cfg.path());
    FAIL();
  } catch (const std::runtime_error& e) {
    const std::string w = e.what();
    EXPECT_NE(w.find("run.ensemble"), std::string::npos) << w;
    EXPECT_NE(w.find("C_buf"), std::string::npos) << w;
  }
}
