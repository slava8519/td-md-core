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
  EXPECT_NO_THROW(io::load_config(project_root() + "/config/config_ring.yaml"));
}

TEST(Config, ShippedConfigsEmitNoWarnings) {
  testing::internal::CaptureStderr();
  io::load_config(project_root() + "/config/config_m0.yaml");
  io::load_config(project_root() + "/config/config_auto.yaml");
  io::load_config(project_root() + "/config/config_ring.yaml");
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

// --- M3: potential.type=lj, truncation schemes (cutoff.hpp) ---

TEST(Config, LJPotentialParses) {
  TempConfig cfg(R"(
run: { steps: 1 }
geometry: { file: reference_data/al_fcc_72.data }
potential: { type: lj, r_cut: 3.0, truncation: force_shift, lj: { epsilon: 1.0, sigma: 1.0 } }
)");
  io::Config c = io::load_config(cfg.path());
  EXPECT_EQ(c.pot_type, "lj");
  EXPECT_EQ(c.truncation, "force_shift");
  EXPECT_DOUBLE_EQ(c.lj_epsilon, 1.0);
  EXPECT_DOUBLE_EQ(c.lj_sigma, 1.0);
}

TEST(Config, BadPotentialTypeIsFatal) {
  std::string y(kValid);
  y.replace(y.find("type: morse"), 11, "type: reaxx");
  expect_throws_with(y, "potential.type");
}

TEST(Config, BadTruncationIsFatal) {
  TempConfig cfg(R"(
run: { steps: 1 }
geometry: { file: reference_data/al_fcc_72.data }
potential: { type: lj, r_cut: 3.0, truncation: smooth }
)");
  EXPECT_THROW(io::load_config(cfg.path()), std::runtime_error);
}

TEST(Config, LegacyShiftMapsToTruncation) {
  std::string y(kValid);  // shift: true
  {
    TempConfig cfg(y);
    EXPECT_EQ(io::load_config(cfg.path()).truncation, "shift");
  }
  y.replace(y.find("shift: true"), 11, "shift: false");
  {
    TempConfig cfg(y);
    EXPECT_EQ(io::load_config(cfg.path()).truncation, "cut");
  }
}

TEST(Config, TruncationWinsOverLegacyShiftWithWarning) {
  std::string y(kValid);
  y.replace(y.find("shift: true"), 11, "shift: false, truncation: shift");
  TempConfig cfg(y);
  testing::internal::CaptureStderr();
  io::Config c = io::load_config(cfg.path());
  const std::string err = testing::internal::GetCapturedStderr();
  EXPECT_EQ(c.truncation, "shift");
  EXPECT_NE(err.find("potential.shift is ignored"), std::string::npos) << err;
}

TEST(Config, NonPositiveLJParamsAreFatal) {
  TempConfig cfg(R"(
run: { steps: 1 }
geometry: { file: reference_data/al_fcc_72.data }
potential: { type: lj, r_cut: 3.0, lj: { epsilon: -1.0, sigma: 1.0 } }
)");
  EXPECT_THROW(io::load_config(cfg.path()), std::runtime_error);
}

// Review M3: parameter ranges are scoped to the ACTIVE potential — a stale
// block of the inactive type must not be fatal.
TEST(Config, InactivePotentialBlockIsNotValidated) {
  TempConfig lj_with_bad_morse(R"(
run: { steps: 1 }
geometry: { file: reference_data/al_fcc_72.data }
potential: { type: lj, r_cut: 3.0, lj: { epsilon: 1.0, sigma: 1.0 }, morse: { D: 0 } }
)");
  EXPECT_NO_THROW(io::load_config(lj_with_bad_morse.path()));
  TempConfig morse_with_bad_lj(R"(
run: { steps: 1 }
geometry: { file: reference_data/al_fcc_72.data }
potential: { type: morse, r_cut: 4.0, lj: { epsilon: -1.0 } }
)");
  EXPECT_NO_THROW(io::load_config(morse_with_bad_lj.path()));
}

// --- M4: decomposition / ring keys (CLI conveyor path) ---

TEST(Config, RingKeysParse) {
  TempConfig cfg(std::string(kValid) +
                 "\ndecomposition: { axis: z, mode: by_n_zones, n_zones: 2, "
                 "ring: { backend: streams, n_nodes: 4 } }\n");
  io::Config c = io::load_config(cfg.path());
  EXPECT_EQ(c.n_zones, 2);
  EXPECT_EQ(c.ring_nodes, 4);
  EXPECT_EQ(c.ring_backend, "streams");
}

TEST(Config, ZoneWidthBelowRcutIsFatal) {  // ConfigSchema: причинность
  expect_throws_with(std::string(kValid) +
                         "\ndecomposition: { mode: by_zone_width, "
                         "zone_width: 2.0 }\n",
                     "zone_width");
}

TEST(Config, StepsPerNodeBeyondOneIsFatal) {  // k>1 — M5a (Гл. 3.4)
  expect_throws_with(std::string(kValid) +
                         "\ndecomposition: { ring: { steps_per_node: 3 } }\n",
                     "steps_per_node");
}

TEST(Config, BadRingBackendIsFatal) {
  expect_throws_with(std::string(kValid) +
                         "\ndecomposition: { ring: { backend: warp } }\n",
                     "decomposition.ring.backend");
}

// Review M3: a scheme name in the legacy bool key is the likely migration
// typo — it must land in the collected B10 report with a hint, not surface
// as a raw yaml-cpp bad-conversion.
TEST(Config, SchemeNameInLegacyShiftKeyIsFatalWithHint) {
  expect_throws_with(R"(
run: { steps: 1 }
geometry: { file: reference_data/al_fcc_72.data }
potential: { type: morse, r_cut: 4.0, shift: force_shift }
)",
                     "potential.truncation");
}
