// M3.5 — Test_NVE_Invariant (Roadmap): 50 000 steps, dt = 1 fs, T(0) = 300 K,
// precision mode: deterministic_fp64 (Real=double, B1 fixed-point assembly).
//
// The acceptance wording is "многозонный конвейер не хуже 1-зонного эталона
// той же длины". Under B1 it holds in the strongest possible form: the pair
// set of a zone decomposition equals the monolithic one (width >= rcut), each
// pair's contribution is quantized identically, and integer addition is
// associative — so the multizone ring is BITWISE equal to the 1-zone run,
// secular trend included. This test asserts that identity AND measures the
// trend itself.
//
// Secular trend: linear LSQ fit of E_full(t) over all passes (the fit
// separates the trend from the bounded Verlet oscillation), reported in
// kT/(ns·dof) (OpenMM convention; reference points in
// _meta/MIXED_PRECISION_BESTPRACTICES_2026-06-11.md §3.5). The absolute value
// is dominated by the force discontinuity of the `shift` truncation at r_cut
// (аудит C1) — the CALIBRATED ceiling below is 3x the measured value of this
// fixture, recorded 2026-06-12; it guards against regressions, not against
// the truncation-scheme floor.
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <string>

#include "tdmd/core/conveyor.hpp"
#include "tdmd/core/thermal.hpp"
#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/potentials/cutoff.hpp"
#include "tdmd/potentials/morse.hpp"
#include "tdmd/units.hpp"

using namespace tdmd;

static std::string project_root() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT);
#else
  return ".";
#endif
}

namespace {

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

// LSQ slope of E(t), eV/ps.
double secular_slope(const std::vector<core::PassStats>& s, double dt) {
  const std::size_t n = s.size();
  double tm = 0.0, em = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    tm += double(i + 1) * dt;
    em += s[i].pe + s[i].ke;
  }
  tm /= double(n);
  em /= double(n);
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double dt_i = double(i + 1) * dt - tm;
    num += dt_i * (s[i].pe + s[i].ke - em);
    den += dt_i * dt_i;
  }
  return num / den;
}

}  // namespace

TEST(NveInvariant, MultizoneConveyorNotWorseThanSingleZone50k) {
  const double kT = 300.0, kDt = 0.001;  // 1 fs
  const long kSteps = 50000;

  core::Box box;
  core::AtomSoA<double> init;
  ASSERT_TRUE(io::read_lammps_data(
      project_root() + "/reference_data/al_fcc_72.data", init, box));
  box.periodic = {false, false, false};
  core::thermal::maxwell_init(init, kT, 1);
  const auto p0 = core::thermal::momentum(init);

  core::ConveyorOptions o;
  o.steps = kSteps;
  o.n_zones = 1;
  o.n_nodes = 1;
  o.auto_step = false;
  o.dt_initial = kDt;

  core::AtomSoA<double> a1 = init;
  auto r1 = core::run_conveyor(a1, box, 4.0, MorsePair{}, o);
  ASSERT_EQ(r1.halt, core::Halt::None) << r1.halt_msg;

  o.n_zones = 2;   // free z + 2 zones: complete bipartition of the pair set
  o.n_nodes = 2;
  core::AtomSoA<double> a2 = init;
  auto r2 = core::run_conveyor(a2, box, 4.0, MorsePair{}, o);
  ASSERT_EQ(r2.halt, core::Halt::None) << r2.halt_msg;

  // "не хуже" in the strongest form: bitwise identity of the trajectories
  // (B1: identical pair set => identical quantized force sums). All six
  // arrays — a single-axis divergence must not slip through (review M3.5).
  const std::size_t nb = a1.x.size() * sizeof(double);
  EXPECT_EQ(0, std::memcmp(a1.x.data(), a2.x.data(), nb));
  EXPECT_EQ(0, std::memcmp(a1.y.data(), a2.y.data(), nb));
  EXPECT_EQ(0, std::memcmp(a1.z.data(), a2.z.data(), nb));
  EXPECT_EQ(0, std::memcmp(a1.vx.data(), a2.vx.data(), nb));
  EXPECT_EQ(0, std::memcmp(a1.vy.data(), a2.vy.data(), nb));
  EXPECT_EQ(0, std::memcmp(a1.vz.data(), a2.vz.data(), nb));
  ASSERT_EQ(r1.stats.size(), r2.stats.size());
  for (std::size_t i = 0; i < r1.stats.size(); ++i) {
    // PE is fixed-point (B1) — bitwise across zone groupings. KE is an FP64
    // diagnostic summed in zone order, so 1 vs 2 zones differ by resummation
    // ulps; the per-atom data behind it is bitwise (asserted above).
    ASSERT_EQ(r1.stats[i].pe, r2.stats[i].pe) << "pass " << i + 1;
    ASSERT_NEAR(r1.stats[i].ke, r2.stats[i].ke, 1e-12) << "pass " << i + 1;
  }

  // Secular trend in kT/(ns·dof).
  const int dof = core::thermal::dof_thermal(init.n);
  const double slope = secular_slope(r1.stats, kDt);             // eV/ps
  const double trend = slope * 1000.0 / (units::kB * kT * dof);  // kT/(ns·dof)
  ::testing::Test::RecordProperty("secular_trend_kT_per_ns_dof", trend);
  std::printf("[NVE] secular trend: %.3e kT/(ns·dof) [deterministic_fp64, "
              "morse+shift, dt=1fs, 50k steps, Al-72 free]\n", trend);
  // Calibrated ceiling: 3x the measured value of this exact fixture
  // (measured -1.204e-2 on 2026-06-12). The floor is set by the `shift`
  // force discontinuity at r_cut (|F(r_c)| ~ 0.165 eV/Å for Morse-Al, аудит
  // C1) — NOT by the conveyor: the multizone run is bitwise equal to the
  // 1-zone reference above. force_shift would lower the floor; the replica
  // keeps the dissertation's scheme.
  EXPECT_LT(std::fabs(trend), 3.7e-2);

  // Momentum conservation over 50k steps (Newton-3 + fixed point: the total
  // of quantized pair forces cancels exactly; residual is per-atom FP64
  // rounding of the kicks).
  const auto p = core::thermal::momentum(a2);
  for (int d = 0; d < 3; ++d) EXPECT_NEAR(p[d], p0[d], 1e-8) << "axis " << d;
}
