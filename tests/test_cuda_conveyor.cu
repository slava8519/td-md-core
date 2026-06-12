// M4 (label: cuda) — GpuTimeConveyor acceptance: the M3.5 test set on the GPU
// (Roadmap M4 criterion).
//
//   * deterministic_fp64 cell of the INV-9 matrix: 1 stream vs N streams
//     BITWISE — and, stronger, GPU conveyor == CPU conveyor bitwise for the
//     transcendental-free LJ under --fmad=false (every per-zone computation
//     was bit-validated in Test_CUDA_Zones; this pins the orchestration);
//   * Δt-handoff through GPU reductions: auto-mode dt sequence == CPU's;
//   * PBC rotation closure on the GPU vs the CPU oracle;
//   * §3.6 replica on the GPU: Al-72, free boundaries, auto-step, 25 900
//     steps, 1 node vs 4 nodes — zero deviation (Morse: GPU-internal bitwise);
//   * INV-4 HALT fires through the device reduction path.
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "tdmd/core/conveyor.hpp"
#include "tdmd/core/thermal.hpp"
#include "tdmd/cuda/conveyor_gpu.cuh"
#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/potentials/cutoff.hpp"
#include "tdmd/potentials/morse.hpp"
#include "tdmd/potentials/pair_lj.hpp"
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

using cuda::LJDev;
using cuda::MorseDev;

LJDev make_lj(double eps, double sigma, double rcut) {
  potentials::LJParams<double> p{eps, sigma};
  return {p, potentials::CutoffScheme::make(
                 potentials::Truncation::Shift, rcut,
                 [&](double r, double& u, double& f) {
                   potentials::pair_lj(r, p, u, f);
                 })};
}
MorseDev make_morse(double rcut) {
  potentials::MorseParams<double> p{0.29614, 1.11892, 3.29692};
  return {p, potentials::CutoffScheme::make(
                 potentials::Truncation::Shift, rcut,
                 [&](double r, double& u, double& f) {
                   potentials::pair_morse(r, p, u, f);
                 })};
}

constexpr double kRcut = 4.0;

core::AtomSoA<double> make_fcc(core::Box& box, int cx, int cy, int cz,
                               bool pz, double a0 = 4.05) {
  core::AtomSoA<double> at;
  at.resize(4 * cx * cy * cz);
  box.lo = {0.0, 0.0, 0.0};
  box.hi = {cx * a0, cy * a0, cz * a0};
  box.periodic = {true, true, pz};
  static const double basis[4][3] = {
      {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  int k = 0;
  for (int ix = 0; ix < cx; ++ix)
    for (int iy = 0; iy < cy; ++iy)
      for (int iz = 0; iz < cz; ++iz)
        for (int b = 0; b < 4; ++b, ++k) {
          at.x[k] = (ix + basis[b][0]) * a0;
          at.y[k] = (iy + basis[b][1]) * a0;
          at.z[k] = (iz + basis[b][2]) * a0 + 0.25 * a0;
          at.type[k] = 1;
          at.mass[k] = 26.9815;
        }
  return at;
}

::testing::AssertionResult bitwise_eq(const core::AtomSoA<double>& a,
                                      const core::AtomSoA<double>& b) {
  if (a.n != b.n) return ::testing::AssertionFailure() << "size mismatch";
  auto cmp = [&](const std::vector<double>& u, const std::vector<double>& v,
                 const char* name) {
    return std::memcmp(u.data(), v.data(), u.size() * sizeof(double)) == 0
               ? ""
               : name;
  };
  std::string bad;
  bad += cmp(a.x, b.x, "x ");
  bad += cmp(a.y, b.y, "y ");
  bad += cmp(a.z, b.z, "z ");
  bad += cmp(a.vx, b.vx, "vx ");
  bad += cmp(a.vy, b.vy, "vy ");
  bad += cmp(a.vz, b.vz, "vz ");
  if (bad.empty()) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << "bitwise mismatch in: " << bad;
}

core::ConveyorOptions opts_fixed(long steps, int n_zones, int n_nodes,
                                 double dt) {
  core::ConveyorOptions o;
  o.steps = steps;
  o.n_zones = n_zones;
  o.n_nodes = n_nodes;
  o.auto_step = false;
  o.dt_initial = dt;
  return o;
}

template <typename PairF>
core::AtomSoA<double> run_gpu(const core::AtomSoA<double>& init,
                              const core::Box& box,
                              const core::ConveyorOptions& o, const PairF& p,
                              core::ConveyorResult* out = nullptr) {
  core::AtomSoA<double> a = init;
  auto r = cuda::run_conveyor_gpu(a, box, kRcut, p, o);
  EXPECT_EQ(r.halt, core::Halt::None) << r.halt_msg;
  EXPECT_EQ(r.steps_done, o.steps);
  if (out) *out = std::move(r);
  return a;
}

template <typename PairF>
core::AtomSoA<double> run_cpu(const core::AtomSoA<double>& init,
                              const core::Box& box,
                              const core::ConveyorOptions& o, const PairF& p,
                              core::ConveyorResult* out = nullptr) {
  core::AtomSoA<double> a = init;
  auto r = core::run_conveyor(a, box, kRcut, p, o);
  EXPECT_EQ(r.halt, core::Halt::None) << r.halt_msg;
  if (out) *out = std::move(r);
  return a;
}

}  // namespace

// --- LJ fixed dt: GPU == CPU conveyor bitwise; 1 stream vs N streams ---

TEST(CudaConveyor, LjFixedBitwiseVsCpuAnd1vsNStreams) {
  core::Box box;
  auto init = make_fcc(box, 2, 2, 6, /*pz=*/false);
  core::thermal::maxwell_init(init, 300.0, 51);
  const auto lj = make_lj(0.4, 2.55, kRcut);
  const auto o1 = opts_fixed(40, 4, 1, 0.002);

  core::ConveyorResult rc, rg;
  auto cpu = run_cpu(init, box, o1, lj, &rc);
  auto gpu = run_gpu(init, box, o1, lj, &rg);
  EXPECT_TRUE(bitwise_eq(cpu, gpu));  // orchestration pinned to the CPU ring
  ASSERT_EQ(rc.stats.size(), rg.stats.size());
  for (std::size_t i = 0; i < rc.stats.size(); ++i)
    ASSERT_EQ(rc.stats[i].pe, rg.stats[i].pe) << "pe pass " << i + 1;

  for (int z : {2, 3}) {  // INV-9: 1 stream vs N streams, bitwise
    auto oz = o1;
    oz.n_nodes = z;
    auto got = run_gpu(init, box, oz, lj);
    EXPECT_TRUE(bitwise_eq(gpu, got)) << "z=" << z;
  }
}

// --- auto dt: the Λ-chain fed by DEVICE reductions matches the CPU ---

TEST(CudaConveyor, LjAutoDtMatchesCpuConveyor) {
  core::Box box;
  auto init = make_fcc(box, 2, 2, 6, /*pz=*/false);
  core::thermal::maxwell_init(init, 300.0, 53);
  const auto lj = make_lj(0.4, 2.55, kRcut);

  core::ConveyorOptions o;
  o.steps = 120;
  o.n_zones = 4;
  o.n_nodes = 1;
  o.auto_step = true;
  o.dt_initial = 0.001;
  o.ts.C1 = 0.01;
  o.ts.C3 = 1.0;

  core::ConveyorResult rc, rg;
  auto cpu = run_cpu(init, box, o, lj, &rc);
  auto gpu = run_gpu(init, box, o, lj, &rg);
  EXPECT_TRUE(bitwise_eq(cpu, gpu));
  ASSERT_EQ(rc.stats.size(), rg.stats.size());
  for (std::size_t i = 0; i < rc.stats.size(); ++i) {
    ASSERT_EQ(rc.stats[i].dt, rg.stats[i].dt) << "dt pass " << i + 1;
    ASSERT_EQ(rc.stats[i].v_max, rg.stats[i].v_max) << "v_max pass " << i + 1;
  }

  o.n_nodes = 2;  // and across stream counts
  core::ConveyorResult rg2;
  auto gpu2 = run_gpu(init, box, o, lj, &rg2);
  EXPECT_TRUE(bitwise_eq(gpu, gpu2));
  for (std::size_t i = 0; i < rg.stats.size(); ++i)
    ASSERT_EQ(rg.stats[i].dt, rg2.stats[i].dt);
}

// --- PBC rotation closure on the GPU ---

TEST(CudaConveyor, PbcClosureBitwiseVsCpu) {
  core::Box box;
  auto init = make_fcc(box, 2, 2, 8, /*pz=*/true);
  core::thermal::maxwell_init(init, 300.0, 57);
  const auto lj = make_lj(0.4, 2.55, kRcut);
  const auto o = opts_fixed(50, 4, 1, 0.002);

  auto cpu = run_cpu(init, box, o, lj);
  auto gpu = run_gpu(init, box, o, lj);
  EXPECT_TRUE(bitwise_eq(cpu, gpu));

  auto oz = o;
  oz.n_nodes = 3;
  auto gpu3 = run_gpu(init, box, oz, lj);
  EXPECT_TRUE(bitwise_eq(gpu, gpu3));
}

// --- §3.6 replica on the GPU: Al-72, free boundaries, auto-step (C1=10
// дисс.), 25 900 steps, 1 node vs 4 nodes — zero deviation of coordinates
// and velocities. Morse => the cross-platform comparison is GPU-internal. ---

TEST(CudaConveyor, Replica36OnGpu) {
  core::Box box;
  core::AtomSoA<double> init;
  ASSERT_TRUE(io::read_lammps_data(
      project_root() + "/reference_data/al_fcc_72.data", init, box));
  box.periodic = {false, false, false};
  core::thermal::maxwell_init(init, 300.0, 1);
  const auto mo = make_morse(kRcut);

  core::ConveyorOptions o;
  o.steps = 25900;
  o.n_zones = 2;
  o.n_nodes = 1;
  o.auto_step = true;
  o.dt_initial = 0.001;
  o.ts.cell_size = 2.33;

  core::ConveyorResult r1, r4;
  auto g1 = run_gpu(init, box, o, mo, &r1);
  o.n_nodes = 4;
  auto g4 = run_gpu(init, box, o, mo, &r4);
  EXPECT_TRUE(bitwise_eq(g1, g4));  // нулевое отклонение, как рис. 43–44
  ASSERT_EQ(r1.stats.size(), r4.stats.size());
  for (std::size_t i = 0; i < r1.stats.size(); ++i)
    ASSERT_EQ(r1.stats[i].dt, r4.stats[i].dt) << "dt pass " << i + 1;

  // informational: GPU vs CPU conveyor (Morse — tolerance-class comparison;
  // measured agreement has been exact thanks to the Q24.40 quantum)
  core::AtomSoA<double> c = init;
  auto rc = core::run_conveyor(c, box, kRcut, mo, [&] {
    auto oc = o;
    oc.n_nodes = 1;
    return oc;
  }());
  ASSERT_EQ(rc.halt, core::Halt::None) << rc.halt_msg;
  double maxd = 0.0;
  for (int i = 0; i < c.n; ++i) {
    maxd = std::max({maxd, std::fabs(c.x[i] - g1.x[i]),
                     std::fabs(c.y[i] - g1.y[i]),
                     std::fabs(c.z[i] - g1.z[i])});
  }
  std::printf("Test_CUDA_Conveyor[replica36]: GPU-vs-CPU max|dx| over 25900 "
              "auto-steps = %.3e A [deterministic_fp64, morse+shift]\n",
              maxd);
}

// --- production_mixed (M4/B5): FP32 pair math + int32 fixed-point transport.
// The B5 design makes the snap sequence z-independent (one pack per zone per
// pass at ANY node count), so production_mixed is bitwise BOTH run-to-run
// AND 1-vs-z — an upgrade over the INV-9 matrix's original "допусково". ---

namespace {
cuda::LJDevF32 make_lj_f32(float eps, float sigma, double rcut) {
  potentials::LJParams<float> p{eps, sigma};
  return {p, potentials::CutoffScheme::make(
                 potentials::Truncation::Shift, rcut,
                 [&](double r, double& u, double& f) {
                   float uf, ff;
                   potentials::pair_lj(float(r), p, uf, ff);
                   u = double(uf);
                   f = double(ff);
                 })};
}
}  // namespace

TEST(CudaConveyor, MixedRunToRunAnd1vsZBitwise) {
  core::Box box;
  auto init = make_fcc(box, 2, 2, 6, /*pz=*/false);
  core::thermal::maxwell_init(init, 300.0, 61);
  const auto lj32 = make_lj_f32(0.4f, 2.55f, kRcut);

  auto o = opts_fixed(60, 4, 1, 0.002);
  o.mixed_transport = true;

  auto g1a = run_gpu(init, box, o, lj32);
  auto g1b = run_gpu(init, box, o, lj32);
  EXPECT_TRUE(bitwise_eq(g1a, g1b));  // run-to-run (the INV-9 matrix cell)

  for (int z : {2, 3}) {  // the B5 upgrade: bitwise across node counts
    auto oz = o;
    oz.n_nodes = z;
    auto gz = run_gpu(init, box, oz, lj32);
    EXPECT_TRUE(bitwise_eq(g1a, gz)) << "z=" << z;
  }
}

TEST(CudaConveyor, MixedAccuracyVsFp64) {
  core::Box box;
  auto init = make_fcc(box, 2, 2, 6, /*pz=*/false);
  core::thermal::maxwell_init(init, 300.0, 67);

  auto o = opts_fixed(60, 4, 1, 0.002);
  auto ref = run_gpu(init, box, o, make_lj(0.4, 2.55, kRcut));
  auto om = o;
  om.mixed_transport = true;
  auto mix = run_gpu(init, box, om, make_lj_f32(0.4f, 2.55f, kRcut));

  double maxd = 0.0;
  for (int i = 0; i < ref.n; ++i)
    maxd = std::max({maxd, std::fabs(ref.x[i] - mix.x[i]),
                     std::fabs(ref.y[i] - mix.y[i]),
                     std::fabs(ref.z[i] - mix.z[i])});
  std::printf("Test_CUDA_Conveyor[mixed]: production_mixed vs "
              "deterministic_fp64 max|dx| over 60 steps = %.3e A\n",
              maxd);
  EXPECT_LT(maxd, 1e-2);  // FP32 pair-math accuracy class, short horizon
}

TEST(CudaConveyor, MixedTransportRangeHalts) {
  core::Box box;
  box.lo = {0.0, 0.0, 0.0};
  box.hi = {12.0, 12.0, 12.0};
  box.periodic = {false, false, false};
  core::AtomSoA<double> a;
  a.resize(2);
  a.x = {-200.0, 6.0};  // first atom far beyond the int32 offset range
  a.y = {6.0, 6.0};
  a.z = {6.0, 6.0};
  a.type = {1, 1};
  a.mass = {26.9815, 26.9815};

  auto o = opts_fixed(3, 1, 1, 0.001);
  o.mixed_transport = true;
  auto r = cuda::run_conveyor_gpu(a, box, kRcut, make_lj_f32(0.4f, 2.55f, kRcut), o);
  EXPECT_EQ(r.halt, core::Halt::Internal) << r.halt_msg;
  EXPECT_NE(r.halt_msg.find("transport range"), std::string::npos) << r.halt_msg;
}

// --- anti-deadlock on the GPU ring (§7.4): odd and even stream counts,
// >= 2z steps each (M4 criterion: the M3.5 test set on the GPU). ---

TEST(CudaConveyor, AntiDeadlockOddEvenRings) {
  core::Box box;
  auto init = make_fcc(box, 2, 2, 6, /*pz=*/false);
  const auto lj = make_lj(0.4, 2.55, kRcut);
  for (int z = 1; z <= 5; ++z) {
    const long steps = 2 * z + 3;
    core::AtomSoA<double> a = init;
    auto r = cuda::run_conveyor_gpu(a, box, kRcut, lj,
                                    opts_fixed(steps, 4, z, 0.002));
    EXPECT_EQ(r.halt, core::Halt::None) << "z=" << z << ": " << r.halt_msg;
    EXPECT_EQ(r.steps_done, steps) << "z=" << z;
  }
}

// --- NVE invariant on the GPU (M3.5 methodology): 50k steps, dt=1 fs,
// T=300 K; multizone ring bitwise == single-zone (B1 on the device); secular
// trend in kT/(ns·dof) under the CPU-calibrated ceiling (the floor is the
// shift-truncation force discontinuity, not the conveyor). ---

TEST(CudaConveyor, NveInvariant50kOnGpu) {
  core::Box box;
  core::AtomSoA<double> init;
  ASSERT_TRUE(io::read_lammps_data(
      project_root() + "/reference_data/al_fcc_72.data", init, box));
  box.periodic = {false, false, false};
  core::thermal::maxwell_init(init, 300.0, 1);
  const auto p0 = core::thermal::momentum(init);
  const auto mo = make_morse(kRcut);

  auto o = opts_fixed(50000, 1, 1, 0.001);
  core::ConveyorResult r1, r2;
  auto a1 = run_gpu(init, box, o, mo, &r1);
  o.n_zones = 2;
  o.n_nodes = 2;
  auto a2 = run_gpu(init, box, o, mo, &r2);

  EXPECT_TRUE(bitwise_eq(a1, a2));  // multizone == single-zone, bitwise
  ASSERT_EQ(r1.stats.size(), r2.stats.size());
  for (std::size_t i = 0; i < r1.stats.size(); ++i)
    ASSERT_EQ(r1.stats[i].pe, r2.stats[i].pe) << "pass " << i + 1;

  // secular trend (LSQ over E(t)), kT/(ns·dof)
  const std::size_t n = r1.stats.size();
  double tm = 0.0, em = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    tm += double(i + 1) * 0.001;
    em += r1.stats[i].pe + r1.stats[i].ke;
  }
  tm /= double(n);
  em /= double(n);
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double dt_i = double(i + 1) * 0.001 - tm;
    num += dt_i * (r1.stats[i].pe + r1.stats[i].ke - em);
    den += dt_i * dt_i;
  }
  const int dof = core::thermal::dof_thermal(init.n);
  const double trend = (num / den) * 1000.0 / (units::kB * 300.0 * dof);
  std::printf("Test_CUDA_Conveyor[nve]: secular trend %.3e kT/(ns·dof) "
              "[deterministic_fp64, morse+shift, dt=1fs, 50k, GPU ring]\n",
              trend);
  EXPECT_LT(std::fabs(trend), 3.7e-2);  // CPU-calibrated ceiling (M3.5)

  const auto p = core::thermal::momentum(a2);
  for (int d = 0; d < 3; ++d) EXPECT_NEAR(p[d], p0[d], 1e-8) << "axis " << d;
}

// --- INV-4 fires through the device reduction path ---

TEST(CudaConveyor, CausalityHaltFires) {
  core::Box box;
  box.lo = {0.0, 0.0, 0.0};
  box.hi = {40.0, 12.0, 12.0};
  box.periodic = {false, false, false};
  core::AtomSoA<double> a;
  a.resize(2);
  a.x[0] = 17.9;
  a.x[1] = 22.1;
  a.y[0] = a.y[1] = 6.0;
  a.z[0] = a.z[1] = 6.0;
  a.vx[0] = 20.0;
  a.vx[1] = -20.0;
  a.type = {1, 1};
  a.mass = {1.0, 1.0};

  const auto mo = make_morse(kRcut);
  auto o = opts_fixed(10, 1, 1, 0.02);
  auto r = cuda::run_conveyor_gpu(a, box, kRcut, mo, o);
  EXPECT_EQ(r.halt, core::Halt::Causality) << r.halt_msg;
}
