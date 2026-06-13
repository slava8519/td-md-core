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

// CUB/libcu++ exposes a global ::cuda namespace, and nvcc-generated host
// stubs reference cuda::std unqualified — `using namespace tdmd` would make
// `cuda` ambiguous there. Targeted aliases instead:
namespace core = tdmd::core;
namespace potentials = tdmd::potentials;
namespace io = tdmd::io;
namespace units = tdmd::units;
namespace tdcu = tdmd::cuda;

static std::string project_root() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT);
#else
  return ".";
#endif
}

namespace {

using tdcu::LJDev;
using tdcu::MorseDev;

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
  bad += cmp(a.fx, b.fx, "fx ");
  bad += cmp(a.fy, b.fy, "fy ");
  bad += cmp(a.fz, b.fz, "fz ");
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
  auto r = tdcu::run_conveyor_gpu(a, box, kRcut, p, o);
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

// --- PR-1b-i: Verlet-list reuse path == cell path bitwise at K=1 ---
// verlet_reuse on, no criterion yet => the list is materialised EVERY pass
// (K=1), so the trajectory must equal the cell path bit-for-bit. Exercises the
// ring integration (per-launch materialise/force, the rcut+skin grid, the
// closure/rotation roles, co-residency) and 1-vs-z for the Verlet path. The
// K>1 reuse + criterion + Physical Oracle land in PR-1b-ii.

TEST(CudaConveyor, VerletReuseK1BitwiseVsCells) {
  const auto lj = make_lj(0.4, 2.55, kRcut);
  auto check = [&](bool pbc, bool autodt) {
    core::Box box;
    auto init = make_fcc(box, 2, 2, 6, pbc);  // w=6.075 Å > rcut+skin=5 Å
    core::thermal::maxwell_init(init, 300.0, 51);
    core::ConveyorOptions oc;
    oc.steps = 40; oc.n_zones = 4; oc.n_nodes = 1;
    oc.dt_initial = 0.001;
    if (autodt) {  // conservative auto-step (as LjAutoDtMatchesCpuConveyor) —
      oc.auto_step = true; oc.ts.C1 = 0.01; oc.ts.C3 = 1.0;  // else INV-4 trips
    }
    const auto cells = run_gpu(init, box, oc, lj);  // cell path, z=1
    for (int z : {1, 2, 3}) {  // verlet path == cells AND z-independent
      auto ov = oc;
      ov.n_nodes = z;
      ov.verlet_reuse = true;
      ov.verlet_skin = 1.0;
      const auto verlet = run_gpu(init, box, ov, lj);
      EXPECT_TRUE(bitwise_eq(cells, verlet))
          << "pbc=" << pbc << " auto=" << autodt << " z=" << z;
    }
  };
  check(false, false);
  check(true, false);
  check(true, true);
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

  for (int z : {2, 3, 4}) {  // even AND odd stream counts under rotation
    auto oz = o;
    oz.n_nodes = z;
    auto gpuz = run_gpu(init, box, oz, lj);
    EXPECT_TRUE(bitwise_eq(gpu, gpuz)) << "z=" << z;
  }
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
tdcu::LJDevF32 make_lj_f32(float eps, float sigma, double rcut) {
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

  // auto dt through the mixed transport: the Λ-chain decides dt from END
  // reductions over UNPACKED int32 state — the regime where the
  // snap-once-per-pass z-independence argument has teeth (review M5a)
  auto oa = o;
  oa.steps = 80;
  oa.auto_step = true;
  oa.dt_initial = 0.001;
  oa.ts.C1 = 0.01;
  oa.ts.C3 = 1.0;
  core::ConveyorResult ra1, ra3;
  auto ga1 = run_gpu(init, box, oa, lj32, &ra1);
  oa.n_nodes = 3;
  auto ga3 = run_gpu(init, box, oa, lj32, &ra3);
  EXPECT_TRUE(bitwise_eq(ga1, ga3));
  for (std::size_t i = 0; i < ra1.stats.size(); ++i)
    ASSERT_EQ(ra1.stats[i].dt, ra3.stats[i].dt) << "mixed auto dt, pass " << i + 1;
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
  auto r = tdcu::run_conveyor_gpu(a, box, kRcut, make_lj_f32(0.4f, 2.55f, kRcut), o);
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
    auto r = tdcu::run_conveyor_gpu(a, box, kRcut, lj,
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
  std::vector<double> tax(n);  // cumulative time axis — survives auto-dt
  double tcum = 0.0, tm = 0.0, em = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    tcum += r1.stats[i].dt;
    tax[i] = tcum;
    tm += tax[i];
    em += r1.stats[i].pe + r1.stats[i].ke;
  }
  tm /= double(n);
  em /= double(n);
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double dt_i = tax[i] - tm;
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

// --- cell-list culling (zone_cells.cuh): bitwise A/B vs the tile path on
// REAL grids (>= 3 cells per dim — the small fixtures above degenerate to
// 1-cell dims). The culled path must be bit-identical: same acceptance
// predicate, order-free integer sums (B1). Covers the slab grid, the
// periodic-z whole-box grid (n_zones=1 + PBC z), and the rotation closure
// with nearest-image query folds. ---

TEST(CudaConveyor, CellListsBitwiseVsTiles) {
  core::Box box;
  auto init = make_fcc(box, 6, 6, 6, /*pz=*/true);  // 864 atoms, 6 cells/dim
  core::thermal::maxwell_init(init, 300.0, 83);

  auto run_ab = [&](const core::Box& bx, const core::AtomSoA<double>& in,
                    int n_zones, int z, const char* label) {
    auto o = opts_fixed(30, n_zones, z, 0.002);
    o.cell_lists = false;
    core::ConveyorResult rt, rc;
    auto tiles = run_gpu(in, bx, o, make_lj(0.4, 2.55, kRcut), &rt);
    o.cell_lists = true;
    auto cells = run_gpu(in, bx, o, make_lj(0.4, 2.55, kRcut), &rc);
    EXPECT_TRUE(bitwise_eq(tiles, cells)) << label;
    ASSERT_EQ(rt.stats.size(), rc.stats.size());
    for (std::size_t i = 0; i < rt.stats.size(); ++i) {
      ASSERT_EQ(rt.stats[i].pe, rc.stats[i].pe) << label << " pass " << i + 1;
      ASSERT_EQ(rt.stats[i].v_max, rc.stats[i].v_max) << label << " pass "
                                                      << i + 1;
    }
  };
  run_ab(box, init, 4, 2, "pbc n=4");   // slab grids + rotation closure
  run_ab(box, init, 1, 1, "pbc n=1");   // periodic-z whole-box grid (wrapz)
  core::Box fb = box;
  fb.periodic = {true, true, false};
  run_ab(fb, init, 4, 3, "free n=4");   // free-z slabs, outer zones unbounded
  run_ab(fb, init, 6, 2, "free n=6");   // width 4.05 ~ rcut: dense closure

  // adversarial geometry fuzz: all-free box (clamping on every axis), 3
  // zones, jittered lattice + outliers flung beyond the OUTER faces (the
  // unbounded sides of edge zones — no StaleZone) — edge-cell clamping must
  // not lose a single candidate pair vs the tile oracle
  for (uint64_t seed : {101ull, 202ull, 303ull}) {
    core::Box ob;
    auto oa = make_fcc(ob, 5, 5, 6, /*pz=*/false);
    ob.periodic = {false, false, false};
    core::thermal::SplitMix64 rng(seed);
    for (int i = 0; i < oa.n; ++i) {  // jitter breaks lattice symmetry
      oa.x[i] += 0.3 * (2.0 * rng.uniform() - 1.0);
      oa.y[i] += 0.3 * (2.0 * rng.uniform() - 1.0);
      oa.z[i] += 0.3 * (2.0 * rng.uniform() - 1.0);
    }
    for (int i = 0; i < 8; ++i) {     // outliers beyond free faces
      const int j = int(rng.uniform() * oa.n);
      oa.x[j] += (i % 2 ? 25.0 : -25.0);
      if (i < 4) oa.z[j] += (i % 2 ? 30.0 : -30.0);
    }
    run_ab(ob, oa, 3, 2, "fuzz free outliers");
  }

  // empty zones on the culled path (build/scan/query of zero atoms,
  // including the rotated head position carrying the dt header)
  {
    core::Box eb;
    eb.lo = {0.0, 0.0, 0.0};
    eb.hi = {12.0, 12.0, 15.0};
    eb.periodic = {true, true, true};
    core::AtomSoA<double> ea;
    ea.resize(4);
    const double pos[4][3] = {
        {3.0, 3.0, 2.0}, {6.0, 6.0, 2.5}, {9.0, 9.0, 3.0}, {4.0, 8.0, 2.2}};
    for (int i = 0; i < 4; ++i) {
      ea.x[i] = pos[i][0]; ea.y[i] = pos[i][1]; ea.z[i] = pos[i][2];
      ea.type[i] = 1;
      ea.mass[i] = 26.9815;
    }
    run_ab(eb, ea, 3, 2, "empty zones pbc");
  }

  // mixed transport over the culled path
  auto om = opts_fixed(30, 4, 2, 0.002);
  om.mixed_transport = true;
  om.cell_lists = false;
  auto mt = run_gpu(init, box, om, make_lj_f32(0.4f, 2.55f, kRcut));
  om.cell_lists = true;
  auto mc = run_gpu(init, box, om, make_lj_f32(0.4f, 2.55f, kRcut));
  EXPECT_TRUE(bitwise_eq(mt, mc));
}

// --- INV-7 slot-pool reduction: n_zones = 16 > S = 8 (z >= 2) with the
// streamed P1 preload. The CPU ring keeps its full pool — bitwise equality
// proves the pool size and upload order are protocol-invisible. z = 1 takes
// the self-loop path (S = n+2, single pool) with the same streamed preload. ---

TEST(CudaConveyor, SlotPoolReductionBitwise) {
  core::Box box;
  auto init = make_fcc(box, 2, 2, 32, /*pz=*/false);  // 512 atoms, 16 slabs
  core::thermal::maxwell_init(init, 300.0, 89);
  const auto lj = make_lj(0.4, 2.55, kRcut);
  const auto o1 = opts_fixed(40, 16, 1, 0.002);

  auto cpu = run_cpu(init, box, o1, lj);
  for (int z : {1, 2, 3}) {
    auto oz = o1;
    oz.n_nodes = z;
    auto gpu = run_gpu(init, box, oz, lj);
    EXPECT_TRUE(bitwise_eq(cpu, gpu)) << "z=" << z;
  }

  core::Box pb = box;             // PBC closure with the reduced pool
  pb.periodic = {true, true, true};
  auto pinit = make_fcc(pb, 2, 2, 32, /*pz=*/true);
  core::thermal::maxwell_init(pinit, 300.0, 97);
  auto pcpu = run_cpu(pinit, pb, o1, lj);
  for (int z : {2, 3}) {
    auto oz = o1;
    oz.n_nodes = z;
    auto pgpu = run_gpu(pinit, pb, oz, lj);
    EXPECT_TRUE(bitwise_eq(pcpu, pgpu)) << "pbc z=" << z;
  }
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
  auto r = tdcu::run_conveyor_gpu(a, box, kRcut, mo, o);
  EXPECT_EQ(r.halt, core::Halt::Causality) << r.halt_msg;
}
