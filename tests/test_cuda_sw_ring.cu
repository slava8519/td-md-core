// M6 / SW-ladder T5: live GPU SW ring — GpuSwRing = SwRing<double, GpuSwWinForce<double>>.
// The GpuSwWinForce policy drops into the PROVEN SwRing orchestration (T3b) via the policy-
// injected ctor; the inverted firewall accepts φ3 needs_transpose. CPU↔GPU is TOLERANCE
// (exp/pow); GPU-INTERNAL (1-vs-z, run-to-run) is BITWISE. Gates:
//   G8  1-vs-z BITWISE (GPU z>1 ≡ z=1, free + PBC) + per-pass PE + run-to-run
//   G9  CPU↔GPU ring TOLERANCE (GpuSwRing vs CPU SwRing trajectory < 1e-9)
//   G10 ⭐ FP64-oracle trajectory (multi-node GpuSwRing ≡ sw_direct_fp64 serial-VV, free+PBC)
//   G11 firewall ACCEPTS SW, rejects bad descriptors; seam-is-live
//   G12 anti-deadlock z=1..5
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "tdmd/core/integrator.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/thermal.hpp"  // maxwell_init, zero_momentum
#include "tdmd/cuda/sw_window_force_gpu.cuh"
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/potentials/sw.hpp"
#include "tdmd/potentials/sw_ring.hpp"
#include "tdmd/units.hpp"

namespace core = tdmd::core;
namespace pot = tdmd::potentials;
namespace cuda = tdmd::cuda;

namespace {
constexpr int kNz = 12;
core::ConveyorOptions ropt(long steps, int n_zones, int n_nodes, double dt) {
  core::ConveyorOptions o;
  o.steps = steps; o.n_zones = n_zones; o.n_nodes = n_nodes;
  o.auto_step = false; o.dt_initial = dt; o.reach_mult = 2; o.symmetric_reach = true;
  return o;
}
core::ConveyorResult gpu_run(core::AtomSoA<double>& a, const core::Box& box,
                             const pot::SwParams& sp, const core::ConveyorOptions& o) {
  pot::SwPotential<double> spot(sp);
  return pot::run_sw_ring(a, box, spot, o, cuda::GpuSwWinForce<double>(sp, box));
}
bool state_eq(const core::AtomSoA<double>& a, const core::AtomSoA<double>& b) {
  for (int i = 0; i < a.n; ++i)
    if (a.x[i] != b.x[i] || a.y[i] != b.y[i] || a.z[i] != b.z[i] ||
        a.vx[i] != b.vx[i] || a.vy[i] != b.vy[i] || a.vz[i] != b.vz[i]) return false;
  return true;
}
double state_dev(const core::AtomSoA<double>& a, const core::AtomSoA<double>& b) {
  double m = 0; for (int i = 0; i < a.n; ++i)
    m = std::max({m, std::fabs(a.x[i]-b.x[i]), std::fabs(a.y[i]-b.y[i]), std::fabs(a.z[i]-b.z[i])});
  return m;
}
template <typename F>
void serial_vv(core::AtomSoA<double>& a, long steps, double dt, F force) {
  core::zero_forces(a); force(a);
  for (long h = 0; h < steps; ++h) {
    for (int i = 0; i < a.n; ++i) { double im = tdmd::units::ftm2v / a.mass[i];
      a.vx[i]+=0.5*dt*im*a.fx[i]; a.vy[i]+=0.5*dt*im*a.fy[i]; a.vz[i]+=0.5*dt*im*a.fz[i];
      a.x[i]+=dt*a.vx[i]; a.y[i]+=dt*a.vy[i]; a.z[i]+=dt*a.vz[i]; }
    core::zero_forces(a); force(a);
    for (int i = 0; i < a.n; ++i) { double im = tdmd::units::ftm2v / a.mass[i];
      a.vx[i]+=0.5*dt*im*a.fx[i]; a.vy[i]+=0.5*dt*im*a.fy[i]; a.vz[i]+=0.5*dt*im*a.fz[i]; }
  }
}
}  // namespace

// G8 — GPU-INTERNAL 1-vs-z BITWISE (z>1 ≡ z=1) + per-pass PE + run-to-run, free + PBC.
TEST(CudaSwRing, OneVsZBitwise) {
  pot::SwParams sp;
  for (bool pbc : {false, true}) {
    core::Box box; auto init = tdmd::gen::make_diamond_si(2, 2, kNz, 5.431, 0.0, box);
    box.periodic = {true, true, pbc};
    core::thermal::maxwell_init(init, 300.0, 12345); core::thermal::zero_momentum(init);
    const long steps = 30; const double dt = 0.0005;
    const std::vector<int> zs = pbc ? std::vector<int>{2, 3} : std::vector<int>{2, 3};

    core::AtomSoA<double> ref = init;
    const auto r1 = gpu_run(ref, box, sp, ropt(steps, 6, 1, dt));
    for (int z : zs) {
      core::AtomSoA<double> a = init;
      const auto rz = gpu_run(a, box, sp, ropt(steps, 6, z, dt));
      EXPECT_TRUE(state_eq(a, ref)) << "GPU 1-vs-z (pbc=" << pbc << ") z=" << z;
      for (std::size_t h = 0; h < r1.stats.size(); ++h)
        EXPECT_EQ(rz.stats[h].pe, r1.stats[h].pe) << "per-pass PE z=" << z << " h=" << h;
    }
  }
  // run-to-run bitwise.
  core::Box box; auto init = tdmd::gen::make_diamond_si(2, 2, kNz, 5.431, 0.0, box);
  box.periodic = {true, true, false}; core::thermal::maxwell_init(init, 300.0, 99); core::thermal::zero_momentum(init);
  core::AtomSoA<double> a = init, b = init;
  gpu_run(a, box, sp, ropt(30, 6, 3, 0.0005)); gpu_run(b, box, sp, ropt(30, 6, 3, 0.0005));
  EXPECT_TRUE(state_eq(a, b)) << "GPU ring not run-to-run bitwise";
}

// G9 — CPU↔GPU ring TOLERANCE: GpuSwRing trajectory vs the CPU SwRing within the exp band.
TEST(CudaSwRing, CpuGpuTolerance) {
  pot::SwParams sp; pot::SwPotential<double> spot(sp);
  core::Box box; auto init = tdmd::gen::make_diamond_si(2, 2, kNz, 5.431, 0.0, box);
  box.periodic = {true, true, false}; core::thermal::maxwell_init(init, 300.0, 7); core::thermal::zero_momentum(init);
  const long steps = 20; const double dt = 0.0005;
  core::AtomSoA<double> cpu = init, gpu = init;
  pot::run_sw_ring(cpu, box, spot, ropt(steps, 6, 3, dt));  // default CPU SwWinForce
  gpu_run(gpu, box, sp, ropt(steps, 6, 3, dt));
  EXPECT_LT(state_dev(gpu, cpu), 1e-9) << "GPU ring outside the CPU exp/pow tolerance band";
}

// G10 ⭐ — multi-node GpuSwRing matches the INDEPENDENT sw_direct_fp64-driven serial VV to
// round-off over a short run (MB2 — a dropped donor in the cyclic window is blind to 1-vs-z).
TEST(CudaSwRing, MatchesFp64OracleTrajectory) {
  pot::SwParams sp;
  for (bool pbc : {false, true}) {
    core::Box box; auto init = tdmd::gen::make_diamond_si(2, 2, kNz, 5.431, 0.0, box);
    box.periodic = {true, true, pbc};
    core::thermal::maxwell_init(init, 300.0, 2024); core::thermal::zero_momentum(init);
    const long steps = 3; const double dt = 0.0004;
    core::AtomSoA<double> oref = init;
    serial_vv(oref, steps, dt, [&](core::AtomSoA<double>& a){ pot::sw_direct_fp64(a, box, sp, true); });
    core::AtomSoA<double> g = init;
    gpu_run(g, box, sp, ropt(steps, pbc ? 6 : 5, pbc ? 3 : 4, dt));
    EXPECT_LT(state_dev(g, oref), 1e-6) << "GPU ring ≠ FP64-oracle trajectory (pbc=" << pbc << ")";
  }
}

// G11 — the inverted firewall ACCEPTS SW (φ3 needs_transpose), rejects bad descriptors.
TEST(CudaSwRing, FirewallAcceptsTranspose) {
  using pot::PassDecl; using pot::PassKind;
  pot::SwPotential<double> spot{};
  EXPECT_NO_THROW(cuda::GpuSwWinForce<double>::assert_supported(spot.passes()));
  static_assert(requires { cuda::GpuSwWinForce<double>::assert_supported(
      std::span<const pot::PassDecl>{}); }, "firewall seam must be live");
  const PassDecl eam3[3] = {{PassKind::Density,true,false,false,44},{PassKind::Embedding,false,false,false,30},{PassKind::Force,true,false,false,40}};
  EXPECT_THROW(cuda::GpuSwWinForce<double>::assert_supported(eam3), std::runtime_error);
  const PassDecl sym[2] = {{PassKind::Force,true,false,false,40},{PassKind::Force,true,false,false,40}};  // φ3 NOT transpose
  EXPECT_THROW(cuda::GpuSwWinForce<double>::assert_supported(sym), std::runtime_error);
}

// G12 — anti-deadlock across node counts.
TEST(CudaSwRing, AntiDeadlock) {
  pot::SwParams sp;
  core::Box box; auto init = tdmd::gen::make_diamond_si(2, 2, kNz, 5.431, 0.0, box);
  box.periodic = {true, true, false}; core::thermal::maxwell_init(init, 300.0, 3); core::thermal::zero_momentum(init);
  for (int z = 1; z <= 5; ++z) {
    core::AtomSoA<double> a = init;
    const auto r = gpu_run(a, box, sp, ropt(2 * z + 3, 6, z, 0.0004));
    EXPECT_EQ(int(r.halt), int(core::Halt::None)) << "halt z=" << z;
    EXPECT_EQ(r.steps_done, 2 * z + 3) << "incomplete z=" << z;
  }
}
