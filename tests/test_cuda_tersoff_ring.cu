// M6 / Tersoff-ladder Te5: live GPU Tersoff ring — GpuTersoffRing = TersoffRing<double,
// GpuTersoffWinForce<double>>. The GPU policy drops into the PROVEN TersoffRing orchestration
// (Te3b) via the policy-injected ctor; the inverted [BondOrder,Force] firewall accepts the
// transpose. CPU↔GPU is TOLERANCE (exp/pow/sin); GPU-INTERNAL (1-vs-z, run-to-run) is BITWISE
// (B1 + the host canonical ζ-sort). Gates:
//   G9  ⭐ 1-vs-z BITWISE (GPU z>1 ≡ z=1, free + PBC) + per-pass PE + run-to-run — THE canonical-ζ gate
//   G10 CPU↔GPU ring TOLERANCE (GpuTersoffRing vs CPU TersoffRing < 1e-9)
//   G11 ⭐ FP64-oracle trajectory (multi-node GpuTersoffRing ≡ tersoff_direct_fp64 serial-VV)
//   G12 firewall ACCEPTS [BondOrder,Force], rejects [Force,Force]/EAM-3/symmetric-trap; seam-live
//   G13 anti-deadlock z=1..5
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "tdmd/core/integrator.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/thermal.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/tersoff_window_force_gpu.cuh"
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/potentials/tersoff.hpp"
#include "tdmd/potentials/tersoff_ring.hpp"
#include "tdmd/units.hpp"

namespace core = tdmd::core;
namespace pot = tdmd::potentials;
namespace tdcu = tdmd::cuda;

namespace {
constexpr int kNz = 12;
core::ConveyorOptions ropt(long steps, int n_zones, int n_nodes, double dt) {
  core::ConveyorOptions o;
  o.steps = steps; o.n_zones = n_zones; o.n_nodes = n_nodes;
  o.auto_step = false; o.dt_initial = dt; o.reach_mult = 2; o.symmetric_reach = true;
  return o;
}
core::ConveyorResult gpu_run(core::AtomSoA<double>& a, const core::Box& box,
                             const pot::TersoffParams& p, const core::ConveyorOptions& o) {
  pot::TersoffPotential<double> tpot(p);
  return pot::run_tersoff_ring(a, box, tpot, o, tdcu::GpuTersoffWinForce<double>(p, box));
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

// G9 ⭐ — GPU-INTERNAL 1-vs-z BITWISE (z>1 ≡ z=1) + per-pass PE + run-to-run, free + PBC. THE
// canonical-ζ gate at ring scale: without the host sort, 1-vs-z diverges in the velocity deep bits.
TEST(CudaTersoffRing, OneVsZBitwise) {
  pot::TersoffParams p;
  for (bool pbc : {false, true}) {
    core::Box box; auto init = tdmd::gen::make_diamond_si(2, 2, kNz, 5.431, 0.0, box);
    box.periodic = {true, true, pbc};
    core::thermal::maxwell_init(init, 300.0, 12345); core::thermal::zero_momentum(init);
    const long steps = 30; const double dt = 0.0005;

    core::AtomSoA<double> ref = init;
    const auto r1 = gpu_run(ref, box, p, ropt(steps, 6, 1, dt));
    for (int z : {2, 3}) {
      core::AtomSoA<double> a = init;
      const auto rz = gpu_run(a, box, p, ropt(steps, 6, z, dt));
      EXPECT_TRUE(state_eq(a, ref)) << "GPU 1-vs-z (pbc=" << pbc << ") z=" << z;
      for (std::size_t h = 0; h < r1.stats.size(); ++h)
        EXPECT_EQ(rz.stats[h].pe, r1.stats[h].pe) << "per-pass PE z=" << z << " h=" << h;
    }
  }
  core::Box box; auto init = tdmd::gen::make_diamond_si(2, 2, kNz, 5.431, 0.0, box);
  box.periodic = {true, true, false}; core::thermal::maxwell_init(init, 300.0, 99); core::thermal::zero_momentum(init);
  core::AtomSoA<double> a = init, b = init;
  gpu_run(a, box, p, ropt(30, 6, 3, 0.0005)); gpu_run(b, box, p, ropt(30, 6, 3, 0.0005));
  EXPECT_TRUE(state_eq(a, b)) << "GPU Tersoff ring not run-to-run bitwise";
}

// G10 — CPU↔GPU ring TOLERANCE: GpuTersoffRing trajectory vs the CPU TersoffRing within the exp band.
TEST(CudaTersoffRing, CpuGpuTolerance) {
  pot::TersoffParams p; pot::TersoffPotential<double> tpot(p);
  core::Box box; auto init = tdmd::gen::make_diamond_si(2, 2, kNz, 5.431, 0.0, box);
  box.periodic = {true, true, false}; core::thermal::maxwell_init(init, 300.0, 7); core::thermal::zero_momentum(init);
  const long steps = 20; const double dt = 0.0005;
  core::AtomSoA<double> cpu = init, gpu = init;
  pot::run_tersoff_ring(cpu, box, tpot, ropt(steps, 6, 3, dt));  // default CPU TersoffWinForce
  gpu_run(gpu, box, p, ropt(steps, 6, 3, dt));
  EXPECT_LT(state_dev(gpu, cpu), 1e-9) << "GPU ring outside the CPU exp/pow tolerance band";
}

// G11 ⭐ — multi-node GpuTersoffRing matches the INDEPENDENT tersoff_direct_fp64-driven serial VV
// to round-off (MB2 — a dropped donor in the cyclic window is blind to 1-vs-z).
TEST(CudaTersoffRing, MatchesFp64OracleTrajectory) {
  pot::TersoffParams p;
  for (bool pbc : {false, true}) {
    core::Box box; auto init = tdmd::gen::make_diamond_si(2, 2, kNz, 5.431, 0.0, box);
    box.periodic = {true, true, pbc};
    core::thermal::maxwell_init(init, 300.0, 2024); core::thermal::zero_momentum(init);
    const long steps = 3; const double dt = 0.0004;
    const core::PairGeom geom(box, p.rcut());
    core::AtomSoA<double> oref = init;
    serial_vv(oref, steps, dt, [&](core::AtomSoA<double>& a){ pot::tersoff_direct_fp64(a, geom, p, true); });
    core::AtomSoA<double> g = init;
    gpu_run(g, box, p, ropt(steps, pbc ? 6 : 5, pbc ? 3 : 4, dt));
    EXPECT_LT(state_dev(g, oref), 1e-6) << "GPU ring ≠ FP64-oracle trajectory (pbc=" << pbc << ")";
  }
}

// G12 ⭐ — the inverted [BondOrder,Force] firewall ACCEPTS Tersoff, rejects SW's [Force,Force]
// (proving it is NOT a GpuSwWinForce copy), the EAM 3-pass, and the symmetric trap; seam-is-live.
TEST(CudaTersoffRing, FirewallAcceptsBondOrder) {
  using pot::PassDecl; using pot::PassKind;
  pot::TersoffPotential<double> tpot{};
  EXPECT_NO_THROW(tdcu::GpuTersoffWinForce<double>::assert_supported(tpot.passes()));
  static_assert(requires { tdcu::GpuTersoffWinForce<double>::assert_supported(
      std::span<const pot::PassDecl>{}); }, "firewall seam must be live");
  const PassDecl sw2[2] = {{PassKind::Force,true,false,false,40},{PassKind::Force,true,true,false,40}};
  EXPECT_THROW(tdcu::GpuTersoffWinForce<double>::assert_supported(sw2), std::runtime_error);  // SW desc
  const PassDecl eam3[3] = {{PassKind::Density,true,false,false,44},{PassKind::Embedding,false,false,false,30},{PassKind::Force,true,false,false,40}};
  EXPECT_THROW(tdcu::GpuTersoffWinForce<double>::assert_supported(eam3), std::runtime_error);  // count
  const PassDecl sym[2] = {{PassKind::BondOrder,true,false,false,0},{PassKind::Force,true,false,false,40}};  // Force NOT transpose
  EXPECT_THROW(tdcu::GpuTersoffWinForce<double>::assert_supported(sym), std::runtime_error);  // symmetric trap
}

// G13 — anti-deadlock across node counts.
TEST(CudaTersoffRing, AntiDeadlock) {
  pot::TersoffParams p;
  core::Box box; auto init = tdmd::gen::make_diamond_si(2, 2, kNz, 5.431, 0.0, box);
  box.periodic = {true, true, false}; core::thermal::maxwell_init(init, 300.0, 3); core::thermal::zero_momentum(init);
  for (int z = 1; z <= 5; ++z) {
    core::AtomSoA<double> a = init;
    const auto r = gpu_run(a, box, p, ropt(2 * z + 3, 6, z, 0.0004));
    EXPECT_EQ(int(r.halt), int(core::Halt::None)) << "halt z=" << z;
    EXPECT_EQ(r.steps_done, 2 * z + 3) << "incomplete z=" << z;
  }
}

// G14 ⭐ — LIVE-RING cells integration (Tersoff-Te5b → ring): the cells path (now cull=true production
// default) is BITWISE to the all-window ring IN THE LIVE RING (z=1 AND 1-vs-z), free + PBC. Tersoff's
// FP64 ζ-sum needs the canonical-ζ cull ⇒ this proves the cull stays bitwise in the host ring.
TEST(CudaTersoffRing, LiveRingCellsBitwiseToAllWindow) {
  pot::TersoffParams p; pot::TersoffPotential<double> tpot(p); const double dt = 0.0004; const long steps = 8;
  for (bool pbc : {false, true}) {
    core::Box box; auto init = tdmd::gen::make_diamond_si(2, 2, kNz, 5.431, 0.0, box);
    box.periodic = {true, true, pbc}; core::thermal::maxwell_init(init, 300.0, 21); core::thermal::zero_momentum(init);
    core::AtomSoA<double> aw = init, cl = init, c2 = init;
    pot::run_tersoff_ring(aw, box, tpot, ropt(steps, 6, 1, dt), tdcu::GpuTersoffWinForce<double>(p, box, /*cull=*/false));
    pot::run_tersoff_ring(cl, box, tpot, ropt(steps, 6, 1, dt), tdcu::GpuTersoffWinForce<double>(p, box, /*cull=*/true));
    EXPECT_TRUE(state_eq(aw, cl)) << "Tersoff cells-ring ≠ all-window-ring (z=1, pbc=" << pbc << ")";
    pot::run_tersoff_ring(c2, box, tpot, ropt(steps, 6, 3, dt), tdcu::GpuTersoffWinForce<double>(p, box, /*cull=*/true));
    EXPECT_TRUE(state_eq(cl, c2)) << "Tersoff cells-ring 1-vs-z (z=3) not bitwise (pbc=" << pbc << ")";
  }
}
