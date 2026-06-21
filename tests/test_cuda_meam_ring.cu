// MEAM-ladder Me5: live GPU MEAM ring — GpuMeamRing = MeamRing<double, GpuMeamWinForce<double>>.
// The GPU policy drops into the PROVEN MeamRing orchestration (Me3b) via the policy-injected ctor;
// the inverted [Density,Embedding,Force(needs_transpose)] firewall accepts the screening transpose.
// CPU↔GPU is TOLERANCE (exp/log/pow); GPU-INTERNAL (1-vs-z, run-to-run) is BITWISE (B1 + the host
// canonical sort). Gates:
//   G9  ⭐ ring(z=1) ≡ serial-VV(meam_zone_pass-on-GPU) GPU-internal bitwise (slab + diamond)
//   G10 ⭐ 1-vs-z BITWISE (z∈{2,3} ≡ z=1, free + PBC, slab) + per-pass PE + run-to-run
//   G11 ⭐ FP64-oracle trajectory (multi-node GpuMeamRing ≡ meam_direct_fp64 serial-VV, slab)
//   G12 firewall ACCEPTS [Density,Embedding,Force]; rejects Tersoff/EAM-symmetric/iterative; seam-live
//   G13 anti-deadlock z=1..5
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "tdmd/core/integrator.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/thermal.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/meam_window_force_gpu.cuh"
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/gen/partial_screen_slab.hpp"
#include "tdmd/potentials/meam.hpp"
#include "tdmd/potentials/meam_ring.hpp"
#include "tdmd/units.hpp"

namespace core = tdmd::core;
namespace pot = tdmd::potentials;
namespace tdcu = tdmd::cuda;

namespace {
core::ConveyorOptions ropt(long steps, int n_zones, int n_nodes, double dt) {
  core::ConveyorOptions o;
  o.steps = steps; o.n_zones = n_zones; o.n_nodes = n_nodes;
  o.auto_step = false; o.dt_initial = dt; o.reach_mult = 2; o.symmetric_reach = true;
  return o;
}
core::ConveyorResult gpu_run(core::AtomSoA<double>& a, const core::Box& box,
                             const pot::MeamParams& p, const core::ConveyorOptions& o) {
  pot::MeamPotential<double> mpot(p);
  return pot::run_meam_ring(a, box, mpot, o, tdcu::GpuMeamWinForce<double>(p, box));
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
core::AtomSoA<double> slab(core::Box& box, bool pbc, unsigned seed) {
  auto s = tdmd::gen::make_partial_screen_slab(8); box = s.box; box.periodic = {true, true, pbc};
  core::AtomSoA<double> a = s.atoms;
  core::thermal::maxwell_init(a, 150.0, seed); core::thermal::zero_momentum(a);
  return a;
}
core::AtomSoA<double> dia(core::Box& box, bool pbc, unsigned seed) {
  auto a = tdmd::gen::make_diamond_si(2, 2, 12, 5.431, 0.15, box); box.periodic = {true, true, pbc};
  core::thermal::maxwell_init(a, 300.0, seed); core::thermal::zero_momentum(a);
  return a;
}
}  // namespace

// G9 — GPU ring(z=1) matches the CPU ring (the PROVEN bitwise-to-serial-meam_zone_pass oracle)
// within the CPU↔GPU TOLERANCE band (NOT bitwise — exp/log/pow ~1 ulp + the FP density/screening
// sums). MEASURED ~9e-14 on the dense diamond (the per-zone windows are identical on both sides;
// the residual is the transcendental ulp). This anchors the GPU ring to the serial oracle through
// the CPU ring; GPU-INTERNAL bitwise (1-vs-z, run-to-run) is G10. Slab + diamond, free + PBC.
TEST(CudaMeamRing, SingleNodeMatchesCpuRing) {
  pot::MeamParams p; const double dt = 0.0003; const long steps = 5;
  for (int which = 0; which < 2; ++which)
    for (bool pbc : {false, true}) {
      core::Box box; core::AtomSoA<double> init = which == 0 ? slab(box, pbc, 11) : dia(box, pbc, 11);
      pot::MeamPotential<double> mpot(p);
      core::AtomSoA<double> cpu = init, gpu = init;
      pot::run_meam_ring(cpu, box, mpot, ropt(steps, 6, 1, dt));          // default CPU MeamWinForce
      gpu_run(gpu, box, p, ropt(steps, 6, 1, dt));
      EXPECT_LT(state_dev(gpu, cpu), 1e-6)
          << "GPU ring(z=1) outside the CPU exp/log tolerance band (which=" << which << " pbc=" << pbc << ")";
    }
}

// G10 ⭐ — GPU-INTERNAL 1-vs-z BITWISE (z∈{2,3} ≡ z=1, free + PBC) + per-pass PE + run-to-run, slab.
TEST(CudaMeamRing, OneVsZBitwise) {
  pot::MeamParams p; const double dt = 0.0003; const long steps = 8;
  for (bool pbc : {false, true}) {
    core::Box box; core::AtomSoA<double> init = slab(box, pbc, 7);
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
  // run-to-run
  core::Box box; core::AtomSoA<double> init = slab(box, false, 99);
  core::AtomSoA<double> a = init, b = init;
  gpu_run(a, box, p, ropt(8, 6, 3, dt)); gpu_run(b, box, p, ropt(8, 6, 3, dt));
  EXPECT_TRUE(state_eq(a, b)) << "GPU MEAM ring not run-to-run bitwise";
}

// G11 ⭐ — multi-node GpuMeamRing matches the INDEPENDENT meam_direct_fp64-driven serial VV to
// round-off (the screening-completeness on the ring — a dropped cyclic-window screening-k is blind
// to 1-vs-z). Slab, free + PBC.
TEST(CudaMeamRing, MatchesFp64OracleTrajectory) {
  pot::MeamParams p; const double dt = 0.0003; const long steps = 3;
  for (bool pbc : {false, true}) {
    core::Box box; core::AtomSoA<double> init = slab(box, pbc, 2024);
    const core::PairGeom geom(box, p.rc);
    core::AtomSoA<double> oref = init;
    serial_vv(oref, steps, dt, [&](core::AtomSoA<double>& a){ pot::meam_direct_fp64(a, geom, p, true); });
    core::AtomSoA<double> g = init;
    gpu_run(g, box, p, ropt(steps, 6, 3, dt));
    EXPECT_LT(state_dev(g, oref), 1e-6) << "GPU ring ≠ FP64-oracle trajectory (pbc=" << pbc << ")";
  }
}

// G12 — the inverted [Density,Embedding,Force(needs_transpose)] firewall ACCEPTS MEAM, rejects
// Tersoff's [BondOrder,Force], the EAM symmetric-Force trap, and iterative; seam-is-live.
TEST(CudaMeamRing, FirewallAcceptsScreeningTranspose) {
  using pot::PassDecl; using pot::PassKind;
  pot::MeamPotential<double> mpot{};
  EXPECT_NO_THROW(tdcu::GpuMeamWinForce<double>::assert_supported(mpot.passes()));
  static_assert(requires { tdcu::GpuMeamWinForce<double>::assert_supported(std::span<const pot::PassDecl>{}); },
                "firewall seam must be live");
  const PassDecl ters2[2] = {{PassKind::BondOrder,true,false,false,0},{PassKind::Force,true,true,false,40}};
  EXPECT_THROW(tdcu::GpuMeamWinForce<double>::assert_supported(ters2), std::runtime_error);
  const PassDecl eam_sym[3] = {{PassKind::Density,true,false,false,44},{PassKind::Embedding,false,false,false,30},{PassKind::Force,true,false,false,40}};
  EXPECT_THROW(tdcu::GpuMeamWinForce<double>::assert_supported(eam_sym), std::runtime_error);
  PassDecl it[3] = {{PassKind::Density,true,false,false,44},{PassKind::Embedding,false,false,false,30},{PassKind::Force,true,true,false,40}};
  it[0].iterative = true;
  EXPECT_THROW(tdcu::GpuMeamWinForce<double>::assert_supported(it), std::runtime_error);
}

// G13 — anti-deadlock across node counts.
TEST(CudaMeamRing, AntiDeadlock) {
  pot::MeamParams p;
  core::Box box; core::AtomSoA<double> init = slab(box, false, 3);
  for (int z = 1; z <= 5; ++z) {
    core::AtomSoA<double> a = init;
    const auto r = gpu_run(a, box, p, ropt(2 * z + 3, 6, z, 0.0003));
    EXPECT_EQ(int(r.halt), int(core::Halt::None)) << "halt z=" << z;
    EXPECT_EQ(r.steps_done, 2 * z + 3) << "incomplete z=" << z;
  }
}
