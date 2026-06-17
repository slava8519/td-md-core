// M6 E5b-2 — EAM on the GPU ring, SINGLE-NODE (z=1) path: bitwise == the serial
// zone_eam_pass-driven velocity-Verlet oracle (the SingleNodeMatchesSerialVV
// gate). Validates the per-zone 3-zone window gather + the E5 kernels (reused
// verbatim) + scatter + device VV, over a multi-zone decomposition. The spline
// (EamSetfl) is the Math on BOTH sides — the GPU kernels' contract (CPU↔GPU
// bitwise was proven per-window in test_cuda_eam; here over a trajectory). The
// STREAMING multi-node ring (z>1, send-delay, the restricted-window superset
// oracles A-D) is E5b-3. Compiled with --fmad=false (tdmd_eam_cuda_flags).
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <array>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/eam_conveyor_gpu.cuh"
#include "tdmd/cuda/eam_window_force_gpu.cuh"  // E5b-3b: GPU window-force policy
#include "tdmd/potentials/eam.hpp"  // eam_direct_fp64 (independent O(N²) oracle)
#include "tdmd/potentials/eam_analytic.hpp"
#include "tdmd/potentials/eam_ring.hpp"  // E5b-3b: EamRing (orchestration oracle)
#include "tdmd/potentials/eam_spline.hpp"
#include "tdmd/potentials/eam_zone.hpp"
#include "tdmd/units.hpp"

namespace core = tdmd::core;
namespace potentials = tdmd::potentials;
namespace units = tdmd::units;
namespace tdcu = tdmd::cuda;

namespace {
constexpr double kRcut = 3.0;

core::AtomSoA<double> make_fcc(int nx, int ny, int nz, double alat, core::Box& box) {
  box.lo = {0, 0, 0};
  box.hi = {nx * alat, ny * alat, nz * alat};
  box.periodic = {true, true, false};  // free z (3-zone EAM residence)
  const double b[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  std::vector<std::array<double, 3>> pos;
  for (int ix = 0; ix < nx; ++ix)
    for (int iy = 0; iy < ny; ++iy)
      for (int iz = 0; iz < nz; ++iz)
        for (auto& bb : b)
          pos.push_back({(ix + bb[0]) * alat, (iy + bb[1]) * alat, (iz + bb[2]) * alat});
  core::AtomSoA<double> a;
  a.resize(int(pos.size()));
  for (int i = 0; i < a.n; ++i) {
    a.x[i] = pos[i][0]; a.y[i] = pos[i][1]; a.z[i] = pos[i][2];
    a.type[i] = 1; a.mass[i] = 26.98;
    a.vx[i] = 0.001 * ((i * 7) % 5 - 2);
    a.vy[i] = 0.001 * ((i * 3) % 5 - 2);
    a.vz[i] = 0.001 * ((i * 11) % 5 - 2);
  }
  return a;
}

potentials::EamSetfl<double> make_setfl() {
  potentials::AnalyticEam<double> m; m.rcut = kRcut; m.finalize();
  return potentials::EamSetfl<double>::from_analytic(m, 4000, 4000, 60.0);
}

// serial VV driven by the zone_eam_pass oracle — the SAME windows/arithmetic the
// GPU single-node path uses (just CPU + double accumulation decoded from int64).
void serial_vv(core::AtomSoA<double>& a, const core::Box& box,
               const potentials::EamPotential<double, potentials::EamSetfl<double>>& pot,
               const core::ZoneDecomposition& zd, long steps, double dt) {
  core::zero_forces(a);
  potentials::zone_eam_pass(a, box, zd, pot);
  for (long h = 0; h < steps; ++h) {
    for (int i = 0; i < a.n; ++i) {
      const double inv_m = units::ftm2v / a.mass[i];
      a.vx[i] += 0.5 * dt * inv_m * a.fx[i]; a.vy[i] += 0.5 * dt * inv_m * a.fy[i]; a.vz[i] += 0.5 * dt * inv_m * a.fz[i];
      a.x[i] += dt * a.vx[i]; a.y[i] += dt * a.vy[i]; a.z[i] += dt * a.vz[i];
    }
    core::zero_forces(a);
    potentials::zone_eam_pass(a, box, zd, pot);
    for (int i = 0; i < a.n; ++i) {
      const double inv_m = units::ftm2v / a.mass[i];
      a.vx[i] += 0.5 * dt * inv_m * a.fx[i]; a.vy[i] += 0.5 * dt * inv_m * a.fy[i]; a.vz[i] += 0.5 * dt * inv_m * a.fz[i];
    }
  }
}

::testing::AssertionResult bitwise_eq(const core::AtomSoA<double>& a,
                                      const core::AtomSoA<double>& b) {
  if (a.n != b.n) return ::testing::AssertionFailure() << "size";
  for (int i = 0; i < a.n; ++i)
    if (a.x[i] != b.x[i] || a.y[i] != b.y[i] || a.z[i] != b.z[i] ||
        a.vx[i] != b.vx[i] || a.vy[i] != b.vy[i] || a.vz[i] != b.vz[i])
      return ::testing::AssertionFailure() << "mismatch at atom " << i;
  return ::testing::AssertionSuccess();
}
}  // namespace

// The single-node (z=1) GPU EAM trajectory must equal the serial zone_eam_pass VV
// bit-for-bit — proving the gather + the 3 E5 kernels + scatter + device VV are
// the same multiset of int64 contributions as the serial oracle. A dropped donor
// in any 3-zone window would diverge here (the serial oracle is the independent
// reference 1-vs-z cannot substitute for).
TEST(CudaEamRing, SingleNodeMatchesSerialVV) {
  const auto setfl = make_setfl();
  const potentials::EamPotential<double, potentials::EamSetfl<double>> pot(setfl);

  for (int n_zones : {2, 3, 4}) {  // width = 24.3/n_zones >= 2·rcut=6 for n<=4
    core::Box box;
    auto init = make_fcc(2, 2, 6, 4.05, box);  // 96 atoms, Lz=24.3
    const auto zd = core::ZoneDecomposition::build(init, box, n_zones, kRcut, 2);

    core::AtomSoA<double> ref = init, gpu = init;
    serial_vv(ref, box, pot, zd, /*steps=*/20, /*dt=*/0.001);
    tdcu::eam_gpu_run_singlenode(gpu, box, zd, setfl, /*steps=*/20, /*dt=*/0.001);

    EXPECT_TRUE(bitwise_eq(ref, gpu)) << "n_zones=" << n_zones;
  }
}

namespace {
double max_force_dev(const core::AtomSoA<double>& ref, const std::vector<double>& gx,
                     const std::vector<double>& gy, const std::vector<double>& gz) {
  double mx = 0.0;
  for (int i = 0; i < ref.n; ++i) {
    mx = std::max(mx, std::fabs(ref.fx[i] - gx[i]));
    mx = std::max(mx, std::fabs(ref.fy[i] - gy[i]));
    mx = std::max(mx, std::fabs(ref.fz[i] - gz[i]));
  }
  return mx;
}
}  // namespace

// Independent-oracle gate (do NOT accept the EAM ring on 1-vs-z / vs serial alone,
// since both share zone_eam_window). eam_direct_fp64 is the full-system O(N²) FP64
// EAM — it shares NO window/zone logic, so it catches a dropped donor the zone
// path would otherwise reproduce deterministically.
//
// Oracle B: the symmetric 3-zone window (width≥2·rcut) is COMPLETE ⇒ GPU forces
//   match the full O(N²) oracle (not bitwise — int64 quant vs FP64 sum — but <1e-10).
// Oracle A (forward-only POISON): gathering only {j,j+1} drops the lower donor ⇒
//   ρ truncated for atoms near the lower face ⇒ forces DIVERGE. Proves the gate
//   has TEETH: a green Oracle B is only meaningful because Oracle A would fail.
TEST(CudaEamRing, SingleNodeForcesMatchFp64OracleAndForwardOnlyPoisonDiverges) {
  const auto setfl = make_setfl();
  for (int n_zones : {2, 3, 4}) {
    core::Box box;
    const auto init = make_fcc(2, 2, 6, 4.05, box);
    const auto zd = core::ZoneDecomposition::build(init, box, n_zones, kRcut, 2);

    // independent reference: full-system O(N²) FP64 EAM on the SAME spline math.
    core::AtomSoA<double> ref = init;
    core::zero_forces(ref);
    potentials::eam_direct_fp64<double, potentials::EamSetfl<double>>(ref, box, setfl, true);

    // Oracle B — correct symmetric window matches the oracle (complete window).
    core::AtomSoA<double> g = init;
    std::vector<double> bx, by, bz;
    tdcu::eam_gpu_run_singlenode(g, box, zd, setfl, /*steps=*/0, 0.001, /*symmetric=*/true, &bx, &by, &bz);
    EXPECT_LT(max_force_dev(ref, bx, by, bz), 1e-10) << "n_zones=" << n_zones << " symmetric ≠ O(N²) oracle";

    // Oracle A — forward-only poison MUST diverge (a dropped donor is detectable).
    core::AtomSoA<double> p = init;
    std::vector<double> px, py, pz;
    tdcu::eam_gpu_run_singlenode(p, box, zd, setfl, /*steps=*/0, 0.001, /*symmetric=*/false, &px, &py, &pz);
    EXPECT_GT(max_force_dev(ref, px, py, pz), 1e-6) << "n_zones=" << n_zones
        << " forward-only did NOT diverge — the gate is blind to a dropped donor!";
  }
}

// =================== E5b-3b: STREAMING multi-node (z>1) GPU EAM ring =========
// EamGpuRing = EamRing<double, EamSetfl<double>, GpuEamWindowForce>: the CPU
// EamRing orchestration (jthread/node, SPSC channels, the second-forward-hop send
// delay, Λ-chain dt, FSM/INV-4, PBC defer_head) is the PROVEN bitwise oracle; the
// ONLY swapped part is the per-window force, now the GPU E5 kernels. Because the
// int64 window-force is order-free and the gather preserves the window multiset
// (key=atom id), the GPU ring is bitwise ≡ the CPU ring BY CONSTRUCTION — and we
// ASSERT it. The independent eam_direct_fp64 oracle (no shared window/zone logic)
// is what catches a DROPPED DONOR that 1-vs-z would reproduce deterministically.
namespace {
using GpuRing = potentials::EamRing<double, potentials::EamSetfl<double>, tdcu::GpuEamWindowForce>;
using CpuRing = potentials::EamRing<double, potentials::EamSetfl<double>>;
using SetflPot = potentials::EamPotential<double, potentials::EamSetfl<double>>;

core::ConveyorOptions ring_opts(long steps, int n_zones, int n_nodes, double dt) {
  core::ConveyorOptions o;
  o.steps = steps; o.n_zones = n_zones; o.n_nodes = n_nodes;
  o.auto_step = false; o.dt_initial = dt;
  o.reach_mult = 2; o.symmetric_reach = true;
  return o;
}

// run the GPU EamRing (fresh device policy) on a copy of `init`.
core::ConveyorResult run_gpu_ring(core::AtomSoA<double>& a, const core::Box& box,
                                  const SetflPot& pot,
                                  const potentials::EamSetfl<double>& setfl,
                                  const core::ConveyorOptions& o) {
  return potentials::run_eam_ring(a, box, pot, o, tdcu::GpuEamWindowForce(setfl));
}

// PBC slab: an exact lattice period in z ⇒ seamless cyclic nn. reach_mult=2 needs
// periodic n_zones >= 5 AND width >= 2·rcut=6; with n_zones=5 we need Lz >= 30, so
// 8 z-cells (Lz=32.4, width 6.48 >= 6). 128 atoms.
core::AtomSoA<double> make_fcc_pbc(core::Box& box) {
  auto a = make_fcc(2, 2, 8, 4.05, box);  // Lz = 32.4
  box.periodic[2] = true;
  return a;
}
}  // namespace

// --- the load-bearing gate: GPU EamRing ≡ CPU EamRing BITWISE for z∈{2,3} (free-z) ---
// width = 24.3/n_zones; n_zones=5 ⇒ 4.86 >= 2·rcut=6? No: need >=6 ⇒ n_zones<=4
// for free-z. Use n_zones=4 (width 6.075 > 6). z (nodes) ∈ {2,3}.
TEST(CudaEamGpuRing, GpuRingMatchesCpuRingBitwiseFree) {
  const auto setfl = make_setfl();
  const SetflPot pot(setfl);
  const double dt = 0.001;
  const long steps = 12;
  const int n_zones = 4;  // width 6.075 Å >= 2·rcut=6

  core::Box box;
  const auto init = make_fcc(2, 2, 6, 4.05, box);  // 96 atoms, free-z

  // CPU EamRing reference (z=1) — the proven orchestration oracle.
  core::AtomSoA<double> cpu_ref = init;
  const auto rc = potentials::run_eam_ring(cpu_ref, box, pot, ring_opts(steps, n_zones, 1, dt));
  ASSERT_EQ(int(rc.halt), int(core::Halt::None)) << rc.halt_msg;

  for (int z : {2, 3}) {
    core::AtomSoA<double> g = init;
    const auto rg = run_gpu_ring(g, box, pot, setfl, ring_opts(steps, n_zones, z, dt));
    ASSERT_EQ(int(rg.halt), int(core::Halt::None)) << "z=" << z << " " << rg.halt_msg;
    EXPECT_TRUE(bitwise_eq(cpu_ref, g)) << "GPU ring z=" << z << " ≠ CPU ring (free-z)";
    // INV-9 energy: the per-pass PE (decoded from the order-free int64 EnergyAccum)
    // must be bitwise-equal too (the GPU force kernel's int64 pe ≡ the CPU
    // eam_window_force pe.add() multiset). KE follows the (bitwise-equal) velocities.
    ASSERT_EQ(rc.stats.size(), rg.stats.size());
    for (size_t h = 0; h < rc.stats.size(); ++h)
      EXPECT_EQ(rc.stats[h].pe, rg.stats[h].pe)
          << "z=" << z << " pass " << h << " PE not bitwise-equal (INV-9)";
  }
}

// --- PBC variant: GPU EamRing ≡ CPU EamRing bitwise (cyclic window + defer_head) ---
TEST(CudaEamGpuRing, GpuRingMatchesCpuRingBitwisePbc) {
  const auto setfl = make_setfl();
  const SetflPot pot(setfl);
  const double dt = 0.001;
  const long steps = 10;
  const int n_zones = 5;  // PBC reach_mult=2: >=5 distinct cyclic zones, width 6.48 >= 6

  core::Box box;
  const auto init = make_fcc_pbc(box);  // fully periodic z, Lz=32.4

  core::AtomSoA<double> cpu_ref = init;
  const auto rc = potentials::run_eam_ring(cpu_ref, box, pot, ring_opts(steps, n_zones, 1, dt));
  ASSERT_EQ(int(rc.halt), int(core::Halt::None)) << rc.halt_msg;

  for (int z : {2, 3}) {
    core::AtomSoA<double> g = init;
    const auto rg = run_gpu_ring(g, box, pot, setfl, ring_opts(steps, n_zones, z, dt));
    ASSERT_EQ(int(rg.halt), int(core::Halt::None)) << "z=" << z << " " << rg.halt_msg;
    EXPECT_TRUE(bitwise_eq(cpu_ref, g)) << "GPU ring z=" << z << " ≠ CPU ring (PBC)";
  }
}

// --- 1-vs-z: GPU EamRing z∈{2,3} ≡ GPU EamRing z=1 bitwise (inherited, asserted) ---
TEST(CudaEamGpuRing, OneVsZBitwise) {
  const auto setfl = make_setfl();
  const SetflPot pot(setfl);
  const double dt = 0.001;
  const long steps = 12;
  const int n_zones = 4;

  core::Box box;
  const auto init = make_fcc(2, 2, 6, 4.05, box);

  core::AtomSoA<double> ref = init;
  const auto r1 = run_gpu_ring(ref, box, pot, setfl, ring_opts(steps, n_zones, 1, dt));
  ASSERT_EQ(int(r1.halt), int(core::Halt::None)) << r1.halt_msg;

  for (int z : {2, 3}) {
    core::AtomSoA<double> g = init;
    const auto rz = run_gpu_ring(g, box, pot, setfl, ring_opts(steps, n_zones, z, dt));
    ASSERT_EQ(int(rz.halt), int(core::Halt::None)) << "z=" << z << " " << rz.halt_msg;
    EXPECT_TRUE(bitwise_eq(ref, g)) << "GPU ring z=" << z << " ≠ GPU ring z=1";
  }
}

// --- anti-deadlock: z=1..5 run to completion (finds the +1 slot-pool margin if
// the GPU path needs it). free-z, short runs. ---
TEST(CudaEamGpuRing, AntiDeadlock) {
  const auto setfl = make_setfl();
  const SetflPot pot(setfl);
  core::Box box;
  const auto init = make_fcc(2, 2, 6, 4.05, box);
  const int n_zones = 4;
  for (int z : {1, 2, 3, 4, 5}) {
    core::AtomSoA<double> g = init;
    const long steps = 2 * z + 3;
    const auto r = run_gpu_ring(g, box, pot, setfl, ring_opts(steps, n_zones, z, 0.001));
    EXPECT_EQ(int(r.halt), int(core::Halt::None)) << "z=" << z << " " << r.halt_msg;
    EXPECT_EQ(r.steps_done, steps) << "z=" << z << " did not run to completion";
  }
}

// --- INDEPENDENT oracle (decisive): a multi-node GPU EamRing's step-0 forces ≡
// eam_direct_fp64 (the full-system O(N²) FP64 EAM that shares NO window/zone
// logic) to <1e-10. This is what catches a dropped donor 1-vs-z cannot. We read
// the forces the ring computed at t0 by running steps=0 conceptually: the ring's
// first pass (h=1) does the t0 force via zone_eam_pass on the CPU; instead we
// take the GPU ring forces from a fresh single-pass and compare. Here we compute
// the t0 forces with the GPU single-node window path (same kernels) over the
// SAME zd the multi-node ring uses, then assert it matches the O(N²) oracle —
// and separately assert the multi-node GPU ring's trajectory ≡ the CPU ring
// (above), closing the loop: orchestration is the oracle, kernels match O(N²). ---
TEST(CudaEamGpuRing, SingleNodeStep0ForcesMatchFp64Oracle) {  // single-node window path
  const auto setfl = make_setfl();
  const int n_zones = 4;

  core::Box box;
  const auto init = make_fcc(2, 2, 6, 4.05, box);
  const auto zd = core::ZoneDecomposition::build(init, box, n_zones, kRcut, 2);

  // independent reference: full-system O(N²) FP64 EAM on the SAME spline math.
  core::AtomSoA<double> ref = init;
  core::zero_forces(ref);
  potentials::eam_direct_fp64<double, potentials::EamSetfl<double>>(ref, box, setfl, true);

  // GPU EAM forces over the SAME 3-zone-window decomposition (steps=0 → t0 forces).
  core::AtomSoA<double> g = init;
  std::vector<double> gx, gy, gz;
  tdcu::eam_gpu_run_singlenode(g, box, zd, setfl, /*steps=*/0, 0.001, /*symmetric=*/true, &gx, &gy, &gz);
  EXPECT_LT(max_force_dev(ref, gx, gy, gz), 1e-10)
      << "GPU EAM (zd reach_mult=2) ≠ O(N²) FP64 oracle";
}

namespace {
// independent FP64 serial velocity-Verlet driven by eam_direct_fp64 (the full-
// system O(N²) FP64 EAM — shares NO window/zone/ring logic). The decisive cross-
// check for the multi-node GPU ring: a dropped donor would truncate ρ on the
// GPU ring (silently, identically on every z ⇒ invisible to 1-vs-z) but the
// O(N²) oracle has the COMPLETE neighbourhood ⇒ they diverge. int64-quant vs
// FP64-sum ⇒ tolerance-equal (not bitwise), <1e-9 over a short run.
void serial_vv_fp64_oracle(core::AtomSoA<double>& a, const core::Box& box,
                           const potentials::EamSetfl<double>& setfl, long steps, double dt) {
  core::zero_forces(a);
  potentials::eam_direct_fp64<double, potentials::EamSetfl<double>>(a, box, setfl, true);
  for (long h = 0; h < steps; ++h) {
    for (int i = 0; i < a.n; ++i) {
      const double inv_m = units::ftm2v / a.mass[i];
      a.vx[i] += 0.5 * dt * inv_m * a.fx[i]; a.vy[i] += 0.5 * dt * inv_m * a.fy[i]; a.vz[i] += 0.5 * dt * inv_m * a.fz[i];
      a.x[i] += dt * a.vx[i]; a.y[i] += dt * a.vy[i]; a.z[i] += dt * a.vz[i];
    }
    core::zero_forces(a);
    potentials::eam_direct_fp64<double, potentials::EamSetfl<double>>(a, box, setfl, true);
    for (int i = 0; i < a.n; ++i) {
      const double inv_m = units::ftm2v / a.mass[i];
      a.vx[i] += 0.5 * dt * inv_m * a.fx[i]; a.vy[i] += 0.5 * dt * inv_m * a.fy[i]; a.vz[i] += 0.5 * dt * inv_m * a.fz[i];
    }
  }
}
double max_pos_dev(const core::AtomSoA<double>& a, const core::AtomSoA<double>& b) {
  double mx = 0.0;
  for (int i = 0; i < a.n; ++i) {
    mx = std::max(mx, std::fabs(a.x[i] - b.x[i]));
    mx = std::max(mx, std::fabs(a.y[i] - b.y[i]));
    mx = std::max(mx, std::fabs(a.z[i] - b.z[i]));
  }
  return mx;
}
}  // namespace

// Oracle D (decisive, independent): a MULTI-NODE GPU EamRing run ≡ the FP64 O(N²)
// serial-VV oracle, to tolerance, over several steps. The GPU ring shares NO code
// with eam_direct_fp64 ⇒ a dropped donor (invisible to 1-vs-z) would diverge here.
// Short run so atoms stay in their slabs (static membership); tolerance because
// the ring is int64-quant fixed-point vs the oracle's FP64 sum.
TEST(CudaEamGpuRing, MultiNodeTrajectoryMatchesFp64OracleTolerance) {
  const auto setfl = make_setfl();
  const SetflPot pot(setfl);
  const double dt = 0.001;
  const long steps = 8;
  const int n_zones = 4;  // free-z, width 6.075 >= 6

  core::Box box;
  const auto init = make_fcc(2, 2, 6, 4.05, box);

  core::AtomSoA<double> oracle = init;
  serial_vv_fp64_oracle(oracle, box, setfl, steps, dt);

  for (int z : {2, 3}) {
    core::AtomSoA<double> g = init;
    const auto rg = run_gpu_ring(g, box, pot, setfl, ring_opts(steps, n_zones, z, dt));
    ASSERT_EQ(int(rg.halt), int(core::Halt::None)) << "z=" << z << " " << rg.halt_msg;
    EXPECT_LT(max_pos_dev(oracle, g), 1e-9)
        << "GPU ring z=" << z << " trajectory diverges from the FP64 O(N²) oracle "
           "(possible dropped donor — invisible to 1-vs-z!)";
  }
}

// G1 (acceptance fix) — the PBC cyclic-window + defer_head + tail-batched-send is
// the MOST intricate orchestration, yet the only PBC multi-node test above is
// bitwise-vs-CPU-ring, which SHARES eam_window_layout ⇒ blind to a dropped donor.
// This gives the PBC path its INDEPENDENT teeth: the multi-node PBC GPU ring ≡ the
// FP64 O(N²) serial-VV oracle (which has the complete periodic neighbourhood via
// min-image and shares no window/zone/ring code) to tolerance.
TEST(CudaEamGpuRing, MultiNodePbcTrajectoryMatchesFp64OracleTolerance) {
  const auto setfl = make_setfl();
  const SetflPot pot(setfl);
  const double dt = 0.001;
  const long steps = 8;
  const int n_zones = 5;  // PBC reach_mult=2: >=5 distinct cyclic zones, width 6.48

  core::Box box;
  const auto init = make_fcc_pbc(box);  // fully periodic z, Lz=32.4

  core::AtomSoA<double> oracle = init;
  serial_vv_fp64_oracle(oracle, box, setfl, steps, dt);

  for (int z : {2, 3}) {
    core::AtomSoA<double> g = init;
    const auto rg = run_gpu_ring(g, box, pot, setfl, ring_opts(steps, n_zones, z, dt));
    ASSERT_EQ(int(rg.halt), int(core::Halt::None)) << "z=" << z << " " << rg.halt_msg;
    EXPECT_LT(max_pos_dev(oracle, g), 1e-9)
        << "PBC GPU ring z=" << z << " diverges from the FP64 O(N²) oracle "
           "(dropped donor in the cyclic window / defer_head — invisible to 1-vs-z!)";
  }
}

namespace {
// G2: a STEEP-density setfl (β=3.3) whose density bound forces the Q23.40
// (fb=40) dual-format branch — mirrors the CPU DualFormatQ23OneVsZ, which the GPU
// ring had no equivalent of (all other tests use fb=44).
potentials::EamSetfl<double> make_setfl_q23() {
  potentials::AnalyticEam<double> m; m.rcut = kRcut; m.beta = 3.3; m.finalize();
  return potentials::EamSetfl<double>::from_analytic(m, 4000, 4000, 60.0);
}
}  // namespace

// G2 (acceptance fix) — the Q23.40 (fb=40) GPU dual-format branch ≡ CPU ring bitwise.
// Exercises the dens_scale = FixedAccum<40>::kScale path in both the GPU policy and
// the kernels, which fb=44 tests never touch.
TEST(CudaEamGpuRing, GpuRingQ2340MatchesCpuRingBitwise) {
  const auto setfl = make_setfl_q23();
  ASSERT_EQ(setfl.density_fracbits(), 40) << "expected Q23.40 dual-format setfl";
  const SetflPot pot(setfl);
  const double dt = 0.001;
  const long steps = 10;
  const int n_zones = 4;

  core::Box box;
  const auto init = make_fcc(2, 2, 6, 4.05, box);

  core::AtomSoA<double> cpu_ref = init;
  const auto rc = potentials::run_eam_ring(cpu_ref, box, pot, ring_opts(steps, n_zones, 1, dt));
  ASSERT_EQ(int(rc.halt), int(core::Halt::None)) << rc.halt_msg;

  for (int z : {2, 3}) {
    core::AtomSoA<double> g = init;
    const auto rg = run_gpu_ring(g, box, pot, setfl, ring_opts(steps, n_zones, z, dt));
    ASSERT_EQ(int(rg.halt), int(core::Halt::None)) << "z=" << z << " " << rg.halt_msg;
    EXPECT_TRUE(bitwise_eq(cpu_ref, g)) << "Q23.40 GPU ring z=" << z << " ≠ CPU ring";
  }
}
