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
#include "tdmd/potentials/eam.hpp"
#include "tdmd/potentials/eam_analytic.hpp"
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
