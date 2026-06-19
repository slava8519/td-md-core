// M6 / SW-ladder T4: NVE energy + momentum conservation on the threaded SwRing. No new
// machinery — validates the T3b ring as a physical integrator. Methodology = the M6 EAM
// drift (tools/eam_drift.cu): symplectic VV has NO secular energy leak, so max|ΔE| ≈ the
// round-off floor; the ceiling is CALIBRATED to the measured floor, NOT pre-registered (the
// M6 lesson). SW-specific finding to MEASURE: EAM's symmetric q(j)=−q(i) conserves momentum
// int64-EXACTLY; SW's triplet quantizes f_i,f_j,f_k independently (rint not additive) ⇒
// Σf≠0 exactly ⇒ momentum drifts at the quantization floor. [ENG].
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "tdmd/core/integrator.hpp"  // kinetic_energy
#include "tdmd/core/soa.hpp"
#include "tdmd/core/thermal.hpp"     // maxwell_init, zero_momentum
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/potentials/sw.hpp"
#include "tdmd/potentials/sw_ring.hpp"
#include "tdmd/units.hpp"

namespace core = tdmd::core;
namespace pot = tdmd::potentials;
namespace thermal = tdmd::core::thermal;
using tdmd::gen::make_diamond_si;

namespace {
core::ConveyorOptions nve_opts(long steps, int n_zones, int n_nodes, double dt) {
  core::ConveyorOptions o;
  o.steps = steps; o.n_zones = n_zones; o.n_nodes = n_nodes;
  o.auto_step = false; o.dt_initial = dt;
  o.reach_mult = 2; o.symmetric_reach = true;
  return o;
}
// p_cm magnitude (amu·Å/ps).
double cm_momentum(const core::AtomSoA<double>& a) {
  double px = 0, py = 0, pz = 0;
  for (int i = 0; i < a.n; ++i) { px += a.mass[i] * a.vx[i]; py += a.mass[i] * a.vy[i]; pz += a.mass[i] * a.vz[i]; }
  return std::sqrt(px * px + py * py + pz * pz);
}
}  // namespace

// THE NVE gate. Thermalize diamond-Si, equilibrate, then a measured NVE run; assert bounded
// energy conservation (calibrated floor) + momentum drift (calibrated floor). PBC-z so the
// two-wing seam is exercised under dynamics.
TEST(SwNVE, EnergyAndMomentumConserved) {
  pot::SwParams sp; pot::SwPotential<double> spot(sp);
  core::Box box;
  auto a = make_diamond_si(2, 2, 12, 5.431, 0.0, box);  // perfect lattice; thermal gives the motion
  box.periodic = {true, true, true};
  thermal::maxwell_init(a, /*T=*/300.0, /*seed=*/12345);
  thermal::zero_momentum(a);

  const double dt = 0.001;  // 1 fs
  pot::run_sw_ring(a, box, spot, nve_opts(/*equil=*/500, 6, 3, dt));  // settle PE/KE

  const long steps = 3000;
  const double p0 = cm_momentum(a);
  const auto res = pot::run_sw_ring(a, box, spot, nve_opts(steps, 6, 3, dt));
  ASSERT_EQ(int(res.halt), int(core::Halt::None));
  ASSERT_EQ(res.steps_done, steps);

  // energy trajectory E(h) = pe + ke; e0 = the post-equilibration total.
  const double e0 = res.e0;
  double max_exc = 0.0;
  std::vector<double> ts, es;
  for (long h = 0; h < steps; ++h) {
    const double E = res.stats[std::size_t(h)].pe + res.stats[std::size_t(h)].ke;
    max_exc = std::max(max_exc, std::fabs(E - e0));
    ts.push_back(double(h) * dt); es.push_back(E);
  }
  const double rel_exc = max_exc / std::fabs(e0);

  // secular slope via least squares over (t, E) — the leak rate (eV/ps).
  double st = 0, se = 0, stt = 0, ste = 0; const double nN = double(es.size());
  for (std::size_t i = 0; i < es.size(); ++i) { st += ts[i]; se += es[i]; stt += ts[i] * ts[i]; ste += ts[i] * es[i]; }
  const double slope = (nN * ste - st * se) / (nN * stt - st * st);  // eV/ps
  const int dof = 3 * a.n - 3;
  const double kT = tdmd::units::kB * 300.0;
  const double drift_norm = slope * 1000.0 / (kT * dof);  // kT/(ns·dof)

  const double dp = std::fabs(cm_momentum(a) - p0);
  std::printf("[SW NVE] N=%d dof=%d e0=%.5f eV | rel_exc=%.3e | drift=%+.2e kT/(ns·dof) | "
              "max|dE|=%.3e eV | dp=%.3e amu·Å/ps\n", a.n, dof, e0, rel_exc, drift_norm, max_exc, dp);

  // CALIBRATED ceilings (measured floor ×~3-7, M3.5/M6 convention). MEASURED @300K/1fs:
  // rel_exc≈1.6e-6 — VV-TRUNCATION-dominated (dt-halving ratio 3.37 ⇒ ~94% dt² + ~6% round-off
  // floor; NO force-shift leak — the SW exp envelope vanishes value+derivative smoothly, unlike
  // Morse+shift ~1e-2; max|ΔE| is run-length-INVARIANT to 30 ps ⇒ bounded oscillation, so the
  // secular slope is a least-squares fit artifact, not a leak). dp≈1.5e-8 — the SW NON-SYMMETRIC
  // -TRIPLET quantization floor (~25% of triplets carry an O(1)-quantum residual; proven by the
  // MomentumFloorIsInt64Quantization gate below), vs EAM's exact q(j)=−q(i) ~1e-11; dp grows
  // ∝√steps (random walk) so the ceiling re-calibrates for longer runs, not silently trips.
  EXPECT_LT(rel_exc, 5e-6) << "SW NVE energy not bounded — a force-shift leak?";
  EXPECT_LT(dp, 1e-7) << "SW momentum drift above the int64 non-symmetric-triplet floor";
}

// Long-run determinism at NVE scale: the threaded ring (z>1) is bitwise-identical to z=1
// over a full thermalized NVE trajectory (the Λ-chain handoff under varying forces).
TEST(SwNVE, MultiNodeBitwiseOverTrajectory) {
  pot::SwParams sp; pot::SwPotential<double> spot(sp);
  core::Box box;
  auto base = make_diamond_si(2, 2, 12, 5.431, 0.0, box);
  box.periodic = {true, true, true};
  thermal::maxwell_init(base, 350.0, 777);
  thermal::zero_momentum(base);

  const long steps = 1000; const double dt = 0.001;
  core::AtomSoA<double> ref = base;
  pot::run_sw_ring(ref, box, spot, nve_opts(steps, 6, 1, dt));
  for (int z : {2, 6}) {
    core::AtomSoA<double> a = base;
    pot::run_sw_ring(a, box, spot, nve_opts(steps, 6, z, dt));
    for (int i = 0; i < a.n; ++i) {
      ASSERT_EQ(a.x[i], ref.x[i]) << "NVE z=" << z << " ≠ z=1 at atom " << i;
      ASSERT_EQ(a.y[i], ref.y[i]); ASSERT_EQ(a.z[i], ref.z[i]);
      ASSERT_EQ(a.vx[i], ref.vx[i]); ASSERT_EQ(a.vy[i], ref.vy[i]); ASSERT_EQ(a.vz[i], ref.vz[i]);
    }
  }
}

// MUST-FIX (acceptance) — PROVE the momentum finding in CI, not just prose: the int64
// non-symmetric-triplet quantization is what drifts momentum, NOT a force-asymmetry bug. The
// FP64 oracle (f_i=−(f_j+f_k) in double) conserves Σf to round-off; the int64 transpose-replay
// quantizes each owner independently (rint not additive) ⇒ Σf≠0 at the quantum. A real
// asymmetry bug would break BOTH; the gap proves quantization. Perturbed lattice ⇒ φ₃≠0.
TEST(SwNVE, MomentumFloorIsInt64QuantizationNotABug) {
  pot::SwParams sp;
  core::Box box;
  auto a = make_diamond_si(2, 2, 12, 5.431, 0.20, box);  // perturbed ⇒ angular forces nonzero
  box.periodic = {true, true, true};

  auto sumf = [](const core::AtomSoA<double>& s) {
    double x = 0, y = 0, z = 0;
    for (int i = 0; i < s.n; ++i) { x += s.fx[i]; y += s.fy[i]; z += s.fz[i]; }
    return std::sqrt(x * x + y * y + z * z);
  };
  core::AtomSoA<double> o = a; core::zero_forces(o);
  pot::sw_direct_fp64(o, box, sp, /*with_forces=*/true);  // FP64 scatter: Σf = round-off
  const double sf_fp64 = sumf(o);
  core::AtomSoA<double> q = a; core::zero_forces(q);
  pot::sw_run_fixed(q, box, sp);                           // int64 transpose-replay: Σf ≠ 0
  const double sf_int64 = sumf(q);

  EXPECT_LT(sf_fp64, 1e-12) << "FP64 oracle Σf not round-off — a REAL force-asymmetry bug";
  EXPECT_GT(sf_int64, 0.0) << "int64 Σf exactly zero — the finding would be vacuous";
  EXPECT_LT(sf_int64, 1e-9) << "int64 Σf far above the quantum — not just B1 quantization";
}
