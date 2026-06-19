// M6 / Tersoff-ladder Te4: NVE energy + momentum conservation on the threaded TersoffRing. No
// new machinery — validates the Te3b ring as a physical integrator. Methodology = M6 (calibrate
// the ceiling to the MEASURED floor, NOT pre-register). The C1 cosine cutoff (vs SW's smooth exp)
// is NOT exercised at 300 K (diamond-Si nn 2.35 ≪ R−D=2.8; 2nd shell 3.84 > R+D=3.2 ⇒ no taper
// atoms) ⇒ expect VV-truncation-dominated conservation. Momentum: Tersoff's transpose-replay
// quantizes f_i,f_j,f_k independently (the SW-T4 floor replicates), vs EAM's exact q(j)=−q(i). [ENG].
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "tdmd/core/integrator.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/thermal.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/potentials/tersoff.hpp"
#include "tdmd/potentials/tersoff_ring.hpp"
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
double cm_momentum(const core::AtomSoA<double>& a) {
  double px = 0, py = 0, pz = 0;
  for (int i = 0; i < a.n; ++i) { px += a.mass[i] * a.vx[i]; py += a.mass[i] * a.vy[i]; pz += a.mass[i] * a.vz[i]; }
  return std::sqrt(px * px + py * py + pz * pz);
}
}  // namespace

// THE NVE gate. Thermalize diamond-Si, equilibrate, then a measured NVE run; assert bounded
// energy conservation (calibrated floor) + momentum drift (calibrated floor). PBC so the two-bond
// seam is exercised under dynamics.
TEST(TersoffNVE, EnergyAndMomentumConserved) {
  pot::TersoffParams p; pot::TersoffPotential<double> tpot(p);
  core::Box box;
  auto a = make_diamond_si(2, 2, 12, 5.431, 0.0, box);  // perfect lattice; thermal gives the motion
  box.periodic = {true, true, true};
  thermal::maxwell_init(a, /*T=*/300.0, /*seed=*/12345);
  thermal::zero_momentum(a);

  const double dt = 0.001;  // 1 fs
  pot::run_tersoff_ring(a, box, tpot, nve_opts(/*equil=*/500, 6, 3, dt));

  const long steps = 3000;
  const double p0 = cm_momentum(a);
  const auto res = pot::run_tersoff_ring(a, box, tpot, nve_opts(steps, 6, 3, dt));
  ASSERT_EQ(int(res.halt), int(core::Halt::None));
  ASSERT_EQ(res.steps_done, steps);

  const double e0 = res.e0;
  double max_exc = 0.0;
  std::vector<double> ts, es;
  for (long h = 0; h < steps; ++h) {
    const double E = res.stats[std::size_t(h)].pe + res.stats[std::size_t(h)].ke;
    max_exc = std::max(max_exc, std::fabs(E - e0));
    ts.push_back(double(h) * dt); es.push_back(E);
  }
  const double rel_exc = max_exc / std::fabs(e0);

  double st = 0, se = 0, stt = 0, ste = 0; const double nN = double(es.size());
  for (std::size_t i = 0; i < es.size(); ++i) { st += ts[i]; se += es[i]; stt += ts[i] * ts[i]; ste += ts[i] * es[i]; }
  const double slope = (nN * ste - st * se) / (nN * stt - st * st);
  const int dof = 3 * a.n - 3;
  const double kT = tdmd::units::kB * 300.0;
  const double drift_norm = slope * 1000.0 / (kT * dof);

  const double dp = std::fabs(cm_momentum(a) - p0);
  std::printf("[Tersoff NVE] N=%d dof=%d e0=%.5f eV | rel_exc=%.3e | drift=%+.2e kT/(ns·dof) | "
              "max|dE|=%.3e eV | dp=%.3e amu·A/ps\n", a.n, dof, e0, rel_exc, drift_norm, max_exc, dp);

  // CALIBRATED ceilings (measured floor ×~3-6, M6 convention). MEASURED @300K/1fs (N=384):
  // rel_exc=8.1e-7 — VV-TRUNCATION-dominated (dt-halving 1fs→0.5fs ratio 3.98 ⇒ ~pure dt²; even
  // cleaner than SW's 3.37). The C1 cosine cutoff does NOT leak: at 300 K no atom enters the
  // taper [2.8,3.2] (nn 2.35, 2nd shell 3.84 outside R+D=3.2) ⇒ the fc'' discontinuity is never
  // sampled (the Te1 fc_d structural-silence finding, now under dynamics). dp=4.7e-9 — the int64
  // NON-SYMMETRIC transpose-replay quantization floor (proven by MomentumFloorIsInt64... below),
  // smaller than SW's 1.5e-8, vs EAM's exact q(j)=−q(i) ~1e-11; dp grows ∝√steps (random walk).
  EXPECT_LT(rel_exc, 2.5e-6) << "Tersoff NVE energy not bounded — a C1-cutoff leak?";
  EXPECT_LT(dp, 3e-8) << "Tersoff momentum drift above the int64 non-symmetric floor";
}

// Long-run determinism at NVE scale: the threaded ring (z>1) is bitwise-identical to z=1 over a
// full thermalized NVE trajectory (the Λ-chain handoff + the canonical ζ-sort under varying forces).
TEST(TersoffNVE, MultiNodeBitwiseOverTrajectory) {
  pot::TersoffParams p; pot::TersoffPotential<double> tpot(p);
  core::Box box;
  auto base = make_diamond_si(2, 2, 12, 5.431, 0.0, box);
  box.periodic = {true, true, true};
  thermal::maxwell_init(base, 350.0, 777);
  thermal::zero_momentum(base);

  const long steps = 1000; const double dt = 0.001;
  core::AtomSoA<double> ref = base;
  pot::run_tersoff_ring(ref, box, tpot, nve_opts(steps, 6, 1, dt));
  for (int z : {2, 6}) {
    core::AtomSoA<double> a = base;
    pot::run_tersoff_ring(a, box, tpot, nve_opts(steps, 6, z, dt));
    for (int i = 0; i < a.n; ++i) {
      ASSERT_EQ(a.x[i], ref.x[i]) << "NVE z=" << z << " ≠ z=1 at atom " << i;
      ASSERT_EQ(a.y[i], ref.y[i]); ASSERT_EQ(a.z[i], ref.z[i]);
      ASSERT_EQ(a.vx[i], ref.vx[i]); ASSERT_EQ(a.vy[i], ref.vy[i]); ASSERT_EQ(a.vz[i], ref.vz[i]);
    }
  }
}

// PROVE the momentum finding in CI (the SW-T4 discriminator replicates): the int64 non-symmetric
// transpose-replay quantization drifts Σf, NOT a force-asymmetry bug. The FP64 oracle conserves Σf
// to round-off; the int64 path quantizes each owner independently (rint not additive) ⇒ Σf≠0 at
// the quantum. A real asymmetry bug would break BOTH; the gap proves quantization.
TEST(TersoffNVE, MomentumFloorIsInt64QuantizationNotABug) {
  pot::TersoffParams p;
  core::Box box;
  auto a = make_diamond_si(2, 2, 12, 5.431, 0.20, box);
  box.periodic = {true, true, true};
  const core::PairGeom geom(box, p.rcut());

  auto sumf = [](const core::AtomSoA<double>& s) {
    double x = 0, y = 0, z = 0;
    for (int i = 0; i < s.n; ++i) { x += s.fx[i]; y += s.fy[i]; z += s.fz[i]; }
    return std::sqrt(x * x + y * y + z * z);
  };
  core::AtomSoA<double> o = a; core::zero_forces(o);
  pot::tersoff_direct_fp64(o, geom, p, /*with_forces=*/true);
  const double sf_fp64 = sumf(o);
  core::AtomSoA<double> q = a; core::zero_forces(q);
  pot::tersoff_run_fixed(q, geom, p);
  const double sf_int64 = sumf(q);

  EXPECT_LT(sf_fp64, 1e-12) << "FP64 oracle Σf not round-off — a REAL force-asymmetry bug";
  EXPECT_GT(sf_int64, 0.0) << "int64 Σf exactly zero — the finding would be vacuous";
  EXPECT_LT(sf_int64, 1e-9) << "int64 Σf far above the quantum — not just B1 quantization";
}
