// M6 / MEAM-ladder Me4: NVE energy + momentum conservation on the threaded MeamRing. No new
// machinery — validates the Me3b ring as a physical integrator. Methodology = M6 (calibrate the
// ceiling to the MEASURED floor). KEY MEAM nuance: the screening force is DEAD on the diamond
// (binary S ⇒ symmetric writes ⇒ momentum int64-EXACT), but on the partial-screening slab the
// screening 3rd-atom transpose quantizes f_i,f_j,f_k independently ⇒ int64 Σf≠0 (the SW-T4/Te4
// floor — here it appears ONLY where screening is active). [ENG], single-element Si.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "tdmd/core/integrator.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/thermal.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/gen/partial_screen_slab.hpp"
#include "tdmd/potentials/meam.hpp"
#include "tdmd/potentials/meam_ring.hpp"
#include "tdmd/units.hpp"

namespace core = tdmd::core;
namespace pot = tdmd::potentials;
namespace thermal = tdmd::core::thermal;

namespace {
core::ConveyorOptions nve_opts(long steps, int n_zones, int n_nodes, double dt) {
  core::ConveyorOptions o;
  o.steps = steps; o.n_zones = n_zones; o.n_nodes = n_nodes;
  o.auto_step = false; o.dt_initial = dt; o.reach_mult = 2; o.symmetric_reach = true;
  return o;
}
double sumf(const core::AtomSoA<double>& a) {
  double x=0,y=0,z=0; for (int i=0;i<a.n;++i){x+=a.fx[i];y+=a.fy[i];z+=a.fz[i];} return std::sqrt(x*x+y*y+z*z);
}
}  // namespace

// THE NVE gate — diamond z-slab (the density+pair+embedding NVE; screening is binary-S here).
// MEASURED (shared-snapshot dt sweep): rel_exc dt-halving ratio 4.06/4.02 — clean dt² VV-truncation,
// and the SECULAR slope ALSO scales dt² (2.62e-3→6.56e-4→1.64e-4 eV/ps) ⇒ NO force-shift leak (the
// MEAM screening fcut cutoff smoothly zeros value+derivative, unlike Morse+shift). MEAM Si is
// STIFFER than SW/Tersoff (steeper embedding+screening curvature ⇒ larger truncation at the same
// dt: 4.1e-6 @ 0.5 fs vs SW 1.6e-6 / Tersoff 8.1e-7) — physical, the dt² scaling proves it bounded.
TEST(MeamNVE, EnergyConserved) {
  pot::MeamParams p; pot::MeamPotential<double> mpot{};
  core::Box box; auto a = tdmd::gen::make_diamond_si(2, 2, 12, 5.431, 0.10, box);
  box.periodic = {true, true, true};
  thermal::maxwell_init(a, 300.0, 12345); thermal::zero_momentum(a);
  const double dt = 0.0005;  // 0.5 fs — conservative for stiff MEAM Si
  pot::run_meam_ring(a, box, mpot, nve_opts(2000, 6, 3, dt));  // equilibrate (relax the perturbation)

  const long steps = 2000;
  const auto res = pot::run_meam_ring(a, box, mpot, nve_opts(steps, 6, 3, dt));
  ASSERT_EQ(int(res.halt), int(core::Halt::None));
  ASSERT_EQ(res.steps_done, steps);
  const double e0 = res.e0;
  double max_exc = 0;
  for (long h = 0; h < steps; ++h)
    max_exc = std::max(max_exc, std::fabs(res.stats[std::size_t(h)].pe + res.stats[std::size_t(h)].ke - e0));
  const double rel_exc = max_exc / std::fabs(e0);
  std::printf("[MEAM NVE] N=%d e0=%.5f eV | rel_exc=%.3e | max|dE|=%.3e eV\n", a.n, e0, rel_exc, max_exc);
  // CALIBRATED ceiling (measured floor 4.1e-6 ×~3, M6 convention). Bounded VV-truncation.
  EXPECT_LT(rel_exc, 1.3e-5) << "MEAM NVE energy not bounded";
}

// Long-run determinism: the threaded ring (z>1) is bitwise-identical to z=1 over a thermalized NVE
// trajectory (the Λ-chain + the canonical sort under varying forces), diamond + slab.
TEST(MeamNVE, MultiNodeBitwiseOverTrajectory) {
  pot::MeamPotential<double> mpot{};
  const long steps = 600; const double dt = 0.0005;
  for (int which = 0; which < 2; ++which) {
    core::AtomSoA<double> base; core::Box box;
    if (which == 0) { base = tdmd::gen::make_diamond_si(2, 2, 12, 5.431, 0.10, box); box.periodic = {true, true, true}; thermal::maxwell_init(base, 350.0, 777); }
    else { auto s = tdmd::gen::make_partial_screen_slab(8); base = s.atoms; box = s.box; box.periodic = {true, true, true}; thermal::maxwell_init(base, 150.0, 777); }
    thermal::zero_momentum(base);
    core::AtomSoA<double> ref = base;
    pot::run_meam_ring(ref, box, mpot, nve_opts(steps, 6, 1, dt));
    for (int z : {2, 3}) {
      core::AtomSoA<double> aa = base;
      pot::run_meam_ring(aa, box, mpot, nve_opts(steps, 6, z, dt));
      for (int i = 0; i < aa.n; ++i) {
        ASSERT_EQ(aa.x[i], ref.x[i]) << "which=" << which << " z=" << z << " atom " << i;
        ASSERT_EQ(aa.vx[i], ref.vx[i]); ASSERT_EQ(aa.vy[i], ref.vy[i]); ASSERT_EQ(aa.vz[i], ref.vz[i]);
      }
    }
  }
}

// PROVE the momentum finding: the screening 3rd-atom transpose ⇒ int64 Σf≠0 (a B1 quantization
// consequence), NOT a force-asymmetry bug. On the SLAB (screening active) the FP64 oracle conserves
// Σf to round-off while the int64 path drifts at the quantum; on the DIAMOND (binary S, symmetric
// writes) BOTH are exact — the discriminator that isolates the screening-transpose floor.
TEST(MeamNVE, MomentumFloorIsScreeningTransposeQuantization) {
  pot::MeamParams p;
  // slab — screening active ⇒ the floor appears.
  auto s = tdmd::gen::make_partial_screen_slab(8);
  core::AtomSoA<double> so = s.atoms, sq = s.atoms; core::zero_forces(so); core::zero_forces(sq);
  pot::meam_direct_fp64(so, core::PairGeom(s.box, p.rc), p, true);
  pot::meam_run_fixed_force(sq, core::PairGeom(s.box, p.rc), p);
  EXPECT_LT(sumf(so), 1e-12) << "slab FP64 oracle Σf not round-off — a force-asymmetry bug";
  EXPECT_GT(sumf(sq), 0.0) << "slab int64 Σf exactly zero — the screening-transpose finding is vacuous";
  EXPECT_LT(sumf(sq), 1e-9) << "slab int64 Σf far above the quantum — a bug";
  // diamond — binary S (symmetric writes) ⇒ int64 Σf EXACT (the floor is screening-specific).
  core::Box box; auto d = tdmd::gen::make_diamond_si(2, 2, 2, 5.431, 0.20, box); box.periodic = {true, true, true};
  core::AtomSoA<double> dq = d; core::zero_forces(dq);
  pot::meam_run_fixed_force(dq, core::PairGeom(box, p.rc), p);
  EXPECT_LT(sumf(dq), 1e-12) << "diamond int64 Σf not exact — screening unexpectedly active (not binary-S)";
}
