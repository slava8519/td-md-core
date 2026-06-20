// M6 / MEAM-ladder Me3b: threaded MeamRing — MEAM on the time-parallel ring. The 4th fork of the
// proven ring (EamRing→SwRing→TersoffRing→MeamRing); diff is the swaps + the MeamWinForce policy.
// Genuine deltas: the two-PE-accum int64 fold (pe.raw += pe_embed.raw + pe_pair.raw), the canonical
// ζ-analog sort (the screening product Π_k S + the FP density-derivative sums are order-sensitive;
// the ring window is UNSORTED ⇒ sort to match the serial path — measured load-bearing), and the
// firewall INVERSION of [Density,Embedding,Force(needs_transpose)] (EAM's shape, but accept the
// screening 3rd-atom transpose). Gates:
//   G1 ring(z=1) state bitwise == serial VV driven by meam_zone_pass (slab + diamond)
//   G2 1-vs-z bitwise (free + PBC) — the Λ-chain + canonical sort under varying forces
//   G3 ⭐ ring ≈ serial VV driven by the INDEPENDENT meam_direct_fp64 oracle (screening completeness)
//   G6 anti-deadlock z=1..5
//   G7 ⭐ firewall: accepts [Density,Embedding,Force]; rejects Tersoff/EAM-symmetric/iterative/count
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "tdmd/core/integrator.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/thermal.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/gen/partial_screen_slab.hpp"
#include "tdmd/potentials/meam.hpp"
#include "tdmd/potentials/meam_ring.hpp"
#include "tdmd/potentials/meam_zone.hpp"
#include "tdmd/units.hpp"

namespace core = tdmd::core;
namespace pot = tdmd::potentials;
namespace units = tdmd::units;

namespace {
core::ConveyorOptions ropt(long steps, int n_zones, int n_nodes, double dt) {
  core::ConveyorOptions o;
  o.steps = steps; o.n_zones = n_zones; o.n_nodes = n_nodes;
  o.auto_step = false; o.dt_initial = dt; o.reach_mult = 2; o.symmetric_reach = true;
  return o;
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
    for (int i = 0; i < a.n; ++i) { double im = units::ftm2v / a.mass[i];
      a.vx[i]+=0.5*dt*im*a.fx[i]; a.vy[i]+=0.5*dt*im*a.fy[i]; a.vz[i]+=0.5*dt*im*a.fz[i];
      a.x[i]+=dt*a.vx[i]; a.y[i]+=dt*a.vy[i]; a.z[i]+=dt*a.vz[i]; }
    core::zero_forces(a); force(a);
    for (int i = 0; i < a.n; ++i) { double im = units::ftm2v / a.mass[i];
      a.vx[i]+=0.5*dt*im*a.fx[i]; a.vy[i]+=0.5*dt*im*a.fy[i]; a.vz[i]+=0.5*dt*im*a.fz[i]; }
  }
}
// (slab, diamond) fixtures with thermal velocities; pbc_z toggles the z boundary.
struct Fix { core::AtomSoA<double> a; core::Box box; pot::MeamPotential<double> pot{}; };
Fix slab_fix(bool pbc_z, unsigned seed) {
  Fix f; auto s = tdmd::gen::make_partial_screen_slab(8); f.a = s.atoms; f.box = s.box;
  f.box.periodic = {true, true, pbc_z};
  core::thermal::maxwell_init(f.a, 200.0, seed); core::thermal::zero_momentum(f.a);
  return f;
}
Fix dia_fix(bool pbc_z, unsigned seed) {
  Fix f; f.a = tdmd::gen::make_diamond_si(2, 2, 12, 5.431, 0.15, f.box);
  f.box.periodic = {true, true, pbc_z};
  core::thermal::maxwell_init(f.a, 300.0, seed); core::thermal::zero_momentum(f.a);
  return f;
}
}  // namespace

// G1 — ring(z=1) bitwise == serial-VV driven by meam_zone_pass, slab AND diamond.
TEST(MeamRing, SingleNodeMatchesSerialVV) {
  pot::MeamParams p; const double dt = 0.0003; const long steps = 6;
  for (int which = 0; which < 2; ++which) {
    Fix f = which == 0 ? slab_fix(false, 11) : dia_fix(false, 11);
    core::AtomSoA<double> ref = f.a;
    const auto zd = core::ZoneDecomposition::build(ref, f.box, 6, p.rc, 2);
    serial_vv(ref, steps, dt, [&](core::AtomSoA<double>& a) { pot::meam_zone_pass(a, f.box, zd, p); });
    core::AtomSoA<double> ring = f.a;
    pot::run_meam_ring(ring, f.box, f.pot, ropt(steps, 6, 1, dt));
    EXPECT_TRUE(state_eq(ring, ref)) << "z=1 ring ≠ serial-VV (which=" << which << ")";
  }
}

// G2 — 1-vs-z bitwise (free + PBC), slab AND diamond.
TEST(MeamRing, OneVsZBitwise) {
  pot::MeamParams p; const double dt = 0.0003; const long steps = 8;
  for (int which = 0; which < 2; ++which)
    for (bool pbc : {false, true}) {
      Fix f = which == 0 ? slab_fix(pbc, 7) : dia_fix(pbc, 7);
      core::AtomSoA<double> ref = f.a;
      pot::run_meam_ring(ref, f.box, f.pot, ropt(steps, 6, 1, dt));
      for (int z : {2, 3}) {
        core::AtomSoA<double> a = f.a;
        pot::run_meam_ring(a, f.box, f.pot, ropt(steps, 6, z, dt));
        EXPECT_TRUE(state_eq(a, ref)) << "1-vs-z which=" << which << " pbc=" << pbc << " z=" << z;
      }
    }
}

// G3 ⭐ — ring trajectory matches a serial VV driven by the INDEPENDENT meam_direct_fp64 oracle
// (the screening-completeness on the ring — a dropped cyclic-window screening-k is blind to 1-vs-z).
TEST(MeamRing, MatchesFp64OracleTrajectory) {
  pot::MeamParams p; const double dt = 0.0003; const long steps = 3;
  for (int which = 0; which < 2; ++which)
    for (bool pbc : {false, true}) {
      Fix f = which == 0 ? slab_fix(pbc, 2024) : dia_fix(pbc, 2024);
      const core::PairGeom geom(f.box, p.rc);
      core::AtomSoA<double> oref = f.a;
      serial_vv(oref, steps, dt, [&](core::AtomSoA<double>& a) { pot::meam_direct_fp64(a, geom, p, true); });
      core::AtomSoA<double> ring = f.a;
      pot::run_meam_ring(ring, f.box, f.pot, ropt(steps, 6, 3, dt));
      EXPECT_LT(state_dev(ring, oref), 1e-6) << "ring ≠ oracle which=" << which << " pbc=" << pbc;
    }
}

// G6 — anti-deadlock across node counts (slab, free-z).
TEST(MeamRing, AntiDeadlock) {
  Fix f = slab_fix(false, 3);
  for (int z = 1; z <= 5; ++z) {
    core::AtomSoA<double> a = f.a;
    const auto r = pot::run_meam_ring(a, f.box, f.pot, ropt(2 * z + 3, 6, z, 0.0003));
    EXPECT_EQ(int(r.halt), int(core::Halt::None)) << "halt z=" << z;
    EXPECT_EQ(r.steps_done, 2 * z + 3) << "incomplete z=" << z;
  }
}

// G7 ⭐ — the INVERTED firewall: MeamWinForce accepts [Density,Embedding,Force(needs_transpose)]
// (EAM's shape, but accept the screening 3rd-atom transpose); rejects Tersoff's [BondOrder,Force],
// the EAM symmetric-Force trap, iterative, wrong count.
TEST(MeamRing, FirewallAcceptsScreeningTranspose) {
  using pot::PassDecl; using pot::PassKind;
  pot::MeamPotential<double> mpot{};
  EXPECT_NO_THROW(pot::MeamWinForce<double>::assert_supported(mpot.passes()));
  static_assert(requires { pot::MeamWinForce<double>::assert_supported(std::span<const pot::PassDecl>{}); });

  const PassDecl ters2[2] = {{PassKind::BondOrder,true,false,false,0},{PassKind::Force,true,true,false,40}};
  EXPECT_THROW(pot::MeamWinForce<double>::assert_supported(ters2), std::runtime_error);  // Tersoff desc
  const PassDecl eam_sym[3] = {{PassKind::Density,true,false,false,44},{PassKind::Embedding,false,false,false,30},{PassKind::Force,true,false,false,40}};
  EXPECT_THROW(pot::MeamWinForce<double>::assert_supported(eam_sym), std::runtime_error);  // symmetric trap
  PassDecl it[3] = {{PassKind::Density,true,false,false,44},{PassKind::Embedding,false,false,false,30},{PassKind::Force,true,true,false,40}};
  it[0].iterative = true;
  EXPECT_THROW(pot::MeamWinForce<double>::assert_supported(it), std::runtime_error);  // iterative
}
