// M6 / Tersoff-ladder Te3b: threaded TersoffRing — bond-order angular force on the time-parallel
// ring. Design wf_0dd9c6c5-129. A SIBLING FORK of sw_ring.hpp (orchestration character-identical;
// diff is the S1-S5 swaps + the TersoffWinForce policy block). Gates:
//   F-NOOP   (PR-level) `diff sw_ring tersoff_ring` is only the swaps; test_sw_ring/eam_ring green
//   G1       TersoffRing(z=1) state bitwise == serial VV driven by tersoff_zone_pass (the SORTED
//            serial window — THIS is the live ζ-order Hazard-B discriminator)
//   G2       1-vs-z bitwise (free-z + PBC, fixed + auto dt) — the Λ-chain handoff / order-free
//   G3 ⭐    ring state ≈ serial VV driven by the INDEPENDENT tersoff_direct_fp64 oracle (MB2)
//   G-PE ⭐   id-SHUFFLED: correct ring PE == serial & oracle; the local-key POISON PE is BITWISE
//            == correct (Hazard A: φ₂ repulsive key count-once-invariant — MEASURED, SW T3b)
//   G6       anti-deadlock z=1..6
//   G7 ⭐    firewall: TersoffWinForce ACCEPTS [BondOrder, Force]; REJECTS SW's [Force,Force]
//            (proves it is NOT a SwWinForce copy), the count, iterative, the symmetric trap
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "tdmd/core/integrator.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/potentials/tersoff.hpp"
#include "tdmd/potentials/tersoff_ring.hpp"
#include "tdmd/potentials/tersoff_zone.hpp"
#include "tdmd/units.hpp"

namespace core = tdmd::core;
namespace pot = tdmd::potentials;
namespace units = tdmd::units;
using tdmd::gen::make_diamond_si;

namespace {
constexpr int kNz = 12;

template <typename ForceFn>
void serial_vv(core::AtomSoA<double>& a, long steps, double dt, ForceFn force) {
  core::zero_forces(a); force(a);
  for (long h = 0; h < steps; ++h) {
    for (int i = 0; i < a.n; ++i) {
      const double inv_m = units::ftm2v / a.mass[i];
      a.vx[i] += 0.5 * dt * inv_m * a.fx[i]; a.vy[i] += 0.5 * dt * inv_m * a.fy[i]; a.vz[i] += 0.5 * dt * inv_m * a.fz[i];
      a.x[i] += dt * a.vx[i]; a.y[i] += dt * a.vy[i]; a.z[i] += dt * a.vz[i];
    }
    core::zero_forces(a); force(a);
    for (int i = 0; i < a.n; ++i) {
      const double inv_m = units::ftm2v / a.mass[i];
      a.vx[i] += 0.5 * dt * inv_m * a.fx[i]; a.vy[i] += 0.5 * dt * inv_m * a.fy[i]; a.vz[i] += 0.5 * dt * inv_m * a.fz[i];
    }
  }
}
bool state_bitwise_equal(const core::AtomSoA<double>& a, const core::AtomSoA<double>& b) {
  if (a.n != b.n) return false;
  for (int i = 0; i < a.n; ++i)
    if (a.x[i] != b.x[i] || a.y[i] != b.y[i] || a.z[i] != b.z[i] ||
        a.vx[i] != b.vx[i] || a.vy[i] != b.vy[i] || a.vz[i] != b.vz[i])
      return false;
  return true;
}
double max_state_dev(const core::AtomSoA<double>& a, const core::AtomSoA<double>& b) {
  double m = 0.0;
  for (int i = 0; i < a.n; ++i)
    m = std::max({m, std::fabs(a.x[i] - b.x[i]), std::fabs(a.y[i] - b.y[i]), std::fabs(a.z[i] - b.z[i])});
  return m;
}
core::ConveyorOptions ring_opts(long steps, int n_zones, int n_nodes, double dt) {
  core::ConveyorOptions o;
  o.steps = steps; o.n_zones = n_zones; o.n_nodes = n_nodes;
  o.auto_step = false; o.dt_initial = dt;
  o.reach_mult = 2; o.symmetric_reach = true;
  return o;
}
core::AtomSoA<double> shuffle_atoms(const core::AtomSoA<double>& a, unsigned seed) {
  std::vector<int> perm(a.n);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), std::mt19937(seed));
  core::AtomSoA<double> b; b.resize(a.n);
  for (int i = 0; i < a.n; ++i) {
    b.x[i] = a.x[perm[i]]; b.y[i] = a.y[perm[i]]; b.z[i] = a.z[perm[i]];
    b.vx[i] = a.vx[perm[i]]; b.vy[i] = a.vy[perm[i]]; b.vz[i] = a.vz[perm[i]];
    b.type[i] = a.type[perm[i]]; b.mass[i] = a.mass[perm[i]];
  }
  return b;
}
// φ₂ pairs in the (unsorted) slot-order ring window whose global-key order disagrees with local.
std::pair<long, long> count_key_flips(const core::AtomSoA<double>& a, const core::Box& box,
                                      const core::ZoneDecomposition& zd, const pot::TersoffParams& p) {
  const core::PairGeom geom(box, p.rcut());
  long flips = 0, total = 0;
  for (int zi = 0; zi < zd.n_zones; ++zi) {
    int ws[3]; const int nw = pot::eam_window_layout(zi, zd.n_zones, box.periodic[2], ws);
    std::vector<int> gid;
    for (int t = 0; t < nw; ++t)
      for (int g : zd.members[ws[t]]) gid.push_back(g);
    for (std::size_t o = 0; o < gid.size(); ++o)
      for (std::size_t b = o + 1; b < gid.size(); ++b) {
        double dx = a.x[gid[o]] - a.x[gid[b]], dy = a.y[gid[o]] - a.y[gid[b]], dz = a.z[gid[o]] - a.z[gid[b]], r2;
        if (!geom.reduce(dx, dy, dz, r2)) continue;
        ++total;
        if (!(gid[o] < gid[b])) ++flips;
      }
  }
  return {flips, total};
}
}  // namespace

// G1 — single-node Tersoff ring is bitwise-identical to the serial tersoff_zone_pass-driven VV.
// The ring gathers the window UNSORTED but TersoffWinForce::compute() CANONICALIZES it (sorts by
// global key) so the FP64 ζ-sum matches the SORTED serial path bitwise (Hazard B resolved). This
// gives the strong validation chain: ring ≡ serial ≡ tersoff_run_fixed ≡ FD/LAMMPS. (Without the
// sort, the FIRST force eval is still bitwise — the ζ delta is sub-quantum — but it amplifies via
// VV from step ≥ 2 into the deep velocity bits; hence state_bitwise_equal must check velocities.)
TEST(TersoffRing, SingleNodeMatchesSerialVV) {
  pot::TersoffParams p; pot::TersoffPotential<double> tpot(p);
  core::Box box;
  auto init = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  box.periodic = {true, true, false};  // free-z
  const long steps = 8; const double dt = 0.0005;

  core::AtomSoA<double> ref = init;
  const auto zd = core::ZoneDecomposition::build(ref, box, 6, p.rcut(), 2);
  serial_vv(ref, steps, dt, [&](core::AtomSoA<double>& a) { pot::tersoff_zone_pass(a, box, zd, p); });

  core::AtomSoA<double> ring = init;
  pot::run_tersoff_ring(ring, box, tpot, ring_opts(steps, 6, 1, dt));
  EXPECT_TRUE(state_bitwise_equal(ring, ref)) << "z=1 Tersoff ring ≠ serial VV bitwise";
}

// G2 — 1-vs-z bitwise (free-z + PBC, fixed + auto dt): the Λ-chain dt handoff / order-free int64.
TEST(TersoffRing, OneVsZBitwise) {
  pot::TersoffParams p; pot::TersoffPotential<double> tpot(p);
  for (bool pbc : {false, true}) {
    core::Box box;
    auto init = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
    box.periodic = {true, true, pbc};
    const long steps = 10; const double dt = 0.0004;
    const std::vector<int> zs = pbc ? std::vector<int>{2, 3, 6} : std::vector<int>{2, 3, 5};

    core::AtomSoA<double> ref = init;
    pot::run_tersoff_ring(ref, box, tpot, ring_opts(steps, 6, 1, dt));
    for (int z : zs) {
      core::AtomSoA<double> a = init;
      pot::run_tersoff_ring(a, box, tpot, ring_opts(steps, 6, z, dt));
      EXPECT_TRUE(state_bitwise_equal(a, ref)) << "1-vs-z (pbc=" << pbc << ") z=" << z;
    }
  }
  core::Box box;
  auto init = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  box.periodic = {true, true, false};
  auto aopt = [&](int z) { auto o = ring_opts(12, 6, z, 0.0004); o.auto_step = true;
                           o.ts.C1 = 0.002; o.ts.C3 = 1.0; o.ts.C_buf = 2.5; return o; };
  core::AtomSoA<double> r1 = init; pot::run_tersoff_ring(r1, box, tpot, aopt(1));
  for (int z : {2, 3, 5}) {
    core::AtomSoA<double> a = init; pot::run_tersoff_ring(a, box, tpot, aopt(z));
    EXPECT_TRUE(state_bitwise_equal(a, r1)) << "auto-dt 1-vs-z z=" << z;
  }
}

// G3 ⭐ — the ring trajectory matches a serial VV driven by the INDEPENDENT O(N²) FP64 oracle
// (tersoff_direct_fp64) to round-off. A dropped k-triplet / wrong ζ donor (1-vs-z is blind)
// would diverge >> this tolerance.
TEST(TersoffRing, MatchesFp64OracleTrajectory) {
  pot::TersoffParams p; pot::TersoffPotential<double> tpot(p);
  for (bool pbc : {false, true}) {
    core::Box box;
    auto init = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
    box.periodic = {true, true, pbc};
    const long steps = 3; const double dt = 0.0004;
    const core::PairGeom geom(box, p.rcut());

    core::AtomSoA<double> oref = init;
    serial_vv(oref, steps, dt, [&](core::AtomSoA<double>& a) { pot::tersoff_direct_fp64(a, geom, p, true); });
    core::AtomSoA<double> ring = init;
    pot::run_tersoff_ring(ring, box, tpot, ring_opts(steps, pbc ? 6 : 5, pbc ? 3 : 4, dt));
    EXPECT_LT(max_state_dev(ring, oref), 1e-6) << "ring ≠ FP64-oracle trajectory (pbc=" << pbc << ")";
  }
}

// G-PE ⭐ — the ring's per-step PE == serial tersoff_zone_pass AND ≈ the independent oracle.
// MEASURE-FIRST (SW T3b): on the id-shuffled fixture the ring window IS unsorted (count_key_flips
// finds φ₂ pairs whose global-key order != local), YET the local-key POISON gives a BITWISE PE ⇒
// the φ₂ repulsive global key is DEFENSIVE, NOT load-bearing on the slot gather (count-once-invariant).
TEST(TersoffRing, RingPeMatchesOracleAndGlobalKeyIsCountOnceInvariant) {
  pot::TersoffParams p; pot::TersoffPotential<double> tpot(p);
  core::Box box;
  auto base = make_diamond_si(2, 2, kNz, 5.431, 0.30, box);
  box.periodic = {true, true, true};
  auto init = shuffle_atoms(base, 2024);

  const auto zd = core::ZoneDecomposition::build(init, box, 6, p.rcut(), 2);
  const auto [flips, total] = count_key_flips(init, box, zd, p);
  ASSERT_GT(total, 0);
  EXPECT_GT(flips, 0) << "window fully sorted — the count-once-invariance finding is vacuous";

  const long steps = 1; const double dt = 0.0004;
  const core::PairGeom geom(box, p.rcut());
  core::AtomSoA<double> ring = init;
  const auto rc = pot::run_tersoff_ring(ring, box, tpot, ring_opts(steps, 6, 3, dt));
  core::AtomSoA<double> poison = init;
  const auto rp = pot::run_tersoff_ring(poison, box, tpot, ring_opts(steps, 6, 3, dt),
                                        pot::TersoffWinForceLocalKey<double>(tpot.ters));

  core::AtomSoA<double> at1 = ring;
  const double pe_serial = pot::tersoff_zone_pass(at1, box,
      core::ZoneDecomposition::build(at1, box, 6, p.rcut(), 2), p).pe;
  core::AtomSoA<double> at1o = ring;
  const double pe_oracle = pot::tersoff_direct_fp64(at1o, geom, p, false).pe;

  ASSERT_EQ(rc.stats.size(), 1u);
  ASSERT_EQ(rp.stats.size(), 1u);
  EXPECT_EQ(rc.stats[0].pe, pe_serial) << "ring PE ≠ serial tersoff_zone_pass PE";
  EXPECT_NEAR(rc.stats[0].pe, pe_oracle, 1e-6) << "ring PE ≠ independent oracle PE";
  EXPECT_TRUE(state_bitwise_equal(poison, ring)) << "the energy gate must not touch forces";
  EXPECT_EQ(rp.stats[0].pe, rc.stats[0].pe)
      << flips << "/" << total << " key-order flips, yet expected bitwise-identical PE";
}

// G6 — anti-deadlock across node counts (free-z).
TEST(TersoffRing, AntiDeadlock) {
  pot::TersoffParams p; pot::TersoffPotential<double> tpot(p);
  core::Box box;
  auto init = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  box.periodic = {true, true, false};
  for (int z = 1; z <= 6; ++z) {
    core::AtomSoA<double> a = init;
    const auto r = pot::run_tersoff_ring(a, box, tpot, ring_opts(2 * z + 3, 6, z, 0.0004));
    EXPECT_EQ(int(r.halt), int(core::Halt::None)) << "halt at z=" << z;
    EXPECT_EQ(r.steps_done, 2 * z + 3) << "incomplete at z=" << z;
  }
}

// G7 ⭐ — the INVERTED firewall: TersoffWinForce accepts [BondOrder, Force] (the third-atom-k
// transpose), and REJECTS SW's [Force,Force] (proving it is NOT a SwWinForce copy), the count,
// iterative, and the symmetric trap.
TEST(TersoffRing, FirewallAcceptsBondOrderRejectsBad) {
  using pot::PassDecl; using pot::PassKind;
  pot::TersoffPotential<double> tpot{};
  EXPECT_NO_THROW(pot::TersoffWinForce<double>::assert_supported(tpot.passes()));  // Tersoff = accept

  // SW's [Force, Force] — a SwWinForce copy would ACCEPT this; TersoffWinForce must REJECT
  // (pass 0 != BondOrder). Proves the inversion is real, not a relabel.
  const PassDecl sw2[2] = {{PassKind::Force, true, false, false, 40},
                           {PassKind::Force, true, true, false, 40}};
  EXPECT_THROW(pot::TersoffWinForce<double>::assert_supported(sw2), std::runtime_error);

  const PassDecl eam3[3] = {{PassKind::Density, true, false, false, 44},
                            {PassKind::Embedding, false, false, false, 30},
                            {PassKind::Force, true, false, false, 40}};
  EXPECT_THROW(pot::TersoffWinForce<double>::assert_supported(eam3), std::runtime_error);  // wrong count

  PassDecl it[2] = {{PassKind::BondOrder, true, false, false, 0},
                    {PassKind::Force, true, true, false, 40}};
  it[1].iterative = true;
  EXPECT_THROW(pot::TersoffWinForce<double>::assert_supported(it), std::runtime_error);  // iterative

  // symmetric trap: [BondOrder, Force(needs_transpose=false)] — a symmetric accumulator would
  // silently run the third-atom write wrong.
  const PassDecl sym[2] = {{PassKind::BondOrder, true, false, false, 0},
                           {PassKind::Force, true, false, false, 40}};
  EXPECT_THROW(pot::TersoffWinForce<double>::assert_supported(sym), std::runtime_error);
}
