// M6 / SW-ladder T3b: threaded SwRing — Stillinger-Weber on the time-parallel ring.
// Design wf_4752ee3c-cde. A SIBLING FORK of eam_ring.hpp (orchestration character-
// identical; diff is the 5 EAM→SW swaps + SwWinForce). Gates:
//   F-NOOP   (PR-level) `diff eam_ring sw_ring` is only the swaps; test_eam_ring green
//   G1       SwRing(z=1) state bitwise == serial VV driven by sw_zone_pass
//   G2       1-vs-z bitwise (free-z + PBC, fixed + auto dt) — the Λ-chain handoff
//   G3       ring state ≈ serial VV driven by the INDEPENDENT sw_direct_fp64 oracle (MB2)
//   G-PE ⭐   on an id-SHUFFLED fixture: correct ring per-step PE == serial & oracle; and
//            the local-key POISON PE is BITWISE-IDENTICAL (count-once-invariant) — MEASURED,
//            overturning the design's "global key becomes load-bearing in T3b" claim: the
//            key is DEFENSIVE (robust to gather reorder), not load-bearing on the slot gather
//   G6       anti-deadlock z=1..6
//   G7       firewall: assert_supported ACCEPTS SW (needs_transpose) + rejects bad descriptors
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
#include "tdmd/potentials/sw.hpp"
#include "tdmd/potentials/sw_ring.hpp"
#include "tdmd/potentials/sw_zone.hpp"
#include "tdmd/units.hpp"

namespace core = tdmd::core;
namespace pot = tdmd::potentials;
namespace units = tdmd::units;
using tdmd::gen::make_diamond_si;

namespace {
constexpr int kNz = 12;

// Serial velocity-Verlet with a pluggable force (sw_zone_pass = structural sibling;
// sw_direct_fp64 = independent O(N²) oracle) — the SAME integration arithmetic the ring uses.
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

// Self-certify the ring window is UNSORTED: replicate the slot-order gather (members of
// {j-1,j,j+1} concatenated, ascending within each zone) and count φ₂ pairs whose GLOBAL-key
// order disagrees with the LOCAL (window) order — exactly the pairs the global key fixes.
std::pair<long, long> count_key_flips(const core::AtomSoA<double>& a, const core::Box& box,
                                      const core::ZoneDecomposition& zd, const pot::SwParams& sp) {
  const core::PairGeom geom(box, sp.rcut());
  long flips = 0, total = 0;
  for (int zi = 0; zi < zd.n_zones; ++zi) {
    int ws[3]; const int nw = pot::eam_window_layout(zi, zd.n_zones, box.periodic[2], ws);
    std::vector<int> gid;  // window global ids in slot order
    for (int t = 0; t < nw; ++t)
      for (int g : zd.members[ws[t]]) gid.push_back(g);
    for (std::size_t o = 0; o < gid.size(); ++o)
      for (std::size_t b = o + 1; b < gid.size(); ++b) {
        double dx = a.x[gid[o]] - a.x[gid[b]], dy = a.y[gid[o]] - a.y[gid[b]], dz = a.z[gid[o]] - a.z[gid[b]], r2;
        if (!geom.reduce(dx, dy, dz, r2)) continue;
        ++total;  // a φ₂ pair in the window; local order is (o<b) ≡ true here
        if (!(gid[o] < gid[b])) ++flips;  // global-key order disagrees with local order
      }
  }
  return {flips, total};
}
}  // namespace

// G1 — single-node SW ring is bitwise-identical to the serial sw_zone_pass-driven VV.
TEST(SwRing, SingleNodeMatchesSerialVV) {
  pot::SwParams sp; pot::SwPotential<double> spot(sp);
  core::Box box;
  auto init = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  box.periodic = {true, true, false};  // free-z
  const long steps = 8; const double dt = 0.0005;

  core::AtomSoA<double> ref = init;
  const auto zd = core::ZoneDecomposition::build(ref, box, 6, sp.rcut(), 2);
  serial_vv(ref, steps, dt, [&](core::AtomSoA<double>& a) { pot::sw_zone_pass(a, box, zd, sp); });

  core::AtomSoA<double> ring = init;
  pot::run_sw_ring(ring, box, spot, ring_opts(steps, 6, 1, dt));
  EXPECT_TRUE(state_bitwise_equal(ring, ref)) << "z=1 SW ring ≠ serial VV bitwise";
}

// G2 — 1-vs-z bitwise (free-z + PBC, fixed + auto dt): the Λ-chain dt handoff / order-free.
TEST(SwRing, OneVsZBitwise) {
  pot::SwParams sp; pot::SwPotential<double> spot(sp);
  for (bool pbc : {false, true}) {
    core::Box box;
    auto init = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
    box.periodic = {true, true, pbc};
    const long steps = 10; const double dt = 0.0004;
    const std::vector<int> zs = pbc ? std::vector<int>{2, 3, 6} : std::vector<int>{2, 3, 5};

    core::AtomSoA<double> ref = init;
    pot::run_sw_ring(ref, box, spot, ring_opts(steps, 6, 1, dt));
    for (int z : zs) {
      core::AtomSoA<double> a = init;
      pot::run_sw_ring(a, box, spot, ring_opts(steps, 6, z, dt));
      EXPECT_TRUE(state_bitwise_equal(a, ref)) << "1-vs-z (pbc=" << pbc << ") z=" << z;
    }
  }
  // auto-dt 1-vs-z (free-z) — the Λ-chain under a varying step.
  core::Box box;
  auto init = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  box.periodic = {true, true, false};
  auto aopt = [&](int z) { auto o = ring_opts(12, 6, z, 0.0004); o.auto_step = true;
                           o.ts.C1 = 0.002; o.ts.C3 = 1.0; o.ts.C_buf = 2.5; return o; };
  core::AtomSoA<double> r1 = init; pot::run_sw_ring(r1, box, spot, aopt(1));
  for (int z : {2, 3, 5}) {
    core::AtomSoA<double> a = init; pot::run_sw_ring(a, box, spot, aopt(z));
    EXPECT_TRUE(state_bitwise_equal(a, r1)) << "auto-dt 1-vs-z z=" << z;
  }
}

// G3 ⭐ — the ring trajectory matches a serial VV driven by the INDEPENDENT O(N²) FP64
// oracle (sw_direct_fp64) to round-off over a short run. A dropped triplet in the ring
// (1-vs-z is blind) would diverge >> this tolerance.
TEST(SwRing, MatchesFp64OracleTrajectory) {
  pot::SwParams sp; pot::SwPotential<double> spot(sp);
  for (bool pbc : {false, true}) {
    core::Box box;
    auto init = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
    box.periodic = {true, true, pbc};
    const long steps = 3; const double dt = 0.0004;

    core::AtomSoA<double> oref = init;
    serial_vv(oref, steps, dt, [&](core::AtomSoA<double>& a) { pot::sw_direct_fp64(a, box, sp, true); });
    core::AtomSoA<double> ring = init;
    pot::run_sw_ring(ring, box, spot, ring_opts(steps, pbc ? 6 : 5, pbc ? 3 : 4, dt));
    EXPECT_LT(max_state_dev(ring, oref), 1e-6) << "ring ≠ FP64-oracle trajectory (pbc=" << pbc << ")";
  }
}

// G-PE-ring ⭐ — the ring's per-step PE matches the serial sw_zone_pass AND the independent
// FP64 oracle. MEASURE-FIRST FINDING (overturns the T3-acceptance Gap-A premise that the
// global key "becomes load-bearing in T3b"): on the id-shuffled fixture the ring window IS
// unsorted (count_key_flips finds φ₂ pairs whose global-key order != local order), YET the
// local-key POISON gives a BITWISE-IDENTICAL PE. Reason: a φ₂ pair spans only ADJACENT zones
// (rcut < zone width), and eam_window_layout emits [pred][center][succ], so any two adjacent
// zones keep the same relative block order in BOTH windows (the cyclic PREDECESSOR first —
// e.g. at the seam zone n-1 precedes 0 in both {n-2,n-1,0} and {n-1,0,1}) ⇒ the local-index
// gate counts each pair exactly once, same as the global key. The flips change WHICH zone
// attributes the energy, not WHETHER it is counted once (order-free int64 EnergyAccum ⇒ the
// fired-pair multiset is bitwise-identical). So the φ₂ global key is DEFENSIVE
// (robust to any future gather reorder), NOT load-bearing on this gather. We keep it (free,
// obviously-correct), and this gate documents the finding instead of a non-existent bug.
TEST(SwRing, RingPeMatchesOracleAndGlobalKeyIsCountOnceInvariant) {
  pot::SwParams sp; pot::SwPotential<double> spot(sp);
  core::Box box;
  auto base = make_diamond_si(2, 2, kNz, 5.431, 0.30, box);
  box.periodic = {true, true, true};
  auto init = shuffle_atoms(base, 2024);  // break zone-id ↔ atom-id correlation

  const auto zd = core::ZoneDecomposition::build(init, box, 6, sp.rcut(), 2);
  const auto [flips, total] = count_key_flips(init, box, zd, sp);
  ASSERT_GT(total, 0);
  EXPECT_GT(flips, 0) << "window fully sorted — the count-once-invariance finding is vacuous";

  const long steps = 1; const double dt = 0.0004;
  core::AtomSoA<double> ring = init;
  const auto rc = pot::run_sw_ring(ring, box, spot, ring_opts(steps, 6, 3, dt));
  core::AtomSoA<double> poison = init;
  const auto rp = pot::run_sw_ring(poison, box, spot, ring_opts(steps, 6, 3, dt),
                                   pot::SwWinForceLocalKey<double>(spot.sw));

  // after 1 full step, final positions == the force-pass positions ⇒ a serial pass there
  // gives the same total energy (each pair/triplet counted once).
  core::AtomSoA<double> at1 = ring;
  const double pe_serial = pot::sw_zone_pass(at1, box,
      core::ZoneDecomposition::build(at1, box, 6, sp.rcut(), 2), sp).pe;
  core::AtomSoA<double> at1o = ring;
  const double pe_oracle = pot::sw_direct_fp64(at1o, box, sp, false).pe;

  ASSERT_EQ(rc.stats.size(), 1u);
  ASSERT_EQ(rp.stats.size(), 1u);
  EXPECT_EQ(rc.stats[0].pe, pe_serial) << "ring PE ≠ serial sw_zone_pass PE";
  EXPECT_NEAR(rc.stats[0].pe, pe_oracle, 1e-6) << "ring PE ≠ independent oracle PE";
  EXPECT_TRUE(state_bitwise_equal(poison, ring)) << "the energy gate must not touch forces";
  // THE FINDING: despite `flips` key-order flips, the local-key poison PE is BITWISE-IDENTICAL
  // ⇒ count-once-invariant ⇒ the global key is defensive, not load-bearing on the slot gather.
  EXPECT_EQ(rp.stats[0].pe, rc.stats[0].pe)
      << flips << "/" << total << " key-order flips, yet expected bitwise-identical PE";
}

// G6 — anti-deadlock across node counts (free-z).
TEST(SwRing, AntiDeadlock) {
  pot::SwParams sp; pot::SwPotential<double> spot(sp);
  core::Box box;
  auto init = make_diamond_si(2, 2, kNz, 5.431, 0.20, box);
  box.periodic = {true, true, false};
  for (int z = 1; z <= 6; ++z) {
    core::AtomSoA<double> a = init;
    const auto r = pot::run_sw_ring(a, box, spot, ring_opts(2 * z + 3, 6, z, 0.0004));
    EXPECT_EQ(int(r.halt), int(core::Halt::None)) << "halt at z=" << z;
    EXPECT_EQ(r.steps_done, 2 * z + 3) << "incomplete at z=" << z;
  }
}

// G7 — the INVERTED firewall: SwWinForce accepts SW (φ₃ needs_transpose), rejects bad
// descriptors; the EAM GPU firewall still refuses SW (we did not weaken it).
TEST(SwRing, FirewallAcceptsTransposeRejectsBad) {
  using pot::PassDecl; using pot::PassKind;
  pot::SwPotential<double> spot{};
  EXPECT_NO_THROW(pot::SwWinForce<double>::assert_supported(spot.passes()));  // SW = accept

  const PassDecl eam3[3] = {{PassKind::Density, true, false, false, 44},
                            {PassKind::Embedding, false, false, false, 30},
                            {PassKind::Force, true, false, false, 40}};
  EXPECT_THROW(pot::SwWinForce<double>::assert_supported(eam3), std::runtime_error);  // wrong count
  PassDecl it[2] = {{PassKind::Force, true, false, false, 40},
                    {PassKind::Force, true, true, false, 40}};
  it[0].iterative = true;
  EXPECT_THROW(pot::SwWinForce<double>::assert_supported(it), std::runtime_error);  // iterative
  const PassDecl sym[2] = {{PassKind::Force, true, false, false, 40},
                           {PassKind::Force, true, false, false, 40}};  // φ₃ NOT transpose
  EXPECT_THROW(pot::SwWinForce<double>::assert_supported(sym), std::runtime_error);  // symmetric trap
}
