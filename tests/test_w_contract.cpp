// PR-0a — the W-contract descriptor + donation_layout + ledger + firewall teeth.
// The design of record: the workflow-merged PR0A_DESIGN (audit §3.1/§3.3/§5). Every
// test carries a kill-mutation in its comment (the tooth is non-vacuous — the project
// has FIVE structurally-dead-fixture recidives: Te1 fc_d, Me1 partial-S, Me5 taper,
// Me5b OOB, M4-S). CPU-only, no CUDA/LAMMPS.
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <set>
#include <span>
#include <vector>

#include "tdmd/potentials/eam.hpp"        // kEamPassDecls, eam_pass_decls, assert_eam_symmetric_passes
#include "tdmd/potentials/eam_analytic.hpp"  // AnalyticEam (cheap Math for T-SRC)
#include "tdmd/potentials/eam_ring.hpp"   // CpuEamWindowForce (T-CPT positive)
#include "tdmd/potentials/eam_zone.hpp"   // donation_layout, want_closure_mask, helpers
#include "tdmd/potentials/many_body.hpp"  // PassDecl, WClass, validate_pass_decls, concept
#include "tdmd/potentials/meam_ring.hpp"      // MeamPotential + MeamWinForce (T-CPT)
#include "tdmd/potentials/sw_ring.hpp"        // SwPotential + SwWinForce (T-CPT)
#include "tdmd/potentials/tersoff_ring.hpp"   // TersoffPotential + TersoffWinForce (T-CPT)

using namespace tdmd::potentials;
namespace core = tdmd::core;

namespace {
PassDecl mk(PassKind k, int fb = 40, WClass w = WClass::kCOnly, uint8_t roles = 0,
            bool transpose = false, bool iterative = false) {
  PassDecl d{k, true, transpose, iterative, fb, w, roles};
  return d;
}
constexpr uint8_t kAllRoles =
    donor_bit(DonorRole::kSelf) | donor_bit(DonorRole::kCrossLo) | donor_bit(DonorRole::kCrossHi);
}  // namespace

// ---------------------------------------------------------------------------
// validate_pass_decls firewall teeth (§2.1)
// ---------------------------------------------------------------------------

// T1 — kAccumB1 requires accum_fracbits>0 (int64 FixedAccum is WHAT makes W-1 hold).
// KILL: delete the fb>0 rule → this passes.
TEST(WContract, ValidateRejectsAccumB1WithoutFracbits) {
  std::array p{mk(PassKind::Density, /*fb*/ 0, WClass::kAccumB1, donor_bit(DonorRole::kSelf))};
  EXPECT_THROW(validate_pass_decls(p), std::runtime_error);
}

// T2 — kAccumB1 forbidden on BondOrder (ζ is an FP64 fold — the kAccumCanonSuffix class).
// KILL: delete the BondOrder rule → this passes.
TEST(WContract, ValidateRejectsAccumB1OnBondOrder) {
  std::array p{mk(PassKind::BondOrder, /*fb*/ 40, WClass::kAccumB1, donor_bit(DonorRole::kSelf))};
  EXPECT_THROW(validate_pass_decls(p), std::runtime_error);
}

// T3 — the five structural guards. KILL any one → its EXPECT_THROW fails.
TEST(WContract, ValidateGuards) {
  // (a) size > 8
  std::vector<PassDecl> nine(9, mk(PassKind::Force));
  EXPECT_THROW(validate_pass_decls(nine), std::runtime_error);
  // (a') empty
  EXPECT_THROW(validate_pass_decls(std::span<const PassDecl>{}), std::runtime_error);
  // (b) reserved donor bit (kReserved3 = bit 3)
  std::array b{mk(PassKind::Density, 44, WClass::kAccumB1, uint8_t(donor_bit(DonorRole::kReserved3) |
                                                                    donor_bit(DonorRole::kSelf)))};
  EXPECT_THROW(validate_pass_decls(b), std::runtime_error);
  // (c) donor_roles != 0 on a kCOnly pass
  std::array c{mk(PassKind::Force, 40, WClass::kCOnly, donor_bit(DonorRole::kSelf))};
  EXPECT_THROW(validate_pass_decls(c), std::runtime_error);
  // (d) kAccumB1 with empty donor_roles
  std::array d{mk(PassKind::Density, 44, WClass::kAccumB1, /*roles*/ 0)};
  EXPECT_THROW(validate_pass_decls(d), std::runtime_error);
  // (e) iterative pass that is not kCOnly
  std::array e{mk(PassKind::QeqIter, 40, WClass::kAccumB1, donor_bit(DonorRole::kSelf),
                  /*transpose*/ false, /*iterative*/ true)};
  EXPECT_THROW(validate_pass_decls(e), std::runtime_error);
}

// T-VAC — the POSITIVE anchor: all shipped descriptors + a legal synthetic kAccumB1 PASS
// validation. KILL: degenerate validate into "always throws" (anti-vacuity for T1–T3).
TEST(WContract, ValidateAcceptsAllShippedDescriptors) {
  EXPECT_NO_THROW(validate_pass_decls(eam_pass_decls()));
  EXPECT_NO_THROW(validate_pass_decls(SwPotential<double>().passes()));
  EXPECT_NO_THROW(validate_pass_decls(TersoffPotential<double>().passes()));
  EXPECT_NO_THROW(validate_pass_decls(MeamPotential<double>().passes()));
  std::array legal{mk(PassKind::Density, 44, WClass::kAccumB1, kAllRoles)};
  EXPECT_NO_THROW(validate_pass_decls(legal));
  std::array canon{mk(PassKind::BondOrder, 0, WClass::kAccumCanonSuffix, /*roles*/ 0)};
  EXPECT_NO_THROW(validate_pass_decls(canon));  // legal future value, gated by its consumer PR
}

// T-INERT — every shipped descriptor is inert (kCOnly + no roles). KILL: flip any shipped
// tail field, OR insert a field in the MIDDLE of PassDecl (the compile-time part).
TEST(WContract, ShippedDescriptorsAreInert) {
  auto check = [](std::span<const PassDecl> ps) {
    for (const auto& d : ps) {
      EXPECT_EQ(d.w_class, WClass::kCOnly);
      EXPECT_EQ(d.donor_roles, 0);
    }
  };
  check(eam_pass_decls());
  check(SwPotential<double>().passes());
  check(TersoffPotential<double>().passes());
  check(MeamPotential<double>().passes());
  // compile-time: the tail fields default and a 5-positional aggregate still builds
  // (breaks if a field is inserted mid-struct — the positional-init hazard).
  constexpr PassDecl five{PassKind::Force, true, false, false, 40};
  static_assert(five.w_class == WClass::kCOnly, "w_class must default to kCOnly");
  static_assert(five.donor_roles == 0, "donor_roles must default to 0");
}

// ---------------------------------------------------------------------------
// donation_layout schedule (§3.1)
// ---------------------------------------------------------------------------
namespace {
// helper: collect all self slots and cross pairs over a full pass, per (n,pbc).
struct Schedule {
  std::vector<int> selfs;
  std::vector<std::pair<int, int>> crosses;  // normalized {min,max} EXCEPT the seam, which
                                             // is stored as emitted (n-1, 0)
};
Schedule collect(int n, bool pbc) {
  Schedule s;
  for (int j = 0; j < n; ++j) {
    const DonationBatches b = donation_layout(j, n, pbc);
    for (int i = 0; i < b.n_self; ++i) s.selfs.push_back(b.self[i]);
    for (int i = 0; i < b.n_cross; ++i) s.crosses.push_back({b.cross[i][0], b.cross[i][1]});
  }
  return s;
}
}  // namespace

// T4 — exactly once per pair (INV-8 generalization). KILL: duplicate/drop any batch.
TEST(WContract, DonationLayoutExactlyOncePerPair) {
  auto run = [](int n, bool pbc) {
    const Schedule s = collect(n, pbc);
    // selfs = {0..n-1} each exactly once
    std::multiset<int> selfset(s.selfs.begin(), s.selfs.end());
    for (int k = 0; k < n; ++k) EXPECT_EQ(selfset.count(k), 1u) << "n=" << n << " self k=" << k;
    EXPECT_EQ(s.selfs.size(), std::size_t(n));
    // crosses = adjacent pairs, each once (n==1 has no pair even under PBC — a zone
    // cannot cross with itself; donation_layout returns just self(0))
    const std::size_t want = (n == 1) ? 0u : (pbc ? std::size_t(n) : std::size_t(n - 1));
    std::set<std::pair<int, int>> uniq;
    for (auto [a, c] : s.crosses) uniq.insert({std::min(a, c), std::max(a, c)});
    EXPECT_EQ(uniq.size(), want) << "n=" << n << " pbc=" << pbc;
    EXPECT_EQ(s.crosses.size(), want) << "duplicate cross batch";
  };
  for (int n = 1; n <= 8; ++n) run(n, false);
  run(1, true);
  for (int n = 5; n <= 8; ++n) run(n, true);
}

// T5 ⭐ — the PBC seam (n-1,0) is emitted EXACTLY at j=n-1 and nowhere else, so by the
// call contract (batches BEFORE finalize within a position) it precedes finalize(n-1).
// KILL: move the seam emission "to the tail" (the refuted variant) or drop it.
TEST(WContract, PbcSeamFiresBeforeFinalizeN1) {
  for (int n = 5; n <= 10; ++n) {
    int seam_pos = -1, seam_count = 0;
    for (int j = 0; j < n; ++j) {
      const DonationBatches b = donation_layout(j, n, true);
      for (int i = 0; i < b.n_cross; ++i) {
        const int a = b.cross[i][0], c = b.cross[i][1];
        if ((a == n - 1 && c == 0) || (a == 0 && c == n - 1)) { seam_pos = j; ++seam_count; }
      }
    }
    EXPECT_EQ(seam_count, 1) << "seam must be emitted exactly once, n=" << n;
    EXPECT_EQ(seam_pos, n - 1) << "seam must be emitted at j=n-1 (before finalize(n-1)), n=" << n;
  }
}

// T5b — feasibility: every emitted self ∈ arrived {0..j+1}; every cross endpoint is in the
// running self-union up to and including position j. KILL: emit self(j+2) or a cross before
// its endpoint's self.
TEST(WContract, DonationLayoutFeasibility) {
  auto run = [](int n, bool pbc) {
    std::set<int> self_union;
    for (int j = 0; j < n; ++j) {
      const DonationBatches b = donation_layout(j, n, pbc);
      for (int i = 0; i < b.n_self; ++i) {
        EXPECT_LE(b.self[i], j + 1) << "self beyond arrived, n=" << n << " j=" << j;
        EXPECT_GE(b.self[i], 0);
        self_union.insert(b.self[i]);
      }
      for (int i = 0; i < b.n_cross; ++i) {
        EXPECT_TRUE(self_union.count(b.cross[i][0])) << "cross endpoint before self, n=" << n;
        EXPECT_TRUE(self_union.count(b.cross[i][1])) << "cross endpoint before self, n=" << n;
      }
    }
  };
  for (int n = 1; n <= 8; ++n) run(n, false);
  run(1, true);
  for (int n = 5; n <= 8; ++n) run(n, true);
}

// T6 — free edges drop; n==1 is a lone self; pbc n∈{2,3,4} throws. KILL: emit a seam in free /
// remove the pbc small-n guard.
TEST(WContract, FreeEdgesDropAndN1AndSmallPbcThrow) {
  // free j=n-1 → no cross (the top edge has no successor)
  for (int n = 2; n <= 6; ++n) {
    const DonationBatches b = donation_layout(n - 1, n, false);
    EXPECT_EQ(b.n_cross, 0) << "free top zone must not cross, n=" << n;
    EXPECT_EQ(b.n_self, 0) << "free top zone self already emitted at j=n-2, n=" << n;
  }
  // n==1 (free and pbc) → a single self(0), no cross
  for (bool pbc : {false, true}) {
    const DonationBatches b = donation_layout(0, 1, pbc);
    EXPECT_EQ(b.n_self, 1);
    EXPECT_EQ(b.self[0], 0);
    EXPECT_EQ(b.n_cross, 0);
  }
  // pbc n∈{2,3,4} → throw (eam_window_layout would silently duplicate slots)
  for (int n : {2, 3, 4}) EXPECT_THROW(donation_layout(0, n, true), std::invalid_argument);
  // out-of-range j
  EXPECT_THROW(donation_layout(5, 5, false), std::invalid_argument);
  EXPECT_THROW(donation_layout(-1, 5, false), std::invalid_argument);
}

// T-SIM ⭐ — the main anti-off-by-one: an INDEPENDENT scan-event simulation (NOT the ring
// code) verifies every batch's readers finalize at a position ≥ its emission. arrival/drift
// of slot s at scan max(0,s-1); finalize(j) in-scan at j, EXCEPT pbc j=0 → tail (defer_head,
// modeled as position n). ALSO (acceptance SHOULD, self-sufficiency): the normative-table
// equality emit == ready for EVERY batch EXCEPT the pbc seam, whose emit(n-1) == deadline
// while ready = n-2 — a late self/cross (emit > ready) now fails HERE directly, not only
// via T5b's cross-endpoint dependency. KILL: any ±1 in the emission or the table.
TEST(WContract, ScheduleConsistentWithScanSimulation) {
  auto fin_pos = [](int j, int n, bool pbc) { return (pbc && j == 0) ? n : j; };
  auto run = [&](int n, bool pbc) {
    for (int j = 0; j < n; ++j) {
      const DonationBatches b = donation_layout(j, n, pbc);
      const int emit = j;
      for (int i = 0; i < b.n_self; ++i) {
        const int k = b.self[i];
        const int ready = std::max(0, k - 1);
        EXPECT_EQ(emit, ready) << "self emit != ready (normative table), n=" << n << " k=" << k;
        EXPECT_LE(emit, fin_pos(k, n, pbc)) << "self read before emit, n=" << n << " k=" << k;
      }
      for (int i = 0; i < b.n_cross; ++i) {
        const int a = b.cross[i][0], c = b.cross[i][1];
        const bool seam = pbc && n > 1 && ((a == n - 1 && c == 0) || (a == 0 && c == n - 1));
        // non-seam cross(k,k+1): ready = k (slot k+1 arrives+drifts at scan k) ⇒ emit == k;
        // the seam is the ONE exception: ready = n-2, emit = deadline = n-1.
        EXPECT_EQ(emit, seam ? n - 1 : std::min(a, c))
            << "cross emit != table, n=" << n << " {" << a << "," << c << "}";
        EXPECT_LE(std::max(0, a - 1), emit) << "cross arrival a, n=" << n;
        EXPECT_LE(std::max(0, c - 1), emit) << "cross arrival c, n=" << n;
        EXPECT_LE(emit, fin_pos(a, n, pbc)) << "cross read a before emit, n=" << n;
        EXPECT_LE(emit, fin_pos(c, n, pbc)) << "cross read c before emit, n=" << n;
      }
    }
  };
  for (int n = 1; n <= 16; ++n) run(n, false);
  run(1, true);
  for (int n = 5; n <= 16; ++n) run(n, true);
}

// T8 — the finalize(j) window (eam_window_layout) is closed by the schedule: the incident
// edges (j-1,j),(j,j+1) are emitted at/before fin_pos(j); the (j+1,j+2) edge is NOT required
// (L-CLOSE openness). KILL: require (j+1,j+2) / lose self(j-1).
TEST(WContract, ConsistentWithEamWindowLayout) {
  auto emit_pos_of_cross = [](int lo, int hi, int n, bool pbc) -> int {
    // find the position j at which donation_layout emits the unordered edge {lo,hi}
    for (int j = 0; j < n; ++j) {
      const DonationBatches b = donation_layout(j, n, pbc);
      for (int i = 0; i < b.n_cross; ++i) {
        const int a = b.cross[i][0], c = b.cross[i][1];
        if ((a == lo && c == hi) || (a == hi && c == lo)) return j;
      }
    }
    return -1;
  };
  auto fin_pos = [](int j, int n, bool pbc) { return (pbc && j == 0) ? n : j; };
  auto run = [&](int n, bool pbc) {
    for (int j = 0; j < n; ++j) {
      int w[3];
      const int nw = eam_window_layout(j, n, pbc, w);
      (void)nw;
      // incident edges of j (slot space; pbc wraps)
      const bool has_lo = pbc ? true : j > 0;
      const bool has_hi = pbc ? true : j + 1 < n;
      if (has_lo) {
        const int lo = pbc ? (j - 1 + n) % n : j - 1;
        const int p = emit_pos_of_cross(lo, j, n, pbc);
        EXPECT_GE(p, 0) << "lower edge missing, n=" << n << " j=" << j;
        EXPECT_LE(p, fin_pos(j, n, pbc)) << "lower edge not closed by finalize(j)";
      }
      if (has_hi) {
        const int hi = pbc ? (j + 1) % n : j + 1;
        const int p = emit_pos_of_cross(j, hi, n, pbc);
        EXPECT_GE(p, 0) << "upper edge missing, n=" << n << " j=" << j;
        EXPECT_LE(p, fin_pos(j, n, pbc)) << "upper edge not closed by finalize(j)";
      }
    }
  };
  for (int n = 2; n <= 8; ++n) run(n, false);
  for (int n = 5; n <= 8; ++n) run(n, true);
}

// ---------------------------------------------------------------------------
// want_closure_mask / ledger (§4)
// ---------------------------------------------------------------------------

// T9 — all-kCOnly shipped descriptors → want mask 0 everywhere (the F-NOOP anchor).
// KILL: derive the mask from kind instead of w_class, or leak a bit from a kCOnly pass.
TEST(WContract, WantMaskAllCOnlyIsZero) {
  auto run = [](std::span<const PassDecl> ps, int n, bool pbc) {
    for (int j = 0; j < n; ++j) EXPECT_EQ(want_closure_mask(ps, j, n, pbc), 0u);
  };
  for (int n = 1; n <= 6; ++n) {
    run(eam_pass_decls(), n, false);
    run(SwPotential<double>().passes(), n, false);
    run(TersoffPotential<double>().passes(), n, false);
    run(MeamPotential<double>().passes(), n, false);
  }
  run(eam_pass_decls(), 5, true);
}

// T10 — synthetic kAccumB1 [Density, roles=self|lo|hi]: interior→3 bits; free j=0 no lo;
// free j=n-1 no hi; pbc→all 3 per j; pbc n=1→only self; positions checked via closure_bit
// AND raw constants (double-entry catches a packing mutation); 2 passes→different nibbles.
// KILL: change the packing (4↔8, role↔pass) / lose the edge-dropping.
TEST(WContract, WantMaskSyntheticAccumB1) {
  const std::array one{mk(PassKind::Density, 44, WClass::kAccumB1, kAllRoles)};
  const uint32_t self0 = closure_bit(0, DonorRole::kSelf);
  const uint32_t lo0 = closure_bit(0, DonorRole::kCrossLo);
  const uint32_t hi0 = closure_bit(0, DonorRole::kCrossHi);
  // raw double-entry: pass 0 nibble = bits {0,1,2}
  EXPECT_EQ(self0, 1u << 0);
  EXPECT_EQ(lo0, 1u << 1);
  EXPECT_EQ(hi0, 1u << 2);
  // free interior (n=5, j=2): all three
  EXPECT_EQ(want_closure_mask(one, 2, 5, false), self0 | lo0 | hi0);
  // free j=0: no LO (no lower neighbour)
  EXPECT_EQ(want_closure_mask(one, 0, 5, false), self0 | hi0);
  // free j=n-1: no HI
  EXPECT_EQ(want_closure_mask(one, 4, 5, false), self0 | lo0);
  // pbc: every j has all three
  for (int j = 0; j < 5; ++j) EXPECT_EQ(want_closure_mask(one, j, 5, true), self0 | lo0 | hi0);
  // pbc n=1: cyclic degenerate → only self (no distinct neighbour), still self bit set
  EXPECT_EQ(want_closure_mask(one, 0, 1, true), self0);
  // two kAccumB1 passes → bits land in DIFFERENT nibbles (pass 0 vs pass 1)
  const std::array two{mk(PassKind::Density, 44, WClass::kAccumB1, donor_bit(DonorRole::kSelf)),
                       mk(PassKind::Force, 40, WClass::kAccumB1, donor_bit(DonorRole::kSelf))};
  const uint32_t m = want_closure_mask(two, 2, 5, false);
  EXPECT_EQ(m, closure_bit(0, DonorRole::kSelf) | closure_bit(1, DonorRole::kSelf));
  EXPECT_EQ(closure_bit(1, DonorRole::kSelf), 1u << 4);  // nibble stride = 4
  // >8 passes → throw (through validate inside want_closure_mask)
  std::vector<PassDecl> nine(9, mk(PassKind::Force));
  EXPECT_THROW(want_closure_mask(nine, 0, 5, false), std::runtime_error);
  // the domain guard (shared with donation_layout) fires THROUGH want_closure_mask too:
  // pbc n∈{2..4} and j out of range (acceptance SHOULD — the guard was otherwise unmapped)
  EXPECT_THROW(want_closure_mask(one, 0, 3, true), std::invalid_argument);
  EXPECT_THROW(want_closure_mask(one, 5, 5, false), std::invalid_argument);
}

// T-CROSS — close_cross_batch marks CrossHi in the LO zone, CrossLo in the HI zone.
// KILL: mirror the roles.
TEST(WContract, CloseCrossBatchMarksBothZones) {
  uint32_t lo = 0, hi = 0;
  close_cross_batch(lo, hi, /*pass*/ 0);
  EXPECT_EQ(lo, closure_bit(0, DonorRole::kCrossHi));
  EXPECT_EQ(hi, closure_bit(0, DonorRole::kCrossLo));
}

// T-ROT — rotation helpers match the ring convention r=(h-1)%n over 2 full turns, n=5.
// KILL: h%n instead of (h-1)%n.
TEST(WContract, RotationHelpersMatchRingConvention) {
  EXPECT_EQ(pass_rotation(1, 5, false), 0);   // free: r always 0
  EXPECT_EQ(pass_rotation(7, 5, false), 0);
  const int expect_r[11] = {/*h=1*/ 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0};  // (h-1)%5, h=1..11
  for (int h = 1; h <= 11; ++h) EXPECT_EQ(pass_rotation(h, 5, true), expect_r[h - 1]) << "h=" << h;
  // slot_zone_id(r, slot, n) = (r+slot)%n
  EXPECT_EQ(slot_zone_id(3, 4, 5), 2);
  EXPECT_EQ(slot_zone_id(0, 0, 5), 0);
}

// ---------------------------------------------------------------------------
// the shared EAM gate + single source (§5)
// ---------------------------------------------------------------------------

// T-EAM — assert_eam_symmetric_passes accepts EAM, rejects count/kind/transpose/iterative on
// EVERY slot (anti off-by-one loop), and rejects a wclass-illegal-but-shape-legal descriptor
// THROUGH the gate (proving validate runs inside it). KILL: gut any branch / drop the validate call.
TEST(WContract, AssertEamSymmetricPassesTeeth) {
  EXPECT_NO_THROW(assert_eam_symmetric_passes(eam_pass_decls(), "t"));
  // wrong count
  std::array two{mk(PassKind::Density, 44), mk(PassKind::Force)};
  EXPECT_THROW(assert_eam_symmetric_passes(two, "t"), std::runtime_error);
  std::array four{mk(PassKind::Density, 44), mk(PassKind::Embedding, 30), mk(PassKind::Force),
                  mk(PassKind::Force)};
  EXPECT_THROW(assert_eam_symmetric_passes(four, "t"), std::runtime_error);
  // wrong kind at each slot
  for (int slot = 0; slot < 3; ++slot) {
    std::array<PassDecl, 3> p{mk(PassKind::Density, 44), mk(PassKind::Embedding, 30),
                              mk(PassKind::Force)};
    p[slot].kind = PassKind::BondOrder;
    EXPECT_THROW(assert_eam_symmetric_passes(p, "t"), std::runtime_error) << "slot " << slot;
  }
  // needs_transpose at each slot
  for (int slot = 0; slot < 3; ++slot) {
    std::array<PassDecl, 3> p{mk(PassKind::Density, 44), mk(PassKind::Embedding, 30),
                              mk(PassKind::Force)};
    p[slot].needs_transpose = true;
    EXPECT_THROW(assert_eam_symmetric_passes(p, "t"), std::runtime_error) << "slot " << slot;
  }
  // iterative at each slot
  for (int slot = 0; slot < 3; ++slot) {
    std::array<PassDecl, 3> p{mk(PassKind::Density, 44), mk(PassKind::Embedding, 30),
                              mk(PassKind::Force)};
    p[slot].iterative = true;
    EXPECT_THROW(assert_eam_symmetric_passes(p, "t"), std::runtime_error) << "slot " << slot;
  }
  // wclass-illegal but shape-legal (Embedding.donor_roles set on kCOnly) → throw THROUGH the gate
  std::array<PassDecl, 3> bad{mk(PassKind::Density, 44), mk(PassKind::Embedding, 30),
                              mk(PassKind::Force)};
  bad[1].donor_roles = donor_bit(DonorRole::kSelf);  // roles on a kCOnly pass → validate throws
  EXPECT_THROW(assert_eam_symmetric_passes(bad, "t"), std::runtime_error);
}

// T-SRC — EamPotential::passes() IS kEamPassDecls (one source of truth, no duplicate).
// KILL: re-duplicate the descriptor into the class.
TEST(WContract, EamPassDeclsSingleSource) {
  EamPotential<double, AnalyticEam<double>> pot{AnalyticEam<double>{}};
  EXPECT_EQ(pot.passes().data(), kEamPassDecls);
  EXPECT_EQ(pot.passes().size(), 3u);
  EXPECT_EQ(eam_pass_decls().data(), kEamPassDecls);
}

// ---------------------------------------------------------------------------
// T-CPT — concept teeth (compile-time). POSITIVE: the real policies model it.
// NEGATIVE: local bad policies do NOT — exactly the "silent bypass" the promotion kills.
// ---------------------------------------------------------------------------
namespace {
struct BadPolicyNoAssert {  // has compute, MISSING assert_supported
  void compute(const double*, const double*, const double*, const long*, int, const int*, int,
               const core::PairGeom&, double, std::vector<core::fixed::ForceAccum>&,
               std::vector<core::fixed::ForceAccum>&, std::vector<core::fixed::ForceAccum>&,
               core::fixed::EnergyAccum&, double&) const {}
};
struct BadPolicyNonStatic {  // assert_supported is NON-static
  void assert_supported(std::span<const PassDecl>) const {}
  void compute(const double*, const double*, const double*, const long*, int, const int*, int,
               const core::PairGeom&, double, std::vector<core::fixed::ForceAccum>&,
               std::vector<core::fixed::ForceAccum>&, std::vector<core::fixed::ForceAccum>&,
               core::fixed::EnergyAccum&, double&) const {}
};
struct BadPolicyWrongArity {  // compute with the wrong arity
  static void assert_supported(std::span<const PassDecl>) {}
  void compute(const double*) const {}
};
}  // namespace
static_assert(WindowForcePolicy<CpuEamWindowForce<AnalyticEam<double>>>,
              "the CPU EAM policy must model the concept (its assert_supported is new in PR-0a)");
static_assert(WindowForcePolicy<SwWinForce<double>>, "SW policy must model the concept");
static_assert(WindowForcePolicy<TersoffWinForce<double>>, "Tersoff policy must model the concept");
static_assert(WindowForcePolicy<MeamWinForce<double>>, "MEAM policy must model the concept");
static_assert(WindowForcePolicy<SwWinForceLocalKey<double>>,
              "the SW poison policy must model the concept (delegating assert_supported)");
static_assert(WindowForcePolicy<TersoffWinForceLocalKey<double>>,
              "the Tersoff poison policy must model the concept (delegating assert_supported)");
static_assert(!WindowForcePolicy<BadPolicyNoAssert>, "missing assert_supported must NOT model");
static_assert(!WindowForcePolicy<BadPolicyNonStatic>, "non-static assert_supported must NOT model");
static_assert(!WindowForcePolicy<BadPolicyWrongArity>, "wrong compute arity must NOT model");

TEST(WContract, ConceptTeethCompiled) { SUCCEED() << "the static_asserts above are the tooth"; }
