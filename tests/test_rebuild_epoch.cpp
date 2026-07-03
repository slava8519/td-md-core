// PR-0c — the lazy-materialization EPOCH primitive (core/rebuild_epoch.hpp) + the
// device-free ZoneNeighborStore scaffold (cuda/zone_nbr_store.cuh). Pure host, testable
// now (like PR-0b's skin_budget); PR-4 consumes it in the live MPI ring. Each test carries
// its kill-mutation. Including zone_nbr_store.cuh from a g++ CPU TU mechanically enforces
// that the scaffold stays device-free.
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "tdmd/core/rebuild_epoch.hpp"
#include "tdmd/cuda/zone_nbr_store.cuh"  // the scaffold: re-export + §3.5 contract (device-free)

using tdmd::core::rebuild_epoch::advance;
using tdmd::core::rebuild_epoch::stale;

// T1 — advance ticks IFF rebuild. KILL: unconditional +1 (tick without rebuild) ⇒ the
// false branch RED; or (rebuild?0:1) ⇒ both branches RED.
TEST(RebuildEpoch, AdvanceTicksIffRebuild) {
  EXPECT_EQ(advance(0u, false), 0u);
  EXPECT_EQ(advance(0u, true), 1u);
  EXPECT_EQ(advance(7u, false), 7u);
  EXPECT_EQ(advance(7u, true), 8u);
}

// T2 — monotone non-decreasing over a canonical rebuild-bool sequence; ticks exactly on
// true. KILL: any decrement / reset-to-0 on a non-rebuild pass ⇒ RED.
TEST(RebuildEpoch, MonotoneNonDecreasing) {
  const std::vector<bool> seq{true, false, false, true, true, false, true, false, false};
  std::uint32_t e = 0, ticks = 0;
  for (bool r : seq) {
    const std::uint32_t prev = e;
    e = advance(e, r);
    EXPECT_GE(e, prev) << "epoch must never decrease";
    ticks += r ? 1u : 0u;
    EXPECT_EQ(e, ticks) << "epoch must equal the running rebuild count";
  }
  EXPECT_EQ(e, 4u);
}

// T3 — the L-SUP bridge: advance(prev,r) > prev  IFF  r. KILL: a tick on the wrong
// condition (e.g. advance ignoring `rebuild`) breaks the equivalence.
TEST(RebuildEpoch, EpochRebuildConsistency) {
  for (std::uint32_t prev : {0u, 1u, 5u, 4000000u})
    for (bool r : {false, true}) {
      const bool strictly_grew = advance(prev, r) > prev;
      EXPECT_EQ(strictly_grew, r)
          << "epoch strictly grows IFF a rebuild occurred (prev=" << prev << " r=" << r << ")";
    }
}

// T4 — the stale predicate is STRICT less: equal reuse, lagging rebuild, ahead reuse
// (total). KILL: < -> <= ⇒ stale(k,k) RED (rebuild storm, laziness lost); < -> != ⇒
// stale(k+1,k) RED.
TEST(RebuildEpoch, StalePredicateStrictLess) {
  EXPECT_FALSE(stale(5u, 5u)) << "equal epochs REUSE (laziness)";
  EXPECT_TRUE(stale(4u, 5u)) << "lagging epoch REBUILDS (conservativeness)";
  EXPECT_FALSE(stale(6u, 5u)) << "ahead-of-head is total/robust ⇒ reuse";
  EXPECT_TRUE(stale(0u, 1u));
  EXPECT_FALSE(stale(0u, 0u));
}

// T5 — integration: the head advances epochs over a rebuild-decision sequence; a lagging
// rank calls stale() and catches up (local=hdr). Assert: the rank materializes EXACTLY on
// the passes the head rebuilt (never skips a mandatory rebuild — conservativeness; never
// rebuilds on a reuse pass — laziness). KILL: <= ⇒ rebuild storm (materializes on reuse);
// advance without a tick ⇒ the rank lives with a stale list (misses a rebuild).
TEST(RebuildEpoch, LazyReuseConservativeness) {
  const std::vector<bool> head_rebuilds{true, false, true, true, false, false, true, false};
  std::uint32_t head_epoch = 0, rank_local = 0;
  std::size_t rank_materializations = 0, head_rebuild_count = 0;
  for (bool r : head_rebuilds) {
    head_epoch = advance(head_epoch, r);        // head decides + broadcasts
    if (r) ++head_rebuild_count;
    if (stale(rank_local, head_epoch)) {        // rank sees the broadcast epoch
      ++rank_materializations;                  // ... and (re)materializes its list
      rank_local = head_epoch;                  // ... catching up to the head
    }
    // INVARIANT every pass: after servicing, the rank is never behind the head
    EXPECT_FALSE(stale(rank_local, head_epoch)) << "rank must not lag after servicing";
  }
  // the rank rebuilt once per head-rebuild pass — no storm, no skip
  EXPECT_EQ(rank_materializations, head_rebuild_count)
      << "materializations must equal head rebuilds (conservativeness + laziness)";
  EXPECT_EQ(rank_materializations, 4u);
}

// T6 — purity/determinism (proxy for z-independence: rebuild_epoch is a z-independent
// broadcast scalar). Identical inputs ⇒ identical outputs.
TEST(RebuildEpoch, DeterministicPure) {
  for (std::uint32_t e : {0u, 3u, 99u})
    for (bool r : {false, true}) {
      EXPECT_EQ(advance(e, r), advance(e, r));
      EXPECT_EQ(stale(e, e + 1u), stale(e, e + 1u));
    }
}

// T-SCAFFOLD (acceptance MUST-FIX) — invoke the primitive THROUGH the cuda::nbr_store::
// re-export alias, so deleting the `using` in zone_nbr_store.cuh fails a test (else the
// scaffold's only executable element is inert-but-untested — the structurally-dead class).
TEST(RebuildEpoch, ScaffoldReexportIsLive) {
  EXPECT_EQ(tdmd::cuda::nbr_store::advance(41u, true), 42u);
  EXPECT_EQ(tdmd::cuda::nbr_store::advance(41u, false), 41u);
  EXPECT_TRUE(tdmd::cuda::nbr_store::stale(3u, 4u));
  EXPECT_FALSE(tdmd::cuda::nbr_store::stale(4u, 4u));
  // the alias IS the core primitive (same result), not a shadow copy
  EXPECT_EQ(tdmd::cuda::nbr_store::advance(10u, true), advance(10u, true));
}
