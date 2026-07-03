#pragma once
#include <cstdint>

// PR-0c (W-contract ladder) — the lazy-materialization EPOCH primitive, the next
// single-source-of-truth scalar the many-body neighbor port (PR-4, ZoneNeighborStore)
// consumes. Pure host C++ (no CUDA) ⇒ independently unit-testable (tests/test_rebuild_
// epoch.cpp), exactly like core/skin_budget.hpp (PR-0b). INERT in PR-0c: nothing writes
// or reads the header field yet; rebuild_now stays the ACTIVE rebuild decision. PR-4
// activates the write (head, sent==0) + the read (arrival 0) + the per-rank local_epoch[].
//
// Design of record: docs/_meta/AUDIT_W_PHASE_NEIGHBORS_2026-07-02.md §3.5 (MPI lazy epoch)
// + §3.2 (L-SUP bitwiseness) + docs/_meta/PR0C_* design doc.
//
// THE MECHANISM (what PR-4 will do, here for context): the head broadcasts rebuild_epoch
// into EVERY header (NL-INV-4). A rank keeps local_epoch[zone_id] (per-rank state, NOT
// shipped) and (re)materializes zone z's list from ITS OWN current positions iff
// stale(local_epoch[z], hdr.rebuild_epoch). Between ticks a rank REUSES its list.
//
// L-SUP BITWISENESS (AUDIT §3.2): reuse is bitwise-identical to a per-pass rebuild because
// the charged list (skin >= 2*R_buf) is a SUPERSET of the true neighbours AND the force
// kernel's exact r^2<rc^2 retest zeroes any extra candidate (W-1 + L-CLOSE). The load-
// bearing bridge is EPOCH-vs-REBUILD CONSISTENCY: advance() strictly increases IFF a
// rebuild occurred ⇒ a lagging rank rebuilds on EXACTLY the passes the head did — never
// lazily skips a mandatory rebuild (conservativeness), never rebuilds on a reuse pass
// (laziness). Tooth T3/T5.
//
// z-INDEPENDENCE: rebuild_epoch is a z-independent broadcast scalar (the head's per-pass
// decision in every header), the same induction as dt_next/skin_consumed ⇒ the epoch path
// inherits 1-vs-z bitwise. WRAP: uint32 never wraps in a runnable trajectory (>4e9 rebuilds
// needed; rebuild-every-step over 1e6 steps = 1e6 << 2^32) ⇒ no wrap logic needed.
namespace tdmd::core::rebuild_epoch {

// Advance the head's rebuild epoch. prev = epoch shipped on the last header; rebuild =
// decide_pass's rebuild flag for THIS pass. Returns the epoch to broadcast this pass:
//   epoch(h) = epoch(h-1) + (rebuild ? 1 : 0)
// Monotone non-decreasing; STRICTLY increases IFF a rebuild occurred (the L-SUP bridge).
constexpr std::uint32_t advance(std::uint32_t prev, bool rebuild) {
  return prev + (rebuild ? 1u : 0u);
}

// Lazy-materialization predicate. A rank whose local list is at generation `local` must
// (re)materialize the zone's list iff the head has advanced STRICTLY past it: equal epochs
// REUSE (list current — laziness), lagging epochs REBUILD (conservativeness). Total/robust
// for local > hdr (an unreachable ahead-of-head local) ⇒ reuse.
constexpr bool stale(std::uint32_t local, std::uint32_t hdr) {
  return local < hdr;
}

}  // namespace tdmd::core::rebuild_epoch
