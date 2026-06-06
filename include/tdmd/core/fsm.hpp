#pragma once
#include <array>
#include <stdexcept>
#include "tdmd/core/zone.hpp"

// Zone finite-state machine — the heart of the Time-Decomposition method.
// Strictly the transition table of docs/TD_MD_Core_ZoneFSM_v1_0.md §4 and the
// invariants of §6. Pure logic (M1): guards/actions that need physics or
// transport are performed by the caller; this only enforces the legal type
// changes and the partiality invariant (INV-3).
namespace tdmd::core {

struct ZoneFSM {
  // Each state allows exactly one event; each event maps from exactly one state
  // (the canonical cycle c→d→w→p→o→c). Any other (state,event) is forbidden.
  static constexpr bool allowed(ZoneType from, ZoneEvent e) {
    switch (from) {
      case ZoneType::c: return e == ZoneEvent::RECV;
      case ZoneType::d: return e == ZoneEvent::SPHERE;
      case ZoneType::w: return e == ZoneEvent::START;
      case ZoneType::p: return e == ZoneEvent::END;
      case ZoneType::o: return e == ZoneEvent::SEND;
    }
    return false;
  }

  // Target state for a legal transition. Throws on a forbidden transition.
  static constexpr ZoneType next(ZoneType from, ZoneEvent e) {
    if (!allowed(from, e))
      throw std::logic_error("ZoneFSM::next — forbidden transition");
    switch (e) {
      case ZoneEvent::RECV:   return ZoneType::d;
      case ZoneEvent::SPHERE: return ZoneType::w;
      case ZoneEvent::START:  return ZoneType::p;
      case ZoneEvent::END:    return ZoneType::o;
      case ZoneEvent::SEND:   return ZoneType::c;
    }
    throw std::logic_error("ZoneFSM::next — unknown event");
  }

  // Applies a transition to a zone and maintains INV-3 (force_complete == true
  // iff type == o) and the contribution mask. Throws on a forbidden transition.
  static void apply(Zone& z, ZoneEvent e) {
    z.type = next(z.type, e);  // throws if not allowed
    switch (e) {
      case ZoneEvent::RECV:   z.force_complete = false; z.contrib_mask = 0; break;
      case ZoneEvent::SPHERE: z.force_complete = false; break;  // partial forces
      case ZoneEvent::START:  z.force_complete = false; break;
      case ZoneEvent::END:    z.force_complete = true;  break;  // INV-3
      case ZoneEvent::SEND:   z.force_complete = false; z.contrib_mask = 0; break;
    }
  }
};

// --- start-up / scheduling helpers (ZoneFSM §7), still pure logic ---

// Initial zone type at t0 (§7.3): node P1 starts with data (d), the rest free (c).
// node_id is 1-based (P1..Pz).
inline ZoneType initial_zone_type(int node_id) {
  return (node_id == 1) ? ZoneType::d : ZoneType::c;
}

// First zone S1 has no predecessor, so it receives an artificial SPHERE seed
// (d→w) at the start of a step (§7.1).
inline void seed_first_zone(Zone& s1) { ZoneFSM::apply(s1, ZoneEvent::SPHERE); }

// Anti-deadlock I/O order (§7.4): odd node sends then receives; even node
// receives then sends. node_id is 1-based.
enum class IoOp { SEND, RECV };
inline std::array<IoOp, 2> node_io_order(int node_id) {
  if (node_id % 2 != 0) return {IoOp::SEND, IoOp::RECV};  // odd
  return {IoOp::RECV, IoOp::SEND};                        // even
}

} // namespace tdmd::core
