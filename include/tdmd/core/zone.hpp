#pragma once
#include <cstdint>

// Computation zone — the unit of the Time-Decomposition ring conveyor.
// States and the Zone record follow docs/TD_MD_Core_ZoneFSM_v1_0.md §2, §8.1
// and Skeleton_Interfaces §3. (M1: pure logic — no GPU, no transport.)
namespace tdmd::core {

// Five dynamic zone types, letters exactly per dissertation Гл.2.1 (стр.560-564).
//   c FREE    : memory cleared, no data; ready to receive
//   d DATA    : atoms received, not yet used in force calc
//   w PARTIAL : partial forces from one computed neighbour (Newton-3); incomplete
//   p COMPUTE : being computed now (internal + pair with next zone)
//   o READY   : computed, force_complete; ready to send onward
enum class ZoneType { c, d, w, p, o };

// Five transition triggers — the five type-change rules (ZoneFSM §3).
enum class ZoneEvent { RECV, SPHERE, START, END, SEND };

struct Zone {
  int      id = 0;
  int      step_h = 0;          // time step this zone belongs to
  int      n_atoms = 0;
  ZoneType type = ZoneType::c;
  double   left_bound_glob = 0.0;  // zone left boundary (for FP32 offsets later)
  bool     force_complete = false; // INV-3: true iff type == o
  uint32_t contrib_mask = 0;       // which neighbours have contributed forces
  double   v_max_local = 0.0;      // local v_max (A8 — no global reduction)
  double   R_buf_local = 0.0;      // local buffer width (eq. 33)
};

inline const char* to_string(ZoneType t) {
  switch (t) {
    case ZoneType::c: return "c";
    case ZoneType::d: return "d";
    case ZoneType::w: return "w";
    case ZoneType::p: return "p";
    case ZoneType::o: return "o";
  }
  return "?";
}

} // namespace tdmd::core
