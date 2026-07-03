#pragma once
#include <cstdint>
#include <limits>

// PR-0b (W-contract ladder) — the Λ-skin recurrence + K-aware fallback hysteresis,
// EXTRACTED byte-identical from GpuTimeConveyor::decide_pass (cuda/conveyor_gpu.cuh)
// so the many-body ring port (PR-4, ZoneNeighborStore) consumes ONE source instead
// of re-deriving it. Pure host C++ (no CUDA) ⇒ independently unit-testable
// (tests/test_skin_budget.cpp). The GPU conveyor now delegates to this function.
//
// Design of record: docs/_meta/AUDIT_W_PHASE_NEIGHBORS_2026-07-02.md §3.5 (skin port).
//
// z-INDEPENDENCE (the load-bearing property, unchanged by the extraction): every
// input is a z-independent scalar over the lagged Λ-forecast (skin_in carried in the
// header, R_buf/d_lagged/lag from the Λ-chain), so the whole decision — skin budget,
// rebuild flag, AND verlet_active — is bitwise identical for any node count z. Same
// induction as dt_next (ZoneFSM §6). The writer is the pass head (the sent==0 block
// in send_slot), broadcast into EVERY header (NL-INV-4); the consumer is arrival 0.
namespace tdmd::core::skin_budget {

// The four config scalars the decision needs (ConveyorOptions.verlet_*). K_on/K_off
// (3.0/1.5, the M4-B calibration) are NOT revisited without a bake-off.
struct SkinParams {
  double skin;    // verlet_skin — list radius = rcut + skin (Å)
  double K_on;    // enable reuse at K_pred >= K_on (I1, PR-2)
  double K_off;   // fall back to the cell-raster path below K_off
  bool   hybrid;  // measured-displacement criterion (2*d_lagged + 2*L*R_buf)
};

// One pass's skin/rebuild/verlet-mode decision.
//   * charge 2*R_buf — the per-step pair-approach bound INV-4 enforces (NL-INV-2a).
//   * K_pred = skin/(2*R_buf): the predicted reuse factor. Two-threshold hysteresis
//     (K_on > K_off) flips verlet_active without chatter; below break-even we fall
//     back to the cell-raster path (worst case == the current engine, not "+tax").
//     A 0->1 turn-on forces a rebuild (no list built yet).
//   * hybrid: 2*d_lagged + 2*L*R_buf is ALSO a valid upper bound on the current
//     pair-approach (d_lagged = max displacement as of t-L, + L steps of R_buf);
//     min() of the two upper bounds is the tightest SAFE bound ⇒ rebuilds no sooner
//     than conservative (larger K). Post-rebuild stale d_lagged is harmless: skin_out
//     is then tiny, so min() picks it — no rebuild storm (no epoch tracking needed).
inline void decide_pass(const SkinParams& p, double skin_in, double R_buf,
                        std::uint8_t va_prev, double d_lagged, double lag,
                        double& skin_out, bool& rebuild, std::uint8_t& va_next) {
  const double charge = 2.0 * R_buf;
  const double K_pred = charge > 1e-300
                            ? p.skin / charge
                            : std::numeric_limits<double>::infinity();
  va_next = va_prev;
  if (va_prev == 0 && K_pred >= p.K_on) va_next = 1;
  else if (va_prev == 1 && K_pred < p.K_off) va_next = 0;
  if (!va_next) {                 // fallback (cell-raster): budget idle
    rebuild = false; skin_out = 0.0;
  } else if (va_prev == 0) {      // turned ON: no list yet -> force rebuild
    rebuild = true; skin_out = 0.0;
  } else {
    skin_out = skin_in + charge;       // conservative accumulator (carried)
    double skin_used = skin_out;
    if (p.hybrid) {
      const double hyb = 2.0 * d_lagged + 2.0 * lag * R_buf;
      skin_used = hyb < skin_used ? hyb : skin_used;
    }
    rebuild = (skin_used >= p.skin);
    if (rebuild) skin_out = 0.0;
  }
}

}  // namespace tdmd::core::skin_budget
