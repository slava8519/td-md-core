#pragma once
#include <cmath>
#include <cstdio>
#include <string>
#include <algorithm>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/integrator.hpp"
#include "tdmd/core/buffer.hpp"

// Single-zone NVE driver (M0–M2.6 loop), extracted from main() (M3, audit
// решение 7) so tests can drive it directly — the run-to-run bitwise
// determinism test is the safety net for the M3 pair-order change.
// I/O-free: frames/rescue/printing are the caller's concern (callback + result).
namespace tdmd::core {

struct SimOptions {
  long   steps = 0;
  double dt = 0.001;            // ps
  bool   auto_step = false;     // C1/K2/C3 auto-dt (M2)
  buffer::TimeStepCfg ts{};     // coefficients + dt bounds
  double r_min_halt = 0.5;      // Å, overlap HALT (B10)
  long   frame_every = 0;       // 0 = no frame callbacks
};

enum class Halt { None, Overlap, Causality, NonFiniteEnergy };

struct SimResult {
  long   steps_done = 0;        // completed steps (HALT step excluded)
  double e0 = 0.0, e_final = 0.0;
  double drift = 0.0;           // max-min of total energy over the run
  double dt_final = 0.0, v_max = 0.0;
  Halt   halt = Halt::None;
  std::string halt_msg;
};

// Pot: any potential with `double compute(AtomSoA<Real>&, const Box&)` that
// ACCUMULATES forces (caller zeroes) and exposes `double last_min_r2`.
// Frame: void(long step) — invoked for step 0 and every frame_every-th step.
template <typename Real, typename Pot, typename Frame>
SimResult run_simulation(AtomSoA<Real>& atoms, const Box& box, Pot& pot,
                         const SimOptions& o, Frame&& on_frame) {
  SimResult r;
  const double rmin2 = o.r_min_halt * o.r_min_halt;

  auto overlap = [&](long step) {
    if (pot.last_min_r2 >= rmin2) return false;
    r.halt = Halt::Overlap;
    r.halt_msg = "atom overlap at step " + std::to_string(step) +
                 ": min pair distance " + std::to_string(std::sqrt(pot.last_min_r2)) +
                 " A < " + std::to_string(o.r_min_halt) + " A";
    return true;
  };

  zero_forces(atoms);
  double pe = pot.compute(atoms, box);
  if (overlap(0)) return r;
  double ke = kinetic_energy(atoms);
  r.e0 = pe + ke;
  double emin = r.e0, emax = r.e0;
  if (o.frame_every > 0) on_frame(0);

  double dt = o.dt;
  double v_max_prev = buffer::max_speed(atoms);  // local v_max (A8)
  double a_max_prev = buffer::max_accel(atoms);  // forces are current (computed above)
  for (long step = 1; step <= o.steps; ++step) {
    // Buffer for this step is sized from the speed an atom may REACH during
    // the step: v_pred = v_max + a_max·dt (A8; [ENG] refinement of eq.33's
    // v̄_max — see buffer::max_accel). Everything is known BEFORE the step:
    // v_max and forces of the last completed pass — no global reduce.
    const double v_pred = v_max_prev + a_max_prev * dt;
    const double R_buf = buffer::compute_R_buf(v_pred, dt, o.ts.C_buf);

    VelocityVerlet<Real>::first_half(atoms, dt);
    zero_forces(atoms);
    pe = pot.compute(atoms, box);
    if (overlap(step)) return r;
    VelocityVerlet<Real>::second_half(atoms, dt);
    ke = kinetic_energy(atoms);

    const double e = pe + ke;
    if (std::isnan(e) || std::isinf(e)) {
      r.halt = Halt::NonFiniteEnergy;
      r.halt_msg = "non-finite energy at step " + std::to_string(step);
      return r;
    }

    const double v_max_now = buffer::max_speed(atoms);
    // INV-4: no atom may cross the buffer — checked in BOTH fixed and auto
    // modes (B10/M12; ZoneFSM §6 says "always", eq.33 is unconditional).
    // The v_pred-sized buffer covers the rest-start ballistic ramp, so the
    // check holds from step 1; what still trips it is force appearing FASTER
    // than the entry-state forecast (e.g. flying through the cutoff into the
    // repulsive wall) — a true causality hazard.
    if (!buffer::causality_ok(v_max_now, dt, R_buf)) {
      r.halt = Halt::Causality;
      char buf[160];
      std::snprintf(buf, sizeof(buf),
                    "causality (INV-4) at step %ld: v_max*dt=%.4g > R_buf=%.4g (dt=%.17g)",
                    step, v_max_now * dt, R_buf, dt);
      r.halt_msg = buf;
      return r;
    }

    emin = std::min(emin, e);
    emax = std::max(emax, e);
    if (o.frame_every > 0 && step % o.frame_every == 0) on_frame(step);

    v_max_prev = v_max_now;
    a_max_prev = buffer::max_accel(atoms);  // forces of this completed pass
    if (o.auto_step)
      dt = buffer::auto_dt(v_max_now, dt, o.ts,
                           buffer::temperature_limited_dt(atoms, o.ts.K2));
    r.steps_done = step;
  }

  r.e_final = pe + ke;
  r.drift = emax - emin;
  r.dt_final = dt;
  r.v_max = v_max_prev;
  return r;
}

} // namespace tdmd::core
