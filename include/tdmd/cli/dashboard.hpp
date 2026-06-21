#pragma once
#include <cstddef>
#include <string>
#include <vector>

#include "tdmd/units.hpp"

// M7 (UX) — the live ANSI dashboard renderer. PURE FORMATTER: a Snapshot is a
// finished bag of scalars (no atoms/forces/threads/terminal), and the Dashboard
// methods turn it into text. This separation is what makes the renderer unit-
// testable in cloud CI with no real run, no GPU, no TTY (Test_Dashboard feeds a
// synthetic deterministic Snapshot stream and asserts exact substrings).
//
// The dashboard is OBSERVATIONAL — it reads the conveyor's per-pass PassStats
// snapshots (fed via ConveyorOptions::on_pass, the determinism-safe hook). It
// never feeds anything back into the ring. INV-9 (bitwise 1-vs-z, run-to-run)
// is preserved by construction; gate A7 proves it.
namespace tdmd::cli {

// A finished, render-ready bag of scalars for ONE pass. All physics is already
// computed (PassStats + the run-invariant e0/n_dof/R_buf context); the renderer
// only formats. `halted`/`halt_msg` drive the red HALT line. `spark` is the
// recent E(t) history (oldest..newest) for the sparkline.
struct Snapshot {
  long pass = 0;       // completed pass index (1-based; 0 = not started)
  long total = 0;      // run.steps (0 => unknown -> '?' in progress)
  double pe = 0.0;     // potential energy this pass, eV
  double ke = 0.0;     // kinetic energy this pass, eV
  double e0 = 0.0;     // total energy at t0 (the NVE-drift reference)
  double dt = 0.0;     // ps, this pass's timestep
  double v_max = 0.0;  // Å/ps, fastest atom this pass
  double r_buf = 0.0;  // Å, causality buffer width this pass (0 => unknown)
  int n_dof = 0;       // thermal dof (3N-3) -> T = 2ke/(n_dof kB)
  int zones = 1;       // decomposition zones
  int nodes = 1;       // ring nodes (software processors)
  bool halted = false;
  std::string halt_msg;
  std::string rescue_path;    // written on HALT (if rescue enabled)
  std::vector<double> spark;  // E(t) history, oldest..newest

  // --- derived readouts (pure, no rounding surprises) ---
  double e_total() const {
    return pe + ke;
  }
  // NVE drift (E - e0)/|e0|, the headline conservation invariant.
  double rel_drift() const {
    return (e0 != 0.0) ? (e_total() - e0) / std::abs(e0) : 0.0;
  }
  // Instantaneous temperature, K: T = 2 ke / (n_dof kB).
  double temperature() const {
    return (n_dof > 0) ? 2.0 * ke / (double(n_dof) * units::kB) : 0.0;
  }
  // The TD-distinctive readout: how much of the causality buffer the fastest
  // atom consumes this step, v_max*dt / R_buf. <1 healthy; ->1 = imminent
  // Causality HALT. Returns -1 when R_buf is unknown (e.g. a never-run pass).
  double buffer_headroom() const {
    return (r_buf > 0.0) ? v_max * dt / r_buf : -1.0;
  }
};

class Dashboard {
public:
  // tty=true uses ANSI (cursor-home + SGR color); tty=false emits a flat,
  // color-free block (what render_plain returns) — for files/pipes/CI/Slurm.
  explicit Dashboard(bool tty = false) : tty_(tty) {}

  // The pure, fully testable renderer: a deterministic multi-line block, no
  // ANSI control bytes, no color. Substrings are STABLE (the test asserts them).
  std::string render_plain(const Snapshot& s) const;

  // The live ANSI frame: render_plain content, but with cursor-home/clear-eol
  // and SGR color. On a non-TTY this delegates to render_plain (no escapes).
  std::string render_ansi(const Snapshot& s) const;

  // One-line append-only log record (the non-TTY / every-N-passes fallback).
  // Single line, no escapes, parse-friendly key=value form.
  std::string render_logline(const Snapshot& s) const;

  // The E(t) sparkline over s.spark (Unicode blocks ▁▂▃▄▅▆▇█), min..max scaled.
  // Public so the test can assert ordering independently of the full frame.
  static std::string sparkline(const std::vector<double>& v);

private:
  bool tty_ = false;
};

}  // namespace tdmd::cli
