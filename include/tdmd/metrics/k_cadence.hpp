#pragma once
#include <cmath>
#include <cstdio>
#include <vector>

// M4-N — K-cadence meter (perf track, "measure-first"). Measures the REALIZED
// neighbour-list rebuild cadence K_eff against the PREDICTED K_pred = skin/(2·R_buf)
// on an equilibrium→shock sweep (F.4, line "K_pred/K_eff/cadence на свипе").
//
//   K_pred = skin / (2·R_buf),  R_buf = v_pred·dt·C  (the same causality-buffer
//            envelope the conveyor uses; v_pred = v_max + a_max·dt, [ENG] eq.33).
//            The conservative ballistic estimate: the fastest atom moves R_buf
//            per step, two approaching atoms close 2·R_buf, rebuild at skin.
//   K_eff  = realized steps per rebuild. A rebuild fires when the actual max
//            per-atom displacement since the last build reaches skin/2 (standard
//            half-skin Verlet criterion). Positions in the engine are global and
//            UNWRAPPED (PBC lives only in the pair distance), so displacement is
//            x−x_ref directly — no minimum-image needed.
//   conservatism = K_eff / K_pred  (>1 ⇒ the envelope over-rebuilds vs the real
//            diffusive motion). This is the amortization factor the PersistentVerlet
//            bake-off (M4-B) needs: a long-lived list pays its build cost over K_eff
//            steps, not the conservative K_pred. Host-only; pure kinematics, no CUDA.
namespace tdmd::metrics {

struct KCadence {
  double skin = 0.0;
  long steps = 0;
  long rebuilds = 0;        // realized rebuild events (max_disp reached skin/2)
  double sum_R_buf = 0.0;   // Σ per-step R_buf — the skin-budget envelope (eq.33)
  long R_buf_samples = 0;

  // realized steps per rebuild. With zero rebuilds the cadence is a lower bound
  // (≥ steps) — the system never moved skin/2 over the whole run.
  double K_eff() const { return rebuilds > 0 ? double(steps) / double(rebuilds) : double(steps); }
  // Predicted envelope cadence = skin / (2·mean(R_buf)). This is the cadence the
  // skin-budget recurrence skin_consumed += 2·R_buf actually realizes — i.e. the
  // HARMONIC mean of the per-step skin/(2·R_buf), NOT the arithmetic mean of it
  // (R_buf is in the denominator: arithmetic averaging over-weights slow steps,
  // over-states K_pred and can spuriously push conservatism below 1 under a
  // strongly varying v_max — the single-front-shock / small-N regime M4-B hits).
  // Aggregating R_buf and dividing once is exactly the budget the engine spends.
  double K_pred() const {
    if (R_buf_samples == 0) return 0.0;
    const double mean_R = sum_R_buf / double(R_buf_samples);
    return mean_R > 0.0 ? skin / (2.0 * mean_R) : 0.0;
  }
  // >1 ⇒ the conservative envelope over-rebuilds vs the real motion. With C≥1 the
  // predicted per-step budget 2·R_buf=2·v_pred·dt·C ≥ the real worst-atom
  // displacement, so K_pred ≤ K_eff step-wise ⇒ conservatism ≥ 1 unconditionally.
  double conservatism() const {
    const double kp = K_pred();
    return kp > 0.0 ? K_eff() / kp : 0.0;
  }
  void report(const char* tag) const {
    std::printf(
        "[%s] skin=%.2f | steps=%ld rebuilds=%ld | K_eff=%.2f%s K_pred=%.2f | "
        "conservatism(K_eff/K_pred)=%.2f\n",
        tag, skin, steps, rebuilds, K_eff(), rebuilds == 0 ? "(≥)" : "",
        K_pred(), conservatism());
  }
};

// Stateful meter driven step-by-step over a trajectory. begin() snapshots the
// reference positions; step() advances the count, accumulates K_pred, and fires
// a rebuild (resetting the reference) when 2·max_disp ≥ skin.
struct KCadenceMeter {
  double skin = 1.0;
  std::vector<double> rx, ry, rz;  // reference positions at the last rebuild
  KCadence acc;

  void reset_reference(const std::vector<double>& x, const std::vector<double>& y,
                       const std::vector<double>& z) {
    rx = x; ry = y; rz = z;
  }
  void begin(const std::vector<double>& x, const std::vector<double>& y,
             const std::vector<double>& z, double skin_) {
    skin = skin_;
    acc = KCadence{};
    acc.skin = skin_;
    reset_reference(x, y, z);
  }
  // Call once per step AFTER positions advanced. r_buf is the engine's own
  // causality buffer R_buf = v_pred·dt·C for this step (the per-step skin budget
  // is 2·r_buf). Aggregated, not turned into K_pred per step — see KCadence::K_pred.
  // Returns true iff a rebuild fired.
  bool step(const std::vector<double>& x, const std::vector<double>& y,
            const std::vector<double>& z, double r_buf) {
    acc.steps++;
    acc.sum_R_buf += r_buf;
    acc.R_buf_samples++;
    double max_d2 = 0.0;
    const int n = int(x.size());
    for (int i = 0; i < n; ++i) {
      const double dx = x[i] - rx[i], dy = y[i] - ry[i], dz = z[i] - rz[i];
      const double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > max_d2) max_d2 = d2;
    }
    if (2.0 * std::sqrt(max_d2) >= skin) {  // max_disp ≥ skin/2 ⇒ rebuild
      acc.rebuilds++;
      reset_reference(x, y, z);
      return true;
    }
    return false;
  }
};

}  // namespace tdmd::metrics
