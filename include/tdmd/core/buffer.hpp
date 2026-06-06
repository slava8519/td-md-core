#pragma once
#include <cmath>
#include <algorithm>
#include <limits>
#include "tdmd/core/soa.hpp"
#include "tdmd/units.hpp"

// Causality buffer and automatic time-step (M2).
// Source of truth: dissertation Гл.2.1 (буфер, ур.33), Гл.3.3/3.5 (авто-шаг,
// коэффициенты C1/K2/C3). See docs/TD_MD_Core_ZoneFSM_v1_0.md §6 (INV-4) and
// TD_MD_Core_Roadmap_v1_0.md (M2).
namespace tdmd::core::buffer {

// Parameters of the automatic step (config: timestep.*, decomposition.cell_size).
struct TimeStepCfg {
  double C1 = 0.1;          // displacement fraction of cell_size (highest priority)
  double K2 = 50.0;         // max per-atom temperature rise per step, K
  double C3 = 0.5;          // step-change threshold (ур.62): 0=fixed, 1=every step
  double C_buf = 1.5;       // buffer-width coefficient (>= 1)
  double cell_size = 2.33;  // Å, spatial cell (Al ≈ 2.33)
  double dt_max = 0.02;     // ps, upper bound
  double dt_min = 1e-6;     // ps, floor
};

// Buffer width R_buf = v_max · dt · C  (eq. 33, C >= 1). Precondition: C >= 1.
inline double compute_R_buf(double v_max, double dt, double C) {
  return v_max * dt * C;
}

// INV-4: no atom may cross the buffer in one step. True == causality preserved.
inline bool causality_ok(double v_max, double dt, double R_buf) {
  return v_max * dt <= R_buf;
}

// Local v_max (parallel reduction; serial here). Local to the node — NO global
// all-reduce (A8): the node carries the whole model step (Гл.3.5).
template <typename Real>
double max_speed(const AtomSoA<Real>& a) {
  double v2max = 0.0;
  for (int i = 0; i < a.n; ++i) {
    const double v2 = double(a.vx[i]) * a.vx[i] +
                      double(a.vy[i]) * a.vy[i] +
                      double(a.vz[i]) * a.vz[i];
    v2max = std::max(v2max, v2);
  }
  return std::sqrt(v2max);
}

// Best-effort per-atom temperature-rise limit (coefficient K2, in K).
// NOTE: the exact dissertation formula (Гл.3.3) is image-only in the text
// extract; this is a conservative approximation to verify against the .docx.
// ΔT_i over dt ≈ (m_i / (3 k_B)) · |Δ(v_i²)|, with Δ(v²) ≈ 2|v·a|dt + a²dt².
// Returns the largest dt keeping every atom's ΔT ≤ K2.
template <typename Real>
double temperature_limited_dt(const AtomSoA<Real>& a, double K2) {
  if (K2 <= 0.0) return std::numeric_limits<double>::infinity();
  double dt = std::numeric_limits<double>::infinity();
  for (int i = 0; i < a.n; ++i) {
    const double ax = units::ftm2v * a.fx[i] / a.mass[i];
    const double ay = units::ftm2v * a.fy[i] / a.mass[i];
    const double az = units::ftm2v * a.fz[i] / a.mass[i];
    const double v = std::sqrt(double(a.vx[i]) * a.vx[i] +
                               double(a.vy[i]) * a.vy[i] +
                               double(a.vz[i]) * a.vz[i]);
    const double acc = std::sqrt(ax * ax + ay * ay + az * az);
    if (acc <= 0.0) continue;
    // ΔT = (mvv2e·m/3kB)(2 v acc dt + acc^2 dt^2) <= K2 -> quadratic in dt.
    // (T_atom = mvv2e·m·v²/(3kB); the mvv2e factor is essential — metal units.)
    const double k = units::mvv2e * a.mass[i] / (3.0 * units::kB);
    const double A = k * acc * acc, B = 2.0 * k * v * acc, C = -K2;
    const double disc = B * B - 4.0 * A * C;
    const double dt_i = (-B + std::sqrt(disc)) / (2.0 * A);
    dt = std::min(dt, dt_i);
  }
  return dt;
}

// Automatic step (Гл.3.3, 3.5). C1 (displacement, highest priority — стр.1125)
// plus an optional external cap (e.g. K2 via temperature_limited_dt). C3
// hysteresis (ур.62) decides whether to actually switch from dt_current.
inline double auto_dt(double v_max, double dt_current, const TimeStepCfg& cfg,
                      double extra_cap = std::numeric_limits<double>::infinity()) {
  if (v_max <= 0.0) return dt_current;  // nothing moving — keep the step
  // C1: fastest atom travels <= C1 · cell_size per step.
  double dt_target = std::min(cfg.C1 * cfg.cell_size / v_max, extra_cap);
  dt_target = std::clamp(dt_target, cfg.dt_min, cfg.dt_max);

  // C3 (ур.62): keep dt unless the relative change is large enough.
  if (cfg.C3 <= 0.0) return dt_current;  // C3=0: fixed step (стр.1208)
  if (cfg.C3 >= 1.0) return dt_target;   // C3=1: update every step
  const double rel = std::fabs(dt_target - dt_current) / dt_current;
  return (rel >= (1.0 - cfg.C3)) ? dt_target : dt_current;
}

} // namespace tdmd::core::buffer
