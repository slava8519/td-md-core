#pragma once

// Cutoff (truncation) schemes — DRIVER policy, not pair math (pair_morse.hpp
// note: "the cutoff test and the energy shift stay in the driver").
//
// M3 decision (audit C1; recorded in TD_MD_Core_Rationale):
//   cut         U(r) truncated at r_cut, nothing else. NIST SRSW "+LRC"
//               schemes tabulate exactly this pair sum.
//   shift       U(r) - U(rc): energy continuous, force still jumps at rc.
//               Project default since M0 (golden data uses it).
//   force_shift U(r) - U(rc) - U'(rc)(r - rc): energy AND force continuous at
//               rc (NIST "linear-force shift"). Chosen over switching for
//               M3.5+ NVE-drift runs: one branchless fixup per pair, no extra
//               functor evaluations, exactly matches an external NIST scheme.
//   switching   REJECTED: needs a switch-region polynomial + its derivative
//               per pair and a second config knob (r_switch); no external
//               frozen-configuration reference to pin it down.
//
// The scheme is applied in FP64 AFTER the (possibly Real=fp32) pair math —
// identical pair sets and shift constants on every path (mixed-precision doc
// §3.3: cutoff decisions in FP64).
namespace tdmd::potentials {

enum class Truncation { Cut, Shift, ForceShift };

// Constants of the scheme, precomputed once per compute() pass.
struct CutoffScheme {
  Truncation mode = Truncation::Shift;
  double rcut = 0.0;
  double u_c = 0.0;  // U(rc)
  double f_c = 0.0;  // radial force F(rc) = -U'(rc) = f_over_r(rc)·rc

  // eval: void(double r, double& u, double& f_over_r) — FP64 pair functor.
  template <typename EvalFn>
  static CutoffScheme make(Truncation mode, double rcut, EvalFn&& eval) {
    double u = 0.0, f_over_r = 0.0;
    if (mode != Truncation::Cut) eval(rcut, u, f_over_r);
    return {mode, rcut, u, f_over_r * rcut};
  }

  // Adjusts one pair's raw (u, f_over_r) in place; r < rcut guaranteed.
  void apply(double r, double& u, double& f_over_r) const {
    switch (mode) {
      case Truncation::Cut:
        break;
      case Truncation::Shift:
        u -= u_c;
        break;
      case Truncation::ForceShift:
        // U_fs(rc) = 0 and U_fs'(rc) = 0  =>  F_fs(r) = F(r) - F(rc)
        u = u - u_c + (r - rcut) * f_c;
        f_over_r -= f_c / r;
        break;
    }
  }
};

}  // namespace tdmd::potentials
