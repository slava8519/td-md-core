#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>

// M6 PR-E1 — analytic Finnis–Sinclair EAM (Al-flavoured): the CPU FP64 ORACLE
// for the many-body force path. Closed-form ⇒ FD-self-consistency checkable
// (Test_EAM_FD) with NO spline/interpolation ambiguity; PR-E2 tabulates THIS
// onto a HOST_DEVICE cubic spline (setfl) and PR-E5 ports it to the GPU.
//
// Energy (force-shifted truncation on φ and ρ_a ⇒ both VALUE and DERIVATIVE
// vanish at rcut ⇒ E is C¹ ⇒ the analytic force is the EXACT gradient and FD
// matches to FP precision; M6_EAM_MANYBODY_DESIGN §1.1):
//   E      = Σ_{i<j} φ(r_ij) + Σ_i F(ρ_i),   ρ_i = Σ_{j≠i} ρ_a(r_ij)
//   φ(r)   = D[e^{−2α(r−r0)} − 2e^{−α(r−r0)}]      (Morse pair)
//   ρ_a(r) = e^{−β(r−r0ρ)}                          (exp-like electron density)
//   F(ρ)   = −A√ρ,   F'(ρ) = −A/(2√ρ)              (Finnis–Sinclair embedding)
//   f_k    = −Σ_{j≠k}[φ'(r) + (F'(ρ_k)+F'(ρ_j))·ρ_a'(r)]·(x_k−x_j)/r
//
// [ENG][FIXTURE] Parameters are a SELF-CONSISTENT analytic fixture for the FD /
// determinism ladder, NOT a validated Al potential — external validation vs
// LAMMPS pair_style eam/alloy is PR-E7 (LAMMPS-gated). Provenance:
// reference_data/eam_al_analytic/README.md.
namespace tdmd::potentials {

template <typename Real>
struct AnalyticEam {
  // pair φ (Morse, dissertation Al params by default)
  double D = 0.29614, alpha = 1.11892, r0 = 3.29692;
  // density ρ_a (exp-like)
  double rho_amp = 1.0, beta = 1.5, r0_rho = 3.29692;
  // embedding F(ρ) = −A√ρ
  double A_embed = 1.0;
  // cutoff (force-shifted) + density-guard radius
  double rcut = 4.0;
  double r_min_guard = 0.5;  // overlap-halt radius — worst-case ρ_a for the guard

  // --- raw (un-truncated) functions; return value, set derivative in `dv` ---
  double phi_raw(double r, double& dv) const {
    const double e1 = std::exp(-alpha * (r - r0));
    const double e2 = e1 * e1;  // e^{−2α(r−r0)}
    dv = D * (-2.0 * alpha * e2 + 2.0 * alpha * e1);
    return D * (e2 - 2.0 * e1);
  }
  double rhoa_raw(double r, double& dv) const {
    const double e = rho_amp * std::exp(-beta * (r - r0_rho));
    dv = -beta * e;
    return e;
  }

  // --- force-shifted: g_fs(r)=g(r)−g(rc)−g'(rc)(r−rc) ⇒ g_fs(rc)=g_fs'(rc)=0 ---
  void eval_phi(double r, double& v, double& dv) const {
    if (r >= rcut) { v = 0.0; dv = 0.0; return; }
    double draw;
    const double raw = phi_raw(r, draw);
    v = raw - phi_rc_ - dphi_rc_ * (r - rcut);
    dv = draw - dphi_rc_;
  }
  void eval_rhoa(double r, double& v, double& dv) const {
    if (r >= rcut) { v = 0.0; dv = 0.0; return; }
    double draw;
    const double raw = rhoa_raw(r, draw);
    v = raw - rhoa_rc_ - drhoa_rc_ * (r - rcut);
    dv = draw - drhoa_rc_;
  }
  void eval_F(double rho, double& v, double& dv) const {
    if (rho <= 0.0) { v = 0.0; dv = 0.0; return; }  // outside the FS smooth regime
    const double s = std::sqrt(rho);
    v = -A_embed * s;
    dv = -A_embed / (2.0 * s);
  }

  // Precompute the force-shift cutoff constants. MUST be called after setting
  // params and before any eval (else the truncation is not applied → the
  // energy is discontinuous at rcut and FD fails). make_analytic_al() does it.
  void finalize() {
    phi_rc_ = phi_raw(rcut, dphi_rc_);
    rhoa_rc_ = rhoa_raw(rcut, drhoa_rc_);
  }

  // Load-time density-quantum guard (M6 §4.1, OQ2): pick the FixedAccum<>
  // FracBits whose accumulator ceiling holds the worst-case accumulated ρ.
  //   Q19.44 (default, ceiling 2¹⁹) → Q23.40 fallback (ceiling 2²³) → throw.
  // The per-CONTRIBUTION guard in FixedAccum::add nearly coincides with the
  // ACCUMULATOR ceiling, so density integrity rests on this physics bound, not
  // on the add() guard (M6 §4.1, the adversary's overflow finding).
  static constexpr double kMaxCoord = 32.0;  // generous coordination bound
  static constexpr double kSafety = 4.0;
  double rhoa_bound() const {
    double d;
    return rhoa_raw(r_min_guard, d);  // raw ρ_a at the closest physical radius
  }
  int density_fracbits() const {
    const double bound = rhoa_bound() * kMaxCoord * kSafety;
    if (bound < std::ldexp(1.0, 19)) return 44;  // Q19.44
    if (bound < std::ldexp(1.0, 23)) return 40;  // Q23.40 fallback
    throw std::runtime_error(
        "AnalyticEam: density bound exceeds even Q23.40 — ρ_a table too steep "
        "(M6 §4.1 load-time guard)");
  }

  // Embedding F(ρ) is closed-form (valid for any ρ ≥ 0) — no tabulation grid,
  // so no upper bound. The setfl/spline form (eam_spline.hpp) returns its finite
  // grid top, which the drivers HALT past (vs LAMMPS's silent clamp).
  double density_grid_max() const { return std::numeric_limits<double>::infinity(); }

 private:
  double phi_rc_ = 0.0, dphi_rc_ = 0.0, rhoa_rc_ = 0.0, drhoa_rc_ = 0.0;
};

// Finalized default-Al analytic EAM fixture.
template <typename Real>
inline AnalyticEam<Real> make_analytic_al() {
  AnalyticEam<Real> m;
  m.finalize();
  return m;
}

}  // namespace tdmd::potentials
