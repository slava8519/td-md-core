#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tdmd/hal/hal.hpp"  // TDMD_HOST_DEVICE — eval reused on the GPU (PR-E5)
#include "tdmd/potentials/eam_analytic.hpp"

// M6 PR-E2 — EAM via LAMMPS-style 7-coefficient cubic splines (setfl).
// Replicates LAMMPS pair_eam::interpolate() + the evaluation EXACTLY, so our
// forces match LAMMPS to ~FP precision (the print-tolerance cross-check on a
// real Al setfl) AND are TRANSCENDENTAL-FREE — only the knot index (int)(x·rd+1)
// and a 7-coeff Horner remainder — so CPU↔GPU is bit-exact under
// -ffp-contract=off / --fmad=false (answers OQ1; M6_EAM_MANYBODY_DESIGN §4.2).
//
// EamSetfl exposes the SAME (r → value,derivative) eval contract as AnalyticEam,
// so eam_direct_fp64 / EamPotential / eam_run_fixed (eam.hpp) reuse it verbatim
// with Math = EamSetfl<Real>.
namespace tdmd::potentials {

template <typename Real>
struct EamSetfl {
  int Nrho = 0, Nr = 0;
  double drho = 0, rdrho = 0, dr = 0, rdr = 0, rcut = 0;
  // 7 coeffs per knot, rows 1..N (row 0 unused — LAMMPS is 1-based). Flat 7*(N+1).
  std::vector<double> Fspl;     // embedding F(ρ),  grid ρ = i·drho
  std::vector<double> rhoaspl;  // density ρ_a(r),  grid r = i·dr
  std::vector<double> rphispl;  // r·φ(r) (z2r),    grid r = i·dr
  double rhoa_floor = 0.0;      // ρ_a at r_min_guard — the density-quantum bound
  double r_min_guard = 0.5;

  // --- HOST_DEVICE evaluation (verbatim LAMMPS pair_eam) ---
  // value + d/dx of a 7-coeff spline at x (knot spacing folded into rd = 1/Δ).
  TDMD_HOST_DEVICE static void eval_spline(const double* spl, int n, double rd,
                                           double x, double& v, double& dv) {
    double p = x * rd + 1.0;
    int m = int(p);          // truncation toward zero (x>=0) — no rounding mode, no tie
    if (m < 1) m = 1;
    if (m > n - 1) m = n - 1;
    p -= m;
    if (p > 1.0) p = 1.0;
    const double* c = spl + 7 * m;
    v = ((c[3] * p + c[4]) * p + c[5]) * p + c[6];
    dv = (c[0] * p + c[1]) * p + c[2];  // already scaled by 1/Δ in interpolate()
  }
  void eval_F(double rho, double& v, double& dv) const {
    eval_spline(Fspl.data(), Nrho, rdrho, rho, v, dv);
  }
  void eval_rhoa(double r, double& v, double& dv) const {
    eval_spline(rhoaspl.data(), Nr, rdr, r, v, dv);
  }
  void eval_phi(double r, double& v, double& dv) const {
    double z2, z2p;  // z2 = r·φ, z2p = d(r·φ)/dr
    eval_spline(rphispl.data(), Nr, rdr, r, z2, z2p);
    const double rec = 1.0 / r;
    v = z2 * rec;             // φ   = (r·φ)/r
    dv = (z2p - v) * rec;     // φ'  = (z2p − φ)/r   (verbatim LAMMPS psip term)
  }

  // load-time density-quantum guard — same policy AND same physical basis as
  // AnalyticEam (M6 §4.1): ρ_a at r_min_guard, NOT the unphysical r→0 table top
  // (which no accepted pair reaches — that would make the guard over-conservative
  // and inconsistent with the analytic side feeding the same eam.hpp drivers).
  static constexpr double kMaxCoord = 32.0, kSafety = 4.0;
  int density_fracbits() const {
    const double bound = rhoa_floor * kMaxCoord * kSafety;
    if (bound < std::ldexp(1.0, 19)) return 44;
    if (bound < std::ldexp(1.0, 23)) return 40;
    throw std::runtime_error("EamSetfl: density bound exceeds even Q23.40");
  }

  // Top of the F(ρ) tabulation. ρ beyond this is NOT representable — the drivers
  // HALT past it (eam.hpp) instead of silently evaluating the clamped last knot
  // (LAMMPS only warns). Compression/melt configs can exceed it (M6 §4.1).
  double density_grid_max() const { return double(Nrho - 1) * drho; }

  // --- LAMMPS pair_eam::interpolate(): 7-coeff spline from f[1..n] (n>=5). ---
  static std::vector<double> interpolate(int n, double delta,
                                         const std::vector<double>& f) {
    if (n < 5) throw std::runtime_error("EamSetfl: spline needs >= 5 knots");
    std::vector<double> s(7 * (n + 1), 0.0);
    auto C = [&](int m, int k) -> double& { return s[7 * m + k]; };
    for (int m = 1; m <= n; ++m) C(m, 6) = f[m - 1];
    C(1, 5) = C(2, 6) - C(1, 6);
    C(2, 5) = 0.5 * (C(3, 6) - C(1, 6));
    C(n - 1, 5) = 0.5 * (C(n, 6) - C(n - 2, 6));
    C(n, 5) = C(n, 6) - C(n - 1, 6);
    for (int m = 3; m <= n - 2; ++m)
      C(m, 5) = ((C(m - 2, 6) - C(m + 2, 6)) + 8.0 * (C(m + 1, 6) - C(m - 1, 6))) / 12.0;
    for (int m = 1; m <= n - 1; ++m) {
      C(m, 4) = 3.0 * (C(m + 1, 6) - C(m, 6)) - 2.0 * C(m, 5) - C(m + 1, 5);
      C(m, 3) = C(m, 5) + C(m + 1, 5) - 2.0 * (C(m + 1, 6) - C(m, 6));
    }
    C(n, 4) = 0.0;
    C(n, 3) = 0.0;
    for (int m = 1; m <= n; ++m) {
      C(m, 2) = C(m, 5) / delta;
      C(m, 1) = 2.0 * C(m, 4) / delta;
      C(m, 0) = 3.0 * C(m, 3) / delta;
    }
    return s;
  }

  void build_(const std::vector<double>& Ff, const std::vector<double>& rhoaf,
              const std::vector<double>& rphif) {
    rdrho = 1.0 / drho;
    rdr = 1.0 / dr;
    Fspl = interpolate(Nrho, drho, Ff);
    rhoaspl = interpolate(Nr, dr, rhoaf);
    rphispl = interpolate(Nr, dr, rphif);
    assert_rhoa_nonnegative_();  // PR-1 D3 — the donation-contract guard (one funnel
                                 // for BOTH loaders: from_setfl and from_analytic)
    double dv;
    eval_rhoa(r_min_guard, rhoa_floor, dv);  // physical-floor density bound
  }

  // PR-1 D3 (W-contract): the donation contract requires ρ_a(r) >= 0 — the
  // monotonicity of partial ρ (partial <= final) and the rho_cap trigger-set
  // equivalence hold only on non-negative quanta (rint(v·2^fb) >= 0 for v >= 0).
  // EXACT per-interval cubic minimum, not knot sampling: the D3 mandate was the
  // BETWEEN-knot spline overshoot (knot-only checking is the kill-mutation of
  // tooth Т-15б). Interval m in [1, Nr-1], p in [0,1):
  //   v(p) = ((c3·p + c4)·p + c5)·p + c6, ci = rhoaspl[7m+3..6];
  //   min over {v(0), v(1), v(p*)} for real roots p* of 3c3·p² + 2c4·p + c5 in (0,1).
  // The knot m = Nr covers the clamp tail (eval clamps p to the last knot).
  // STRICT `< 0.0`: from_analytic's force-shifted ρ_a is EXACTLY 0 at the rcut
  // knot (measured) — zero must pass. Load-time only; the HOST_DEVICE eval path
  // (bitwise CPU<->GPU contract) is untouched.
  void assert_rhoa_nonnegative_() const {
    auto C = [&](int m, int k) -> double { return rhoaspl[7 * m + k]; };
    double vmin = C(1, 6);
    for (int m = 1; m <= Nr - 1; ++m) {
      const double c3 = C(m, 3), c4 = C(m, 4), c5 = C(m, 5), c6 = C(m, 6);
      auto v = [&](double p) { return ((c3 * p + c4) * p + c5) * p + c6; };
      vmin = std::min(vmin, std::min(v(0.0), v(1.0)));
      // interior extrema: roots of 3c3 p^2 + 2c4 p + c5 = 0 in (0,1)
      const double a = 3.0 * c3, b = 2.0 * c4;
      if (a == 0.0) {
        if (b != 0.0) {
          const double p = -c5 / b;
          if (p > 0.0 && p < 1.0) vmin = std::min(vmin, v(p));
        }
      } else {
        const double disc = b * b - 4.0 * a * c5;
        if (disc >= 0.0) {
          const double sq = std::sqrt(disc);
          for (const double p : {(-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)})
            if (p > 0.0 && p < 1.0) vmin = std::min(vmin, v(p));
        }
      }
    }
    vmin = std::min(vmin, C(Nr, 6));  // clamp-tail knot
    if (vmin < 0.0)
      throw std::runtime_error(
          "EamSetfl: rho_a(r) dips below zero (min " + std::to_string(vmin) +
          ") — the donation contract requires rho_a >= 0: the monotonicity of "
          "partial rho (partial <= final) and the rho_cap trigger-set "
          "equivalence hold only on non-negative quanta (PR-1 D3)");
  }

  // --- single-element setfl (eam/alloy) loader ---
  static EamSetfl from_setfl(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("EamSetfl: cannot open " + path);
    std::string line;
    for (int i = 0; i < 3; ++i) std::getline(in, line);  // 3 comment lines
    std::getline(in, line);                              // Nelem Elem...
    { std::istringstream ss(line); int nel = 0; ss >> nel;
      if (nel != 1) throw std::runtime_error("EamSetfl: single-element only (PR-E2)"); }
    EamSetfl s;
    std::getline(in, line);                              // Nrho drho Nr dr cutoff
    { std::istringstream ss(line);
      if (!(ss >> s.Nrho >> s.drho >> s.Nr >> s.dr >> s.rcut))
        throw std::runtime_error("EamSetfl: bad header line"); }
    // reject malformed spacing/counts before they build a NaN/inf potential
    // (1/drho, /delta in interpolate) — consistent with the nelem!=1 and n<5 guards.
    if (!(s.drho > 0.0 && s.dr > 0.0 && s.Nrho >= 5 && s.Nr >= 5 && s.rcut > 0.0))
      throw std::runtime_error(
          "EamSetfl: bad header — Nrho,Nr (>=5), drho,dr,cutoff must be positive");
    std::getline(in, line);                              // Z mass alat lattice (ignored)
    std::vector<double> Ff(s.Nrho), rhoaf(s.Nr), rphif(s.Nr);
    auto read_n = [&](std::vector<double>& v, const char* what) {
      for (double& x : v)
        if (!(in >> x)) throw std::runtime_error(std::string("EamSetfl: short ") + what);
    };
    read_n(Ff, "F(rho)");
    read_n(rhoaf, "rho_a(r)");
    read_n(rphif, "r*phi(r)");
    s.build_(Ff, rhoaf, rphif);
    return s;
  }

  // --- tabulate an analytic potential onto a setfl grid (own host-once gen) ---
  static EamSetfl from_analytic(const AnalyticEam<Real>& a, int Nrho, int Nr,
                                double rho_max) {
    EamSetfl s;
    s.Nrho = Nrho; s.Nr = Nr; s.rcut = a.rcut;
    s.dr = a.rcut / (Nr - 1);
    s.drho = rho_max / (Nrho - 1);
    s.r_min_guard = a.r_min_guard;
    std::vector<double> Ff(Nrho), rhoaf(Nr), rphif(Nr);
    for (int i = 0; i < Nrho; ++i) { double F, dF; a.eval_F(i * s.drho, F, dF); Ff[i] = F; }
    for (int i = 0; i < Nr; ++i) {
      const double r = i * s.dr;
      double v, dv; a.eval_rhoa(r, v, dv); rhoaf[i] = v;
      double phi, dphi; a.eval_phi(r, phi, dphi); rphif[i] = r * phi;
    }
    s.build_(Ff, rhoaf, rphif);
    return s;
  }
};

}  // namespace tdmd::potentials
