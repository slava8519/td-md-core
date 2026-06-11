#pragma once
#include <cmath>

#include "tdmd/hal/hal.hpp"

// Pair Morse math — the SINGLE source of truth (M3 split, audit M15):
//   U(r)    = D (e^{-2α(r-r0)} - 2 e^{-α(r-r0)})          (дисс. Гл. 3.5)
//   F(r)    = -dU/dr = 2αD (e^{-2α(r-r0)} - e^{-α(r-r0)})
// The host O(N²) reference driver, the M3 clustered driver and the M4 CUDA
// kernel must all instantiate THESE functions — never re-derive the formulas.
// template<Real>: FP64 for the reference/deterministic path, FP32 for
// production_mixed (forces computed in Real, accumulated in int64/FP64 — B1).
namespace tdmd::potentials {

template <typename Real>
struct MorseParams {
  Real D, alpha, r0;
};

// Energy and force/r for one pair at distance r (r > 0, r < r_cut; the cutoff
// test and the energy shift stay in the driver — they are cutoff-scheme policy,
// not pair math). f_over_r multiplies the (dx,dy,dz) vector directly.
template <typename Real>
TDMD_HOST_DEVICE inline void pair_morse(Real r, const MorseParams<Real>& p,
                                        Real& u, Real& f_over_r) {
  using std::exp;
  const Real ea  = exp(-p.alpha * (r - p.r0));
  const Real e2a = ea * ea;
  u = p.D * (e2a - Real(2) * ea);
  f_over_r = Real(2) * p.alpha * p.D * (e2a - ea) / r;
}

}  // namespace tdmd::potentials
