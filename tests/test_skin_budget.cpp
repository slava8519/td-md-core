// PR-0b — the extracted Λ-skin / K-aware-fallback decision (core/skin_budget.hpp).
// Previously a private method of GpuTimeConveyor, testable only via the GPU verlet
// suite; now a pure host function with its own gate (and PR-4's many-body port
// consumes the SAME source). Each test carries its kill-mutation. The bitwise
// z-INDEPENDENCE of the decision is proved by the GPU 1-vs-z verlet tests
// (Test_CUDA_Conveyor); here we pin the arithmetic itself.
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "tdmd/core/skin_budget.hpp"

using tdmd::core::skin_budget::decide_pass;
using tdmd::core::skin_budget::SkinParams;

namespace {
// K_pred = skin/(2*R_buf). To make K_pred = K exactly, pick R_buf = skin/(2*K).
double rbuf_for_kpred(double skin, double K) { return skin / (2.0 * K); }

struct Out {
  double skin_out = -1.0;
  bool rebuild = false;
  std::uint8_t va_next = 42;
};
Out run(const SkinParams& p, double skin_in, double R_buf, std::uint8_t va_prev,
        double d_lagged = 0.0, double lag = 0.0) {
  Out o;
  decide_pass(p, skin_in, R_buf, va_prev, d_lagged, lag, o.skin_out, o.rebuild, o.va_next);
  return o;
}
constexpr SkinParams kP{/*skin=*/1.0, /*K_on=*/3.0, /*K_off=*/1.5, /*hybrid=*/false};
}  // namespace

// charge = 2*R_buf and the turn-on threshold is INCLUSIVE (K_pred >= K_on).
// KILL: change 2.0*R_buf, or make the turn-on strict (>).
TEST(SkinBudget, ChargeAndTurnOnThreshold) {
  // K_pred exactly at K_on (3.0) from OFF ⇒ turns ON (inclusive)
  Out on = run(kP, 0.0, rbuf_for_kpred(kP.skin, kP.K_on), /*va_prev=*/0);
  EXPECT_EQ(on.va_next, 1) << "K_pred == K_on must turn ON (inclusive)";
  // just below K_on ⇒ stays OFF
  Out off = run(kP, 0.0, rbuf_for_kpred(kP.skin, kP.K_on - 1e-6), 0);
  EXPECT_EQ(off.va_next, 0) << "K_pred < K_on must stay OFF from cold";
}

// Two-threshold hysteresis: the band [K_off, K_on) is STICKY both ways — OFF stays
// OFF, ON stays ON, no chatter. KILL: collapse to a single threshold (use K_on for
// both, or K_off for both).
TEST(SkinBudget, HysteresisBandIsSticky) {
  const double mid = 0.5 * (kP.K_on + kP.K_off);  // 2.25, inside the band
  const double rb = rbuf_for_kpred(kP.skin, mid);
  EXPECT_EQ(run(kP, 0.0, rb, /*va_prev=*/0).va_next, 0) << "in-band from OFF stays OFF";
  EXPECT_EQ(run(kP, 0.0, rb, /*va_prev=*/1).va_next, 1) << "in-band from ON stays ON";
  // turn-off is STRICT (< K_off): exactly at K_off from ON stays ON
  EXPECT_EQ(run(kP, 0.0, rbuf_for_kpred(kP.skin, kP.K_off), /*va_prev=*/1).va_next, 1)
      << "K_pred == K_off must NOT turn off (strict <)";
  EXPECT_EQ(run(kP, 0.0, rbuf_for_kpred(kP.skin, kP.K_off - 1e-6), /*va_prev=*/1).va_next, 0)
      << "K_pred < K_off turns OFF";
}

// A 0->1 turn-on forces a rebuild (no list built yet) and zeroes the budget.
// KILL: drop the `va_prev == 0` branch (turn-on would try to accumulate a stale skin).
TEST(SkinBudget, TurnOnForcesRebuild) {
  Out o = run(kP, /*skin_in=*/0.7, rbuf_for_kpred(kP.skin, kP.K_on + 1.0), /*va_prev=*/0);
  EXPECT_EQ(o.va_next, 1);
  EXPECT_TRUE(o.rebuild) << "turn-on must force a rebuild";
  EXPECT_EQ(o.skin_out, 0.0) << "turn-on must zero the budget (no carried skin)";
}

// Fallback mode (verlet inactive): the budget is idle, never rebuilds, regardless
// of skin_in. KILL: leak skin_in into skin_out, or set rebuild=true in fallback.
TEST(SkinBudget, FallbackIsIdle) {
  // K_pred below K_off from ON ⇒ falls back to OFF ⇒ idle
  Out o = run(kP, /*skin_in=*/0.9, rbuf_for_kpred(kP.skin, kP.K_off - 0.1), /*va_prev=*/1);
  EXPECT_EQ(o.va_next, 0);
  EXPECT_FALSE(o.rebuild);
  EXPECT_EQ(o.skin_out, 0.0);
  // and cold OFF stays idle
  Out c = run(kP, 0.5, rbuf_for_kpred(kP.skin, 0.1), /*va_prev=*/0);
  EXPECT_EQ(c.va_next, 0);
  EXPECT_FALSE(c.rebuild);
  EXPECT_EQ(c.skin_out, 0.0);
}

// Steady ON: the conservative accumulator carries skin_out = skin_in + charge and
// rebuilds when it reaches the skin, resetting to 0. KILL: drop the accumulation
// (skin_out = charge only), or the reset-on-rebuild.
TEST(SkinBudget, ConservativeAccumulatorAndReset) {
  // stay ON (K well above K_on) with a small charge so no rebuild yet
  const double rb = rbuf_for_kpred(kP.skin, 100.0);  // charge = skin/100 = 0.01
  Out a = run(kP, /*skin_in=*/0.30, rb, /*va_prev=*/1);
  EXPECT_EQ(a.va_next, 1);
  EXPECT_FALSE(a.rebuild) << "0.30 + 0.01 < 1.0 — no rebuild yet";
  EXPECT_DOUBLE_EQ(a.skin_out, 0.31) << "budget must accumulate skin_in + charge";
  // now cross the skin ⇒ rebuild + reset
  Out b = run(kP, /*skin_in=*/0.995, rb, /*va_prev=*/1);
  EXPECT_TRUE(b.rebuild) << "0.995 + 0.01 >= 1.0 — must rebuild";
  EXPECT_EQ(b.skin_out, 0.0) << "rebuild must reset the budget";
  // rebuild boundary is INCLUSIVE (>=): exactly-at-skin must rebuild. Exact binary
  // sum (1.5 + 0.5 == 2.0, charge=0.5 from R_buf=0.25, K_pred=4.0 stays ON).
  // KILL: change `skin_used >= skin` to strict `>`.
  const SkinParams p2{/*skin=*/2.0, 3.0, 1.5, /*hybrid=*/false};
  Out e = run(p2, /*skin_in=*/1.5, /*R_buf=*/0.25, /*va_prev=*/1);
  EXPECT_EQ(e.va_next, 1) << "K_pred=4.0 stays ON";
  EXPECT_TRUE(e.rebuild) << "skin_in+charge == skin (2.0) must rebuild — inclusive >=";
}

// Hybrid: skin_used = min(conservative, 2*d_lagged + 2*L*R_buf) — the TIGHTER of two
// valid upper bounds ⇒ rebuilds no SOONER than conservative. A small measured
// displacement suppresses a rebuild the conservative bound would have triggered.
// KILL: use max() instead of min(), or drop the hybrid branch entirely.
TEST(SkinBudget, HybridTakesTighterBound) {
  SkinParams hp = kP;
  hp.hybrid = true;
  const double rb = rbuf_for_kpred(kP.skin, 100.0);  // charge 0.01
  // conservative budget already over skin (skin_in high), but the measured
  // displacement is tiny ⇒ hyb = 2*0.01 + 2*0*R_buf = 0.02 < skin ⇒ NO rebuild
  Out h = run(hp, /*skin_in=*/0.999, rb, /*va_prev=*/1, /*d_lagged=*/0.01, /*lag=*/0.0);
  EXPECT_EQ(h.va_next, 1);
  EXPECT_FALSE(h.rebuild) << "the tighter hybrid bound (0.02) must suppress the rebuild";
  // control: the SAME inputs without hybrid DO rebuild (1.009 >= 1.0) — proves the
  // hybrid branch is what changed the outcome (anti-vacuity)
  Out c = run(kP, /*skin_in=*/0.999, rb, /*va_prev=*/1, /*d_lagged=*/0.01, /*lag=*/0.0);
  EXPECT_TRUE(c.rebuild) << "without hybrid the conservative bound rebuilds — control";
  // and a LARGE measured displacement lets the rebuild through (hyb dominates skin)
  Out big = run(hp, /*skin_in=*/0.999, rb, /*va_prev=*/1, /*d_lagged=*/1.0, /*lag=*/0.0);
  EXPECT_TRUE(big.rebuild) << "large d_lagged ⇒ hyb >= skin ⇒ rebuild fires";
}

// The charge>1e-300 guard maps charge≈0 to K_pred=+inf. For skin>0 this coincides
// with plain IEEE (skin/0 = +inf), so the guard is only LOAD-BEARING in the degenerate
// charge==0 ∧ skin==0 case, where plain division is 0/0 = NaN and NaN>=K_on is false
// (verlet would wrongly stay OFF). Config forbids skin<=0, but the pure function must
// stay total/robust for a direct-constructed SkinParams. This tests exactly that
// contract. KILL: drop the guard ⇒ 0/0 = NaN ⇒ va_next stays 0 ⇒ RED (acceptance MUST-FIX:
// the previous skin=1.0 form was VACUOUS — 1.0/0.0=+inf killed nothing).
TEST(SkinBudget, ZeroChargeGuardIsInfiniteNotNaN) {
  const SkinParams degenerate{/*skin=*/0.0, /*K_on=*/3.0, /*K_off=*/1.5, /*hybrid=*/false};
  Out o = run(degenerate, 0.0, /*R_buf=*/0.0, /*va_prev=*/0);
  EXPECT_EQ(o.va_next, 1) << "the guard must map charge==0 to +inf (turn ON), not NaN";
  // and the ordinary skin>0, R_buf=0 case turns ON too (IEEE skin/0 = +inf; guard agrees)
  EXPECT_EQ(run(kP, 0.0, /*R_buf=*/0.0, /*va_prev=*/0).va_next, 1);
}

// Purity/determinism proxy for z-independence: identical inputs ⇒ identical outputs
// (the GPU 1-vs-z verlet test proves the caller feeds z-independent inputs).
TEST(SkinBudget, DeterministicPure) {
  SkinParams hp = kP;
  hp.hybrid = true;
  for (double rb : {0.0, 0.02, 0.2, 0.5}) {
    Out a = run(hp, 0.4, rb, 1, 0.3, 2.0);
    Out b = run(hp, 0.4, rb, 1, 0.3, 2.0);
    EXPECT_EQ(a.skin_out, b.skin_out);
    EXPECT_EQ(a.rebuild, b.rebuild);
    EXPECT_EQ(a.va_next, b.va_next);
  }
}
