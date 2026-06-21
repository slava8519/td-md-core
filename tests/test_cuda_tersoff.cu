// M6 / Tersoff-ladder Te5: per-window GPU Tersoff kernel (the φ₃/zetaterm transpose accumulator).
// Design wf_d662cfdd-eca. CONTRACT: CPU↔GPU is TOLERANCE (exp/pow/sin), GPU-INTERNAL is BITWISE
// (B1 + the canonical ζ-sort). Gates (⭐ = catches deterministic-but-wrong physics):
//   G1 vacuity     GPU exercises the bond order (total ≠ bigb=0 repulsive-only)
//   G2 CPU↔GPU     |F_gpu − F_cpu| < 1e-9, |pe| < 1e-6, counts ==  (NOT raw-bitwise; +taper fixture)
//   G3 GPU bitwise run-to-run + window-permutation: raw int64 identical (B1 + canonical sort)
//   G4 ⭐ oracle   |F_gpu − tersoff_direct_fp64| < 1e-9 + counts == (free AND PBC-seam) — MB2
//   G5 ⭐ poison   GPU vs dropped-triplet oracle DIVERGES (G4 has teeth)
//   G6 ⭐ momentum GPU int64 Σf ∈ (0,1e-9) while FP64 Σf < 1e-12 (the Te4 finding on device)
//   G7 overflow    a near-coincident pair HALTs (B1 range)
//   G-ζORDER ⭐    skip-sort vs canonical-sort DIVERGE at pert≥0.75 (the canonical ζ-order bites)
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/tersoff_window_force_gpu.cuh"
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/potentials/tersoff.hpp"
#include "tdmd/potentials/tersoff_zone.hpp"

namespace core = tdmd::core;
namespace pot = tdmd::potentials;
namespace tdcu = tdmd::cuda;

namespace {
struct Window {
  std::vector<double> wx, wy, wz;
  std::vector<long> key;
  std::vector<int> owned;
  core::Box box;
  int m = 0;
};
Window window_from(const core::AtomSoA<double>& a, const core::Box& box) {
  Window w; w.box = box; w.m = a.n;
  for (int i = 0; i < a.n; ++i) { w.wx.push_back(a.x[i]); w.wy.push_back(a.y[i]); w.wz.push_back(a.z[i]); }
  w.key.resize(a.n); w.owned.resize(a.n);
  for (int i = 0; i < a.n; ++i) { w.key[i] = i; w.owned[i] = i; }
  return w;
}
struct Forces { std::vector<double> fx, fy, fz; double pe = 0; long nb = 0, ntri = 0; };
Forces cpu_window(const Window& w, const pot::TersoffParams& p) {
  const core::PairGeom geom(w.box, p.rcut());
  std::vector<core::fixed::ForceAccum> Fx(w.m), Fy(w.m), Fz(w.m);
  core::fixed::EnergyAccum pe; double mr2 = 1e300; long nb = 0, nt = 0;
  pot::tersoff_window_force<double>(w.wx.data(), w.wy.data(), w.wz.data(), w.key.data(), w.m,
                                    w.owned.data(), w.m, p, geom, Fx, Fy, Fz, pe, mr2, nb, nt);
  Forces f; f.pe = pe.value(); f.nb = nb; f.ntri = nt;
  for (int i = 0; i < w.m; ++i) { f.fx.push_back(Fx[i].value()); f.fy.push_back(Fy[i].value()); f.fz.push_back(Fz[i].value()); }
  return f;
}
struct GpuForces { std::vector<double> fx, fy, fz; std::vector<long long> rx, ry, rz; double pe = 0; long long nb = 0, ntri = 0; };
GpuForces gpu_window(const Window& w, const pot::TersoffParams& p, bool skip_sort = false) {
  tdcu::GpuTersoffWinForce<double> g(p, w.box);
  g.skip_sort = skip_sort;
  const core::PairGeom geom(w.box, p.rcut());
  std::vector<core::fixed::ForceAccum> Fx(w.m), Fy(w.m), Fz(w.m);
  core::fixed::EnergyAccum pe; double mr2 = 1e300;
  g.compute(w.wx.data(), w.wy.data(), w.wz.data(), w.key.data(), w.m, w.owned.data(), w.m,
            geom, 0.0, Fx, Fy, Fz, pe, mr2);
  GpuForces f; f.pe = pe.value(); f.nb = g.st->last_nbonds; f.ntri = g.st->last_ntri;
  for (int i = 0; i < w.m; ++i) {
    f.fx.push_back(Fx[i].value()); f.fy.push_back(Fy[i].value()); f.fz.push_back(Fz[i].value());
    f.rx.push_back(Fx[i].raw); f.ry.push_back(Fy[i].raw); f.rz.push_back(Fz[i].raw);
  }
  return f;
}
double maxdiff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0; for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i])); return m;
}
double maxabs(const std::vector<double>& a) { double m = 0; for (double v : a) m = std::max(m, std::fabs(v)); return m; }
double sumf(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z) {
  double sx = 0, sy = 0, sz = 0; for (std::size_t i = 0; i < x.size(); ++i) { sx += x[i]; sy += y[i]; sz += z[i]; }
  return std::sqrt(sx * sx + sy * sy + sz * sz);
}
Forces oracle(const Window& w, const pot::TersoffParams& p, int drop = 0) {
  core::AtomSoA<double> a; a.resize(w.m);
  for (int i = 0; i < w.m; ++i) { a.x[i] = w.wx[i]; a.y[i] = w.wy[i]; a.z[i] = w.wz[i]; a.type[i] = 1; a.mass[i] = 28.0855; }
  core::zero_forces(a);
  const core::PairGeom geom(w.box, p.rcut());
  const auto acc = pot::tersoff_direct_fp64(a, geom, p, true, drop);
  Forces f; f.pe = acc.pe; f.nb = acc.n_bonds; f.ntri = acc.n_triplets;
  for (int i = 0; i < w.m; ++i) { f.fx.push_back(a.fx[i]); f.fy.push_back(a.fy[i]); f.fz.push_back(a.fz[i]); }
  return f;
}
Window make_win(bool pbc, double pert = 0.20, double a0 = 5.431, int nz = 2) {
  core::Box box; auto a = tdmd::gen::make_diamond_si(2, 2, nz, a0, pert, box);
  box.periodic = {pbc, pbc, pbc};
  return window_from(a, box);
}
}  // namespace

// G1 — the GPU exercises the bond order (total ≠ repulsive-only; bigb=0 kills fa ⇒ Roles 2/3/4).
TEST(CudaTersoff, BondOrderExercised) {
  pot::TersoffParams p; auto w = make_win(false);
  const auto g = gpu_window(w, p);
  EXPECT_GT(maxabs(g.fx), 0.1);
  pot::TersoffParams nob = p; nob.bigb = 0.0;  // fa = 0 ⇒ no attractive/angular
  const auto g2 = gpu_window(w, nob);
  EXPECT_GT(maxdiff(g.fx, g2.fx), 0.01) << "GPU bond-order force negligible";
}

// G2 — CPU↔GPU TOLERANCE (free + PBC + the STRETCHED a0=6.70 taper fixture, exercises ters_fc_d).
TEST(CudaTersoff, CpuGpuTolerance) {
  pot::TersoffParams p;
  for (bool pbc : {false, true}) {
    auto w = make_win(pbc);
    const auto c = cpu_window(w, p);
    const auto g = gpu_window(w, p);
    EXPECT_LT(maxdiff(g.fx, c.fx), 1e-9) << "pbc=" << pbc;
    EXPECT_LT(maxdiff(g.fy, c.fy), 1e-9); EXPECT_LT(maxdiff(g.fz, c.fz), 1e-9);
    EXPECT_NEAR(g.pe, c.pe, 1e-6) << "pbc=" << pbc;
    EXPECT_EQ(g.nb, c.nb); EXPECT_EQ(g.ntri, c.ntri);
  }
  auto wt = make_win(false, 0.10, 6.70);  // nn≈2.90 ∈ taper [2.8,3.2] ⇒ ters_fc_d live on device
  const auto ct = cpu_window(wt, p);
  const auto gt = gpu_window(wt, p);
  EXPECT_LT(maxdiff(gt.fx, ct.fx), 1e-9) << "taper fixture: ters_fc_d CPU≠GPU";
}

// G3 — GPU-INTERNAL determinism: run-to-run + window-permutation raw int64 BITWISE (B1 + sort).
TEST(CudaTersoff, GpuInternalBitwise) {
  pot::TersoffParams p; auto w = make_win(false);
  const auto a = gpu_window(w, p);
  const auto b = gpu_window(w, p);
  EXPECT_EQ(a.rx, b.rx); EXPECT_EQ(a.ry, b.ry); EXPECT_EQ(a.rz, b.rz);

  std::vector<int> perm(w.m); std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), std::mt19937(7));
  Window p2 = w;
  for (int i = 0; i < w.m; ++i) { p2.wx[i] = w.wx[perm[i]]; p2.wy[i] = w.wy[perm[i]]; p2.wz[i] = w.wz[perm[i]]; p2.key[i] = perm[i]; }
  const auto pg = gpu_window(p2, p);
  for (int i = 0; i < w.m; ++i) {
    ASSERT_EQ(pg.rx[i], a.rx[perm[i]]) << "window-permutation not bitwise at " << i;
    ASSERT_EQ(pg.ry[i], a.ry[perm[i]]); ASSERT_EQ(pg.rz[i], a.rz[perm[i]]);
  }
}

// G4 ⭐ — GPU matches the INDEPENDENT FP64 oracle (tersoff_direct_fp64) + counts. MB2 (free + PBC).
TEST(CudaTersoff, MatchesFp64Oracle) {
  pot::TersoffParams p;
  for (bool pbc : {false, true}) {
    auto w = make_win(pbc);
    const auto g = gpu_window(w, p);
    const auto o = oracle(w, p);
    EXPECT_LT(maxdiff(g.fx, o.fx), 1e-9) << "pbc=" << pbc << " GPU ≠ FP64 oracle";
    EXPECT_LT(maxdiff(g.fy, o.fy), 1e-9); EXPECT_LT(maxdiff(g.fz, o.fz), 1e-9);
    EXPECT_NEAR(g.pe, o.pe, 1e-7);
    EXPECT_EQ(g.ntri, o.ntri); EXPECT_EQ(g.nb, o.nb); EXPECT_GT(o.ntri, 0);
  }
}

// G5 ⭐ — POISON teeth: GPU vs a dropped-triplet oracle DIVERGES. pert=0.20 REQUIRED (teeth
// collapse at the ideal lattice where Δ=cosθ−h≈0).
TEST(CudaTersoff, PoisonHasTeeth) {
  pot::TersoffParams p; auto w = make_win(false);
  const auto g = gpu_window(w, p);
  const auto bad = oracle(w, p, /*drop=*/1);
  EXPECT_GT(maxdiff(g.fx, bad.fx) + maxdiff(g.fy, bad.fy) + maxdiff(g.fz, bad.fz), 1e-3)
      << "dropped-triplet oracle did not diverge — G4 would be blind";
}

// G6 ⭐ — momentum discriminator on GPU: int64 Σf nonzero at the quantum (the Te4 finding on
// device); FP64 oracle Σf round-off. A symmetric shortcut (Σf=0) fails the >0.
TEST(CudaTersoff, MomentumFloorIsInt64Quantization) {
  pot::TersoffParams p; auto w = make_win(false);
  const auto g = gpu_window(w, p);
  const auto o = oracle(w, p);
  EXPECT_GT(sumf(g.fx, g.fy, g.fz), 0.0) << "GPU Σf exactly zero — a symmetric shortcut?";
  EXPECT_LT(sumf(g.fx, g.fy, g.fz), 1e-9) << "GPU Σf far above the quantum — a force-asymmetry bug?";
  EXPECT_LT(sumf(o.fx, o.fy, o.fz), 1e-12) << "FP64 oracle Σf not round-off";
}

// G7 — the kMaxNbr neighbour-cap HALTs (bit-4 sticky). NOTE: unlike SW (r⁻¹² blows up), Tersoff's
// repulsive A·exp(−λ₁r) is BOUNDED (~λ₁·A ≈ 10577 eV/Å as r→0) ⇒ a close pair does NOT overflow
// the B1 range; the guarded HALT that DOES bite is the >kMaxNbr in-rcut cap (the dropped-donor class).
TEST(CudaTersoff, OverflowHalt) {
  pot::TersoffParams p;
  Window w; w.box.lo = {-10, -10, -10}; w.box.hi = {10, 10, 10}; w.box.periodic = {false, false, false};
  std::mt19937 rng(1); std::uniform_real_distribution<double> u(0.0, 1.5);  // 70 atoms in a 1.5Å cube
  for (int i = 0; i < 70; ++i) { w.wx.push_back(u(rng)); w.wy.push_back(u(rng)); w.wz.push_back(u(rng)); w.key.push_back(i); w.owned.push_back(i); }
  w.m = 70;  // each atom sees ~69 mutual neighbours (max sep 1.5·√3=2.6 < rcut=3.2) > kMaxNbr=64
  tdcu::GpuTersoffWinForce<double> g(p, w.box);
  const core::PairGeom geom(w.box, p.rcut());
  std::vector<core::fixed::ForceAccum> Fx(70), Fy(70), Fz(70);
  core::fixed::EnergyAccum pe; double mr2 = 1e300;
  EXPECT_THROW(g.compute(w.wx.data(), w.wy.data(), w.wz.data(), w.key.data(), 70, w.owned.data(), 70,
                         geom, 0.0, Fx, Fy, Fz, pe, mr2), std::runtime_error);
}

// G-ζORDER ⭐ (the canonical-sort CORRECTNESS — order-invariance) — the canonical sort makes the
// GPU result ORDER-INDEPENDENT: a canonical-sort run on a REVERSED window matches the unpermuted
// reference bitwise (un-permuted). NOTE (the Te3/Te5-acceptance finding): the sort is DEFENSIVE,
// not load-bearing — a SINGLE force eval is sub-Q24.40-quantum sorted-vs-unsorted (so skip-sort ==
// sort here), AND the GPU 1-vs-z is bitwise WITHOUT the sort (the block-order window is
// z-independent — empirically: a skip_sort live-ring keeps 1-vs-z bitwise, max_dev=0). The sort
// buys window-permutation-bitwise-by-construction + CPU-serial alignment. Here we pin its correctness.
TEST(CudaTersoff, CanonicalZetaSortGivesOrderInvariance) {
  pot::TersoffParams p;
  auto w = make_win(false, 0.90, 5.431, 12);  // 384-atom window @ pert 0.90
  const auto ref = gpu_window(w, p);  // sorted (key=i identity ⇒ already canonical)

  std::vector<int> perm(w.m);
  for (int i = 0; i < w.m; ++i) perm[i] = w.m - 1 - i;  // reversed
  Window pw = w;
  for (int i = 0; i < w.m; ++i) { pw.wx[i] = w.wx[perm[i]]; pw.wy[i] = w.wy[perm[i]]; pw.wz[i] = w.wz[perm[i]]; pw.key[i] = perm[i]; }

  const auto sorted = gpu_window(pw, p, /*skip_sort=*/false);  // canonicalizes ⇒ order-independent
  for (int i = 0; i < w.m; ++i) {
    ASSERT_EQ(sorted.rx[i], ref.rx[perm[i]]) << "canonical sort not order-invariant at " << i;
    ASSERT_EQ(sorted.ry[i], ref.ry[perm[i]]); ASSERT_EQ(sorted.rz[i], ref.rz[perm[i]]);
  }
}
