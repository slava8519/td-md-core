// MEAM-ladder Me5: per-window GPU MEAM kernels (K1 density / K2 embedding / K3 the screened 3-role
// transpose-replay). Design docs/_meta/M5_MEAM_GPU_DESIGN_2026-06-20.md. CONTRACT: CPU↔GPU is
// TOLERANCE (exp/log/pow ~1 ulp), GPU-INTERNAL is BITWISE (B1 + the canonical sort). The diamond is
// BINARY-S ⇒ Role C structurally DEAD ⇒ the screening gates run on the PARTIAL-screening SLAB.
//   G1 kMaxNbr   realized max neighbour count on diamond + slab (MEASURE the 64 headroom)
//   G2 CPU↔GPU   |F_gpu − F_cpu(meam_run_fixed_force/window)| < 1e-9, |pe| < 1e-6 (diamond + slab)
//   G3 GPU bitwise run-to-run + window-permutation raw int64 == ; RE-MEASURE the sort (skip_sort)
//   G4 ⭐ oracle  |F_gpu − meam_direct_fp64| < 1e-9 + counts == (slab free) — the dropped-k witness
//   G5 ⭐ oracle  same on the slab PBC seam (two-wing j–k difference leg)
//   G6 ⭐ poison  drop_class>0 (Role C dropped) DIVERGES from the oracle on the slab (G4 has teeth)
//   G7 ⭐ momentum slab GPU int64 Σf ∈ (0,1e-9); FP64 oracle Σf round-off; diamond Σf int64-exact
//   G8 overflow  a dense cluster (>kMaxNbr in-rc) HALTs (sticky bit, before writeback)
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
#include "tdmd/cuda/meam_window_force_gpu.cuh"
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/gen/partial_screen_slab.hpp"
#include "tdmd/potentials/meam.hpp"
#include "tdmd/potentials/meam_zone.hpp"

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
struct Forces { std::vector<double> fx, fy, fz; double pe = 0; long np = 0, nz = 0; };
struct GpuForces { std::vector<double> fx, fy, fz; std::vector<long long> rx, ry, rz; double pe = 0; long long np = 0, nz = 0; };

// CPU per-window via meam_window_force (the int64 scatter the zone path uses).
Forces cpu_window(const Window& w, const pot::MeamParams& p, int drop = 0) {
  const core::PairGeom geom(w.box, p.rc);
  std::vector<core::fixed::ForceAccum> Fx(w.m), Fy(w.m), Fz(w.m);
  core::fixed::EnergyAccum pe_embed, pe_pair; double mr2 = 1e300; long np = 0, nz = 0;
  pot::meam_window_force<double>(w.wx.data(), w.wy.data(), w.wz.data(), w.key.data(), w.m,
                                 w.owned.data(), w.m, p, geom, Fx, Fy, Fz, pe_embed, pe_pair, mr2,
                                 np, nz, drop);
  Forces f; f.pe = pe_embed.value() + pe_pair.value(); f.np = np; f.nz = nz;
  for (int i = 0; i < w.m; ++i) { f.fx.push_back(Fx[i].value()); f.fy.push_back(Fy[i].value()); f.fz.push_back(Fz[i].value()); }
  return f;
}
GpuForces gpu_window(const Window& w, const pot::MeamParams& p, bool skip_sort = false, int drop = 0) {
  tdcu::GpuMeamWinForce<double> g(p, w.box);
  g.skip_sort = skip_sort; g.drop_class = drop;
  const core::PairGeom geom(w.box, p.rc);
  std::vector<core::fixed::ForceAccum> Fx(w.m), Fy(w.m), Fz(w.m);
  core::fixed::EnergyAccum pe; double mr2 = 1e300;
  g.compute(w.wx.data(), w.wy.data(), w.wz.data(), w.key.data(), w.m, w.owned.data(), w.m,
            geom, 0.0, Fx, Fy, Fz, pe, mr2);
  GpuForces f; f.pe = pe.value(); f.np = g.st->last_npartial; f.nz = g.st->last_nzero;
  for (int i = 0; i < w.m; ++i) {
    f.fx.push_back(Fx[i].value()); f.fy.push_back(Fy[i].value()); f.fz.push_back(Fz[i].value());
    f.rx.push_back(Fx[i].raw); f.ry.push_back(Fy[i].raw); f.rz.push_back(Fz[i].raw);
  }
  return f;
}
// independent FP64 oracle (meam_direct_fp64 — different enumeration, the SOLE completeness witness).
Forces oracle(const Window& w, const pot::MeamParams& p, int drop = 0) {
  core::AtomSoA<double> a; a.resize(w.m);
  for (int i = 0; i < w.m; ++i) { a.x[i] = w.wx[i]; a.y[i] = w.wy[i]; a.z[i] = w.wz[i]; a.type[i] = 1; a.mass[i] = 28.0855; }
  core::zero_forces(a);
  const auto acc = pot::meam_direct_fp64(a, core::PairGeom(w.box, p.rc), p, true, drop);
  Forces f; f.pe = acc.pe; f.np = acc.n_screened_partial; f.nz = acc.n_screened_zero;
  for (int i = 0; i < w.m; ++i) { f.fx.push_back(a.fx[i]); f.fy.push_back(a.fy[i]); f.fz.push_back(a.fz[i]); }
  return f;
}
double maxdiff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0; for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i])); return m;
}
double sumf(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z) {
  double sx = 0, sy = 0, sz = 0; for (std::size_t i = 0; i < x.size(); ++i) { sx += x[i]; sy += y[i]; sz += z[i]; }
  return std::sqrt(sx * sx + sy * sy + sz * sz);
}
Window slab_win(bool pbc) {
  auto s = tdmd::gen::make_partial_screen_slab(8);
  s.box.periodic = {true, true, pbc};
  return window_from(s.atoms, s.box);
}
Window dia_win() {
  core::Box box; auto a = tdmd::gen::make_diamond_si(2, 2, 4, 5.431, 0.15, box);
  box.periodic = {true, true, true};
  return window_from(a, box);
}
// the TAPER-BAND partial-screening triple (Me5-acceptance MUST-FIX): i–j bond 3.95 Å in the radial
// screening taper (rnorm=0.5 ⇒ dfcut≠0 ⇒ the dscrfcn `−coef2` term LIVE on the DEVICE path), k giving
// 0<S<1. The slab/diamond bonds (rnorm≥1 ⇒ coef2=0) leave the device `−coef2` sign untested.
Window taper_win() {
  core::Box box; box.lo = {0, 0, 0}; box.hi = {24, 24, 24}; box.periodic = {false, false, false};
  core::AtomSoA<double> a; a.resize(3);
  const double pos[3][3] = {{10, 10, 10}, {13.95, 10, 10}, {11.975, 12.93, 10}};
  for (int i = 0; i < 3; ++i) { a.x[i] = pos[i][0]; a.y[i] = pos[i][1]; a.z[i] = pos[i][2]; a.type[i] = 1; a.mass[i] = 28.0855; }
  return window_from(a, box);
}
// realized max in-rc neighbour count over the window (the kMaxNbr witness).
int max_nbr(const Window& w, const pot::MeamParams& p) {
  const core::PairGeom geom(w.box, p.rc);
  int mx = 0;
  for (int i = 0; i < w.m; ++i) {
    int c = 0;
    for (int j = 0; j < w.m; ++j) { if (j == i) continue; double dx = w.wx[j]-w.wx[i], dy = w.wy[j]-w.wy[i], dz = w.wz[j]-w.wz[i], r2; if (geom.reduce(dx, dy, dz, r2)) ++c; }
    mx = std::max(mx, c);
  }
  return mx;
}
}  // namespace

// G1 — realized kMaxNbr margin (MEASURE): diamond + slab max in-rc count well under 64.
TEST(CudaMeam, NeighbourCapMargin) {
  pot::MeamParams p;
  const int dia = max_nbr(dia_win(), p);
  const int slab = max_nbr(slab_win(false), p);
  fprintf(stderr, "[G1] realized max in-rc neighbours: diamond=%d slab=%d (cap=%d)\n", dia, slab, tdcu::kMeamMaxNbr);
  EXPECT_LT(dia, tdcu::kMeamMaxNbr);
  EXPECT_LT(slab, tdcu::kMeamMaxNbr);
}

// G2 — CPU↔GPU TOLERANCE (NOT bitwise — exp/log/pow ~1 ulp), diamond + slab (free + PBC).
TEST(CudaMeam, CpuGpuTolerance) {
  pot::MeamParams p;
  for (auto& w : {dia_win(), slab_win(false), slab_win(true)}) {
    const auto c = cpu_window(w, p);
    const auto g = gpu_window(w, p);
    EXPECT_LT(maxdiff(g.fx, c.fx), 1e-9); EXPECT_LT(maxdiff(g.fy, c.fy), 1e-9); EXPECT_LT(maxdiff(g.fz, c.fz), 1e-9);
    EXPECT_NEAR(g.pe, c.pe, 1e-6);
    EXPECT_EQ(g.np, c.np); EXPECT_EQ(g.nz, c.nz);
  }
}

// G3 — GPU-INTERNAL determinism: run-to-run + window-permutation raw int64 BITWISE (B1 + sort).
// RE-MEASURE the sort: skip_sort on the slab — does it flip? (predicted load-bearing for ring≡serial.)
TEST(CudaMeam, GpuInternalBitwiseAndSortVerdict) {
  pot::MeamParams p; auto w = slab_win(false);
  const auto a = gpu_window(w, p);
  const auto b = gpu_window(w, p);
  EXPECT_EQ(a.rx, b.rx); EXPECT_EQ(a.ry, b.ry); EXPECT_EQ(a.rz, b.rz);  // run-to-run

  // window-permutation with the canonical sort ⇒ bitwise-by-construction (un-permuted compare).
  std::vector<int> perm(w.m); std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), std::mt19937(7));
  Window pw = w;
  for (int i = 0; i < w.m; ++i) { pw.wx[i] = w.wx[perm[i]]; pw.wy[i] = w.wy[perm[i]]; pw.wz[i] = w.wz[perm[i]]; pw.key[i] = perm[i]; }
  const auto pg = gpu_window(pw, p, /*skip_sort=*/false);
  for (int i = 0; i < w.m; ++i) {
    ASSERT_EQ(pg.rx[i], a.rx[perm[i]]) << "window-permutation not bitwise at " << i;
    ASSERT_EQ(pg.ry[i], a.ry[perm[i]]); ASSERT_EQ(pg.rz[i], a.rz[perm[i]]);
  }

  // SORT VERDICT (the measure): reversed window, skip_sort vs canonical-sort. Print whether they
  // diverge (load-bearing) or stay bitwise (defensive, the Te3b/Te5 finding).
  std::vector<int> rev(w.m); for (int i = 0; i < w.m; ++i) rev[i] = w.m - 1 - i;
  Window rw = w;
  for (int i = 0; i < w.m; ++i) { rw.wx[i] = w.wx[rev[i]]; rw.wy[i] = w.wy[rev[i]]; rw.wz[i] = w.wz[rev[i]]; rw.key[i] = rev[i]; }
  const auto sorted = gpu_window(rw, p, /*skip_sort=*/false);
  const auto unsorted = gpu_window(rw, p, /*skip_sort=*/true);
  long long delta = 0;
  for (int i = 0; i < w.m; ++i) delta += std::llabs(sorted.rx[i] - unsorted.rx[i]) + std::llabs(sorted.ry[i] - unsorted.ry[i]) + std::llabs(sorted.rz[i] - unsorted.rz[i]);
  fprintf(stderr, "[G3 sort verdict] reversed-window skip_sort vs sort raw-int64 |delta|=%lld (0=defensive, >0=load-bearing)\n", delta);
  // canonical-sort makes the reversed window order-invariant (un-permuted bitwise to the reference).
  for (int i = 0; i < w.m; ++i) {
    ASSERT_EQ(sorted.rx[i], a.rx[rev[i]]) << "canonical sort not order-invariant at " << i;
    ASSERT_EQ(sorted.ry[i], a.ry[rev[i]]); ASSERT_EQ(sorted.rz[i], a.rz[rev[i]]);
  }
}

// G4 ⭐ — GPU vs the INDEPENDENT FP64 oracle (meam_direct_fp64, different enumeration) on the SLAB
// (free): the SOLE dropped-screening-k witness. counts == and partial > 0 (non-vacuous Role C).
TEST(CudaMeam, MatchesFp64OracleSlabFree) {
  pot::MeamParams p; auto w = slab_win(false);
  const auto g = gpu_window(w, p);
  const auto o = oracle(w, p);
  EXPECT_GT(o.np, 0) << "no partial-S triples — Role C untested";
  EXPECT_LT(maxdiff(g.fx, o.fx), 1e-9); EXPECT_LT(maxdiff(g.fy, o.fy), 1e-9); EXPECT_LT(maxdiff(g.fz, o.fz), 1e-9);
  EXPECT_NEAR(g.pe, o.pe, 1e-6);
  EXPECT_EQ(g.np, o.np); EXPECT_EQ(g.nz, o.nz);
}

// G5 ⭐ — same vs the FP64 oracle on the SLAB PBC SEAM (the two-wing j–k difference leg).
TEST(CudaMeam, MatchesFp64OracleSlabPbc) {
  pot::MeamParams p; auto w = slab_win(true);
  const auto g = gpu_window(w, p);
  const auto o = oracle(w, p);
  EXPECT_GT(o.np, 0);
  EXPECT_LT(maxdiff(g.fx, o.fx), 1e-9); EXPECT_LT(maxdiff(g.fy, o.fy), 1e-9); EXPECT_LT(maxdiff(g.fz, o.fz), 1e-9);
  EXPECT_NEAR(g.pe, o.pe, 1e-6);
}

// G5b ⭐ — the DEVICE dscrfcn RADIAL-TAPER derivative (`−coef2`) witness (Me5-acceptance MUST-FIX):
// GPU vs the FP64 oracle on the taper-band partial triple (bond 3.95 Å ⇒ dfcut≠0 ⇒ coef2 LIVE). The
// slab/diamond bonds leave the device `−coef2` SIGN structurally untested (rnorm≥1 ⇒ coef2=0); a
// +coef2 typo in meam_getscreen_d_device is silent there. Here it diverges (CPU FD measured 1.47).
TEST(CudaMeam, MatchesFp64OracleTaperBand) {
  pot::MeamParams p; auto w = taper_win();
  const auto g = gpu_window(w, p);
  const auto o = oracle(w, p);
  EXPECT_EQ(o.np, 1) << "taper triple not partially screened — the device coef2 branch is dead";
  EXPECT_LT(maxdiff(g.fx, o.fx), 1e-9); EXPECT_LT(maxdiff(g.fy, o.fy), 1e-9); EXPECT_LT(maxdiff(g.fz, o.fz), 1e-9);
  EXPECT_NEAR(g.pe, o.pe, 1e-6);
}

// G6 ⭐ — POISON teeth: GPU with Role C dropped DIVERGES from the (full) oracle on the slab.
TEST(CudaMeam, PoisonHasTeeth) {
  pot::MeamParams p; auto w = slab_win(false);
  const auto bad = gpu_window(w, p, /*skip_sort=*/false, /*drop=*/1);
  const auto o = oracle(w, p);
  EXPECT_GT(maxdiff(bad.fx, o.fx) + maxdiff(bad.fy, o.fy) + maxdiff(bad.fz, o.fz), 1e-3)
      << "dropped-Role-C GPU did not diverge — G4 would be blind to a dropped screening-k";
}

// G7 ⭐ — momentum discriminator: slab GPU int64 Σf ∈ (0,1e-9) (the screening transpose quantizes
// f_i,f_j,f_k independently); FP64 oracle Σf round-off; diamond Σf int64-exact (binary-S symmetric).
TEST(CudaMeam, MomentumFloorIsInt64Quantization) {
  pot::MeamParams p;
  auto ws = slab_win(false);
  const auto gs = gpu_window(ws, p);
  const auto os = oracle(ws, p);
  EXPECT_GT(sumf(gs.fx, gs.fy, gs.fz), 0.0) << "slab GPU Σf exactly zero — a symmetric shortcut (Role C dropped)?";
  EXPECT_LT(sumf(gs.fx, gs.fy, gs.fz), 1e-9) << "slab GPU Σf far above the quantum — a force-asymmetry bug?";
  EXPECT_LT(sumf(os.fx, os.fy, os.fz), 1e-12) << "FP64 oracle Σf not round-off";
  auto wd = dia_win();
  const auto gd = gpu_window(wd, p);
  EXPECT_LT(sumf(gd.fx, gd.fy, gd.fz), 1e-9) << "diamond Σf above the quantum (binary-S should cancel in int64)";
}

// G8 — the kMaxNbr neighbour-cap HALTs on a dense cluster (>kMaxNbr in-rc). Sticky bit, thrown
// before writeback (NEVER silent truncation — a dropped neighbour drops a screening-k).
TEST(CudaMeam, OverflowHalt) {
  pot::MeamParams p;
  Window w; w.box.lo = {-10, -10, -10}; w.box.hi = {10, 10, 10}; w.box.periodic = {false, false, false};
  std::mt19937 rng(1); std::uniform_real_distribution<double> u(0.0, 2.5);  // dense cube within rc=4
  for (int i = 0; i < 80; ++i) { w.wx.push_back(u(rng)); w.wy.push_back(u(rng)); w.wz.push_back(u(rng)); w.key.push_back(i); w.owned.push_back(i); }
  w.m = 80;  // each atom sees ~79 mutual neighbours (max sep 2.5·√3≈4.33; many < rc=4) > kMaxNbr=64
  tdcu::GpuMeamWinForce<double> g(p, w.box);
  const core::PairGeom geom(w.box, p.rc);
  std::vector<core::fixed::ForceAccum> Fx(80), Fy(80), Fz(80);
  core::fixed::EnergyAccum pe; double mr2 = 1e300;
  EXPECT_THROW(g.compute(w.wx.data(), w.wy.data(), w.wz.data(), w.key.data(), 80, w.owned.data(), 80,
                         geom, 0.0, Fx, Fy, Fz, pe, mr2), std::runtime_error);
}
