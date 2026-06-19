// M6 / SW-ladder T5: per-window GPU Stillinger-Weber kernel (the φ3 transpose accumulator).
// Design wf_f13d5221-b7e. CONTRACT: CPU↔GPU is TOLERANCE (exp/pow), GPU-INTERNAL is BITWISE.
// Gates (⭐ = catches deterministic-but-wrong physics blind to consistency checks):
//   G1 vacuity     GPU exercises φ3 (total ≠ φ2-only)
//   G2 CPU↔GPU     |F_gpu − F_cpu| < 1e-9, |pe| < 1e-6, n_triplets ==  (NOT raw-bitwise)
//   G3 GPU bitwise run-to-run + window-permutation: raw int64 identical (order-free B1)
//   G4 ⭐ oracle   |F_gpu − sw_direct_fp64| < 1e-9 + count == oracle (free AND PBC-seam) — MB2
//   G5 ⭐ poison   GPU vs dropped-triplet oracle DIVERGES (G4 has teeth)
//   G6 ⭐ momentum GPU int64 Σf ∈ (0, 1e-7) while FP64 Σf < 1e-12 (the non-symmetric finding)
//   G7 overflow    a near-coincident pair HALTs (B1 range, symmetric to the CPU throw)
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/sw_window_force_gpu.cuh"
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/potentials/sw.hpp"
#include "tdmd/potentials/sw_zone.hpp"

namespace core = tdmd::core;
namespace pot = tdmd::potentials;
namespace cuda = tdmd::cuda;

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
// CPU reference forces (sw_window_force) decoded to double; returns (fx,fy,fz,pe,n_tri).
struct Forces { std::vector<double> fx, fy, fz; double pe = 0; long ntri = 0; };
Forces cpu_window(const Window& w, const pot::SwParams& sp) {
  const core::PairGeom geom(w.box, sp.rcut());
  std::vector<core::fixed::ForceAccum> Fx(w.m), Fy(w.m), Fz(w.m);
  core::fixed::EnergyAccum pe; double mr2 = 1e300; long nt = 0;
  pot::sw_window_force<double>(w.wx.data(), w.wy.data(), w.wz.data(), w.key.data(), w.m,
                              w.owned.data(), w.m, sp, geom, Fx, Fy, Fz, pe, mr2, nt);
  Forces f; f.pe = pe.value(); f.ntri = nt;
  for (int i = 0; i < w.m; ++i) { f.fx.push_back(Fx[i].value()); f.fy.push_back(Fy[i].value()); f.fz.push_back(Fz[i].value()); }
  return f;
}
// GPU forces via the policy; raw=true returns the raw int64 (for G3 bitwise).
struct GpuForces { std::vector<double> fx, fy, fz; std::vector<long long> rx, ry, rz; double pe = 0; long long ntri = 0; };
GpuForces gpu_window(const Window& w, const pot::SwParams& sp) {
  cuda::GpuSwWinForce<double> g(sp, w.box);
  const core::PairGeom geom(w.box, sp.rcut());
  std::vector<core::fixed::ForceAccum> Fx(w.m), Fy(w.m), Fz(w.m);
  core::fixed::EnergyAccum pe; double mr2 = 1e300;
  g.compute(w.wx.data(), w.wy.data(), w.wz.data(), w.key.data(), w.m, w.owned.data(), w.m,
            geom, 0.0, Fx, Fy, Fz, pe, mr2);
  GpuForces f; f.pe = pe.value(); f.ntri = g.st->last_ntri;
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
// oracle forces (sw_direct_fp64) on the window atoms (free cluster ⇒ window == full system).
Forces oracle(const Window& w, const pot::SwParams& sp, int drop = 0) {
  core::AtomSoA<double> a; a.resize(w.m);
  for (int i = 0; i < w.m; ++i) { a.x[i] = w.wx[i]; a.y[i] = w.wy[i]; a.z[i] = w.wz[i]; a.type[i] = 1; a.mass[i] = 28.0855; }
  core::zero_forces(a);
  const auto acc = pot::sw_direct_fp64(a, w.box, sp, true, drop);
  Forces f; f.pe = acc.pe; f.ntri = acc.n_triplets;
  for (int i = 0; i < w.m; ++i) { f.fx.push_back(a.fx[i]); f.fy.push_back(a.fy[i]); f.fz.push_back(a.fz[i]); }
  return f;
}
Window make_win(bool pbc, double pert = 0.20) {
  core::Box box; auto a = tdmd::gen::make_diamond_si(2, 2, 2, 5.431, pert, box);
  box.periodic = {pbc, pbc, pbc};
  return window_from(a, box);
}
}  // namespace

// G1 — the GPU exercises the angular term (total ≠ φ2-only pair force).
TEST(CudaSw, AngularTermExercised) {
  pot::SwParams sp; auto w = make_win(false);
  const auto g = gpu_window(w, sp);
  EXPECT_GT(maxabs(g.fx), 0.05);
  // φ2-only reference: zero λ ⇒ no φ3.
  pot::SwParams nophi3 = sp; nophi3.lambda = 0.0;
  const auto g2 = gpu_window(w, nophi3);
  EXPECT_GT(maxdiff(g.fx, g2.fx), 0.01) << "GPU angular force negligible";
}

// G2 — CPU↔GPU TOLERANCE (exp/pow differ libm-vs-CUDA): NOT raw-bitwise.
TEST(CudaSw, CpuGpuTolerance) {
  pot::SwParams sp;
  for (bool pbc : {false, true}) {
    auto w = make_win(pbc);
    const auto c = cpu_window(w, sp);
    const auto g = gpu_window(w, sp);
    EXPECT_LT(maxdiff(g.fx, c.fx), 1e-9) << "pbc=" << pbc;
    EXPECT_LT(maxdiff(g.fy, c.fy), 1e-9); EXPECT_LT(maxdiff(g.fz, c.fz), 1e-9);
    EXPECT_NEAR(g.pe, c.pe, 1e-6) << "pbc=" << pbc;
    EXPECT_EQ(g.ntri, c.ntri) << "triplet count CPU≠GPU";
  }
}

// G3 — GPU-INTERNAL determinism: run-to-run + window-permutation raw int64 BITWISE.
TEST(CudaSw, GpuInternalBitwise) {
  pot::SwParams sp; auto w = make_win(false);
  const auto a = gpu_window(w, sp);
  const auto b = gpu_window(w, sp);
  EXPECT_EQ(a.rx, b.rx); EXPECT_EQ(a.ry, b.ry); EXPECT_EQ(a.rz, b.rz);  // run-to-run

  // window-permutation: shuffle the gather order; owned forces must be bitwise-invariant.
  std::vector<int> perm(w.m); std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), std::mt19937(7));
  Window p = w;
  for (int i = 0; i < w.m; ++i) { p.wx[i] = w.wx[perm[i]]; p.wy[i] = w.wy[perm[i]]; p.wz[i] = w.wz[perm[i]]; p.key[i] = perm[i]; }
  const auto pg = gpu_window(p, sp);
  for (int i = 0; i < w.m; ++i) {
    ASSERT_EQ(pg.rx[i], a.rx[perm[i]]) << "window-permutation not bitwise at " << i;
    ASSERT_EQ(pg.ry[i], a.ry[perm[i]]); ASSERT_EQ(pg.rz[i], a.rz[perm[i]]);
  }
}

// G4 ⭐ — GPU matches the INDEPENDENT FP64 oracle (sw_direct_fp64, scatter-per-center) +
// count == oracle. The MB2 dropped-triplet witness (free AND PBC-seam).
TEST(CudaSw, MatchesFp64Oracle) {
  pot::SwParams sp;
  for (bool pbc : {false, true}) {
    auto w = make_win(pbc);
    const auto g = gpu_window(w, sp);
    const auto o = oracle(w, sp);
    EXPECT_LT(maxdiff(g.fx, o.fx), 1e-9) << "pbc=" << pbc << " GPU ≠ FP64 oracle";
    EXPECT_LT(maxdiff(g.fy, o.fy), 1e-9); EXPECT_LT(maxdiff(g.fz, o.fz), 1e-9);
    EXPECT_NEAR(g.pe, o.pe, 1e-7);
    EXPECT_EQ(g.ntri, o.ntri); EXPECT_GT(o.ntri, 0);
  }
}

// G5 ⭐ — POISON teeth: GPU vs a dropped-triplet-class oracle DIVERGES (G4 is meaningful).
// GEOMETRICALLY FRAGILE: at the ideal lattice (pert=0) Δ=cosθ−cos0≈0 ⇒ φ₃ force≈0 ⇒ the
// divergence collapses; make_win's pert=0.20 is REQUIRED to keep the teeth (the angular
// force O(0.1–1 eV/Å)). Do NOT reduce the fixture perturbation.
TEST(CudaSw, PoisonHasTeeth) {
  pot::SwParams sp; auto w = make_win(false);
  const auto g = gpu_window(w, sp);
  const auto bad = oracle(w, sp, /*drop=*/1);
  EXPECT_GT(maxdiff(g.fx, bad.fx) + maxdiff(g.fy, bad.fy) + maxdiff(g.fz, bad.fz), 1e-3)
      << "dropped-triplet oracle did not diverge — G4 would be blind";
}

// G6 ⭐ — momentum discriminator on GPU: the int64 Σf is nonzero at the quantum (the T4
// finding replicates on device); the FP64 oracle Σf is round-off. A symmetric shortcut
// (Σf=0) would fail the >0.
TEST(CudaSw, MomentumFloorIsInt64Quantization) {
  pot::SwParams sp; auto w = make_win(false);
  const auto g = gpu_window(w, sp);
  const auto o = oracle(w, sp);
  const double sf_gpu = sumf(g.fx, g.fy, g.fz);
  const double sf_oracle = sumf(o.fx, o.fy, o.fz);
  EXPECT_GT(sf_gpu, 0.0) << "GPU Σf exactly zero — a symmetric shortcut?";
  EXPECT_LT(sf_gpu, 1e-9) << "GPU Σf far above the quantum (~1.4e-11 floor) — a force-asymmetry bug?";
  EXPECT_LT(sf_oracle, 1e-12) << "FP64 oracle Σf not round-off";
}

// G7 — a near-coincident pair HALTs (B1 range overflow), symmetric to the CPU throw.
TEST(CudaSw, OverflowHalt) {
  pot::SwParams sp;
  Window w; w.box.lo = {-10, -10, -10}; w.box.hi = {10, 10, 10}; w.box.periodic = {false, false, false};
  w.wx = {0.0, 0.1}; w.wy = {0.0, 0.0}; w.wz = {0.0, 0.0};  // r = 0.1 Å ⇒ huge φ2 force
  w.key = {0, 1}; w.owned = {0, 1}; w.m = 2;
  cuda::GpuSwWinForce<double> g(sp, w.box);
  const core::PairGeom geom(w.box, sp.rcut());
  std::vector<core::fixed::ForceAccum> Fx(2), Fy(2), Fz(2);
  core::fixed::EnergyAccum pe; double mr2 = 1e300;
  EXPECT_THROW(g.compute(w.wx.data(), w.wy.data(), w.wz.data(), w.key.data(), 2, w.owned.data(), 2,
                         geom, 0.0, Fx, Fy, Fz, pe, mr2), std::runtime_error);
}
