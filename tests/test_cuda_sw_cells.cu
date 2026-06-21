// SW-ladder T5b — CELL-LIST culled Stillinger–Weber GPU kernel (zone_sw_cells.cuh). The culled
// φ₂+φ₃ force MUST be RAW-int64-BITWISE-EQUAL to the all-window kernel (SW is int64-order-free ⇒
// NO canonical sort, the EAM-like proof, UNLIKE MEAM's canonical-k cull) AND match the independent
// FP64 oracle (sw_direct_fp64) — the SOLE dropped-triplet witness.
//   G-A   cells ≡ all-window RAW int64 BITWISE (fx/fy/fz + pe + n_triplets), ∀cell_div∈{1,2,3,4},
//         diamond + free cluster + PBC. The diamond/PBC exercise φ₃. WITHOUT a canonical sort —
//         the proof SW differs from MEAM (int64-order-freedom carries the cull by construction).
//   G-B ⭐ cells vs sw_direct_fp64 oracle (BLOCKING — the dropped-φ₃-triplet witness): force<1e-9,
//         PE<1e-6, n_triplets EXACTLY ==, n_triplets>0, ∀cell_div.
//   G-POISON ⭐ stencil-too-small (drop a φ₃ wing/center triplet) → DIVERGE from the oracle.
//   G-OVERFLOW the CELLS path HALTs (the M1 per-thread nb[] guard) on a dense fixture; memcheck clean.
//   G-W   tight-PBC boundary (n_k=2k+1, the off-by-one) via the oracle at k∈{3,4}; momentum preserved.
// Compiled with --fmad=false (tdmd_eam_cuda_flags). zone_sw.cuh / zone_cells.cuh BYTE-UNTOUCHED.
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/sw_window_force_gpu.cuh"  // GpuSwWinForce (cull flag) + the cells kernel
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/potentials/sw.hpp"

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
struct GpuForces {
  std::vector<double> fx, fy, fz;
  std::vector<long long> rx, ry, rz;  // RAW int64 (the G-A bitwise compare)
  double pe = 0; long long ntri = 0;
};
// One GPU run over the window. cull=false ⇒ all-window (the in-process bitwise REFERENCE);
// cull=true ⇒ the cells path. cell_div/poison_s only matter when cull.
GpuForces gpu_window(const Window& w, const pot::SwParams& sp, bool cull, int cell_div = 0,
                     int poison_s = 0) {
  tdcu::GpuSwWinForce<double> g(sp, w.box, cull, cell_div);
  g.poison_s = poison_s;
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
// independent FP64 oracle (sw_direct_fp64 — scatter-per-center, a different enumeration ⇒ the SOLE
// dropped-triplet witness). drop>0 = a poisoned enumeration (the G-POISON-of-the-oracle reference).
struct OracleF { std::vector<double> fx, fy, fz; double pe = 0; long ntri = 0; };
OracleF oracle(const Window& w, const pot::SwParams& sp, int drop = 0) {
  core::AtomSoA<double> a; a.resize(w.m);
  for (int i = 0; i < w.m; ++i) { a.x[i] = w.wx[i]; a.y[i] = w.wy[i]; a.z[i] = w.wz[i]; a.type[i] = 1; a.mass[i] = 28.0855; }
  core::zero_forces(a);
  const auto acc = pot::sw_direct_fp64(a, w.box, sp, true, drop);
  OracleF f; f.pe = acc.pe; f.ntri = acc.n_triplets;
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
// diamond-Si window (PBC or free); pert=0.20 keeps φ₃ non-degenerate (the angular force teeth).
Window dia_win(bool pbc, double pert = 0.20) {
  core::Box box; auto a = tdmd::gen::make_diamond_si(2, 2, 2, 5.431, pert, box);
  box.periodic = {pbc, pbc, pbc};
  return window_from(a, box);
}
// a free 4-atom tetrahedral cluster — every atom is a φ₃ center AND a wing; exercises the wing
// re-scan with a tiny window where a stencil-too-small drop is sharp.
Window cluster_win() {
  core::Box box; box.lo = {-12, -12, -12}; box.hi = {12, 12, 12}; box.periodic = {false, false, false};
  core::AtomSoA<double> a; a.resize(4);
  const double d = 2.35;  // ~Si nn bond (inside rcut = a·σ ≈ 3.77)
  const double pos[4][3] = {{0, 0, 0}, {d, 0.3, -0.2}, {-0.4, d, 0.1}, {0.2, -0.3, d}};
  for (int i = 0; i < 4; ++i) { a.x[i] = pos[i][0]; a.y[i] = pos[i][1]; a.z[i] = pos[i][2]; a.type[i] = 1; a.mass[i] = 28.0855; }
  return window_from(a, box);
}
int max_nbr(const Window& w, const pot::SwParams& sp) {
  const core::PairGeom geom(w.box, sp.rcut());
  int mx = 0;
  for (int i = 0; i < w.m; ++i) {
    int c = 0;
    for (int j = 0; j < w.m; ++j) { if (j == i) continue; double dx = w.wx[j]-w.wx[i], dy = w.wy[j]-w.wy[i], dz = w.wz[j]-w.wz[i], r2; if (geom.reduce(dx, dy, dz, r2)) ++c; }
    mx = std::max(mx, c);
  }
  return mx;
}

// G-A body: cells(cell_div) ≡ all-window, RAW int64 BITWISE, one window. SW is int64-order-free ⇒
// this passes WITHOUT a canonical sort (the proof SW ≠ MEAM).
void check_self_equiv(const Window& w, const pot::SwParams& sp, int cell_div) {
  const auto a = gpu_window(w, sp, /*cull=*/false);                 // all-window reference
  const auto c = gpu_window(w, sp, /*cull=*/true, cell_div);        // cells
  for (int i = 0; i < w.m; ++i) {
    ASSERT_EQ(c.rx[i], a.rx[i]) << "fx raw atom " << i << " cell_div=" << cell_div;
    ASSERT_EQ(c.ry[i], a.ry[i]) << "fy raw atom " << i << " cell_div=" << cell_div;
    ASSERT_EQ(c.rz[i], a.rz[i]) << "fz raw atom " << i << " cell_div=" << cell_div;
  }
  ASSERT_EQ(c.ntri, a.ntri) << "n_triplets cells vs all-window, cell_div=" << cell_div;
  ASSERT_EQ(c.pe, a.pe) << "pe cells vs all-window, cell_div=" << cell_div;  // raw int64 fold
}
}  // namespace

// ============================ G-A — cells ≡ all-window (NO canonical sort — the EAM-like proof) ==
TEST(CudaSwCells, SelfEquivBitwise) {
  pot::SwParams sp;
  for (int k : {1, 2, 3, 4}) {
    check_self_equiv(dia_win(false), sp, k);
    check_self_equiv(dia_win(true), sp, k);
    check_self_equiv(cluster_win(), sp, k);
  }
}

// ============================ G-B ⭐ — cells vs FP64 oracle (the dropped-triplet witness) =======
namespace {
void check_vs_oracle(const Window& w, const pot::SwParams& sp, int cell_div) {
  const auto c = gpu_window(w, sp, /*cull=*/true, cell_div);
  const auto o = oracle(w, sp);
  EXPECT_GT(o.ntri, 0) << "no φ₃ triplets — the angular cull is untested (cells blind)";
  EXPECT_LT(maxdiff(c.fx, o.fx), 1e-9) << "fx cells vs oracle, cell_div=" << cell_div;
  EXPECT_LT(maxdiff(c.fy, o.fy), 1e-9) << "fy cells vs oracle, cell_div=" << cell_div;
  EXPECT_LT(maxdiff(c.fz, o.fz), 1e-9) << "fz cells vs oracle, cell_div=" << cell_div;
  EXPECT_NEAR(c.pe, o.pe, 1e-6) << "pe cells vs oracle, cell_div=" << cell_div;
  EXPECT_EQ(c.ntri, o.ntri) << "n_triplets cells vs oracle, cell_div=" << cell_div;
}
}  // namespace
TEST(CudaSwCells, VsOracleDiamondFree) {
  pot::SwParams sp;
  for (int k : {1, 2, 3, 4}) check_vs_oracle(dia_win(false), sp, k);
}
TEST(CudaSwCells, VsOracleDiamondPbc) {
  pot::SwParams sp;
  for (int k : {1, 2, 3, 4}) check_vs_oracle(dia_win(true), sp, k);
}
TEST(CudaSwCells, VsOracleCluster) {
  pot::SwParams sp;
  for (int k : {1, 2, 3, 4}) check_vs_oracle(cluster_win(), sp, k);
}

// ============================ G-POISON ⭐ — stencil-too-small MUST diverge =======================
// Cells sized rc/k (k≥3) ⇒ a wing-role neighbour m' at ~2·rcut graph distance (or a φ₃-center
// neighbour at ~rcut) sits beyond a forced s=1 stencil ⇒ a triplet is dropped ⇒ the force gap
// blows up AND n_triplets collapses. If neither bites, G-B is blind to a dropped triplet.
TEST(CudaSwCells, PoisonStencilTooSmallHasTeeth) {
  pot::SwParams sp;
  for (int k : {3, 4}) {
    auto w = dia_win(false);
    const auto bad = gpu_window(w, sp, /*cull=*/true, /*cell_div=*/k, /*poison_s=*/1);
    const auto o = oracle(w, sp);
    EXPECT_GT(maxdiff(bad.fx, o.fx) + maxdiff(bad.fy, o.fy) + maxdiff(bad.fz, o.fz), 1e-2)
        << "poison stencil s=1 (cells rc/" << k << ") did NOT drop a triplet — G-B blind";
    EXPECT_LT(bad.ntri, o.ntri)
        << "poison stencil did not collapse n_triplets (k=" << k << "): the dropped triplet is invisible";
  }
}

// ============================ G-OVERFLOW (M1 guard) — the CELLS path HALTs ======================
// The cull GATHERS o's in-rc neighbours into the per-thread nb[kMaxNbr] buffer; a dense fixture
// (80 atoms within ~2.5 Å, every atom a mutual neighbour) overflows it ⇒ WITHOUT the guard a silent
// local-array OOB, WITH the guard a sticky HALT. memcheck must stay clean.
TEST(CudaSwCells, OverflowHaltOnDenseCube) {
  pot::SwParams sp;
  Window w; w.box.lo = {-10, -10, -10}; w.box.hi = {10, 10, 10}; w.box.periodic = {false, false, false};
  std::mt19937 rng(1); std::uniform_real_distribution<double> u(0.0, 2.5);  // dense within rcut
  for (int i = 0; i < 80; ++i) { w.wx.push_back(u(rng)); w.wy.push_back(u(rng)); w.wz.push_back(u(rng)); w.key.push_back(i); w.owned.push_back(i); }
  w.m = 80;
  EXPECT_THROW(gpu_window(w, sp, /*cull=*/true), std::runtime_error) << "cells path silently truncated (OOB?)";
}

// ============================ G-W — tight-PBC degeneracy boundary (k∈{3,4}) =====================
// At sub-rcut k the realized n_k brushes the n=2k+1 wrap boundary. A wing/center neighbour at the
// stencil edge must still be reached. The independent FP64 oracle is the witness; momentum preserved.
TEST(CudaSwCells, TightPbcBoundaryViaOracle) {
  pot::SwParams sp;
  for (int k : {3, 4}) {
    auto w = dia_win(true);
    check_vs_oracle(w, sp, k);
    // confirm the grid actually wraps at this k (the boundary is exercised, not bypassed).
    const double lo[3] = {w.box.lo[0], w.box.lo[1], w.box.lo[2]};
    const double len[3] = {w.box.len(0), w.box.len(1), w.box.len(2)};
    const bool per[3] = {w.box.periodic[0], w.box.periodic[1], w.box.periodic[2]};
    const auto g = tdcu::make_zone_grid(lo, len, per, sp.rcut(), 1, 0, k);
    EXPECT_TRUE(!g.wrapx || g.nx >= 2 * g.sx + 1) << "k=" << k << " x wrap double-visit";
    EXPECT_TRUE(!g.wrapy || g.ny >= 2 * g.sy + 1) << "k=" << k << " y wrap double-visit";
  }
}

// ============================ G-kMaxNbr — cull preserves the in-rc count (<64) ==================
TEST(CudaSwCells, NeighbourCapMargin) {
  pot::SwParams sp;
  const int dia = max_nbr(dia_win(false), sp);
  fprintf(stderr, "[G-kMaxNbr] realized max in-rc neighbours: diamond=%d (cap=%d)\n",
          dia, tdcu::kMaxNbr);
  EXPECT_LT(dia, tdcu::kMaxNbr);
  EXPECT_NO_THROW(gpu_window(dia_win(false), sp, /*cull=*/true));
  EXPECT_NO_THROW(gpu_window(dia_win(true), sp, /*cull=*/true));
}

// ============================ G-momentum — the int64 floor carried through the cull ============
TEST(CudaSwCells, MomentumFloorPreserved) {
  pot::SwParams sp;
  auto w = dia_win(false);
  const auto g = gpu_window(w, sp, /*cull=*/true);
  const auto o = oracle(w, sp);
  const double sf_gpu = sumf(g.fx, g.fy, g.fz);
  EXPECT_GT(sf_gpu, 0.0) << "cells Σf exactly zero — a symmetric shortcut?";
  EXPECT_LT(sf_gpu, 1e-9) << "cells Σf far above the quantum (~1.4e-11 floor) — a force-asymmetry bug?";
  EXPECT_LT(sumf(o.fx, o.fy, o.fz), 1e-12) << "FP64 oracle Σf not round-off";
}
