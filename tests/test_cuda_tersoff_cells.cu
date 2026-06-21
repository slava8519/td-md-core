// Tersoff-ladder Te5b — CELL-LIST culled Tersoff GPU kernel (zone_tersoff_cells.cuh). The culled
// 4-role bond-order force MUST be RAW-int64-BITWISE-EQUAL to the all-window kernel (the canonical-ζ
// cull carries the FP64 ζ-sum's order onto the cell grid — Tersoff is MEAM-LIKE, NOT SW-LIKE) AND
// match the independent FP64 oracle (tersoff_direct_fp64) — the SOLE dropped-bond/triplet witness.
//   G-A   cells ≡ all-window RAW int64 BITWISE (fx/fy/fz + pe + n_bonds + n_triplets),
//         ∀cell_div∈{1,2,3,4}, diamond-Si free + PBC + a free cluster. THE canonical-ζ forcing gate
//         — WITHOUT the cull, ζ reassociates and the raw int64 diverges.
//   G-B ⭐ cells vs tersoff_direct_fp64 oracle (BLOCKING — the dropped-bond/triplet witness):
//         force<1e-9, PE<1e-7, n_bonds/n_triplets EXACTLY ==, n_triplets>0, ∀cell_div.
//   G-POISON ⭐ stencil-too-small (drop a ζ-contributor or a triplet) → DIVERGE from the oracle.
//   G-SORT  skip_sort reversed window → measure |delta| (predicted DEFENSIVE like Te3b/Te5 — MEASURE).
//   G-OVERFLOW the CELLS path HALTs (the M1 nb[]/ζ-buffer guard) on a dense fixture; memcheck clean.
//   G-W   tight-PBC boundary (n_k=2k+1, the off-by-one) via the oracle at k∈{3,4}; momentum preserved.
// Compiled with --fmad=false (tdmd_eam_cuda_flags). zone_tersoff.cuh / zone_cells.cuh BYTE-UNTOUCHED.
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
#include "tdmd/cuda/tersoff_window_force_gpu.cuh"  // GpuTersoffWinForce (cull flag) + the cells kernel
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/potentials/tersoff.hpp"

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
  double pe = 0; long long nb = 0, ntri = 0;
};
// One GPU run over the window. cull=false ⇒ all-window (the in-process bitwise REFERENCE);
// cull=true ⇒ the cells path (canonical-ζ cull). cell_div/skip_sort/poison_s only matter when cull
// (skip_sort also exercises the all-window canonical sort hook).
GpuForces gpu_window(const Window& w, const pot::TersoffParams& p, bool cull, int cell_div = 0,
                     bool skip_sort = false, int poison_s = 0) {
  tdcu::GpuTersoffWinForce<double> g(p, w.box, cull, cell_div);
  g.skip_sort = skip_sort; g.poison_s = poison_s;
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
// independent FP64 oracle (tersoff_direct_fp64 — scatter-per-center, a different enumeration ⇒ the
// SOLE dropped-bond/triplet witness). drop>0 = a poisoned enumeration (the G-POISON oracle ref).
struct OracleF { std::vector<double> fx, fy, fz; double pe = 0; long nb = 0, ntri = 0; };
OracleF oracle(const Window& w, const pot::TersoffParams& p, int drop = 0) {
  core::AtomSoA<double> a; a.resize(w.m);
  for (int i = 0; i < w.m; ++i) { a.x[i] = w.wx[i]; a.y[i] = w.wy[i]; a.z[i] = w.wz[i]; a.type[i] = 1; a.mass[i] = 28.0855; }
  core::zero_forces(a);
  const auto acc = pot::tersoff_direct_fp64(a, core::PairGeom(w.box, p.rcut()), p, true, drop);
  OracleF f; f.pe = acc.pe; f.nb = acc.n_bonds; f.ntri = acc.n_triplets;
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
// diamond-Si window (PBC or free); pert=0.20 keeps the bond-order angular force non-degenerate.
Window dia_win(bool pbc, double pert = 0.20, int nz = 2) {
  core::Box box; auto a = tdmd::gen::make_diamond_si(2, 2, nz, 5.431, pert, box);
  box.periodic = {pbc, pbc, pbc};
  return window_from(a, box);
}
// a free 5-atom cluster — a center with four neighbours (every bond carries ζ-k; exercises the wing
// re-scans with a tiny window where a stencil-too-small drop is sharp).
Window cluster_win() {
  core::Box box; box.lo = {-12, -12, -12}; box.hi = {12, 12, 12}; box.periodic = {false, false, false};
  core::AtomSoA<double> a; a.resize(5);
  const double d = 2.35;  // ~Si nn bond (inside rcut = 3.2 Å)
  const double pos[5][3] = {{0, 0, 0}, {d, 0.3, -0.2}, {-0.4, d, 0.1}, {0.2, -0.3, d}, {-d * 0.6, -d * 0.6, -d * 0.6}};
  for (int i = 0; i < 5; ++i) { a.x[i] = pos[i][0]; a.y[i] = pos[i][1]; a.z[i] = pos[i][2]; a.type[i] = 1; a.mass[i] = 28.0855; }
  return window_from(a, box);
}
int max_nbr(const Window& w, const pot::TersoffParams& p) {
  const core::PairGeom geom(w.box, p.rcut());
  int mx = 0;
  for (int i = 0; i < w.m; ++i) {
    int c = 0;
    for (int j = 0; j < w.m; ++j) { if (j == i) continue; double dx = w.wx[j]-w.wx[i], dy = w.wy[j]-w.wy[i], dz = w.wz[j]-w.wz[i], r2; if (geom.reduce(dx, dy, dz, r2)) ++c; }
    mx = std::max(mx, c);
  }
  return mx;
}

// G-A body: cells(cell_div) ≡ all-window, RAW int64 BITWISE, one window. THE canonical-ζ forcing
// test — without the canonical-ζ cull, ζ reassociates and the raw lanes/forces diverge.
void check_self_equiv(const Window& w, const pot::TersoffParams& p, int cell_div) {
  const auto a = gpu_window(w, p, /*cull=*/false);                 // all-window reference
  const auto c = gpu_window(w, p, /*cull=*/true, cell_div);        // cells
  for (int i = 0; i < w.m; ++i) {
    ASSERT_EQ(c.rx[i], a.rx[i]) << "fx raw atom " << i << " cell_div=" << cell_div;
    ASSERT_EQ(c.ry[i], a.ry[i]) << "fy raw atom " << i << " cell_div=" << cell_div;
    ASSERT_EQ(c.rz[i], a.rz[i]) << "fz raw atom " << i << " cell_div=" << cell_div;
  }
  ASSERT_EQ(c.nb, a.nb) << "n_bonds cells vs all-window, cell_div=" << cell_div;
  ASSERT_EQ(c.ntri, a.ntri) << "n_triplets cells vs all-window, cell_div=" << cell_div;
  ASSERT_EQ(c.pe, a.pe) << "pe cells vs all-window, cell_div=" << cell_div;  // raw int64 fold
}
}  // namespace

// ============================ G-A — cells ≡ all-window (the canonical-ζ forcing gate) ===========
TEST(CudaTersoffCells, SelfEquivBitwise) {
  pot::TersoffParams p;
  for (int k : {1, 2, 3, 4}) {
    check_self_equiv(dia_win(false), p, k);
    check_self_equiv(dia_win(true), p, k);
    check_self_equiv(cluster_win(), p, k);
  }
}

// ============================ G-B ⭐ — cells vs FP64 oracle (the dropped-bond/triplet witness) ===
namespace {
void check_vs_oracle(const Window& w, const pot::TersoffParams& p, int cell_div) {
  const auto c = gpu_window(w, p, /*cull=*/true, cell_div);
  const auto o = oracle(w, p);
  EXPECT_GT(o.ntri, 0) << "no bond-order triplets — the angular cull is untested (cells blind)";
  EXPECT_LT(maxdiff(c.fx, o.fx), 1e-9) << "fx cells vs oracle, cell_div=" << cell_div;
  EXPECT_LT(maxdiff(c.fy, o.fy), 1e-9) << "fy cells vs oracle, cell_div=" << cell_div;
  EXPECT_LT(maxdiff(c.fz, o.fz), 1e-9) << "fz cells vs oracle, cell_div=" << cell_div;
  EXPECT_NEAR(c.pe, o.pe, 1e-7) << "pe cells vs oracle, cell_div=" << cell_div;
  EXPECT_EQ(c.nb, o.nb) << "n_bonds cells vs oracle, cell_div=" << cell_div;
  EXPECT_EQ(c.ntri, o.ntri) << "n_triplets cells vs oracle, cell_div=" << cell_div;
}
}  // namespace
TEST(CudaTersoffCells, VsOracleDiamondFree) {
  pot::TersoffParams p;
  for (int k : {1, 2, 3, 4}) check_vs_oracle(dia_win(false), p, k);
}
TEST(CudaTersoffCells, VsOracleDiamondPbc) {
  pot::TersoffParams p;
  for (int k : {1, 2, 3, 4}) check_vs_oracle(dia_win(true), p, k);
}
TEST(CudaTersoffCells, VsOracleCluster) {
  pot::TersoffParams p;
  for (int k : {1, 2, 3, 4}) check_vs_oracle(cluster_win(), p, k);
}

// ============================ G-POISON ⭐ — stencil-too-small MUST diverge =======================
// Cells sized rc/k (k≥3) ⇒ a Role-3/4 wing neighbour at ~2·rcut graph distance (or a ζ-k at ~rcut)
// sits beyond a forced s=1 stencil ⇒ a ζ-contributor / triplet is dropped ⇒ the force gap blows up
// AND n_triplets collapses. If neither bites, G-B is blind to a dropped contribution.
TEST(CudaTersoffCells, PoisonStencilTooSmallHasTeeth) {
  pot::TersoffParams p;
  for (int k : {3, 4}) {
    auto w = dia_win(false);
    const auto bad = gpu_window(w, p, /*cull=*/true, /*cell_div=*/k, /*skip_sort=*/false,
                                /*poison_s=*/1);
    const auto o = oracle(w, p);
    EXPECT_GT(maxdiff(bad.fx, o.fx) + maxdiff(bad.fy, o.fy) + maxdiff(bad.fz, o.fz), 1e-2)
        << "poison stencil s=1 (cells rc/" << k << ") did NOT drop a ζ-k/triplet — G-B blind";
    EXPECT_LT(bad.ntri, o.ntri)
        << "poison stencil did not collapse n_triplets (k=" << k << "): the dropped contrib is invisible";
  }
}

// ============================ G-SORT — the canonical-ζ order verdict (MEASURE-FIRST) ============
// The DESIGN predicted the canonical-ζ cull would be LOAD-BEARING for Tersoff (the bond-order ζ sum
// Σ_k fc·g·exp is FP-order-sensitive). MEASURE it, don't assume — print |delta| of skip_sort vs
// canonical on BOTH the all-window and the cells path, on the DENSEST fixture (a 384-atom jittered
// diamond). PREDICTED (the Te3b/Te5 finding, recorded in the headers): the ζ reassociation stays
// SUB-QUANTUM at Q24.40 on these Si fixtures ⇒ the canonical-ζ sort is DEFENSIVE here, not load-
// bearing. The cull is kept regardless: it makes cells bitwise-to-all-window BY CONSTRUCTION (G-A).
TEST(CudaTersoffCells, SortVerdictAndOrderInvariance) {
  pot::TersoffParams p;
  auto w = dia_win(false, 0.90, 12);  // 384-atom window @ pert 0.90 (the densest fixture)
  const auto ref = gpu_window(w, p, /*cull=*/false);  // all-window, canonical order (key=i identity)

  std::vector<int> rev(w.m); for (int i = 0; i < w.m; ++i) rev[i] = w.m - 1 - i;
  Window rw = w;
  for (int i = 0; i < w.m; ++i) { rw.wx[i] = w.wx[rev[i]]; rw.wy[i] = w.wy[rev[i]]; rw.wz[i] = w.wz[rev[i]]; rw.key[i] = rev[i]; }
  const auto aw_skip = gpu_window(rw, p, /*cull=*/false, /*cell_div=*/0, /*skip_sort=*/true);
  const auto sorted = gpu_window(rw, p, /*cull=*/true, /*cell_div=*/0, /*skip_sort=*/false);
  const auto unsorted = gpu_window(rw, p, /*cull=*/true, /*cell_div=*/0, /*skip_sort=*/true);
  long long d_aw = 0, d_cl = 0;
  for (int i = 0; i < w.m; ++i) {
    d_aw += std::llabs(ref.rx[rev[i]] - aw_skip.rx[i]);  // all-window: canonical vs skip
    d_cl += std::llabs(sorted.rx[i] - unsorted.rx[i]);   // cells: canonical vs skip
  }
  fprintf(stderr, "[G-SORT m=%d] all-window |sort-skip|=%lld  cells |sort-skip|=%lld "
                  "(0=defensive on this fixture, >0=load-bearing)\n", w.m, d_aw, d_cl);
  // the canonical-ζ cull ⇒ the reversed cells window is order-invariant (un-permuted bitwise to ref).
  for (int i = 0; i < w.m; ++i) {
    ASSERT_EQ(sorted.rx[i], ref.rx[rev[i]]) << "cells canonical sort not order-invariant at " << i;
    ASSERT_EQ(sorted.ry[i], ref.ry[rev[i]]);
    ASSERT_EQ(sorted.rz[i], ref.rz[rev[i]]);
  }
}

// ============================ G-OVERFLOW (M1 guard) — the CELLS path HALTs ======================
// The cull GATHERS o's in-rc neighbours into nb[kMaxNbr] AND the ζ-k candidates into a local buffer;
// a dense fixture (70 atoms within ~1.5 Å, every atom a mutual neighbour) overflows them ⇒ WITHOUT
// the guards a silent local-array OOB, WITH the guards a sticky HALT. memcheck must stay clean.
TEST(CudaTersoffCells, OverflowHaltOnDenseCube) {
  pot::TersoffParams p;
  Window w; w.box.lo = {-10, -10, -10}; w.box.hi = {10, 10, 10}; w.box.periodic = {false, false, false};
  std::mt19937 rng(1); std::uniform_real_distribution<double> u(0.0, 1.5);  // dense within rcut=3.2
  for (int i = 0; i < 70; ++i) { w.wx.push_back(u(rng)); w.wy.push_back(u(rng)); w.wz.push_back(u(rng)); w.key.push_back(i); w.owned.push_back(i); }
  w.m = 70;
  EXPECT_THROW(gpu_window(w, p, /*cull=*/true), std::runtime_error) << "cells path silently truncated (OOB?)";
}

// ============================ G-W — tight-PBC degeneracy boundary (k∈{3,4}) =====================
// At sub-rcut k the realized n_k brushes the n=2k+1 wrap boundary. A wing/ζ-k neighbour at the
// stencil edge must still be reached. The independent FP64 oracle is the witness; momentum preserved.
TEST(CudaTersoffCells, TightPbcBoundaryViaOracle) {
  pot::TersoffParams p;
  for (int k : {3, 4}) {
    auto w = dia_win(true);
    check_vs_oracle(w, p, k);
    // confirm the grid actually wraps at this k (the boundary is exercised, not bypassed).
    const double lo[3] = {w.box.lo[0], w.box.lo[1], w.box.lo[2]};
    const double len[3] = {w.box.len(0), w.box.len(1), w.box.len(2)};
    const bool per[3] = {w.box.periodic[0], w.box.periodic[1], w.box.periodic[2]};
    const auto g = tdcu::make_zone_grid(lo, len, per, p.rcut(), 1, 0, k);
    EXPECT_TRUE(!g.wrapx || g.nx >= 2 * g.sx + 1) << "k=" << k << " x wrap double-visit";
    EXPECT_TRUE(!g.wrapy || g.ny >= 2 * g.sy + 1) << "k=" << k << " y wrap double-visit";
  }
}

// ============================ G-kMaxNbr — cull preserves the in-rc count (<64) ==================
TEST(CudaTersoffCells, NeighbourCapMargin) {
  pot::TersoffParams p;
  const int dia = max_nbr(dia_win(false), p);
  fprintf(stderr, "[G-kMaxNbr] realized max in-rc neighbours: diamond=%d (cap=%d, ζ-cap=%d)\n",
          dia, tdcu::kMaxNbr, tdcu::kZetaMax);
  EXPECT_LT(dia, tdcu::kMaxNbr);
  EXPECT_LT(dia, tdcu::kZetaMax);
  EXPECT_NO_THROW(gpu_window(dia_win(false), p, /*cull=*/true));
  EXPECT_NO_THROW(gpu_window(dia_win(true), p, /*cull=*/true));
}

// ============================ G-momentum — the int64 floor carried through the cull ============
TEST(CudaTersoffCells, MomentumFloorPreserved) {
  pot::TersoffParams p;
  auto w = dia_win(false);
  const auto g = gpu_window(w, p, /*cull=*/true);
  const auto o = oracle(w, p);
  const double sf_gpu = sumf(g.fx, g.fy, g.fz);
  EXPECT_GT(sf_gpu, 0.0) << "cells Σf exactly zero — a symmetric shortcut?";
  EXPECT_LT(sf_gpu, 1e-9) << "cells Σf far above the quantum — a force-asymmetry bug?";
  EXPECT_LT(sumf(o.fx, o.fy, o.fz), 1e-12) << "FP64 oracle Σf not round-off";
}
