// MEAM-ladder Me5b — CELL-LIST culled MEAM GPU kernels (zone_meam_cells.cuh). The culled K1
// density + K3 four-role force MUST be RAW-int64-BITWISE-EQUAL to the all-window kernels (the
// canonical-k cull carries the screening FP product/sum's order onto the cell grid) AND match the
// independent FP64 oracle on the PARTIAL-screening fixtures (the diamond is binary-S ⇒ Role C dead
// ⇒ it cannot witness screening-cull completeness — the recurring structurally-dead trap).
//   G-A   cells ≡ all-window RAW int64 BITWISE (27 dens lanes + fx/fy/fz + pe + np/nz), ∀cell_div,
//         diamond + partial slab (free + PBC) + taper. THE canonical-k forcing gate.
//   G-B ⭐ cells vs meam_direct_fp64 oracle (BLOCKING — the SOLE dropped-screening-k witness):
//         force<1e-9, PE<1e-6, np/nz EXACTLY ==, np>0 (Role C live THROUGH the cull), ∀cell_div.
//   G-POISON ⭐ two knobs, both MUST diverge: (1) stencil-too-small (drops a screening-k at ~1.04rc);
//         (2) drop_class>0 (Role C dropped). If neither bites, the gate is blind.
//   G-SORT  reversed/shuffled window skip_sort vs canonical-k cull |delta| (predicted load-bearing).
//   G-W   tight-PBC boundary (n_k=2k+1, the off-by-one) via the oracle at k=4.
//   G-kMaxNbr (cull doesn't change the in-rc count, <64) + G-momentum (slab Σf∈(0,1e-9), diamond exact).
// Compiled with --fmad=false (tdmd_eam_cuda_flags). zone_meam.cuh / zone_cells.cuh BYTE-UNTOUCHED.
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
#include "tdmd/cuda/meam_window_force_gpu.cuh"  // GpuMeamWinForce (cull flag) + the cells kernels
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/gen/partial_screen_slab.hpp"
#include "tdmd/potentials/meam.hpp"

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
  double pe = 0; long long np = 0, nz = 0;
};

// One GPU run over the window. cull=false ⇒ all-window (the in-process bitwise REFERENCE);
// cull=true ⇒ the cells path (canonical-k cull). cell_div/poison_s/drop only matter when cull.
GpuForces gpu_window(const Window& w, const pot::MeamParams& p, bool cull, int cell_div = 0,
                     bool skip_sort = false, int drop = 0, int poison_s = 0) {
  tdcu::GpuMeamWinForce<double> g(p, w.box, cull, cell_div);
  g.skip_sort = skip_sort; g.drop_class = drop; g.poison_s = poison_s;
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
struct OracleF { std::vector<double> fx, fy, fz; double pe = 0; long np = 0, nz = 0; };
OracleF oracle(const Window& w, const pot::MeamParams& p, int drop = 0) {
  core::AtomSoA<double> a; a.resize(w.m);
  for (int i = 0; i < w.m; ++i) { a.x[i] = w.wx[i]; a.y[i] = w.wy[i]; a.z[i] = w.wz[i]; a.type[i] = 1; a.mass[i] = 28.0855; }
  core::zero_forces(a);
  const auto acc = pot::meam_direct_fp64(a, core::PairGeom(w.box, p.rc), p, true, drop);
  OracleF f; f.pe = acc.pe; f.np = acc.n_screened_partial; f.nz = acc.n_screened_zero;
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
// taper-band partial triple (free box) — the dscrfcn `−coef2` radial-taper branch LIVE.
Window taper_win() {
  core::Box box; box.lo = {0, 0, 0}; box.hi = {24, 24, 24}; box.periodic = {false, false, false};
  core::AtomSoA<double> a; a.resize(3);
  const double pos[3][3] = {{10, 10, 10}, {13.95, 10, 10}, {11.975, 12.93, 10}};
  for (int i = 0; i < 3; ++i) { a.x[i] = pos[i][0]; a.y[i] = pos[i][1]; a.z[i] = pos[i][2]; a.type[i] = 1; a.mass[i] = 28.0855; }
  return window_from(a, box);
}
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

// G-A body: cells(cell_div) ≡ all-window, RAW int64 BITWISE, one window. THE canonical-k forcing
// test — without the canonical-k cull, sij reassociates and the raw lanes/forces diverge.
void check_self_equiv(const Window& w, const pot::MeamParams& p, int cell_div) {
  const auto a = gpu_window(w, p, /*cull=*/false);                 // all-window reference
  const auto c = gpu_window(w, p, /*cull=*/true, cell_div);        // cells
  for (int i = 0; i < w.m; ++i) {
    ASSERT_EQ(c.rx[i], a.rx[i]) << "fx raw atom " << i << " cell_div=" << cell_div;
    ASSERT_EQ(c.ry[i], a.ry[i]) << "fy raw atom " << i << " cell_div=" << cell_div;
    ASSERT_EQ(c.rz[i], a.rz[i]) << "fz raw atom " << i << " cell_div=" << cell_div;
  }
  // pe + the screening counts are the sharp tell of a dropped/reclassified screening-k.
  ASSERT_EQ(c.np, a.np) << "n_partial cells vs all-window, cell_div=" << cell_div;
  ASSERT_EQ(c.nz, a.nz) << "n_zero cells vs all-window, cell_div=" << cell_div;
  // pe equality (raw int64 fold — order-free): compare the decoded value bit-for-bit.
  ASSERT_EQ(c.pe, a.pe) << "pe cells vs all-window, cell_div=" << cell_div;
}
}  // namespace

// ============================ G-A — cells ≡ all-window (the canonical-k forcing gate) ===========
TEST(CudaMeamCells, SelfEquivBitwise) {
  pot::MeamParams p;
  for (int k : {1, 2, 3, 4}) {
    check_self_equiv(dia_win(), p, k);
    check_self_equiv(slab_win(false), p, k);
    check_self_equiv(slab_win(true), p, k);
    check_self_equiv(taper_win(), p, k);
  }
}

// ============================ G-B ⭐ — cells vs FP64 oracle (the dropped-k witness) =============
namespace {
void check_vs_oracle(const Window& w, const pot::MeamParams& p, int cell_div, bool expect_partial) {
  const auto c = gpu_window(w, p, /*cull=*/true, cell_div);
  const auto o = oracle(w, p);
  if (expect_partial) EXPECT_GT(o.np, 0) << "no partial-S triples — Role C untested (cull blind)";
  EXPECT_LT(maxdiff(c.fx, o.fx), 1e-9) << "fx cells vs oracle, cell_div=" << cell_div;
  EXPECT_LT(maxdiff(c.fy, o.fy), 1e-9) << "fy cells vs oracle, cell_div=" << cell_div;
  EXPECT_LT(maxdiff(c.fz, o.fz), 1e-9) << "fz cells vs oracle, cell_div=" << cell_div;
  EXPECT_NEAR(c.pe, o.pe, 1e-6) << "pe cells vs oracle, cell_div=" << cell_div;
  EXPECT_EQ(c.np, o.np) << "n_partial cells vs oracle, cell_div=" << cell_div;
  EXPECT_EQ(c.nz, o.nz) << "n_zero cells vs oracle, cell_div=" << cell_div;
}
}  // namespace
TEST(CudaMeamCells, VsOraclePartialSlabFree) {
  pot::MeamParams p;
  for (int k : {1, 2, 3, 4}) check_vs_oracle(slab_win(false), p, k, /*expect_partial=*/true);
}
TEST(CudaMeamCells, VsOraclePartialSlabPbc) {
  pot::MeamParams p;
  for (int k : {1, 2, 3, 4}) check_vs_oracle(slab_win(true), p, k, /*expect_partial=*/true);
}
TEST(CudaMeamCells, VsOracleTaperBand) {
  pot::MeamParams p;
  for (int k : {1, 2, 3, 4}) {
    const auto o = oracle(taper_win(), p);
    EXPECT_EQ(o.np, 1) << "taper triple not partially screened — the coef2 branch is dead";
    check_vs_oracle(taper_win(), p, k, /*expect_partial=*/true);
  }
}

// ============================ G-POISON ⭐ — both knobs MUST diverge ==============================
// Knob 1: stencil-too-small. Cells sized rc/k (k≥3) ⇒ a screening-k at ~1.04·rc sits one cell
// beyond a forced s=1 stencil ⇒ it is dropped ⇒ the force gap blows up AND n_partial collapses.
TEST(CudaMeamCells, PoisonStencilTooSmallHasTeeth) {
  pot::MeamParams p;
  for (int k : {3, 4}) {
    auto w = slab_win(false);
    const auto bad = gpu_window(w, p, /*cull=*/true, /*cell_div=*/k, /*skip_sort=*/false,
                                /*drop=*/0, /*poison_s=*/1);
    const auto o = oracle(w, p);
    EXPECT_GT(maxdiff(bad.fx, o.fx) + maxdiff(bad.fy, o.fy) + maxdiff(bad.fz, o.fz), 1e-2)
        << "poison stencil s=1 (cells rc/" << k << ") did NOT drop a screening-k — G-B blind";
    EXPECT_LT(bad.np, o.np)
        << "poison stencil did not collapse n_partial (k=" << k << "): the dropped-k is invisible";
  }
}
// Knob 2: drop_class>0 (Role C dropped) DIVERGES from the full oracle, ON THE CELLS PATH.
TEST(CudaMeamCells, PoisonDropRoleCHasTeeth) {
  pot::MeamParams p; auto w = slab_win(false);
  const auto bad = gpu_window(w, p, /*cull=*/true, /*cell_div=*/0, /*skip_sort=*/false, /*drop=*/1);
  const auto o = oracle(w, p);
  EXPECT_GT(maxdiff(bad.fx, o.fx) + maxdiff(bad.fy, o.fy) + maxdiff(bad.fz, o.fz), 1e-3)
      << "dropped-Role-C cells did not diverge — G-B blind to a dropped screening-k";
}

// ============================ G-SORT — the canonical-k order verdict (MEASURE-FIRST) ============
// The DESIGN predicted the canonical-k cull would be LOAD-BEARING for MEAM (screening Π_k S /
// Σ_k dscrfcn are FP-order-sensitive). MEASURE it, don't assume — print |delta| of skip_sort vs
// canonical on BOTH the all-window and the cells path, on the DENSEST fixture (diamond: 16
// screening-k per bond). FINDING (recorded in the header): the all-window path is ITSELF order-
// invariant on these Si fixtures (the screening reassociation stays SUB-QUANTUM at Q24.40) ⇒ the
// canonical-k sort is DEFENSIVE here, not load-bearing — the same measure-first verdict as Te3b /
// SW T3b / Te5. The cull is kept regardless: it makes cells bitwise-to-all-window BY CONSTRUCTION
// (independent of whether the reassociation happens to stay sub-quantum), and is robust to a denser
// multi-k partial fixture (deferred measure-first). The hard correctness proof is G-A (cells ≡
// all-window raw int64 ∀ cell_div), which passed on the diamond's 16-screening-k bonds.
TEST(CudaMeamCells, SortVerdictAndOrderInvariance) {
  pot::MeamParams p;
  for (auto& w : {dia_win(), slab_win(false)}) {
    const auto ref = gpu_window(w, p, /*cull=*/false);  // all-window, canonical order

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
    // canonical cull ⇒ the reversed cells window is order-invariant (un-permuted bitwise to ref).
    for (int i = 0; i < w.m; ++i) {
      ASSERT_EQ(sorted.rx[i], ref.rx[rev[i]]) << "cells canonical sort not order-invariant at " << i;
      ASSERT_EQ(sorted.ry[i], ref.ry[rev[i]]);
      ASSERT_EQ(sorted.rz[i], ref.rz[rev[i]]);
    }
  }
}

// ============================ G-W — tight-PBC degeneracy boundary (k=4) =========================
// A tunable-L partial-slab where the realized n_k brushes the n=2k+1 wrap boundary. A screening-k
// at ~1.04·rc must still be reached by the ±k stencil. Verified by the independent FP64 oracle.
TEST(CudaMeamCells, TightPbcBoundaryViaOracle) {
  pot::MeamParams p;
  // a compact PBC partial slab (Lz tuned small but ≥ 2·rc so min-image stays unambiguous); the
  // grid's k=4 sub-rcut binning makes nz brush the wrap boundary. The oracle is the witness.
  for (int k : {3, 4}) {
    auto w = slab_win(true);
    check_vs_oracle(w, p, k, /*expect_partial=*/true);
    // confirm the grid actually wraps at this k (the boundary is exercised, not bypassed)
    const double lo[3] = {w.box.lo[0], w.box.lo[1], w.box.lo[2]};
    const double len[3] = {w.box.len(0), w.box.len(1), w.box.len(2)};
    const bool per[3] = {w.box.periodic[0], w.box.periodic[1], w.box.periodic[2]};
    const auto g = tdcu::make_zone_grid(lo, len, per, p.rc, 1, 0, k);
    EXPECT_TRUE(!g.wrapx || g.nx >= 2 * g.sx + 1) << "k=" << k << " x wrap double-visit";
    EXPECT_TRUE(!g.wrapy || g.ny >= 2 * g.sy + 1) << "k=" << k << " y wrap double-visit";
  }
}

// ============================ G-kMaxNbr — cull preserves the in-rc count (<64) ==================
TEST(CudaMeamCells, NeighbourCapMargin) {
  pot::MeamParams p;
  const int dia = max_nbr(dia_win(), p);
  const int slab = max_nbr(slab_win(false), p);
  fprintf(stderr, "[G-kMaxNbr] realized max in-rc neighbours: diamond=%d slab=%d (cap=%d)\n",
          dia, slab, tdcu::kMeamMaxNbr);
  EXPECT_LT(dia, tdcu::kMeamMaxNbr);
  EXPECT_LT(slab, tdcu::kMeamMaxNbr);
  // and the cells path must not HALT on these (no spurious overflow from the cull).
  EXPECT_NO_THROW(gpu_window(dia_win(), p, /*cull=*/true));
  EXPECT_NO_THROW(gpu_window(slab_win(false), p, /*cull=*/true));
}

// G-OVERFLOW (M5b acceptance M1) — the CELLS path must HALT (never silently OOB) when a per-thread
// buffer overflows. The all-window OverflowHalt (test_cuda_meam.cu) covers only the all-window path;
// the cull GATHERS screening-k into a local buffer (kkey/krik2/krjk2[64]) that the original cull did
// NOT bound-check — a dense fixture (the 80-atom cube, every atom ~mutual-neighbour) overflows it ⇒
// without the guard a silent local OOB, WITH the guard a sticky HALT. memcheck must stay clean.
TEST(CudaMeamCells, OverflowHaltOnDenseCube) {
  pot::MeamParams p;
  Window w; w.box.lo = {-10, -10, -10}; w.box.hi = {10, 10, 10}; w.box.periodic = {false, false, false};
  std::mt19937 rng(1); std::uniform_real_distribution<double> u(0.0, 2.5);  // dense within rc=4
  for (int i = 0; i < 80; ++i) { w.wx.push_back(u(rng)); w.wy.push_back(u(rng)); w.wz.push_back(u(rng)); w.key.push_back(i); w.owned.push_back(i); }
  w.m = 80;
  EXPECT_THROW(gpu_window(w, p, /*cull=*/true), std::runtime_error) << "cells path silently truncated (OOB?)";
}

// ============================ G-momentum — carried forward through the cull ====================
TEST(CudaMeamCells, MomentumFloorPreserved) {
  pot::MeamParams p;
  auto ws = slab_win(false);
  const auto gs = gpu_window(ws, p, /*cull=*/true);
  EXPECT_GT(sumf(gs.fx, gs.fy, gs.fz), 0.0) << "slab cells Σf exactly zero — a symmetric shortcut?";
  EXPECT_LT(sumf(gs.fx, gs.fy, gs.fz), 1e-9) << "slab cells Σf far above the quantum";
  auto wd = dia_win();
  const auto gd = gpu_window(wd, p, /*cull=*/true);
  EXPECT_LT(sumf(gd.fx, gd.fy, gd.fz), 1e-9) << "diamond cells Σf above the quantum (binary-S)";
}
