// M6 E5c — CELL-LIST culled EAM GPU kernels (zone_eam_cells.cuh).
//   Test A (bitwise self-equiv): the culled density/force kernels produce raw
//     int64 ρ / fx / fy / fz / pe / overflow bit-for-bit equal to the all-window
//     (O(m²)) kernels on the SAME gathered window. min_r2: only the Overlap-HALT
//     boolean is compared (min-over-examined differs by candidate set). fb=44
//     AND fb=40 (steep β=3.3 setfl).
//   Test B (the dropped-donor gate, BLOCKING): the culled density+force are
//     cross-checked against eam_direct_fp64 (all-pairs FP64, no grid, no
//     quantize) on the SAME window/config. Per-atom ρ and force agree to ~1e-12,
//     PE matches. This is the ONLY witness that the in-cutoff candidate set is
//     COMPLETE — a grid that drops a donor truncates ρ deterministically and is
//     invisible to A (cells-vs-itself).
// Compiled with --fmad=false (tdmd_eam_cuda_flags). zone_eam.cuh /
// zone_cells.cuh / the E5b ring are BYTE-UNTOUCHED.
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/zone_eam.cuh"
#include "tdmd/cuda/zone_eam_cells.cuh"
#include "tdmd/cuda/zone_eam_verlet.cuh"    // E5c tight-list (verlet) EAM kernels
#include "tdmd/potentials/eam.hpp"          // eam_direct_fp64 (Test B oracle)
#include "tdmd/potentials/eam_analytic.hpp"
#include "tdmd/potentials/eam_spline.hpp"
#include "tdmd/potentials/eam_zone.hpp"

// CUB/libcu++ exposes a global ::cuda namespace, and nvcc-generated host stubs
// reference cuda::std unqualified — `using namespace tdmd` would make `cuda`
// ambiguous there (the EAM cells header includes the CUB scan). Targeted
// aliases instead (the test_cuda_zones.cu / test_cuda_conveyor.cu convention).
namespace core = tdmd::core;
namespace potentials = tdmd::potentials;
namespace tdcu = tdmd::cuda;

namespace {
long long d2ll(double v) { long long b; std::memcpy(&b, &v, 8); return b; }

template <typename T>
T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  EXPECT_EQ(cudaMalloc(&d, v.size() * sizeof(T)), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
  return d;
}

// One gathered window: a jittered free-z FCC cluster, all atoms owned. Same
// shape as test_cuda_eam.cu's make_window, so the candidate set is the heavy
// (~150-neighbour-bounded) case the bake-off cares about.
struct Window {
  std::vector<double> wx, wy, wz;
  std::vector<long> key;
  std::vector<int> owned;
  core::Box box;
  int m = 0;
};
Window make_window(unsigned seed) {
  Window w;
  const double a0 = 4.05;
  const double b[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> jit(-0.06, 0.06);
  for (int ix = 0; ix < 3; ++ix)
    for (int iy = 0; iy < 3; ++iy)
      for (int iz = 0; iz < 3; ++iz)
        for (auto& bb : b) {
          w.wx.push_back((ix + bb[0]) * a0 + jit(rng));
          w.wy.push_back((iy + bb[1]) * a0 + jit(rng));
          w.wz.push_back((iz + bb[2]) * a0 + jit(rng));
        }
  w.m = int(w.wx.size());
  w.key.resize(w.m);
  w.owned.resize(w.m);
  for (int i = 0; i < w.m; ++i) { w.key[i] = i; w.owned[i] = i; }
  w.box.lo = {-10, -10, -10}; w.box.hi = {25, 25, 25}; w.box.periodic = {false, false, false};
  return w;
}

// PERIODIC window (exact lattice period, positions wrapped into [0,L)) — exercises
// the grid wrapx/y/z fold + min-image under the dropped-donor oracle (the free-z
// make_window never takes the wrap branch; a donor dropped ACROSS the periodic
// seam would be invisible to cells-vs-self but caught by eam_direct_fp64's min-image).
Window make_window_pbc(unsigned seed) {
  Window w;
  const double a0 = 4.05, L = 3 * a0;  // 3×3×3 cells ⇒ exact period; L=12.15 ≥ 2·rcut
  const double b[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> jit(-0.06, 0.06);
  auto wrap = [&](double x) { return x - L * std::floor(x / L); };
  for (int ix = 0; ix < 3; ++ix)
    for (int iy = 0; iy < 3; ++iy)
      for (int iz = 0; iz < 3; ++iz)
        for (auto& bb : b) {
          w.wx.push_back(wrap((ix + bb[0]) * a0 + jit(rng)));
          w.wy.push_back(wrap((iy + bb[1]) * a0 + jit(rng)));
          w.wz.push_back(wrap((iz + bb[2]) * a0 + jit(rng)));
        }
  w.m = int(w.wx.size());
  w.key.resize(w.m); w.owned.resize(w.m);
  for (int i = 0; i < w.m; ++i) { w.key[i] = i; w.owned[i] = i; }
  w.box.lo = {0, 0, 0}; w.box.hi = {L, L, L}; w.box.periodic = {true, true, true};
  return w;
}

struct DevSetfl { tdcu::EamSetflView view; double *F, *ra, *rp; };
DevSetfl upload_setfl(const potentials::EamSetfl<double>& s) {
  DevSetfl d;
  d.F = upload(s.Fspl); d.ra = upload(s.rhoaspl); d.rp = upload(s.rphispl);
  d.view = {d.F, d.ra, d.rp, s.Nrho, s.Nr, s.rdrho, s.rdr, s.rcut};
  return d;
}
void free_setfl(DevSetfl& d) { cudaFree(d.F); cudaFree(d.ra); cudaFree(d.rp); }

// Raw outputs of a 3-pass EAM run over the window.
struct GpuOut {
  std::vector<long long> rho, fx, fy, fz;
  long long pe = 0; unsigned long long min_r2_bits = 0; int overflow = 0;
};

// Backend selector: all-window O(m²) kernels (zone_eam.cuh) vs cell-list culled
// kernels (zone_eam_cells.cuh). The density/embedding/force order is identical;
// only the candidate generation of pass 1 + pass 3 differs.
enum class Backend { AllWindow, Cells, Verlet };

GpuOut run_gpu(const Window& w, const potentials::EamSetfl<double>& setfl,
               double dens_scale, double rho_cap, Backend be, double skin = 1.0) {
  const int m = w.m;
  DevSetfl ds = upload_setfl(setfl);
  double* dx = upload(w.wx); double* dy = upload(w.wy); double* dz = upload(w.wz);
  long* dkey = upload(w.key); int* downed = upload(w.owned);
  const core::PairGeom geom(w.box, setfl.rcut);
  std::vector<long long> z64(m, 0);
  long long* d_rho = upload(z64);
  std::vector<double> z(m, 0.0); double* d_fp = upload(z);
  long long* d_fx = upload(z64); long long* d_fy = upload(z64); long long* d_fz = upload(z64);
  long long* d_pe = upload(std::vector<long long>{0});
  unsigned long long sentinel = static_cast<unsigned long long>(d2ll(1e300));
  unsigned long long* d_mr = upload(std::vector<unsigned long long>{sentinel});
  int* d_of = upload(std::vector<int>{0});
  const int blk = tdcu::kZoneBlock;
  const int gd = (m + blk - 1) / blk;

  const double box_lo[3] = {w.box.lo[0], w.box.lo[1], w.box.lo[2]};
  const double box_len[3] = {w.box.hi[0] - w.box.lo[0], w.box.hi[1] - w.box.lo[1],
                             w.box.hi[2] - w.box.lo[2]};
  const bool per[3] = {w.box.periodic[0], w.box.periodic[1], w.box.periodic[2]};
  tdcu::EamCellGrid cg;
  tdcu::EamVerletList vl;
  if (be == Backend::Cells)
    cg = tdcu::eam_build_window_grid(dx, dy, dz, m, box_lo, box_len, per, setfl.rcut);
  else if (be == Backend::Verlet)
    vl = tdcu::eam_build_verlet_list(dx, dy, dz, m, box_lo, box_len, per, setfl.rcut, skin);

  if (be == Backend::AllWindow) {
    tdcu::eam_density_kernel<<<gd, blk>>>(dx, dy, dz, m, geom, ds.view, dens_scale, d_rho, d_of);
  } else if (be == Backend::Cells) {
    tdcu::eam_density_cells_kernel<<<gd, blk>>>(dx, dy, dz, m, geom, ds.view, dens_scale,
                                                cg.g, cg.d_starts, cg.d_counts, cg.d_order,
                                                d_rho, d_of);
  } else {
    tdcu::eam_density_verlet_kernel<<<gd, blk>>>(dx, dy, dz, m, geom, ds.view, dens_scale,
                                                 vl.d_off, vl.d_idx, d_rho, d_of);
  }
  tdcu::eam_embedding_kernel<<<gd, blk>>>(m, ds.view, dens_scale, rho_cap, d_rho, d_fp, d_of);
  if (be == Backend::AllWindow) {
    tdcu::eam_force_kernel<<<gd, blk>>>(dx, dy, dz, dkey, m, downed, m, geom, ds.view,
                                        dens_scale, d_rho, d_fp, d_fx, d_fy, d_fz, d_pe, d_mr, d_of);
  } else if (be == Backend::Cells) {
    tdcu::eam_force_cells_kernel<<<gd, blk>>>(dx, dy, dz, dkey, m, downed, m, geom, ds.view,
                                              dens_scale, d_rho, d_fp, cg.g, cg.d_starts,
                                              cg.d_counts, cg.d_order, d_fx, d_fy, d_fz, d_pe,
                                              d_mr, d_of);
  } else {
    tdcu::eam_force_verlet_kernel<<<gd, blk>>>(dx, dy, dz, dkey, m, downed, m, geom, ds.view,
                                               dens_scale, d_rho, d_fp, vl.d_off, vl.d_idx,
                                               d_fx, d_fy, d_fz, d_pe, d_mr, d_of);
  }
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  GpuOut o;
  o.rho.resize(m); o.fx.resize(m); o.fy.resize(m); o.fz.resize(m);
  cudaMemcpy(o.rho.data(), d_rho, m * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(o.fx.data(), d_fx, m * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(o.fy.data(), d_fy, m * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(o.fz.data(), d_fz, m * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(&o.pe, d_pe, 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(&o.min_r2_bits, d_mr, 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(&o.overflow, d_of, 4, cudaMemcpyDeviceToHost);
  if (be == Backend::Cells) tdcu::eam_cells_free(cg);
  if (be == Backend::Verlet) tdcu::eam_verlet_free(vl);
  for (void* p : {(void*)dx, (void*)dy, (void*)dz, (void*)dkey, (void*)downed, (void*)d_rho,
                  (void*)d_fp, (void*)d_fx, (void*)d_fy, (void*)d_fz, (void*)d_pe, (void*)d_mr, (void*)d_of})
    cudaFree(p);
  free_setfl(ds);
  return o;
}

potentials::EamSetfl<double> analytic_setfl(double beta) {
  potentials::AnalyticEam<double> m; m.rcut = 3.0; m.beta = beta; m.finalize();
  return potentials::EamSetfl<double>::from_analytic(m, 4000, 4000, 60.0);
}

// raw int64 -> physical (for Test B comparison vs the FP64 oracle)
double force_of(long long raw) { return double(raw) / core::fixed::ForceAccum::kScale; }

// Test A / A' body: a culled backend (Cells or Verlet) ≡ all-window, bit-for-bit,
// for one fb. pbc selects the free-z or the periodic-seam window. skin only feeds
// the Verlet build (ignored by Cells).
template <int FB>
void check_self_equiv(double beta, unsigned seed, Backend culled = Backend::Cells,
                      bool pbc = false, double skin = 1.0) {
  const auto setfl = analytic_setfl(beta);
  ASSERT_EQ(setfl.density_fracbits(), FB);
  const double dens_scale = core::fixed::FixedAccum<FB>::kScale;
  const double rho_cap = setfl.density_grid_max();
  const auto w = pbc ? make_window_pbc(seed) : make_window(seed);

  const auto a = run_gpu(w, setfl, dens_scale, rho_cap, Backend::AllWindow);
  const auto c = run_gpu(w, setfl, dens_scale, rho_cap, culled, skin);

  EXPECT_EQ(a.overflow, 0);
  EXPECT_EQ(c.overflow, a.overflow) << "overflow flags must match";
  for (int i = 0; i < w.m; ++i) {
    EXPECT_EQ(c.rho[i], a.rho[i]) << "rho raw, atom " << i;
    EXPECT_EQ(c.fx[i], a.fx[i]) << "fx raw, atom " << i;
    EXPECT_EQ(c.fy[i], a.fy[i]) << "fy raw, atom " << i;
    EXPECT_EQ(c.fz[i], a.fz[i]) << "fz raw, atom " << i;
  }
  EXPECT_EQ(c.pe, a.pe) << "pe raw";
  // min_r2: only the Overlap-HALT boolean is comparable (min over EXAMINED
  // candidates differs by candidate set — never compare bits). Here no pair is
  // below any overlap threshold (jittered lattice), so both are >0 and finite.
  double a_mr2, c_mr2;
  unsigned long long ab = a.min_r2_bits, cb = c.min_r2_bits;
  std::memcpy(&a_mr2, &ab, 8); std::memcpy(&c_mr2, &cb, 8);
  EXPECT_GT(a_mr2, 0.0);
  EXPECT_GT(c_mr2, 0.0);
}
}  // namespace

// ============================ Test A ====================================

TEST(CudaEamCells, SelfEquivQ1944) {  // bitwise cells ≡ all-window, fb=44
  check_self_equiv<44>(1.5, 7);
}
TEST(CudaEamCells, SelfEquivQ2340) {  // steep β=3.3 (fracbits==40)
  check_self_equiv<40>(3.3, 11);
}

// ============================ Test B ====================================
// THE blocking dropped-donor gate: the culled density+force vs eam_direct_fp64
// (all-pairs FP64, no grid, no quantize). If the window grid dropped any donor,
// ρ would be truncated on that atom ⇒ its F'(ρ) wrong ⇒ neighbour forces wrong;
// the FP64 oracle (which walks ALL pairs) is the only witness of completeness.

namespace {
void check_vs_oracle(double beta, unsigned seed, int fb, bool pbc = false,
                     Backend culled = Backend::Cells, double skin = 1.0) {
  const auto setfl = analytic_setfl(beta);
  ASSERT_EQ(setfl.density_fracbits(), fb);
  const double dens_scale = (fb == 44) ? core::fixed::FixedAccum<44>::kScale
                                       : core::fixed::FixedAccum<40>::kScale;
  const double rho_cap = setfl.density_grid_max();
  const auto w = pbc ? make_window_pbc(seed) : make_window(seed);

  // FP64 oracle: all-pairs density + force, no grid, no quantize. The SAME
  // EamSetfl spline math feeds it (Math = EamSetfl<double>).
  core::AtomSoA<double> at;
  at.resize(w.m);
  for (int i = 0; i < w.m; ++i) { at.x[i] = w.wx[i]; at.y[i] = w.wy[i]; at.z[i] = w.wz[i]; }
  core::zero_forces(at);
  std::vector<double> rho_oracle;
  const auto acc = potentials::eam_direct_fp64<double, potentials::EamSetfl<double>>(
      at, w.box, setfl, /*with_forces=*/true, &rho_oracle);

  // culled GPU run
  const auto c = run_gpu(w, setfl, dens_scale, rho_cap, culled, skin);
  EXPECT_EQ(c.overflow, 0);

  // per-atom density: the fixed-point ρ vs the FP64 oracle ρ (~quantization +
  // spline-eval agreement). A DROPPED DONOR shows here as a too-small ρ.
  for (int i = 0; i < w.m; ++i) {
    const double rho_gpu = double(c.rho[i]) / dens_scale;
    EXPECT_NEAR(rho_gpu, rho_oracle[i], 1e-9)
        << "rho atom " << i << " (cells=" << rho_gpu << " oracle=" << rho_oracle[i]
        << ") — a gap this large means a DROPPED DENSITY DONOR";
  }
  // per-atom force: the culled force vs the oracle (a truncated ρ ⇒ wrong F'(ρ)
  // ⇒ wrong embedding force here even if the φ-pair part matched).
  for (int i = 0; i < w.m; ++i) {
    EXPECT_NEAR(force_of(c.fx[i]), at.fx[i], 1e-9) << "fx atom " << i;
    EXPECT_NEAR(force_of(c.fy[i]), at.fy[i], 1e-9) << "fy atom " << i;
    EXPECT_NEAR(force_of(c.fz[i]), at.fz[i], 1e-9) << "fz atom " << i;
  }
  // total PE (Σ F(ρ) + Σ_{i<j} φ): the oracle's pe vs the culled fixed-point pe.
  const double pe_gpu = double(c.pe) / core::fixed::EnergyAccum::kScale;
  EXPECT_NEAR(pe_gpu, acc.pe, 1e-7) << "total PE (cells vs FP64 oracle)";
}
}  // namespace

TEST(CudaEamCells, VsOracleQ1944) {  // dropped-donor gate, fb=44
  check_vs_oracle(1.5, 7, 44);
}
TEST(CudaEamCells, VsOracleQ2340) {  // dropped-donor gate, steep β=3.3
  check_vs_oracle(3.3, 11, 40);
}
TEST(CudaEamCells, VsOraclePbc) {  // dropped-donor gate across the PERIODIC seam
  check_vs_oracle(1.5, 13, 44, /*pbc=*/true);  // exercises grid wrap + min-image
}

// ========================== Test A' (verlet) ============================
// The TIGHT per-atom verlet list ≡ all-window, bit-for-bit (same B1 argument as
// cells: the list within rcut+skin is a SUPERSET of the in-cutoff set; the exact
// r2<rcut² re-test keeps the in-cutoff multiset identical ⇒ raw int64 bit-equal).
// fb=44 + fb=40 (steep β=3.3), free-z + a PBC variant. skin=1.0.

TEST(CudaEamVerlet, SelfEquivQ1944) {  // bitwise verlet ≡ all-window, fb=44, free-z
  check_self_equiv<44>(1.5, 7, Backend::Verlet, /*pbc=*/false, /*skin=*/1.0);
}
TEST(CudaEamVerlet, SelfEquivQ2340) {  // steep β=3.3 (fracbits==40)
  check_self_equiv<40>(3.3, 11, Backend::Verlet, /*pbc=*/false, /*skin=*/1.0);
}
TEST(CudaEamVerlet, SelfEquivPbc) {  // verlet ≡ all-window across the PERIODIC seam
  check_self_equiv<44>(1.5, 13, Backend::Verlet, /*pbc=*/true, /*skin=*/1.0);
}

// ========================== Test B' (verlet) ============================
// THE blocking dropped-donor gate for the tight list: verlet density+force vs
// eam_direct_fp64 (all-pairs FP64, no grid, no quantize). A list that drops a ρ
// donor (e.g. a build grid too small for rcut+skin, or a half list) truncates ρ
// deterministically — invisible to A' (verlet-vs-itself); only the FP64 oracle,
// which walks ALL pairs, witnesses completeness. skin=1.0.

TEST(CudaEamVerlet, VsOracleQ1944) {  // dropped-donor gate, fb=44, free-z
  check_vs_oracle(1.5, 7, 44, /*pbc=*/false, Backend::Verlet, /*skin=*/1.0);
}
TEST(CudaEamVerlet, VsOracleQ2340) {  // dropped-donor gate, steep β=3.3
  check_vs_oracle(3.3, 11, 40, /*pbc=*/false, Backend::Verlet, /*skin=*/1.0);
}
TEST(CudaEamVerlet, VsOraclePbc) {  // dropped-donor gate across the PERIODIC seam
  check_vs_oracle(1.5, 13, 44, /*pbc=*/true, Backend::Verlet, /*skin=*/1.0);
}
