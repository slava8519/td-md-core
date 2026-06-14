// M6 PR-E5 — GPU EAM kernels (zone_eam.cuh) BITWISE-EQUAL to CPU eam_window_force
// in deterministic_fp64, over one gathered window. + the device near-knot
// flag-audit (closes the E2-P2 host-only flag-audit finding). Standalone; the
// GPU-ring integration is E5b. Compiled with --fmad=false (tdmd_eam_cuda_flags).
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/zone_eam.cuh"
#include "tdmd/potentials/eam_analytic.hpp"
#include "tdmd/potentials/eam_spline.hpp"
#include "tdmd/potentials/eam_zone.hpp"

using namespace tdmd;

namespace {
// host bit-casts (the device __double_as_longlong / pos_double_bits are device-only)
long long d2ll(double v) { long long b; std::memcpy(&b, &v, 8); return b; }

template <typename T>
T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  EXPECT_EQ(cudaMalloc(&d, v.size() * sizeof(T)), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
  return d;
}

// One gathered window: a small free-z FCC cluster, all atoms owned.
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

// Build a device EamSetflView (caller frees the returned ptrs).
struct DevSetfl { cuda::EamSetflView view; double *F, *ra, *rp; };
DevSetfl upload_setfl(const potentials::EamSetfl<double>& s) {
  DevSetfl d;
  d.F = upload(s.Fspl); d.ra = upload(s.rhoaspl); d.rp = upload(s.rphispl);
  d.view = {d.F, d.ra, d.rp, s.Nrho, s.Nr, s.rdrho, s.rdr, s.rcut};
  return d;
}
void free_setfl(DevSetfl& d) { cudaFree(d.F); cudaFree(d.ra); cudaFree(d.rp); }

// Run the 3 GPU kernels over the window; return raw rho/forces/pe/min_r2/overflow.
struct GpuOut {
  std::vector<long long> rho, fx, fy, fz;
  long long pe = 0; unsigned long long min_r2_bits = 0; int overflow = 0;
};
GpuOut run_gpu(const Window& w, const potentials::EamSetfl<double>& setfl,
               double dens_scale, double rho_cap) {
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
  // device pos_double_bits = __double_as_longlong (raw bits; monotone for r2>=0).
  // Seed with the SAME sentinel the CPU oracle uses (1e300), so an empty window
  // (n_owned=0, E5b) bit-matches instead of diverging on the inf-vs-1e300 init.
  unsigned long long sentinel = static_cast<unsigned long long>(d2ll(1e300));
  unsigned long long* d_mr = upload(std::vector<unsigned long long>{sentinel});
  int* d_of = upload(std::vector<int>{0});
  const int blk = cuda::kZoneBlock;
  cuda::eam_density_kernel<<<(m + blk - 1) / blk, blk>>>(dx, dy, dz, m, geom, ds.view, dens_scale, d_rho, d_of);
  cuda::eam_embedding_kernel<<<(m + blk - 1) / blk, blk>>>(m, ds.view, dens_scale, rho_cap, d_rho, d_fp, d_of);
  cuda::eam_force_kernel<<<(m + blk - 1) / blk, blk>>>(dx, dy, dz, dkey, m, downed, m, geom, ds.view, dens_scale, d_rho, d_fp, d_fx, d_fy, d_fz, d_pe, d_mr, d_of);
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
  for (void* p : {(void*)dx, (void*)dy, (void*)dz, (void*)dkey, (void*)downed, (void*)d_rho,
                  (void*)d_fp, (void*)d_fx, (void*)d_fy, (void*)d_fz, (void*)d_pe, (void*)d_mr, (void*)d_of})
    cudaFree(p);
  free_setfl(ds);
  return o;
}

// CPU golden density raw (mirrors eam_window_force pass1, which doesn't expose ρ).
template <typename DensAccum>
std::vector<long long> cpu_density(const Window& w, const potentials::EamSetfl<double>& s) {
  const core::PairGeom geom(w.box, s.rcut);
  std::vector<long long> rho(w.m);
  for (int aa = 0; aa < w.m; ++aa) {
    DensAccum acc;
    for (int bb = 0; bb < w.m; ++bb) {
      if (bb == aa) continue;
      double dx = w.wx[aa] - w.wx[bb], dy = w.wy[aa] - w.wy[bb], dz = w.wz[aa] - w.wz[bb], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
      double v, dv; s.eval_rhoa(std::sqrt(r2), v, dv); acc.add(v);
    }
    rho[aa] = acc.raw;
  }
  return rho;
}

// Bitwise CPU↔GPU over the window for a given setfl (Q19.44 or Q23.40).
template <int FB>
void check_bitwise(const potentials::EamSetfl<double>& setfl, unsigned seed) {
  const auto w = make_window(seed);
  const core::PairGeom geom(w.box, setfl.rcut);
  const double rho_cap = setfl.density_grid_max();
  const double dens_scale = core::fixed::FixedAccum<FB>::kScale;
  ASSERT_EQ(setfl.density_fracbits(), FB);

  // CPU golden: force/pe via eam_window_force, density via the mirror loop.
  std::vector<core::fixed::ForceAccum> wFx(w.m), wFy(w.m), wFz(w.m);
  core::fixed::EnergyAccum pe; double min_r2 = 1e300;
  potentials::eam_window_force<potentials::EamSetfl<double>, core::fixed::FixedAccum<FB>>(
      w.wx.data(), w.wy.data(), w.wz.data(), w.key.data(), w.m, w.owned.data(), w.m,
      setfl, geom, rho_cap, wFx, wFy, wFz, pe, min_r2);
  const auto rho_gold = cpu_density<core::fixed::FixedAccum<FB>>(w, setfl);

  const auto g = run_gpu(w, setfl, dens_scale, rho_cap);
  EXPECT_EQ(g.overflow, 0);
  for (int i = 0; i < w.m; ++i) {
    EXPECT_EQ(g.rho[i], rho_gold[i]) << "rho raw, atom " << i;
    EXPECT_EQ(g.fx[i], wFx[i].raw) << "fx raw, atom " << i;
    EXPECT_EQ(g.fy[i], wFy[i].raw) << "fy raw, atom " << i;
    EXPECT_EQ(g.fz[i], wFz[i].raw) << "fz raw, atom " << i;
  }
  EXPECT_EQ(g.pe, pe.raw) << "pe raw";
  EXPECT_EQ(g.min_r2_bits, static_cast<unsigned long long>(d2ll(min_r2)));
}

potentials::EamSetfl<double> analytic_setfl(double beta) {
  potentials::AnalyticEam<double> m; m.rcut = 3.0; m.beta = beta; m.finalize();
  return potentials::EamSetfl<double>::from_analytic(m, 4000, 4000, 60.0);
}
}  // namespace

// --- HEADLINE: Q19.44 bitwise CPU↔GPU (density + force + pe + min_r2) ---
TEST(CudaEam, BitwiseQ1944) {
  check_bitwise<44>(analytic_setfl(1.5), 7);
}

// --- DUAL-FORMAT: Q23.40 fallback path bitwise (steep ρa, fracbits==40) ---
TEST(CudaEam, BitwiseQ2340) {
  check_bitwise<40>(analytic_setfl(3.3), 11);
}

// --- HALT symmetry: a near-overlap pair (ρ past the grid AND/OR force past
// Q24.40) ⇒ GPU sets the sticky overflow flag where CPU eam_window_force throws
// (both map to Halt::Internal host-side). ---
TEST(CudaEam, OverflowHaltSymmetry) {
  const auto setfl = analytic_setfl(1.5);
  Window w;  // two atoms at 0.15 Å — ρa(0.15)≈112 > grid 60, force also huge
  w.wx = {0.0, 0.15}; w.wy = {0.0, 0.0}; w.wz = {0.0, 0.0};
  w.key = {0, 1}; w.owned = {0, 1}; w.m = 2;
  w.box.lo = {-10, -10, -10}; w.box.hi = {10, 10, 10}; w.box.periodic = {false, false, false};
  const core::PairGeom geom(w.box, setfl.rcut);
  const double rho_cap = setfl.density_grid_max();
  std::vector<core::fixed::ForceAccum> fx(2), fy(2), fz(2);
  core::fixed::EnergyAccum pe; double mr = 1e300;
  EXPECT_ANY_THROW((potentials::eam_window_force<potentials::EamSetfl<double>, core::fixed::FixedAccum<44>>(
      w.wx.data(), w.wy.data(), w.wz.data(), w.key.data(), 2, w.owned.data(), 2,
      setfl, geom, rho_cap, fx, fy, fz, pe, mr)));  // ρ-cap or quantize overflow
  const auto g = run_gpu(w, setfl, core::fixed::FixedAccum<44>::kScale, rho_cap);
  EXPECT_NE(g.overflow, 0) << "GPU did not flag the over-range pair";
}

// --- on-device near-knot flag-audit: device eval_spline/eval_phi == host,
// bit-for-bit (a forgotten --fmad would fuse the Horner/prefactor and diverge).
__global__ void eval_kernel(cuda::EamSetflView eam, const double* xs, int n,
                            long long* outF, long long* outdF, long long* outphi) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double v, dv; eam.eval_rhoa(xs[i], v, dv);
  outF[i] = __double_as_longlong(v); outdF[i] = __double_as_longlong(dv);
  double p, dp; eam.eval_phi(xs[i], p, dp);
  outphi[i] = __double_as_longlong(p);
}
TEST(CudaEam, NearKnotFlagAudit) {
  const auto setfl = analytic_setfl(1.5);
  // sweep x straddling knot seams (p = x*rdr+1 near integers) in the physical band
  std::vector<double> xs;
  for (int k = 800; k < 1400; ++k) {
    const double rk = k / setfl.rdr;  // exact knot
    xs.push_back(rk); xs.push_back(rk + 1e-9); xs.push_back(rk - 1e-9);
  }
  const int n = int(xs.size());
  DevSetfl ds = upload_setfl(setfl);
  double* dxs = upload(xs);
  std::vector<long long> z(n, 0);
  long long* dF = upload(z); long long* ddF = upload(z); long long* dphi = upload(z);
  eval_kernel<<<(n + 127) / 128, 128>>>(ds.view, dxs, n, dF, ddF, dphi);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<long long> gF(n), gdF(n), gphi(n);
  cudaMemcpy(gF.data(), dF, n * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(gdF.data(), ddF, n * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(gphi.data(), dphi, n * 8, cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; ++i) {
    double v, dv; setfl.eval_rhoa(xs[i], v, dv);
    double p, dp; setfl.eval_phi(xs[i], p, dp);
    EXPECT_EQ(gF[i], d2ll(v)) << "rhoa value, x=" << xs[i];
    EXPECT_EQ(gdF[i], d2ll(dv)) << "rhoa deriv, x=" << xs[i];
    EXPECT_EQ(gphi[i], d2ll(p)) << "phi value, x=" << xs[i];
  }
  for (void* pp : {(void*)dxs, (void*)dF, (void*)ddF, (void*)dphi}) cudaFree(pp);
  free_setfl(ds);

  // discrimination proof: the Horner step IS fuse-sensitive (split != fma) for a
  // representative knot — so a forgotten --fmad WOULD diverge and fail above.
  bool discriminates = false;
  for (int k = 800; k < 1400 && !discriminates; ++k) {
    const double* c = setfl.rhoaspl.data() + 7 * k;
    const double p = 0.37;
    if ((c[0] * p + c[1]) != std::fma(c[0], p, c[1])) discriminates = true;
  }
  EXPECT_TRUE(discriminates) << "near-knot test is not fuse-discriminating";
}
