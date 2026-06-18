// E5c device-resident PROBE (measure-first gate, design wf_d75b4a72-21c). The live
// EAM ring serializes z>1 on the null stream under a mutex (SPEEDUP_z=0.99). Before a
// ~1000-line per-node-stream rewrite, this measures the CEILING: do N independent EAM
// kernel chains (density→embedding→force, cell_div=3 = the AUTO default) on N separate
// streams + disjoint scratch beat running them serially? ncu pre-check: kernels are
// latency-bound (SM 38%, FP64 24% abs, 7% occupancy, DRAM ~1%) ⇒ headroom EXISTS; this
// quantifies the realizable fraction. One global sync per leg ⇒ this is the CEILING
// (the live ring's per-node syncs can only be worse).
//
// Pre-registered bands (frozen): SPEEDUP_conc(2,k=3) < 1.20× ⇒ NO-GO (shelve to M5b);
// 1.20–1.5× ⇒ PARTIAL (build #1, STOP if live SPEEDUP_z<1.3×); ≥1.5× ⇒ GO.
//   build: -DTDMD_WITH_CUDA=ON, --fmad=false ; run: ./probe_eam_concurrency [--nc N] [--cellk K]
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/zone_eam.cuh"        // eam_embedding_kernel, EamSetflView, kZoneBlock
#include "tdmd/cuda/zone_eam_cells.cuh"  // eam_density/force_cells_kernel, eam_build_window_grid
#include "tdmd/potentials/eam_spline.hpp"

namespace core = tdmd::core;
namespace potentials = tdmd::potentials;
namespace tdcu = tdmd::cuda;

namespace {
template <typename T> T* up(const std::vector<T>& v) {
  T* d = nullptr; cudaMalloc(&d, v.size() * sizeof(T));
  cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); return d;
}

// one per-stream chain's device state (fully disjoint scratch — mirrors #1 per-node).
struct Chain {
  cudaStream_t stream{};
  double *wx{}, *wy{}, *wz{}, *d_fp{};
  long* wkey{}; int* owned{};
  long long *d_rho{}, *d_fx{}, *d_fy{}, *d_fz{}, *d_pe{};
  unsigned long long* d_mr{};
  int* d_of{};
  tdcu::EamCellGrid grid{};
  int m{};
};

void launch_chain(const Chain& c, const core::PairGeom& geom, const tdcu::EamSetflView& view,
                  double dens_scale, double rho_cap, cudaStream_t s) {
  const int blk = tdcu::kZoneBlock, g = (c.m + blk - 1) / blk;
  tdcu::eam_density_cells_kernel<<<g, blk, 0, s>>>(c.wx, c.wy, c.wz, c.m, geom, view, dens_scale,
      c.grid.g, c.grid.d_starts, c.grid.d_counts, c.grid.d_order, c.d_rho, c.d_of);
  tdcu::eam_embedding_kernel<<<g, blk, 0, s>>>(c.m, view, dens_scale, rho_cap, c.d_rho, c.d_fp, c.d_of);
  tdcu::eam_force_cells_kernel<<<g, blk, 0, s>>>(c.wx, c.wy, c.wz, c.wkey, c.m, c.owned, c.m, geom,
      view, dens_scale, c.d_rho, c.d_fp, c.grid.g, c.grid.d_starts, c.grid.d_counts, c.grid.d_order,
      c.d_fx, c.d_fy, c.d_fz, c.d_pe, c.d_mr, c.d_of);
}
}  // namespace

int main(int argc, char** argv) {
  int nc = 13; int cell_div = 3;  // 13³·4 = 8788 ≈ Axis-B zone window (~8986)
  std::string setfl_path = "reference_data/eam_al/Al_zhou.eam.alloy";
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--nc") nc = std::stoi(argv[++i]);
    else if (a == "--cellk") cell_div = std::stoi(argv[++i]);
    else if (a == "--setfl") setfl_path = argv[++i];
  }
  cudaDeviceProp pr{}; cudaGetDeviceProperties(&pr, 0);
  const auto setfl = potentials::EamSetfl<double>::from_setfl(setfl_path);
  const double a0 = 4.05;
  core::Box box; box.lo = {0,0,0}; box.hi = {nc*a0, nc*a0, nc*a0}; box.periodic = {false,false,false};
  const double b[4][3] = {{0,0,0},{0.5,0.5,0},{0.5,0,0.5},{0,0.5,0.5}};
  std::vector<double> wx, wy, wz; std::vector<long> key; std::vector<int> owned;
  for (int ix=0; ix<nc; ++ix) for (int iy=0; iy<nc; ++iy) for (int iz=0; iz<nc; ++iz) for (auto& bb : b) {
    wx.push_back((ix+bb[0])*a0); wy.push_back((iy+bb[1])*a0); wz.push_back((iz+bb[2])*a0);
  }
  const int m = int(wx.size());
  for (int i=0; i<m; ++i) { key.push_back(i); owned.push_back(i); }

  double* dF = up(setfl.Fspl); double* dra = up(setfl.rhoaspl); double* drp = up(setfl.rphispl);
  tdcu::EamSetflView view{dF, dra, drp, setfl.Nrho, setfl.Nr, setfl.rdrho, setfl.rdr, setfl.rcut};
  const int fb = setfl.density_fracbits();
  const double dens_scale = (fb==44) ? core::fixed::FixedAccum<44>::kScale : core::fixed::FixedAccum<40>::kScale;
  const double rho_cap = setfl.density_grid_max();
  const core::PairGeom geom(box, setfl.rcut);
  const double box_lo[3]={0,0,0}, box_len[3]={nc*a0,nc*a0,nc*a0}; const bool per[3]={false,false,false};

  std::printf("probe_eam_concurrency: %s | window m=%d, cell_div=%d | %d SMs | fp64 --fmad=false\n",
              pr.name, m, cell_div, pr.multiProcessorCount);
  std::printf("  pre-registered: SPEEDUP_conc(2) <1.20 NO-GO | 1.20-1.5 PARTIAL | >=1.5 GO\n");

  std::vector<long long> z64(m, 0); std::vector<double> zdv(m, 0.0);
  auto make_chain = [&]() {
    Chain c; c.m = m;
    cudaStreamCreate(&c.stream);
    c.wx = up(wx); c.wy = up(wy); c.wz = up(wz); c.wkey = up(key); c.owned = up(owned);
    c.d_rho = up(z64); c.d_fp = up(zdv); c.d_fx = up(z64); c.d_fy = up(z64); c.d_fz = up(z64);
    c.d_pe = up(std::vector<long long>{0}); c.d_mr = up(std::vector<unsigned long long>{0x7FF0000000000000ULL});
    c.d_of = up(std::vector<int>{0});
    c.grid = tdcu::eam_build_window_grid(c.wx, c.wy, c.wz, m, box_lo, box_len, per, setfl.rcut, cell_div);
    return c;
  };

  cudaFree(0);  // warmup
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  const int reps = 8;
  for (int nstream : {2, 3}) {
    std::vector<Chain> ch; for (int s=0; s<nstream; ++s) ch.push_back(make_chain());
    // SERIAL: all chains back-to-back on the null stream, one sync.
    float t_serial = 1e30f;
    for (int r=0; r<reps; ++r) {
      cudaEventRecord(e0);
      for (auto& c : ch) launch_chain(c, geom, view, dens_scale, rho_cap, 0);
      cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms=0; cudaEventElapsedTime(&ms, e0, e1); t_serial = std::min(t_serial, ms);
    }
    // CONCURRENT: chain i on stream[i], one device sync.
    float t_conc = 1e30f;
    for (int r=0; r<reps; ++r) {
      cudaEventRecord(e0);
      for (auto& c : ch) launch_chain(c, geom, view, dens_scale, rho_cap, c.stream);
      cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms=0; cudaEventElapsedTime(&ms, e0, e1); t_conc = std::min(t_conc, ms);
    }
    const double sp = t_serial / t_conc;
    const char* verdict = sp < 1.20 ? "NO-GO" : sp < 1.5 ? "PARTIAL" : "GO";
    std::printf("  nstream=%d: serial=%.3f ms  concurrent=%.3f ms  SPEEDUP_conc=%.2f  -> %s\n",
                nstream, t_serial, t_conc, sp, verdict);
    for (auto& c : ch) { cudaStreamDestroy(c.stream); tdcu::eam_cells_free(c.grid); }
  }
  return 0;
}
