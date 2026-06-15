// M4-N — GPU EAM ms/step phase-breakdown bench (perf track, "measure-first").
// Times the 3 EAM kernels (density / embedding / force) separately, reports the
// stacked-bar breakdown + hit-rate (real/examined pairs) + occupancy + atom-
// steps/s. This is the DATA that drives the neighbour-backend bake-off
// (all-window O(N²) baseline here; the low hit-rate quantifies the cell-list
// opportunity). deterministic_fp64, the EamSetfl (spline) Math.
//   build: -DTDMD_WITH_CUDA=ON ; run: ./bench_eam --cells 8 --steps 50
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/zone_eam.cuh"
#include "tdmd/metrics/eam_breakdown.hpp"
#include "tdmd/potentials/eam_analytic.hpp"
#include "tdmd/potentials/eam_spline.hpp"

using namespace tdmd;

namespace {
template <typename T>
T* up(const std::vector<T>& v) {
  T* d = nullptr;
  cudaMalloc(&d, v.size() * sizeof(T));
  cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

core::AtomSoA<double> make_fcc(core::Box& box, int nc, double a0) {
  box.lo = {0, 0, 0}; box.hi = {nc * a0, nc * a0, nc * a0};
  box.periodic = {true, true, true};
  const double b[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  core::AtomSoA<double> at;
  std::vector<std::array<double, 3>> p;
  for (int ix = 0; ix < nc; ++ix)
    for (int iy = 0; iy < nc; ++iy)
      for (int iz = 0; iz < nc; ++iz)
        for (auto& bb : b)
          p.push_back({(ix + bb[0]) * a0, (iy + bb[1]) * a0, (iz + bb[2]) * a0});
  at.resize(int(p.size()));
  for (int i = 0; i < at.n; ++i) { at.x[i] = p[i][0]; at.y[i] = p[i][1]; at.z[i] = p[i][2]; }
  return at;
}

// occupancy % for a kernel (theoretical, via the runtime).
template <typename K>
double occ(K kernel, int block) {
  int maxblocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblocks, kernel, block, 0);
  cudaDeviceProp pr{}; cudaGetDeviceProperties(&pr, 0);
  return 100.0 * double(maxblocks * block) / double(pr.maxThreadsPerMultiProcessor);
}
}  // namespace

int main(int argc, char** argv) {
  int nc = 8; long steps = 50; double rcut = 4.0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--cells") nc = std::stoi(argv[++i]);
    else if (a == "--steps") steps = std::stol(argv[++i]);
    else if (a == "--rcut") rcut = std::stod(argv[++i]);
  }
  cudaDeviceProp pr{}; cudaGetDeviceProperties(&pr, 0);
  std::printf("bench_eam: %s, %d SMs\n", pr.name, pr.multiProcessorCount);

  core::Box box;
  auto at = make_fcc(box, nc, 4.05);
  const int m = at.n;
  potentials::AnalyticEam<double> am; am.rcut = rcut; am.finalize();
  const auto setfl = potentials::EamSetfl<double>::from_analytic(am, 4000, 4000, 60.0);
  const int fb = setfl.density_fracbits();
  const double dens_scale = (fb == 44) ? core::fixed::FixedAccum<44>::kScale
                                       : core::fixed::FixedAccum<40>::kScale;
  const double rho_cap = setfl.density_grid_max();
  const core::PairGeom geom(box, setfl.rcut);

  // window = the whole system, all owned (the all-window O(N²) baseline).
  std::vector<double> wx(m), wy(m), wz(m);
  std::vector<long> key(m); std::vector<int> owned(m);
  for (int i = 0; i < m; ++i) { wx[i] = at.x[i]; wy[i] = at.y[i]; wz[i] = at.z[i]; key[i] = i; owned[i] = i; }

  // device setfl view
  double* dF = up(setfl.Fspl); double* dra = up(setfl.rhoaspl); double* drp = up(setfl.rphispl);
  cuda::EamSetflView view{dF, dra, drp, setfl.Nrho, setfl.Nr, setfl.rdrho, setfl.rdr, setfl.rcut};
  double* dx = up(wx); double* dy = up(wy); double* dz = up(wz);
  long* dkey = up(key); int* downed = up(owned);
  std::vector<long long> z64(m, 0); std::vector<double> zd(m, 0.0);
  long long* d_rho = up(z64); double* d_fp = up(zd);
  long long* d_fx = up(z64); long long* d_fy = up(z64); long long* d_fz = up(z64);
  long long* d_pe = up(std::vector<long long>{0});
  unsigned long long* d_mr = up(std::vector<unsigned long long>{0x7FF0000000000000ULL});
  int* d_of = up(std::vector<int>{0});
  const int blk = cuda::kZoneBlock;
  const int gd = (m + blk - 1) / blk;
  auto dens = [&] { cuda::eam_density_kernel<<<gd, blk>>>(dx, dy, dz, m, geom, view, dens_scale, d_rho, d_of); };
  auto emb  = [&] { cuda::eam_embedding_kernel<<<gd, blk>>>(m, view, dens_scale, rho_cap, d_rho, d_fp, d_of); };
  auto frc  = [&] { cuda::eam_force_kernel<<<gd, blk>>>(dx, dy, dz, dkey, m, downed, m, geom, view, dens_scale, d_rho, d_fp, d_fx, d_fy, d_fz, d_pe, d_mr, d_of); };

  cudaEvent_t e0, e1, e2, e3; for (auto* e : {&e0, &e1, &e2, &e3}) cudaEventCreate(e);
  // warmup
  dens(); emb(); frc(); cudaDeviceSynchronize();

  metrics::EamPhaseBreakdown bd; bd.n_atoms = m; bd.steps = steps;
  for (long s = 0; s < steps; ++s) {
    float td = 0, te = 0, tf = 0;
    cudaEventRecord(e0); dens(); cudaEventRecord(e1); emb(); cudaEventRecord(e2); frc(); cudaEventRecord(e3);
    cudaEventSynchronize(e3);
    cudaEventElapsedTime(&td, e0, e1); cudaEventElapsedTime(&te, e1, e2); cudaEventElapsedTime(&tf, e2, e3);
    bd.density_ms += td; bd.embedding_ms += te; bd.force_ms += tf;
  }
  // hit-rate: examined = m*(m-1) (force kernel walks the full window per owned),
  // real = within-cutoff pairs (CPU count over the same config).
  bd.examined_pairs = 1LL * m * (m - 1);
  long long real = 0;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j) {
      if (j == i) continue;
      double ddx = wx[i] - wx[j], ddy = wy[i] - wy[j], ddz = wz[i] - wz[j], r2;
      if (geom.reduce(ddx, ddy, ddz, r2)) ++real;
    }
  bd.real_pairs = real;

  int of = 0; cudaMemcpy(&of, d_of, 4, cudaMemcpyDeviceToHost);
  bd.report("all-window");
  std::printf("occupancy: density %.1f%% | embedding %.1f%% | force %.1f%% | overflow=%d\n",
              occ(cuda::eam_density_kernel, blk), occ(cuda::eam_embedding_kernel, blk),
              occ(cuda::eam_force_kernel, blk), of);
  std::printf("NOTE: all-window O(N^2) baseline. hit-rate %.4f ⇒ a cell-list/Verlet "
              "backend cuts examined pairs ~%.0fx (the M4-B bake-off opportunity).\n",
              bd.hit_rate(), bd.hit_rate() > 0 ? 1.0 / bd.hit_rate() : 0.0);
  return 0;
}
