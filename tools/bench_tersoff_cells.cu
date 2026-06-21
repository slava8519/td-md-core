// Tersoff-ladder Te5b — the cell-list culling bake-off + the Tersoff entry of the angular flagship
// (10⁶) atom-steps/s table. Times the heavy Tersoff force kernel (the 4-role bond-order transpose-
// replay + the ζ k-loops) all-window vs cell-list culled, on a jittered diamond-Si lattice,
// reporting R_cull = t(all-window)/t(cells) per N, the realized AUTO cell_div, the realized kMaxNbr,
// and the atom-steps/s at the largest N. ISOLATED-kernel throughput (free-z / z=1 / fp64) — NOT the
// live ring (host-orchestrated, concurrency-dead at z=1), NOT a bitwise-to-EAM comparison; quote it
// INTERNAL (NOT EAM-comparable). Tersoff is MEAM-LIKE (FP64 ζ-sum ⇒ the canonical-ζ cull makes cells
// == all-window BITWISE — proven by the test, the bench measures throughput only). NB (the MEAM-K3
// caveat): the 4-role transpose-replay may be REGISTER-BOUND (kMaxNbr=64 local nb[] buffer + the
// ζ-candidate buffer), capping achievable occupancy.
//   build: -DTDMD_WITH_CUDA=ON ; run: ./bench_tersoff_cells [--n 16384] [--reps 8] [--flagship] [--cellk K]
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/zone_tersoff.cuh"        // all-window Tersoff force kernel
#include "tdmd/cuda/zone_tersoff_cells.cuh"  // Te5b: cell-list culled Tersoff kernel + TersoffCellGrid
#include "tdmd/potentials/tersoff.hpp"

namespace core = tdmd::core;
namespace pot = tdmd::potentials;
namespace tdcu = tdmd::cuda;

namespace {
template <typename T>
T* up(const std::vector<T>& v) {
  T* d = nullptr; cudaMalloc(&d, v.size() * sizeof(T));
  cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); return d;
}

// jittered diamond-Si lattice scaled to ~N atoms (8 atoms/cell; PBC; a0=5.431). The bench measures
// THROUGHPUT, not correctness (that is the test's job: cells == all-window + the FP64 oracle).
core::AtomSoA<double> make_diamond(core::Box& box, int target_n, double jitter = 0.10) {
  const double a0 = 5.431;
  int nc = std::max(1, int(std::lround(std::cbrt(double(target_n) / 8.0))));
  box.lo = {0, 0, 0}; box.hi = {nc * a0, nc * a0, nc * a0};
  box.periodic = {true, true, true};
  const double b[8][3] = {{0, 0, 0}, {0.25, 0.25, 0.25}, {0.5, 0.5, 0}, {0.75, 0.75, 0.25},
                          {0.5, 0, 0.5}, {0.75, 0.25, 0.75}, {0, 0.5, 0.5}, {0.25, 0.75, 0.75}};
  std::mt19937 rng(7); std::uniform_real_distribution<double> jit(-jitter, jitter);
  std::vector<std::array<double, 3>> p;
  for (int ix = 0; ix < nc; ++ix)
    for (int iy = 0; iy < nc; ++iy)
      for (int iz = 0; iz < nc; ++iz)
        for (auto& bb : b)
          p.push_back({(ix + bb[0]) * a0 + jit(rng), (iy + bb[1]) * a0 + jit(rng),
                       (iz + bb[2]) * a0 + jit(rng)});
  core::AtomSoA<double> at; at.resize(int(p.size()));
  for (int i = 0; i < at.n; ++i) { at.x[i] = p[i][0]; at.y[i] = p[i][1]; at.z[i] = p[i][2];
                                   at.type[i] = 1; at.mass[i] = 28.0855; }
  return at;
}
}  // namespace

int main(int argc, char** argv) {
  std::vector<int> sizes;
  int reps = 8; bool flagship = false; int cellk_override = 0;  // 0 = AUTO
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--n") sizes = {std::stoi(argv[++i])};
    else if (a == "--reps") reps = std::stoi(argv[++i]);
    else if (a == "--cellk") cellk_override = std::stoi(argv[++i]);  // force sub-rcut k (else AUTO)
    else if (a == "--flagship") flagship = true;
  }
  if (sizes.empty()) sizes = {864, 4000, 16384};   // all-window baseline tractable up to ~16k
  if (flagship) sizes = {1000000};

  cudaDeviceProp pr{}; cudaGetDeviceProperties(&pr, 0);
  std::printf("bench_tersoff_cells: %s, %d SMs, %.1f GiB\n", pr.name, pr.multiProcessorCount,
              pr.totalGlobalMem / 1073741824.0);

  pot::TersoffParams sp;
  const int blk = tdcu::kZoneBlock;

  std::printf("\n(Kcull = cells force kernel µs; Rk = t(all-window)/t(cells); GiB = device mem)\n");
  std::printf("%8s %5s %7s %9s %7s %6s %11s %6s\n", "N", "k", "ncells", "Kcull/µs", "Rk",
              "kmax", "a-steps/s", "GiB");

  for (int N : sizes) {
    core::Box box; auto at = make_diamond(box, N);
    const int m = at.n;
    const core::PairGeom geom(box, sp.rcut());
    std::vector<double> wx(m), wy(m), wz(m); std::vector<long> key(m); std::vector<int> owned(m);
    for (int i = 0; i < m; ++i) { wx[i]=at.x[i]; wy[i]=at.y[i]; wz[i]=at.z[i]; key[i]=i; owned[i]=i; }

    double* dwx = up(wx); double* dwy = up(wy); double* dwz = up(wz);
    long* dkey = up(key); int* downed = up(owned);
    std::vector<long long> zf(m, 0);
    long long* d_fx = up(zf); long long* d_fy = up(zf); long long* d_fz = up(zf);
    std::vector<long long> z1{0};
    long long* d_pe = up(z1); long long* d_nb = up(z1); long long* d_ntri = up(z1);
    unsigned long long sent; double v = 1e300; std::memcpy(&sent, &v, 8);
    unsigned long long* d_mr = up(std::vector<unsigned long long>{sent});
    int* d_of = up(std::vector<int>{0});

    const double box_lo[3] = {box.lo[0], box.lo[1], box.lo[2]};
    const double box_len[3] = {box.len(0), box.len(1), box.len(2)};
    const bool per[3] = {box.periodic[0], box.periodic[1], box.periodic[2]};

    // AUTO cell_div (the same heuristic as GpuTersoffWindowState::ensure_grid_geometry) — or override.
    const auto g1 = tdcu::make_zone_grid(box_lo, box_len, per, sp.rcut(), 1, 0, 1);
    const double atoms_per = g1.ncells() > 0 ? double(m) / double(g1.ncells()) : 1.0;
    int cell_div = cellk_override > 0 ? cellk_override : int(std::lround(std::cbrt(atoms_per / 2.5)));
    cell_div = cell_div < 1 ? 1 : (cell_div > 4 ? 4 : cell_div);
    tdcu::TersoffCellGrid grid = tdcu::tersoff_build_window_grid(dwx, dwy, dwz, m, box_lo, box_len,
                                                                 per, sp.rcut(), cell_div);

    const int ng = (m + blk - 1) / blk;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    // The all-window Tersoff force is O(m²) per thread with an INNER O(m) ζ/wing re-scan ⇒ ~O(m³)
    // total — intractable above a few thousand (that IS why no angular flagship exists). Cap the
    // all-window BASELINE; above it report cells-only throughput (the flagship). Few reps + 1 warmup
    // for the slow all-window; the cells path is fast ⇒ full reps + 2 warmups.
    auto time_cl = [&](auto launch) {
      for (int w = 0; w < 2; ++w) launch();
      cudaDeviceSynchronize(); cudaEventRecord(e0);
      for (int r = 0; r < reps; ++r) launch();
      cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms = 0; cudaEventElapsedTime(&ms, e0, e1); return double(ms) / reps;
    };
    auto time_aw = [&](auto launch) {
      const int rr = m <= 4096 ? 4 : 2;
      launch(); cudaDeviceSynchronize(); cudaEventRecord(e0);
      for (int r = 0; r < rr; ++r) launch();
      cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms = 0; cudaEventElapsedTime(&ms, e0, e1); return double(ms) / rr;
    };
    const int aw_cap = 5000;  // above this the all-window baseline is intractable ⇒ cells-only
    const bool do_aw = m <= aw_cap;

    const double t_aw = do_aw ? time_aw([&]{
      tdcu::tersoff_force_kernel<<<ng, blk>>>(dwx, dwy, dwz, dkey, m, downed, m, geom, sp,
          d_fx, d_fy, d_fz, d_pe, d_nb, d_ntri, d_mr, d_of); }) : 0.0;
    const double t_cl = time_cl([&]{
      tdcu::tersoff_force_cells_kernel<<<ng, blk>>>(dwx, dwy, dwz, dkey, m, downed, m, geom, sp,
          grid.g, grid.d_starts, grid.d_counts, grid.d_order, d_fx, d_fy, d_fz, d_pe, d_nb, d_ntri,
          d_mr, d_of); });

    const double Rk = do_aw ? t_aw / t_cl : 0.0;
    const double a_steps = double(m) / (t_cl / 1000.0);  // atom-steps/s (cells, isolated force kernel)

    // realized kMaxNbr (host probe — the cull doesn't change the in-rc count). O(m²) ⇒ SAMPLE a
    // bounded subset of centers for large m (the lattice is uniform; a sample bounds the max).
    int kmax = 0;
    const int probe_n = std::min(m, 2000);
    for (int ii = 0; ii < probe_n; ++ii) {
      const int i = (m / probe_n) * ii;
      int c = 0; for (int j = 0; j < m; ++j) { if (j==i) continue; double dx=wx[j]-wx[i],dy=wy[j]-wy[i],dz=wz[j]-wz[i],r2; if (geom.reduce(dx,dy,dz,r2)) ++c; }
      kmax = std::max(kmax, c);
    }
    int of = 0; cudaMemcpy(&of, d_of, 4, cudaMemcpyDeviceToHost);
    std::size_t mem_free = 0, mem_tot = 0; cudaMemGetInfo(&mem_free, &mem_tot);
    const double gib_used = (mem_tot - mem_free) / 1073741824.0;

    if (do_aw)
      std::printf("%8d %5d %7d %9.1f %7.2f %6d %11.3e %6.2f%s\n", m, cell_div, grid.ncells,
                  t_cl * 1000, Rk, kmax, a_steps, gib_used, of ? " OVERFLOW!" : "");
    else
      std::printf("%8d %5d %7d %9.1f %7s %6d %11.3e %6.2f%s\n", m, cell_div, grid.ncells,
                  t_cl * 1000, "(cells)", kmax, a_steps, gib_used, of ? " OVERFLOW!" : "");

    tdcu::tersoff_cells_free(grid);
    for (void* q : {(void*)dwx,(void*)dwy,(void*)dwz,(void*)dkey,(void*)downed,(void*)d_fx,
                    (void*)d_fy,(void*)d_fz,(void*)d_pe,(void*)d_nb,(void*)d_ntri,(void*)d_mr,(void*)d_of})
      cudaFree(q);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
  }
  std::printf("\nNOTE: a-steps/s is the ISOLATED Tersoff force kernel throughput (free-z/z=1/fp64),\n"
              "the Tersoff entry of the angular flagship table — NOT the live ring (concurrency-dead\n"
              "at z=1) and NOT a bitwise EAM comparison. R_cull = t(all-window)/t(cells). Tersoff is\n"
              "MEAM-LIKE (FP64 ζ-sum) ⇒ cells == all-window BITWISE via the canonical-ζ cull (proven\n"
              "by Test_CUDA_Tersoff_Cells, not the bench). The 4-role replay may be register-bound.\n");
  return 0;
}
