// MEAM-ladder Me5b — the cell-list culling bake-off + the FIRST angular flagship-scale (10⁶)
// atom-steps/s number. Times the heavy screened MEAM kernels (K1 density + K3 four-role force)
// all-window vs cell-list culled, on a jittered diamond-Si lattice, reporting R_cull = t(all-
// window)/t(cells) per N, the per-kernel split, the realized AUTO cell_div + over-fetch, the
// realized kMaxNbr, and the atom-steps/s at the largest N. ISOLATED-kernel throughput (free-z /
// z=1 / fp64) — NOT the live ring (host-orchestrated, concurrency-dead at z=1), NOT a bitwise-to-
// EAM comparison. K2 (embedding) is O(m) and identical on both paths (timed but not the headline).
//   build: -DTDMD_WITH_CUDA=ON ; run: ./bench_meam_cells [--n 16384] [--reps 8] [--flagship]
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/zone_meam.cuh"          // all-window K1/K2/K3
#include "tdmd/cuda/zone_meam_cells.cuh"    // Me5b: cell-list culled K1/K3 + MeamCellGrid
#include "tdmd/potentials/meam.hpp"

namespace core = tdmd::core;
namespace pot = tdmd::potentials;
namespace tdcu = tdmd::cuda;

namespace {
template <typename T>
T* up(const std::vector<T>& v) {
  T* d = nullptr; cudaMalloc(&d, v.size() * sizeof(T));
  cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); return d;
}

// jittered diamond-Si lattice scaled to ~N atoms (8 atoms/cell; PBC; a0=5.431). Jitter breaks
// the binary-S degeneracy a little but the bench measures THROUGHPUT, not screening correctness
// (that is the test's job on the partial slab).
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
  std::printf("bench_meam_cells: %s, %d SMs, %.1f GiB\n", pr.name, pr.multiProcessorCount,
              pr.totalGlobalMem / 1073741824.0);

  pot::MeamParams p;
  const pot::MeamScreenCParams scp = {p.rc, p.delr, p.ebound, p.Cmin, p.Cmax};
  const auto fp = pot::meam_force_params(p);
  const double dens_scale = core::fixed::ForceAccum::kScale;
  const int blk = tdcu::kZoneBlock;

  // φ-spline upload (MeamParamsView)
  auto upd = [](const std::vector<double>& h) { double* d; cudaMalloc(&d, h.size()*8);
    cudaMemcpy(d, h.data(), h.size()*8, cudaMemcpyHostToDevice); return d; };
  pot::MeamParamsView view{upd(p.phirar), upd(p.phirar1), upd(p.phirar2), upd(p.phirar3),
                           upd(p.phirar4), upd(p.phirar5), upd(p.phirar6), p.nr, p.rdrar};

  std::printf("\n(K1cull/K3cull = cells kernel µs; Rk1/Rk3/Rtot = t(all-window)/t(cells); GiB = device mem)\n");
  std::printf("%8s %5s %7s %9s %9s %7s %7s %7s %6s %11s %6s\n", "N", "k", "ncells", "K1cull/µs",
              "K3cull/µs", "Rk1", "Rk3", "Rtot", "kmax", "a-steps/s", "GiB");

  for (int N : sizes) {
    core::Box box; auto at = make_diamond(box, N);
    const int m = at.n;
    const core::PairGeom geom(box, p.rc);
    std::vector<double> wx(m), wy(m), wz(m); std::vector<long> key(m); std::vector<int> owned(m);
    for (int i = 0; i < m; ++i) { wx[i]=at.x[i]; wy[i]=at.y[i]; wz[i]=at.z[i]; key[i]=i; owned[i]=i; }

    double* dwx = up(wx); double* dwy = up(wy); double* dwz = up(wz);
    long* dkey = up(key); int* downed = up(owned);
    std::vector<long long> z64(std::size_t(tdcu::kDensLanes) * m, 0);
    long long* d_dens = up(z64);
    std::vector<long long> zf(m, 0);
    long long* d_fx = up(zf); long long* d_fy = up(zf); long long* d_fz = up(zf);
    std::vector<long long> z1{0};
    long long* d_pe_pair = up(z1); long long* d_pe_embed = up(z1);
    long long* d_np = up(z1); long long* d_nz = up(z1);
    unsigned long long sent; double v = 1e300; std::memcpy(&sent, &v, 8);
    unsigned long long* d_mr = up(std::vector<unsigned long long>{sent});
    int* d_of = up(std::vector<int>{0});
    std::vector<int> hflag(m, 1);
    int* d_flag = up(hflag);
    pot::MeamEmbedDeriv* d_ed;
    cudaMalloc(&d_ed, std::size_t(m) * sizeof(pot::MeamEmbedDeriv));

    const double box_lo[3] = {box.lo[0], box.lo[1], box.lo[2]};
    const double box_len[3] = {box.len(0), box.len(1), box.len(2)};
    const bool per[3] = {box.periodic[0], box.periodic[1], box.periodic[2]};

    // AUTO cell_div (the same heuristic as GpuMeamWindowState::ensure_grid_geometry) — or override.
    const auto g1 = tdcu::make_zone_grid(box_lo, box_len, per, p.rc, 1, 0, 1);
    const double atoms_per = g1.ncells() > 0 ? double(m) / double(g1.ncells()) : 1.0;
    int cell_div = cellk_override > 0 ? cellk_override : int(std::lround(std::cbrt(atoms_per / 2.5)));
    cell_div = cell_div < 1 ? 1 : (cell_div > 4 ? 4 : cell_div);
    tdcu::MeamCellGrid grid = tdcu::meam_build_window_grid(dwx, dwy, dwz, m, box_lo, box_len, per,
                                                           p.rc, cell_div);

    const int ng = (m + blk - 1) / blk;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    // The all-window MEAM force is O(m²) per thread with an INNER O(m) screening scan ⇒ ~O(m³)
    // total — intractable above ~16k (that IS why no angular flagship exists). Cap the all-window
    // BASELINE; above it report cells-only throughput (the flagship). Few reps + 1 warmup for the
    // slow all-window; the cells path is fast ⇒ full reps + 2 warmups.
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

    // density + embedding need to run before force (force reads d_dens/d_ed). Build them once
    // (all-window for ≤cap; cells otherwise — identical result by G-A) so the force inputs are valid.
    if (do_aw)
      tdcu::meam_density_kernel<<<ng, blk>>>(dwx, dwy, dwz, m, geom, scp, fp, dens_scale, d_dens, d_of);
    else
      tdcu::meam_density_cells_kernel<<<ng, blk>>>(dwx, dwy, dwz, dkey, m, geom, scp, fp, dens_scale,
          grid.g, grid.d_starts, grid.d_counts, grid.d_order, d_dens, d_of);
    tdcu::meam_embed_kernel<<<ng, blk>>>(m, d_flag, fp, p.A, p.Ec, p.ibar, p.gsmooth, p.emb_lin_neg,
                                         p.rho_ref, dens_scale, d_dens, d_ed, d_pe_embed);
    cudaDeviceSynchronize();

    // K1 timings (density).
    const double t_k1_aw = do_aw ? time_aw([&]{
      tdcu::meam_density_kernel<<<ng, blk>>>(dwx, dwy, dwz, m, geom, scp, fp, dens_scale, d_dens, d_of); }) : 0.0;
    const double t_k1_cl = time_cl([&]{
      tdcu::meam_density_cells_kernel<<<ng, blk>>>(dwx, dwy, dwz, dkey, m, geom, scp, fp, dens_scale,
          grid.g, grid.d_starts, grid.d_counts, grid.d_order, d_dens, d_of); });
    // re-seed dens/embed for the force pass (cells K1 == all-window K1 by G-A; either is valid).
    tdcu::meam_density_cells_kernel<<<ng, blk>>>(dwx, dwy, dwz, dkey, m, geom, scp, fp, dens_scale,
        grid.g, grid.d_starts, grid.d_counts, grid.d_order, d_dens, d_of);
    tdcu::meam_embed_kernel<<<ng, blk>>>(m, d_flag, fp, p.A, p.Ec, p.ibar, p.gsmooth, p.emb_lin_neg,
                                         p.rho_ref, dens_scale, d_dens, d_ed, d_pe_embed);
    cudaDeviceSynchronize();

    // K3 timings (force, four-role transpose-replay).
    const double t_k3_aw = do_aw ? time_aw([&]{
      tdcu::meam_force_kernel<<<ng, blk>>>(dwx, dwy, dwz, dkey, m, downed, m, geom, scp, fp, view,
          dens_scale, d_dens, d_ed, d_fx, d_fy, d_fz, d_pe_pair, d_np, d_nz, d_mr, d_of, 0); }) : 0.0;
    const double t_k3_cl = time_cl([&]{
      tdcu::meam_force_cells_kernel<<<ng, blk>>>(dwx, dwy, dwz, dkey, m, downed, m, geom, scp, fp,
          view, dens_scale, d_dens, d_ed, grid.g, grid.d_starts, grid.d_counts, grid.d_order,
          d_fx, d_fy, d_fz, d_pe_pair, d_np, d_nz, d_mr, d_of, 0); });

    const double Rk1 = do_aw ? t_k1_aw / t_k1_cl : 0.0, Rk3 = do_aw ? t_k3_aw / t_k3_cl : 0.0;
    const double t_tot_cl = t_k1_cl + t_k3_cl;
    const double Rtot = do_aw ? (t_k1_aw + t_k3_aw) / t_tot_cl : 0.0;
    const double a_steps = double(m) / (t_tot_cl / 1000.0);  // atom-steps/s (cells, K1+K3)

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
      std::printf("%8d %5d %7d %9.1f %9.1f %7.2f %7.2f %7.2f %6d %11.3e %6.2f%s\n", m, cell_div,
                  grid.ncells, t_k1_cl * 1000, t_k3_cl * 1000, Rk1, Rk3, Rtot, kmax, a_steps,
                  gib_used, of ? " OVERFLOW!" : "");
    else
      std::printf("%8d %5d %7d %9.1f %9.1f %7s %7s %7s %6d %11.3e %6.2f%s\n", m, cell_div,
                  grid.ncells, t_k1_cl * 1000, t_k3_cl * 1000, "—", "—", "(cells)", kmax, a_steps,
                  gib_used, of ? " OVERFLOW!" : "");

    tdcu::meam_cells_free(grid);
    for (void* q : {(void*)dwx,(void*)dwy,(void*)dwz,(void*)dkey,(void*)downed,(void*)d_dens,
                    (void*)d_fx,(void*)d_fy,(void*)d_fz,(void*)d_pe_pair,(void*)d_pe_embed,
                    (void*)d_np,(void*)d_nz,(void*)d_mr,(void*)d_of,(void*)d_flag,(void*)d_ed})
      cudaFree(q);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
  }
  std::printf("\nNOTE: a-steps/s is K1+K3 ISOLATED-kernel throughput (free-z/z=1/fp64), the FIRST\n"
              "angular flagship-scale number — NOT the live ring (concurrency-dead at z=1) and NOT\n"
              "a bitwise EAM comparison. R_cull = t(all-window)/t(cells) per kernel.\n");
  return 0;
}
