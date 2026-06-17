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
#include "tdmd/cuda/zone_eam_cells.cuh"    // E5c: cell-list culled EAM kernels
#include "tdmd/cuda/zone_eam_verlet.cuh"   // E5c: tight per-atom verlet-list EAM kernels
#include "tdmd/cuda/zone_eam_newton3.cuh"  // E5c: half-list + Newton-3 EAM kernels
#include "tdmd/metrics/eam_breakdown.hpp"
#include "tdmd/potentials/eam_analytic.hpp"
#include "tdmd/potentials/eam_spline.hpp"

// CUB/libcu++ exposes a global ::cuda namespace, and nvcc-generated host stubs
// reference cuda::std unqualified — `using namespace tdmd` would make `cuda`
// ambiguous there (the E5c cells header pulls in the CUB scan). Targeted aliases
// (the test_cuda_zones.cu / test_cuda_conveyor.cu convention).
namespace core = tdmd::core;
namespace potentials = tdmd::potentials;
namespace metrics = tdmd::metrics;
namespace tdcu = tdmd::cuda;

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
  int nc = 8; long steps = 50; double rcut = 4.0; double skin = 1.0;
  std::string backend = "allwindow";  // E5c bake-off: allwindow | cells | verlet | newton3
  std::string setfl_path;  // like-for-like: a real setfl (e.g. Al_zhou.eam.alloy)
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--cells") nc = std::stoi(argv[++i]);
    else if (a == "--steps") steps = std::stol(argv[++i]);
    else if (a == "--rcut") rcut = std::stod(argv[++i]);
    else if (a == "--backend") backend = argv[++i];
    else if (a == "--setfl") setfl_path = argv[++i];
    else if (a == "--skin") skin = std::stod(argv[++i]);
  }
  if (backend != "allwindow" && backend != "cells" && backend != "verlet" &&
      backend != "newton3") {
    std::printf("bench_eam: --backend must be allwindow|cells|verlet|newton3 (got '%s')\n", backend.c_str());
    return 2;
  }
  const bool use_cells = (backend == "cells");
  const bool use_verlet = (backend == "verlet");
  const bool use_newton3 = (backend == "newton3");
  cudaDeviceProp pr{}; cudaGetDeviceProperties(&pr, 0);
  std::printf("bench_eam: %s, %d SMs\n", pr.name, pr.multiProcessorCount);

  core::Box box;
  auto at = make_fcc(box, nc, 4.05);
  const int m = at.n;
  // potential: a real setfl (--setfl, identical pair math to LAMMPS) OR the
  // analytic-tabulated EAM (default). from_setfl carries its own rcut.
  potentials::EamSetfl<double> setfl = [&] {
    if (!setfl_path.empty()) {
      auto s = potentials::EamSetfl<double>::from_setfl(setfl_path);
      rcut = s.rcut;
      std::printf("bench_eam: setfl=%s, rcut=%.4f Å (like-for-like)\n", setfl_path.c_str(), s.rcut);
      return s;
    }
    potentials::AnalyticEam<double> am; am.rcut = rcut; am.finalize();
    return potentials::EamSetfl<double>::from_analytic(am, 4000, 4000, 60.0);
  }();
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
  tdcu::EamSetflView view{dF, dra, drp, setfl.Nrho, setfl.Nr, setfl.rdrho, setfl.rdr, setfl.rcut};
  double* dx = up(wx); double* dy = up(wy); double* dz = up(wz);
  long* dkey = up(key); int* downed = up(owned);
  std::vector<long long> z64(m, 0); std::vector<double> zd(m, 0.0);
  long long* d_rho = up(z64); double* d_fp = up(zd);
  long long* d_fx = up(z64); long long* d_fy = up(z64); long long* d_fz = up(z64);
  long long* d_pe = up(std::vector<long long>{0});
  unsigned long long* d_mr = up(std::vector<unsigned long long>{0x7FF0000000000000ULL});
  int* d_of = up(std::vector<int>{0});
  const int blk = tdcu::kZoneBlock;
  const int gd = (m + blk - 1) / blk;

  // E5c cell grid (over the WHOLE window — dropped-donor-safe extent). Rebuilt
  // every step (the realistic per-step work: after the drift the grid moves).
  const double box_lo[3] = {box.lo[0], box.lo[1], box.lo[2]};
  const double box_len[3] = {box.hi[0] - box.lo[0], box.hi[1] - box.lo[1], box.hi[2] - box.lo[2]};
  const bool per[3] = {box.periodic[0], box.periodic[1], box.periodic[2]};
  tdcu::EamCellGrid cg{};
  tdcu::EamVerletList vl{};
  // The per-step candidate structure: cells rebuild the rcut grid; verlet
  // rebuilds the tight (rcut+skin) CSR list. allwindow has none. In a real engine
  // the verlet list is rebuilt only every K steps (skin amortizes it — LAMMPS's
  // Neigh was ~0% at equilibrium), so we time the build SEPARATELY (one-time) and
  // the iterate-only kernels per step (the steady-state cost).
  auto build_struct = [&] {
    if (use_cells) {
      if (cg.d_order) tdcu::eam_cells_free(cg);
      cg = tdcu::eam_build_window_grid(dx, dy, dz, m, box_lo, box_len, per, setfl.rcut);
    } else if (use_verlet || use_newton3) {  // both walk a per-atom verlet CSR
      if (vl.d_off) tdcu::eam_verlet_free(vl);
      vl = tdcu::eam_build_verlet_list(dx, dy, dz, m, box_lo, box_len, per, setfl.rcut, skin);
    }
  };

  // The Newton-3 kernels atomicAdd onto d_rho/d_fx/d_fy/d_fz, so those must be
  // re-zeroed each step before the density/force passes (the full-neighbour
  // kernels write directly — no zeroing needed, hence this is newton3-only and the
  // memset is part of the steady-state per-step cost, timed inside the kernels' window).
  auto dens = [&] {
    if (use_cells)
      tdcu::eam_density_cells_kernel<<<gd, blk>>>(dx, dy, dz, m, geom, view, dens_scale,
                                                  cg.g, cg.d_starts, cg.d_counts, cg.d_order,
                                                  d_rho, d_of);
    else if (use_verlet)
      tdcu::eam_density_verlet_kernel<<<gd, blk>>>(dx, dy, dz, m, geom, view, dens_scale,
                                                   vl.d_off, vl.d_idx, d_rho, d_of);
    else if (use_newton3) {
      cudaMemsetAsync(d_rho, 0, size_t(m) * sizeof(long long));
      tdcu::eam_density_n3_kernel<<<gd, blk>>>(dx, dy, dz, m, geom, view, dens_scale,
                                               vl.d_off, vl.d_idx, d_rho, d_of);
    } else
      tdcu::eam_density_kernel<<<gd, blk>>>(dx, dy, dz, m, geom, view, dens_scale, d_rho, d_of);
  };
  auto emb  = [&] { tdcu::eam_embedding_kernel<<<gd, blk>>>(m, view, dens_scale, rho_cap, d_rho, d_fp, d_of); };
  auto frc  = [&] {
    if (use_cells)
      tdcu::eam_force_cells_kernel<<<gd, blk>>>(dx, dy, dz, dkey, m, downed, m, geom, view,
                                                dens_scale, d_rho, d_fp, cg.g, cg.d_starts,
                                                cg.d_counts, cg.d_order, d_fx, d_fy, d_fz, d_pe,
                                                d_mr, d_of);
    else if (use_verlet)
      tdcu::eam_force_verlet_kernel<<<gd, blk>>>(dx, dy, dz, dkey, m, downed, m, geom, view,
                                                 dens_scale, d_rho, d_fp, vl.d_off, vl.d_idx,
                                                 d_fx, d_fy, d_fz, d_pe, d_mr, d_of);
    else if (use_newton3) {
      cudaMemsetAsync(d_fx, 0, size_t(m) * sizeof(long long));
      cudaMemsetAsync(d_fy, 0, size_t(m) * sizeof(long long));
      cudaMemsetAsync(d_fz, 0, size_t(m) * sizeof(long long));
      tdcu::eam_embed_energy_n3_kernel<<<gd, blk>>>(m, downed, m, view, dens_scale,
                                                    d_rho, d_pe, d_of);
      tdcu::eam_force_n3_kernel<<<gd, blk>>>(dx, dy, dz, m, geom, view, d_fp,
                                             vl.d_off, vl.d_idx, d_fx, d_fy, d_fz, d_pe,
                                             d_mr, d_of);
    } else
      tdcu::eam_force_kernel<<<gd, blk>>>(dx, dy, dz, dkey, m, downed, m, geom, view, dens_scale, d_rho, d_fp, d_fx, d_fy, d_fz, d_pe, d_mr, d_of);
  };

  cudaEvent_t e0, e1, e2, e3; for (auto* e : {&e0, &e1, &e2, &e3}) cudaEventCreate(e);
  cudaEvent_t b0, b1; cudaEventCreate(&b0); cudaEventCreate(&b1);
  double build_ms_total = 0.0;   // cells: summed over steps; verlet: one-time
  long long list_nnz = 0;        // verlet: total neighbours (avg = nnz/m)

  // warmup + (for verlet) build the tight list ONCE. The realistic steady-state
  // per-step cost is the iterate-only kernels; the list rebuild is amortized over
  // its rebuild interval in a real engine (skin-bounded — LAMMPS Neigh ≈0% at
  // equilibrium), so it's reported as a SEPARATE one-time number, not per step.
  if (use_cells) build_struct();
  if (use_verlet || use_newton3) {
    cudaEventRecord(b0); build_struct(); cudaEventRecord(b1); cudaEventSynchronize(b1);
    float tb = 0; cudaEventElapsedTime(&tb, b0, b1); build_ms_total = tb;
    list_nnz = vl.nnz;
  }
  dens(); emb(); frc(); cudaDeviceSynchronize();

  metrics::EamPhaseBreakdown bd; bd.n_atoms = m; bd.steps = steps;
  for (long s = 0; s < steps; ++s) {
    if (use_cells) {
      cudaEventRecord(b0); build_struct(); cudaEventRecord(b1); cudaEventSynchronize(b1);
      float tb = 0; cudaEventElapsedTime(&tb, b0, b1); build_ms_total += tb;
    }
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
  if (use_cells) tdcu::eam_cells_free(cg);
  if (use_verlet || use_newton3) tdcu::eam_verlet_free(vl);

  // Candidate enumeration cost per atom — what makes the over-fetch visible:
  //   real/atom    = in-cutoff (rcut) neighbours/atom (the irreducible work)
  //   cells exam/atom = the 27-cell (rcut-grid) candidates/atom (the over-fetch)
  //   verlet nbr/atom = the tight (rcut+skin) list size/atom (what we measure)
  const double real_per_atom = double(real) / double(m);
  // cells examined/atom: replicate the 27-cell rcut-grid neighbourhood count on CPU.
  long long cells_exam = 0;
  {
    tdcu::CellGrid g = tdcu::make_zone_grid(box_lo, box_len, per, setfl.rcut, 1, 0);
    std::vector<int> cell_of(m);
    std::vector<std::vector<int>> cells_atoms(g.ncells());
    for (int i = 0; i < m; ++i) {
      int ix, iy, iz; g.coords(wx[i], wy[i], wz[i], ix, iy, iz);
      const int c = g.idx(ix, iy, iz); cell_of[i] = c; cells_atoms[c].push_back(i);
    }
    for (int i = 0; i < m; ++i) {
      int cxi, cyi, czi; g.coords(wx[i], wy[i], wz[i], cxi, cyi, czi);
      const int dzlo = (g.nz == 1) ? 0 : -1, dzhi = (g.nz == 1) ? 0 : 1;
      const int dylo = (g.ny == 1) ? 0 : -1, dyhi = (g.ny == 1) ? 0 : 1;
      const int dxlo = (g.nx == 1) ? 0 : -1, dxhi = (g.nx == 1) ? 0 : 1;
      for (int dz = dzlo; dz <= dzhi; ++dz) {
        int zc = czi + dz; if (g.wrapz) zc = (zc + g.nz) % g.nz; else if (zc < 0 || zc >= g.nz) continue;
        for (int dy = dylo; dy <= dyhi; ++dy) {
          int yc = cyi + dy; if (g.wrapy) yc = (yc + g.ny) % g.ny; else if (yc < 0 || yc >= g.ny) continue;
          for (int dx = dxlo; dx <= dxhi; ++dx) {
            int xc = cxi + dx; if (g.wrapx) xc = (xc + g.nx) % g.nx; else if (xc < 0 || xc >= g.nx) continue;
            cells_exam += (long long)cells_atoms[g.idx(xc, yc, zc)].size();  // includes self in own cell
          }
        }
      }
    }
  }
  const double cells_exam_per_atom = double(cells_exam) / double(m);

  bd.report(backend.c_str());
  if (use_cells)
    std::printf("grid-build: %.4f ms/step (rebuilt over the whole window each step)\n",
                build_ms_total / double(steps));
  if (use_verlet || use_newton3)
    std::printf("list-build: %.4f ms (ONE-TIME — amortized over the rebuild interval; "
                "steady-state per-step cost is the iterate-only kernels above%s)\n",
                build_ms_total,
                use_newton3 ? " + the per-step memset of the int64 atomicAdd targets" : "");
  if (use_cells)
    std::printf("occupancy: density %.1f%% | embedding %.1f%% | force %.1f%% | overflow=%d\n",
                occ(tdcu::eam_density_cells_kernel, blk), occ(tdcu::eam_embedding_kernel, blk),
                occ(tdcu::eam_force_cells_kernel, blk), of);
  else if (use_verlet)
    std::printf("occupancy: density %.1f%% | embedding %.1f%% | force %.1f%% | overflow=%d\n",
                occ(tdcu::eam_density_verlet_kernel, blk), occ(tdcu::eam_embedding_kernel, blk),
                occ(tdcu::eam_force_verlet_kernel, blk), of);
  else if (use_newton3)
    std::printf("occupancy: density %.1f%% | embedding %.1f%% | force %.1f%% | overflow=%d\n",
                occ(tdcu::eam_density_n3_kernel, blk), occ(tdcu::eam_embedding_kernel, blk),
                occ(tdcu::eam_force_n3_kernel, blk), of);
  else
    std::printf("occupancy: density %.1f%% | embedding %.1f%% | force %.1f%% | overflow=%d\n",
                occ(tdcu::eam_density_kernel, blk), occ(tdcu::eam_embedding_kernel, blk),
                occ(tdcu::eam_force_kernel, blk), of);

  // The over-fetch table: real (irreducible) vs cells-27-cell vs verlet-tight.
  std::printf("candidate enumeration: real/atom=%.1f | cells exam/atom=%.1f (%.2fx over-fetch)",
              real_per_atom, cells_exam_per_atom,
              real_per_atom > 0 ? cells_exam_per_atom / real_per_atom : 0.0);
  if (use_verlet || use_newton3)
    std::printf(" | %s nbr/atom=%.1f (skin=%.2f, %.2fx over-fetch, %lld total%s)",
                use_newton3 ? "n3-full-list" : "verlet",
                double(list_nnz) / double(m), skin,
                real_per_atom > 0 ? (double(list_nnz) / double(m)) / real_per_atom : 0.0,
                list_nnz,
                use_newton3 ? "; the bb>aa gate halves the EVAL to ~half this" : "");
  std::printf("\n");

  std::printf(
      "NOTE: backend=%s. all-window hit-rate %.4f ⇒ predicted cull = 1/hit-rate = %.1fx\n"
      "      on the all-pairs candidate set. Run allwindow vs cells vs verlet at the same\n"
      "      --cells to read the REALIZED density/force speedup; the tight verlet list\n"
      "      shows how much of the gap is the cells 27-cell over-fetch (~8x).\n",
      backend.c_str(), bd.hit_rate(), bd.hit_rate() > 0 ? 1.0 / bd.hit_rate() : 0.0);
  return 0;
}
