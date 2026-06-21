#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"     // Box
#include "tdmd/core/zones.hpp"   // PairGeom
#include "tdmd/cuda/zone_sw.cuh"  // sw_force_kernel, kMaxNbr, kZoneBlock
#include "tdmd/cuda/zone_sw_cells.cuh"  // T5b: cell-list culled SW kernel + SwCellGrid
#include "tdmd/potentials/many_body.hpp"  // PassDecl/PassKind (the firewall)
#include "tdmd/potentials/sw.hpp"         // SwParams

// M6 / SW-ladder T5 — GpuSwWinForce: the GPU window-force POLICY that drops into SwRing
// (sw_ring.hpp's policy-injected ctor) so GpuSwRing = SwRing<double, GpuSwWinForce<double>>
// inherits ALL the proven host orchestration unchanged. Mirrors GpuEamWindowForce, STRIPPED
// (SW has no spline upload, no density/embedding pre-pass, no dens_scale/rho_cap, single
// Q24.40 format). Given ONE already-gathered 3-zone window it launches sw_force_kernel and
// writes the int64 accumulators decoded IDENTICALLY to the CPU SwWinForce. CPU↔GPU is
// TOLERANCE (exp/pow), GPU-internal is BITWISE — see the zone_sw.cuh contract banner.
namespace tdmd::cuda {
namespace sw_wf_detail {
template <typename T>
inline T* sw_wf_malloc(std::size_t n) {
  void* d = nullptr;
  if (cudaMalloc(&d, n * sizeof(T)) != cudaSuccess)
    throw std::runtime_error("GpuSwWinForce: cudaMalloc failed");
  return static_cast<T*>(d);
}
inline int sw_ng(int n) { return (n + kZoneBlock - 1) / kZoneBlock; }
}  // namespace sw_wf_detail

struct GpuSwWindowState {
  potentials::SwParams sp{};  // by value — NO device tables (no spline)
  int cap_m = 0;
  double *wx = nullptr, *wy = nullptr, *wz = nullptr;
  long* wkey = nullptr;
  int* d_owned = nullptr;
  long long *d_fx = nullptr, *d_fy = nullptr, *d_fz = nullptr, *d_pe = nullptr, *d_ntri = nullptr;
  unsigned long long* d_mr = nullptr;
  int* d_of = nullptr;
  unsigned long long sentinel = 0;  // pos_double_bits(1e300) — empty-window seed (CPU 1e300)
  long long last_ntri = 0;  // last window's triplet count — TEST-ONLY (G4); the ring SINKS
                            // n_triplets, so a write-without-read race across z is benign.
  std::mutex mu;

  // T5b — cell-list culling of the per-window SW force. cells ≡ all-window RAW int64 BITWISE by
  // B1 — SW is int64-ORDER-FREE (no screening, single triplet eval ⇒ NO canonical sort, the
  // EAM-like proof, UNLIKE MEAM's canonical-k cull). The box is captured at construction (static
  // membership for the run); compute() only gets a PairGeom, which lacks box.lo the grid needs.
  // cull=false keeps the all-window kernel as the in-process BITWISE REFERENCE.
  double box_lo[3] = {0, 0, 0}, box_len[3] = {0, 0, 0};
  bool periodic[3] = {false, false, false};
  double rcut = 0.0;
  bool cull = false;     // DEFAULT OFF until G-A/G-B/G-POISON green, then AUTO.
  int cell_div = 0;      // 0 = AUTO (target ~2.5 atoms/cell, resolved in ensure_grid_geometry).
                         // 1 = legacy ~rcut cells / ±1. Any k is bitwise == all-window (G-A).
  unsigned long long cells_passes = 0;
  SwCellGrid grid_{};
  bool grid_built_ = false;

  explicit GpuSwWindowState(const potentials::SwParams& s, const core::Box& box, bool cull_,
                            int cell_div_)
      : sp(s), cull(cull_), cell_div(cell_div_) {
    box_lo[0] = box.lo[0]; box_lo[1] = box.lo[1]; box_lo[2] = box.lo[2];
    box_len[0] = box.len(0); box_len[1] = box.len(1); box_len[2] = box.len(2);
    periodic[0] = box.periodic[0]; periodic[1] = box.periodic[1]; periodic[2] = box.periodic[2];
    rcut = s.rcut();
    double v = 1e300;
    std::memcpy(&sentinel, &v, 8);
    d_pe = sw_wf_detail::sw_wf_malloc<long long>(1);
    d_ntri = sw_wf_detail::sw_wf_malloc<long long>(1);
    d_mr = sw_wf_detail::sw_wf_malloc<unsigned long long>(1);
    d_of = sw_wf_detail::sw_wf_malloc<int>(1);
    grow(64);
  }
  void free_scratch() {
    for (void* p : {(void*)wx, (void*)wy, (void*)wz, (void*)wkey, (void*)d_owned,
                    (void*)d_fx, (void*)d_fy, (void*)d_fz})
      if (p) cudaFree(p);
    wx = wy = wz = nullptr; wkey = nullptr; d_owned = nullptr; d_fx = d_fy = d_fz = nullptr;
  }
  void grow(int m) {
    if (m <= cap_m) return;
    free_scratch();
    cap_m = m;
    wx = sw_wf_detail::sw_wf_malloc<double>(cap_m);
    wy = sw_wf_detail::sw_wf_malloc<double>(cap_m);
    wz = sw_wf_detail::sw_wf_malloc<double>(cap_m);
    wkey = sw_wf_detail::sw_wf_malloc<long>(cap_m);
    d_owned = sw_wf_detail::sw_wf_malloc<int>(cap_m);
    d_fx = sw_wf_detail::sw_wf_malloc<long long>(cap_m);
    d_fy = sw_wf_detail::sw_wf_malloc<long long>(cap_m);
    d_fz = sw_wf_detail::sw_wf_malloc<long long>(cap_m);
    // cell-list per-atom buffers grow with the window (ncells is box-static, allocated once in
    // ensure_grid_geometry). Mirrors GpuMeamWindowState::grow.
    if (grid_.d_cell_of) { cudaFree(grid_.d_cell_of); cudaFree(grid_.d_order); }
    grid_.d_cell_of = sw_wf_detail::sw_wf_malloc<int>(cap_m);
    grid_.d_order = sw_wf_detail::sw_wf_malloc<int>(cap_m);
    grid_.m = cap_m;
  }

  // Build the window cell grid GEOMETRY once (box-static). Per-pass only counts/order refresh (in
  // compute). One rc-padded whole-window grid (n_zones=1) — reach is sub-cell ⇒ covers o's
  // neighbours + the wing role's 2·rcut. AUTO cell_div heuristic COPIED VERBATIM from
  // GpuMeamWindowState::ensure_grid_geometry: target ~2.5 atoms/cell, derive k from the realized
  // k=1 occupancy m/ncells_k1. Bitwise-safe for ANY k (zone_sw_cells.cuh G-A). Clamped [1,4].
  void ensure_grid_geometry(int m_hint) {
    if (grid_built_) return;
    if (cell_div <= 0) {
      const auto g1 = make_zone_grid(box_lo, box_len, periodic, rcut, 1, 0, 1);
      const int nc1 = g1.ncells();
      const double atoms_per = nc1 > 0 ? double(m_hint) / double(nc1) : 1.0;
      int k = int(std::lround(std::cbrt(atoms_per / 2.5)));
      cell_div = std::clamp(k, 1, 4);
    }
    grid_.g = make_zone_grid(box_lo, box_len, periodic, rcut, /*n_zones=*/1, /*zone_id=*/0, cell_div);
    grid_.ncells = grid_.g.ncells();
    grid_.d_counts = sw_wf_detail::sw_wf_malloc<int>(std::size_t(grid_.ncells));
    grid_.d_starts = sw_wf_detail::sw_wf_malloc<int>(std::size_t(grid_.ncells));
    grid_.d_cursor = sw_wf_detail::sw_wf_malloc<int>(std::size_t(grid_.ncells));
    std::size_t cub_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_bytes, grid_.d_counts, grid_.d_starts, grid_.ncells);
    grid_.cub_bytes = cub_bytes;
    grid_.d_cub = sw_wf_detail::sw_wf_malloc<char>(cub_bytes);
    grid_built_ = true;
  }
  ~GpuSwWindowState() {
    free_scratch();
    sw_cells_free(grid_);
    for (void* p : {(void*)d_pe, (void*)d_ntri, (void*)d_mr, (void*)d_of})
      if (p) cudaFree(p);
  }
};

template <typename Real>
struct GpuSwWinForce {
  std::shared_ptr<GpuSwWindowState> st;
  int poison_s = 0;  // TEST-ONLY (G-POISON): force a too-small cell stencil (G-A teeth)

  GpuSwWinForce(const potentials::SwParams& sp, const core::Box& box, bool cull = false,
                int cell_div = 0)
      : st(std::make_shared<GpuSwWindowState>(sp, box, cull, cell_div)) {
    // SW two-wing min-image guard (host; mirrors SwRing ctor + sw_zone_pass): every
    // periodic box dim > 2·rcut, else a wing bond could wrap to a wrong image (the
    // device kernel trusts geom.reduce picked the unique image).
    for (int d = 0; d < 3; ++d)
      if (box.periodic[d] && box.len(d) < 2.0 * sp.rcut())
        throw std::invalid_argument("GpuSwWinForce: periodic box dim < 2·rcut — min-image ambiguous");
  }

  // FIREWALL — the INVERSE of GpuEamWindowForce::assert_supported (which THROWS on
  // needs_transpose). SW's φ₃ IS needs_transpose and THIS kernel implements the
  // transpose-replay ⇒ ACCEPT it; reject count≠2 / non-Force / iterative / the
  // symmetric-only-φ₃ trap. Copies SwWinForce::assert_supported (sw_ring.hpp) so the GPU
  // SW seam asserts its own descriptor from day one (closes the FIREWALL GAP for SW).
  static void assert_supported(std::span<const potentials::PassDecl> passes) {
    if (passes.size() != 2)
      throw std::runtime_error("GpuSwWinForce: SW is the 2-pass [φ2 Force, φ3 Force] "
          "sequence — got " + std::to_string(passes.size()) + " passes");
    for (const auto& p : passes) {
      if (p.kind != potentials::PassKind::Force)
        throw std::runtime_error("GpuSwWinForce: only PassKind::Force is implemented");
      if (p.iterative)
        throw std::runtime_error("GpuSwWinForce: iterative pass unimplemented");
    }
    if (passes[0].needs_transpose || !passes[1].needs_transpose)
      throw std::runtime_error("GpuSwWinForce: expected pass 0 symmetric (φ2), pass 1 "
          "needs_transpose (φ3) — descriptor does not match the SW force structure");
  }

  void compute(const double* wx, const double* wy, const double* wz, const long* key,
               int m, const int* owned, int n_owned, const core::PairGeom& geom,
               double /*rho_cap — SW has no density*/,
               std::vector<core::fixed::ForceAccum>& wFx,
               std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz,
               core::fixed::EnergyAccum& pe, double& min_r2) const {
    GpuSwWindowState& s = *st;
    std::lock_guard<std::mutex> lk(s.mu);  // serialize device work (correctness, like EAM)
    if (m <= 0) return;
    s.grow(m);

    cudaMemcpy(s.wx, wx, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wy, wy, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wz, wz, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wkey, key, m * sizeof(long), cudaMemcpyHostToDevice);
    if (n_owned > 0)
      cudaMemcpy(s.d_owned, owned, n_owned * sizeof(int), cudaMemcpyHostToDevice);
    const long long z64 = 0;
    const int z32 = 0;
    cudaMemcpy(s.d_pe, &z64, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_ntri, &z64, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_mr, &s.sentinel, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_of, &z32, 4, cudaMemcpyHostToDevice);
    // zero the force buffers before launch (the kernel writes only the n_owned owned
    // slots; the D2H below copies all m ⇒ without this the non-owned slots are read
    // uninitialized — benign, never consumed, but it trips initcheck. Acceptance fix).
    cudaMemsetAsync(s.d_fx, 0, m * sizeof(long long));
    cudaMemsetAsync(s.d_fy, 0, m * sizeof(long long));
    cudaMemsetAsync(s.d_fz, 0, m * sizeof(long long));

    // s.cull=false keeps the all-window kernel as the in-process BITWISE REFERENCE. The cull is
    // raw-int64-bitwise to it by B1 (SW is int64-order-free — NO canonical sort, the EAM-like
    // proof). NB: the all-window path needs NO window sort (unlike MEAM) — the gather order is
    // irrelevant to the int64 force/PE multiset.
    if (s.cull) {
      s.ensure_grid_geometry(m);
      SwCellGrid& g = s.grid_;
      cudaMemsetAsync(g.d_counts, 0, std::size_t(g.ncells) * sizeof(int));
      cell_count_kernel<<<sw_wf_detail::sw_ng(m), kZoneBlock>>>(s.wx, s.wy, s.wz, m, g.g,
                                                               g.d_cell_of, g.d_counts);
      cub::DeviceScan::ExclusiveSum(g.d_cub, g.cub_bytes, g.d_counts, g.d_starts, g.ncells);
      cudaMemcpyAsync(g.d_cursor, g.d_starts, std::size_t(g.ncells) * sizeof(int),
                      cudaMemcpyDeviceToDevice);
      cell_scatter_kernel<<<sw_wf_detail::sw_ng(m), kZoneBlock>>>(g.d_cell_of, m, g.d_cursor,
                                                                 g.d_order);
      if (poison_s > 0) { g.g.sx = g.g.sy = g.g.sz = poison_s; }  // TEST-ONLY (G-POISON teeth)
      if (n_owned > 0)
        sw_force_cells_kernel<<<sw_wf_detail::sw_ng(n_owned), kZoneBlock>>>(
            s.wx, s.wy, s.wz, s.wkey, m, s.d_owned, n_owned, geom, s.sp, g.g, g.d_starts,
            g.d_counts, g.d_order, s.d_fx, s.d_fy, s.d_fz, s.d_pe, s.d_ntri, s.d_mr, s.d_of);
      ++s.cells_passes;
    } else {
      sw_force_kernel<<<sw_wf_detail::sw_ng(n_owned), kZoneBlock>>>(
          s.wx, s.wy, s.wz, s.wkey, m, s.d_owned, n_owned, geom, s.sp,
          s.d_fx, s.d_fy, s.d_fz, s.d_pe, s.d_ntri, s.d_mr, s.d_of);
    }

    std::vector<long long> hfx(m), hfy(m), hfz(m);
    cudaMemcpy(hfx.data(), s.d_fx, m * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(hfy.data(), s.d_fy, m * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(hfz.data(), s.d_fz, m * sizeof(long long), cudaMemcpyDeviceToHost);
    long long h_pe = 0, h_ntri = 0;  // h_ntri sunk (the MB2 count lives on the serial path)
    cudaMemcpy(&h_pe, s.d_pe, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_ntri, s.d_ntri, 8, cudaMemcpyDeviceToHost);
    unsigned long long h_mr = 0;
    cudaMemcpy(&h_mr, s.d_mr, 8, cudaMemcpyDeviceToHost);
    int of = 0;
    cudaMemcpy(&of, s.d_of, 4, cudaMemcpyDeviceToHost);
    if (cudaDeviceSynchronize() != cudaSuccess)
      throw std::runtime_error("GpuSwWinForce: kernel launch/sync failed");
    if (of) throw std::runtime_error("GpuSwWinForce: force/neighbour overflow HALT");

    for (int o = 0; o < n_owned; ++o) {
      const int loc = owned[o];
      wFx[loc].raw = hfx[loc]; wFy[loc].raw = hfy[loc]; wFz[loc].raw = hfz[loc];
    }
    pe.raw += h_pe;  // associative int64 add ⇒ == CPU multiset (up to the transcendental)
    double gpu_mr2;
    std::memcpy(&gpu_mr2, &h_mr, 8);
    if (gpu_mr2 < min_r2) min_r2 = gpu_mr2;
    s.last_ntri = h_ntri;  // TEST-ONLY (the ring never reads this)
  }
};

}  // namespace tdmd::cuda
