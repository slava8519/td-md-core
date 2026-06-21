#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/zone_tersoff.cuh"      // tersoff_force_kernel, kMaxNbr, kZoneBlock
#include "tdmd/cuda/zone_tersoff_cells.cuh"  // Te5b: cell-list culled Tersoff kernel + TersoffCellGrid
#include "tdmd/potentials/many_body.hpp"   // PassDecl/PassKind (the firewall)
#include "tdmd/potentials/tersoff.hpp"     // TersoffParams
#include "tdmd/potentials/tersoff_ring.hpp"  // TersoffPotential, TersoffRing (the policy-ctor)

// M6 / Tersoff-ladder Te5 — GpuTersoffWinForce: the GPU window-force POLICY that drops into
// TersoffRing (the Te3b policy-injected ctor) so GpuTersoffRing = TersoffRing<double,
// GpuTersoffWinForce<double>> inherits ALL the proven host orchestration unchanged. Mirrors
// GpuSwWinForce, with TWO genuine deltas: (1) the [BondOrder, Force] firewall (NOT SW's
// [Force,Force] — a GpuSwWinForce copy would wrongly reject pass 0 == BondOrder); (2) the
// HOST-SIDE CANONICAL ζ-SORT (sort the window by global key so the device kernel sums the FP64
// ζ canonically). MEASURED (the Te3 finding, S2 corrected): the sort is DEFENSIVE — GPU 1-vs-z is
// bitwise WITHOUT it (the block-order window is z-independent) and a single eval is sub-quantum;
// the sort is kept for window-permutation-bitwise-by-construction + canonical alignment with the
// CPU serial. CPU↔GPU is TOLERANCE (exp/pow/sin), GPU-internal is BITWISE — see zone_tersoff.cuh.
namespace tdmd::cuda {
namespace ters_wf_detail {
template <typename T>
inline T* ters_wf_malloc(std::size_t n) {
  void* d = nullptr;
  if (cudaMalloc(&d, n * sizeof(T)) != cudaSuccess)
    throw std::runtime_error("GpuTersoffWinForce: cudaMalloc failed");
  return static_cast<T*>(d);
}
inline int ters_ng(int n) { return (n + kZoneBlock - 1) / kZoneBlock; }
}  // namespace ters_wf_detail

struct GpuTersoffWindowState {
  potentials::TersoffParams tp{};  // by value — NO device tables (no spline)
  int cap_m = 0;
  double *wx = nullptr, *wy = nullptr, *wz = nullptr;
  long* wkey = nullptr;
  int* d_owned = nullptr;
  long long *d_fx = nullptr, *d_fy = nullptr, *d_fz = nullptr;
  long long *d_pe = nullptr, *d_nbonds = nullptr, *d_ntri = nullptr;
  unsigned long long* d_mr = nullptr;
  int* d_of = nullptr;
  unsigned long long sentinel = 0;  // pos_double_bits(1e300) — empty-window seed
  long long last_nbonds = 0, last_ntri = 0;  // TEST-ONLY; the ring SINKS counts (benign WAR race)
  std::mutex mu;

  // Te5b — cell-list culling of the per-window Tersoff force. cells ≡ all-window RAW int64 BITWISE
  // by B1 + the canonical-ζ cull (zone_tersoff_cells.cuh): the int64 force lanes are order-free, and
  // zeta_center_cells key-sorts the stencil ζ-k candidates so the FP64 ζ-sum reassociates in the
  // SAME order the host-sorted all-window zeta_center walks. The box is captured at construction
  // (static membership for the run); compute() only gets a PairGeom, which lacks box.lo the grid
  // needs. cull=false keeps the all-window kernel as the in-process BITWISE REFERENCE.
  double box_lo[3] = {0, 0, 0}, box_len[3] = {0, 0, 0};
  bool periodic[3] = {false, false, false};
  double rcut = 0.0;
  bool cull = false;     // DEFAULT OFF until G-A/G-B/G-POISON green, then AUTO.
  int cell_div = 0;      // 0 = AUTO (target ~2.5 atoms/cell, resolved in ensure_grid_geometry).
                         // 1 = legacy ~rcut cells / ±1. Any k is bitwise == all-window (G-A).
  unsigned long long cells_passes = 0;
  TersoffCellGrid grid_{};
  bool grid_built_ = false;

  explicit GpuTersoffWindowState(const potentials::TersoffParams& t, const core::Box& box,
                                 bool cull_, int cell_div_)
      : tp(t), cull(cull_), cell_div(cell_div_) {
    box_lo[0] = box.lo[0]; box_lo[1] = box.lo[1]; box_lo[2] = box.lo[2];
    box_len[0] = box.len(0); box_len[1] = box.len(1); box_len[2] = box.len(2);
    periodic[0] = box.periodic[0]; periodic[1] = box.periodic[1]; periodic[2] = box.periodic[2];
    rcut = t.rcut();
    double v = 1e300;
    std::memcpy(&sentinel, &v, 8);
    d_pe = ters_wf_detail::ters_wf_malloc<long long>(1);
    d_nbonds = ters_wf_detail::ters_wf_malloc<long long>(1);
    d_ntri = ters_wf_detail::ters_wf_malloc<long long>(1);
    d_mr = ters_wf_detail::ters_wf_malloc<unsigned long long>(1);
    d_of = ters_wf_detail::ters_wf_malloc<int>(1);
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
    wx = ters_wf_detail::ters_wf_malloc<double>(cap_m);
    wy = ters_wf_detail::ters_wf_malloc<double>(cap_m);
    wz = ters_wf_detail::ters_wf_malloc<double>(cap_m);
    wkey = ters_wf_detail::ters_wf_malloc<long>(cap_m);
    d_owned = ters_wf_detail::ters_wf_malloc<int>(cap_m);
    d_fx = ters_wf_detail::ters_wf_malloc<long long>(cap_m);
    d_fy = ters_wf_detail::ters_wf_malloc<long long>(cap_m);
    d_fz = ters_wf_detail::ters_wf_malloc<long long>(cap_m);
    // cell-list per-atom buffers grow with the window (ncells is box-static, allocated once in
    // ensure_grid_geometry). Mirrors GpuSwWindowState::grow.
    if (grid_.d_cell_of) { cudaFree(grid_.d_cell_of); cudaFree(grid_.d_order); }
    grid_.d_cell_of = ters_wf_detail::ters_wf_malloc<int>(cap_m);
    grid_.d_order = ters_wf_detail::ters_wf_malloc<int>(cap_m);
    grid_.m = cap_m;
  }

  // Build the window cell grid GEOMETRY once (box-static). Per-pass only counts/order refresh (in
  // compute). One rc-padded whole-window grid (n_zones=1) — reach is sub-cell ⇒ covers o's
  // neighbours + Roles 3/4's 2·rcut + the ζ-k. AUTO cell_div heuristic COPIED VERBATIM from
  // GpuSwWindowState::ensure_grid_geometry: target ~2.5 atoms/cell, derive k from the realized k=1
  // occupancy m/ncells_k1. Bitwise-safe for ANY k (zone_tersoff_cells.cuh G-A). Clamped [1,4].
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
    grid_.d_counts = ters_wf_detail::ters_wf_malloc<int>(std::size_t(grid_.ncells));
    grid_.d_starts = ters_wf_detail::ters_wf_malloc<int>(std::size_t(grid_.ncells));
    grid_.d_cursor = ters_wf_detail::ters_wf_malloc<int>(std::size_t(grid_.ncells));
    std::size_t cub_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_bytes, grid_.d_counts, grid_.d_starts, grid_.ncells);
    grid_.cub_bytes = cub_bytes;
    grid_.d_cub = ters_wf_detail::ters_wf_malloc<char>(cub_bytes);
    grid_built_ = true;
  }
  ~GpuTersoffWindowState() {
    free_scratch();
    tersoff_cells_free(grid_);
    for (void* p : {(void*)d_pe, (void*)d_nbonds, (void*)d_ntri, (void*)d_mr, (void*)d_of})
      if (p) cudaFree(p);
  }
};
static_assert(std::is_trivially_copyable_v<potentials::TersoffParams>,
              "TersoffParams is passed by value into the kernel; a future non-trivial member "
              "(e.g. a std::vector for multi-element) would silently slice — fail loudly here");

template <typename Real>
struct GpuTersoffWinForce {
  std::shared_ptr<GpuTersoffWindowState> st;
  bool skip_sort = false;  // TEST-ONLY (G-ζORDER): skip the canonical sort to prove it bites
  int poison_s = 0;        // TEST-ONLY (G-POISON): force a too-small cell stencil (G-A teeth)

  GpuTersoffWinForce(const potentials::TersoffParams& tp, const core::Box& box, bool cull = true,
                     int cell_div = 0)  // cull DEFAULT-ON (gates green; AUTO): live-ring cull translates bitwise
      : st(std::make_shared<GpuTersoffWindowState>(tp, box, cull, cell_div)) {
    for (int d = 0; d < 3; ++d)
      if (box.periodic[d] && box.len(d) < 2.0 * tp.rcut())
        throw std::invalid_argument("GpuTersoffWinForce: periodic box dim < 2·rcut — min-image ambiguous");
  }

  // FIREWALL — the INVERSE of GpuEamWindowForce (which THROWS on needs_transpose). Tersoff's
  // Force pass writes to a third atom k (needs_transpose) and the 4-role transpose-replay IS the
  // mechanism ⇒ ACCEPT it. THE genuine delta vs GpuSwWinForce: pass 0 is BondOrder (a copy of
  // GpuSwWinForce::assert_supported demands every pass be Force ⇒ would WRONGLY reject pass 0).
  static void assert_supported(std::span<const potentials::PassDecl> passes) {
    if (passes.size() != 2)
      throw std::runtime_error("GpuTersoffWinForce: Tersoff is the 2-pass [BondOrder ζ, Force] "
          "sequence — got " + std::to_string(passes.size()) + " passes");
    if (passes[0].kind != potentials::PassKind::BondOrder)
      throw std::runtime_error("GpuTersoffWinForce: pass 0 must be BondOrder (a Force here is the "
          "SW [Force,Force] descriptor, not Tersoff)");
    if (passes[1].kind != potentials::PassKind::Force)
      throw std::runtime_error("GpuTersoffWinForce: pass 1 must be Force");
    for (const auto& p : passes)
      if (p.iterative)
        throw std::runtime_error("GpuTersoffWinForce: iterative pass (QEq/CG, ReaxFF) unimplemented");
    if (passes[0].needs_transpose || !passes[1].needs_transpose)
      throw std::runtime_error("GpuTersoffWinForce: expected pass 0 symmetric (BondOrder ζ), "
          "pass 1 needs_transpose (Force) — descriptor does not match the Tersoff force structure");
  }

  void compute(const double* wx, const double* wy, const double* wz, const long* key,
               int m, const int* owned, int n_owned, const core::PairGeom& geom,
               double /*rho_cap — Tersoff has no density*/,
               std::vector<core::fixed::ForceAccum>& wFx,
               std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz,
               core::fixed::EnergyAccum& pe, double& min_r2) const {
    GpuTersoffWindowState& s = *st;
    std::lock_guard<std::mutex> lk(s.mu);
    if (m <= 0) return;
    s.grow(m);

    // CANONICAL ζ-ORDER (DEFENSIVE — mirrors TersoffWinForce::compute): sort the window by global
    // key so the device kernel sums the FP64 ζ canonically. The GPU 1-vs-z is bitwise WITHOUT this
    // (the block-order window is z-independent) and a single eval is sub-quantum (the Te3 finding),
    // so the sort is not strictly load-bearing for the GPU gates — kept for window-permutation-
    // bitwise-by-construction + canonical alignment with the CPU serial. skip_sort = test hook.
    std::vector<int> perm(m);
    std::iota(perm.begin(), perm.end(), 0);
    if (!skip_sort) std::sort(perm.begin(), perm.end(), [&](int a, int b) { return key[a] < key[b]; });
    std::vector<double> sx(m), sy(m), sz(m);
    std::vector<long> sk(m);
    std::vector<int> inv(m);
    for (int t = 0; t < m; ++t) { const int o = perm[t]; sx[t] = wx[o]; sy[t] = wy[o]; sz[t] = wz[o]; sk[t] = key[o]; inv[o] = t; }
    std::vector<int> sowned(n_owned);
    for (int t = 0; t < n_owned; ++t) sowned[t] = inv[owned[t]];

    cudaMemcpy(s.wx, sx.data(), m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wy, sy.data(), m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wz, sz.data(), m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wkey, sk.data(), m * sizeof(long), cudaMemcpyHostToDevice);
    if (n_owned > 0) cudaMemcpy(s.d_owned, sowned.data(), n_owned * sizeof(int), cudaMemcpyHostToDevice);
    const long long z64 = 0; const int z32 = 0;
    cudaMemcpy(s.d_pe, &z64, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_nbonds, &z64, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_ntri, &z64, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_mr, &s.sentinel, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_of, &z32, 4, cudaMemcpyHostToDevice);
    cudaMemsetAsync(s.d_fx, 0, m * sizeof(long long));  // initcheck fix (D2H reads all m)
    cudaMemsetAsync(s.d_fy, 0, m * sizeof(long long));
    cudaMemsetAsync(s.d_fz, 0, m * sizeof(long long));

    // s.cull=false keeps the all-window kernel as the in-process BITWISE REFERENCE. The cull is raw-
    // int64-bitwise to it by B1 + the canonical-ζ cull (zone_tersoff_cells.cuh). The canonical sort
    // above STAYS (it is the reference ζ-k order the cells canonical-ζ cull matches).
    if (s.cull) {
      s.ensure_grid_geometry(m);
      TersoffCellGrid& g = s.grid_;
      cudaMemsetAsync(g.d_counts, 0, std::size_t(g.ncells) * sizeof(int));
      cell_count_kernel<<<ters_wf_detail::ters_ng(m), kZoneBlock>>>(s.wx, s.wy, s.wz, m, g.g,
                                                                    g.d_cell_of, g.d_counts);
      cub::DeviceScan::ExclusiveSum(g.d_cub, g.cub_bytes, g.d_counts, g.d_starts, g.ncells);
      cudaMemcpyAsync(g.d_cursor, g.d_starts, std::size_t(g.ncells) * sizeof(int),
                      cudaMemcpyDeviceToDevice);
      cell_scatter_kernel<<<ters_wf_detail::ters_ng(m), kZoneBlock>>>(g.d_cell_of, m, g.d_cursor,
                                                                      g.d_order);
      if (poison_s > 0) { g.g.sx = g.g.sy = g.g.sz = poison_s; }  // TEST-ONLY (G-POISON teeth)
      if (n_owned > 0)
        tersoff_force_cells_kernel<<<ters_wf_detail::ters_ng(n_owned), kZoneBlock>>>(
            s.wx, s.wy, s.wz, s.wkey, m, s.d_owned, n_owned, geom, s.tp, g.g, g.d_starts, g.d_counts,
            g.d_order, s.d_fx, s.d_fy, s.d_fz, s.d_pe, s.d_nbonds, s.d_ntri, s.d_mr, s.d_of);
      ++s.cells_passes;
    } else {
      tersoff_force_kernel<<<ters_wf_detail::ters_ng(n_owned), kZoneBlock>>>(
          s.wx, s.wy, s.wz, s.wkey, m, s.d_owned, n_owned, geom, s.tp,
          s.d_fx, s.d_fy, s.d_fz, s.d_pe, s.d_nbonds, s.d_ntri, s.d_mr, s.d_of);
    }

    std::vector<long long> hfx(m), hfy(m), hfz(m);
    cudaMemcpy(hfx.data(), s.d_fx, m * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(hfy.data(), s.d_fy, m * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(hfz.data(), s.d_fz, m * sizeof(long long), cudaMemcpyDeviceToHost);
    long long h_pe = 0, h_nb = 0, h_nt = 0;
    cudaMemcpy(&h_pe, s.d_pe, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_nb, s.d_nbonds, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_nt, s.d_ntri, 8, cudaMemcpyDeviceToHost);
    unsigned long long h_mr = 0;
    cudaMemcpy(&h_mr, s.d_mr, 8, cudaMemcpyDeviceToHost);
    int of = 0;
    cudaMemcpy(&of, s.d_of, 4, cudaMemcpyDeviceToHost);
    if (cudaDeviceSynchronize() != cudaSuccess)
      throw std::runtime_error("GpuTersoffWinForce: kernel launch/sync failed");
    if (of) throw std::runtime_error("GpuTersoffWinForce: force/neighbour overflow HALT");

    // un-permute: sorted slot t ↔ original window slot perm[t] (the kernel wrote the sorted layout).
    for (int t = 0; t < m; ++t) { wFx[perm[t]].raw = hfx[t]; wFy[perm[t]].raw = hfy[t]; wFz[perm[t]].raw = hfz[t]; }
    pe.raw += h_pe;
    double gpu_mr2;
    std::memcpy(&gpu_mr2, &h_mr, 8);
    if (gpu_mr2 < min_r2) min_r2 = gpu_mr2;
    s.last_nbonds = h_nb; s.last_ntri = h_nt;  // TEST-ONLY
  }
};

template <typename Real = double>
using GpuTersoffRing = potentials::TersoffRing<Real, GpuTersoffWinForce<Real>>;

}  // namespace tdmd::cuda
