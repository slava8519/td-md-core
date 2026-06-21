#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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
#include "tdmd/cuda/zone_meam.cuh"          // K1/K2/K3, kDensLanes, kMeamMaxNbr, kZoneBlock
#include "tdmd/cuda/zone_meam_cells.cuh"    // Me5b: cell-list culled K1/K3 + MeamCellGrid + canonical-k cull
#include "tdmd/potentials/many_body.hpp"    // PassDecl/PassKind (the firewall)
#include "tdmd/potentials/meam.hpp"         // MeamParams, MeamParamsView, the POD param builders
#include "tdmd/potentials/meam_ring.hpp"    // MeamPotential, MeamRing (the policy-ctor), MeamWinForce

// MEAM-ladder Me5 — GpuMeamWinForce: the GPU window-force POLICY that drops into MeamRing (the
// Me3b policy-injected ctor) so GpuMeamRing = MeamRing<double, GpuMeamWinForce<double>> inherits
// ALL the proven host orchestration unchanged (FSM, Λ-chain, defer_head, PBC cyclic window). It
// mirrors GpuEamWindowForce (spline upload + growable scratch + mutex) fused with
// GpuTersoffWinForce (the host canonical sort + un-permute + the inverted [Density,Embedding,
// Force(needs_transpose)] firewall). CONTRACT: CPU↔GPU is TOLERANCE (exp/log/pow ~1 ulp), GPU-
// INTERNAL is BITWISE by B1 int64 — see zone_meam.cuh.
namespace tdmd::cuda {
namespace meam_wf_detail {
template <typename T>
inline T* meam_wf_malloc(std::size_t n) {
  void* d = nullptr;
  if (cudaMalloc(&d, n * sizeof(T)) != cudaSuccess)
    throw std::runtime_error("GpuMeamWinForce: cudaMalloc failed");
  return static_cast<T*>(d);
}
inline int meam_ng(int n) { return (n + kZoneBlock - 1) / kZoneBlock; }
}  // namespace meam_wf_detail

// Shared device state: the φ-spline view (uploaded once) + the POD scalar params + growable
// per-call scratch (27-lane density, MeamEmbedDeriv, force) + the serialization mutex. Held by
// shared_ptr so the policy is cheap to copy (MeamRing stores it by value) yet all copies share
// one allocation. dens_scale = ForceAccum::kScale (Q24.40 — matches the CPU FixedDens lanes).
struct GpuMeamWindowState {
  potentials::MeamScreenCParams scp{};
  potentials::MeamForceParams fp{};
  potentials::MeamParamsView view{};
  // embedding scalars
  double A = 0, Ec = 0, gsmooth = 0, rho_ref = 0;
  int ibar = 0, emb_lin_neg = 0;
  double dens_scale = core::fixed::ForceAccum::kScale;  // Q24.40, == CPU FixedDens

  // device φ-spline arrays (uploaded once)
  double *d_phirar = nullptr, *d_phirar1 = nullptr, *d_phirar2 = nullptr, *d_phirar3 = nullptr;
  double *d_phirar4 = nullptr, *d_phirar5 = nullptr, *d_phirar6 = nullptr;

  // growable scratch (sized to the largest window m seen so far)
  int cap_m = 0;
  double *wx = nullptr, *wy = nullptr, *wz = nullptr;
  long* wkey = nullptr;
  int *d_owned = nullptr, *d_owned_flag = nullptr;
  long long* d_dens = nullptr;             // [kDensLanes * cap_m]
  potentials::MeamEmbedDeriv* d_ed = nullptr;  // [cap_m]
  long long *d_fx = nullptr, *d_fy = nullptr, *d_fz = nullptr;

  // per-call scalars
  long long *d_pe_embed = nullptr, *d_pe_pair = nullptr, *d_npartial = nullptr, *d_nzero = nullptr;
  unsigned long long* d_mr = nullptr;
  int* d_of = nullptr;
  unsigned long long sentinel = 0;  // pos_double_bits(1e300) — empty-window seed (the CPU min_r2 seed)
  long long last_npartial = 0, last_nzero = 0;  // TEST-ONLY (the ring SINKS counts)
  std::mutex mu;

  // Me5b — cell-list culling of the per-window screened density (K1) + force (K3). cells ≡
  // all-window RAW int64 BITWISE by B1 + the canonical-k cull (zone_meam_cells.cuh): the 27-lane
  // density lanes are order-free int64, and meam_getscreen_d_cells_device key-sorts the stencil
  // k-candidates so the FP screening product/sum reassociate in the SAME order the host-sorted
  // all-window helper walks. The box is captured at construction (static membership for the run);
  // compute() only gets a PairGeom, which lacks box.lo the grid needs. cull=false keeps the
  // all-window K1/K3 as the in-process BITWISE REFERENCE.
  double box_lo[3] = {0, 0, 0}, box_len[3] = {0, 0, 0};
  bool periodic[3] = {false, false, false};
  double rcut = 0.0;
  bool cull = false;     // DEFAULT OFF until G-B/G-POISON green on the partial fixtures, then AUTO.
  int cell_div = 0;      // 0 = AUTO (target ~2.5 atoms/cell, resolved in ensure_grid_geometry).
                         // 1 = legacy ~rcut cells / ±1. Any k is bitwise == all-window (G-A).
  unsigned long long cells_passes = 0;
  MeamCellGrid grid_{};
  bool grid_built_ = false;

  GpuMeamWindowState(const potentials::MeamParams& p, const core::Box& box, bool cull_, int cell_div_)
      : cull(cull_), cell_div(cell_div_) {
    scp = {p.rc, p.delr, p.ebound, p.Cmin, p.Cmax};
    fp = potentials::meam_force_params(p);
    A = p.A; Ec = p.Ec; gsmooth = p.gsmooth; rho_ref = p.rho_ref;
    ibar = p.ibar; emb_lin_neg = p.emb_lin_neg;
    box_lo[0] = box.lo[0]; box_lo[1] = box.lo[1]; box_lo[2] = box.lo[2];
    box_len[0] = box.len(0); box_len[1] = box.len(1); box_len[2] = box.len(2);
    periodic[0] = box.periodic[0]; periodic[1] = box.periodic[1]; periodic[2] = box.periodic[2];
    rcut = p.rc;
    upload_spline(p);
    double v = 1e300;
    std::memcpy(&sentinel, &v, 8);
    d_pe_embed = meam_wf_detail::meam_wf_malloc<long long>(1);
    d_pe_pair = meam_wf_detail::meam_wf_malloc<long long>(1);
    d_npartial = meam_wf_detail::meam_wf_malloc<long long>(1);
    d_nzero = meam_wf_detail::meam_wf_malloc<long long>(1);
    d_mr = meam_wf_detail::meam_wf_malloc<unsigned long long>(1);
    d_of = meam_wf_detail::meam_wf_malloc<int>(1);
    grow(64);
  }

  void upload_spline(const potentials::MeamParams& p) {
    const int nr = p.nr;
    auto up = [&](const std::vector<double>& h) {
      double* d = meam_wf_detail::meam_wf_malloc<double>(h.size());
      cudaMemcpy(d, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice);
      return d;
    };
    d_phirar = up(p.phirar);  d_phirar1 = up(p.phirar1); d_phirar2 = up(p.phirar2);
    d_phirar3 = up(p.phirar3); d_phirar4 = up(p.phirar4); d_phirar5 = up(p.phirar5);
    d_phirar6 = up(p.phirar6);
    view = potentials::MeamParamsView{d_phirar, d_phirar1, d_phirar2, d_phirar3, d_phirar4,
                                      d_phirar5, d_phirar6, nr, p.rdrar};
  }

  void free_scratch() {
    for (void* p : {(void*)wx, (void*)wy, (void*)wz, (void*)wkey, (void*)d_owned,
                    (void*)d_owned_flag, (void*)d_dens, (void*)d_ed, (void*)d_fx, (void*)d_fy,
                    (void*)d_fz})
      if (p) cudaFree(p);
    wx = wy = wz = nullptr; wkey = nullptr; d_owned = d_owned_flag = nullptr;
    d_dens = d_fx = d_fy = d_fz = nullptr; d_ed = nullptr;
  }
  void grow(int m) {
    if (m <= cap_m) return;
    free_scratch();
    cap_m = m;
    wx = meam_wf_detail::meam_wf_malloc<double>(cap_m);
    wy = meam_wf_detail::meam_wf_malloc<double>(cap_m);
    wz = meam_wf_detail::meam_wf_malloc<double>(cap_m);
    wkey = meam_wf_detail::meam_wf_malloc<long>(cap_m);
    d_owned = meam_wf_detail::meam_wf_malloc<int>(cap_m);
    d_owned_flag = meam_wf_detail::meam_wf_malloc<int>(cap_m);
    d_dens = meam_wf_detail::meam_wf_malloc<long long>(std::size_t(kDensLanes) * cap_m);
    d_ed = meam_wf_detail::meam_wf_malloc<potentials::MeamEmbedDeriv>(cap_m);
    d_fx = meam_wf_detail::meam_wf_malloc<long long>(cap_m);
    d_fy = meam_wf_detail::meam_wf_malloc<long long>(cap_m);
    d_fz = meam_wf_detail::meam_wf_malloc<long long>(cap_m);
    // cell-list per-atom buffers grow with the window (the cell COUNT — ncells — is box-static,
    // allocated once in ensure_grid_geometry).
    if (grid_.d_cell_of) { cudaFree(grid_.d_cell_of); cudaFree(grid_.d_order); }
    grid_.d_cell_of = meam_wf_detail::meam_wf_malloc<int>(cap_m);
    grid_.d_order = meam_wf_detail::meam_wf_malloc<int>(cap_m);
    grid_.m = cap_m;
  }

  // Build the window cell grid GEOMETRY once (box-static). Per-pass only the counts/order are
  // refreshed (in compute). One rc-padded whole-window grid (n_zones=1) — reach is sub-cell
  // (√ebound·rc < 2·rc) ⇒ this covers density donors + force neighbours + screening-k. The AUTO
  // cell_div heuristic is COPIED VERBATIM from GpuEamWindowState::ensure_grid_geometry: target
  // ~2.5 atoms/cell, derive k from the realized k=1 occupancy m/ncells_k1. MEAM-Si rc=4.0 is far
  // denser per cell than EAM's Al_zhou rc=10.1 — the realized k is MEASURED by the bench, not
  // assumed. Bitwise-safe for ANY k (zone_meam_cells.cuh G-A). Clamped [1,4].
  void ensure_grid_geometry(int m_hint) {
    if (grid_built_) return;
    if (cell_div <= 0) {
      const auto g1 = make_zone_grid(box_lo, box_len, periodic, rcut, 1, 0, 1);
      const int nc1 = g1.ncells();
      const double atoms_per = nc1 > 0 ? double(m_hint) / double(nc1) : 1.0;
      int k = int(std::lround(std::cbrt(atoms_per / 2.5)));
      cell_div = k < 1 ? 1 : (k > 4 ? 4 : k);
    }
    grid_.g = make_zone_grid(box_lo, box_len, periodic, rcut, /*n_zones=*/1, /*zone_id=*/0, cell_div);
    grid_.ncells = grid_.g.ncells();
    grid_.d_counts = meam_wf_detail::meam_wf_malloc<int>(std::size_t(grid_.ncells));
    grid_.d_starts = meam_wf_detail::meam_wf_malloc<int>(std::size_t(grid_.ncells));
    grid_.d_cursor = meam_wf_detail::meam_wf_malloc<int>(std::size_t(grid_.ncells));
    std::size_t cub_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_bytes, grid_.d_counts, grid_.d_starts, grid_.ncells);
    grid_.cub_bytes = cub_bytes;
    grid_.d_cub = meam_wf_detail::meam_wf_malloc<char>(cub_bytes);
    grid_built_ = true;
  }
  ~GpuMeamWindowState() {
    free_scratch();
    meam_cells_free(grid_);
    for (void* p : {(void*)d_phirar, (void*)d_phirar1, (void*)d_phirar2, (void*)d_phirar3,
                    (void*)d_phirar4, (void*)d_phirar5, (void*)d_phirar6, (void*)d_pe_embed,
                    (void*)d_pe_pair, (void*)d_npartial, (void*)d_nzero, (void*)d_mr, (void*)d_of})
      if (p) cudaFree(p);
  }
};

template <typename Real>
struct GpuMeamWinForce {
  std::shared_ptr<GpuMeamWindowState> st;
  bool skip_sort = false;   // TEST-ONLY (G3): skip the canonical sort to RE-MEASURE the verdict
  int drop_class = 0;       // TEST-ONLY (G6 poison): drop Role C → the MB2 teeth
  int poison_s = 0;         // TEST-ONLY (G-POISON): force a too-small cell stencil (G-A teeth)

  GpuMeamWinForce(const potentials::MeamParams& p, const core::Box& box, bool cull = false,
                  int cell_div = 0)
      : st(std::make_shared<GpuMeamWindowState>(p, box, cull, cell_div)) {
    // Three-leg min-image guard (MEAM's j–k leg is a DIFFERENCE of two min-imaged vectors).
    for (int d = 0; d < 3; ++d)
      if (box.periodic[d] && box.len(d) < 2.0 * p.rc)
        throw std::invalid_argument("GpuMeamWinForce: periodic box dim < 2·rc — min-image ambiguous");
  }

  // FIREWALL — COPIED VERBATIM from MeamWinForce::assert_supported (meam_ring.hpp): accept
  // [Density, Embedding, Force(needs_transpose=true)] (the screening 3rd-atom-k transpose-replay
  // IS the mechanism), reject the EAM symmetric-Force trap / Tersoff's [BondOrder,Force] /
  // iterative. This INVERTS GpuEamWindowForce (which throws on needs_transpose).
  static void assert_supported(std::span<const potentials::PassDecl> passes) {
    if (passes.size() != 3)
      throw std::runtime_error("GpuMeamWinForce: MEAM is the 3-pass [Density, Embedding, Force] "
          "sequence — got " + std::to_string(passes.size()) + " passes");
    const potentials::PassKind want[3] = {potentials::PassKind::Density,
        potentials::PassKind::Embedding, potentials::PassKind::Force};
    for (std::size_t p = 0; p < 3; ++p)
      if (passes[p].kind != want[p])
        throw std::runtime_error("GpuMeamWinForce: unexpected pass kind at " + std::to_string(p) +
            " — MEAM is Density→Embedding→Force (a BondOrder here is the Tersoff descriptor)");
    for (const auto& p : passes)
      if (p.iterative)
        throw std::runtime_error("GpuMeamWinForce: iterative pass (QEq/CG, ReaxFF) unimplemented");
    if (passes[0].needs_transpose || passes[1].needs_transpose)
      throw std::runtime_error("GpuMeamWinForce: Density/Embedding passes must be symmetric");
    if (!passes[2].needs_transpose)
      throw std::runtime_error("GpuMeamWinForce: expected pass 2 (Force) needs_transpose=true (the "
          "screening 3rd-atom-k non-symmetric write) — a symmetric Force here is the EAM "
          "descriptor, which has no third-atom-k slot and would silently run wrong");
  }

  void compute(const double* wx, const double* wy, const double* wz, const long* key,
               int m, const int* owned, int n_owned, const core::PairGeom& geom,
               double /*rho_cap — MEAM density is internal*/,
               std::vector<core::fixed::ForceAccum>& wFx,
               std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz,
               core::fixed::EnergyAccum& pe, double& min_r2) const {
    GpuMeamWindowState& s = *st;
    std::lock_guard<std::mutex> lk(s.mu);
    if (m <= 0) return;
    s.grow(m);

    // CANONICAL FP-ORDER (mirrors MeamWinForce::compute): the ring gathers the window UNSORTED
    // (block order); the serial meam_zone_pass sorts via zone_eam_window. The screening product
    // Π_k S and the four-density partial sums are FP-non-associative ⇒ sort by global key so the
    // device sums them in the SAME order as the serial ⇒ ring-GPU ≡ serial bitwise. skip_sort is
    // the G3 RE-MEASURE hook. int64 force accumulation is order-free (B1); only the density/
    // screening FP sums need this — a no-op when already sorted (the serial-shaped z=1 window).
    std::vector<int> perm(m);
    std::iota(perm.begin(), perm.end(), 0);
    if (!skip_sort) std::sort(perm.begin(), perm.end(), [&](int a, int b) { return key[a] < key[b]; });
    std::vector<double> sx(m), sy(m), sz(m);
    std::vector<long> sk(m);
    std::vector<int> inv(m);
    for (int t = 0; t < m; ++t) { const int o = perm[t]; sx[t] = wx[o]; sy[t] = wy[o]; sz[t] = wz[o]; sk[t] = key[o]; inv[o] = t; }
    std::vector<int> sowned(n_owned);
    std::vector<int> hflag(m, 0);
    for (int t = 0; t < n_owned; ++t) { sowned[t] = inv[owned[t]]; hflag[inv[owned[t]]] = 1; }

    cudaMemcpy(s.wx, sx.data(), m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wy, sy.data(), m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wz, sz.data(), m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wkey, sk.data(), m * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_owned_flag, hflag.data(), m * sizeof(int), cudaMemcpyHostToDevice);
    if (n_owned > 0) cudaMemcpy(s.d_owned, sowned.data(), n_owned * sizeof(int), cudaMemcpyHostToDevice);

    const long long z64 = 0; const int z32 = 0;
    cudaMemcpy(s.d_pe_embed, &z64, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_pe_pair, &z64, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_npartial, &z64, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_nzero, &z64, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_mr, &s.sentinel, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_of, &z32, 4, cudaMemcpyHostToDevice);
    // initcheck: K3 writes only owned force slots, but D2H reads all m ⇒ memset first.
    cudaMemsetAsync(s.d_fx, 0, m * sizeof(long long));
    cudaMemsetAsync(s.d_fy, 0, m * sizeof(long long));
    cudaMemsetAsync(s.d_fz, 0, m * sizeof(long long));

    // K1 density → K2 embedding → K3 force, all on the null stream (ordered within one compute()).
    // K2 (embedding, no neighbour scan) is reused VERBATIM on both paths. s.cull=false keeps the
    // all-window K1/K3 as the in-process BITWISE REFERENCE. The canonical sort above STAYS (it is
    // the reference k-order the cells canonical-k cull matches).
    if (s.cull) {
      s.ensure_grid_geometry(m);
      MeamCellGrid& g = s.grid_;
      cudaMemsetAsync(g.d_counts, 0, std::size_t(g.ncells) * sizeof(int));
      cell_count_kernel<<<meam_wf_detail::meam_ng(m), kZoneBlock>>>(s.wx, s.wy, s.wz, m, g.g,
                                                                    g.d_cell_of, g.d_counts);
      cub::DeviceScan::ExclusiveSum(g.d_cub, g.cub_bytes, g.d_counts, g.d_starts, g.ncells);
      cudaMemcpyAsync(g.d_cursor, g.d_starts, std::size_t(g.ncells) * sizeof(int),
                      cudaMemcpyDeviceToDevice);
      cell_scatter_kernel<<<meam_wf_detail::meam_ng(m), kZoneBlock>>>(g.d_cell_of, m, g.d_cursor,
                                                                     g.d_order);
      if (poison_s > 0) { g.g.sx = g.g.sy = g.g.sz = poison_s; }  // TEST-ONLY (G-POISON teeth)
      meam_density_cells_kernel<<<meam_wf_detail::meam_ng(m), kZoneBlock>>>(
          s.wx, s.wy, s.wz, s.wkey, m, geom, s.scp, s.fp, s.dens_scale, g.g, g.d_starts, g.d_counts,
          g.d_order, s.d_dens, s.d_of);
      meam_embed_kernel<<<meam_wf_detail::meam_ng(m), kZoneBlock>>>(
          m, s.d_owned_flag, s.fp, s.A, s.Ec, s.ibar, s.gsmooth, s.emb_lin_neg, s.rho_ref,
          s.dens_scale, s.d_dens, s.d_ed, s.d_pe_embed);
      if (n_owned > 0)
        meam_force_cells_kernel<<<meam_wf_detail::meam_ng(n_owned), kZoneBlock>>>(
            s.wx, s.wy, s.wz, s.wkey, m, s.d_owned, n_owned, geom, s.scp, s.fp, s.view, s.dens_scale,
            s.d_dens, s.d_ed, g.g, g.d_starts, g.d_counts, g.d_order, s.d_fx, s.d_fy, s.d_fz,
            s.d_pe_pair, s.d_npartial, s.d_nzero, s.d_mr, s.d_of, drop_class);
      ++s.cells_passes;
    } else {
      meam_density_kernel<<<meam_wf_detail::meam_ng(m), kZoneBlock>>>(
          s.wx, s.wy, s.wz, m, geom, s.scp, s.fp, s.dens_scale, s.d_dens, s.d_of);
      meam_embed_kernel<<<meam_wf_detail::meam_ng(m), kZoneBlock>>>(
          m, s.d_owned_flag, s.fp, s.A, s.Ec, s.ibar, s.gsmooth, s.emb_lin_neg, s.rho_ref,
          s.dens_scale, s.d_dens, s.d_ed, s.d_pe_embed);
      if (n_owned > 0)
        meam_force_kernel<<<meam_wf_detail::meam_ng(n_owned), kZoneBlock>>>(
            s.wx, s.wy, s.wz, s.wkey, m, s.d_owned, n_owned, geom, s.scp, s.fp, s.view, s.dens_scale,
            s.d_dens, s.d_ed, s.d_fx, s.d_fy, s.d_fz, s.d_pe_pair, s.d_npartial, s.d_nzero, s.d_mr,
            s.d_of, drop_class);
    }

    std::vector<long long> hfx(m), hfy(m), hfz(m);
    cudaMemcpy(hfx.data(), s.d_fx, m * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(hfy.data(), s.d_fy, m * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(hfz.data(), s.d_fz, m * sizeof(long long), cudaMemcpyDeviceToHost);
    long long h_pe_embed = 0, h_pe_pair = 0, h_np = 0, h_nz = 0;
    cudaMemcpy(&h_pe_embed, s.d_pe_embed, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_pe_pair, s.d_pe_pair, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_np, s.d_npartial, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_nz, s.d_nzero, 8, cudaMemcpyDeviceToHost);
    unsigned long long h_mr = 0;
    cudaMemcpy(&h_mr, s.d_mr, 8, cudaMemcpyDeviceToHost);
    int of = 0;
    cudaMemcpy(&of, s.d_of, 4, cudaMemcpyDeviceToHost);
    if (cudaDeviceSynchronize() != cudaSuccess)
      throw std::runtime_error("GpuMeamWinForce: kernel launch/sync failed");
    if (of) throw std::runtime_error("GpuMeamWinForce: density/force/neighbour overflow HALT");

    // un-permute: sorted slot t ↔ original window slot perm[t] (the kernel wrote the sorted layout).
    for (int t = 0; t < m; ++t) { wFx[perm[t]].raw = hfx[t]; wFy[perm[t]].raw = hfy[t]; wFz[perm[t]].raw = hfz[t]; }
    pe.raw += h_pe_embed + h_pe_pair;  // INT64 fold — order-free, == serial's pe multiset
    double gpu_mr2;
    std::memcpy(&gpu_mr2, &h_mr, 8);
    if (gpu_mr2 < min_r2) min_r2 = gpu_mr2;
    s.last_npartial = h_np; s.last_nzero = h_nz;  // TEST-ONLY
  }
};

template <typename Real = double>
using GpuMeamRing = potentials::MeamRing<Real, GpuMeamWinForce<Real>>;

}  // namespace tdmd::cuda
