#pragma once
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#ifdef TDMD_EAM_RING_TIMERS
#include <chrono>
#endif

#include <span>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"   // Box (the cell grid needs box.lo — F1)
#include "tdmd/core/zones.hpp"  // PairGeom
#include "tdmd/cuda/zone_eam.cuh"  // eam_density/embedding/force kernels, EamSetflView
#include "tdmd/cuda/zone_eam_cells.cuh"  // E5c-integration: culled kernels + window grid
#include "tdmd/potentials/eam_donation.hpp"  // PR-2: EamDonationState/ZoneBlockView + concept
#include "tdmd/potentials/eam.hpp"  // PR-0a: assert_eam_symmetric_passes (NOT transitively
                                    // reachable via zone_eam.cuh→eam_spline→eam_analytic;
                                    // cycle-safe — eam.hpp pulls no cuda headers)
#include "tdmd/potentials/eam_spline.hpp"
#include "tdmd/potentials/many_body.hpp"  // PassDecl/PassKind — the descriptor firewall

// M6 E5b-3b — the GPU WINDOW-FORCE POLICY for the streaming multi-node EAM ring.
//
// EamRing (eam_ring.hpp) is the PROVEN bitwise orchestration oracle; its ONLY
// moving part is the per-window EAM force. This policy runs that single piece on
// the GPU — given ONE already-gathered 3-zone window {wx,wy,wz,key,m,owned,
// n_owned} it launches the E5 kernels (eam_density → eam_embedding → eam_force,
// zone_eam.cuh, reused VERBATIM) and returns the int64 force accumulators + pe +
// min_r2, BITWISE-EQUAL to eam_window_force (eam_zone.hpp). test_cuda_eam already
// proves the kernels match per-window; this policy just wires the per-call
// upload → 3 kernels → download of the int64 raws, decoded identically.
//
// Why bitwise (not ~1e-12): the spline is transcendental-free (Horner + an int
// knot index) ⇒ CPU↔GPU is strictly == under --fmad=false (the TU MUST link
// tdmd_eam_cuda_flags). int64 accumulation is associative ⇒ the per-atom raw is
// order-free ⇒ depends only on the window MULTISET, which the gather preserves
// (key = atom id, identical on both sides). So EamGpuRing inherits ALL of
// EamRing's orchestration ⇒ bitwise ≡ the CPU ring by construction.
//
// CONCURRENCY: EamRing spawns one jthread per node, all sharing the single
// policy member ⇒ compute() is called concurrently. Correctness-first (E5b is a
// correctness gate, perf is E5c): a mutex serializes the device work; the shared
// scratch grows to the max window seen. The host orchestration (FSM, transport,
// staging) still overlaps — only the kernel launches serialize. A GPU-RESIDENT
// D2D StreamTransport + per-stream buffers is the perf optimization, deferred to
// E5c; THIS delivers the CORRECT streaming z>1 GPU EAM ring.
namespace tdmd::cuda {

template <typename T>
inline T* eam_wf_malloc(std::size_t n) {
  T* d = nullptr;
  if (cudaMalloc(&d, n * sizeof(T)) != cudaSuccess)
    throw std::runtime_error("GpuEamWindowForce: cudaMalloc failed");
  return d;
}

// Shared device state: the spline view (uploaded once) + growable per-call
// scratch + the serialization mutex. Held by shared_ptr so the policy is cheap
// to copy/move (EamRing stores it by value) yet all copies share one allocation.
struct GpuEamWindowState {
  EamSetflView view{};
  double *dF = nullptr, *dra = nullptr, *drp = nullptr;
  double dens_scale = 0.0, rho_cap = 0.0;

  // growable scratch (sized to the largest window m seen so far)
  int cap_m = 0;
  double *wx = nullptr, *wy = nullptr, *wz = nullptr;
  long *wkey = nullptr;
  int *d_owned = nullptr;
  long long *d_rho = nullptr, *d_fx = nullptr, *d_fy = nullptr, *d_fz = nullptr;
  double *d_fp = nullptr;
  // per-call scalars
  long long *d_pe = nullptr;
  unsigned long long *d_mr = nullptr;
  int *d_of = nullptr;
  unsigned long long sentinel = 0;  // pos_double_bits(1e300) — empty-window seed
  std::mutex mu;

#ifdef TDMD_EAM_RING_TIMERS
  // per-phase attribution (bench-only, compile-gated ⇒ the proven bitwise hot path
  // is byte-identical in default builds). Piggybacks on compute()'s existing
  // end-of-pass cudaDeviceSynchronize — adds NO extra sync, so it does not perturb
  // the wall-time. compute_wall ≈ t_h2d_s + t_rest_s; t_kernel_ms is the PURE GPU
  // kernel time (hidden inside the blocking D2H wait of t_rest_s). f=kernel/wall.
  cudaEvent_t ev_ks_ = nullptr, ev_ke_ = nullptr;
  double t_h2d_s = 0, t_rest_s = 0, t_kernel_ms = 0;
  unsigned long long n_calls = 0;
#endif

  // E5c-integration — cell-list culling of the per-window density+force (cells ≡
  // all-window BITWISE by B1, proven in test_cuda_eam_cells). The box is captured
  // at construction (static membership/box for the run) — compute() only gets a
  // PairGeom, which lacks box.lo the grid needs (F1). cull=false keeps the O(m²)
  // all-window path as the in-process bitwise reference. cells_passes is the
  // non-vacuity witness (the culled path was actually taken in-ring).
  double box_lo[3] = {0, 0, 0}, box_len[3] = {0, 0, 0};
  bool periodic[3] = {false, false, false};
  double rcut = 0.0;
  bool cull = true;
  int cell_div = 0;  // sub-rcut binning (E5c-subrcut): cells ~rcut/cell_div, ±cell_div
                     // stencil. 0 = AUTO (target ~2.5 atoms/cell, resolved in
                     // ensure_grid_geometry — the production default, 1.71× on Al_zhou).
                     // 1 = legacy ~rcut cells / ±1. Any k is bitwise == all-window (A-k).
  unsigned long long cells_passes = 0;
  EamCellGrid grid_{};       // persistent — geometry built once, refreshed per pass
  bool grid_built_ = false;

  GpuEamWindowState(const potentials::EamSetfl<double>& setfl, const core::Box& box,
                    bool cull_, int cell_div_ = 0)  // 0 = AUTO (see cell_div field)
      : cull(cull_), cell_div(cell_div_) {
    box_lo[0] = box.lo[0]; box_lo[1] = box.lo[1]; box_lo[2] = box.lo[2];
    box_len[0] = box.len(0); box_len[1] = box.len(1); box_len[2] = box.len(2);
    periodic[0] = box.periodic[0]; periodic[1] = box.periodic[1]; periodic[2] = box.periodic[2];
    rcut = setfl.rcut;
    init_setfl(setfl);
  }

  void init_setfl(const potentials::EamSetfl<double>& setfl) {
    std::vector<double> Fspl = setfl.Fspl, rhoaspl = setfl.rhoaspl, rphispl = setfl.rphispl;
    dF = eam_wf_malloc<double>(Fspl.size());
    dra = eam_wf_malloc<double>(rhoaspl.size());
    drp = eam_wf_malloc<double>(rphispl.size());
    cudaMemcpy(dF, Fspl.data(), Fspl.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dra, rhoaspl.data(), rhoaspl.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(drp, rphispl.data(), rphispl.size() * 8, cudaMemcpyHostToDevice);
    view = EamSetflView{dF, dra, drp, setfl.Nrho, setfl.Nr, setfl.rdrho, setfl.rdr, setfl.rcut};
    const int fb = setfl.density_fracbits();
    dens_scale = (fb == 44) ? core::fixed::FixedAccum<44>::kScale
                            : core::fixed::FixedAccum<40>::kScale;
    rho_cap = setfl.density_grid_max();
    // min_r2 seed = 1e300 (F5: NOT +inf — matches the CPU EAM oracle's 1e300, so
    // CPU↔GPU stays bitwise even on empty windows where no pair updates it).
    double v = 1e300;
    std::memcpy(&sentinel, &v, 8);
    d_pe = eam_wf_malloc<long long>(1);
    d_mr = eam_wf_malloc<unsigned long long>(1);
    d_of = eam_wf_malloc<int>(1);
    grow(64);  // initial scratch
#ifdef TDMD_EAM_RING_TIMERS
    cudaEventCreate(&ev_ks_); cudaEventCreate(&ev_ke_);
#endif
  }

  void grow(int m) {
    if (m <= cap_m) return;
    free_scratch();
    cap_m = m;
    wx = eam_wf_malloc<double>(cap_m);
    wy = eam_wf_malloc<double>(cap_m);
    wz = eam_wf_malloc<double>(cap_m);
    wkey = eam_wf_malloc<long>(cap_m);
    d_owned = eam_wf_malloc<int>(cap_m);
    d_rho = eam_wf_malloc<long long>(cap_m);
    d_fp = eam_wf_malloc<double>(cap_m);
    d_fx = eam_wf_malloc<long long>(cap_m);
    d_fy = eam_wf_malloc<long long>(cap_m);
    d_fz = eam_wf_malloc<long long>(cap_m);
    // cell-list per-atom buffers grow with the window (the cell COUNT — ncells —
    // is box-static, allocated once in ensure_grid_geometry).
    if (grid_.d_cell_of) { cudaFree(grid_.d_cell_of); cudaFree(grid_.d_order); }
    grid_.d_cell_of = eam_wf_malloc<int>(cap_m);
    grid_.d_order = eam_wf_malloc<int>(cap_m);
    grid_.m = cap_m;
  }

  // Build the window cell grid GEOMETRY once (box-static: membership/box fixed for
  // the run ⇒ make_zone_grid output + ncells are invariant). Per-pass only the
  // counts/order are refreshed (in compute). Whole-box, periodic-z grid (F3) —
  // the SAME full-Lz min-image fold geom.reduce uses, so it is a sound superset
  // filter for the gathered cyclic subset (a z-AABB slab would miss seam pairs).
  void ensure_grid_geometry(int m_hint) {
    if (grid_built_) return;
    // AUTO (cell_div==0): target ~2.5 atoms per cell — the MEASURED-optimal cell
    // occupancy (E5c-subrcut bake-off: k=3 at Al_zhou rcut=10.1 ⇒ 62/27≈2.3 atoms/cell).
    // It is a hardware-calibrated sweet spot (cell-enumeration overhead vs over-fetch
    // saving), ~rcut/density-INDEPENDENT, so derive k from the realized k=1 occupancy
    // m/ncells_k1. Adaptive: Al_zhou ⇒ k=3 (1.71× ring); short-rcut/sparse ⇒ k=1 (no
    // regression). Bitwise-safe for ANY k (test_cuda_eam_cells A-k). Clamped [1,4].
    if (cell_div <= 0) {
      const auto g1 = make_zone_grid(box_lo, box_len, periodic, rcut, 1, 0, 1);
      const int nc1 = g1.ncells();
      const double atoms_per = nc1 > 0 ? double(m_hint) / double(nc1) : 1.0;
      int k = int(std::lround(std::cbrt(atoms_per / 2.5)));
      cell_div = k < 1 ? 1 : (k > 4 ? 4 : k);
    }
    grid_.g = make_zone_grid(box_lo, box_len, periodic, rcut, /*n_zones=*/1, /*zone_id=*/0, cell_div);
    grid_.ncells = grid_.g.ncells();
    grid_.d_counts = eam_wf_malloc<int>(std::size_t(grid_.ncells));
    grid_.d_starts = eam_wf_malloc<int>(std::size_t(grid_.ncells));
    grid_.d_cursor = eam_wf_malloc<int>(std::size_t(grid_.ncells));
    std::size_t cub_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_bytes, grid_.d_counts, grid_.d_starts,
                                  grid_.ncells);
    grid_.cub_bytes = cub_bytes;
    grid_.d_cub = eam_wf_malloc<char>(cub_bytes);
    grid_built_ = true;
  }

  void free_scratch() {
    for (void* p : {(void*)wx, (void*)wy, (void*)wz, (void*)wkey, (void*)d_owned,
                    (void*)d_rho, (void*)d_fp, (void*)d_fx, (void*)d_fy, (void*)d_fz})
      if (p) cudaFree(p);
    wx = wy = wz = d_fp = nullptr;
    wkey = nullptr;
    d_owned = nullptr;
    d_rho = d_fx = d_fy = d_fz = nullptr;
  }

  ~GpuEamWindowState() {
    free_scratch();
    for (void* p : {(void*)grid_.d_cell_of, (void*)grid_.d_order, (void*)grid_.d_counts,
                    (void*)grid_.d_starts, (void*)grid_.d_cursor, grid_.d_cub})
      if (p) cudaFree(p);
    for (void* p : {(void*)dF, (void*)dra, (void*)drp, (void*)d_pe, (void*)d_mr, (void*)d_of})
      if (p) cudaFree(p);
#ifdef TDMD_EAM_RING_TIMERS
    if (ev_ks_) cudaEventDestroy(ev_ks_);
    if (ev_ke_) cudaEventDestroy(ev_ke_);
#endif
  }
};

// The window-force policy object EamRing stores by value. Cheap to copy/move
// (shared_ptr to the device state). Matches the CpuEamWindowForce::compute
// signature exactly, so EamRing<Real,Math,GpuEamWindowForce> just works.
struct GpuEamWindowForce {
  std::shared_ptr<GpuEamWindowState> st;

  // box is REQUIRED — the cell grid needs box.lo (F1); compute() only gets a
  // PairGeom. cull=true ⇒ cell-list culling (cells ≡ all-window bitwise, B1);
  // cull=false ⇒ the O(m²) all-window path (the in-process bitwise reference).
  GpuEamWindowForce(const potentials::EamSetfl<double>& setfl, const core::Box& box,
                    bool cull = true, int cell_div = 0)  // 0 = AUTO (production default)
      : st(std::make_shared<GpuEamWindowState>(setfl, box, cull, cell_div)) {}

  // non-vacuity witness: how many compute() calls took the culled path.
  unsigned long long cells_passes() const { return st->cells_passes; }

  // DESCRIPTOR FIREWALL (correctness, not perf). This GPU policy implements the EAM
  // SYMMETRIC 3-pass force: the int64 accumulator writes q(j)=−q(i) (zone_force.cuh),
  // valid ONLY because EAM's (F'_i+F'_j)·dρ bracket is symmetric. A future MEAM/Tersoff
  // angular / bond-order term is NON-symmetric (writes force to a THIRD atom k, declared
  // PassDecl.needs_transpose) — and an iterative QEq/CG solver (PassDecl.iterative) is a
  // different control flow entirely. Today compute() hardcodes the [Density,Embedding,
  // Force] kernel sequence (heterogeneous device signatures ⇒ no device-side polymorphic
  // loop); WITHOUT this gate a potential could ship needs_transpose=true, this policy would
  // silently run the symmetric accumulator and produce DETERMINISTIC, bitwise-stable,
  // 1-vs-z-identical — and physically WRONG forces, invisible to every consistency gate.
  // EamRing calls this UNCONDITIONALLY before the run (PR-0a; the old `if constexpr
  // requires` opt-in is gone); EAM's passes() ⇒ exactly [Density,Embedding,Force], all
  // symmetric ⇒ a pure no-op (no behavior change).
  //
  // FIREWALL SCOPE (PR-0a — BOTH named gaps CLOSED): the two escape hatches this comment
  // used to name are now shut. (1) The z=1 driver eam_gpu_run_singlenode gained a mandatory
  // descriptor gate (eam_conveyor_gpu.cuh — its `FIREWALL GAP` anchor now reads CLOSED). (2)
  // The opt-in `if constexpr requires` was promoted to the C++20 concept WindowForcePolicy
  // (many_body.hpp) with a class-scope static_assert in every ring ⇒ a policy that omits/
  // mis-signs assert_supported is now a COMPILE error, not a silent bypass (the overdue
  // MB1/MB2 promise). This gate itself now DELEGATES to the single EAM source of truth
  // (assert_eam_symmetric_passes, eam.hpp) ⇒ no parity drift between CPU/GPU/driver; the
  // accept/reject SEMANTICS are preserved (message texts normalized, `who`-prefixed —
  // the teeth check throw/no-throw, not strings; design R4).
  static void assert_supported(std::span<const potentials::PassDecl> passes) {
    potentials::assert_eam_symmetric_passes(passes, "GpuEamWindowForce");
  }

#ifdef TDMD_EAM_RING_TIMERS
  // per-phase attribution accessors (bench-only). compute_wall ≈ h2d+rest; the pure
  // GPU kernel time is t_kernel_ms (ms). f = (t_kernel_ms/1000)/compute_wall.
  double timer_h2d_s() const { return st->t_h2d_s; }
  double timer_rest_s() const { return st->t_rest_s; }
  double timer_kernel_s() const { return st->t_kernel_ms / 1000.0; }
  unsigned long long timer_calls() const { return st->n_calls; }
#endif

  void compute(const double* wx, const double* wy, const double* wz, const long* key,
               int m, const int* owned, int n_owned, const core::PairGeom& geom,
               double /*rho_cap_unused*/, std::vector<core::fixed::ForceAccum>& wFx,
               std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz,
               core::fixed::EnergyAccum& pe, double& min_r2) const {
    using namespace eam_sn_detail;  // kB, ng
    GpuEamWindowState& s = *st;
    std::lock_guard<std::mutex> lk(s.mu);  // serialize device work (correctness)
    if (m <= 0) return;  // empty window: no owned forces, pe/min_r2 unchanged
    s.grow(m);

#ifdef TDMD_EAM_RING_TIMERS
    const auto _t_h2d0 = std::chrono::steady_clock::now();
#endif
    // upload the gathered window (key = the actual atom ids — the φ-once order,
    // IDENTICAL to the CPU eam_window_force call; NOT window-local indices).
    cudaMemcpy(s.wx, wx, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wy, wy, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wz, wz, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.wkey, key, m * sizeof(long), cudaMemcpyHostToDevice);
    if (n_owned > 0)
      cudaMemcpy(s.d_owned, owned, n_owned * sizeof(int), cudaMemcpyHostToDevice);

    const long long zpe = 0;
    const int zof = 0;
    cudaMemcpy(s.d_pe, &zpe, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_mr, &s.sentinel, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_of, &zof, 4, cudaMemcpyHostToDevice);

#ifdef TDMD_EAM_RING_TIMERS
    const auto _t_rest0 = std::chrono::steady_clock::now();
    s.t_h2d_s += std::chrono::duration<double>(_t_rest0 - _t_h2d0).count();
    cudaEventRecord(s.ev_ks_);  // null-stream ⇒ orders after the H2D, before the kernels
#endif
    // ρ → F'(ρ) → force. embedding (O(m)) is identical on both paths; density and
    // force are the heavy passes, culled when s.cull. ALL on the null stream (same
    // as the force kernels) ⇒ grid → density → force ordered within one compute();
    // the mutex serializes across z nodes ⇒ race-free, no events (do NOT split
    // streams without an event — E5c per-stream is deferred).
    if (s.cull) {
      // refresh the box-static grid over the freshly-uploaded window (count → scan
      // → scatter; geometry built once). cells ≡ all-window int64 BITWISE (B1:
      // 27-cell candidates are a SUPERSET; geom.reduce r²<rc² re-test + quantize +
      // order-free int64 sum unchanged) ⇒ the ring trajectory stays bitwise.
      s.ensure_grid_geometry(m);
      EamCellGrid& g = s.grid_;
      cudaMemsetAsync(g.d_counts, 0, std::size_t(g.ncells) * sizeof(int));
      cell_count_kernel<<<ng(m), kB>>>(s.wx, s.wy, s.wz, m, g.g, g.d_cell_of, g.d_counts);
      cub::DeviceScan::ExclusiveSum(g.d_cub, g.cub_bytes, g.d_counts, g.d_starts, g.ncells);
      cudaMemcpyAsync(g.d_cursor, g.d_starts, std::size_t(g.ncells) * sizeof(int),
                      cudaMemcpyDeviceToDevice);
      cell_scatter_kernel<<<ng(m), kB>>>(g.d_cell_of, m, g.d_cursor, g.d_order);
      eam_density_cells_kernel<<<ng(m), kB>>>(s.wx, s.wy, s.wz, m, geom, s.view,
                                              s.dens_scale, g.g, g.d_starts, g.d_counts,
                                              g.d_order, s.d_rho, s.d_of);
      eam_embedding_kernel<<<ng(m), kB>>>(m, s.view, s.dens_scale, s.rho_cap, s.d_rho,
                                          s.d_fp, s.d_of);
      if (n_owned > 0)
        eam_force_cells_kernel<<<ng(n_owned), kB>>>(s.wx, s.wy, s.wz, s.wkey, m, s.d_owned,
                                                    n_owned, geom, s.view, s.dens_scale,
                                                    s.d_rho, s.d_fp, g.g, g.d_starts,
                                                    g.d_counts, g.d_order, s.d_fx, s.d_fy,
                                                    s.d_fz, s.d_pe, s.d_mr, s.d_of);
      ++s.cells_passes;
    } else {  // all-window O(m²) — the in-process bitwise reference (verbatim E5).
      eam_density_kernel<<<ng(m), kB>>>(s.wx, s.wy, s.wz, m, geom, s.view, s.dens_scale,
                                        s.d_rho, s.d_of);
      eam_embedding_kernel<<<ng(m), kB>>>(m, s.view, s.dens_scale, s.rho_cap, s.d_rho,
                                          s.d_fp, s.d_of);
      if (n_owned > 0)
        eam_force_kernel<<<ng(n_owned), kB>>>(s.wx, s.wy, s.wz, s.wkey, m, s.d_owned,
                                              n_owned, geom, s.view, s.dens_scale,
                                              s.d_rho, s.d_fp, s.d_fx, s.d_fy, s.d_fz,
                                              s.d_pe, s.d_mr, s.d_of);
    }
#ifdef TDMD_EAM_RING_TIMERS
    cudaEventRecord(s.ev_ke_);  // after the force kernel, before the D2H
#endif

    // download the int64 raws + scalars
    int of = 0;
    long long h_pe = 0;
    unsigned long long h_mr = 0;
    std::vector<long long> hfx, hfy, hfz;
    if (n_owned > 0) {
      hfx.resize(m); hfy.resize(m); hfz.resize(m);
      cudaMemcpy(hfx.data(), s.d_fx, m * 8, cudaMemcpyDeviceToHost);
      cudaMemcpy(hfy.data(), s.d_fy, m * 8, cudaMemcpyDeviceToHost);
      cudaMemcpy(hfz.data(), s.d_fz, m * 8, cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(&of, s.d_of, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_pe, s.d_pe, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_mr, s.d_mr, 8, cudaMemcpyDeviceToHost);
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
      throw std::runtime_error(std::string("GpuEamWindowForce: ") + cudaGetErrorString(err));
#ifdef TDMD_EAM_RING_TIMERS
    // the sync above completed the kernel events ⇒ read with NO extra sync. t_rest_s
    // = kernel-launch + (blocking D2H that waits on the kernels) + this sync.
    s.t_rest_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - _t_rest0).count();
    float _kms = 0; cudaEventElapsedTime(&_kms, s.ev_ks_, s.ev_ke_);
    s.t_kernel_ms += _kms; ++s.n_calls;
#endif

    // HALT symmetry: the CPU eam_window_force THROWS on ρ>rho_cap; the kernel
    // sets overflow bit 2 (rho-cap) / bit 1 (quantize). Throw ⇒ node_main maps
    // it to Halt::Internal, mirroring the CPU oracle's throw.
    if (of)
      throw std::runtime_error("GpuEamWindowForce: density/rho-cap overflow HALT");

    // merge results into the ring's running accumulators (decoded IDENTICALLY to
    // the CPU policy: ForceAccum.raw is the int64; pe.raw += d_pe; min_r2 = min).
    if (n_owned > 0) {
      for (int o = 0; o < n_owned; ++o) {
        const int loc = owned[o];
        wFx[std::size_t(loc)].raw = hfx[std::size_t(loc)];
        wFy[std::size_t(loc)].raw = hfy[std::size_t(loc)];
        wFz[std::size_t(loc)].raw = hfz[std::size_t(loc)];
      }
    }
    pe.raw += h_pe;  // int64 add is associative ⇒ == CPU's pe.add() multiset sum
    double gpu_mr2;
    std::memcpy(&gpu_mr2, &h_mr, 8);
    if (gpu_mr2 < min_r2) min_r2 = gpu_mr2;
  }

  // PR-2 (live donation ring) — the GPU policy models DonatingWindowForcePolicy so
  // EamGpuRing compiles. In PR-2 the GPU ring RECOMPUTES density per-window (bitwise ≡
  // today); device-resident donation is PR-3b (gated R_W>=1.15, NULL-rollback). The two
  // donation hooks are INERT (the ring still calls them + owns the ledger, so END never
  // starves, but no rho is accumulated device-side); compose IGNORES the (all-zero) rho_w
  // and runs the existing E5 device recompute ⇒ EamGpuRing output == today, bitwise.
  template <class DA>
  void on_zone_arrival(potentials::EamDonationState<DA>&, int, const potentials::ZoneBlockView&,
                       const core::PairGeom&) const {}
  template <class DA>
  void on_edge(potentials::EamDonationState<DA>&, int, int, const potentials::ZoneBlockView&,
               const potentials::ZoneBlockView&, const core::PairGeom&) const {}
  template <class DA>
  void compose(const double* wx, const double* wy, const double* wz, const long* key, int m,
               const int* owned, int n_owned, const core::PairGeom& geom, double rho_cap,
               const DA* /*rho_w — ignored; GPU recomputes density in compute() until PR-3b*/,
               std::vector<core::fixed::ForceAccum>& wFx,
               std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz, core::fixed::EnergyAccum& pe,
               double& min_r2, int /*zone_j*/) const {
    compute(wx, wy, wz, key, m, owned, n_owned, geom, rho_cap, wFx, wFy, wFz, pe, min_r2);
  }
};

}  // namespace tdmd::cuda
