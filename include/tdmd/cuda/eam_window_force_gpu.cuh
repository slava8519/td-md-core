#pragma once
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
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
#include "tdmd/cuda/zone_eam_donation.cuh"  // PR-3a: the cross donation kernel pair
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

// PR-3a — test-only donation poison knobs (policy-level fields, NEVER kernel defaults —
// the P-k precedent). Default INERT. They express the intra-batch faults the ring ledger
// is honestly blind to (the T-5..T-8 class, on device):
//   stencil_override — force the stencil radius (±s) at donation-kernel launch; ONLY
//     meaningful with cell_div >= 2 (at k=1 the override-to-1 is the identity — recorded,
//     design amendment M5);
//   one_sided — route the cross batch's B-side writes into a discard lane (the classic
//     half-bug; ledger stays clean, physics red — FP64-oracle witness);
//   drop_outside_nominal_slab — park atoms whose wrapped z lies outside the zone's NOMINAL
//     [zone_lo, zone_hi] far outside the box in the slab MIRROR upload only: mimics a
//     slab-extent CSR losing a drifted donor (the REAL Tier-0 drift-binning hazard the
//     whole-box grid discharges by construction; design amendment M4 — the A5 kill);
//   defer_seam_to_next_compose — (PR-3b, B2) buffer the pbc seam cross launch past
//     finalize(n-1)'s lane read (fires at the entry of the TAIL finalize(0) compose —
//     design amendment S9); pins the in-scan seam deadline as load-bearing ON DEVICE.
struct GpuDonationPoison {
  int stencil_override = -1;
  bool one_sided = false;
  bool drop_outside_nominal_slab = false;
  bool defer_seam_to_next_compose = false;
};

// PR-3a — a captured per-(pass, window) lane snapshot (test-only; the A1/B-D2 capture
// seam, design J11). GPU side: D2H of the <=3 window lane segments at compose. The CPU
// twin (a wrapper policy logging dstate.rho[label]) lives in the test TU.
struct EamLaneCapture {
  long pass = 0;
  int zone_j = -1;
  int nb = 0;
  int label[3] = {-1, -1, -1};
  std::vector<long long> lane[3];  // raw int64 rho, member order
};

struct GpuEamDonationNodeStoreImpl;  // fwd (PR-3a; defined below the state)

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

  // PR-3a — the device-donation knob + observability (design §3.2). donate_device=false
  // (the DEFAULT) ⇒ the hooks return before ANY device call and this state is byte-inert:
  // the default path is byte-identical to HEAD. donate_device=true ⇒ the hooks run the
  // donation kernels into the per-NODE lanes (GpuEamDonationNodeStore); in PR-3a compose
  // still RECOMPUTES density (knob-ON is purely observational — gate A0); the donated
  // compose (consuming the lanes) is PR-3b.
  bool donate_device = false;
  // non-vacuity counters (mutex-held increments; the cells_passes() pattern). Semantics
  // PINNED (design amendment S4): self/cross count every EXECUTED batch INCLUDING the
  // vacuous n==0 ones (incremented before the early-return); A4 checks the absence of
  // kernel LAUNCHES for n==0, not of the increment. donated_composes stays 0 until PR-3b.
  unsigned long long donation_self_batches_ = 0;
  unsigned long long donation_cross_batches_ = 0;
  unsigned long long donated_composes_ = 0;
  GpuDonationPoison poison;   // test-only, default inert
  bool capture_lanes = false;  // test-only (A1/B-D2): log lane snapshots at compose
  std::vector<EamLaneCapture> capture_log;
  // registry of ring-created NodeStores (A6: the sticky donation-overflow flag lives in
  // the per-node impl; the ring owns the stores privately, so the observability accessor
  // reads them through this weak registry, post-run).
  std::vector<std::weak_ptr<GpuEamDonationNodeStoreImpl>> node_impls_;
#ifdef TDMD_EAM_RING_TIMERS
  // PR-3a (design amendment M3): donation-kernel device time. The hooks launch with NO
  // sync, so their GPU time would otherwise land in the NEXT compose's blocking-H2D
  // bucket (null-stream ordering) and falsify the work-removal attribution. TIMERS builds
  // bracket each donation launch with this event pair and cudaEventSynchronize IMMEDIATELY
  // (a TIMERS-ONLY sync — headline R_W numbers come from the default build; attribution
  // from the TIMERS build). Expectation (restated): (compose kernels + donation kernels)
  // total device-time drop ≈ the wall drop; H2D bytes unchanged; the donated leg's h2d
  // bucket LEGITIMATELY shrinks by what moved here.
  cudaEvent_t ev_ds_ = nullptr, ev_de_ = nullptr;
  double t_donation_kernel_ms = 0;
  unsigned long long n_donation_launches = 0;
#endif

  GpuEamWindowState(const potentials::EamSetfl<double>& setfl, const core::Box& box,
                    bool cull_, int cell_div_ = 0,  // 0 = AUTO (see cell_div field)
                    bool donate_device_ = false)    // PR-3a: default byte-identical to HEAD
      : cull(cull_), cell_div(cell_div_), donate_device(donate_device_) {
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
    cudaEventCreate(&ev_ds_); cudaEventCreate(&ev_de_);
#endif
  }

  void grow(int m) {
    if (m <= cap_m) return;
    // PR-3a (design amendment M9 — use-after-free guard): the donation hooks release the
    // mutex with kernels QUEUED on the null stream that reference grid_.d_cell_of/d_order;
    // pre-PR-3a compute() fully synced before unlock, so free-after-queue could not happen.
    // grow() is grow-only/rare ⇒ an unconditional sync before freeing costs nothing and
    // restores the invariant "no queued work ever references a freed buffer".
    if (cap_m > 0) cudaDeviceSynchronize();
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
    if (ev_ds_) cudaEventDestroy(ev_ds_);
    if (ev_de_) cudaEventDestroy(ev_de_);
#endif
  }
};

// PR-3a — the per-NODE device donation store (design §3.1): per-zone (LABEL-keyed, NEVER
// slot — the G-ROT/rotation tooth guards the mapping ring-side) position slab mirrors +
// persistent int64 rho lanes. Ring-created (EamRing::run() builds one per node jthread via
// make_node_state — the audit ownership shape; per-node pass-local, hard constraint 1).
// NOT in the z-shared GpuEamWindowState and NOT in the growable window scratch. Fully LAZY:
// knob-off runs allocate NOTHING device-side here.
//
// d_of_dn STICKY-WITHOUT-PER-PASS-RESET BANNER (design OQ5/R5): the donation overflow flag
// is zeroed ONCE at first use and sticky thereafter — safe because a HALT is TERMINAL for
// run() (no pass-retry / partial-replay API exists, SPEC(6)). If in-flight mid-ring rescue
// (the deferred M7 item) ever lands, a pass-boundary reset MUST be added here.
struct GpuEamDonationNodeStoreImpl {
  long token = 0;  // begin_pass(h) sets it (host-only; no device work)
  struct ZoneLane {
    double *x = nullptr, *y = nullptr, *z = nullptr;  // slab mirror, 24 B/atom (POST-drift)
    long long* rho = nullptr;  // persistent per-zone int64 rho lane (fb-agnostic raw)
    int cap = 0, n = 0;        // grow-only capacity / members this pass
    long stamp = -1;           // == token <=> uploaded this pass  [PR-4: rebuild_epoch seat]
  };
  std::vector<ZoneLane> zone;  // index == zone LABEL (fsm.id)
  double *cxx = nullptr, *cxy = nullptr, *cxz = nullptr;  // cross concat scratch (grow-only)
  int ccap = 0;
  long long* rho_discard = nullptr;  // one_sided poison sink (grow-only, test-only)
  int discard_cap = 0;
  int* d_of_dn = nullptr;  // sticky donation overflow (bit 1 = quantize) — see BANNER above

  explicit GpuEamDonationNodeStoreImpl(int n_zones) : zone(std::size_t(n_zones)) {}
  GpuEamDonationNodeStoreImpl(const GpuEamDonationNodeStoreImpl&) = delete;
  GpuEamDonationNodeStoreImpl& operator=(const GpuEamDonationNodeStoreImpl&) = delete;

  void lane_grow(ZoneLane& L, int n) {
    if (n <= L.cap) return;
    // M9 twin: previous-pass kernels referencing the old lane arrays are long completed
    // (every pass syncs at each compose), but the uniform rule is sync-before-free.
    cudaDeviceSynchronize();
    for (void* p : {(void*)L.x, (void*)L.y, (void*)L.z, (void*)L.rho})
      if (p) cudaFree(p);
    L.cap = n;
    L.x = eam_wf_malloc<double>(std::size_t(n));
    L.y = eam_wf_malloc<double>(std::size_t(n));
    L.z = eam_wf_malloc<double>(std::size_t(n));
    L.rho = eam_wf_malloc<long long>(std::size_t(n));
  }
  void concat_grow(int m) {
    if (m <= ccap) return;
    cudaDeviceSynchronize();  // M9 twin
    for (void* p : {(void*)cxx, (void*)cxy, (void*)cxz})
      if (p) cudaFree(p);
    ccap = m;
    cxx = eam_wf_malloc<double>(std::size_t(m));
    cxy = eam_wf_malloc<double>(std::size_t(m));
    cxz = eam_wf_malloc<double>(std::size_t(m));
  }
  long long* discard_lane(int n) {  // one_sided poison sink
    if (n > discard_cap) {
      cudaDeviceSynchronize();  // M9 twin
      if (rho_discard) cudaFree(rho_discard);
      discard_cap = n;
      rho_discard = eam_wf_malloc<long long>(std::size_t(n));
      // the cross kernel's lane epilogue is `+=` (RMW) — it READS the sink; zero it so
      // the poison leg is initcheck-clean (the values are discarded either way — J10).
      cudaMemset(rho_discard, 0, std::size_t(n) * 8);
    }
    return rho_discard;
  }
  int* ensure_of() {  // lazy alloc + zero ONCE (sticky; see BANNER)
    if (!d_of_dn) {
      d_of_dn = eam_wf_malloc<int>(1);
      cudaMemset(d_of_dn, 0, sizeof(int));
    }
    return d_of_dn;
  }

  ~GpuEamDonationNodeStoreImpl() {
    for (auto& L : zone)
      for (void* p : {(void*)L.x, (void*)L.y, (void*)L.z, (void*)L.rho})
        if (p) cudaFree(p);
    for (void* p : {(void*)cxx, (void*)cxy, (void*)cxz, (void*)rho_discard, (void*)d_of_dn})
      if (p) cudaFree(p);
  }
};

// The handle the ring stores by value (std::vector<NodeState>) — shared_ptr keeps the
// impl non-copyable device state stable across vector growth.
struct GpuEamDonationNodeStore {
  std::shared_ptr<GpuEamDonationNodeStoreImpl> impl;
  void begin_pass(long h) {
    if (impl) impl->token = h;
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
  // donate_device=false (DEFAULT, PR-3a) ⇒ the donation hooks are device-free no-ops and
  // the whole policy is byte-identical to HEAD; =true ⇒ hooks run donation kernels into
  // the per-node lanes (observational in PR-3a — compose still recomputes; gate A0).
  GpuEamWindowForce(const potentials::EamSetfl<double>& setfl, const core::Box& box,
                    bool cull = true, int cell_div = 0,  // 0 = AUTO (production default)
                    bool donate_device = false)
      : st(std::make_shared<GpuEamWindowState>(setfl, box, cull, cell_div, donate_device)) {}

  // non-vacuity witness: how many compute() calls took the culled path.
  unsigned long long cells_passes() const { return st->cells_passes; }

  // PR-3a — per-NODE donation state (ring-owned; created once per node jthread in run()).
  using NodeState = GpuEamDonationNodeStore;
  NodeState make_node_state(int n_zones) const {
    // Fully lazy: no device allocation here (knob-off runs stay device-free on this path).
    auto impl = std::make_shared<GpuEamDonationNodeStoreImpl>(n_zones);
    {
      std::lock_guard<std::mutex> lk(st->mu);
      st->node_impls_.push_back(impl);  // A6 observability registry (weak)
    }
    return NodeState{std::move(impl)};
  }

  // PR-3a — donation observability (test/bench accessors; S4-pinned counter semantics).
  unsigned long long donation_self_batches() const { return st->donation_self_batches_; }
  unsigned long long donation_cross_batches() const { return st->donation_cross_batches_; }
  unsigned long long donated_composes() const { return st->donated_composes_; }
  const std::vector<EamLaneCapture>& capture_log() const { return st->capture_log; }
  // A6: OR of the sticky donation overflow flags across every ring-created NodeStore.
  // Tolerates a HALTed run (design amendment S5) — call AFTER run(); D2H is synchronous.
  int donation_overflow() const {
    int acc = 0;
    std::lock_guard<std::mutex> lk(st->mu);
    for (const auto& w : st->node_impls_)
      if (auto impl = w.lock(); impl && impl->d_of_dn) {
        int of = 0;
        cudaMemcpy(&of, impl->d_of_dn, sizeof(int), cudaMemcpyDeviceToHost);
        acc |= of;
      }
    return acc;
  }
#ifdef TDMD_EAM_RING_TIMERS
  double timer_donation_kernel_s() const { return st->t_donation_kernel_ms / 1000.0; }
  unsigned long long timer_donation_launches() const { return st->n_donation_launches; }
#endif

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

  // PR-3a (live device-donation substrate — design §3.3/§4.3). donate_device=false
  // (DEFAULT): the hooks return BEFORE any device call ⇒ byte-identical to HEAD. =true:
  // the SELF batch runs the frozen density kernel VERBATIM over the zone's slab mirror
  // (its single overwrite-write per lane element IS the per-pass reset by construction —
  // the schedule fires self(L) strictly before any cross touching L; the memsetAsync
  // below is DEFENSIVE, NOT load-bearing — the staleness guard is the stamp fence); the
  // CROSS batch D2D-concats two slabs and runs the one new kernel pair. NO sync, NO D2H
  // in the hooks (the next compose's blocking ops are null-stream-ordered behind them).
  // The host EamDonationState (the CPU mirror) is NOT touched — device lanes are the
  // GPU-side rho store; compose ignores both in PR-3a (recompute; gate A0 observational).
  template <class DA>
  void on_zone_arrival(NodeState& ns, potentials::EamDonationState<DA>& /*st_host*/, int label,
                       const potentials::ZoneBlockView& blk, const core::PairGeom& geom) const {
    GpuEamWindowState& s = *st;
    if (!s.donate_device) return;  // knob-off: NO device call (A0/A8 by construction)
    // fb trap (A3): the lane raws are compared against DA-format CPU raws — the policy's
    // ctor-frozen dens_scale must BE DA::kScale (both keyed off runtime density_fracbits).
    if (DA::kScale != s.dens_scale)
      throw std::logic_error("GpuEamWindowForce: donation DA::kScale != policy dens_scale "
                             "(fb dispatch mismatch — the T-12/A3 trap)");
    std::lock_guard<std::mutex> lk(s.mu);
    ++s.donation_self_batches_;  // S4: counts EXECUTED batches incl. vacuous
    auto& impl = *ns.impl;
    auto& L = impl.zone.at(std::size_t(label));
    impl.lane_grow(L, blk.n);
    L.n = blk.n;
    L.stamp = impl.token;  // the staleness fence (PR-4: rebuild_epoch seat)
    if (blk.n == 0) return;  // vacuous batch: stamp set, ledger closes ring-side, no launch
    // slab H2D — the very POST-drift bytes the finalize gather reads (donate_position runs
    // after ensure_drift; positions are immutable for the rest of the pass).
    if (s.poison.drop_outside_nominal_slab) {
      // A5 kill (amendment M4): park drifted atoms far outside the box in the MIRROR only
      // — mimics a slab-extent CSR losing the drifted donor (all its pairs vanish from the
      // batch, both directions) ⇒ lanes + FP64-oracle go red. Nominal slab from the label.
      const int nz = int(impl.zone.size());
      const double w = s.box_len[2] / double(nz);
      const double lo = s.box_lo[2] + double(label) * w, hi = lo + w;
      std::vector<double> px(blk.x, blk.x + blk.n), py(blk.y, blk.y + blk.n),
          pz(blk.z, blk.z + blk.n);
      for (int i = 0; i < blk.n; ++i) {
        double zw = pz[std::size_t(i)];
        if (s.periodic[2])
          zw -= s.box_len[2] * std::floor((zw - s.box_lo[2]) / s.box_len[2]);
        if (zw < lo || zw > hi) pz[std::size_t(i)] = s.box_lo[2] + s.box_len[2] + 1e6;
      }
      cudaMemcpy(L.x, px.data(), std::size_t(blk.n) * 8, cudaMemcpyHostToDevice);
      cudaMemcpy(L.y, py.data(), std::size_t(blk.n) * 8, cudaMemcpyHostToDevice);
      cudaMemcpy(L.z, pz.data(), std::size_t(blk.n) * 8, cudaMemcpyHostToDevice);
    } else {
      cudaMemcpy(L.x, blk.x, std::size_t(blk.n) * 8, cudaMemcpyHostToDevice);
      cudaMemcpy(L.y, blk.y, std::size_t(blk.n) * 8, cudaMemcpyHostToDevice);
      cudaMemcpy(L.z, blk.z, std::size_t(blk.n) * 8, cudaMemcpyHostToDevice);
    }
    cudaMemsetAsync(L.rho, 0, std::size_t(blk.n) * 8);  // defensive (see banner above)
    int* of = impl.ensure_of();
    using namespace eam_sn_detail;  // kB, ng
#ifdef TDMD_EAM_RING_TIMERS
    cudaEventRecord(s.ev_ds_);
#endif
    if (s.cull) {
      s.ensure_grid_geometry(3 * blk.n);  // AUTO hint frozen as 3·blk.n (design OQ3)
      s.grow(blk.n);  // M8: grid per-atom arrays must cover the slab BEFORE first compose
      EamCellGrid& g = s.grid_;
      cudaMemsetAsync(g.d_counts, 0, std::size_t(g.ncells) * sizeof(int));
      cell_count_kernel<<<ng(blk.n), kB>>>(L.x, L.y, L.z, blk.n, g.g, g.d_cell_of, g.d_counts);
      cub::DeviceScan::ExclusiveSum(g.d_cub, g.cub_bytes, g.d_counts, g.d_starts, g.ncells);
      cudaMemcpyAsync(g.d_cursor, g.d_starts, std::size_t(g.ncells) * sizeof(int),
                      cudaMemcpyDeviceToDevice);
      cell_scatter_kernel<<<ng(blk.n), kB>>>(g.d_cell_of, blk.n, g.d_cursor, g.d_order);
      CellGrid gp = g.g;  // stencil poison: launch-time copy, kernels stay clean (P-k)
      if (s.poison.stencil_override > 0)
        gp.sx = gp.sy = gp.sz = s.poison.stencil_override;
      eam_density_cells_kernel<<<ng(blk.n), kB>>>(L.x, L.y, L.z, blk.n, geom, s.view,
                                                  s.dens_scale, gp, g.d_starts, g.d_counts,
                                                  g.d_order, L.rho, of);
    } else {
      eam_density_kernel<<<ng(blk.n), kB>>>(L.x, L.y, L.z, blk.n, geom, s.view, s.dens_scale,
                                            L.rho, of);
    }
#ifdef TDMD_EAM_RING_TIMERS
    cudaEventRecord(s.ev_de_);
    cudaEventSynchronize(s.ev_de_);  // TIMERS-ONLY sync (amendment M3)
    float ms = 0; cudaEventElapsedTime(&ms, s.ev_ds_, s.ev_de_);
    s.t_donation_kernel_ms += ms; ++s.n_donation_launches;
#endif
    // NO sync, NO D2H — the donation result is consumed device-side (PR-3b) / by gates.
  }

  template <class DA>
  void on_edge(NodeState& ns, potentials::EamDonationState<DA>& /*st_host*/, int la, int lb,
               const potentials::ZoneBlockView& a, const potentials::ZoneBlockView& b,
               const core::PairGeom& geom) const {
    GpuEamWindowState& s = *st;
    if (!s.donate_device) return;
    if (DA::kScale != s.dens_scale)
      throw std::logic_error("GpuEamWindowForce: donation DA::kScale != policy dens_scale "
                             "(fb dispatch mismatch — the T-12/A3 trap)");
    std::lock_guard<std::mutex> lk(s.mu);
    ++s.donation_cross_batches_;  // S4: counts EXECUTED batches incl. vacuous
    auto& impl = *ns.impl;
    auto& LA = impl.zone.at(std::size_t(la));
    auto& LB = impl.zone.at(std::size_t(lb));
    // A7 stamp fence: CoRes(A,B) guarantees both selfs already executed this pass — a
    // stale/missing lane here is a label-keying or missed-arrival bug, deterministic HALT.
    if (LA.stamp != impl.token || LA.n != a.n || LB.stamp != impl.token || LB.n != b.n)
      throw std::logic_error("GpuEamWindowForce: cross batch on a lane not stamped this "
                             "pass (label " + std::to_string(la) + "," + std::to_string(lb) +
                             ") — stale zone lane");
    if (a.n == 0 || b.n == 0) return;  // vacuous: nothing to donate, ledger closes ring-side
    const int m = a.n + b.n;
    impl.concat_grow(m);
    // D2D concat [A|B] from the slab mirrors (already POST-drift on device).
    cudaMemcpyAsync(impl.cxx, LA.x, std::size_t(a.n) * 8, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(impl.cxy, LA.y, std::size_t(a.n) * 8, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(impl.cxz, LA.z, std::size_t(a.n) * 8, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(impl.cxx + a.n, LB.x, std::size_t(b.n) * 8, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(impl.cxy + a.n, LB.y, std::size_t(b.n) * 8, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(impl.cxz + a.n, LB.z, std::size_t(b.n) * 8, cudaMemcpyDeviceToDevice);
    // one_sided poison: route the B-side writes into a discard sink (ledger stays clean,
    // physics red — the FP64-oracle/A1 witness of the intra-batch half-bug class).
    long long* rho_b = s.poison.one_sided ? impl.discard_lane(b.n) : LB.rho;
    int* of = impl.ensure_of();
    using namespace eam_sn_detail;
#ifdef TDMD_EAM_RING_TIMERS
    cudaEventRecord(s.ev_ds_);
#endif
    if (s.cull) {
      s.ensure_grid_geometry(3 * m);
      s.grow(m);  // M8
      EamCellGrid& g = s.grid_;
      cudaMemsetAsync(g.d_counts, 0, std::size_t(g.ncells) * sizeof(int));
      cell_count_kernel<<<ng(m), kB>>>(impl.cxx, impl.cxy, impl.cxz, m, g.g, g.d_cell_of,
                                       g.d_counts);
      cub::DeviceScan::ExclusiveSum(g.d_cub, g.cub_bytes, g.d_counts, g.d_starts, g.ncells);
      cudaMemcpyAsync(g.d_cursor, g.d_starts, std::size_t(g.ncells) * sizeof(int),
                      cudaMemcpyDeviceToDevice);
      cell_scatter_kernel<<<ng(m), kB>>>(g.d_cell_of, m, g.d_cursor, g.d_order);
      CellGrid gp = g.g;
      if (s.poison.stencil_override > 0)
        gp.sx = gp.sy = gp.sz = s.poison.stencil_override;
      eam_donate_cross_cells_kernel<<<ng(m), kB>>>(impl.cxx, impl.cxy, impl.cxz, a.n, m,
                                                   geom, s.view, s.dens_scale, gp,
                                                   g.d_starts, g.d_counts, g.d_order,
                                                   LA.rho, rho_b, of);
    } else {
      eam_donate_cross_kernel<<<ng(m), kB>>>(impl.cxx, impl.cxy, impl.cxz, a.n, m, geom,
                                             s.view, s.dens_scale, LA.rho, rho_b, of);
    }
#ifdef TDMD_EAM_RING_TIMERS
    cudaEventRecord(s.ev_de_);
    cudaEventSynchronize(s.ev_de_);  // TIMERS-ONLY sync (amendment M3)
    float ms = 0; cudaEventElapsedTime(&ms, s.ev_ds_, s.ev_de_);
    s.t_donation_kernel_ms += ms; ++s.n_donation_launches;
#endif
  }

  template <class DA>
  void compose(NodeState& ns, const double* wx, const double* wy, const double* wz,
               const long* key, int m, const potentials::WindowBlocks& wb, const int* owned,
               int n_owned, const core::PairGeom& geom, double rho_cap,
               const DA* /*rho_w — host mirror, ignored; the GPU rho lives in the lanes*/,
               std::vector<core::fixed::ForceAccum>& wFx,
               std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz, core::fixed::EnergyAccum& pe,
               double& min_r2, int zone_j) const {
    GpuEamWindowState& s = *st;
    // A1/B-D2 capture seam (test-only): snapshot the <=3 window lane segments (blocking
    // D2H — waits for the queued donation kernels on the null stream) BEFORE the recompute.
    if (s.donate_device && s.capture_lanes && ns.impl) {
      std::lock_guard<std::mutex> lk(s.mu);
      EamLaneCapture rec;
      rec.pass = ns.impl->token;
      rec.zone_j = zone_j;
      rec.nb = wb.nb;
      for (int t = 0; t < wb.nb; ++t) {
        rec.label[t] = wb.label[t];
        const auto& L = ns.impl->zone.at(std::size_t(wb.label[t]));
        // capture fence (acceptance wf_1c8cf3af hygiene): a wb mis-build (slot-vs-label)
        // must die HERE as a clean logic_error, not cascade into a noisy downstream halt.
        if (L.stamp != ns.impl->token || L.n != wb.n[t])
          throw std::logic_error("GpuEamWindowForce: capture on unstamped/mismatched lane "
                                 "(label " + std::to_string(wb.label[t]) + ")");
        rec.lane[t].resize(std::size_t(wb.n[t]));
        if (wb.n[t] > 0 &&
            cudaMemcpy(rec.lane[t].data(), L.rho, std::size_t(wb.n[t]) * 8,
                       cudaMemcpyDeviceToHost) != cudaSuccess)
          throw std::runtime_error("GpuEamWindowForce: capture D2H failed");
      }
      s.capture_log.push_back(std::move(rec));
    }
    // PR-3a: compose RECOMPUTES density on device — verbatim today (knob-ON is purely
    // observational; the donated compose consuming the lanes is PR-3b, gated R_W>=1.15).
    compute(wx, wy, wz, key, m, owned, n_owned, geom, rho_cap, wFx, wFy, wFz, pe, min_r2);
  }
};

}  // namespace tdmd::cuda
