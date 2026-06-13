#pragma once
// M4 — GpuTimeConveyor: the TD ring on cudaStream_t (single-GPU emulation,
// ZoneFSM §8.3 mapping: node = stream, SEND/RECV = cudaMemcpyAsync D2D,
// INV-2 lag = cudaEvent_t record/wait). CUDA-only header.
//
// Division of labour. The PROTOCOL is the M3.5 CPU conveyor verbatim — host
// jthread per node, ZoneFSM orchestration (T1–T5, seed §7.1), §7.4 parity,
// positional Λ-chain Δt-handoff, rotation PBC closure, INV-4/Overlap/
// StaleZone/NonFinite guards. What moves to the device is the per-zone MATH:
// force pass (zone_force.cuh), drift/kick/locals (zone_integrate.cuh). All of
// it is bit-validated against the CPU expressions (Test_CUDA_Zones), so for a
// transcendental-free potential under --fmad=false the GPU conveyor must be
// BITWISE equal to the CPU conveyor — and trivially "1 stream vs N streams"
// bitwise (the deterministic_fp64 cell of the INV-9 matrix).
//
// StreamTransport (per ring edge k -> (k+1)%z):
//   * headers (ZoneHeader + atom count + dst slot) travel host-side through
//     the bounded SPSC channel — backpressure and HALT shutdown as in M3.5;
//   * payloads travel device-side: one packed cudaMemcpyAsync D2D from the
//     producer's slot into the consumer's slot, ordered by TWO event rings:
//       arrival[s]  recorded in the PRODUCER stream after the copy; the
//                   consumer does cudaStreamWaitEvent before touching slot s
//                   (INV-2: hardware-enforced "END+SEND happens-before RECV");
//       free[s]     recorded in the CONSUMER stream after its onward copy
//                   has READ slot s; the producer waits on it before
//                   overwriting (slot-reuse safety, INV-5/6).
//     Ordering invariant that makes cudaStreamWaitEvent sound: an event is
//     ALWAYS recorded before the header that announces it is pushed, and the
//     wait is issued only after the header is popped — a wait can never see
//     a never-recorded event (which CUDA treats as a no-op).
//   * per-zone END scalars (v_max²/a_max²/k2cap/KE/min_r2/flags) come back
//     through a small pinned staging block + stream sync — the host decides
//     INV-4/halts/Δt exactly like the CPU conveyor.
//
// Zone slot = one packed device allocation: 10 double arrays (x,y,z,v,f,
// mass) + 3 int64 raw accumulators, capacity = max zone size. Memory per
// node ~ (104 B + slack)·N_zone·S slots; the M4 10⁷-atom budget check is the
// bench PR's job. Static membership + guard as in M3.5 (migration — later).
#include <cuda_runtime.h>

#include <atomic>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// NVTX phase markers (M4 deferral; tech-stack item). Header-only NVTX3 from
// the toolkit; zero-cost no-ops unless TDMD_WITH_NVTX. Host-side ranges —
// nsys correlates the enqueued kernels/copies itself.
#ifdef TDMD_WITH_NVTX
#include <nvtx3/nvToolsExt.h>
#define TDMD_NVTX_PUSH(msg) nvtxRangePushA(msg)
#define TDMD_NVTX_POP() nvtxRangePop()
#else
#define TDMD_NVTX_PUSH(msg) ((void)0)
#define TDMD_NVTX_POP() ((void)0)
#endif

#include "tdmd/core/buffer.hpp"
#include "tdmd/core/conveyor.hpp"
#include "tdmd/core/fsm.hpp"
#include "tdmd/core/transport.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/zone_cells.cuh"
#include "tdmd/cuda/zone_force.cuh"
#include "tdmd/cuda/zone_integrate.cuh"
#include "tdmd/cuda/zone_verlet.cuh"

namespace tdmd::cuda {

#define TDMD_CU(call)                                                      \
  do {                                                                     \
    const cudaError_t e_ = (call);                                         \
    if (e_ != cudaSuccess)                                                 \
      throw std::runtime_error(std::string("CUDA: ") +                     \
                               cudaGetErrorString(e_) + " @ " #call);      \
  } while (0)

// --- small device helpers -------------------------------------------------

static __global__ void reset_zone_scalars_kernel(unsigned long long* v2,
                                                 unsigned long long* a2,
                                                 unsigned long long* kc) {
  *v2 = 0ULL;
  *a2 = 0ULL;
  *kc = 0x7FF0000000000000ULL;  // +inf bits
}

// [ENG] static-membership guard on device (conveyor membership_ok twin).
static __global__ void membership_guard_kernel(const double* z, int n,
                                               double lo, double hi, double g,
                                               double lo_box, double Lz,
                                               bool pbc, bool first, bool last,
                                               int* stale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double zw = z[i];
  if (pbc) zw -= Lz * floor((zw - lo_box) / Lz);
  double excess = 0.0;
  if (zw < lo) {
    excess = lo - zw;
    if (pbc)
      excess = fmin(excess, zw + Lz - hi);
    else if (first)
      excess = 0.0;
  } else if (zw > hi) {
    excess = zw - hi;
    if (pbc)
      excess = fmin(excess, lo + Lz - zw);
    else if (last)
      excess = 0.0;
  }
  if (excess > g) atomicOr(stale, 2);  // bit1: stale (bit0 = overflow)
}

// production_mixed transport (M4/B5 decision): coordinates/velocities/forces
// ship as int32 fixed-point offsets with POWER-OF-TWO quanta — pack is one
// rint per component per send (<= half-quantum loss, uniform across the box,
// 16-200x finer than FP32 offsets at zone scale), unpack is EXACT (int->
// double and 2^-k scaling are exact). Every zone is packed exactly once per
// pass regardless of the node count, so the snap sequence — and therefore
// the production_mixed trajectory — is bitwise z-INDEPENDENT (upgrade of the
// INV-9 matrix cell; see Rationale). Out-of-range components set flag bit2 ->
// Halt::Internal at the producer's pass end (detection latency <= 1 pass).

__device__ inline int snap_i32(double v, double inv_q, int* flags) {
  const double q = rint(v * inv_q);
  if (!(fabs(q) <= 2147483647.0)) {
    atomicOr(flags, 4);
    return 0;
  }
  return (int)q;
}

static __global__ void pack_zone_kernel(const double* base, int cap, int n,
                                        double inv_qp, double inv_qv,
                                        double inv_qf, double ox, double oy,
                                        double oz, double* pk_mass, int* pk,
                                        int* flags) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  pk[0 * cap + i] = snap_i32(base[0 * cap + i] - ox, inv_qp, flags);
  pk[1 * cap + i] = snap_i32(base[1 * cap + i] - oy, inv_qp, flags);
  pk[2 * cap + i] = snap_i32(base[2 * cap + i] - oz, inv_qp, flags);
  pk[3 * cap + i] = snap_i32(base[3 * cap + i], inv_qv, flags);
  pk[4 * cap + i] = snap_i32(base[4 * cap + i], inv_qv, flags);
  pk[5 * cap + i] = snap_i32(base[5 * cap + i], inv_qv, flags);
  pk[6 * cap + i] = snap_i32(base[6 * cap + i], inv_qf, flags);
  pk[7 * cap + i] = snap_i32(base[7 * cap + i], inv_qf, flags);
  pk[8 * cap + i] = snap_i32(base[8 * cap + i], inv_qf, flags);
  pk_mass[i] = base[9 * cap + i];
}

static __global__ void unpack_zone_kernel(const double* pk_mass,
                                          const int* pk, int cap, int n,
                                          double qp, double qv, double qf,
                                          double ox, double oy, double oz,
                                          double* base) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  base[0 * cap + i] = ox + double(pk[0 * cap + i]) * qp;  // exact
  base[1 * cap + i] = oy + double(pk[1 * cap + i]) * qp;
  base[2 * cap + i] = oz + double(pk[2 * cap + i]) * qp;
  base[3 * cap + i] = double(pk[3 * cap + i]) * qv;
  base[4 * cap + i] = double(pk[4 * cap + i]) * qv;
  base[5 * cap + i] = double(pk[5 * cap + i]) * qv;
  base[6 * cap + i] = double(pk[6 * cap + i]) * qf;
  base[7 * cap + i] = double(pk[7 * cap + i]) * qf;
  base[8 * cap + i] = double(pk[8 * cap + i]) * qf;
  base[9 * cap + i] = pk_mass[i];
}

// Scoped NVTX range (RAII — survives the early returns of run_pass).
struct NvtxScope {
  explicit NvtxScope(const char* msg) { TDMD_NVTX_PUSH(msg); }
  ~NvtxScope() { TDMD_NVTX_POP(); }
  NvtxScope(const NvtxScope&) = delete;
  NvtxScope& operator=(const NvtxScope&) = delete;
};

// --- StreamTransport ------------------------------------------------------

struct GpuHeader {
  core::ZoneHeader hdr;
  int n_atoms = 0;
  int dst_slot = 0;  // consumer slot the payload was copied into
};

// One ring edge: host header channel + device payload lane with the two
// event rings described above.
class StreamEdge {
 public:
  StreamEdge(int n_slots, std::size_t header_capacity)
      : chan_(header_capacity), credits_(std::size_t(n_slots)),
        arrival_(n_slots), free_(n_slots) {
    for (auto& e : arrival_) TDMD_CU(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    for (auto& e : free_) TDMD_CU(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  }
  ~StreamEdge() {
    for (auto& e : arrival_) cudaEventDestroy(e);
    for (auto& e : free_) cudaEventDestroy(e);
  }
  core::SpscChannelT<GpuHeader> chan_;
  // Free-slot credits, consumer -> producer. The consumer pushes slot id s
  // AFTER recording free_[s] in its stream; the producer pops BEFORE waiting
  // on free_[s] — so the wait always sees a recorded event (an unrecorded
  // event would make cudaStreamWaitEvent a no-op and allow copy-in to
  // overtake copy-out; under PBC rotation the head slot is forwarded LAST,
  // which a modular dst numbering would reuse too early — found by test).
  core::SpscChannelT<int> credits_;
  std::vector<cudaEvent_t> arrival_;
  std::vector<cudaEvent_t> free_;
};

class StreamTransport {
 public:
  StreamTransport(int n_nodes, int n_slots, std::size_t header_capacity) {
    edges_.reserve(n_nodes);
    for (int i = 0; i < n_nodes; ++i)
      edges_.push_back(std::make_unique<StreamEdge>(n_slots, header_capacity));
  }
  StreamEdge& edge(int e) { return *edges_[e]; }
  void shutdown() {
    for (auto& e : edges_) {
      e->chan_.shutdown();
      e->credits_.shutdown();
    }
  }

 private:
  std::vector<std::unique_ptr<StreamEdge>> edges_;
};

// --- M5a: process-boundary edge (the ring partitioned across MPI ranks) ---
//
// The ITransport contract fixes the memory space: payloads cross a process
// boundary as HOST buffers (the system OpenMPI 4.1.6 is not CUDA-aware —
// device pointers do not travel; host-staging D2H -> MPI -> H2D is the base
// path). The conveyor does the staging; the edge moves bytes. Wire format =
// exactly the intra-rank image (packed doubles, or the int32 image in
// production_mixed — the B5 traffic win applies to MPI hops too), so the
// transport is bit-exact and rank partitioning cannot change trajectories.
struct IBoundaryEdge {
  // Blocking; the payload buffer may be reused after return.
  virtual void send(const GpuHeader& h, const void* payload,
                    std::size_t bytes) = 0;
  // Blocking; false == ring shutdown (halt poison from another rank).
  virtual bool recv(GpuHeader& h, void* payload, std::size_t bytes) = 0;
  // Propagate a HALT across the boundary (poison message with a hop TTL).
  virtual void poison() = 0;
  virtual ~IBoundaryEdge() = default;
};

// This process's share of the global logical ring. Logical node count is
// z_ranks x k (M5a "k шагов на узел", Гл. 3.4/ур. 51: a zone makes k cheap
// intra-rank hops per one MPI hop — traffic drops k-fold, memory grows with
// the in-flight window, the trajectory is bitwise UNCHANGED because the
// protocol never depends on the node count).
struct RingPart {
  int z_global = 0;            // 0 => single-process (z_global = o.n_nodes)
  int node0 = 0;               // global index of this process's first node
  int z_local = 0;             // logical nodes hosted by this process
  IBoundaryEdge* in = nullptr;   // feeds local node 0
  IBoundaryEdge* out = nullptr;  // drains local node z_local-1
};

// --- per-node device storage ----------------------------------------------

// Packed zone slot: base[k*cap + i], k = 0..9 -> x y z vx vy vz fx fy fz mass.
struct DevSlot {
  double* base = nullptr;
  long long* raw = nullptr;  // [3*cap]: rfx rfy rfz
  // production_mixed packed wire image: [double mass × cap][int32 × 9 × cap]
  double* pk_mass = nullptr;
  int* pk = nullptr;
  // cell-list culling (zone_cells.cuh): [cell_of cap][order cap]
  // [counts NC][starts NC][cursor NC] — one allocation
  int* cells = nullptr;
  int cap = 0;
  int* cell_of() const { return cells; }
  int* order() const { return cells + cap; }
  int* counts(int nc) const { (void)nc; return cells + 2 * std::size_t(cap); }
  int* starts(int nc) const { return cells + 2 * std::size_t(cap) + nc; }
  int* cursor(int nc) const { return cells + 2 * std::size_t(cap) + 2 * std::size_t(nc); }
  double* arr(int k) const { return base + std::size_t(k) * cap; }
  long long* rfx() const { return raw; }
  long long* rfy() const { return raw + cap; }
  long long* rfz() const { return raw + 2 * std::size_t(cap); }
  static constexpr std::size_t kArrays = 10;
};

// Pinned END-scalar staging (one per node, reused after each sync).
struct EndScalars {
  unsigned long long v2, a2, kc, min_r2;
  long long ke, pe;
  int flags;  // bit0 overflow, bit1 stale
};

// --- the GPU conveyor -----------------------------------------------------

template <typename PairF>
class GpuTimeConveyor {
 public:
  GpuTimeConveyor(core::AtomSoA<double>& atoms, const core::Box& box,
                  double rcut, PairF pot, const core::ConveyorOptions& o,
                  RingPart part = {})
      : atoms_(atoms), box_(box), rcut_(rcut), pot_(pot), o_(o), part_(part),
        geom_(box, rcut), geom_list_(box, rcut + o.verlet_skin),
        grid_rcut_(o.verlet_reuse ? rcut + o.verlet_skin : rcut) {
    if (o_.steps < 1) throw std::invalid_argument("gpu conveyor: steps >= 1");
    if (o_.n_nodes < 1) throw std::invalid_argument("gpu conveyor: n_nodes >= 1");
    if (!(o_.dt_initial > 0.0))
      throw std::invalid_argument("gpu conveyor: dt_initial must be > 0");
    if (o_.verlet_hybrid && !o_.verlet_reuse)
      throw std::invalid_argument("gpu conveyor: verlet_hybrid needs verlet_reuse");
    if (o_.verlet_drift && !o_.verlet_hybrid)
      throw std::invalid_argument("gpu conveyor: verlet_drift needs verlet_hybrid");
    if (part_.z_global > 0 &&
        (part_.z_local < 1 || part_.node0 < 0 ||
         part_.node0 + part_.z_local > part_.z_global || !part_.in || !part_.out))
      throw std::invalid_argument("gpu conveyor: malformed RingPart");
  }

  core::ConveyorResult run() {
    zd_ = core::ZoneDecomposition::build(atoms_, box_, o_.n_zones, rcut_);
    n_ = zd_.n_zones;
    z_ = (part_.z_global > 0) ? part_.z_local : o_.n_nodes;
    Z_ = (part_.z_global > 0) ? part_.z_global : z_;
    node0_ = part_.node0;
    // INV-7 slot-pool sizing. The ring as a WHOLE holds all n zones at all
    // times, so Z·S >= n + slack is a LAW, not a tunable: a constant S
    // deadlocks once Z·S - n loses its slack (measured: z=2, n=16, S=8 —
    // 16 = n exactly — wedged with node 0 awaiting downstream credits and
    // node 1 awaiting node-0 slots hoarded by the preload reserve). This is
    // the dissertation's own '>= 6 зон на процессор' for 80 zones / 13
    // procs ~ n/z. Hence S = ceil(n/Z) + 4 (pipeline slack; the pass-1
    // liveness inequality n <= S + (S - W_res) then holds with margin),
    // floored by the per-node working window (~8: positions j, j+1, ENDed
    // pending, the held PBC head, deposits). A single-node ring holds
    // everything: S = n+2. P1's preload is STREAMED (lazy per-zone uploads
    // from atoms_) out of a reserved sub-pool of W_res = min(n, 6) slots.
    self_loop_ = (part_.z_global == 0 && z_ == 1);
    S_ = self_loop_
             ? n_ + 2
             : std::min(n_ + 2, std::max(8, (n_ + Z_ - 1) / Z_ + 4));
    pbc_z_ = box_.periodic[2];
    for (const auto& m : zd_.members) cap_ = std::max(cap_, int(m.size()));
    cap_ = std::max(cap_, 1);

    // t0 forces + Λ pre-history — identical to the CPU conveyor.
    core::zero_forces(atoms_);
    const double pe0 = o_.skip_t0_forces
                           ? 0.0  // bench-only: zero first kick (documented)
                           : core::zone_force_pass(atoms_, box_, zd_, rcut_, pot_);
    res_.e0 = pe0 + core::kinetic_energy(atoms_);
    lam0_ = {core::buffer::max_speed(atoms_), core::buffer::max_accel(atoms_),
             core::buffer::temperature_limited_dt(atoms_, o_.ts.K2)};
    if (o_.skip_t0_forces && !(lam0_.v > 0.0))
      throw std::invalid_argument(
          "skip_t0_forces needs a thermal start: with v=0 and a=0 the INV-4 "
          "forecast gives R_buf=0 and any motion HALTs (bench-only flag)");
    res_.stats.assign(std::size_t(o_.steps), {});
    final_.assign(std::size_t(n_), HostZone{});

    if (o_.mixed_transport) {
      // smallest power-of-two quantum covering the range in int32 (B5)
      auto pick_q = [](double range) {
        int e = 0;
        std::frexp(range, &e);          // range = f·2^e, f in [0.5, 1)
        return std::ldexp(1.0, e - 31);  // q·2^31 = 2^e >= range
      };
      const double L = std::max({box_.len(0), box_.len(1), box_.len(2)});
      qp_ = pick_q(L + 64.0);   // positions: box offsets + excursion pad
      qv_ = std::ldexp(1.0, -24);  // velocities: ±128 Å/ps
      qf_ = std::ldexp(1.0, -20);  // forces: ±2048 eV/Å
    }
    if (o_.verlet_reuse) {
      // The list halo must hold rcut+skin and the StaleZone slack g=(w-rcut-
      // skin)/2 must stay >= 0, so a multi-zone slab must be wider than
      // rcut+skin (SPEC §5.5). A persistent neighbour can drift up to skin/2
      // into the adjacent zone and still be binned.
      if (n_ > 1 && zd_.width <= rcut_ + o_.verlet_skin)
        throw std::invalid_argument(
            "verlet_reuse: zone width must exceed rcut + skin (SPEC §5.5)");
      // CSR stride bound: ~atoms within a (rcut+skin) sphere at the mean
      // density, x2 headroom + floor. A truncated atom trips the overflow flag
      // (Halt::Internal), never a silent miss. NB [ENG]: this flat per-atom
      // stride is generous for correctness at test scale; the 1e7 memory
      // strides (CSR capacity = cap*stride). A truncated atom trips the
      // overflow flag (Halt::Internal), never a silent miss.
      //   SELF: full (rcut+skin) sphere.
      //   NEXT/PREV (sparse cross-role): the three roles PARTITION the same
      //   neighbour sphere, so an atom's cross-partners in ONE adjacent zone
      //   are at most the HALF-sphere reaching across that face — half the
      //   storage, guaranteed-safe (the per-atom cap bounds it; no geometry
      //   estimate, no overflow path). Cuts the 1e7 list footprint ~2x.
      const double vol = box_.len(0) * box_.len(1) * box_.len(2);
      const double dens = vol > 0 ? double(atoms_.n) / vol : 0.0;
      const double rl = rcut_ + o_.verlet_skin;
      const double kPi = 3.14159265358979323846;
      const double sphere = (4.0 / 3.0) * kPi * rl * rl * rl * dens;
      max_neigh_ = std::max(64, int(2.0 * sphere) + 16);
      max_neigh_cross_ = std::max(32, int(2.0 * 0.5 * sphere) + 8);  // half-sphere
    }
    if (o_.cell_lists || o_.verlet_reuse) {
      const double len[3] = {box_.len(0), box_.len(1), box_.len(2)};
      const bool per[3] = {box_.periodic[0], box_.periodic[1], box_.periodic[2]};
      ncells_ = make_zone_grid(box_.lo.data(), len, per, grid_rcut_, n_, 0).ncells();
    }
    alloc_nodes();
    // Pool partition: ids [0, S-W) circulate as producer credits / the
    // boundary free list; ids [S-W, S) are node 0's pass-1 upload reserve
    // (drained into the main pool after pass 1). Self-loop: ONE pool.
    const bool owns0 = (node0_ == 0);
    const int W = self_loop_ ? S_ : (owns0 ? std::min(n_, 6) : 0);
    if (owns0)
      for (int s2 = S_ - W; s2 < S_; ++s2) p1free_.push_back(s2);
    transport_ = std::make_unique<StreamTransport>(z_, S_, std::size_t(n_) + 2);
    for (int k = 0; k < z_; ++k) {  // intra edge k feeds local node (k+1)%z
      if (part_.in && k == z_ - 1) continue;  // wrap edge replaced by boundary
      if (self_loop_) continue;               // single pool, no channel
      const int kn = (k + 1) % z_;
      const int hi = (kn == 0 && !part_.in && owns0) ? S_ - W : S_;
      for (int s2 = 0; s2 < hi; ++s2) transport_->edge(k).credits_.send(int(s2));
    }
    if (part_.in) {  // local node 0 manages its own slot pool + host staging
      for (int s2 = 0; s2 < S_ - W; ++s2) bfree_.push_back(s2);
      TDMD_CU(cudaHostAlloc(&in_stage_, wire_bytes(), 0));
      TDMD_CU(cudaHostAlloc(&out_stage_, wire_bytes(), 0));
      TDMD_CU(cudaEventCreateWithFlags(&in_evt_, cudaEventDisableTiming));
    }
    if (owns0) p1host_.resize(DevSlot::kArrays * std::size_t(cap_));

    {
      std::vector<std::jthread> nodes;
      nodes.reserve(std::size_t(z_));
      try {
        for (int k = 0; k < z_; ++k)
          nodes.emplace_back([this, k] { node_main(k); });
      } catch (const std::exception& ex) {
        set_halt(core::Halt::Internal,
                 std::string("gpu conveyor: node spawn failed: ") + ex.what());
      }
    }
    for (auto& nb : node_) TDMD_CU(cudaStreamSynchronize(nb.stream));

    if (halt_on_.load()) {
      res_.halt = halt_kind_;
      res_.halt_msg = halt_msg_;
      long done = 0;
      while (done < o_.steps && res_.stats[std::size_t(done)].dt > 0.0) ++done;
      res_.steps_done = done;
      res_.stats.resize(std::size_t(done));
    } else {
      res_.steps_done = o_.steps;
      if (final_seen_.load()) {  // multi-rank: one rank owns the final pass
        scatter_final();
        res_.has_final = true;
      }
    }
    res_.verlet_rebuilds = verlet_rebuilds_.load();
    if (part_.in) {
      cudaFreeHost(in_stage_);
      cudaFreeHost(out_stage_);
      cudaEventDestroy(in_evt_);
    }
    free_nodes();
    return res_;
  }

 private:
  using Lambda = core::conveyor_detail::Lambda;

  struct HostZone {  // final-pass payload landed back on the host
    std::vector<double> a;  // packed, 10*cap
    int n = 0;
  };

  struct NodeBuf {
    cudaStream_t stream{};
    std::vector<DevSlot> slot;
    // per-zone / per-pass device scalars
    unsigned long long *d_v2{}, *d_a2{}, *d_kc{}, *d_mr2{};
    long long *d_ke{}, *d_pe{};
    int* d_flags{};
    EndScalars* h_end{};  // pinned, n_zones entries — per-position END
                          // snapshots for the ONCE-PER-PASS batched sync
    void* d_cubtmp = nullptr;  // cell-list / verlet scan temp
    std::size_t cubtmp_bytes = 0;
    // PR-1 Verlet reuse scratch (ephemeral per launch at K=1, one stream so
    // reuse across launches is stream-ordered-safe): CSR build for the current
    // (A,B). v_idx capacity = cap*max_neigh_.
    int *v_counts = nullptr, *v_offsets = nullptr, *v_idx = nullptr;
    // PR-3/PR-4 hybrid criterion: per-pass max sq displacement + (PR-4) Q24.40
    // drift sums (reset per pass, accumulated across zones, read after sync).
    unsigned long long* d_d2 = nullptr;
    long long *d_sx = nullptr, *d_sy = nullptr, *d_sz = nullptr;
  };

  // Host-side per-pass slot bookkeeping (the CPU conveyor Slot minus payload).
  struct PassSlot {
    core::Zone fsm{};
    Lambda lam_in{};
    int dev = -1;  // device slot index
    int n_atoms = 0;
    bool present = false, drifted = false, computed = false;
    double v_max = 0, a_max = 0, k2cap = 0;
    long long ke_raw = 0;
  };

  void alloc_nodes() {
    node_.resize(std::size_t(z_));
    for (auto& nb : node_) {
      TDMD_CU(cudaStreamCreate(&nb.stream));
      nb.slot.resize(std::size_t(S_));
      for (auto& s : nb.slot) {
        s.cap = cap_;
        TDMD_CU(cudaMalloc(&s.base, DevSlot::kArrays * sizeof(double) * cap_));
        TDMD_CU(cudaMalloc(&s.raw, 3 * sizeof(long long) * cap_));
        if (o_.mixed_transport) {
          char* pkb = nullptr;
          TDMD_CU(cudaMalloc(&pkb, packed_bytes()));
          s.pk_mass = reinterpret_cast<double*>(pkb);
          s.pk = reinterpret_cast<int*>(pkb + sizeof(double) * cap_);
        }
        if (o_.cell_lists || o_.verlet_reuse)
          TDMD_CU(cudaMalloc(&s.cells,
                             sizeof(int) * (2 * std::size_t(cap_) +
                                            3 * std::size_t(ncells_))));
      }
      if (o_.cell_lists || o_.verlet_reuse) {
        // CUB temp for the per-zone exclusive scan: cells scan over ncells_,
        // the verlet CSR scan over cap_+1 — size for the larger.
        nb.cubtmp_bytes = 0;
        std::size_t b1 = 0, b2 = 0;
        TDMD_CU(cub::DeviceScan::ExclusiveSum(nullptr, b1, (int*)nullptr,
                                              (int*)nullptr, ncells_));
        TDMD_CU(cub::DeviceScan::ExclusiveSum(nullptr, b2, (int*)nullptr,
                                              (int*)nullptr, cap_ + 1));
        nb.cubtmp_bytes = std::max(b1, b2);
        TDMD_CU(cudaMalloc(&nb.d_cubtmp, nb.cubtmp_bytes));
      }
      if (o_.verlet_reuse) {
        TDMD_CU(cudaMalloc(&nb.v_counts, sizeof(int) * (cap_ + 1)));
        TDMD_CU(cudaMalloc(&nb.v_offsets, sizeof(int) * (cap_ + 1)));
        TDMD_CU(cudaMalloc(&nb.v_idx,
                           sizeof(int) * std::size_t(cap_) * max_neigh_));
      }
      TDMD_CU(cudaMalloc(&nb.d_v2, 8));
      TDMD_CU(cudaMalloc(&nb.d_a2, 8));
      TDMD_CU(cudaMalloc(&nb.d_kc, 8));
      TDMD_CU(cudaMalloc(&nb.d_mr2, 8));
      TDMD_CU(cudaMalloc(&nb.d_ke, 8));
      TDMD_CU(cudaMalloc(&nb.d_pe, 8));
      TDMD_CU(cudaMalloc(&nb.d_flags, 4));
      TDMD_CU(cudaHostAlloc(&nb.h_end, sizeof(EndScalars) * std::size_t(n_), 0));
      if (o_.verlet_hybrid) {  // PR-3/PR-4 per-pass displacement aggregates
        TDMD_CU(cudaMalloc(&nb.d_d2, 8));
        TDMD_CU(cudaMalloc(&nb.d_sx, 8));
        TDMD_CU(cudaMalloc(&nb.d_sy, 8));
        TDMD_CU(cudaMalloc(&nb.d_sz, 8));
      }
    }
    if (o_.verlet_reuse) {  // shared per-(zone,role) persistent lists
      vl_off_.assign(std::size_t(n_) * kRoles, nullptr);
      vl_idx_.assign(std::size_t(n_) * kRoles, nullptr);
      for (std::size_t s = 0; s < vl_off_.size(); ++s) {
        const int role = int(s % kRoles);             // 0=SELF, 1/2=cross
        const int stride = role == 0 ? max_neigh_ : max_neigh_cross_;
        TDMD_CU(cudaMalloc(&vl_off_[s], sizeof(int) * (cap_ + 1)));
        TDMD_CU(cudaMalloc(&vl_idx_[s],
                           sizeof(int) * std::size_t(cap_) * stride));
      }
      if (o_.verlet_hybrid) {  // PR-3 rebuild-epoch positions x_ref per zone
        vl_xref_.assign(std::size_t(n_), nullptr);
        for (auto& p : vl_xref_)
          TDMD_CU(cudaMalloc(&p, sizeof(double) * 3 * std::size_t(cap_)));
      }
    }
  }

  void free_nodes() {
    for (auto* p : vl_off_) cudaFree(p);
    for (auto* p : vl_idx_) cudaFree(p);
    for (auto* p : vl_xref_) cudaFree(p);
    vl_off_.clear();
    vl_idx_.clear();
    vl_xref_.clear();
    for (auto& nb : node_) {
      for (auto& s : nb.slot) {
        cudaFree(s.base);
        cudaFree(s.raw);
        cudaFree(s.pk_mass);  // base pointer of the packed image (or null)
        cudaFree(s.cells);
      }
      cudaFree(nb.d_cubtmp);
      cudaFree(nb.v_counts);
      cudaFree(nb.v_offsets);
      cudaFree(nb.v_idx);
      cudaFree(nb.d_d2);
      cudaFree(nb.d_sx);
      cudaFree(nb.d_sy);
      cudaFree(nb.d_sz);
      for (auto* p : {nb.d_v2, nb.d_a2, nb.d_kc, nb.d_mr2}) cudaFree(p);
      cudaFree(nb.d_ke);
      cudaFree(nb.d_pe);
      cudaFree(nb.d_flags);
      cudaFreeHost(nb.h_end);
      cudaStreamDestroy(nb.stream);
    }
    node_.clear();
  }

  // §7.3 streamed preload: upload zone `zone_id`'s t0 state into the given
  // slot ON DEMAND (gathered from atoms_ — no bulk host image). Synchronous
  // memcpy on the legacy default stream: ordered after any enqueued reads of
  // a recycled slot, and the gather buffer is reused safely.
  void upload_zone(int dev_slot, int zone_id) {
    const auto& mem = zd_.members[std::size_t(zone_id)];
    std::fill(p1host_.begin(), p1host_.end(), 0.0);
    for (std::size_t t = 0; t < mem.size(); ++t) {
      const int i = mem[t];
      p1host_[0 * cap_ + t] = atoms_.x[i];
      p1host_[1 * cap_ + t] = atoms_.y[i];
      p1host_[2 * cap_ + t] = atoms_.z[i];
      p1host_[3 * cap_ + t] = atoms_.vx[i];
      p1host_[4 * cap_ + t] = atoms_.vy[i];
      p1host_[5 * cap_ + t] = atoms_.vz[i];
      p1host_[6 * cap_ + t] = atoms_.fx[i];
      p1host_[7 * cap_ + t] = atoms_.fy[i];
      p1host_[8 * cap_ + t] = atoms_.fz[i];
      p1host_[9 * cap_ + t] = atoms_.mass[i];
    }
    TDMD_CU(cudaMemcpy(node_[0].slot[std::size_t(dev_slot)].base,
                       p1host_.data(), DevSlot::kArrays * sizeof(double) * cap_,
                       cudaMemcpyHostToDevice));
  }

  void node_main(int k) {
    try {
      for (long h = node0_ + k + 1; h <= o_.steps; h += Z_) {
        if (halt_on_.load(std::memory_order_acquire)) return;
        if (!run_pass(k, h)) return;
      }
    } catch (const std::exception& ex) {
      set_halt(core::Halt::Internal,
               std::string("gpu conveyor internal error: ") + ex.what());
    }
  }

  bool run_pass(int k, long h) {
    char nvtx_name[48];
    std::snprintf(nvtx_name, sizeof(nvtx_name), "pass %ld @node %ld", h,
                  long(node0_ + k));
    NvtxScope nvtx_pass(nvtx_name);
    NodeBuf& nb = node_[std::size_t(k)];
    StreamEdge& in = transport_->edge((k - 1 + z_) % z_);
    StreamEdge& out = transport_->edge(k);
    const bool b_in = (part_.in && k == 0);            // boundary in-edge
    const bool b_out = (part_.out && k == z_ - 1);     // boundary out-edge
    const auto io = core::node_io_order(int(node0_ + k) + 1);  // §7.4 parity
    const int r = pbc_z_ ? int((h - 1) % n_) : 0;     // rotation
    const bool defer_head = pbc_z_ && n_ > 1;
    const bool preload = (h == 1);

    std::vector<PassSlot> slot(static_cast<std::size_t>(n_));
    std::deque<int> outq;
    std::vector<int> ended;  // END order (checks replay in this order)
    bool synced = false;
    int sent = 0, arrived = 0;
    double dt = 0.0, R_buf = 0.0, dt_next = 0.0;
    // PR-1b-ii Verlet skin budget (per-pass, carried in the header like dt):
    // skin_in arrives at pos 0; the head decides the NEXT pass's rebuild and
    // broadcasts skin_out/rebuild_next into EVERY header. rebuild_now_pass is
    // THIS pass's broadcast decision (uniform across zones).
    double skin_in = 0.0, skin_out = 0.0;
    bool rebuild_now_pass = true, rebuild_next = false;
    uint8_t verlet_active_pass = 0, va_next = 0;  // PR-2: K-aware fallback mode
    Lambda agg{0.0, 0.0, std::numeric_limits<double>::infinity()};
    double ke = 0.0;

    // per-pass device scalar reset (min_r2 = +inf, pe = 0, flags = 0)
    TDMD_CU(cudaMemcpyAsync(nb.d_mr2, &inf_bits_, 8, cudaMemcpyHostToDevice,
                            nb.stream));
    TDMD_CU(cudaMemsetAsync(nb.d_pe, 0, 8, nb.stream));
    TDMD_CU(cudaMemsetAsync(nb.d_flags, 0, 4, nb.stream));
    if (o_.verlet_hybrid) {  // PR-3/PR-4 per-pass displacement aggregates -> 0
      TDMD_CU(cudaMemsetAsync(nb.d_d2, 0, 8, nb.stream));
      TDMD_CU(cudaMemsetAsync(nb.d_sx, 0, 8, nb.stream));
      TDMD_CU(cudaMemsetAsync(nb.d_sy, 0, 8, nb.stream));
      TDMD_CU(cudaMemsetAsync(nb.d_sz, 0, 8, nb.stream));
    }

    auto fail = [&](core::Halt kind, const std::string& msg) {
      set_halt(kind, msg);
      return false;
    };

    auto ensure_arrival = [&](int j) -> bool {
      while (arrived <= j) {
        PassSlot& s = slot[std::size_t(arrived)];
        const int want_id = (r + arrived) % n_;
        if (preload) {
          if (p1free_.empty())
            throw std::logic_error("gpu conveyor: preload reserve exhausted");
          s.dev = p1free_.front();
          p1free_.pop_front();
          upload_zone(s.dev, want_id);  // streamed §7.3 (INV-7 pool)
          s.n_atoms = int(zd_.members[std::size_t(want_id)].size());
          s.lam_in = lam0_;
          if (arrived == 0) {
            dt = o_.dt_initial;
            skin_in = 0.0;          // cold start
            verlet_active_pass = o_.verlet_default ? 1 : 0;  // PR-2 default mode
            rebuild_now_pass = verlet_active_pass != 0;      // build iff active
          }
          s.fsm.type = core::initial_zone_type(1);
        } else if (b_in) {
          // M5a boundary: header+payload arrive as host bytes; this node owns
          // its slot pool locally (no cross-rank credits). Staging reuse is
          // gated by the previous H2D's event (never-recorded => no-op).
          TDMD_CU(cudaEventSynchronize(in_evt_));
          GpuHeader gh;
          if (!part_.in->recv(gh, in_stage_, wire_bytes())) {
            // poison from another rank: report honestly (idempotent CAS — a
            // locally-originated halt keeps its own kind/message)
            set_halt(core::Halt::Internal,
                     "halt propagated across the ring boundary");
            return false;
          }
          if (gh.hdr.zone_id != want_id || gh.hdr.step_h != h - 1 ||
              gh.hdr.sent_pos != arrived)
            throw std::logic_error("gpu conveyor: boundary arrival out of order");
          if (bfree_.empty())
            throw std::logic_error("gpu conveyor: boundary slot pool exhausted");
          s.dev = bfree_.front();
          bfree_.pop_front();
          s.n_atoms = gh.n_atoms;
          s.lam_in = {gh.hdr.v_full, gh.hdr.a_full, gh.hdr.k2cap_full,
                      gh.hdr.d_full,
                      {gh.hdr.drift_full[0], gh.hdr.drift_full[1],
                       gh.hdr.drift_full[2]}};
          if (arrived == 0) {
            dt = gh.hdr.dt_next;
            skin_in = gh.hdr.skin_consumed;            // carried budget
            rebuild_now_pass = gh.hdr.rebuild_now;      // broadcast decisions
            verlet_active_pass = gh.hdr.verlet_active;  // (uniform per pass)
          }
          const DevSlot& d = nb.slot[std::size_t(s.dev)];
          if (o_.mixed_transport) {
            TDMD_CU(cudaMemcpyAsync(d.pk_mass, in_stage_, wire_bytes(),
                                    cudaMemcpyHostToDevice, nb.stream));
            if (s.n_atoms > 0) {
              const int grid = (s.n_atoms + kIntBlock - 1) / kIntBlock;
              unpack_zone_kernel<<<grid, kIntBlock, 0, nb.stream>>>(
                  d.pk_mass, d.pk, cap_, s.n_atoms, qp_, qv_, qf_, box_.lo[0],
                  box_.lo[1], box_.lo[2], d.base);
              TDMD_CU(cudaGetLastError());
            }
          } else {
            TDMD_CU(cudaMemcpyAsync(d.base, in_stage_, wire_bytes(),
                                    cudaMemcpyHostToDevice, nb.stream));
          }
          TDMD_CU(cudaEventRecord(in_evt_, nb.stream));
          s.fsm.type = core::ZoneType::o;
          core::ZoneFSM::apply(s.fsm, core::ZoneEvent::RECV);  // T1
        } else {
          GpuHeader gh;
          if (!in.chan_.recv(gh)) return false;  // shutdown
          if (gh.hdr.zone_id != want_id || gh.hdr.step_h != h - 1 ||
              gh.hdr.sent_pos != arrived)
            throw std::logic_error("gpu conveyor: ring arrival out of order");
          s.dev = gh.dst_slot;
          s.n_atoms = gh.n_atoms;
          s.lam_in = {gh.hdr.v_full, gh.hdr.a_full, gh.hdr.k2cap_full,
                      gh.hdr.d_full,
                      {gh.hdr.drift_full[0], gh.hdr.drift_full[1],
                       gh.hdr.drift_full[2]}};
          if (arrived == 0) {
            dt = gh.hdr.dt_next;
            skin_in = gh.hdr.skin_consumed;            // carried budget
            rebuild_now_pass = gh.hdr.rebuild_now;      // broadcast decisions
            verlet_active_pass = gh.hdr.verlet_active;  // (uniform per pass)
          }
          // INV-2: the payload copy is ordered before this wait by the
          // record-before-push / pop-before-wait protocol invariant.
          TDMD_CU(cudaStreamWaitEvent(nb.stream, in.arrival_[std::size_t(s.dev)], 0));
          if (o_.mixed_transport && s.n_atoms > 0) {
            const DevSlot& d = nb.slot[std::size_t(s.dev)];
            const int grid = (s.n_atoms + kIntBlock - 1) / kIntBlock;
            unpack_zone_kernel<<<grid, kIntBlock, 0, nb.stream>>>(
                d.pk_mass, d.pk, cap_, s.n_atoms, qp_, qv_, qf_, box_.lo[0],
                box_.lo[1], box_.lo[2], d.base);
            TDMD_CU(cudaGetLastError());
          }
          s.fsm.type = core::ZoneType::o;
          core::ZoneFSM::apply(s.fsm, core::ZoneEvent::RECV);  // T1
        }
        s.fsm.id = want_id;
        s.fsm.step_h = h;
        s.fsm.n_atoms = s.n_atoms;
        s.present = true;
        if (arrived == std::min(1, n_ - 1)) {
          const Lambda& lf = slot[std::size_t(arrived)].lam_in;
          const double lag = (n_ == 1) ? 1.0 : double(n_ - 1);
          const double v_pred = lf.v + lf.a * dt * lag;
          R_buf = core::buffer::compute_R_buf(v_pred, dt, o_.ts.C_buf);
        }
        ++arrived;
      }
      return true;
    };

    auto ensure_drift = [&](int j) {
      PassSlot& s = slot[std::size_t(j)];
      if (s.drifted) return;
      NvtxScope nv("drift+cells");
      const DevSlot& d = nb.slot[std::size_t(s.dev)];
      if (s.n_atoms > 0) {
        const int grid = (s.n_atoms + kIntBlock - 1) / kIntBlock;
        zone_drift_kernel<<<grid, kIntBlock, 0, nb.stream>>>(
            d.arr(0), d.arr(1), d.arr(2), d.arr(3), d.arr(4), d.arr(5),
            d.arr(6), d.arr(7), d.arr(8), d.arr(9), s.n_atoms, dt);
        TDMD_CU(cudaGetLastError());
        // PR-3: on a rebuild pass snapshot the new epoch x_ref = x_h (post-
        // drift), so this pass measures d=0 and reuse passes measure growth.
        if (o_.verlet_hybrid && rebuild_now_pass) {
          double* xr = vl_xref_[std::size_t(s.fsm.id)];
          xref_copy_kernel<<<grid, kIntBlock, 0, nb.stream>>>(
              d.arr(0), d.arr(1), d.arr(2), xr, xr + cap_, xr + 2 * cap_,
              s.n_atoms);
          TDMD_CU(cudaGetLastError());
        }
      }
      TDMD_CU(cudaMemsetAsync(d.raw, 0, 3 * sizeof(long long) * cap_,
                              nb.stream));
      // verlet_reuse + active: build cells only on a rebuild pass (the
      // materialisation needs them); a reuse pass skips the build — the
      // amortisation win. Fallback (verlet_active=0, cell-raster path) and the
      // plain cell path build every pass.
      const bool build_cells =
          o_.verlet_reuse
              ? (verlet_active_pass ? rebuild_now_pass : true)
              : o_.cell_lists;
      if (build_cells) {  // rebuild cell list at x_h
        TDMD_CU(cudaMemsetAsync(d.counts(ncells_), 0, sizeof(int) * ncells_,
                                nb.stream));
        if (s.n_atoms > 0) {
          const CellGrid g = zone_grid(s.fsm.id);
          const int grid = (s.n_atoms + kIntBlock - 1) / kIntBlock;
          cell_count_kernel<<<grid, kIntBlock, 0, nb.stream>>>(
              d.arr(0), d.arr(1), d.arr(2), s.n_atoms, g, d.cell_of(),
              d.counts(ncells_));
          TDMD_CU(cudaGetLastError());
        }
        std::size_t tb = nb.cubtmp_bytes;
        TDMD_CU(cub::DeviceScan::ExclusiveSum(nb.d_cubtmp, tb,
                                              d.counts(ncells_),
                                              d.starts(ncells_), ncells_,
                                              nb.stream));
        TDMD_CU(cudaMemcpyAsync(d.cursor(ncells_), d.starts(ncells_),
                                sizeof(int) * ncells_,
                                cudaMemcpyDeviceToDevice, nb.stream));
        if (s.n_atoms > 0) {
          const int grid = (s.n_atoms + kIntBlock - 1) / kIntBlock;
          cell_scatter_kernel<<<grid, kIntBlock, 0, nb.stream>>>(
              d.cell_of(), s.n_atoms, d.cursor(ncells_), d.order());
          TDMD_CU(cudaGetLastError());
        }
      }
      s.drifted = true;
    };

    auto launch_pairs = [&](PassSlot& A, PassSlot& B, bool same, bool energy,
                            int role) {
      if (A.n_atoms == 0) return;
      const DevSlot& da = nb.slot[std::size_t(A.dev)];
      const DevSlot& db = nb.slot[std::size_t(B.dev)];
      ZoneForceArgs args{da.arr(0), da.arr(1), da.arr(2), A.n_atoms,
                         db.arr(0), db.arr(1), db.arr(2), B.n_atoms,
                         da.rfx(),  da.rfy(),  da.rfz(),  nb.d_pe,
                         nb.d_mr2,  nb.d_flags, same,     energy};
      const int grid = (A.n_atoms + kZoneBlock - 1) / kZoneBlock;
      if (o_.verlet_reuse && verlet_active_pass) {
        // PR-1b-ii: persistent (zone,role) list. On a rebuild pass materialise
        // from B's cells at rcut+skin into the shared buffer; on a reuse pass
        // skip the build and force from the stored list. The verlet force
        // kernel re-tests at geom_ (rcut): bitwise ≡ the cell path while the
        // list is a superset (guaranteed by the 2*R_buf skin budget). The
        // shared write/read is ordered by the existing arrival-event chain;
        // the stored LOCAL indices are slot-independent (static membership).
        const std::size_t vk = std::size_t(A.fsm.id) * kRoles + role;
        int* off = vl_off_[vk];
        int* idx = vl_idx_[vk];
        const int stride = role == 0 ? max_neigh_ : max_neigh_cross_;
        const int gc = (A.n_atoms + kVerletBlock - 1) / kVerletBlock;
        if (rebuild_now_pass) {
          const CellGrid bgrid = zone_grid(B.fsm.id);
          TDMD_CU(cudaMemsetAsync(nb.v_counts, 0, sizeof(int) * (A.n_atoms + 1),
                                  nb.stream));
          verlet_count_kernel<<<gc, kVerletBlock, 0, nb.stream>>>(
              da.arr(0), da.arr(1), da.arr(2), A.n_atoms, db.arr(0), db.arr(1),
              db.arr(2), geom_list_, bgrid, db.starts(ncells_),
              db.counts(ncells_), db.order(), same, stride, nb.d_flags,
              nb.v_counts);
          std::size_t tb = nb.cubtmp_bytes;
          TDMD_CU(cub::DeviceScan::ExclusiveSum(nb.d_cubtmp, tb, nb.v_counts,
                                                off, A.n_atoms + 1, nb.stream));
          verlet_fill_kernel<<<gc, kVerletBlock, 0, nb.stream>>>(
              da.arr(0), da.arr(1), da.arr(2), A.n_atoms, db.arr(0), db.arr(1),
              db.arr(2), geom_list_, bgrid, db.starts(ncells_),
              db.counts(ncells_), db.order(), same, stride, off, idx);
        }
        zone_pair_verlet_kernel<PairF><<<gc, kVerletBlock, 0, nb.stream>>>(
            args, geom_, pot_, off, idx);
      } else if (o_.cell_lists || o_.verlet_reuse) {  // cell path — also the
        // PR-2 fallback (verlet_active=0); bitwise ≡ tiles (B1)
        const CellGrid bgrid = zone_grid(B.fsm.id);
        zone_pair_cells_kernel<PairF><<<grid, kZoneBlock, 0, nb.stream>>>(
            args, geom_, pot_, bgrid, db.starts(ncells_), db.counts(ncells_),
            db.order());
      } else {
        zone_pair_kernel<PairF><<<grid, kZoneBlock, 0, nb.stream>>>(args, geom_,
                                                                    pot_);
      }
      TDMD_CU(cudaGetLastError());
    };

    // T4 enqueue: kernels + an ASYNC snapshot of the END scalars into
    // h_end[j]. NO host sync here — the per-zone D2H+sync was the ring's
    // dominant overhead (NVTX: end+sync 32.6%). The guards (Overlap/
    // overflow/StaleZone/INV-4) replay at sync_check(), ONCE per pass,
    // before the LAST send: identical data, identical order, identical
    // first failure — and a failed pass never emits its last zone, so
    // downstream can never complete pass h+1 and the stats-prefix contract
    // holds without extra bookkeeping.
    auto end_zone = [&](int j) -> bool {
      NvtxScope nv("end-enqueue");
      PassSlot& s = slot[std::size_t(j)];
      const DevSlot& d = nb.slot[std::size_t(s.dev)];
      const uint32_t want_mask = (j == 0) ? (defer_head ? 2u : 0u) : 1u;
      if (s.fsm.contrib_mask != want_mask)
        throw std::logic_error("gpu conveyor: INV-3 mask incomplete at END");
      reset_zone_scalars_kernel<<<1, 1, 0, nb.stream>>>(nb.d_v2, nb.d_a2,
                                                        nb.d_kc);
      TDMD_CU(cudaMemsetAsync(nb.d_ke, 0, 8, nb.stream));
      if (s.n_atoms > 0) {
        const int grid = (s.n_atoms + kIntBlock - 1) / kIntBlock;
        zone_end_kernel<<<grid, kIntBlock, 0, nb.stream>>>(
            d.arr(3), d.arr(4), d.arr(5), d.arr(6), d.arr(7), d.arr(8),
            d.rfx(), d.rfy(), d.rfz(), d.arr(9), s.n_atoms, dt, o_.ts.K2,
            nb.d_v2, nb.d_a2, nb.d_kc, nb.d_ke, nb.d_flags);
        TDMD_CU(cudaGetLastError());
        if (n_ >= 3) {
          const double w = zd_.width, g = 0.5 * (w - rcut_);
          const double lo = box_.lo[2] + s.fsm.id * w;
          membership_guard_kernel<<<grid, kIntBlock, 0, nb.stream>>>(
              d.arr(2), s.n_atoms, lo, lo + w, g, box_.lo[2], box_.len(2),
              pbc_z_, s.fsm.id == 0, s.fsm.id == n_ - 1,
              nb.d_flags);  // bit1
        }
        if (o_.verlet_hybrid) {  // PR-3: max ||x_h - x_ref - D0|| into pass d_d2
          double* xr = vl_xref_[std::size_t(s.fsm.id)];
          const int gd = (s.n_atoms + kVerletBlock - 1) / kVerletBlock;
          // PR-4 (Theorem 1): subtract the lagged mean drift D0 (same value for
          // EVERY atom of the pass — the head's lagged aggregate — so it cancels
          // in pair differences; ANY D0 is safe). 0 when verlet_drift is off.
          double d0x = 0, d0y = 0, d0z = 0;
          if (o_.verlet_drift) {
            const Lambda& L0 = slot[0].lam_in;
            d0x = L0.drift[0]; d0y = L0.drift[1]; d0z = L0.drift[2];
          }
          zone_dmax_kernel<<<gd, kVerletBlock, 0, nb.stream>>>(
              d.arr(0), d.arr(1), d.arr(2), xr, xr + cap_, xr + 2 * cap_,
              s.n_atoms, d0x, d0y, d0z, nb.d_d2);
          TDMD_CU(cudaGetLastError());
          if (o_.verlet_drift)  // this pass's drift D0 = (Σ displacement)/N
            zone_drift_sum_kernel<<<gd, kVerletBlock, 0, nb.stream>>>(
                d.arr(0), d.arr(1), d.arr(2), xr, xr + cap_, xr + 2 * cap_,
                s.n_atoms, nb.d_sx, nb.d_sy, nb.d_sz, nb.d_flags);
        }
      }
      EndScalars* hs = &nb.h_end[std::size_t(j)];
      TDMD_CU(cudaMemcpyAsync(&hs->v2, nb.d_v2, 8, cudaMemcpyDeviceToHost, nb.stream));
      TDMD_CU(cudaMemcpyAsync(&hs->a2, nb.d_a2, 8, cudaMemcpyDeviceToHost, nb.stream));
      TDMD_CU(cudaMemcpyAsync(&hs->kc, nb.d_kc, 8, cudaMemcpyDeviceToHost, nb.stream));
      TDMD_CU(cudaMemcpyAsync(&hs->min_r2, nb.d_mr2, 8, cudaMemcpyDeviceToHost, nb.stream));
      TDMD_CU(cudaMemcpyAsync(&hs->ke, nb.d_ke, 8, cudaMemcpyDeviceToHost, nb.stream));
      TDMD_CU(cudaMemcpyAsync(&hs->pe, nb.d_pe, 8, cudaMemcpyDeviceToHost, nb.stream));
      TDMD_CU(cudaMemcpyAsync(&hs->flags, nb.d_flags, 4, cudaMemcpyDeviceToHost, nb.stream));
      ended.push_back(j);
      core::ZoneFSM::apply(s.fsm, core::ZoneEvent::END);  // T4
      outq.push_back(j);
      return true;
    };

    // The batched END sync: ONE cudaStreamSynchronize per pass, guards
    // replayed over the snapshots in END order — same data, same order,
    // same first failure as the old per-zone path; bitwise-identical agg,
    // ke and dt decisions.
    auto sync_check = [&]() -> bool {
      if (synced) return true;
      NvtxScope nv("end+sync");
      TDMD_CU(cudaStreamSynchronize(nb.stream));
      synced = true;
      for (int q : ended) {
        PassSlot& s = slot[std::size_t(q)];
        const EndScalars& e = nb.h_end[std::size_t(q)];
        double mr2, v2, a2, kc;
        std::memcpy(&mr2, &e.min_r2, 8);
        std::memcpy(&v2, &e.v2, 8);
        std::memcpy(&a2, &e.a2, 8);
        std::memcpy(&kc, &e.kc, 8);
        if (mr2 < o_.r_min_halt * o_.r_min_halt)
          return fail(core::Halt::Overlap,
                      "atom overlap at step " + std::to_string(h) +
                          ": min pair distance " +
                          std::to_string(std::sqrt(mr2)) + " A < " +
                          std::to_string(o_.r_min_halt) + " A");
        if (e.flags & 1)
          return fail(core::Halt::Internal,
                      "fixed-point overflow on GPU at step " + std::to_string(h));
        if (e.flags & 2)
          return fail(core::Halt::StaleZone,
                      "stale zone membership at step " + std::to_string(h) +
                          " zone " + std::to_string(s.fsm.id) +
                          ": atom left its slab by more than (width-r_cut)/2");
        s.v_max = std::sqrt(v2);
        s.a_max = std::sqrt(a2);
        s.k2cap = kc;
        s.fsm.v_max_local = s.v_max;
        s.fsm.R_buf_local = R_buf;
        agg.v = std::max(agg.v, s.v_max);
        agg.a = std::max(agg.a, s.a_max);
        agg.k2cap = std::min(agg.k2cap, s.k2cap);
        ke += double(e.ke) / core::fixed::EnergyAccum::kScale;
        if (!core::buffer::causality_ok(s.v_max, dt, R_buf)) {
          char buf[176];
          std::snprintf(buf, sizeof(buf),
                        "causality (INV-4) at step %ld zone %d: "
                        "v_max*dt=%.4g > R_buf=%.4g (dt=%.17g)",
                        h, s.fsm.id, s.v_max * dt, R_buf, dt);
          return fail(core::Halt::Causality, buf);
        }
      }
      if (o_.verlet_hybrid) {  // pass-global max displacement (PR-3) + drift (PR-4)
        unsigned long long d2bits = 0;
        TDMD_CU(cudaMemcpy(&d2bits, nb.d_d2, 8, cudaMemcpyDeviceToHost));
        double d2;
        std::memcpy(&d2, &d2bits, 8);
        agg.d = std::sqrt(d2);
        if (o_.verlet_drift) {  // PR-4: D0 = (Σ displacement)/N, z-independent
          long long sx = 0, sy = 0, sz = 0;
          TDMD_CU(cudaMemcpy(&sx, nb.d_sx, 8, cudaMemcpyDeviceToHost));
          TDMD_CU(cudaMemcpy(&sy, nb.d_sy, 8, cudaMemcpyDeviceToHost));
          TDMD_CU(cudaMemcpy(&sz, nb.d_sz, 8, cudaMemcpyDeviceToHost));
          const double invN = 1.0 / double(atoms_.n);
          const double sc = core::fixed::ForceAccum::kScale;
          agg.drift[0] = (double(sx) / sc) * invN;
          agg.drift[1] = (double(sy) / sc) * invN;
          agg.drift[2] = (double(sz) / sc) * invN;
        }
      }
      return true;
    };


    auto flush_sends = [&]() -> bool {
      if (outq.empty()) return !halt_on_.load(std::memory_order_relaxed);
      NvtxScope nv("flush");
      while (!outq.empty()) {
        const int j = outq.front();
        outq.pop_front();
        PassSlot& s = slot[std::size_t(j)];
        const DevSlot& src = nb.slot[std::size_t(s.dev)];
        // freeing a source slot back to its owner pool: the pass-1 reserve
        // recycles uploads; the self-loop keeps ONE pool; boundary-fed node
        // 0 uses its local list (stream order covers reuse); intra-fed
        // nodes pair a recorded free event with an upstream credit.
        auto free_src = [&](int dev) {
          if (self_loop_ || preload) {
            p1free_.push_back(dev);
          } else if (b_in) {
            bfree_.push_back(dev);
          } else {
            TDMD_CU(cudaEventRecord(in.free_[std::size_t(dev)], nb.stream));
            in.credits_.send(int(dev));
          }
        };
        if (h == o_.steps) {  // ring ends — land the payload on the host
          HostZone& hz = final_[std::size_t(s.fsm.id)];
          hz.n = s.n_atoms;
          hz.a.assign(DevSlot::kArrays * std::size_t(cap_), 0.0);
          TDMD_CU(cudaMemcpyAsync(hz.a.data(), src.base,
                                  DevSlot::kArrays * sizeof(double) * cap_,
                                  cudaMemcpyDeviceToHost, nb.stream));
          TDMD_CU(cudaStreamSynchronize(nb.stream));
          final_seen_.store(true, std::memory_order_release);
          free_src(s.dev);  // streamed-preload uploads need recycled slots
          continue;
        }
        const int kn = (k + 1) % z_;
        NodeBuf& cn = node_[std::size_t(kn)];
        if (b_out) {
          // M5a boundary: stage the wire image to pinned host memory and hand
          // it to the edge (D2H -> MPI -> H2D on the peer). One staging
          // buffer suffices: send() returns only when it may be reused.
          if (sent + 1 >= n_ && !sync_check()) return false;  // agg + guards
          GpuHeader gh;
          gh.hdr.zone_id = s.fsm.id;
          gh.hdr.step_h = h;
          gh.hdr.sent_pos = sent;
          const Lambda lamb =
              (sent + 1 < n_) ? slot[std::size_t(sent + 1)].lam_in : agg;
          if (sent == 0) {
            dt_next = o_.auto_step
                          ? core::buffer::auto_dt(lamb.v, dt, o_.ts, lamb.k2cap)
                          : o_.dt_initial;
            decide_pass(skin_in, R_buf, verlet_active_pass, lamb.d,
                        (n_ == 1) ? 1.0 : double(n_ - 1), skin_out, rebuild_next,
                        va_next);
          }
          gh.hdr.dt_next = (sent == 0) ? dt_next : 0.0;
          gh.hdr.v_full = lamb.v;
          gh.hdr.a_full = lamb.a;
          gh.hdr.k2cap_full = lamb.k2cap;
          gh.hdr.d_full = lamb.d;            // PR-3 lagged max displacement
          gh.hdr.drift_full[0] = lamb.drift[0];
          gh.hdr.drift_full[1] = lamb.drift[1];
          gh.hdr.drift_full[2] = lamb.drift[2];
          gh.hdr.skin_consumed = skin_out;  // broadcast to EVERY header
          gh.hdr.rebuild_now = rebuild_next ? 1 : 0;
          gh.hdr.verlet_active = va_next;
          gh.n_atoms = s.n_atoms;
          gh.dst_slot = -1;  // the receiver picks its own slot
          if (o_.mixed_transport) {
            if (s.n_atoms > 0) {
              const int grid = (s.n_atoms + kIntBlock - 1) / kIntBlock;
              pack_zone_kernel<<<grid, kIntBlock, 0, nb.stream>>>(
                  src.base, cap_, s.n_atoms, 1.0 / qp_, 1.0 / qv_, 1.0 / qf_,
                  box_.lo[0], box_.lo[1], box_.lo[2], src.pk_mass, src.pk,
                  nb.d_flags);
              TDMD_CU(cudaGetLastError());
            }
            TDMD_CU(cudaMemcpyAsync(out_stage_, src.pk_mass, wire_bytes(),
                                    cudaMemcpyDeviceToHost, nb.stream));
          } else {
            TDMD_CU(cudaMemcpyAsync(out_stage_, src.base, wire_bytes(),
                                    cudaMemcpyDeviceToHost, nb.stream));
          }
          TDMD_CU(cudaStreamSynchronize(nb.stream));
          core::ZoneFSM::apply(s.fsm, core::ZoneEvent::SEND);  // T5
          part_.out->send(gh, out_stage_, wire_bytes());
          free_src(s.dev);
          ++sent;
          continue;
        }
        int dst = -1;
        if (self_loop_) {  // one node, one thread, one pool
          if (p1free_.empty())
            throw std::logic_error("gpu conveyor: self-loop pool exhausted");
          dst = p1free_.front();
          p1free_.pop_front();
        } else if (!out.credits_.recv(dst)) {
          return false;  // free-slot credit (shutdown)
        }
        // slot-reuse safety: the credit is pushed only after the consumer
        // recorded free_[dst], so this wait is never a no-op on a live slot
        TDMD_CU(cudaStreamWaitEvent(nb.stream, out.free_[std::size_t(dst)], 0));
        if (o_.mixed_transport) {  // B5: int32 wire image (1.8x less traffic)
          if (s.n_atoms > 0) {
            const int grid = (s.n_atoms + kIntBlock - 1) / kIntBlock;
            pack_zone_kernel<<<grid, kIntBlock, 0, nb.stream>>>(
                src.base, cap_, s.n_atoms, 1.0 / qp_, 1.0 / qv_, 1.0 / qf_,
                box_.lo[0], box_.lo[1], box_.lo[2], src.pk_mass, src.pk,
                nb.d_flags);
            TDMD_CU(cudaGetLastError());
          }
          TDMD_CU(cudaMemcpyAsync(cn.slot[std::size_t(dst)].pk_mass,
                                  src.pk_mass, packed_bytes(),
                                  cudaMemcpyDeviceToDevice, nb.stream));
        } else {
          TDMD_CU(cudaMemcpyAsync(cn.slot[std::size_t(dst)].base, src.base,
                                  DevSlot::kArrays * sizeof(double) * cap_,
                                  cudaMemcpyDeviceToDevice, nb.stream));
        }
        TDMD_CU(cudaEventRecord(out.arrival_[std::size_t(dst)], nb.stream));
        // my source slot becomes reusable once that copy has READ it
        free_src(s.dev);

        if (sent + 1 >= n_ && !sync_check()) return false;  // agg + guards
        GpuHeader gh;
        gh.hdr.zone_id = s.fsm.id;
        gh.hdr.step_h = h;
        gh.hdr.sent_pos = sent;
        const Lambda lam =
            (sent + 1 < n_) ? slot[std::size_t(sent + 1)].lam_in : agg;
        if (sent == 0) {
          dt_next = o_.auto_step
                        ? core::buffer::auto_dt(lam.v, dt, o_.ts, lam.k2cap)
                        : o_.dt_initial;
          decide_pass(skin_in, R_buf, verlet_active_pass, lam.d,
                      (n_ == 1) ? 1.0 : double(n_ - 1), skin_out, rebuild_next,
                      va_next);
        }
        gh.hdr.dt_next = (sent == 0) ? dt_next : 0.0;
        gh.hdr.v_full = lam.v;
        gh.hdr.a_full = lam.a;
        gh.hdr.k2cap_full = lam.k2cap;
        gh.hdr.d_full = lam.d;            // PR-3 lagged max displacement
        gh.hdr.drift_full[0] = lam.drift[0];
        gh.hdr.drift_full[1] = lam.drift[1];
        gh.hdr.drift_full[2] = lam.drift[2];
        gh.hdr.skin_consumed = skin_out;  // broadcast to EVERY header
        gh.hdr.rebuild_now = rebuild_next ? 1 : 0;
        gh.hdr.verlet_active = va_next;
        gh.n_atoms = s.n_atoms;
        gh.dst_slot = dst;
        core::ZoneFSM::apply(s.fsm, core::ZoneEvent::SEND);  // T5
        out.chan_.send(std::move(gh));
        ++sent;
      }
      return !halt_on_.load(std::memory_order_relaxed);
    };

    auto compute_zone = [&](int j) -> bool {
      if (!ensure_arrival(j)) return false;
      if (j + 1 < n_ && !ensure_arrival(j + 1)) return false;
      ensure_drift(j);
      if (j + 1 < n_) ensure_drift(j + 1);
      PassSlot& s = slot[std::size_t(j)];
      if (j == 0) {
        core::seed_first_zone(s.fsm);  // §7.1
        if (o_.verlet_reuse && verlet_active_pass && rebuild_now_pass)
          verlet_rebuilds_.fetch_add(1, std::memory_order_relaxed);  // per pass
      }
      if (j > 0 && !slot[std::size_t(j - 1)].computed)
        throw std::logic_error("gpu conveyor: INV-1 violated");
      core::ZoneFSM::apply(s.fsm, core::ZoneEvent::START);  // T3
      {
        NvtxScope nv("pairs");
        // role: 0=SELF, 1=NEXT (partner is A's +1 / wrap), 2=PREV (partner is
        // A's -1 / wrap) — the persistent Verlet list key (PR-1b-ii).
        launch_pairs(s, s, /*same=*/true, /*energy=*/true, /*role=*/0);
        if (j + 1 < n_) {
          PassSlot& nx = slot[std::size_t(j + 1)];
          core::ZoneFSM::apply(nx.fsm, core::ZoneEvent::SPHERE);  // T2
          nx.fsm.contrib_mask |= 1u;
          launch_pairs(s, nx, false, true, 1);   // A=j: partner j+1 is NEXT
          launch_pairs(nx, s, false, false, 2);  // A=j+1: partner j is PREV
        } else if (defer_head && j == n_ - 1) {
          PassSlot& hd = slot[0];
          hd.fsm.contrib_mask |= 2u;
          launch_pairs(s, hd, false, true, 1);   // A=n-1: NEXT wraps to 0
          launch_pairs(hd, s, false, false, 2);  // A=0:   PREV wraps to n-1
        }
        }
      s.computed = true;
      if (!(defer_head && j == 0) && !end_zone(j)) return false;
      if (defer_head && j == n_ - 1 && !end_zone(0)) return false;
      return true;
    };

    for (int j = 0; j < n_; ++j) {
      for (core::IoOp op : io) {
        if (op == core::IoOp::SEND) {
          if (!flush_sends()) return false;
        } else {
          if (!ensure_arrival(std::min(j + 1, n_ - 1))) return false;
        }
      }
      if (halt_on_.load(std::memory_order_acquire)) return false;
      if (!compute_zone(j)) return false;
    }
    if (!flush_sends()) return false;
    if (!sync_check()) return false;  // final pass has no sends — sync here

    if (preload && !self_loop_) {
      // pass 1 done: the upload reserve joins the main circulation. Intra
      // consumers pair credits with RECORDED free events (the unrecorded-
      // event wait is a no-op — the M5a lesson), so record them now: every
      // pending read of these slots is already enqueued on this stream.
      while (!p1free_.empty()) {
        const int dev = p1free_.front();
        p1free_.pop_front();
        if (b_in) {
          bfree_.push_back(dev);
        } else {
          TDMD_CU(cudaEventRecord(in.free_[std::size_t(dev)], nb.stream));
          in.credits_.send(int(dev));
        }
      }
    }

    if (o_.mixed_transport) {  // pack-overflow flags from this pass's sends
      TDMD_CU(cudaMemcpyAsync(&nb.h_end[0].flags, nb.d_flags, 4,
                              cudaMemcpyDeviceToHost, nb.stream));
      TDMD_CU(cudaStreamSynchronize(nb.stream));
      if (nb.h_end[0].flags & 4)
        return fail(core::Halt::Internal,
                    "production_mixed transport range overflow at step " +
                        std::to_string(h) +
                        " (atom beyond the int32 offset range)");
    }
    const double pass_pe =
        double(nb.h_end[std::size_t(ended.back())].pe) /
        core::fixed::EnergyAccum::kScale;
    if (!std::isfinite(pass_pe + ke))
      return fail(core::Halt::NonFiniteEnergy,
                  "non-finite energy at step " + std::to_string(h));
    res_.stats[std::size_t(h - 1)] = {pass_pe, ke,    dt,
                                      agg.v,   agg.a, agg.k2cap};
    return true;
  }

  void scatter_final() {
    for (int zid = 0; zid < n_; ++zid) {
      const auto& mem = zd_.members[std::size_t(zid)];
      const HostZone& hz = final_[std::size_t(zid)];
      if (hz.n != int(mem.size()))
        throw std::logic_error("gpu conveyor: final zone size mismatch");
      for (std::size_t t = 0; t < mem.size(); ++t) {
        const std::size_t i = std::size_t(mem[t]);
        atoms_.x[i] = hz.a[0 * cap_ + t];
        atoms_.y[i] = hz.a[1 * cap_ + t];
        atoms_.z[i] = hz.a[2 * cap_ + t];
        atoms_.vx[i] = hz.a[3 * cap_ + t];
        atoms_.vy[i] = hz.a[4 * cap_ + t];
        atoms_.vz[i] = hz.a[5 * cap_ + t];
        atoms_.fx[i] = hz.a[6 * cap_ + t];
        atoms_.fy[i] = hz.a[7 * cap_ + t];
        atoms_.fz[i] = hz.a[8 * cap_ + t];
      }
    }
  }

  void set_halt(core::Halt kind, std::string msg) {
    bool expected = false;
    if (halt_on_.compare_exchange_strong(expected, true)) {
      std::lock_guard lk(halt_mu_);
      halt_kind_ = kind;
      halt_msg_ = std::move(msg);
      transport_->shutdown();
      if (part_.out) part_.out->poison();  // M5a: stop the other ranks too
    }
  }

  core::AtomSoA<double>& atoms_;
  const core::Box& box_;
  const double rcut_;
  PairF pot_;
  const core::ConveyorOptions o_;
  RingPart part_;
  const core::PairGeom geom_;
  // PR-1 Verlet reuse: list-build geometry at rcut+skin (the force kernel still
  // re-tests at rcut via geom_). grid_rcut_ sizes the cell grid halo so the
  // 27-cell sweep covers rcut+skin; equals rcut_ when the feature is off.
  const core::PairGeom geom_list_;
  double grid_rcut_;

  std::size_t packed_bytes() const {
    return sizeof(double) * cap_ + 9 * sizeof(int) * std::size_t(cap_);
  }
  // PR-1b-ii skin recurrence (NL-INV-2a) + PR-2 K-aware fallback (I1). All
  // inputs are z-independent scalars over the lagged Λ-forecast, so the whole
  // decision (skin budget, rebuild, AND verlet_active) is bitwise z-independent.
  //   * charge 2*R_buf — the per-step pair-approach bound INV-4 enforces.
  //   * K_pred = skin/(2*R_buf): predicted reuse factor. Two-threshold
  //     hysteresis (K_on > K_off) flips verlet_active without chatter; below
  //     break-even we fall back to the cell-raster path (worst case == current
  //     engine, not "+tax"). A 0->1 turn-on forces a rebuild (no list yet).
  void decide_pass(double skin_in, double R_buf, uint8_t va_prev, double d_lagged,
                   double lag, double& skin_out, bool& rebuild,
                   uint8_t& va_next) const {
    const double charge = 2.0 * R_buf;
    const double K_pred = charge > 1e-300
                              ? o_.verlet_skin / charge
                              : std::numeric_limits<double>::infinity();
    va_next = va_prev;
    if (va_prev == 0 && K_pred >= o_.verlet_K_on) va_next = 1;
    else if (va_prev == 1 && K_pred < o_.verlet_K_off) va_next = 0;
    if (!va_next) {                 // fallback (cell-raster): budget idle
      rebuild = false; skin_out = 0.0;
    } else if (va_prev == 0) {      // turned ON: no list yet -> force rebuild
      rebuild = true; skin_out = 0.0;
    } else {
      skin_out = skin_in + charge;       // conservative accumulator (carried)
      double skin_used = skin_out;
      if (o_.verlet_hybrid) {
        // PR-3 hybrid: 2*d_lagged + 2*L*R_buf is also a valid upper bound on
        // the current pair-approach (d_lagged = max displacement as of t-L,
        // + L steps of R_buf). min() of two upper bounds is the tightest SAFE
        // bound => rebuilds no sooner than conservative (larger K). The
        // post-rebuild stale d_lagged is harmless: skin_out is then tiny, so
        // min() picks it — no rebuild storm (no epoch tracking needed).
        const double hyb = 2.0 * d_lagged + 2.0 * lag * R_buf;
        skin_used = hyb < skin_used ? hyb : skin_used;
      }
      rebuild = (skin_used >= o_.verlet_skin);
      if (rebuild) skin_out = 0.0;
    }
  }
  CellGrid zone_grid(int zone_id) const {
    const double len[3] = {box_.len(0), box_.len(1), box_.len(2)};
    const bool per[3] = {box_.periodic[0], box_.periodic[1], box_.periodic[2]};
    return make_zone_grid(box_.lo.data(), len, per, grid_rcut_, n_, zone_id);
  }
  std::size_t wire_bytes() const {  // the boundary payload image
    return o_.mixed_transport ? packed_bytes()
                              : DevSlot::kArrays * sizeof(double) * cap_;
  }

  core::ZoneDecomposition zd_;
  int n_ = 1, z_ = 1, Z_ = 1, S_ = 3, cap_ = 0;
  int ncells_ = 1;
  int max_neigh_ = 1;       // PR-1: SELF per-atom Verlet stride
  int max_neigh_cross_ = 1;  // PR-1b-ii sparse: NEXT/PREV stride (half-sphere)
  // PR-1b-ii: persistent Verlet lists keyed by (zone-id, role), role ∈
  // {SELF,NEXT,PREV} — a zone pairs only with itself + its slab neighbours.
  // SHARED across nodes (a list built at the rebuild step is read at later
  // reuse steps by whichever node owns them): correct because the existing
  // arrival-event chain orders consecutive steps' access to each zone, and the
  // stored LOCAL indices are slot-independent (static membership). idx capacity
  // = cap*max_neigh_, offsets = cap+1, per (zone,role) => 3*n_ buffers.
  static constexpr int kRoles = 3;  // 0=SELF, 1=NEXT, 2=PREV
  std::vector<int*> vl_off_;   // [zone*kRoles + role], cap+1 ints
  std::vector<int*> vl_idx_;   // [zone*kRoles + role], cap*max_neigh_ ints
  std::vector<double*> vl_xref_;  // PR-3 [zone], 3*cap doubles (rebuild-epoch x)
  int node0_ = 0;
  std::deque<int> bfree_;          // local slot pool of a boundary-fed node 0
  std::deque<int> p1free_;         // pass-1 upload reserve / self-loop pool
  std::vector<double> p1host_;     // reusable gather buffer (node 0 thread)
  bool self_loop_ = false;
  double* in_stage_ = nullptr;     // pinned boundary staging (M5a)
  double* out_stage_ = nullptr;
  cudaEvent_t in_evt_{};
  std::atomic<bool> final_seen_{false};
  double qp_ = 0.0, qv_ = 0.0, qf_ = 0.0;  // B5 quanta (mixed only)
  bool pbc_z_ = false;
  Lambda lam0_{};
  const unsigned long long inf_bits_ = 0x7FF0000000000000ULL;
  std::vector<NodeBuf> node_;
  std::unique_ptr<StreamTransport> transport_;
  std::vector<HostZone> final_;
  core::ConveyorResult res_;
  std::atomic<long> verlet_rebuilds_{0};  // PR-1b-ii rebuild count (all nodes)

  std::atomic<bool> halt_on_{false};
  std::mutex halt_mu_;
  core::Halt halt_kind_ = core::Halt::None;
  std::string halt_msg_;
};

template <typename PairF>
core::ConveyorResult run_conveyor_gpu(core::AtomSoA<double>& atoms,
                                      const core::Box& box, double rcut,
                                      PairF pot,
                                      const core::ConveyorOptions& o,
                                      RingPart part = {}) {
  GpuTimeConveyor<PairF> tc(atoms, box, rcut, std::move(pot), o, part);
  return tc.run();
}

}  // namespace tdmd::cuda
