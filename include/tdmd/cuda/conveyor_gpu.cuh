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

#include "tdmd/core/buffer.hpp"
#include "tdmd/core/conveyor.hpp"
#include "tdmd/core/fsm.hpp"
#include "tdmd/core/transport.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/zone_force.cuh"
#include "tdmd/cuda/zone_integrate.cuh"

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

// --- per-node device storage ----------------------------------------------

// Packed zone slot: base[k*cap + i], k = 0..9 -> x y z vx vy vz fx fy fz mass.
struct DevSlot {
  double* base = nullptr;
  long long* raw = nullptr;  // [3*cap]: rfx rfy rfz
  int cap = 0;
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
                  double rcut, PairF pot, const core::ConveyorOptions& o)
      : atoms_(atoms), box_(box), rcut_(rcut), pot_(pot), o_(o),
        geom_(box, rcut) {
    if (o_.steps < 1) throw std::invalid_argument("gpu conveyor: steps >= 1");
    if (o_.n_nodes < 1) throw std::invalid_argument("gpu conveyor: n_nodes >= 1");
    if (!(o_.dt_initial > 0.0))
      throw std::invalid_argument("gpu conveyor: dt_initial must be > 0");
  }

  core::ConveyorResult run() {
    zd_ = core::ZoneDecomposition::build(atoms_, box_, o_.n_zones, rcut_);
    n_ = zd_.n_zones;
    z_ = o_.n_nodes;
    S_ = n_ + 2;  // slots per node
    pbc_z_ = box_.periodic[2];
    for (const auto& m : zd_.members) cap_ = std::max(cap_, int(m.size()));
    cap_ = std::max(cap_, 1);

    // t0 forces + Λ pre-history — identical to the CPU conveyor.
    core::zero_forces(atoms_);
    const double pe0 = core::zone_force_pass(atoms_, box_, zd_, rcut_, pot_);
    res_.e0 = pe0 + core::kinetic_energy(atoms_);
    lam0_ = {core::buffer::max_speed(atoms_), core::buffer::max_accel(atoms_),
             core::buffer::temperature_limited_dt(atoms_, o_.ts.K2)};
    res_.stats.assign(std::size_t(o_.steps), {});
    final_.assign(std::size_t(n_), HostZone{});

    alloc_nodes();
    preload_node0();
    transport_ = std::make_unique<StreamTransport>(z_, S_, std::size_t(n_) + 2);
    for (int k = 0; k < z_; ++k) {  // edge k feeds node (k+1)%z
      const int kn = (k + 1) % z_;
      for (int s2 = (kn == 0 ? n_ : 0); s2 < S_; ++s2)
        transport_->edge(k).credits_.send(int(s2));
    }

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
      scatter_final();
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
    EndScalars* h_end{};  // pinned
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
      }
      TDMD_CU(cudaMalloc(&nb.d_v2, 8));
      TDMD_CU(cudaMalloc(&nb.d_a2, 8));
      TDMD_CU(cudaMalloc(&nb.d_kc, 8));
      TDMD_CU(cudaMalloc(&nb.d_mr2, 8));
      TDMD_CU(cudaMalloc(&nb.d_ke, 8));
      TDMD_CU(cudaMalloc(&nb.d_pe, 8));
      TDMD_CU(cudaMalloc(&nb.d_flags, 4));
      TDMD_CU(cudaHostAlloc(&nb.h_end, sizeof(EndScalars), 0));
    }
  }

  void free_nodes() {
    for (auto& nb : node_) {
      for (auto& s : nb.slot) {
        cudaFree(s.base);
        cudaFree(s.raw);
      }
      for (auto* p : {nb.d_v2, nb.d_a2, nb.d_kc, nb.d_mr2}) cudaFree(p);
      cudaFree(nb.d_ke);
      cudaFree(nb.d_pe);
      cudaFree(nb.d_flags);
      cudaFreeHost(nb.h_end);
      cudaStreamDestroy(nb.stream);
    }
    node_.clear();
  }

  // §7.3: P1 starts with every zone in d — upload t0 state into node 0
  // slots 0..n-1. Λ pre-history and dt(1) are synthesized like the CPU.
  void preload_node0() {
    std::vector<double> h(DevSlot::kArrays * std::size_t(cap_), 0.0);
    for (int j = 0; j < n_; ++j) {
      const auto& mem = zd_.members[std::size_t(j)];
      for (std::size_t t = 0; t < mem.size(); ++t) {
        const int i = mem[t];
        h[0 * cap_ + t] = atoms_.x[i];
        h[1 * cap_ + t] = atoms_.y[i];
        h[2 * cap_ + t] = atoms_.z[i];
        h[3 * cap_ + t] = atoms_.vx[i];
        h[4 * cap_ + t] = atoms_.vy[i];
        h[5 * cap_ + t] = atoms_.vz[i];
        h[6 * cap_ + t] = atoms_.fx[i];
        h[7 * cap_ + t] = atoms_.fy[i];
        h[8 * cap_ + t] = atoms_.fz[i];
        h[9 * cap_ + t] = atoms_.mass[i];
      }
      TDMD_CU(cudaMemcpy(node_[0].slot[std::size_t(j)].base, h.data(),
                         DevSlot::kArrays * sizeof(double) * cap_,
                         cudaMemcpyHostToDevice));
    }
  }

  void node_main(int k) {
    try {
      for (long h = k + 1; h <= o_.steps; h += z_) {
        if (halt_on_.load(std::memory_order_acquire)) return;
        if (!run_pass(k, h)) return;
      }
    } catch (const std::exception& ex) {
      set_halt(core::Halt::Internal,
               std::string("gpu conveyor internal error: ") + ex.what());
    }
  }

  bool run_pass(int k, long h) {
    NodeBuf& nb = node_[std::size_t(k)];
    StreamEdge& in = transport_->edge((k - 1 + z_) % z_);
    StreamEdge& out = transport_->edge(k);
    const auto io = core::node_io_order(k + 1);       // §7.4 parity
    const int r = pbc_z_ ? int((h - 1) % n_) : 0;     // rotation
    const bool defer_head = pbc_z_ && n_ > 1;
    const bool preload = (h == 1);

    std::vector<PassSlot> slot(static_cast<std::size_t>(n_));
    std::deque<int> outq;
    int sent = 0, arrived = 0;
    double dt = 0.0, R_buf = 0.0, dt_next = 0.0;
    Lambda agg{0.0, 0.0, std::numeric_limits<double>::infinity()};
    double ke = 0.0;

    // per-pass device scalar reset (min_r2 = +inf, pe = 0, flags = 0)
    TDMD_CU(cudaMemcpyAsync(nb.d_mr2, &inf_bits_, 8, cudaMemcpyHostToDevice,
                            nb.stream));
    TDMD_CU(cudaMemsetAsync(nb.d_pe, 0, 8, nb.stream));
    TDMD_CU(cudaMemsetAsync(nb.d_flags, 0, 4, nb.stream));

    auto fail = [&](core::Halt kind, const std::string& msg) {
      set_halt(kind, msg);
      return false;
    };

    auto ensure_arrival = [&](int j) -> bool {
      while (arrived <= j) {
        PassSlot& s = slot[std::size_t(arrived)];
        const int want_id = (r + arrived) % n_;
        if (preload) {
          s.dev = arrived;  // preload occupies slots 0..n-1
          s.n_atoms = int(zd_.members[std::size_t(want_id)].size());
          s.lam_in = lam0_;
          if (arrived == 0) dt = o_.dt_initial;
          s.fsm.type = core::initial_zone_type(1);
        } else {
          GpuHeader gh;
          if (!in.chan_.recv(gh)) return false;  // shutdown
          if (gh.hdr.zone_id != want_id || gh.hdr.step_h != h - 1 ||
              gh.hdr.sent_pos != arrived)
            throw std::logic_error("gpu conveyor: ring arrival out of order");
          s.dev = gh.dst_slot;
          s.n_atoms = gh.n_atoms;
          s.lam_in = {gh.hdr.v_full, gh.hdr.a_full, gh.hdr.k2cap_full};
          if (arrived == 0) dt = gh.hdr.dt_next;
          // INV-2: the payload copy is ordered before this wait by the
          // record-before-push / pop-before-wait protocol invariant.
          TDMD_CU(cudaStreamWaitEvent(nb.stream, in.arrival_[std::size_t(s.dev)], 0));
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
      const DevSlot& d = nb.slot[std::size_t(s.dev)];
      if (s.n_atoms > 0) {
        const int grid = (s.n_atoms + kIntBlock - 1) / kIntBlock;
        zone_drift_kernel<<<grid, kIntBlock, 0, nb.stream>>>(
            d.arr(0), d.arr(1), d.arr(2), d.arr(3), d.arr(4), d.arr(5),
            d.arr(6), d.arr(7), d.arr(8), d.arr(9), s.n_atoms, dt);
        TDMD_CU(cudaGetLastError());
      }
      TDMD_CU(cudaMemsetAsync(d.raw, 0, 3 * sizeof(long long) * cap_,
                              nb.stream));
      s.drifted = true;
    };

    auto launch_pairs = [&](PassSlot& A, PassSlot& B, bool same, bool energy) {
      if (A.n_atoms == 0) return;
      const DevSlot& da = nb.slot[std::size_t(A.dev)];
      const DevSlot& db = nb.slot[std::size_t(B.dev)];
      ZoneForceArgs args{da.arr(0), da.arr(1), da.arr(2), A.n_atoms,
                         db.arr(0), db.arr(1), db.arr(2), B.n_atoms,
                         da.rfx(),  da.rfy(),  da.rfz(),  nb.d_pe,
                         nb.d_mr2,  nb.d_flags, same,     energy};
      const int grid = (A.n_atoms + kZoneBlock - 1) / kZoneBlock;
      zone_pair_kernel<PairF><<<grid, kZoneBlock, 0, nb.stream>>>(args, geom_,
                                                                  pot_);
      TDMD_CU(cudaGetLastError());
    };

    auto end_zone = [&](int j) -> bool {
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
      }
      // D2H the END scalars (pinned) + sync — the host decides like the CPU.
      TDMD_CU(cudaMemcpyAsync(&nb.h_end->v2, nb.d_v2, 8, cudaMemcpyDeviceToHost, nb.stream));
      TDMD_CU(cudaMemcpyAsync(&nb.h_end->a2, nb.d_a2, 8, cudaMemcpyDeviceToHost, nb.stream));
      TDMD_CU(cudaMemcpyAsync(&nb.h_end->kc, nb.d_kc, 8, cudaMemcpyDeviceToHost, nb.stream));
      TDMD_CU(cudaMemcpyAsync(&nb.h_end->min_r2, nb.d_mr2, 8, cudaMemcpyDeviceToHost, nb.stream));
      TDMD_CU(cudaMemcpyAsync(&nb.h_end->ke, nb.d_ke, 8, cudaMemcpyDeviceToHost, nb.stream));
      TDMD_CU(cudaMemcpyAsync(&nb.h_end->pe, nb.d_pe, 8, cudaMemcpyDeviceToHost, nb.stream));
      TDMD_CU(cudaMemcpyAsync(&nb.h_end->flags, nb.d_flags, 4, cudaMemcpyDeviceToHost, nb.stream));
      TDMD_CU(cudaStreamSynchronize(nb.stream));

      double mr2, v2, a2, kc;
      std::memcpy(&mr2, &nb.h_end->min_r2, 8);
      std::memcpy(&v2, &nb.h_end->v2, 8);
      std::memcpy(&a2, &nb.h_end->a2, 8);
      std::memcpy(&kc, &nb.h_end->kc, 8);
      if (mr2 < o_.r_min_halt * o_.r_min_halt)
        return fail(core::Halt::Overlap,
                    "atom overlap at step " + std::to_string(h) +
                        ": min pair distance " + std::to_string(std::sqrt(mr2)) +
                        " A < " + std::to_string(o_.r_min_halt) + " A");
      if (nb.h_end->flags & 1)
        return fail(core::Halt::Internal,
                    "fixed-point overflow on GPU at step " + std::to_string(h));
      if (nb.h_end->flags & 2)
        return fail(core::Halt::StaleZone,
                    "stale zone membership at step " + std::to_string(h) +
                        " zone " + std::to_string(s.fsm.id) +
                        ": atom left its slab by more than (width-r_cut)/2");
      s.v_max = std::sqrt(v2);
      s.a_max = std::sqrt(a2);
      s.k2cap = kc;
      s.ke_raw = nb.h_end->ke;
      s.fsm.v_max_local = s.v_max;
      s.fsm.R_buf_local = R_buf;
      agg.v = std::max(agg.v, s.v_max);
      agg.a = std::max(agg.a, s.a_max);
      agg.k2cap = std::min(agg.k2cap, s.k2cap);
      ke += double(s.ke_raw) / core::fixed::EnergyAccum::kScale;
      if (!core::buffer::causality_ok(s.v_max, dt, R_buf)) {
        char buf[176];
        std::snprintf(buf, sizeof(buf),
                      "causality (INV-4) at step %ld zone %d: "
                      "v_max*dt=%.4g > R_buf=%.4g (dt=%.17g)",
                      h, s.fsm.id, s.v_max * dt, R_buf, dt);
        return fail(core::Halt::Causality, buf);
      }
      core::ZoneFSM::apply(s.fsm, core::ZoneEvent::END);  // T4
      outq.push_back(j);
      return true;
    };

    auto flush_sends = [&]() -> bool {
      while (!outq.empty()) {
        const int j = outq.front();
        outq.pop_front();
        PassSlot& s = slot[std::size_t(j)];
        const DevSlot& src = nb.slot[std::size_t(s.dev)];
        if (h == o_.steps) {  // ring ends — land the payload on the host
          HostZone& hz = final_[std::size_t(s.fsm.id)];
          hz.n = s.n_atoms;
          hz.a.assign(DevSlot::kArrays * std::size_t(cap_), 0.0);
          TDMD_CU(cudaMemcpyAsync(hz.a.data(), src.base,
                                  DevSlot::kArrays * sizeof(double) * cap_,
                                  cudaMemcpyDeviceToHost, nb.stream));
          TDMD_CU(cudaStreamSynchronize(nb.stream));
          continue;
        }
        const int kn = (k + 1) % z_;
        NodeBuf& cn = node_[std::size_t(kn)];
        int dst = -1;
        if (!out.credits_.recv(dst)) return false;  // free-slot credit
        // slot-reuse safety: the credit is pushed only after the consumer
        // recorded free_[dst], so this wait is never a no-op on a live slot
        TDMD_CU(cudaStreamWaitEvent(nb.stream, out.free_[std::size_t(dst)], 0));
        TDMD_CU(cudaMemcpyAsync(cn.slot[std::size_t(dst)].base, src.base,
                                DevSlot::kArrays * sizeof(double) * cap_,
                                cudaMemcpyDeviceToDevice, nb.stream));
        TDMD_CU(cudaEventRecord(out.arrival_[std::size_t(dst)], nb.stream));
        // my source slot becomes reusable once that copy has READ it:
        // record the event FIRST, then hand the credit back upstream
        TDMD_CU(cudaEventRecord(in.free_[std::size_t(s.dev)], nb.stream));
        in.credits_.send(int(s.dev));

        GpuHeader gh;
        gh.hdr.zone_id = s.fsm.id;
        gh.hdr.step_h = h;
        gh.hdr.sent_pos = sent;
        const Lambda lam =
            (sent + 1 < n_) ? slot[std::size_t(sent + 1)].lam_in : agg;
        if (sent == 0)
          dt_next = o_.auto_step
                        ? core::buffer::auto_dt(lam.v, dt, o_.ts, lam.k2cap)
                        : o_.dt_initial;
        gh.hdr.dt_next = (sent == 0) ? dt_next : 0.0;
        gh.hdr.v_full = lam.v;
        gh.hdr.a_full = lam.a;
        gh.hdr.k2cap_full = lam.k2cap;
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
      if (j == 0) core::seed_first_zone(s.fsm);  // §7.1
      if (j > 0 && !slot[std::size_t(j - 1)].computed)
        throw std::logic_error("gpu conveyor: INV-1 violated");
      core::ZoneFSM::apply(s.fsm, core::ZoneEvent::START);  // T3
      launch_pairs(s, s, /*same=*/true, /*energy=*/true);
      if (j + 1 < n_) {
        PassSlot& nx = slot[std::size_t(j + 1)];
        core::ZoneFSM::apply(nx.fsm, core::ZoneEvent::SPHERE);  // T2
        nx.fsm.contrib_mask |= 1u;
        launch_pairs(s, nx, false, true);   // A-side: energy once
        launch_pairs(nx, s, false, false);  // B-side: Newton-3 partner
      } else if (defer_head && j == n_ - 1) {
        PassSlot& hd = slot[0];
        hd.fsm.contrib_mask |= 2u;
        launch_pairs(s, hd, false, true);   // closure (rotation §7.2)
        launch_pairs(hd, s, false, false);
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

    const double pass_pe = double(nb.h_end->pe) / core::fixed::EnergyAccum::kScale;
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
    }
  }

  core::AtomSoA<double>& atoms_;
  const core::Box& box_;
  const double rcut_;
  PairF pot_;
  const core::ConveyorOptions o_;
  const core::PairGeom geom_;

  core::ZoneDecomposition zd_;
  int n_ = 1, z_ = 1, S_ = 3, cap_ = 0;
  bool pbc_z_ = false;
  Lambda lam0_{};
  const unsigned long long inf_bits_ = 0x7FF0000000000000ULL;
  std::vector<NodeBuf> node_;
  std::unique_ptr<StreamTransport> transport_;
  std::vector<HostZone> final_;
  core::ConveyorResult res_;

  std::atomic<bool> halt_on_{false};
  std::mutex halt_mu_;
  core::Halt halt_kind_ = core::Halt::None;
  std::string halt_msg_;
};

template <typename PairF>
core::ConveyorResult run_conveyor_gpu(core::AtomSoA<double>& atoms,
                                      const core::Box& box, double rcut,
                                      PairF pot,
                                      const core::ConveyorOptions& o) {
  GpuTimeConveyor<PairF> tc(atoms, box, rcut, std::move(pot), o);
  return tc.run();
}

}  // namespace tdmd::cuda
