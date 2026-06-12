#pragma once
// M5a — MpiTransport: the IBoundaryEdge implementation over MPI with
// HOST-staging (D2H -> MPI -> H2D is done by the conveyor; this class moves
// host bytes). Base path per the M5a plan: the system OpenMPI 4.1.6 is not
// CUDA-aware (opal_built_with_cuda_support=false, measured) — device
// pointers must not enter MPI calls.
//
// One object per rank serves BOTH ring directions: recv from rank-1 (the
// conveyor's in-edge of local node 0) and send to rank+1 (the out-edge of
// local node z_local-1). Sends are MPI_Isend over a small buffer pool —
// non-blocking transport sidesteps the rendezvous-ring deadlock class
// entirely (the M3.5 review measured capacity<=3 self-loop deadlocks; a
// buffer pool is the MPI analogue of channel capacity). Requires
// MPI_THREAD_MULTIPLE: with z_local > 1 the in- and out-edges live on
// different node threads.
//
// HALT propagation: poison() sends a header with sent_pos = -1 and a hop TTL
// in step_h; a receiver forwards it once (TTL-1) and reports shutdown to its
// conveyor, so a halt on any rank stops the whole ring in <= nranks hops.
#include <mpi.h>

#include <atomic>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <vector>

#include "tdmd/cuda/conveyor_gpu.cuh"

namespace tdmd::mpi {

class MpiRingEdge final : public cuda::IBoundaryEdge {
 public:
  // Each ring instance gets a UNIQUE tag (process-global counter — identical
  // on every rank as long as rings are constructed in the same order): a
  // HALTED ring leaves unconsumed messages in flight, and a shared tag would
  // let the NEXT ring consume them as its own (found by test: garbage
  // headers, SIGSEGV). The destructor additionally drains stragglers; call
  // it only after all ranks left run() (e.g. behind a barrier).
  // pool_slots: Isend window. MUST be >= n_zones + 2 — a smaller window
  // reintroduces blocking-send semantics on rendezvous-size payloads, and
  // the M3.5 capacity analysis (self-loop deadlock at capacity < n) applies
  // to the boundary too (review M5a: measured hang at n_zones >= 11 with a
  // 4-deep pool). The conveyor's intra channels carry n+2 for the same
  // reason.
  MpiRingEdge(MPI_Comm comm, int rank, int nranks, std::size_t payload_bytes,
              int pool_slots = 4)
      : comm_(comm),
        prev_((rank - 1 + nranks) % nranks),
        next_((rank + 1) % nranks),
        nranks_(nranks),
        bytes_(payload_bytes),
        tag_(next_tag()),
        pool_(std::size_t(pool_slots < 4 ? 4 : pool_slots)) {
    for (auto& p : pool_) p.buf.resize(sizeof(cuda::GpuHeader) + bytes_);
    rbuf_.resize(sizeof(cuda::GpuHeader) + bytes_);
  }

  ~MpiRingEdge() override {
    // A clean run: every send was matched — plain waits complete. A HALTED
    // ring: in-flight payloads to a stopped rank can never match; waiting
    // would hang the job (review M5a, demonstrated on rendezvous payloads).
    // Drain-then-test: keep matching the upstream's stragglers while our own
    // sends complete; cancel whatever remains unmatched.
    bool pending = true;
    int spins = 0;
    while (pending) {
      drain_once();
      pending = false;
      for (auto& p : pool_) {
        if (p.req == MPI_REQUEST_NULL) continue;
        int done = 0;
        MPI_Test(&p.req, &done, MPI_STATUS_IGNORE);
        if (!done) pending = true;
      }
      if (pending && ++spins > 2000) {  // ~2 s — peer is gone for good
        for (auto& p : pool_)
          if (p.req != MPI_REQUEST_NULL) {
            MPI_Cancel(&p.req);
            MPI_Request_free(&p.req);
            p.req = MPI_REQUEST_NULL;
          }
        break;
      }
      if (pending) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    drain_once();
  }

  void send(const cuda::GpuHeader& h, const void* payload,
            std::size_t bytes) override {
    if (poisoned_.load(std::memory_order_acquire)) return;
    if (bytes != bytes_) throw std::logic_error("mpi edge: payload size");
    Pending& p = pool_[std::size_t(next_buf_)];
    next_buf_ = (next_buf_ + 1) % int(pool_.size());
    // pool-slot reuse: test-loop instead of a blind MPI_Wait — if the ring
    // gets poisoned while we wait, the peer may never match this request
    while (p.req != MPI_REQUEST_NULL) {
      int done = 0;
      MPI_Test(&p.req, &done, MPI_STATUS_IGNORE);
      if (done) break;
      if (poisoned_.load(std::memory_order_acquire)) {
        MPI_Cancel(&p.req);
        MPI_Request_free(&p.req);
        p.req = MPI_REQUEST_NULL;
        return;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    std::memcpy(p.buf.data(), &h, sizeof h);
    std::memcpy(p.buf.data() + sizeof h, payload, bytes);
    MPI_Isend(p.buf.data(), int(p.buf.size()), MPI_BYTE, next_, tag_, comm_,
              &p.req);
  }

  bool recv(cuda::GpuHeader& h, void* payload, std::size_t bytes) override {
    if (poisoned_.load(std::memory_order_acquire)) return false;
    if (bytes != bytes_) throw std::logic_error("mpi edge: payload size");
    MPI_Recv(rbuf_.data(), int(rbuf_.size()), MPI_BYTE, prev_, tag_, comm_,
             MPI_STATUS_IGNORE);
    std::memcpy(&h, rbuf_.data(), sizeof h);
    if (h.hdr.sent_pos < 0) {  // poison from another rank
      poisoned_.store(true, std::memory_order_release);
      const long ttl = h.hdr.step_h;
      if (ttl > 0) {
        cuda::GpuHeader fwd = h;
        fwd.hdr.step_h = ttl - 1;
        MPI_Send(&fwd, int(sizeof fwd), MPI_BYTE, next_, tag_, comm_);
      }
      return false;
    }
    std::memcpy(payload, rbuf_.data() + sizeof h, bytes);
    return true;
  }

  void poison() override {
    if (poisoned_.load(std::memory_order_acquire) ||
        poison_sent_.exchange(true, std::memory_order_acq_rel))
      return;
    cuda::GpuHeader p{};
    p.hdr.sent_pos = -1;
    p.hdr.step_h = nranks_ - 1;  // hop TTL
    MPI_Send(&p, int(sizeof p), MPI_BYTE, next_, tag_, comm_);
  }

 private:
  static int next_tag() {
    static int counter = 57;
    return counter++;
  }
  struct Pending {
    std::vector<char> buf;
    MPI_Request req = MPI_REQUEST_NULL;
  };

  void drain_once() {  // match upstream stragglers (poison/zones of a halt)
    int flag = 1;
    while (true) {
      MPI_Iprobe(prev_, tag_, comm_, &flag, MPI_STATUS_IGNORE);
      if (!flag) break;
      MPI_Recv(rbuf_.data(), int(rbuf_.size()), MPI_BYTE, prev_, tag_, comm_,
               MPI_STATUS_IGNORE);
    }
  }

  MPI_Comm comm_;
  int prev_, next_, nranks_;
  std::size_t bytes_;
  int tag_;
  std::vector<Pending> pool_;
  int next_buf_ = 0;
  std::vector<char> rbuf_;
  std::atomic<bool> poisoned_{false}, poison_sent_{false};
};

}  // namespace tdmd::mpi
