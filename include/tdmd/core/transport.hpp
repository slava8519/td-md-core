#pragma once
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

// M3.5 — zone transport for the TimeConveyor-CPU ring.
//
// ITransport is the CPU stand-in for hal/transport.hpp (Skeleton §2.1) [ENG]:
// the GPU form is asynchronous (post_send/post_recv + events); the conveyor
// LOGIC only needs ordered blocking send/recv per ring edge, so M3.5 stubs the
// contract with bounded SPSC channels (Roadmap M3.5: "SPSC-очереди как
// заглушка ITransport"). M4's StreamTransport implements the same contract
// over cudaMemcpyAsync + cudaEvent (INV-2).
//
// Edge k connects node k -> node (k+1) % z. Exactly one producer and one
// consumer per edge (SPSC); a plain mutex+condvar bounded queue is the
// timeboxed implementation — ring performance is explicitly NOT an M3.5 goal.
namespace tdmd::core {

// Per-zone message header. dt_next implements the Δt-handoff (B2, дисс. §3.3):
// the sender of a pass decides the NEXT pass's dt and ships it on the FIRST
// zone it sends. v_full/a_full/k2cap_full are the Λ-chain (ZoneFSM §6 B2):
// full-pass aggregates forwarded positionally so that the value riding the
// j-th sent zone of pass h is the aggregate of pass h-(n-1-j) — the freshest
// full pass inside the pass head's light cone. This makes the dt sequence and
// the INV-4 forecast bitwise independent of the node count z.
struct ZoneHeader {
  int    zone_id = 0;
  long   step_h = 0;       // the pass whose completed state this payload is
  int    sent_pos = 0;     // send-order index within the pass (= arrival pos)
  double dt_next = 0.0;    // on sent_pos==0: dt of pass step_h+1 (B2)
  double v_full = 0.0;     // Λ-chain: lagged full-pass v_max
  double a_full = 0.0;     // Λ-chain: lagged full-pass a_max
  double k2cap_full = 0.0; // Λ-chain: lagged full-pass K2 dt cap (min)
};

// Zone payload. deterministic_fp64 mode ships plain FP64 global coordinates
// (B5: FP32 offsets are a production_mixed traffic optimization, M4).
struct ZoneMsg {
  ZoneHeader hdr;
  std::vector<int>    id;
  std::vector<double> mass;
  std::vector<double> x, y, z, vx, vy, vz, fx, fy, fz;
  int n() const { return int(x.size()); }
};

// Bounded SPSC channel. shutdown() (HALT path) wakes every blocked side:
// send becomes a no-op, recv returns false — node threads unwind cleanly.
// Templated (M4): the CPU ring ships whole ZoneMsg payloads through it; the
// GPU ring ships only small zone HEADERS host-side while payloads move
// device-side (cudaMemcpyAsync + events — conveyor_gpu.cuh StreamTransport).
template <typename Msg>
class SpscChannelT {
 public:
  explicit SpscChannelT(std::size_t capacity) : cap_(capacity) {}

  void send(Msg&& m) {
    std::unique_lock lk(mu_);
    cv_send_.wait(lk, [&] { return q_.size() < cap_ || down_; });
    if (down_) return;
    q_.push_back(std::move(m));
    cv_recv_.notify_one();
  }

  bool recv(Msg& out) {
    std::unique_lock lk(mu_);
    cv_recv_.wait(lk, [&] { return !q_.empty() || down_; });
    if (down_) return false;
    out = std::move(q_.front());
    q_.pop_front();
    cv_send_.notify_one();
    return true;
  }

  void shutdown() {
    { std::lock_guard lk(mu_); down_ = true; }
    cv_send_.notify_all();
    cv_recv_.notify_all();
  }

 private:
  std::size_t cap_;
  std::deque<Msg> q_;
  std::mutex mu_;
  std::condition_variable cv_send_, cv_recv_;
  bool down_ = false;
};

using SpscChannel = SpscChannelT<ZoneMsg>;

struct ITransport {
  virtual void send(int edge, ZoneMsg&& m) = 0;
  virtual bool recv(int edge, ZoneMsg& out) = 0;  // false == shut down
  virtual void shutdown() = 0;
  virtual ~ITransport() = default;
};

class RingTransport final : public ITransport {
 public:
  RingTransport(int n_edges, std::size_t capacity) {
    edges_.reserve(n_edges);
    for (int i = 0; i < n_edges; ++i)
      edges_.push_back(std::make_unique<SpscChannel>(capacity));
  }
  void send(int edge, ZoneMsg&& m) override { edges_[edge]->send(std::move(m)); }
  bool recv(int edge, ZoneMsg& out) override { return edges_[edge]->recv(out); }
  void shutdown() override {
    for (auto& e : edges_) e->shutdown();
  }

 private:
  std::vector<std::unique_ptr<SpscChannel>> edges_;
};

}  // namespace tdmd::core
