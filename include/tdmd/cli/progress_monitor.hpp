#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

#include "tdmd/cli/dashboard.hpp"
#include "tdmd/core/conveyor.hpp"

// M7 (UX) — the bridge between the conveyor's concurrent per-pass on_pass
// callbacks (or the direct stepper's on_frame) and the Dashboard renderer.
//
// THREAD-SAFETY CONTRACT (load-bearing): z ring node threads call on_pass()
// CONCURRENTLY. Each call does the absolute minimum under one short lock —
// copy scalars into the latest Snapshot, advance the max-pass counter, push one
// E into the sparkline ring — then returns. Rendering (formatting + ANSI + I/O)
// happens on a SEPARATE render thread at ~10 Hz, NEVER inside on_pass. So the
// hook stays cheap and lock-contention on the hot path is bounded; the renderer
// snapshots under the same lock. The monitor itself does NOTHING to the ring's
// atoms/forces/dt — it is observational, preserving INV-9 (gate A7).
namespace tdmd::cli {

class ProgressMonitor {
public:
  // out: where rendered frames go (typically stderr so stdout stays the
  // parseable summary). tty: live ANSI vs plain append-only log. total: run
  // length (for the progress fraction). log_every: in non-TTY mode, emit one
  // log line every `log_every` passes.
  ProgressMonitor(std::FILE* out, bool tty, long total, const Snapshot& ctx, long log_every = 100)
      : out_(out), tty_(tty), dash_(tty), log_every_(log_every) {
    std::lock_guard lk(mu_);
    snap_ = ctx;
    snap_.total = total;
  }

  ~ProgressMonitor() {
    stop();
  }

  ProgressMonitor(const ProgressMonitor&) = delete;
  ProgressMonitor& operator=(const ProgressMonitor&) = delete;

  // Start the ~10 Hz render thread (live-ANSI mode only). In plain mode there
  // is no render thread — log lines are emitted inline from on_pass_stats at
  // the log_every cadence (so a piped/CI run shows steady progress).
  void start() {
    if (tty_) {
      running_.store(true);
      render_thr_ = std::thread([this] { render_loop(); });
    }
  }

  void stop() {
    if (running_.exchange(false) && render_thr_.joinable()) render_thr_.join();
    // A final frame so the last completed pass is always visible.
    render_once();
  }

  // The conveyor hook target: bind as
  //   co.on_pass = [&](long h, const core::PassStats& st){ mon.on_pass_stats(h, st); };
  // Minimal work under the lock; thread-safe for concurrent z-node calls.
  void on_pass_stats(long pass_h, const core::PassStats& st) {
    bool emit_log = false;
    {
      std::lock_guard lk(mu_);
      // Seed e0 from the first observed total energy if the caller did not set
      // it (the CLI does not pre-compute e0). The ring's true e0 is pe0+ke0 at
      // t0; pass 1's energy differs by one VV step, so the rel-drift readout is
      // approximate for the first frame and exact thereafter — it is a live
      // monitor, not the authoritative summary (that stays on stdout).
      if (snap_.e0 == 0.0 && !e0_seen_) {
        snap_.e0 = st.pe + st.ke;
        e0_seen_ = true;
      }
      // Keep the latest (highest) pass — z nodes complete out of order.
      if (pass_h >= snap_.pass) {
        snap_.pass = pass_h;
        snap_.pe = st.pe;
        snap_.ke = st.ke;
        snap_.dt = st.dt;
        snap_.v_max = st.v_max;
        // R_buf for the headroom readout: the buffer this pass would size from
        // the recorded v_max and dt (compute_R_buf == v_max*dt*C_buf). We do
        // not have C_buf here in PassStats, so headroom uses the pass v_max/dt
        // against the context R_buf seed if set; else recompute with C_buf=1
        // (a conservative upper headroom). Kept simple — it is a readout.
        if (c_buf_ > 0.0) snap_.r_buf = st.v_max * st.dt * c_buf_;
      }
      // Sparkline ring (bounded).
      spark_.push_back(st.pe + st.ke);
      if (spark_.size() > kSparkMax) spark_.pop_front();
      snap_.spark.assign(spark_.begin(), spark_.end());
      if (!tty_ && (pass_h % log_every_ == 0)) emit_log = true;
    }
    if (emit_log) render_logline_now();
  }

  // The direct stepper hook target: it has no PassStats, so the caller supplies
  // a fully-built Snapshot (energies/T/dt computed by the stepper's callback).
  void on_snapshot(const Snapshot& s) {
    bool emit_log = false;
    {
      std::lock_guard lk(mu_);
      const long total = snap_.total;
      snap_ = s;
      snap_.total = total;
      spark_.push_back(s.e_total());
      if (spark_.size() > kSparkMax) spark_.pop_front();
      snap_.spark.assign(spark_.begin(), spark_.end());
      if (!tty_ && (s.pass % log_every_ == 0)) emit_log = true;
    }
    if (emit_log) render_logline_now();
  }

  // Mark the run halted (drives the red HALT line on the final frame).
  void set_halt(const std::string& msg, const std::string& rescue_path = "") {
    std::lock_guard lk(mu_);
    snap_.halted = true;
    snap_.halt_msg = msg;
    snap_.rescue_path = rescue_path;
  }

  // C_buf is needed to turn (v_max,dt) into the headroom readout; set once.
  void set_c_buf(double c_buf) {
    std::lock_guard lk(mu_);
    c_buf_ = c_buf;
  }

private:
  static constexpr std::size_t kSparkMax = 60;

  Snapshot snapshot_copy() {
    std::lock_guard lk(mu_);
    return snap_;
  }

  void render_loop() {
    while (running_.load()) {
      render_once();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));  // ~10 Hz
    }
  }

  void render_once() {
    if (!tty_) return;  // plain mode renders via log lines, not frames
    const Snapshot s = snapshot_copy();
    if (s.pass == 0 && !s.halted) return;  // nothing yet
    const std::string frame = dash_.render_ansi(s);
    std::fwrite(frame.data(), 1, frame.size(), out_);
    std::fflush(out_);
  }

  void render_logline_now() {
    const Snapshot s = snapshot_copy();
    std::string line = dash_.render_logline(s) + "\n";
    std::fwrite(line.data(), 1, line.size(), out_);
    std::fflush(out_);
  }

  std::FILE* out_;
  bool tty_;
  Dashboard dash_;
  long log_every_;
  double c_buf_ = 0.0;

  std::mutex mu_;
  Snapshot snap_;
  std::deque<double> spark_;
  bool e0_seen_ = false;

  std::atomic<bool> running_{false};
  std::thread render_thr_;
};

}  // namespace tdmd::cli
