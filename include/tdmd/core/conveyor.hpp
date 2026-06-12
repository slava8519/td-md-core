#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "tdmd/core/buffer.hpp"
#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/fsm.hpp"
#include "tdmd/core/integrator.hpp"
#include "tdmd/core/simulation.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/transport.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/units.hpp"

// M3.5 — TimeConveyor-CPU: the TD ring (дисс. Гл. 2.1–2.2) of z software
// nodes on std::jthread + SPSC channels, orchestrated zone-by-zone through
// ZoneFSM. Node P_k computes pass (= model time step) h = k+1, k+1+z, …;
// a zone is sent to the next node immediately after its END, BEFORE the
// node starts the next zone (дисс. стр. ~1549) — per-zone pipelining.
//
// Per-pass per-zone cycle (letters per dissertation):
//   RECV   o→d  accept atoms (output of pass h−1); [ENG] the velocity-Verlet
//               first half (drift to x_h with dt(h)) + force-matrix clear are
//               data prep done here — d still means "not yet used in forces".
//   SPHERE d→w  partial cross forces land when the PREDECESSOR zone starts
//               computing (T2); the pass head gets the artificial seed (§7.1).
//   START  w→c  internal pairs + cross pairs with the NEXT zone (Newton-3,
//               INV-8: every adjacent pair counted exactly once).
//   END    c→p  forces complete -> fixed-point reduce (B1), second half kick,
//               zone-local v_max/a_max/K2 reductions (A8 — no global reduce),
//               INV-4 buffer check, static-membership guard.
//   SEND   p→o  ship to the next node (parity order §7.4: odd nodes send
//               first, even nodes receive first — anti-deadlock).
//
// PBC closure along z (§7.2/B3) — [ENG] ROTATION variant: the pass head is
// HELD in `c` until the tail computes; the closure pair tail↔head is then
// evaluated on the SAME node at the SAME pass positions x_h, the head ENDs
// last and is sent last. Hence the pass order rotates by one zone per step:
// order(h) = [r, r+1, …] mod n, r = (h−1) mod n. The evaluated pair set and
// positions are identical to the dissertation's cross-node partial-force
// packet (s_{n+1} ≡ s_1 of the next pass receiving SPHERE); what differs is
// transport only — no second message type, no force_complete=false transit,
// INV-3/INV-8 intact. Revisit in M4 if the GPU layout prefers the packet.
//
// Δt-handoff (B2, дисс. §3.3): the sender decides dt(h+1) and ships it with
// the FIRST zone it sends. Under full pipelining the head of pass h causally
// knows complete passes only up to h−(n−1) (the light cone of the wedge
// S_1..S_{1+m} of pass h−m), so the freshest FULL-pass v_max usable for a
// z-INDEPENDENT dt rule has lag n−1 (n=1: lag 1 ≡ the serial rule). The
// Λ-chain in ZoneHeader forwards exactly that: the j-th sent zone of pass h
// carries the aggregate of pass h−(n−1−j) (tail carries its own pass), so
//   dt(h+1) = auto_dt(v_max(h−n+1), dt(h), K2cap(h−n+1))
// is computable at first-send for ANY z — the acceptance criterion "bitwise
// 1 node vs z nodes incl. timestep.mode=auto" holds by construction. The
// n−1-step staleness is covered by C_buf and policed by INV-4 (the
// dissertation's own §3.6 ran C3=0, i.e. constant dt — the subtlety never
// surfaced there). INV-4 forecast: v_pred = v_full + a_full·dt·L, L=max(1,n−1)
// ([ENG] lag generalization of the M3 cold-start refinement).
//
// Static zone membership [ENG]: atoms are binned once at t0; migration is
// deferred to M4 (the zone payload format changes there anyway, B5). The
// honest guard: at END every member must lie within its slab ± (width−rcut)/2
// (pair completeness bound), else HALT StaleZone. n=2 with free z is a
// bipartition — always complete, no guard needed; n>=3 guarded.
//
// PairFn contract: void(double r, double& u, double& f_over_r), FP64, with
// the truncation scheme already applied (potentials/cutoff.hpp). It is
// invoked CONCURRENTLY from every node thread — it must be const-callable
// and stateless/thread-safe. The project's stateful potential drivers
// (last_min_r2/last_virial bookkeeping) are NOT suitable as-is: passing a
// mutable functor is a silent data race (TSan-verified in review).
//
// HALT semantics: the ring stops via an atomic flag + transport shutdown;
// the result keeps t0 atom state (no consistent mid-ring geometry exists —
// §9's rescue dump of the in-flight ring is an M4 work item) and the stats
// prefix of provably completed passes. FixedAccum overflow / FSM violations
// surface as Halt::Internal (the §9 rescue path), not as UB or assert.
namespace tdmd::core {

struct ConveyorOptions {
  long   steps = 0;
  int    n_zones = 1;
  int    n_nodes = 1;
  bool   auto_step = false;
  double dt_initial = 0.001;     // ps; also the fixed dt when !auto_step
  buffer::TimeStepCfg ts{};
  double r_min_halt = 0.5;       // Å, overlap HALT (B10)
  // production_mixed TRANSPORT (M4/B5, GPU ring only — this CPU conveyor is
  // always the deterministic_fp64 reference): zone payloads ship as int32
  // fixed-point offsets (power-of-two quantum; pack = one rint per send,
  // unpack exact). Pair-math precision is the functor's choice (LJDevF32/
  // MorseDevF32), independent of this flag.
  bool   mixed_transport = false;
};

// Per-pass record. v_max/a_max/k2cap are the pass aggregates that feed the
// Λ-chain — recorded so tests can replay the documented Δt recurrence
// dt(h+1) = auto_dt(v_max(h−n+1), dt(h), k2cap(h−n+1)) as an oracle.
// dt == 0 marks a never-written entry (dt of a completed pass is always > 0).
struct PassStats {
  double pe = 0.0, ke = 0.0, dt = 0.0;
  double v_max = 0.0, a_max = 0.0, k2cap = 0.0;
};

// On HALT: atoms keep their t0 state (a consistent mid-ring geometry does not
// exist — zones of several passes are in flight; the §9 rescue dump of the
// ring state is an M4 work item), stats hold the longest prefix of passes
// whose record was written — a pass that finished concurrently with the halt
// but aborted before its stats write is conservatively NOT counted.
struct ConveyorResult {
  long   steps_done = 0;
  Halt   halt = Halt::None;
  std::string halt_msg;
  double e0 = 0.0;               // pe0 + ke0 at t0
  std::vector<PassStats> stats;  // [h-1] for h = 1..steps_done
};

namespace conveyor_detail {

struct Lambda {
  double v = 0.0, a = 0.0, k2cap = 0.0;
};

struct Slot {
  Zone    fsm{};
  ZoneMsg msg;
  Lambda  lam_in{};
  std::vector<fixed::ForceAccum> ax, ay, az;  // B1 fixed-point force matrix
  bool present = false, drifted = false;
  bool computed = false;  // pair evaluation done this pass (INV-1 guard:
                          // FSM state alone won't do — SEND moves p->o
                          // before the successor's compute)
  double v_max = 0.0, a_max = 0.0, k2cap = 0.0;  // zone locals at END
};

}  // namespace conveyor_detail

template <typename Real, typename PairFn>
class TimeConveyor {
 public:
  TimeConveyor(AtomSoA<Real>& atoms, const Box& box, double rcut, PairFn pair,
               const ConveyorOptions& o)
      : atoms_(atoms), box_(box), rcut_(rcut), pair_(std::move(pair)), o_(o),
        geom_(box, rcut) {
    if (o_.steps < 1) throw std::invalid_argument("conveyor: steps must be >= 1");
    if (o_.n_nodes < 1) throw std::invalid_argument("conveyor: n_nodes must be >= 1");
    if (!(o_.dt_initial > 0.0))
      throw std::invalid_argument("conveyor: dt_initial must be > 0");
  }

  ConveyorResult run() {
    zd_ = ZoneDecomposition::build(atoms_, box_, o_.n_zones, rcut_);
    n_ = zd_.n_zones;
    z_ = o_.n_nodes;
    pbc_z_ = box_.periodic[2];

    // t0 forces via the serial w-pass (same pair core -> same bits) — needed
    // for the first drift; t0 scalars seed the Λ-chain pre-history.
    zero_forces(atoms_);
    const double pe0 = zone_force_pass(atoms_, box_, zd_, rcut_, pair_);
    res_.e0 = pe0 + kinetic_energy(atoms_);
    lam0_ = {buffer::max_speed(atoms_), buffer::max_accel(atoms_),
             buffer::temperature_limited_dt(atoms_, o_.ts.K2)};
    res_.stats.assign(std::size_t(o_.steps), {});
    final_.assign(std::size_t(n_), ZoneMsg{});

    // Pass-1 preload (§7.3: P1 starts with every zone in d, the rest of the
    // ring all-o = empty in-queues).
    std::vector<ZoneMsg> preload(static_cast<std::size_t>(n_));
    for (int j = 0; j < n_; ++j) preload[std::size_t(j)] = make_preload(j);

    transport_ = std::make_unique<RingTransport>(z_, std::size_t(n_) + 2);
    {
      std::vector<std::jthread> nodes;
      nodes.reserve(std::size_t(z_));
      try {
        for (int k = 0; k < z_; ++k)
          nodes.emplace_back([this, k, &preload] {
            node_main(k, k == 0 ? &preload : nullptr);
          });
      } catch (const std::exception& ex) {
        // Partial spawn (e.g. EAGAIN): without a halt the spawned nodes would
        // block forever in recv/send and ~jthread's join would hang run().
        set_halt(Halt::Internal,
                 std::string("conveyor: node spawn failed: ") + ex.what());
      }
    }  // jthreads join here

    if (halt_on_.load()) {
      res_.halt = halt_kind_;
      res_.halt_msg = halt_msg_;
      // Longest written prefix of stats == passes provably completed; the
      // halting pass (and any pass aborted before its stats write) is out.
      long done = 0;
      while (done < o_.steps && res_.stats[std::size_t(done)].dt > 0.0) ++done;
      res_.steps_done = done;
      res_.stats.resize(std::size_t(done));
    } else {
      res_.steps_done = o_.steps;
      scatter_final();
    }
    return res_;
  }

 private:
  using Slot = conveyor_detail::Slot;
  using Lambda = conveyor_detail::Lambda;

  // --- node thread ---------------------------------------------------------

  void node_main(int k, std::vector<ZoneMsg>* preload) {
    try {
      for (long h = k + 1; h <= o_.steps; h += z_) {
        if (halt_on_.load(std::memory_order_acquire)) return;
        if (!run_pass(k, h, h == 1 ? preload : nullptr)) return;
      }
    } catch (const std::exception& ex) {
      set_halt(Halt::Internal,
               std::string("conveyor internal error: ") + ex.what());
    }
  }

  bool run_pass(int k, long h, std::vector<ZoneMsg>* preload) {
    const int in_edge = (k - 1 + z_) % z_;
    const int out_edge = k;
    const auto io = node_io_order(k + 1);            // §7.4 parity (1-based)
    const int r = pbc_z_ ? int((h - 1) % n_) : 0;    // pass-order rotation
    const bool defer_head = pbc_z_ && n_ > 1;        // §7.2 closure ([ENG])

    std::vector<Slot> slot(static_cast<std::size_t>(n_));
    std::deque<int> outq;                            // ENDed, awaiting SEND
    int sent = 0, arrived = 0;
    double dt = 0.0, R_buf = 0.0, dt_next = 0.0;
    Lambda agg{0.0, 0.0, std::numeric_limits<double>::infinity()};
    fixed::EnergyAccum pe;
    double ke = 0.0;
    double min_r2 = std::numeric_limits<double>::infinity();

    auto fail = [&](Halt kind, const std::string& msg) {
      set_halt(kind, msg);
      return false;
    };

    // T1 (+ preload variant): accept arrivals up to position j.
    auto ensure_arrival = [&](int j) -> bool {
      while (arrived <= j) {
        Slot& s = slot[std::size_t(arrived)];
        if (preload) {
          s.msg = std::move((*preload)[std::size_t(arrived)]);
        } else if (!transport_->recv(in_edge, s.msg)) {
          return false;  // ring shut down (HALT elsewhere)
        }
        const int want_id = (r + arrived) % n_;
        if (s.msg.hdr.zone_id != want_id || s.msg.hdr.step_h != h - 1 ||
            s.msg.hdr.sent_pos != arrived)
          throw std::logic_error("conveyor: ring arrival out of order");
        s.lam_in = {s.msg.hdr.v_full, s.msg.hdr.a_full, s.msg.hdr.k2cap_full};
        s.fsm.id = want_id;
        s.fsm.step_h = h;
        s.fsm.n_atoms = s.msg.n();
        if (preload) {
          s.fsm.type = initial_zone_type(1);         // §7.3: P1 starts in d
        } else {
          s.fsm.type = ZoneType::o;
          ZoneFSM::apply(s.fsm, ZoneEvent::RECV);    // T1
        }
        s.present = true;
        if (arrived == 0) dt = s.msg.hdr.dt_next;    // B2: apply, don't recompute
        if (arrived == std::min(1, n_ - 1)) {
          // INV-4 forecast — freshest full pass in the head's light cone:
          // pass h-L, L = max(1, n-1). (The dt(h+1) DECISION happens later,
          // at the first SEND, from the same Λ source as Λ_out(0) — for n=1
          // that is this pass's own aggregate, reproducing the serial rule.)
          const Lambda& lf = slot[std::size_t(arrived)].lam_in;
          const double lag = (n_ == 1) ? 1.0 : double(n_ - 1);
          const double v_pred = lf.v + lf.a * dt * lag;   // [ENG] ramp cover
          R_buf = buffer::compute_R_buf(v_pred, dt, o_.ts.C_buf);
        }
        ++arrived;
      }
      return true;
    };

    // [ENG] data prep at RECV: velocity-Verlet first half (EXACT expressions
    // of integrator.hpp) + force-matrix clear (T1 action).
    auto ensure_drift = [&](int j) {
      Slot& s = slot[std::size_t(j)];
      if (s.drifted) return;
      const std::size_t m = std::size_t(s.msg.n());
      for (std::size_t i = 0; i < m; ++i) {
        const double inv_m = units::ftm2v / s.msg.mass[i];
        s.msg.vx[i] += 0.5 * dt * inv_m * s.msg.fx[i];
        s.msg.vy[i] += 0.5 * dt * inv_m * s.msg.fy[i];
        s.msg.vz[i] += 0.5 * dt * inv_m * s.msg.fz[i];
        s.msg.x[i] += dt * s.msg.vx[i];
        s.msg.y[i] += dt * s.msg.vy[i];
        s.msg.z[i] += dt * s.msg.vz[i];
      }
      s.ax.assign(m, {});
      s.ay.assign(m, {});
      s.az.assign(m, {});
      s.drifted = true;
    };

    // One evaluated pair — same FP core as the serial w-pass (PairGeom).
    auto do_pair = [&](Slot& A, int s, Slot& B, int t) {
      double dx = A.msg.x[std::size_t(s)] - B.msg.x[std::size_t(t)];
      double dy = A.msg.y[std::size_t(s)] - B.msg.y[std::size_t(t)];
      double dz = A.msg.z[std::size_t(s)] - B.msg.z[std::size_t(t)];
      double r2;
      const bool eval = geom_.reduce(dx, dy, dz, r2);
      // min_r2 BEFORE the acceptance cut (the direct drivers' order): a
      // coincident pair (r2 < 1e-18, skipped from evaluation) must still
      // trip the Overlap HALT, not masquerade as a healthy run (B10).
      min_r2 = std::min(min_r2, r2);
      if (!eval) return;
      double u, f_over_r;
      pair_(std::sqrt(r2), u, f_over_r);
      pe.add(u);
      A.ax[std::size_t(s)].add(f_over_r * dx);
      A.ay[std::size_t(s)].add(f_over_r * dy);
      A.az[std::size_t(s)].add(f_over_r * dz);
      B.ax[std::size_t(t)].add(-f_over_r * dx);
      B.ay[std::size_t(t)].add(-f_over_r * dy);
      B.az[std::size_t(t)].add(-f_over_r * dz);
    };

    // T4: reduce, integrate second half, zone-local reductions, checks.
    auto end_zone = [&](int j) -> bool {
      Slot& s = slot[std::size_t(j)];
      const std::size_t m = std::size_t(s.msg.n());
      if (min_r2 < o_.r_min_halt * o_.r_min_halt)
        return fail(Halt::Overlap,
                    "atom overlap at step " + std::to_string(h) +
                        ": min pair distance " +
                        std::to_string(std::sqrt(min_r2)) + " A < " +
                        std::to_string(o_.r_min_halt) + " A");
      for (std::size_t i = 0; i < m; ++i) {
        s.msg.fx[i] = s.ax[i].value();
        s.msg.fy[i] = s.ay[i].value();
        s.msg.fz[i] = s.az[i].value();
      }
      double v2 = 0.0, a2 = 0.0;
      double kcap = std::numeric_limits<double>::infinity();
      for (std::size_t i = 0; i < m; ++i) {
        const double inv_m = units::ftm2v / s.msg.mass[i];
        s.msg.vx[i] += 0.5 * dt * inv_m * s.msg.fx[i];
        s.msg.vy[i] += 0.5 * dt * inv_m * s.msg.fy[i];
        s.msg.vz[i] += 0.5 * dt * inv_m * s.msg.fz[i];
      }
      for (std::size_t i = 0; i < m; ++i) {
        const double vi2 =
            buffer::speed2(s.msg.vx[i], s.msg.vy[i], s.msg.vz[i]);
        v2 = std::max(v2, vi2);
        a2 = std::max(a2, buffer::accel2(s.msg.fx[i], s.msg.fy[i],
                                         s.msg.fz[i], s.msg.mass[i]));
        kcap = std::min(
            kcap, buffer::k2_limited_dt_atom(s.msg.fx[i], s.msg.fy[i],
                                             s.msg.fz[i], s.msg.vx[i],
                                             s.msg.vy[i], s.msg.vz[i],
                                             s.msg.mass[i], o_.ts.K2));
        ke += 0.5 * units::mvv2e * s.msg.mass[i] * vi2;
      }
      // INV-3 completeness ledger: by END the contribution mask must be FULL
      // for this zone's layout (T2 'mask |= prev'): bit0 = predecessor's
      // cross contribution, bit1 = PBC closure into the held head. A mismatch
      // is a forbidden state -> rescue path (§9) via Halt::Internal.
      const uint32_t want_mask = (j == 0) ? (defer_head ? 2u : 0u) : 1u;
      if (s.fsm.contrib_mask != want_mask)
        throw std::logic_error(
            "conveyor: INV-3 contribution mask incomplete at END");
      s.v_max = std::sqrt(v2);
      s.a_max = std::sqrt(a2);
      s.k2cap = kcap;
      s.fsm.v_max_local = s.v_max;
      s.fsm.R_buf_local = R_buf;
      agg.v = std::max(agg.v, s.v_max);
      agg.a = std::max(agg.a, s.a_max);
      agg.k2cap = std::min(agg.k2cap, s.k2cap);
      // INV-4 (eq. 33): no atom may cross the buffer this step.
      if (!buffer::causality_ok(s.v_max, dt, R_buf)) {
        char buf[176];
        std::snprintf(buf, sizeof(buf),
                      "causality (INV-4) at step %ld zone %d: "
                      "v_max*dt=%.4g > R_buf=%.4g (dt=%.17g)",
                      h, s.fsm.id, s.v_max * dt, R_buf, dt);
        return fail(Halt::Causality, buf);
      }
      // [ENG] static-membership guard (migration deferred to M4).
      if (n_ >= 3 && !membership_ok(s)) {
        return fail(Halt::StaleZone,
                    "stale zone membership at step " + std::to_string(h) +
                        " zone " + std::to_string(s.fsm.id) +
                        ": atom left its slab by more than (width-r_cut)/2; "
                        "atom migration is an M4 work item");
      }
      ZoneFSM::apply(s.fsm, ZoneEvent::END);  // T4: force_complete = true
      outq.push_back(j);
      return true;
    };

    // T5 + result collection on the final pass. Λ_out of the j-th SENT zone
    // is Λ_in of arrival j+1 of THIS pass (positional chain), the last sent
    // carries the pass aggregate. dt(h+1) is DECIDED here, at the first send
    // (B2: «решает узел-отправитель, передаётся с первой зоной»), from the
    // same Λ source as Λ_out(0): for n>=2 that is v_max(h-n+1); for n=1 it
    // is this pass's own aggregate — exactly the serial rule
    // dt(h+1)=auto_dt(v_max(h), dt(h)). NB: the FSM SEND event fires when
    // ownership is handed to the transport (message moved); delivery
    // confirmation (§4 T5 guard) is the bounded queue's backpressure.
    auto flush_sends = [&]() -> bool {
      while (!outq.empty()) {
        const int j = outq.front();
        outq.pop_front();
        Slot& s = slot[std::size_t(j)];
        s.ax = {};  // force matrix is dead past END — do not keep
        s.ay = {};  // O(N_total) of accumulators per node (INV-7 hygiene)
        s.az = {};
        if (h == o_.steps) {  // ring ends here — hand the zone to the result
          final_[std::size_t(s.fsm.id)] = std::move(s.msg);
          continue;
        }
        ZoneMsg m = std::move(s.msg);
        m.hdr.zone_id = s.fsm.id;
        m.hdr.step_h = h;
        m.hdr.sent_pos = sent;
        const Lambda lam =
            (sent + 1 < n_) ? slot[std::size_t(sent + 1)].lam_in : agg;
        if (sent == 0)
          dt_next = o_.auto_step ? buffer::auto_dt(lam.v, dt, o_.ts, lam.k2cap)
                                 : o_.dt_initial;
        m.hdr.dt_next = (sent == 0) ? dt_next : 0.0;
        m.hdr.v_full = lam.v;
        m.hdr.a_full = lam.a;
        m.hdr.k2cap_full = lam.k2cap;
        ZoneFSM::apply(s.fsm, ZoneEvent::SEND);  // T5: p -> o
        transport_->send(out_edge, std::move(m));
        ++sent;
      }
      return !halt_on_.load(std::memory_order_relaxed);
    };

    auto compute_zone = [&](int j) -> bool {
      if (!ensure_arrival(j)) return false;
      if (j + 1 < n_ && !ensure_arrival(j + 1)) return false;
      ensure_drift(j);
      if (j + 1 < n_) ensure_drift(j + 1);
      Slot& s = slot[std::size_t(j)];
      if (j == 0) seed_first_zone(s.fsm);  // §7.1 artificial SPHERE (d->w)
      // INV-1: the same-pass predecessor has already computed. The FSM state
      // alone cannot witness this (SEND legally moves p->o before our turn),
      // so a per-pass flag carries it; violation -> rescue path (§9).
      if (j > 0 && !slot[std::size_t(j - 1)].computed)
        throw std::logic_error(
            "conveyor: INV-1 violated — predecessor zone not computed");
      ZoneFSM::apply(s.fsm, ZoneEvent::START);  // T3: w -> c
      const std::size_t mj = std::size_t(s.msg.n());
      for (std::size_t a = 0; a < mj; ++a)      // internal pairs, a < b
        for (std::size_t b = a + 1; b < mj; ++b)
          do_pair(s, int(a), s, int(b));
      if (j + 1 < n_) {
        Slot& nx = slot[std::size_t(j + 1)];
        ZoneFSM::apply(nx.fsm, ZoneEvent::SPHERE);  // T2: d -> w
        nx.fsm.contrib_mask |= 1u;                  // T2: mask |= prev
        for (int a = 0; a < s.msg.n(); ++a)
          for (int b = 0; b < nx.msg.n(); ++b) do_pair(s, a, nx, b);
      } else if (defer_head && j == n_ - 1) {
        Slot& hd = slot[0];                     // §7.2 closure ([ENG] rotation)
        hd.fsm.contrib_mask |= 2u;              // closure contribution bit
        for (int a = 0; a < s.msg.n(); ++a)
          for (int b = 0; b < hd.msg.n(); ++b) do_pair(s, a, hd, b);
      }
      s.computed = true;
      // END: zone j is complete (its S_{j-1} cross landed at compute(j-1));
      // with PBC the head waits for the tail's closure and ENDs after it.
      if (!(defer_head && j == 0) && !end_zone(j)) return false;
      if (defer_head && j == n_ - 1 && !end_zone(0)) return false;
      return true;
    };

    // --- the pass: per-zone subintervals with §7.4 parity I/O ---
    for (int j = 0; j < n_; ++j) {
      for (IoOp op : io) {
        if (op == IoOp::SEND) {
          if (!flush_sends()) return false;
        } else {
          if (!ensure_arrival(std::min(j + 1, n_ - 1))) return false;
        }
      }
      if (halt_on_.load(std::memory_order_acquire)) return false;
      if (!compute_zone(j)) return false;
    }
    if (!flush_sends()) return false;

    const double pass_pe = pe.value();
    if (!std::isfinite(pass_pe + ke))
      return fail(Halt::NonFiniteEnergy,
                  "non-finite energy at step " + std::to_string(h));
    res_.stats[std::size_t(h - 1)] = {pass_pe, ke, dt, agg.v, agg.a, agg.k2cap};
    return true;
  }

  // --- helpers --------------------------------------------------------------

  ZoneMsg make_preload(int j) {
    ZoneMsg m;
    const auto& mem = zd_.members[std::size_t(j)];
    const std::size_t n = mem.size();
    m.id.reserve(n);   m.mass.reserve(n);
    m.x.reserve(n);    m.y.reserve(n);   m.z.reserve(n);
    m.vx.reserve(n);   m.vy.reserve(n);  m.vz.reserve(n);
    m.fx.reserve(n);   m.fy.reserve(n);  m.fz.reserve(n);
    for (int i : mem) {
      m.id.push_back(atoms_.id[std::size_t(i)]);
      m.mass.push_back(atoms_.mass[std::size_t(i)]);
      m.x.push_back(atoms_.x[std::size_t(i)]);
      m.y.push_back(atoms_.y[std::size_t(i)]);
      m.z.push_back(atoms_.z[std::size_t(i)]);
      m.vx.push_back(double(atoms_.vx[std::size_t(i)]));
      m.vy.push_back(double(atoms_.vy[std::size_t(i)]));
      m.vz.push_back(double(atoms_.vz[std::size_t(i)]));
      m.fx.push_back(atoms_.fx[std::size_t(i)]);
      m.fy.push_back(atoms_.fy[std::size_t(i)]);
      m.fz.push_back(atoms_.fz[std::size_t(i)]);
    }
    m.hdr.zone_id = j;
    m.hdr.step_h = 0;
    m.hdr.sent_pos = j;
    m.hdr.dt_next = (j == 0) ? o_.dt_initial : 0.0;  // dt(1)
    m.hdr.v_full = lam0_.v;                          // Λ pre-history = t0
    m.hdr.a_full = lam0_.a;
    m.hdr.k2cap_full = lam0_.k2cap;
    return m;
  }

  void scatter_final() {
    for (int zid = 0; zid < n_; ++zid) {
      const auto& mem = zd_.members[std::size_t(zid)];
      const ZoneMsg& m = final_[std::size_t(zid)];
      if (m.n() != int(mem.size()))
        throw std::logic_error("conveyor: final zone size mismatch");
      for (std::size_t t = 0; t < mem.size(); ++t) {
        const std::size_t i = std::size_t(mem[t]);
        if (atoms_.id[i] != m.id[t])
          throw std::logic_error("conveyor: final scatter id mismatch");
        atoms_.x[i] = m.x[t];
        atoms_.y[i] = m.y[t];
        atoms_.z[i] = m.z[t];
        atoms_.vx[i] = Real(m.vx[t]);
        atoms_.vy[i] = Real(m.vy[t]);
        atoms_.vz[i] = Real(m.vz[t]);
        atoms_.fx[i] = m.fx[t];
        atoms_.fy[i] = m.fy[t];
        atoms_.fz[i] = m.fz[t];
      }
    }
  }

  // Pair-completeness guard for static membership: a member of zone `id`
  // must stay within its slab ± (width−rcut)/2. Outer slabs of a free axis
  // are unbounded outward; n==2 free is a bipartition (no guard — caller
  // skips n_<3); PBC distances are cyclic.
  bool membership_ok(const Slot& s) const {
    const double w = zd_.width;
    const double g = 0.5 * (w - rcut_);
    const double lo_box = box_.lo[2], Lz = box_.len(2);
    const double lo = lo_box + s.fsm.id * w;
    const double hi = lo + w;
    const bool first = (s.fsm.id == 0), last = (s.fsm.id == n_ - 1);
    for (double zc : s.msg.z) {
      double zw = zc;
      if (pbc_z_) zw -= Lz * std::floor((zw - lo_box) / Lz);
      double excess = 0.0;
      if (zw < lo) {
        excess = lo - zw;
        if (pbc_z_) excess = std::min(excess, zw + Lz - hi);
        else if (first) excess = 0.0;
      } else if (zw > hi) {
        excess = zw - hi;
        if (pbc_z_) excess = std::min(excess, lo + Lz - zw);
        else if (last) excess = 0.0;
      }
      if (excess > g) return false;
    }
    return true;
  }

  void set_halt(Halt kind, std::string msg) {
    bool expected = false;
    if (halt_on_.compare_exchange_strong(expected, true)) {
      std::lock_guard lk(halt_mu_);
      halt_kind_ = kind;
      halt_msg_ = std::move(msg);
      transport_->shutdown();
    }
  }

  AtomSoA<Real>& atoms_;
  const Box& box_;
  const double rcut_;
  PairFn pair_;
  const ConveyorOptions o_;
  const PairGeom geom_;

  ZoneDecomposition zd_;
  int n_ = 1, z_ = 1;
  bool pbc_z_ = false;
  conveyor_detail::Lambda lam0_{};
  std::unique_ptr<RingTransport> transport_;
  std::vector<ZoneMsg> final_;
  ConveyorResult res_;

  std::atomic<bool> halt_on_{false};
  std::mutex halt_mu_;
  Halt halt_kind_ = Halt::None;
  std::string halt_msg_;
};

template <typename Real, typename PairFn>
ConveyorResult run_conveyor(AtomSoA<Real>& atoms, const Box& box, double rcut,
                            PairFn pair, const ConveyorOptions& o) {
  TimeConveyor<Real, PairFn> tc(atoms, box, rcut, std::move(pair), o);
  return tc.run();
}

}  // namespace tdmd::core
