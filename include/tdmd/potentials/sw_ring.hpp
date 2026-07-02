#pragma once
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "tdmd/core/buffer.hpp"
#include "tdmd/core/conveyor.hpp"     // ConveyorOptions, ConveyorResult, Halt, PassStats (reused verbatim)
#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/fsm.hpp"
#include "tdmd/core/integrator.hpp"  // kinetic_energy
#include "tdmd/core/soa.hpp"
#include "tdmd/core/transport.hpp"
#include "tdmd/units.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/potentials/eam.hpp"
#include "tdmd/potentials/eam_zone.hpp"  // zone_eam_window / eam_window_layout (reused verbatim)
#include "tdmd/potentials/sw.hpp"        // SwParams, SwPotential, sw_triplet
#include "tdmd/potentials/sw_zone.hpp"   // sw_window_force, sw_zone_pass

// M6 / SW-ladder T3b — SwRing: threaded Stillinger-Weber angular TD ring. Design
// of record: wf_4752ee3c-cde. A SIBLING FORK of eam_ring.hpp (EamRing was itself
// forked from the pair TimeConveyor — the house style): the orchestration body
// (threads, SPSC channels, send-delay/second-forward-hop, FSM, Λ-chain dt handoff,
// PBC defer_head, the unsorted slot-order window gather, finalize/scatter) is
// COPIED CHARACTER-FOR-CHARACTER from EamRing. `diff eam_ring.hpp sw_ring.hpp`
// shows ONLY the five EAM→SW swaps + SwWinForce — an INSPECTION-CHECKABLE proof
// that the EAM ring path is byte-untouched (stronger than a codegen-equivalence
// argument over a refactored template; EamRing/EamGpuRing stay bitwise, F-NOOP).
//
// The five swaps (the entire delta): (1) pot_ type EamPotential→SwPotential (no
// Math param); (2) rcut_ = pot.sw.rcut() (a METHOD); (3) fb_/density DELETED (SW
// has no density pass); (4) rho_cap = 0.0, passed and IGNORED by SwWinForce; (5)
// t0 forces via sw_zone_pass. The firewall line is copied VERBATIM — for SW it
// fires (SwWinForce HAS assert_supported) and ACCEPTS needs_transpose (the φ₃
// transpose-replay IS the correct mechanism — the INVERSE of the GPU EAM firewall).
//
// Per owned zone S_j the node finalizes the CENTER zone over the three-zone window
// {S_{j-1}, S_j, S_{j+1}} (SW force range 2·rcut, effective_range={2,true}=EAM's),
// resident via the one-zone send delay. The per-window force is sw_window_force
// (the transpose-replay), byte-identical to the serial sw_zone_pass oracle. The
// ring window is gathered UNSORTED (slot order); the φ₂ LOWER-GLOBAL-KEY gate is
// DEFENSIVE (robust to any gather reorder) but — MEASURED, overturning the T3
// acceptance's Gap-A claim — NOT load-bearing here: a φ₂ pair spans only adjacent
// zones and the slot order is zone-consistent in every window, so the local index
// also counts each pair once (test_sw_ring proves poison PE bitwise == correct).
namespace tdmd::potentials {

using core::AtomSoA;
using core::Box;
using core::ConveyorOptions;
using core::ConveyorResult;
using core::Halt;
using core::ITransport;
using core::RingTransport;
using core::ZoneMsg;

// Default (CPU) SW window-force policy: a thin, STATELESS wrapper over
// sw_window_force (the transpose-replay). compute() matches the EamRing 14-arg
// const signature EXACTLY so the orchestration's gather/finalize/scatter body is
// the verbatim copy; rho_cap is named-discarded (SW has no density), and
// sw_window_force's 15th out-param n_triplets is sunk into a local (the MB2 count
// witness lives on the SERIAL path — never on the shared ring policy, which would
// race across z jthreads). STATELESS-except-const-ptr is what makes 1-vs-z bitwise.
template <typename Real>
struct SwWinForce {
  const SwParams* sp = nullptr;  // owned by SwPotential; pot_ outlives the ring
  explicit SwWinForce(const SwParams& s) : sp(&s) {}

  void compute(const double* wx, const double* wy, const double* wz, const long* key,
               int m, const int* owned, int n_owned, const core::PairGeom& geom,
               double /*rho_cap — SW has no density*/,
               std::vector<core::fixed::ForceAccum>& wFx,
               std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz,
               core::fixed::EnergyAccum& pe, double& min_r2) const {
    long n_triplets_sink = 0;  // serial-only A2 witness; not threaded through the ring
    sw_window_force<Real>(wx, wy, wz, key, m, owned, n_owned, *sp, geom, wFx, wFy, wFz,
                          pe, min_r2, n_triplets_sink);
  }

  // FIREWALL — the INVERSION of GpuEamWindowForce::assert_supported (which THROWS on
  // needs_transpose + demands 3 passes). SW's φ₃ IS needs_transpose and the CPU
  // transpose-replay IS the correct mechanism ⇒ ACCEPT it. Reject ONLY: wrong pass
  // count, non-Force kind, iterative, and a mislabelled SYMMETRIC-only descriptor (the
  // door against a future GPU symmetric accumulator silently running SW). This also
  // closes the FIREWALL GAP at the SW seam from day one (the CPU policy asserts its own
  // descriptor, unlike CpuEamWindowForce which has none).
  static void assert_supported(std::span<const potentials::PassDecl> passes) {
    if (passes.size() != 2)
      throw std::runtime_error("SwWinForce: SW is the 2-pass [φ2 Force, φ3 Force] "
          "sequence — got " + std::to_string(passes.size()) + " passes");
    for (const auto& p : passes) {
      if (p.kind != potentials::PassKind::Force)
        throw std::runtime_error("SwWinForce: only PassKind::Force is implemented "
            "(SW φ2 pair + φ3 triplet); Density/Embedding/BondOrder/QeqIter are not this policy");
      if (p.iterative)
        throw std::runtime_error("SwWinForce: iterative pass (QEq/CG, ReaxFF) unimplemented");
    }
    if (passes[0].needs_transpose || !passes[1].needs_transpose)
      throw std::runtime_error("SwWinForce: expected pass 0 symmetric (φ2), pass 1 "
          "needs_transpose (φ3) — descriptor does not match the SW force structure (a "
          "symmetric-only SW would silently run a symmetric accumulator wrong)");
  }
};
static_assert(std::is_trivially_copyable_v<SwWinForce<double>>,
              "SwWinForce must stay stateless (shared across z jthreads) — no mutable "
              "diagnostic member, or 1-vs-z bitwise breaks via a data race");

// POISON policy (test-only): φ₂ energy gated on the LOCAL index instead of the global
// key. Used by test_sw_ring to MEASURE whether the global key is load-bearing on the
// unsorted ring window — the answer (count-once-invariant ⇒ poison PE bitwise == correct)
// overturned the design's premise. Kept as the evidence for that finding.
template <typename Real>
struct SwWinForceLocalKey {
  const SwParams* sp = nullptr;
  explicit SwWinForceLocalKey(const SwParams& s) : sp(&s) {}
  void compute(const double* wx, const double* wy, const double* wz, const long* /*key*/,
               int m, const int* owned, int n_owned, const core::PairGeom& geom, double,
               std::vector<core::fixed::ForceAccum>& wFx,
               std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz,
               core::fixed::EnergyAccum& pe, double& min_r2) const {
    // rebuild the LOCAL-index key (0..m-1) — the exact Gap-A bug on the ring window.
    std::vector<long> localkey(m);
    for (int i = 0; i < m; ++i) localkey[i] = i;
    long sink = 0;
    sw_window_force<Real>(wx, wy, wz, localkey.data(), m, owned, n_owned, *sp, geom,
                          wFx, wFy, wFz, pe, min_r2, sink);
  }
  // PR-0a: the poison differs from SwWinForce ONLY in compute() (the local-key gating);
  // its descriptor capability is identical ⇒ delegate (the concept mandates presence).
  static void assert_supported(std::span<const potentials::PassDecl> passes) {
    SwWinForce<Real>::assert_supported(passes);
  }
};

template <typename Real, typename WinForce = SwWinForce<Real>>
class SwRing {
  static_assert(potentials::WindowForcePolicy<WinForce>,
      "WinForce must model WindowForcePolicy (static assert_supported + 14-arg const "
      "compute) — the opt-in `if constexpr requires` firewall was silently bypassable "
      "(PR-0a; see many_body.hpp)");

 public:
  // The default-policy ctor (CPU): builds the policy from pot.sw (a SwParams).
  SwRing(AtomSoA<Real>& atoms, const Box& box, const SwPotential<Real>& pot,
         const ConveyorOptions& o)
      : SwRing(atoms, box, pot, o, WinForce(pot.sw)) {}

  // Policy-injected ctor (GPU / custom): the caller supplies the window-force policy
  // (e.g. a GpuSwWinForce holding the device transpose accumulator — T5).
  SwRing(AtomSoA<Real>& atoms, const Box& box, const SwPotential<Real>& pot,
         const ConveyorOptions& o, WinForce winforce)
      : atoms_(atoms), box_(box), pot_(pot), o_(o), rcut_(pot.sw.rcut()),
        winforce_(std::move(winforce)) {
    if (o_.steps < 1) throw std::invalid_argument("sw_ring: steps must be >= 1");
    if (o_.n_nodes < 1) throw std::invalid_argument("sw_ring: n_nodes must be >= 1");
    if (!(o_.dt_initial > 0.0)) throw std::invalid_argument("sw_ring: dt_initial > 0");
    // SW two-wing min-image guard (the ring calls sw_window_force directly, BYPASSING
    // sw_zone_pass's guard): every periodic box dim must exceed 2·rcut, else a wing bond
    // could wrap to a wrong image (deterministically — invisible to 1-vs-z).
    for (int d = 0; d < 3; ++d)
      if (box.periodic[d] && box.len(d) < 2.0 * rcut_)
        throw std::invalid_argument("sw_ring: periodic box dim < 2·rcut — min-image ambiguous");
    // periodic-z is supported: cyclic window + defer_head + tail-batched sends.
    // ZoneDecomposition::build rejects periodic n_zones in 2..4 for reach_mult=2, so PBC
    // only reaches n=1 (free path) or n>=5 (distinct cyclic zones — no window dedup needed).
  }

  ConveyorResult run() {
    zd_ = core::ZoneDecomposition::build(atoms_, box_, o_.n_zones, rcut_, /*reach_mult=*/2);
    n_ = zd_.n_zones;
    z_ = o_.n_nodes;
    // descriptor firewall (PR-0a: UNCONDITIONAL — the concept guarantees the hooks
    // exist). validate_pass_decls checks descriptor self-consistency (W-teeth);
    // assert_supported checks the policy's capability. SW's φ₃ IS needs_transpose and
    // SwWinForce ACCEPTS it (the transpose-replay is the correct mechanism) ⇒ no-op on
    // legal SW ⇒ F-NOOP.
    potentials::validate_pass_decls(pot_.passes());
    WinForce::assert_supported(pot_.passes());

    // t0 forces via the serial oracle (same kernel ⇒ same bits) for the 1st drift.
    core::zero_forces(atoms_);
    const double pe0 = sw_zone_pass(atoms_, box_, zd_, pot_.sw).pe;
    res_.e0 = pe0 + core::kinetic_energy(atoms_);
    lam0_ = {core::buffer::max_speed(atoms_), core::buffer::max_accel(atoms_),
             core::buffer::temperature_limited_dt(atoms_, o_.ts.K2)};
    res_.stats.assign(std::size_t(o_.steps), {});
    final_.assign(std::size_t(n_), ZoneMsg{});

    std::vector<ZoneMsg> preload(static_cast<std::size_t>(n_));
    for (int j = 0; j < n_; ++j) preload[std::size_t(j)] = make_preload(j);

    // capacity n_+3: +1 over the pair ring for the one extra in-flight delayed
    // zone (conservative slack — the tail step is the load-bearing liveness; §5).
    transport_ = std::make_unique<RingTransport>(z_, std::size_t(n_) + 3);
    {
      std::vector<std::jthread> nodes;
      nodes.reserve(std::size_t(z_));
      for (int k = 0; k < z_; ++k)
        nodes.emplace_back([this, k, &preload] { node_main(k, k == 0 ? &preload : nullptr); });
    }  // join

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
      res_.has_final = true;
    }
    return res_;
  }

 private:
  struct Slot {
    core::Zone fsm{};
    ZoneMsg msg;
    core::conveyor_detail::Lambda lam_in{};
    bool present = false, drifted = false, finalized = false, sent = false;
    double v_max = 0.0, a_max = 0.0, k2cap = 0.0;
  };

  void node_main(int k, std::vector<ZoneMsg>* preload) {
    try {
      for (long h = k + 1; h <= o_.steps; h += z_) {
        if (halt_on_.load(std::memory_order_acquire)) return;
        if (!run_pass(k, h, h == 1 ? preload : nullptr)) return;
      }
    } catch (const std::exception& ex) {
      set_halt(Halt::Internal, std::string("eam_ring internal error: ") + ex.what());
    }
  }

  bool run_pass(int k, long h, std::vector<ZoneMsg>* preload) {
    const int in_edge = (k - 1 + z_) % z_;
    const int out_edge = k;
    const auto io = core::node_io_order(k + 1);  // §7.4 parity (1-based)
    const int r = box_.periodic[2] ? int((h - 1) % n_) : 0;  // pass-order rotation
    const bool defer_head = box_.periodic[2] && n_ > 1;      // §7.2 closure ([ENG])
    const core::PairGeom geom(box_, rcut_);
    const double rho_cap = 0.0;  // SW has no density — passed and ignored by SwWinForce

    std::vector<Slot> slot(static_cast<std::size_t>(n_));
    int arrived = 0, sent = 0;
    double dt = 0.0, R_buf = 0.0, dt_next = 0.0;
    core::conveyor_detail::Lambda agg{0.0, 0.0, std::numeric_limits<double>::infinity()};
    core::fixed::EnergyAccum pe;
    double ke = 0.0, min_r2 = std::numeric_limits<double>::infinity();

    auto fail = [&](Halt kind, const std::string& msg) { set_halt(kind, msg); return false; };

    auto ensure_arrival = [&](int j) -> bool {
      while (arrived <= j) {
        Slot& s = slot[std::size_t(arrived)];
        if (preload) {
          s.msg = std::move((*preload)[std::size_t(arrived)]);
        } else if (!transport_->recv(in_edge, s.msg)) {
          return false;
        }
        const int want_id = (r + arrived) % n_;  // PBC: pass rotation (free-z: r=0)
        if (s.msg.hdr.zone_id != want_id || s.msg.hdr.step_h != h - 1 ||
            s.msg.hdr.sent_pos != arrived)
          throw std::logic_error("eam_ring: ring arrival out of order");
        s.lam_in = {s.msg.hdr.v_full, s.msg.hdr.a_full, s.msg.hdr.k2cap_full};
        s.fsm.id = want_id;
        s.fsm.step_h = h;
        s.fsm.n_atoms = s.msg.n();
        if (preload) {
          s.fsm.type = core::initial_zone_type(1);
        } else {
          s.fsm.type = core::ZoneType::o;
          core::ZoneFSM::apply(s.fsm, core::ZoneEvent::RECV);
        }
        s.present = true;
        if (arrived == 0) dt = s.msg.hdr.dt_next;
        if (arrived == std::min(1, n_ - 1)) {
          const core::conveyor_detail::Lambda& lf = slot[std::size_t(arrived)].lam_in;
          const double lag = (n_ == 1) ? 1.0 : double(n_ - 1);
          const double v_pred = lf.v + lf.a * dt * lag;
          R_buf = core::buffer::compute_R_buf(v_pred, dt, o_.ts.C_buf);
        }
        ++arrived;
      }
      return true;
    };

    auto ensure_drift = [&](int j) {
      if (j < 0 || j >= n_) return;
      Slot& s = slot[std::size_t(j)];
      if (!s.present || s.drifted) return;
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
      s.drifted = true;
    };

    // T3 START + force store + second-half kick + zone-local reductions + checks.
    auto end_eam = [&](int j, const std::vector<int>& ownedloc,
                       std::vector<core::fixed::ForceAccum>& wFx,
                       std::vector<core::fixed::ForceAccum>& wFy,
                       std::vector<core::fixed::ForceAccum>& wFz) -> bool {
      Slot& s = slot[std::size_t(j)];
      // EAM has no pair-cross "partial force into successor", so EVERY zone's
      // d->w SPHERE is artificial (its symmetric window data is what's ready),
      // not just zone 0's §7.1 seed. Apply it here for all owned zones.
      core::ZoneFSM::apply(s.fsm, core::ZoneEvent::SPHERE);  // d -> w
      core::ZoneFSM::apply(s.fsm, core::ZoneEvent::START);   // w -> c
      const std::size_t mj = std::size_t(s.msg.n());
      if (min_r2 < o_.r_min_halt * o_.r_min_halt)
        return fail(Halt::Overlap, "atom overlap at step " + std::to_string(h) +
                                       ": min dist " + std::to_string(std::sqrt(min_r2)));
      for (std::size_t i = 0; i < mj; ++i) {  // window-local owned index t -> slot-local i
        const int loc = ownedloc[i];
        s.msg.fx[i] = wFx[std::size_t(loc)].value();
        s.msg.fy[i] = wFy[std::size_t(loc)].value();
        s.msg.fz[i] = wFz[std::size_t(loc)].value();
      }
      double v2 = 0.0, a2 = 0.0, kcap = std::numeric_limits<double>::infinity();
      for (std::size_t i = 0; i < mj; ++i) {
        const double inv_m = units::ftm2v / s.msg.mass[i];
        s.msg.vx[i] += 0.5 * dt * inv_m * s.msg.fx[i];
        s.msg.vy[i] += 0.5 * dt * inv_m * s.msg.fy[i];
        s.msg.vz[i] += 0.5 * dt * inv_m * s.msg.fz[i];
      }
      for (std::size_t i = 0; i < mj; ++i) {
        const double vi2 = core::buffer::speed2(s.msg.vx[i], s.msg.vy[i], s.msg.vz[i]);
        v2 = std::max(v2, vi2);
        a2 = std::max(a2, core::buffer::accel2(s.msg.fx[i], s.msg.fy[i], s.msg.fz[i], s.msg.mass[i]));
        kcap = std::min(kcap, core::buffer::k2_limited_dt_atom(
                                  s.msg.fx[i], s.msg.fy[i], s.msg.fz[i], s.msg.vx[i],
                                  s.msg.vy[i], s.msg.vz[i], s.msg.mass[i], o_.ts.K2));
        ke += 0.5 * units::mvv2e * s.msg.mass[i] * vi2;
      }
      s.v_max = std::sqrt(v2); s.a_max = std::sqrt(a2); s.k2cap = kcap;
      agg.v = std::max(agg.v, s.v_max);
      agg.a = std::max(agg.a, s.a_max);
      agg.k2cap = std::min(agg.k2cap, s.k2cap);
      if (!core::buffer::causality_ok(s.v_max, dt, R_buf)) {
        char buf[176];
        std::snprintf(buf, sizeof(buf),
                      "causality (INV-4) at step %ld zone %d: v_max*dt=%.4g > R_buf=%.4g",
                      h, s.fsm.id, s.v_max * dt, R_buf);
        return fail(Halt::Causality, buf);
      }
      if (n_ >= 3 && !membership_ok(s))
        return fail(Halt::StaleZone, "stale zone membership at step " +
                                         std::to_string(h) + " zone " + std::to_string(s.fsm.id));
      core::ZoneFSM::apply(s.fsm, core::ZoneEvent::END);  // c -> p
      s.finalized = true;
      return true;
    };

    // finalize CENTER zone j over the resident window {j-1, j, j+1} (free-z).
    auto finalize_owned = [&](int j) -> bool {
      // window slot positions — the SINGLE source of truth shared with the GPU
      // EamGpuConveyor gather (E5b F4): PBC cyclic {(j-1)%n,j,(j+1)%n} (n>=5 ⇒
      // distinct), free-z drop out-of-range. (PR-E3b-PBC; factored E5b.)
      int wslots[3];
      const int nw = eam_window_layout(j, n_, box_.periodic[2], wslots);
      // self-drift the cyclic members: the scan-boundary ensure_drift(j-1,j,j+1)
      // clamps out-of-range, so the wrap neighbour (owned 0 / n-1) isn't drifted
      // there. ensure_drift is idempotent (checks s.drifted).
      for (int t = 0; t < nw; ++t) ensure_drift(wslots[t]);
      // residence-precondition (adversarial fix): all window slots live.
      for (int t = 0; t < nw; ++t) {
        const Slot& w = slot[std::size_t(wslots[t])];
        if (!w.present || !w.drifted)  // empty (n()==0) slot is legitimately resident
          throw std::logic_error("eam_ring: window slot not resident at finalize");
      }
      // gather contiguous window (key = atom id for the φ-once order)
      std::vector<double> wx, wy, wz;
      std::vector<long> key;
      std::vector<int> ownedloc;
      for (int t = 0; t < nw; ++t) {
        const Slot& w = slot[std::size_t(wslots[t])];
        const int base = int(wx.size());
        for (int i = 0; i < w.msg.n(); ++i) {
          wx.push_back(w.msg.x[std::size_t(i)]);
          wy.push_back(w.msg.y[std::size_t(i)]);
          wz.push_back(w.msg.z[std::size_t(i)]);
          key.push_back(w.msg.id[std::size_t(i)]);
        }
        if (wslots[t] == j)
          for (int i = 0; i < w.msg.n(); ++i) ownedloc.push_back(base + i);
      }
      const int m = int(wx.size());
      std::vector<core::fixed::ForceAccum> wFx(m), wFy(m), wFz(m);
      // M6 E5b-3b: the SINGLE moving part — delegate to the window-force policy
      // (CPU default = eam_window_force; GPU = the E5 kernels). pe/min_r2 are
      // updated exactly as before; the int64 force result is order-free.
      winforce_.compute(wx.data(), wy.data(), wz.data(), key.data(), m,
                        ownedloc.data(), int(ownedloc.size()), geom, rho_cap, wFx,
                        wFy, wFz, pe, min_r2);
      return end_eam(j, ownedloc, wFx, wFy, wFz);
    };

    // send slot j (must be finalized); Λ-chain handoff (B2) on the first send.
    auto send_slot = [&](int j) -> bool {
      Slot& s = slot[std::size_t(j)];
      if (s.sent) return true;
      if (h == o_.steps) { final_[std::size_t(s.fsm.id)] = std::move(s.msg); s.sent = true; return true; }
      ZoneMsg m = std::move(s.msg);
      m.hdr.zone_id = s.fsm.id;
      m.hdr.step_h = h;
      m.hdr.sent_pos = sent;
      const core::conveyor_detail::Lambda lam = (sent + 1 < n_) ? slot[std::size_t(sent + 1)].lam_in : agg;
      if (sent == 0)
        dt_next = o_.auto_step ? core::buffer::auto_dt(lam.v, dt, o_.ts, lam.k2cap) : o_.dt_initial;
      m.hdr.dt_next = (sent == 0) ? dt_next : 0.0;
      m.hdr.v_full = lam.v; m.hdr.a_full = lam.a; m.hdr.k2cap_full = lam.k2cap;
      core::ZoneFSM::apply(s.fsm, core::ZoneEvent::SEND);  // p -> o
      transport_->send(out_edge, std::move(m));
      s.sent = true;
      ++sent;
      return !halt_on_.load(std::memory_order_relaxed);
    };

    // --- the pass. free-z: finalize CENTER j at scan j, send the DELAYED j-1.
    // PBC (defer_head): finalize owned 1..n-1 in-scan, defer owned 0 to the tail
    // (its cyclic window {n-1,0,1} needs slot n-1 = last arrival); ALL sends are
    // tail-batched in slot order 1,2,...,n-1,0 (head last). ---
    for (int j = 0; j < n_; ++j) {
      for (core::IoOp op : io) {
        if (op == core::IoOp::SEND) {
          // free-z: delayed in-scan send (j-2 finalized at scan j-1).
          // defer_head: scan SEND is a NO-OP (sends are tail-batched).
          if (!defer_head && j >= 2 && !send_slot(j - 2)) return false;
        } else {
          if (!ensure_arrival(std::min(j + 1, n_ - 1))) return false;
        }
      }
      if (halt_on_.load(std::memory_order_acquire)) return false;
      if (!ensure_arrival(j)) return false;
      if (j + 1 < n_ && !ensure_arrival(j + 1)) return false;
      ensure_drift(j - 1); ensure_drift(j); ensure_drift(j + 1);
      if (!(defer_head && j == 0))  // PBC: owned 0 deferred to the tail
        if (!finalize_owned(j)) return false;
    }
    if (defer_head) {
      // owned 0 FIRST (reads slots n-1,0,1 + contributes to agg) — then send
      // 1,2,...,n-1, then the head (slot 0) LAST at sent_pos n-1.
      if (!finalize_owned(0)) return false;
      for (int p = 1; p < n_; ++p)
        if (!send_slot(p)) return false;
      if (!send_slot(0)) return false;
    } else {
      // free-z: the last two finalized zones (n-2, n-1) are still unsent.
      if (n_ >= 2 && !send_slot(n_ - 2)) return false;
      if (!send_slot(n_ - 1)) return false;
    }

    const double pass_pe = pe.value();
    if (!std::isfinite(pass_pe + ke))
      return fail(Halt::NonFiniteEnergy, "non-finite energy at step " + std::to_string(h));
    res_.stats[std::size_t(h - 1)] = {pass_pe, ke, dt, agg.v, agg.a, agg.k2cap};
    return true;
  }

  // --- helpers (mirror the pair conveyor) ---
  ZoneMsg make_preload(int j) {
    ZoneMsg m;
    const auto& mem = zd_.members[std::size_t(j)];
    for (int i : mem) {
      m.id.push_back(atoms_.id[std::size_t(i)]);
      m.mass.push_back(atoms_.mass[std::size_t(i)]);
      m.x.push_back(atoms_.x[std::size_t(i)]); m.y.push_back(atoms_.y[std::size_t(i)]); m.z.push_back(atoms_.z[std::size_t(i)]);
      m.vx.push_back(double(atoms_.vx[std::size_t(i)])); m.vy.push_back(double(atoms_.vy[std::size_t(i)])); m.vz.push_back(double(atoms_.vz[std::size_t(i)]));
      m.fx.push_back(atoms_.fx[std::size_t(i)]); m.fy.push_back(atoms_.fy[std::size_t(i)]); m.fz.push_back(atoms_.fz[std::size_t(i)]);
    }
    m.hdr.zone_id = j; m.hdr.step_h = 0; m.hdr.sent_pos = j;
    m.hdr.dt_next = (j == 0) ? o_.dt_initial : 0.0;
    m.hdr.v_full = lam0_.v; m.hdr.a_full = lam0_.a; m.hdr.k2cap_full = lam0_.k2cap;
    return m;
  }

  void scatter_final() {
    for (int zid = 0; zid < n_; ++zid) {
      const auto& mem = zd_.members[std::size_t(zid)];
      const ZoneMsg& m = final_[std::size_t(zid)];
      if (m.n() != int(mem.size())) throw std::logic_error("eam_ring: final zone size mismatch");
      for (std::size_t t = 0; t < mem.size(); ++t) {
        const std::size_t i = std::size_t(mem[t]);
        atoms_.x[i] = m.x[t]; atoms_.y[i] = m.y[t]; atoms_.z[i] = m.z[t];
        atoms_.vx[i] = Real(m.vx[t]); atoms_.vy[i] = Real(m.vy[t]); atoms_.vz[i] = Real(m.vz[t]);
        atoms_.fx[i] = m.fx[t]; atoms_.fy[i] = m.fy[t]; atoms_.fz[i] = m.fz[t];
      }
    }
  }

  bool membership_ok(const Slot& s) const {
    const double w = zd_.width;
    const double g = 0.5 * (w - 2.0 * rcut_);  // many-body reach (reach_mult=2)
    const double lo_box = box_.lo[2], Lz = box_.len(2);
    const double lo = lo_box + s.fsm.id * w, hi = lo + w;
    const bool first = (s.fsm.id == 0), last = (s.fsm.id == n_ - 1);
    for (double zc : s.msg.z) {
      double zw = zc;
      if (box_.periodic[2]) zw -= Lz * std::floor((zw - lo_box) / Lz);  // wrap
      double excess = 0.0;
      if (zw < lo) {
        excess = lo - zw;
        if (box_.periodic[2]) excess = std::min(excess, zw + Lz - hi);  // cyclic min
        else if (first) excess = 0.0;
      } else if (zw > hi) {
        excess = zw - hi;
        if (box_.periodic[2]) excess = std::min(excess, lo + Lz - zw);
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
      halt_kind_ = kind; halt_msg_ = std::move(msg);
      transport_->shutdown();
    }
  }

  AtomSoA<Real>& atoms_;
  const Box& box_;
  const SwPotential<Real>& pot_;
  ConveyorOptions o_;
  double rcut_;
  WinForce winforce_;
  core::ZoneDecomposition zd_;
  int n_ = 0, z_ = 0;
  core::conveyor_detail::Lambda lam0_{};
  std::unique_ptr<ITransport> transport_;
  std::vector<ZoneMsg> final_;
  ConveyorResult res_{};
  std::atomic<bool> halt_on_{false};
  std::mutex halt_mu_;
  Halt halt_kind_ = Halt::None;
  std::string halt_msg_;
};

// Convenience: run the SW ring on `atoms` (mutated in place), return result.
// Default policy (CPU) — byte-identical to the serial sw_zone_pass oracle.
template <typename Real>
ConveyorResult run_sw_ring(AtomSoA<Real>& atoms, const Box& box,
                           const SwPotential<Real>& pot, const ConveyorOptions& o) {
  SwRing<Real> r(atoms, box, pot, o);
  return r.run();
}

// Policy-injected overload: run the ring with a custom window-force policy (e.g. the
// T5 GpuSwWinForce, or the SwWinForceLocalKey poison). Orchestration identical ⇒
// bitwise ≡ the CPU ring by construction.
template <typename Real, typename WinForce>
ConveyorResult run_sw_ring(AtomSoA<Real>& atoms, const Box& box,
                           const SwPotential<Real>& pot, const ConveyorOptions& o,
                           WinForce winforce) {
  SwRing<Real, WinForce> r(atoms, box, pot, o, std::move(winforce));
  return r.run();
}

}  // namespace tdmd::potentials
