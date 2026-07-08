#pragma once
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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
#include "tdmd/potentials/eam_donation.hpp"  // PR-2: donation executors + DonatingWindowForcePolicy
#include "tdmd/potentials/eam_zone.hpp"  // eam_window_force, zone_eam_window

// M6 PR-E3b — EamRing: threaded EAM TD ring (SEPARATE driver; the pair
// TimeConveyor in core/conveyor.hpp stays byte-untouched). Design:
// docs/_meta/M6_E3b_EAMRING_DESIGN_2026-06-14.md.
//
// Per owned zone S_j the node finalizes the CENTER zone j over the SYMMETRIC
// three-zone window {S_{j-1}, S_j, S_{j+1}} (the EAM force range is 2·rcut), made
// resident by a ONE-ZONE SEND DELAY (the second forward-hop): slot j-1 is sent
// only AFTER compute(j), so {S_{j-1}, S_j, S_{j+1}} are co-resident at finalize(j).
// Force kernel = eam_window_force (PR-E3b-1); Λ-chain dt handoff / INV-4 /
// velocity-Verlet split / §7.4 parity reused from the pair conveyor's logic
// (verbatim arithmetic ⇒ bitwise ≡ the serial zone_eam_pass oracle).
//
// THIS PR: free-z (non-periodic) boundaries. PBC-z (cyclic window closure of
// owned 0 in the tail) is a follow-up.
//
// M6 E5b-3b — POLICY-TEMPLATED window force. The ring's orchestration (threads,
// SPSC channels, send-delay/second-forward-hop, FSM, Λ-chain dt handoff, PBC
// defer_head) is the PROVEN bitwise oracle; the ONLY moving part is the per-
// window EAM force computation. EamRing is parametrized on a WINDOW-FORCE POLICY
// `WinForce`; the default `CpuEamWindowForce<Math>` wraps eam_window_force ⇒ the
// CPU ring stays BYTE-IDENTICAL (verified by test_eam_ring / test_eam_zone). The
// GPU policy (cuda/eam_window_force_gpu.cuh) runs the E5 kernels on the SAME
// gathered window ⇒ EamGpuRing inherits ALL orchestration ⇒ bitwise ≡ CPU ring
// by construction (the int64 window-force result is order-free, B1/INV-9).
//
// PR-2 (live donation ring) — FORK DIVERGENCE (audit §5/§7.7 foresaw it, fork #5 to
// unify at ReaxFF): the EAM scan loop (run_pass_impl<DA> + donate_position + the
// compose-from-rho finalize + the ledger gate) now carries the W-contract donation
// path and DIVERGES IRREVERSIBLY from the 3 sibling rings (sw_ring/tersoff_ring/
// meam_ring). Those siblings are PRE-PR-2 forks (git-recoverable), kept BYTE-UNTOUCHED
// (house rule: siblings byte-untouched proves PR-2 leaked nothing into their paths).
// CONSEQUENTLY the sibling banners' "diff eam_ring.hpp sw_ring.hpp shows ONLY the five
// swaps" provenance claim (sw_ring.hpp:30-35, tersoff_ring.hpp:33-34, meam_ring.hpp:33-35)
// is a SUPERSEDED HISTORICAL ARTIFACT as of PR-2 — the density surgery makes that diff no
// longer swap-only. Future orchestration fixes port to eam_ring by hand. The refinement
// concept (DonatingWindowForcePolicy) is ADDITIVE ⇒ SwWinForce/TersoffWinForce/MeamWinForce
// still model the BASE WindowForcePolicy and stay byte-intact. fsm.hpp is byte-untouched.
namespace tdmd::potentials {

using core::AtomSoA;
using core::Box;
using core::ConveyorOptions;
using core::ConveyorResult;
using core::Halt;
using core::ITransport;
using core::RingTransport;
using core::ZoneMsg;

// Default (CPU) window-force policy: a thin, stateless wrapper over
// eam_window_force with the dual-format (Q19.44 / Q23.40) dispatch on
// density_fracbits(). compute() takes the gathered window and writes the int64
// force accumulators + accumulates pe / min_r2 EXACTLY as the pre-refactor
// finalize_owned body did ⇒ the CPU ring is byte-identical. `fb` is the density
// fracbits (44 or 40); the policy is constructed once per run.
template <typename Math>
struct CpuEamWindowForce {
  const Math* math = nullptr;
  int fb = 44;
  explicit CpuEamWindowForce(const Math& m) : math(&m), fb(m.density_fracbits()) {}

  void compute(const double* wx, const double* wy, const double* wz, const long* key,
               int m, const int* owned, int n_owned, const core::PairGeom& geom,
               double rho_cap, std::vector<core::fixed::ForceAccum>& wFx,
               std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz,
               core::fixed::EnergyAccum& pe, double& min_r2) const {
    if (fb == 44)
      eam_window_force<Math, core::fixed::FixedAccum<44>>(
          wx, wy, wz, key, m, owned, n_owned, *math, geom, rho_cap, wFx, wFy, wFz, pe, min_r2);
    else
      eam_window_force<Math, core::fixed::FixedAccum<40>>(
          wx, wy, wz, key, m, owned, n_owned, *math, geom, rho_cap, wFx, wFy, wFz, pe, min_r2);
  }

  // PR-0a (FIREWALL GAP at the CPU-EAM seam — closed): the CPU policy runs the SAME
  // symmetric eam_window_force ⇒ its accept-set ≡ the GPU policy. Delegates to the
  // single EAM gate (eam.hpp). On legal EAM this is a no-op ⇒ F-NOOP (Test_EAM_Ring
  // byte-identical). Was ABSENT ⇒ the `if constexpr requires` firewall was dead on the
  // CPU ring; the concept now MANDATES this (many_body.hpp WindowForcePolicy).
  static void assert_supported(std::span<const potentials::PassDecl> passes) {
    potentials::assert_eam_symmetric_passes(passes, "CpuEamWindowForce");
  }

  // PR-2 (live donation ring) — the three donation hooks (thin wrappers over the
  // PR-1 executors, already bitwise-proven vs the window recompute by lemma W-1).
  // The ring calls these; the ring (not the hook) owns the ledger. After PR-2 the CPU
  // ring composes force from the donated rho ⇒ compute() above is no longer called by
  // the production ring for density (kept for the base concept + as the frozen recompute).
  //
  // PR-3a (concept v2 — device-donation substrate): the CPU NodeState is INERT (the CPU
  // donated rho lives in the ring's pass-scoped dstate, not here) ⇒ the CPU ring stays
  // byte-identical (F-NOOP gate = the CPU suite bitwise). ns/wb are accepted and ignored.
  struct NodeState {
    void begin_pass(long) {}
  };
  NodeState make_node_state(int /*n_zones*/) const { return {}; }

  template <class DA>
  void on_zone_arrival(NodeState& /*ns*/, potentials::EamDonationState<DA>& st, int label,
                       const potentials::ZoneBlockView& blk,
                       const core::PairGeom& geom) const {
    potentials::eam_donate_self<Math, DA>(blk, *math, geom, st.rho[std::size_t(label)]);
  }
  template <class DA>
  void on_edge(NodeState& /*ns*/, potentials::EamDonationState<DA>& st, int la, int lb,
               const potentials::ZoneBlockView& a, const potentials::ZoneBlockView& b,
               const core::PairGeom& geom) const {
    potentials::eam_donate_cross<Math, DA>(a, b, *math, geom, st.rho[std::size_t(la)],
                                           st.rho[std::size_t(lb)]);
  }
  template <class DA>
  void compose(NodeState& /*ns*/, const double* wx, const double* wy, const double* wz,
               const long* key, int m, const potentials::WindowBlocks& /*wb*/,
               const int* owned, int n_owned, const core::PairGeom& geom, double rho_cap,
               const DA* rho_w, std::vector<core::fixed::ForceAccum>& wFx,
               std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz, core::fixed::EnergyAccum& pe,
               double& min_r2, int zone_j) const {
    potentials::eam_window_force_from_rho<Math, DA>(wx, wy, wz, key, m, owned, n_owned, *math,
                                                    geom, rho_cap, rho_w, wFx, wFy, wFz, pe,
                                                    min_r2, potentials::NullDonationTrace{}, zone_j);
  }
};

template <typename Real, typename Math, typename WinForce = CpuEamWindowForce<Math>>
class EamRing {
  // PR-0a: the WinForce contract, PROMOTED from the opt-in `if constexpr requires`
  // (silently bypassable — the overdue MB1/MB2 promise). Fires on ANY instantiation
  // of the ring, even a TU that only constructs it (many_body.hpp).
  // PR-2 upgrades the base WindowForcePolicy to the DonatingWindowForcePolicy REFINEMENT:
  // the WinForce must ALSO carry the three donation hooks (on_zone_arrival/on_edge/compose)
  // the live donation ring calls. Refinement (not a base change) ⇒ the sibling sw/tersoff/
  // meam policies (which static_assert the BASE) stay byte-untouched. Mandatory = compile
  // error, not opt-in (WContract §12.1).
  static_assert(potentials::DonatingWindowForcePolicy<WinForce>,
      "WinForce must model DonatingWindowForcePolicy (base WindowForcePolicy + the PR-2 "
      "donation hooks on_zone_arrival/on_edge/compose) — see eam_donation.hpp");

 public:
  // The default-policy ctor (CPU): builds the policy from pot.math. Byte-compatible
  // with the pre-refactor signature — every existing call site is unchanged.
  EamRing(AtomSoA<Real>& atoms, const Box& box, const EamPotential<Real, Math>& pot,
          const ConveyorOptions& o)
      : EamRing(atoms, box, pot, o, WinForce(pot.math)) {}

  // Policy-injected ctor (GPU / custom): the caller supplies the window-force
  // policy (e.g. a GpuEamWindowForce holding the device spline + per-node stream).
  EamRing(AtomSoA<Real>& atoms, const Box& box, const EamPotential<Real, Math>& pot,
          const ConveyorOptions& o, WinForce winforce)
      : atoms_(atoms), box_(box), pot_(pot), o_(o), rcut_(pot.math.rcut),
        winforce_(std::move(winforce)) {
    if (o_.steps < 1) throw std::invalid_argument("eam_ring: steps must be >= 1");
    if (o_.n_nodes < 1) throw std::invalid_argument("eam_ring: n_nodes must be >= 1");
    if (!(o_.dt_initial > 0.0)) throw std::invalid_argument("eam_ring: dt_initial > 0");
    // periodic-z is supported (PR-E3b-PBC): cyclic window + defer_head + tail-
    // batched sends. ZoneDecomposition::build rejects periodic n_zones in 2..4
    // for reach_mult=2, so PBC only reaches n=1 (free path) or n>=5 (distinct
    // cyclic zones — no window dedup ever needed).
    fb_ = pot_.math.density_fracbits();  // 44 (Q19.44) or 40 (Q23.40)
  }

  // PR-2 TEST-ONLY knob (G10): when set, the ring drops the FIRST self donation's ledger
  // bit (rho still donated), making the ledger INCOMPLETE so the end_eam completeness check
  // must HALT StaleZone. Default false ⇒ production is untouched. Use with z=1 (single node,
  // no race on the one-shot state). NOT forwarded through run_eam_ring — set on a directly
  // constructed EamRing.
  bool test_drop_first_self_ledger_ = false;

  ConveyorResult run() {
    zd_ = core::ZoneDecomposition::build(atoms_, box_, o_.n_zones, rcut_, /*reach_mult=*/2);
    n_ = zd_.n_zones;
    z_ = o_.n_nodes;
    // descriptor firewall (PR-0a: UNCONDITIONAL — the concept guarantees the hooks
    // exist, so the old `if constexpr requires` opt-in is gone). validate_pass_decls
    // checks descriptor self-consistency (W-teeth); assert_supported checks the policy's
    // capability (refuses needs_transpose / unsupported kinds ⇒ the symmetric accumulator
    // cannot silently run wrong). Both no-op on legal EAM ⇒ F-NOOP.
    potentials::validate_pass_decls(pot_.passes());
    WinForce::assert_supported(pot_.passes());
    // PR-2: the donation path requires the D5-flipped EAM Density descriptor (kAccumB1 +
    // self|lo|hi) — without it want_closure_mask is 0 and the ring ledger-check is vacuous.
    potentials::assert_eam_donation_descriptor(pot_.passes());
    td_self_dropped_ = false;  // PR-2 test knob (G10): reset the one-shot ledger-drop
    // PR-3a: per-NODE donation state (one per node jthread — the audit ownership shape;
    // per-node pass-local, hard constraint 1). Created AFTER the firewall asserts and the
    // decomposition build (n_ known), BEFORE the jthreads spawn. CPU NodeState is inert.
    node_stores_.clear();
    node_stores_.reserve(std::size_t(o_.n_nodes));
    for (int k = 0; k < o_.n_nodes; ++k) node_stores_.push_back(winforce_.make_node_state(n_));

    // t0 forces via the serial oracle (same kernel ⇒ same bits) for the 1st drift.
    core::zero_forces(atoms_);
    const double pe0 = zone_eam_pass(atoms_, box_, zd_, pot_).pe;
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
    // PR-2: the atom ids as long, materialized on arrival — ZoneBlockView.key needs a
    // valid const long* (ZoneMsg.id is int; the donation executors evaluate key[t] even
    // under NullPairHook, so nullptr would be a null-deref the sanitizer flags).
    std::vector<long> gkey;
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

  // PR-2: fb-dispatch (44=Q19.44 / 40=Q23.40) mirrors zone_eam_pass_impl / zone_eam_pass_
  // donated_impl — keyed ONLY off density_fracbits() (fb_), NEVER the static descriptor.
  bool run_pass(int k, long h, std::vector<ZoneMsg>* preload) {
    return (fb_ == 44)
               ? run_pass_impl<core::fixed::FixedAccum<44>>(k, h, preload)
               : run_pass_impl<core::fixed::FixedAccum<40>>(k, h, preload);
  }

  template <class DensAccum>
  bool run_pass_impl(int k, long h, std::vector<ZoneMsg>* preload) {
    const int in_edge = (k - 1 + z_) % z_;
    const int out_edge = k;
    const auto io = core::node_io_order(k + 1);  // §7.4 parity (1-based)
    const int r = box_.periodic[2] ? int((h - 1) % n_) : 0;  // pass-order rotation
    const bool defer_head = box_.periodic[2] && n_ > 1;      // §7.2 closure ([ENG])
    const core::PairGeom geom(box_, rcut_);
    const double rho_cap = pot_.math.density_grid_max();

    std::vector<Slot> slot(static_cast<std::size_t>(n_));
    int arrived = 0, sent = 0;
    double dt = 0.0, R_buf = 0.0, dt_next = 0.0;
    core::conveyor_detail::Lambda agg{0.0, 0.0, std::numeric_limits<double>::infinity()};
    core::fixed::EnergyAccum pe;
    double ke = 0.0, min_r2 = std::numeric_limits<double>::infinity();

    // PR-2: pass-scoped per-zone donated density + completeness ledger, keyed by zone_id
    // (label), reset on RECV. A mid-pass HALT (any return false / throw) destroys this
    // stack frame ⇒ the whole pass is discarded, no partial replay (SPEC(6)).
    potentials::EamDonationState<DensAccum> dstate;
    dstate.rho.assign(std::size_t(n_), {});
    dstate.ledger.assign(std::size_t(n_), 0u);
    // PR-3a: the node's device-donation state + pass token (the stamp staleness fence —
    // PR-4's rebuild_epoch seat). Inert no-op for the CPU policy.
    auto& ns = node_stores_[std::size_t(k)];
    ns.begin_pass(h);

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
        // PR-2: materialize the long key + reset this zone's donated rho/ledger on RECV
        // (label = want_id). reset_zone was designed for exactly this (eam_donation.hpp).
        s.gkey.assign(std::size_t(s.msg.n()), 0);
        for (int i = 0; i < s.msg.n(); ++i) s.gkey[std::size_t(i)] = long(s.msg.id[std::size_t(i)]);
        dstate.reset_zone(want_id, std::size_t(s.msg.n()));
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

    // PR-2: a zone's coordinate block for donations (POST-drift; key = the materialized
    // long id array). key IS used (the executors evaluate on_pair(key[t],key[u]) even
    // under the default NullPairHook) ⇒ must be a valid pointer, hence Slot::gkey.
    auto block_view = [&](const Slot& s) -> potentials::ZoneBlockView {
      return {s.msg.x.data(), s.msg.y.data(), s.msg.z.data(), s.gkey.data(), s.msg.n()};
    };

    // PR-2: execute the donation batches READY at scan position jj (self first, then cross)
    // per donation_layout — the arrival-variant schedule (WContract §9). The RING owns the
    // ledger (the free executors touch rho only); exactly-once is a runtime INV-8 (throw on
    // a repeated batch). Edge-order (many_body.hpp): close_cross_batch(lower,upper) marks
    // CrossHi in the lower zone, CrossLo in the upper — the seam (n-1,0) marks n-1 as lower.
    auto donate_position = [&](int jj) -> bool {
      const potentials::DonationBatches b =
          potentials::donation_layout(jj, n_, box_.periodic[2]);
      for (int s = 0; s < b.n_self; ++s) {
        const int sl = b.self[s];
        const int label = slot[std::size_t(sl)].fsm.id;
        const uint32_t bit = potentials::closure_bit(0, potentials::DonorRole::kSelf);
        if (dstate.ledger[std::size_t(label)] & bit)
          return fail(Halt::Internal, "donation: self batch twice zone " + std::to_string(label));
        winforce_.on_zone_arrival(ns, dstate, label, block_view(slot[std::size_t(sl)]), geom);
        // G10 test knob: drop the FIRST self ledger bit (rho still donated ⇒ physics fine),
        // making the ledger INCOMPLETE ⇒ the end_eam ledger-before-END check must HALT. This
        // is the ONLY way to exercise that check (the ledger is ring-owned, not policy-owned).
        if (test_drop_first_self_ledger_ && !td_self_dropped_)
          td_self_dropped_ = true;
        else
          dstate.ledger[std::size_t(label)] |= bit;
      }
      for (int c = 0; c < b.n_cross; ++c) {
        const int sa = b.cross[c][0], sb = b.cross[c][1];
        const int la = slot[std::size_t(sa)].fsm.id, lb = slot[std::size_t(sb)].fsm.id;
        const uint32_t hi = potentials::closure_bit(0, potentials::DonorRole::kCrossHi);
        if (dstate.ledger[std::size_t(la)] & hi)
          return fail(Halt::Internal, "donation: cross batch twice edge " + std::to_string(la));
        winforce_.on_edge(ns, dstate, la, lb, block_view(slot[std::size_t(sa)]),
                          block_view(slot[std::size_t(sb)]), geom);
        potentials::close_cross_batch(dstate.ledger[std::size_t(la)],
                                      dstate.ledger[std::size_t(lb)], 0);
      }
      return true;
    };

    // T3 START + force store + second-half kick + zone-local reductions + checks.
    auto end_eam = [&](int j, const std::vector<int>& ownedloc,
                       std::vector<core::fixed::ForceAccum>& wFx,
                       std::vector<core::fixed::ForceAccum>& wFy,
                       std::vector<core::fixed::ForceAccum>& wFz) -> bool {
      Slot& s = slot[std::size_t(j)];
      // PR-2: the density ledger of THIS zone must be closed (all its self + cross
      // donation batches executed) STRICTLY before SPHERE/END — this is the completeness
      // check (WContract §5/§12.4) AND the reason SPHERE is now MATERIAL: the w-phase
      // carries the donated rho that filled this zone in its d/w, no longer artificial.
      // want uses SLOT j (free-z edge-drop); ledger uses LABEL (s.fsm.id) — the slot->label
      // mapping the rotational tooth (G-ROT) guards.
      const uint32_t want = potentials::want_closure_mask(pot_.passes(), j, n_, box_.periodic[2]);
      if (dstate.ledger[std::size_t(s.fsm.id)] != want)
        return fail(Halt::StaleZone,
                    "eam_ring: donation ledger " + std::to_string(dstate.ledger[std::size_t(s.fsm.id)]) +
                        " != want " + std::to_string(want) + " (missing/late batch) step " +
                        std::to_string(h) + " zone " + std::to_string(s.fsm.id));
      core::ZoneFSM::apply(s.fsm, core::ZoneEvent::SPHERE);  // d -> w (MATERIAL: ledger closed)
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
      // gather contiguous window (key = atom id for the φ-once order) + the donated rho
      // in the SAME block order [pred][center][succ]. rho comes from the PERSISTENT
      // per-zone dstate (keyed by LABEL = slot.fsm.id), NOT recomputed here (PR-2: the
      // ×3 density recompute is gone). raw copy of the int64 accumulator is bit-trivial.
      std::vector<double> wx, wy, wz;
      std::vector<long> key;
      std::vector<int> ownedloc;
      std::vector<DensAccum> rho_w;
      for (int t = 0; t < nw; ++t) {
        const Slot& w = slot[std::size_t(wslots[t])];
        const int base = int(wx.size());
        const auto& rz = dstate.rho[std::size_t(w.fsm.id)];  // LABEL keying (G-ROT tooth)
        for (int i = 0; i < w.msg.n(); ++i) {
          wx.push_back(w.msg.x[std::size_t(i)]);
          wy.push_back(w.msg.y[std::size_t(i)]);
          wz.push_back(w.msg.z[std::size_t(i)]);
          key.push_back(w.msg.id[std::size_t(i)]);
          rho_w.push_back(rz[std::size_t(i)]);
        }
        if (wslots[t] == j)
          for (int i = 0; i < w.msg.n(); ++i) ownedloc.push_back(base + i);
      }
      const int m = int(wx.size());
      // PR-3a: the window's block layout (labels + sizes + center) — built HERE, from data
      // finalize already iterates (the slot→label mapping stays ring-side; hooks/compose see
      // only labels). Block order == the gather order above.
      potentials::WindowBlocks wb;
      wb.nb = nw;
      for (int t = 0; t < nw; ++t) {
        const Slot& w = slot[std::size_t(wslots[t])];
        wb.label[t] = w.fsm.id;
        wb.n[t] = w.msg.n();
        if (wslots[t] == j) wb.center = t;
      }
      std::vector<core::fixed::ForceAccum> wFx(m), wFy(m), wFz(m);
      // PR-2: COMPOSE the C-phase force from the donated rho (pass-2/3 of eam_window_force,
      // verbatim — cap on owned-full rho only). CPU default = eam_window_force_from_rho;
      // GPU recomputes density per-window inside compose (bitwise == today) until the
      // device-donated compose lands in PR-3b. pe/min_r2/φ-once identical to the oracle.
      winforce_.compose(ns, wx.data(), wy.data(), wz.data(), key.data(), m, wb,
                        ownedloc.data(), int(ownedloc.size()), geom, rho_cap, rho_w.data(),
                        wFx, wFy, wFz, pe, min_r2, j);
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
      // PR-2: execute the donation batches READY at scan j (POST-drift coords) BEFORE any
      // finalize reads them — the arrival-variant deadline (WContract §9). The seam
      // cross(n-1,0) fires here at j=n-1, strictly before the tail finalize(0)/finalize(n-1).
      if (!donate_position(j)) return false;
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
  const EamPotential<Real, Math>& pot_;
  ConveyorOptions o_;
  double rcut_;
  int fb_ = 44;
  WinForce winforce_;
  // PR-3a: per-node donation state, ring-owned (created in run(), one per node jthread).
  std::vector<typename WinForce::NodeState> node_stores_;
  core::ZoneDecomposition zd_;
  int n_ = 0, z_ = 0;
  bool td_self_dropped_ = false;  // G10 one-shot state (reset in run(); use with z=1)
  core::conveyor_detail::Lambda lam0_{};
  std::unique_ptr<ITransport> transport_;
  std::vector<ZoneMsg> final_;
  ConveyorResult res_{};
  std::atomic<bool> halt_on_{false};
  std::mutex halt_mu_;
  Halt halt_kind_ = Halt::None;
  std::string halt_msg_;
};

// Convenience: run the EAM ring on `atoms` (mutated in place), return result.
// Default policy (CPU) — byte-identical to the pre-refactor driver.
template <typename Real, typename Math>
ConveyorResult run_eam_ring(AtomSoA<Real>& atoms, const Box& box,
                            const EamPotential<Real, Math>& pot, const ConveyorOptions& o) {
  EamRing<Real, Math> r(atoms, box, pot, o);
  return r.run();
}

// Policy-injected overload: run the ring with a custom window-force policy
// (e.g. the GPU GpuEamWindowForce). The orchestration is identical ⇒ bitwise
// ≡ the CPU ring by construction.
template <typename Real, typename Math, typename WinForce>
ConveyorResult run_eam_ring(AtomSoA<Real>& atoms, const Box& box,
                            const EamPotential<Real, Math>& pot, const ConveyorOptions& o,
                            WinForce winforce) {
  EamRing<Real, Math, WinForce> r(atoms, box, pot, o, std::move(winforce));
  return r.run();
}

}  // namespace tdmd::potentials
