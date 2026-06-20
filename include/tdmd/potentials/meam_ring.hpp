#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
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
#include "tdmd/potentials/eam_zone.hpp"  // zone_eam_window / eam_window_layout (reused verbatim)
#include "tdmd/potentials/meam.hpp"       // MeamParams, MeamPotential, the descriptor
#include "tdmd/potentials/meam_zone.hpp"  // meam_window_force, meam_zone_pass (Me3)

// M6 / MEAM-ladder Me3b — MeamRing: threaded screened-angular many-body TD ring. The FOURTH fork
// of the proven ring (EamRing → SwRing → TersoffRing → MeamRing). A SIBLING FORK of
// tersoff_ring.hpp (which forked sw_ring.hpp ← eam_ring.hpp ← the pair TimeConveyor — the house
// style): the orchestration body (threads, SPSC channels, send-delay/second-forward-hop, FSM,
// Λ-chain dt handoff, PBC defer_head, the unsorted slot-order window gather, finalize/scatter) is
// COPIED CHARACTER-FOR-CHARACTER from TersoffRing. `diff tersoff_ring.hpp meam_ring.hpp` shows
// ONLY the swaps + the MEAM policy block — an INSPECTION-CHECKABLE proof that the EAM/SW/Tersoff
// ring paths are byte-untouched (F-NOOP).
//
// The swaps (the entire delta): (S1) includes; (S2) pot type TersoffPotential→MeamPotential,
// pot.ters→pot.meam, rcut_=pot.meam.rc (a FIELD, not a method); (S3) t0 forces via meam_zone_pass;
// (S4) the policy block — MeamWinForce — the GENUINE deltas vs TersoffWinForce: (i) meam_window_force
// has TWO energy accumulators (pe_embed, pe_pair) which compute() must fold into the ring's single
// `pe` via an INT64 sum (pe.raw += embed.raw + pair.raw — order-free; FP .value()+.value() would
// break ring≡serial bitwise) + sinks the SERIAL-only screening counts (n_screened_partial/_zero);
// (ii) the firewall ACCEPTS the [Density, Embedding, Force(needs_transpose)] descriptor (it INVERTS
// GpuEamWindowForce — which REJECTS needs_transpose — and is NOT TersoffWinForce's [BondOrder,Force]);
// (S5) comment/string text.
//
// Per owned zone S_j the node finalizes the CENTER zone over the three-zone window
// {S_{j-1}, S_j, S_{j+1}} (MEAM screening reach 2·rc, effective_range={2,true}). The per-window
// force is meam_window_force (the screening transpose-replay: pair/embedding + the 3rd-atom-k
// screening write). The ring gathers the window UNSORTED (block order). TWO order hazards, both
// handled (the SW T3b / Te3b discipline):
//   (A) the φ-once / pair-energy lower-global-key — DEFENSIVE not load-bearing (count-once-invariant
//       on the slot gather; Me3 MEASURED the screening-order delta sub-quantum under relabel).
//   (B) the FP order — MEAM's screening product Π_k S(C_ijk) and the four-density partial sums are
//       FP-non-associative. The unsorted ring window would sum these in a different order than the
//       SORTED serial meam_zone_pass (zone_eam_window sorts by global id) ⇒ a Q24.40 quantum could
//       flip ⇒ correct-within-round-off but NOT bitwise to the serial. MeamWinForce::compute()
//       therefore CANONICALIZES — it sorts the window into global-key order before meam_window_force
//       ⇒ the screening/density sums are summed identically to the serial ⇒ the ring is BITWISE to
//       meam_zone_pass (G1), transitively to meam_run_fixed_force / FD / LAMMPS. (The int64 force
//       accumulation is order-free by B1; only the FP density/screening sums need this. A no-op when
//       the window is already sorted — the serial-shaped z=1 window.)
namespace tdmd::potentials {

using core::AtomSoA;
using core::Box;
using core::ConveyorOptions;
using core::ConveyorResult;
using core::Halt;
using core::ITransport;
using core::RingTransport;
using core::ZoneMsg;

// Default (CPU) MEAM window-force policy: a thin, STATELESS wrapper over meam_window_force (the
// screened many-body transpose-replay). compute() matches the EamRing 14-arg const signature
// EXACTLY so the orchestration's gather/finalize/scatter body is the verbatim copy. THE GENUINE
// DELTAS vs TersoffWinForce:
//   • meam_window_force takes TWO EnergyAccum out-params (pe_embed, pe_pair) where the ring carries
//     a SINGLE `pe`. compute() declares local pe_embed/pe_pair, then folds them into `pe` via an
//     INT64 sum (pe.raw += embed.raw + pair.raw) — order-free by B1. State bitwise-equivalence
//     (G1/G2 compare x/v only) rests entirely on the int64 FORCES being order-free; the int64 PE
//     fold keeps `pe` itself 1-vs-z bitwise (a free bonus) but its single FP decode at report time
//     differs in the last ULP from the serial's two-decode `pe_embed.value()+pe_pair.value()` — NO
//     gate compares the PE scalar, so this is immaterial. An FP .value()+.value() fold would lose
//     the int64 order-freedom; the raw-int64 fold preserves it.
//   • meam_window_force also returns the SERIAL-only screening witnesses n_screened_partial /
//     n_screened_zero + a drop_class param. The counts live on the SERIAL path (the MB2/G-VACUITY
//     evidence) — NEVER on the shared ring policy, which would race across z jthreads. compute()
//     sinks them into locals (like Tersoff sinks n_bonds/n_triplets, SW sinks n_triplets).
// STATELESS-except-const-ptr is what makes 1-vs-z bitwise.
template <typename Real>
struct MeamWinForce {
  const MeamParams* mp = nullptr;  // owned by MeamPotential; pot_ outlives the ring
  explicit MeamWinForce(const MeamParams& m) : mp(&m) {}

  void compute(const double* wx, const double* wy, const double* wz, const long* key,
               int m, const int* owned, int n_owned, const core::PairGeom& geom,
               double /*rho_cap — MEAM density is internal to meam_window_force*/,
               std::vector<core::fixed::ForceAccum>& wFx,
               std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz,
               core::fixed::EnergyAccum& pe, double& min_r2) const {
    // CANONICAL FP-ORDER (the screening/density delta vs the int64-order-free EAM — and the same
    // posture as Tersoff's ζ-sum). The ring gathers the window UNSORTED (block order
    // [pred][center][succ]); the serial meam_zone_pass sorts via zone_eam_window (by global INDEX
    // g). We sort by key = atom id; the two coincide because id == index+1 (core/soa.hpp resize — a
    // strict monotone bijection on the MEAM path). So the screening product Π_k S(C_ijk) and the
    // four-density partial sums are summed identically to the serial ⇒ the ring is BITWISE to
    // meam_zone_pass (⇒ transitively to meam_run_fixed_force / FD / LAMMPS — not merely ≈). int64
    // forces are order-free (B1), so only the FP density/screening sums need this; a no-op when
    // already sorted (the serial-shaped z=1 window). (Me3 measured the screening order DEFENSIVE-
    // not-load-bearing under relabel, but the ring window IS unsorted vs the serial sorted — carry
    // the sort so ring≡serial is bitwise.)
    std::vector<int> perm(m);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int a, int b) { return key[a] < key[b]; });
    bool identity = true;
    for (int s = 0; s < m; ++s) if (perm[s] != s) { identity = false; break; }
    if (identity) {  // already global-key sorted ⇒ direct (the serial-shaped window)
      core::fixed::EnergyAccum pe_embed, pe_pair;
      long n_partial = 0, n_zero = 0;
      meam_window_force<Real>(wx, wy, wz, key, m, owned, n_owned, *mp, geom, wFx, wFy, wFz,
                              pe_embed, pe_pair, min_r2, n_partial, n_zero, /*drop_class=*/0);
      pe.raw += pe_embed.raw + pe_pair.raw;  // INT64 fold — order-free, == serial's pe multiset
      return;
    }
    std::vector<double> sx(m), sy(m), sz(m);
    std::vector<long> sk(m);
    std::vector<int> inv(m);
    for (int s = 0; s < m; ++s) { const int o = perm[s]; sx[s] = wx[o]; sy[s] = wy[o]; sz[s] = wz[o]; sk[s] = key[o]; inv[o] = s; }
    std::vector<int> sowned(n_owned);
    for (int t = 0; t < n_owned; ++t) sowned[t] = inv[owned[t]];
    std::vector<core::fixed::ForceAccum> sFx(m), sFy(m), sFz(m);
    core::fixed::EnergyAccum pe_embed, pe_pair;
    long n_partial = 0, n_zero = 0;
    meam_window_force<Real>(sx.data(), sy.data(), sz.data(), sk.data(), m, sowned.data(),
                            n_owned, *mp, geom, sFx, sFy, sFz, pe_embed, pe_pair, min_r2,
                            n_partial, n_zero, /*drop_class=*/0);
    pe.raw += pe_embed.raw + pe_pair.raw;  // INT64 fold — order-free, == serial's pe multiset
    for (int s = 0; s < m; ++s) { wFx[perm[s]] = sFx[s]; wFy[perm[s]] = sFy[s]; wFz[perm[s]] = sFz[s]; }
  }

  // FIREWALL — MEAM's descriptor is [Density, Embedding, Force(needs_transpose=TRUE)] (the SAME
  // SHAPE as EAM's [Density, Embedding, Force] but Force.needs_transpose=TRUE: the screening
  // 3rd-atom-k write is the legitimate non-symmetric write). So this firewall INVERTS
  // GpuEamWindowForce::assert_supported (which THROWS on needs_transpose: the symmetric q(j)=−q(i)
  // int64 accumulator has no third-atom-k slot) — ACCEPT the needs_transpose Force (the screening
  // transpose-replay IS the correct mechanism). It is ALSO NOT a copy of TersoffWinForce (which
  // expects [BondOrder, Force] and would WRONGLY reject MEAM's pass 0 == Density) NOR EamWinForce
  // (which rejects needs_transpose). Reject ONLY: wrong count, non-{Density,Embedding,Force} kinds,
  // a Force pass mislabelled SYMMETRIC (needs_transpose==false — would silently run a symmetric
  // accumulator wrong, the firewall's whole point), and iterative (QEq/CG, ReaxFF).
  static void assert_supported(std::span<const potentials::PassDecl> passes) {
    if (passes.size() != 3)
      throw std::runtime_error("MeamWinForce: MEAM is the 3-pass [Density, Embedding, Force] "
          "sequence — got " + std::to_string(passes.size()) + " passes");
    const potentials::PassKind want[3] = {potentials::PassKind::Density,
        potentials::PassKind::Embedding, potentials::PassKind::Force};
    for (std::size_t p = 0; p < 3; ++p)
      if (passes[p].kind != want[p])
        throw std::runtime_error("MeamWinForce: unexpected pass kind at " + std::to_string(p) +
            " — MEAM is Density→Embedding→Force (a BondOrder here is the Tersoff descriptor, not "
            "MEAM)");
    for (const auto& p : passes)
      if (p.iterative)
        throw std::runtime_error("MeamWinForce: iterative pass (QEq/CG, ReaxFF) unimplemented");
    // The INVERSION: passes[0] (Density) + passes[1] (Embedding) are symmetric; passes[2] (Force)
    // IS needs_transpose — the screening 3rd-atom-k transpose-replay IS the mechanism. ACCEPT it
    // (unlike GpuEamWindowForce, which throws). A SYMMETRIC Force pass 2 would be the EAM trap (no
    // third-atom-k slot — silently wrong); a transpose-mislabelled Density/Embedding is malformed.
    if (passes[0].needs_transpose || passes[1].needs_transpose)
      throw std::runtime_error("MeamWinForce: Density/Embedding passes must be symmetric");
    if (!passes[2].needs_transpose)
      throw std::runtime_error("MeamWinForce: expected pass 2 (Force) needs_transpose=true (the "
          "screening 3rd-atom-k non-symmetric write) — a symmetric Force here is the EAM "
          "descriptor, which has no third-atom-k slot and would silently run wrong");
  }
};
static_assert(std::is_trivially_copyable_v<MeamWinForce<double>>,
              "MeamWinForce must stay stateless (shared across z jthreads) — no mutable "
              "diagnostic member, or 1-vs-z bitwise breaks via a data race");

template <typename Real, typename WinForce = MeamWinForce<Real>>
class MeamRing {
 public:
  // The default-policy ctor (CPU): builds the policy from pot.meam (a MeamParams).
  MeamRing(AtomSoA<Real>& atoms, const Box& box, const MeamPotential<Real>& pot,
         const ConveyorOptions& o)
      : MeamRing(atoms, box, pot, o, WinForce(pot.meam)) {}

  // Policy-injected ctor (GPU / custom): the caller supplies the window-force policy
  // (e.g. a GpuMeamWinForce holding the device screening transpose accumulator — Me5).
  MeamRing(AtomSoA<Real>& atoms, const Box& box, const MeamPotential<Real>& pot,
         const ConveyorOptions& o, WinForce winforce)
      : atoms_(atoms), box_(box), pot_(pot), o_(o), rcut_(pot.meam.rc),
        winforce_(std::move(winforce)) {
    if (o_.steps < 1) throw std::invalid_argument("meam_ring: steps must be >= 1");
    if (o_.n_nodes < 1) throw std::invalid_argument("meam_ring: n_nodes must be >= 1");
    if (!(o_.dt_initial > 0.0)) throw std::invalid_argument("meam_ring: dt_initial > 0");
    // Three-leg min-image guard (the ring calls meam_window_force directly, BYPASSING
    // meam_zone_pass's guard): MEAM's screening triple has THREE independent inter-atom vectors
    // (i–j, i–k, j–k = a DIFFERENCE of two min-imaged vectors) ⇒ a thin periodic box could pick a
    // wrong image (deterministically — invisible to 1-vs-z). 2·rc < L ⇒ all three legs unambiguous.
    for (int d = 0; d < 3; ++d)
      if (box.periodic[d] && box.len(d) < 2.0 * rcut_)
        throw std::invalid_argument("meam_ring: periodic box dim < 2·rc — min-image ambiguous");
    // periodic-z is supported: cyclic window + defer_head + tail-batched sends.
    // ZoneDecomposition::build rejects periodic n_zones in 2..4 for reach_mult=2, so PBC
    // only reaches n=1 (free path) or n>=5 (distinct cyclic zones — no window dedup needed).
  }

  ConveyorResult run() {
    zd_ = core::ZoneDecomposition::build(atoms_, box_, o_.n_zones, rcut_, /*reach_mult=*/2);
    n_ = zd_.n_zones;
    z_ = o_.n_nodes;
    // descriptor firewall: a GPU window-force policy validates pot_.passes() (the MEAM
    // [Density,Embedding,Force(needs_transpose)] descriptor — a symmetric-only accumulator would
    // silently run the screening 3rd-atom write wrong). The CPU policy DOES define assert_supported
    // ⇒ this fires at run() (a self-check on the descriptor wiring; MeamWinForce ACCEPTS the
    // needs_transpose Force).
    if constexpr (requires { WinForce::assert_supported(pot_.passes()); })
      WinForce::assert_supported(pot_.passes());

    // t0 forces via the serial oracle (same kernel ⇒ same bits) for the 1st drift.
    core::zero_forces(atoms_);
    const double pe0 = meam_zone_pass(atoms_, box_, zd_, pot_.meam).pe;
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
      set_halt(Halt::Internal, std::string("meam_ring internal error: ") + ex.what());
    }
  }

  bool run_pass(int k, long h, std::vector<ZoneMsg>* preload) {
    const int in_edge = (k - 1 + z_) % z_;
    const int out_edge = k;
    const auto io = core::node_io_order(k + 1);  // §7.4 parity (1-based)
    const int r = box_.periodic[2] ? int((h - 1) % n_) : 0;  // pass-order rotation
    const bool defer_head = box_.periodic[2] && n_ > 1;      // §7.2 closure ([ENG])
    const core::PairGeom geom(box_, rcut_);
    const double rho_cap = 0.0;  // MEAM density is internal — passed and ignored by MeamWinForce

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
          throw std::logic_error("meam_ring: ring arrival out of order");
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
      // MEAM has no pair-cross "partial force into successor", so EVERY zone's
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
      // gather (E5b F4): PBC cyclic {(j-1)%n,j,(j+1)%n} (n>=5 ⇒ distinct), free-z
      // drop out-of-range. (PR-E3b-PBC; factored E5b.)
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
          throw std::logic_error("meam_ring: window slot not resident at finalize");
      }
      // gather contiguous window (key = atom id for the φ-once / screening order)
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
      // M6 Me3b: the SINGLE moving part — delegate to the window-force policy
      // (CPU default = meam_window_force via MeamWinForce; GPU = the Me5 screening kernels).
      // pe/min_r2 are updated exactly as before; the int64 force result is order-free.
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
      if (m.n() != int(mem.size())) throw std::logic_error("meam_ring: final zone size mismatch");
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
  const MeamPotential<Real>& pot_;
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

// Convenience: run the MEAM ring on `atoms` (mutated in place), return result.
// Default policy (CPU) — byte-identical to the serial meam_zone_pass oracle.
template <typename Real>
ConveyorResult run_meam_ring(AtomSoA<Real>& atoms, const Box& box,
                           const MeamPotential<Real>& pot, const ConveyorOptions& o) {
  MeamRing<Real> r(atoms, box, pot, o);
  return r.run();
}

// Policy-injected overload: run the ring with a custom window-force policy (e.g. the
// Me5 GpuMeamWinForce). Orchestration identical ⇒ bitwise ≡ the CPU ring by construction.
template <typename Real, typename WinForce>
ConveyorResult run_meam_ring(AtomSoA<Real>& atoms, const Box& box,
                           const MeamPotential<Real>& pot, const ConveyorOptions& o,
                           WinForce winforce) {
  MeamRing<Real, WinForce> r(atoms, box, pot, o, std::move(winforce));
  return r.run();
}

}  // namespace tdmd::potentials
