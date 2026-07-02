#pragma once
#include <concepts>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"  // PairGeom — single source of min-image/cutoff (HOST_DEVICE)

// M6 — universal multi-pass potential contract (IManyBodyPotential).
// Design of record: docs/_meta/M6_EAM_MANYBODY_DESIGN_2026-06-14.md.
//
// This header is PR-E0: it defines the descriptor types + the ABSTRACT
// interface only. The concrete EAM instance (analytic Finnis–Sinclair, then a
// HOST_DEVICE cubic-spline setfl form) lands in PR-E1/PR-E2; the multi-pass
// driver over passes() and its three-zone residency on the ring land in PR-E3.
//
// SUBSUMPTION OF PairFn — honest statement (§2 of the design): the existing
// pair hot paths stay TEMPLATE-INLINE with zero indirection — direct_pair_loop
// <PairFn> / ClusteredPairEngine::run_pairs<PairFn> for compute()/run_simulation
// and TimeConveyor<Real,PairFn> for the ring. They are NOT re-routed through
// this virtual interface (that is WHY adding it is a bitwise no-op for the pair
// suite). A pair potential is SEMANTICALLY the degenerate single-pass instance
// {PassKind::Force, reuses_candidates=true, effective_range={1,false}}, but the
// virtual run_pass() is only ever taken by many-body potentials. Two dispatch
// mechanisms coexist (template pair + virtual many-body), both no-vtable in the
// pair inner loop.
namespace tdmd::potentials {

using core::AtomSoA;
using core::Box;
using core::PairGeom;

// A potential is a graph of passes run, in order, over ONE candidate topology.
//   EAM: [Density, Embedding, Force]; Tersoff/SW: +BondOrder; ReaxFF: +QeqIter.
enum class PassKind { Density, Embedding, Force, BondOrder, QeqIter };

// === PR-0a (W-contract, audit §3.3) ===
// W-classification of a pass: HOW the pass MAY execute as int64 donation batches
// at zone co-residency (CoRes) events. SEMANTICS: w_class declares a CAPABILITY,
// not an obligation — a C-only consumer (today: every ring and eam_gpu_run_
// singlenode) legally IGNORES the field and runs the monolithic path; so flipping
// EAM Density → kAccumB1 in PR-1 breaks no consumer. In PR-0a NOTHING consumes the
// field — the kCOnly default reproduces today's behavior bit-for-bit.
// kAccumB1 is DEFINED by int64 associativity (lemma W-1): for ANY partition of the
// contribution multiset into batches and ANY execution order the raw int64 equals
// the monolithic sum. ⇒ a kAccumB1 pass MUST accumulate in a FixedAccum (accum_
// fracbits>0); an FP reduction (Tersoff ζ: FP64 by design, fb=0) is NOT under W-1 —
// its class is kAccumCanonSuffix (DEFER-AND-GATE, v1 does not implement).
// Donation batches do NOT touch min_r2 / PE / φ-once — those stay in the force pass
// of the C-phase; re-specify this if the force pass is ever split (audit §3.3).
enum class WClass : uint8_t {
  kCOnly,             // C-phase only (default = today's behavior)
  kAccumB1,           // int64-accumulable: donation batches in d/w (W-1/W-2)
  kAccumCanonSuffix,  // FP reduction under a block-zone canon + carry (DEFER-AND-GATE)
  kWrapup             // nonlinear per-atom convolution — C-phase
};

// Donor roles of a zone k in the pass slot-space: self(k); cross(k-1,k) as the HI
// endpoint; cross(k,k+1) as the LO endpoint. kReserved3 is held for a future
// screening/transpose-donor class (MEAM); forbidden in v1.
enum class DonorRole : uint8_t { kSelf = 0, kCrossLo = 1, kCrossHi = 2, kReserved3 = 3 };
inline constexpr uint8_t donor_bit(DonorRole r) { return uint8_t(1u << unsigned(r)); }
inline constexpr uint8_t kDonorRolesValidMask = 0x07;  // kReserved3 forbidden in v1

struct PassDecl {
  PassKind kind;
  bool reuses_candidates = true;  // same neighbour set as the previous pass
  bool needs_transpose = false;   // j→i connectivity (ReaxFF 3/4-body); EAM: false
  bool iterative = false;         // inner solver (QEq/CG); EAM: false
  int  accum_fracbits = 40;       // FixedAccum<> this pass writes (Force 40, Density 44, Energy 30)
  // --- PR-0a: INERT defaults (zero behavior change) ---
  WClass  w_class     = WClass::kCOnly;
  uint8_t donor_roles = 0;   // bitmask of donor_bit(DonorRole); 0 <=> kCOnly/kWrapup
};

// PR-0a firewall teeth (audit §3.3, MUST-FIX baked in). Descriptor SELF-consistency,
// independent of the WinForce policy; the rings call this UNCONDITIONALLY before
// WinForce::assert_supported; assert_eam_symmetric_passes calls it itself.
//
// HONEST CAVEAT (mandatory, do not reword): the machine check catches ONLY the
// zeta-class (kAccumB1 declared on a pass whose reduction is FP: markers accum_
// fracbits==0 / kind==BondOrder). The canon-dependent-addend class — MEAM int64
// lanes whose double addends depend on S_ij=Π_k (an order-sensitive product BEFORE
// quantization) — is NOT machine-catchable: such a pass legally carries fracbits>0
// and kind==Density. Its witnesses are only the classifying PR's adversarial design
// + the independent FP64 oracle (*_direct_fp64, class MB2). Do not extend the
// kAccumB1 claim beyond what this function proves.
inline void validate_pass_decls(std::span<const PassDecl> passes) {
  if (passes.empty() || passes.size() > 8)
    throw std::runtime_error("validate_pass_decls: 1..8 passes "
        "(closure_bit packs 4 roles x 8 passes into uint32)");
  for (std::size_t p = 0; p < passes.size(); ++p) {
    const PassDecl& d = passes[p];
    if (d.donor_roles & ~kDonorRolesValidMask)
      throw std::runtime_error("validate_pass_decls: reserved donor_roles bit set (pass "
          + std::to_string(p) + ")");
    if (d.iterative && d.w_class != WClass::kCOnly)
      throw std::runtime_error("validate_pass_decls: iterative pass (QEq/CG) cannot be "
          "donated (pass " + std::to_string(p) + ")");
    if (d.w_class == WClass::kAccumB1) {
      if (d.accum_fracbits <= 0)
        throw std::runtime_error("validate_pass_decls: kAccumB1 requires accum_fracbits>0 "
            "(int64 FixedAccum is WHAT makes donation batches order-free, lemma W-1); "
            "an FP-reduction pass cannot be kAccumB1 (pass " + std::to_string(p) + ")");
      if (d.kind == PassKind::BondOrder)
        throw std::runtime_error("validate_pass_decls: kAccumB1 forbidden on BondOrder "
            "(zeta is an FP64 fold by design — tersoff.hpp; the canon class is "
            "kAccumCanonSuffix) (pass " + std::to_string(p) + ")");
      if (d.donor_roles == 0)
        throw std::runtime_error("validate_pass_decls: kAccumB1 with empty donor_roles "
            "(nothing would ever close the ledger) (pass " + std::to_string(p) + ")");
    } else if (d.donor_roles != 0) {
      // STRICT in v1 (deliberate, К4): roles allowed ONLY on kAccumB1. Canon donations
      // (kAccumCanonSuffix) are not designed — their PR relaxes this together with the
      // semantics + teeth. kAccumB1+roles==0 also throws (likely typo); the first
      // consumer PR may relax it deliberately.
      throw std::runtime_error("validate_pass_decls: donor_roles set on a non-kAccumB1 "
          "pass (pass " + std::to_string(p) + ")");
    }
  }
}

// Completeness-ledger bit layout: closure_bit(pass_idx, role) = pass_idx*4 + role,
// pass_idx < 8 (uint32 = 4 roles x up to 8 passes — an extension of the INV-3 ledger
// core::Zone::contrib_mask). Consumed by PR-1/2; provided pure here.
inline constexpr uint32_t closure_bit(int pass_idx, DonorRole role) {
  return 1u << (unsigned(pass_idx) * 4u + unsigned(role));
}
// Mark a cross-batch in BOTH zones' masks (the one place PR-2 would get wrong by
// hand): CrossHi in the lower zone of the edge, CrossLo in the upper zone.
inline constexpr void close_cross_batch(uint32_t& mask_lo, uint32_t& mask_hi, int pass_idx) {
  mask_lo |= closure_bit(pass_idx, DonorRole::kCrossHi);
  mask_hi |= closure_bit(pass_idx, DonorRole::kCrossLo);
}

// PR-0a: the WinForce-policy contract, PROMOTED from duck-typing (the MB1/MB2
// promise in eam_window_force_gpu.cuh — overdue since MEAM). Requires:
//  - copy_constructible (rings store the policy by value; NOT trivially_copyable —
//    GPU policies hold a shared_ptr to device state);
//  - static assert_supported(span<PassDecl>) — MANDATORY: kills the silently-
//    bypassable `if constexpr requires` (a policy with a typo is now a COMPILE error);
//  - the FROZEN 14-arg const compute() (rho_cap is the 9th param, named-discarded by
//    SW/Tersoff/MEAM; do not change).
// PR-0a SCOPE (honest): the on_zone_arrival/on_edge hooks (audit §3.3) join the
// concept TOGETHER with their consumers (PR-1/2) — a mandatory-but-never-called hook
// today would be structurally-dead code (5 recidive precedents); extending the
// concept is compile-time enforced, impossible to forget later.
template <typename WF>
concept WindowForcePolicy =
    std::copy_constructible<WF> &&
    requires(const WF wf, const double* d, const long* k, int i, const int* ip,
             const core::PairGeom& g, double rc,
             std::vector<core::fixed::ForceAccum>& f,
             core::fixed::EnergyAccum& pe, double& mr,
             std::span<const PassDecl> ps) {
      { WF::assert_supported(ps) } -> std::same_as<void>;
      { wf.compute(d, d, d, k, i, ip, i, g, rc, f, f, f, pe, mr) } -> std::same_as<void>;
    };

// Effective FORCE radius in units of rcut + reach symmetry. This is the single
// source of truth driving the geometry guards (M6_EAM_MANYBODY_DESIGN §1.2/§5):
//   cutoff_multiplier m : pair 1; EAM 2 (f_i needs F'(ρ_j), ρ_j reaches 2·rcut).
//   symmetric_reach     : node must co-reside BOTH immediate neighbours
//                         {S_{i-1}, S_i, S_{i+1}} (three-zone window) — EAM's
//                         density of lower-face neighbours lives in S_{i-1}.
// Feeds: ZoneDecomposition::build(reach_mult=m) [width>=m·rcut, >=2m+1 periodic
// zones]; conveyor membership g=0.5·(w−m·rcut); ring residency window (3 zones
// when symmetric_reach, 2 otherwise); GPU slot-pool +1 co-residency.
struct EffectiveRange {
  int  cutoff_multiplier = 1;
  bool symmetric_reach = false;
};

// EffectiveRange for a config potential.type. Pair types are {1,false} (inert);
// EAM is {2,true}. Centralised so the CLI/conveyor wiring (PR-E3) has one map.
inline EffectiveRange effective_range_for(const std::string& pot_type) {
  if (pot_type == "eam") return {2, /*symmetric_reach=*/true};
  if (pot_type == "sw") return {2, /*symmetric_reach=*/true};  // SW φ₃ wing reaches 2·rcut
  if (pot_type == "tersoff") return {2, true};  // Tersoff k-atom of bond (i,j) reaches 2·rcut
  if (pot_type == "meam") return {2, true};  // MEAM screening k-atom + 2·rcut density reach
  return {1, false};  // morse | lj — pair
}

// One accepted pair after PairGeom::reduce (min-image + cutoff). r = sqrt(r2).
struct PairGeomResult {
  double dx, dy, dz, r2, r;
};

// Per-atom scalars the engine threads through the passes. Density is FIXED-
// POINT (B1, DensityAccum=FixedAccum<44>) so the intermediate ρ is order-free
// deterministic exactly like the forces. nullptr in a pure-pair pass.
template <typename Real>
struct ManyBodyState {
  std::vector<core::fixed::DensityAccum>* rho = nullptr;  // size a.n; per-atom EAM density
  std::vector<double>*                    fp  = nullptr;  // F'(ρ_i)=dF/dρ, reconstructed + spline
  std::vector<core::fixed::ForceAccum>*   fx  = nullptr;  // SAME Q24.40 the pair path writes
  std::vector<core::fixed::ForceAccum>*   fy  = nullptr;
  std::vector<core::fixed::ForceAccum>*   fz  = nullptr;
  core::fixed::EnergyAccum*               pe  = nullptr;
};

// Replays the engine's accepted-pair set: visit_pairs(fn) calls
// fn(i, j, PairGeomResult) once per accepted contribution. The two std::function
// layers let a pass walk the SAME candidate topology (reuses_candidates) the
// engine built once for the zone. A pass reads/writes ONLY the accumulators in
// `st`; it does NO global reduction and NO cross-node communication — purely
// local to the atoms resident in the node's window (owned + symmetric halo).
template <typename Real>
using VisitPairs = std::function<void(
    const std::function<void(int, int, const PairGeomResult&)>&)>;

template <typename Real>
struct IManyBodyPotential {
  virtual ~IManyBodyPotential() = default;
  virtual std::span<const PassDecl> passes() const = 0;
  virtual EffectiveRange effective_range() const = 0;
  // Run one pass over the candidate topology. (PR-E1+ provide EAM.)
  virtual void run_pass(int pass_idx, const AtomSoA<Real>& a, const PairGeom& geom,
                        ManyBodyState<Real>& st,
                        const VisitPairs<Real>& visit_pairs) = 0;
};

}  // namespace tdmd::potentials
