#pragma once
#include <functional>
#include <span>
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

struct PassDecl {
  PassKind kind;
  bool reuses_candidates = true;  // same neighbour set as the previous pass
  bool needs_transpose = false;   // j→i connectivity (ReaxFF 3/4-body); EAM: false
  bool iterative = false;         // inner solver (QEq/CG); EAM: false
  int  accum_fracbits = 40;       // FixedAccum<> this pass writes (Force 40, Density 44, Energy 30)
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
