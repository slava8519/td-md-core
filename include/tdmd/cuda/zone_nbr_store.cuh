#pragma once
// PR-0c (W-contract ladder) — ZoneNeighborStore SCAFFOLD (the contract home).
//
// DEVICE-FREE BY DESIGN in PR-0c: this file has ZERO __global__ / cudaMalloc / device
// buffers and MUST NOT #include <cuda_runtime.h> — it is plain C++ compiled by g++ (the
// CPU test tests/test_rebuild_epoch.cpp includes it, which mechanically enforces the
// device-freeness: the moment a device consumer (PR-1/PR-3a) adds a __global__ here, the
// CPU TU breaks and the author must move the include into an nvcc TU — the "consumer
// arrives with the scaffold" rule made mechanical, PR-0a D2).
//
// The ONLY executable element today is the re-export of the (already unit-tested) epoch
// primitive, so future device consumers write `cuda::nbr_store::stale(...)` against ONE
// source. Everything else is the frozen §3.5 Tier-0/1/2 CONTRACT (a doc-comment shipped
// BEFORE its consumers, exactly like donation_layout's 8-clause SPEC in PR-0a) + explicit
// DEFER records. NO device state is allocated here — an inert buffer no kernel touches is
// precisely the structurally-dead class the project defers (the 5-recidive precedent:
// Te1 fc_d / Me1 partial-S / Me5 taper / Me5b OOB / M4-S).
//
// Design of record: docs/_meta/AUDIT_W_PHASE_NEIGHBORS_2026-07-02.md §3.5 + PR0C_* design.

#include "tdmd/core/rebuild_epoch.hpp"

namespace tdmd::cuda::nbr_store {

// The lazy-materialization epoch primitive, re-exported so device consumers key off ONE
// source. Exercised through this alias by Test_Rebuild_Epoch (else the re-export is dead).
using core::rebuild_epoch::advance;
using core::rebuild_epoch::stale;

// ===========================================================================
// §3.5 ZoneNeighborStore CONTRACT (frozen; consumers land it per the ladder).
//
// Tier-0 — per-zone population CSR over the single global lattice geometry (resolve 1x,
//   AUTO sub-rcut k). DEFAULT ON. [MUST-FIX, AUDIT] specify the binning of DRIFTED atoms:
//   slab + drift-padding >= g + a sticky exit guard, OR a CSR over the global index space
//   with counted memory (counts x n). L-SUP insures an extra REQUESTED slab, but NOT a
//   donor that fell out of ALL CSRs; the G-B fixture is "a donor drifted across a slab
//   boundary". Deterministic clamp beyond the lattice extent + stencil wrap at the box seam.
//   CONSUMER: PR-1 (Tier-0 binning + G-B), PR-3a.
//
// Tier-1 — window = a VIEW-concatenation of <=3 slabs (no re-rasterization). DEFAULT ONLY
//   EAM (+SW, int64-order-free); Tersoff/MEAM keep host gather + canonical-sort until a
//   per-potential proof that in-kernel canonical-k/zeta covers ALL FP folds on an unsorted
//   slab window (the window-sort ring<->serial is load-bearing there). CONSUMER: PR-3a.
//
// Tier-2 — persistent per-(zone,role) Verlet-CSR (rcut+skin), key-sorted at build, k-way
//   role merge (NL-INV-K). opt-in, memory-gated: neighbor.mem_budget_gib (default 3.5) +
//   a VRAM probe in the ctor; EAM Al_zhou ~2.1 GiB @1e6 passes, ~14-21 GiB @1e7 blocks =>
//   fallback = the current cells engine; angular = NO-GO until an ncu enumeration-share
//   probe >15%. CONSUMER: PR-4.
//
// MPI lazy epoch (this file's live primitive above): a rank keeps a per-rank local_epoch[]
// and materializes zone z iff stale(local_epoch[z], hdr.rebuild_epoch); conservativeness =
// skin_consumed monotone from the epoch; bitwiseness = L-SUP. CONSUMER: PR-4 (write head
// sent==0, read arrival-0, the per-rank local_epoch[] state, + G-EPOCH with verlet_hybrid=ON,
// run LOCALLY since Test_MPI_Conveyor is in no cloud CI loop).
//
// DEFERRED — the consumer arrives WITH the scaffold (PR-0a D2, anti-structurally-dead):
//   * Tier-0 CSR binning + G-B straddle fixture      => PR-1 / PR-3a
//   * Tier-1 window-view                              => PR-3a
//   * Tier-2 per-(zone,role) Verlet-CSR + mem-gate    => PR-4
//   * per-rank local_epoch[] device/host state        => PR-4 (an inert buffer now = D2)
//   * epoch WRITE/READ in the live ring + G-EPOCH     => PR-4
// No NeighborTier vocabulary-enum is shipped here: an enum nobody switches on is itself
// softly dead — the tier type arrives with its Tier-0 consumer (PR-1/3a) + the logic.
// ===========================================================================
//
// PR-3a REALIZATION NOTE (append-only; design PR3AB_GPU_DONATION_DESIGN_2026-07-07 §0
// pt.5 + amendment S3 — this note SUPERSEDES the two stale consumer lines above, quoted):
//   * Tier-0: REALIZED in PR-3a as the SECOND sanctioned MUST-FIX shape — "a CSR over
//     the global index space with counted memory": the donation batches refresh the
//     population of the EXISTING whole-box lattice CSR (eam_window_force_gpu.cuh shared
//     grid; geometry resolved once, counted memory, zero new grid allocations). Drift
//     binning is discharged BY CONSTRUCTION (a drifted donor bins at its TRUE cell — no
//     slab boundary exists to escape); the G-B straddle fixture is gate A5
//     (DriftedDonorAcrossZoneBoundary, test_cuda_eam_donation.cu), teeth =
//     drop_outside_nominal_slab poison (the slab-clamp mutation was MEASURED
//     structurally dead — amendment M4). The line "CONSUMER: PR-1 (Tier-0 binning +
//     G-B), PR-3a" above is SATISFIED in this shape.
//   * Tier-1: the SLAB RESIDENCY half landed in PR-3a (per-zone LABEL-keyed device
//     position mirrors + persistent int64 rho lanes in GpuEamDonationNodeStore — the
//     donation-side consumer); the WINDOW-AS-VIEW-CONCAT consumption half is DEFERRED
//     to M5b/on-measure (its only production effect today is H2D 3x->1x ~= 1% of wall —
//     non-claimable; once slabs + WindowBlocks labels exist it is a ~10-line compose()
//     addition). The lines "CONSUMER: PR-3a" (Tier-1) and "Tier-1 window-view => PR-3a"
//     above are SUPERSEDED accordingly: residency PR-3a, consumption DEFER -> M5b.
//   * The ZoneLane.stamp pass-token in GpuEamDonationNodeStore is the PR-4 seat for the
//     rebuild_epoch primitive re-exported above (token-compare -> advance/stale).

}  // namespace tdmd::cuda::nbr_store
