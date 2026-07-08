# PR-3a + PR-3b FROZEN DESIGN — device donations for the EAM GPU ring (W-contract PR-3, judge-merged)

Status: **FROZEN** (adversarial design workflow: 4 readers → 2 designers → judge). Date 2026-07-07.
Base: HEAD `eb5c43f` (PR-2 live donation ring landed; working tree clean — verified by the judge).
All paths absolute under `/home/slava8519/td-md-core/`. Anchors verified against the working tree by 4 readers + judge spot-checks (`eam_ring.hpp` donate_position/finalize_owned call sites, `eam_donation.hpp:527-538` concept, `eam_window_force_gpu.cuh:409-431` inert hooks).

---

## 0. SPLIT DECISION (frozen by the judge)

**PR-3a = the audit FALLBACK shape (Design A), strengthened with Design B's stamp/token machinery. NOT full Tier-1 window view-concat.**

PR-3a lands the complete device-donation SUBSTRATE, default-inert:
- per-node, per-zone (**LABEL-keyed**) device position slabs + persistent int64 ρ lanes (`GpuEamDonationNodeStore`, **ring-created** — audit :280-286 ownership shape),
- donation kernels: SELF = `eam_density_cells_kernel`/`eam_density_kernel` **byte-verbatim reuse** over the slab; CROSS = ONE new kernel pair,
- concept v2 (`DonatingWindowForcePolicy`: `NodeState` + `WindowBlocks`) — the sanctioned second fork-divergence (PR-2 §11), spent ONCE, in 3a,
- everything behind ctor knob `donate_device=false` ⇒ the default path is **byte-identical to HEAD** (3a criterion "bitwise + wall ±3%" holds by construction); knob-ON is exercised by gates comparing device ρ raw-int64 vs CPU-donated ρ (exact `EXPECT_EQ`, never tolerance).

PR-3b flips consumption only: `compose()` D2D-scatters the ≤3 per-zone ρ lanes into `d_rho` in window block order, SKIPS the density kernel, applies the К3 cap move (owned-full-ρ only, via a tiny separate kernel — force kernels stay byte-frozen), default flips `donate_device=true` behind the pre-registered **R_W ≥ 1.15** gate with a surgical **two-commit NULL rollback**. **Force path untouched** (host gather + full-window H2D stays) — exactly the audit :350-352 fallback wording.

### Judge reasoning (why fallback, not full Tier-1)

1. **Bitwise by construction beats bitwise by test.** Full Tier-1 (Design B) forks the force-path candidate enumerator (slab-CSR walk in `eam_force_slabs_kernel` — the most delicate kernel: min_r2-before-cutoff, φ-once on global keys, embedding-PE, int64 tree-reduce) plus a band-CSR build path with escape guards, modular bands and terminal-row clamps — ~9 new/forked kernels whose equivalence is by-argument+by-test. The fallback keeps the frozen `eam_density/embedding/force[_cells]_kernel` **byte-untouched**, reuses one of them verbatim as the SELF batch, and adds exactly ONE new kernel pair (cross) whose body is the cells stencil walk verbatim with two mechanical deltas.
2. **Full Tier-1's only production effect is H2D 3×→1× ≈ 1% of wall — explicitly NON-claimable** (audit :317; ring is 99% kernel-bound, E5c-ring measurement). Zero gated payoff funds a rework of the proven force path (key-semantics fork risk, per-zone AUTO-k geometry, stale-lane indexing, pbc-n=1 gather parity).
3. **Design B's attribution argument dissolves against Design A's actual shape.** B argued "if slabs+CSR land with donations, an R_W miss can't be attributed". But in the fallback, the 3b A/B legs share the **identical, today's-proven** force path and differ ONLY in the density-work source (donation kernels + D2D ρ scatter vs density kernel). Attribution is at least as clean as B's — with none of B's ±3%-wall risk, whose own recorded contingency (B §11.R1) *is* the fallback shape. When a design's contingency equals the rival design and the gated payoff is ~0, the contingency is the plan.
4. **Consumer-arrives rule** (PR-0c D2, E5c-concurrency rollback precedent, PR-2 K1): the consumer of device-resident *window consumption* is M5b; the consumer of per-zone slab *residency* (donation kernels reading position mirrors) arrives in PR-3a. So slabs land in 3a; window-as-view-concat does not. Once slabs + `WindowBlocks` labels exist, the window position D2D-concat is a ~10-line `compose()` addition when M5b measures a consumer.
5. **Tier-contract honesty:** `zone_nbr_store.cuh` gets an APPEND-ONLY re-annotation, not silence: Tier-0's PR-3 realization = the second sanctioned MUST-FIX shape ("CSR over the global index space with counted memory") — the existing whole-box lattice CSR with per-batch population refresh; drift-binning discharged BY CONSTRUCTION (a drifted donor bins at its TRUE cell — no slab boundary to escape), the G-B straddle fixture still mandatory (gate A5). Tier-1: slab residency lands PR-3a (donation-side consumer); window view-concat consumption **DEFER → M5b/on-measure**.
6. **PR-2 §11 letter deviation blessed:** both candidate designs land the hook/compose signature change in 3a, not 3b. Justification: §11's intent was "the signature change travels with the device-store consumer"; this split moves the substrate to 3a, so the change accompanies its consumer, once. 3b then changes NO signatures (`eam_ring.hpp` + `eam_donation.hpp` git-byte-frozen in 3b — Design B's freeze discipline, adopted). Base `WindowForcePolicy` (many_body.hpp) + sw/tersoff/meam siblings byte-frozen in BOTH PRs.

### Adopted from Design B (explicit)
- Pass-token + per-lane `stamp` staleness fence + `StaleStampThrows` gate — the exact seat for PR-4's `rebuild_epoch` (`cuda::nbr_store::advance/stale`).
- `NodeState::begin_pass(long h)` (host-only token set; CPU inert).
- `WindowBlocks.center` field (owner-block index; cheap asserts + K3 attribution convenience).
- Ring-owned `node_stores_` vector created in `run()` (explicit ownership; per-node, per audit shape).
- Honest R_W expectation band (removable fraction r ∈ (0.5·s̄, 0.67·s̄)).
- 3b file-freeze discipline: 3b = GPU policy + kernels + bench + tests + docs ONLY.

### Rejected from Design B (explicit, with reasons)
- Slab-CSR force-path enumeration + per-zone band CSR + escape guard + terminal-row clamp (complexity serving the rejected enumerator; whole-box binning discharges the drift MUST-FIX by construction).
- Owned-range D2H (closes the known uninitialized-D2H initcheck seam but TOUCHES the frozen compute path — deferred as a separate micro-change with its own manual-initcheck evidence; never bundled with the donation flip).
- Device key mirrors (donations never touch keys — φ-once/PE/min_r2 stay C-phase verbatim, WContract §6; saves ~80 MB/node @1e7).

---

## 1. PRE-REGISTERED MEASUREMENT (verbatim, frozen 2026-07-07 BEFORE any design/code)

> PRE-REGISTERED measurement (2026-07-07, idle RTX 5080, `bench_eam --backend cells --cellk 3 --free`, Al_zhou setfl, 50 steps, 8 reps/size; frozen BEFORE any design/code):
> ```
> N=2048:   density 40.8±0.3% / force 59.1% / embed ~0.1%
> N=6912:   density 40.9±0.5% / force 59.1%
> N=16384:  density 39.7±0.1% / force 60.2%
> N=42592:  density 40.1±0.1% / force 59.9%
> N=108000: density 42.2±0.1% / force 57.7%
> ```
> ⇒ density share in cells mode ~40% (audit predicted 30–40%); theoretical R_W ceiling 1/(1−(2/3)·s) = 1.36–1.39 assuming donation removes exactly 2/3 of density work with ZERO overhead. Honest caveats: cross pairs today cost ×2 (not ×3) so true saving is below 2/3·s; donation adds per-batch kernel-launch overhead; small zones risk GPU underload (batches of s atoms instead of 3s). The R_W ≥ 1.15 gate (audit line 315) stands as pre-registered: live ring, size grid (small N + saturating point ≥ ~21k atoms/zone), idle GPU, reps≥8, NULL ⇒ rollback (E5c-concurrency precedent).

Refined removable-fraction model (frozen with the design, Design B's honest band): an intra-zone density pair is computed 3× per pass today (once per window containing the zone), a cross-zone pair 2×; donation computes each exactly 1× ⇒ removable r = (2/3)·intra_share + (1/2)·cross_share of density time ∈ (0.5·s̄, 0.67·s̄) ≈ **0.20–0.27** of kernel time at s̄ = 0.40 ⇒ ceiling R_W ≈ **1.25–1.37** BEFORE donation-kernel underload (s-sized launches vs 3s), +~2n launches/pass, ~3n grid-population refreshes/pass, and D2D-scatter cost. Realistic 1.10–1.30 at ≥21k atoms/zone ⇒ **the 1.15 gate is genuinely at risk — that is the point of the gate.**

---

## 2. ARCHITECTURE + DECISION LOG

```
                ring (eam_ring.hpp, policy-agnostic, ledger-owner, schedule/Λ/parity UNTOUCHED)
 pass h, node k:  arrivals ─ drift ─ donate_position(j) ─ finalize_owned(j) ─ END(j) ─ sends
                                  │                              │
                        hooks(NodeState&, dstate, …)     compose(NodeState&, …, WindowBlocks, …)
                                  │                              │
 GPU policy      on_zone_arrival(label):                 [3a & 3b-recompute] host gather + full-window H2D
 (knob-ON):        H2D slab(label) 24 B/atom, stamp=token   + density/embedding/force — VERBATIM TODAY
                   memset ρ lane; SELF kernel             [3b donated] same H2D of positions/keys/owned;
                   (= eam_density_cells_kernel VERBATIM)    D2D ≤3 ρ lanes → d_rho (block order);
                 on_edge(la,lb): D2D concat 2 slabs,        NO density kernel; embedding_nocap;
                   grid refresh, CROSS kernel               eam_owned_cap_kernel (К3);
                   (ONE new kernel pair)                    force kernel BYTE-VERBATIM; D2H+sync+flags
```

Decision log (J-numbered; A/B = source design):

- **J1 (split)** — §0. Fallback shape; force path byte-frozen in both PRs.
- **J2 (ownership, A+B)** — per-NODE `GpuEamDonationNodeStore`, created by the ring: `EamRing::run()` builds `node_stores_` (one per node jthread, `std::vector<typename WinForce::NodeState>`) after the firewall asserts and `ZoneDecomposition::build` (n_ known), before threads spawn. NOT in the z-shared `GpuEamWindowState`, NOT in the growable window scratch (`grow()` frees without preserving — eam_window_force_gpu.cuh:150-170). Audit :280-286 shape verbatim; hard constraint 1 ("donation state is per-node pass-local") upheld.
- **J3 (concept v2 once, in 3a)** — §0 pt.6. Refinement concept only; base `WindowForcePolicy` + siblings byte-frozen.
- **J4 (donation culling, A)** — ONE whole-box grid geometry (today's `make_zone_grid(n_zones=1)`, F3 periodic-z fold, ulp-skin pad, stencil from REALIZED cell size, AUTO sub-rcut k resolved once via `ensure_grid_geometry`), population REFRESHED PER BATCH into the SHARED `EamCellGrid` arrays (count → CUB scan → scatter — the identical refresh compose already does per window). Consequences: (a) drifted donors bin at their TRUE cell ⇒ Tier-0 drift MUST-FIX discharged by construction (no slab clamping, no escape guard; G-B fixture still gated, A5); (b) pbc seam pairs land in stencil-adjacent wrapped cells — no fold/clamp special-casing; (c) counted memory = zero new grid allocations; (d) ~3n refreshes/pass instead of n — grid build measured negligible (E5c), µs-scale vs kernel seconds. REJECTED (B): per-zone band CSR + escape guard (machinery for the rejected force enumerator).
- **J5 (kernels, A)** — SELF batch reuses `eam_density_cells_kernel` (culled) / `eam_density_kernel` (plain) **byte-verbatim** (its single overwrite-write per lane IS the per-pass reset by construction — the schedule guarantees self(L) strictly before any cross touching L and any compose reading L; a defensive `cudaMemsetAsync` of the lane at arrival is kept as hygiene, NOT load-bearing). CROSS = ONE new kernel pair in a NEW header `zone_eam_donation.cuh`. No device key mirrors, no per-zone grids.
- **J6 (stamp/token, B)** — `NodeState.token` set by `begin_pass(h)`; `ZoneLane.stamp = token` set at `on_zone_arrival`. `on_edge` and the donated `compose` check `stamp == token && lane.n == expected` for every touched label, else `logic_error` → node_main catch → `Halt::Internal`. Catches label-keying bugs and missed arrivals deterministically (not via size coincidence); the stamp field is PR-4's `rebuild_epoch` seat.
- **J7 (3b compose, A over B)** — donated compose = D2D lane scatter + `eam_embedding_nocap_kernel` + tiny `eam_owned_cap_kernel` + **frozen force kernel byte-verbatim**. REJECTED (B): prepending the cap inside a forked force kernel — A's separate 8-line cap kernel keeps the force kernels untouched (stronger by-construction property).
- **J8 (rollback, A)** — PR-3b lands as TWO commits: (c1) donated compose + К3 kernels + all B-gates, default still `donate_device=false`; (c2) default flip + Bench numbers. NULL ⇒ revert **c2 only**: production returns to the recompute leg; the donated path stays knob-gated and fully test-covered (NOT structurally dead — A/B-gates keep instantiating it); PR-3a substrate stays (audit :353-354 NULL clause: "архитектурная часть PR-0..2 остаётся ценной"). If acceptance demands full removal, revert c1+c2; 3a intact either way.
- **J9 (R_W geometry)** — primary gated axis = **pbc-z, n_zones=5, z=1** (the only geometry with the FULL ×3 window redundancy for every zone — free-z edges drop to ×2, understating the claim; pbc n∈{2..4} throws, so 5 is minimal). Corroboration legs free-z n_zones=4 + Axis B, ungated. `--cellk 3` PINNED in both legs (matches the pre-registered share measurement; forbids per-leg AUTO drift). See §7.
- **J10 (deferred hygiene)** — owned-range D2H (initcheck seam) NOT bundled; separate micro-change later, with its own clean manual initcheck run as evidence.
- **J11 (counters + capture seam)** — non-vacuity counters (`donation_self_batches_`, `donation_cross_batches_`, `donated_composes_`) live in the SHARED `GpuEamWindowState`, incremented under the existing mutex, policy accessors (the `cells_passes()` pattern). Seam non-vacuity is proven ARITHMETICALLY (pbc n zones ⇒ exactly n cross batches/pass vs n−1 free — the +1 is the seam; rotation-proof, no label-keyed seam counter needed). Test capture seam = policy-level `capture_lanes` knob (default off): at each knob-ON/donated compose, D2H the ≤3 lane segments of `wb` into a mutex-held host log `{pass h, zone_j, label[3], raw int64 lanes[3]}`; the CPU side uses a test wrapper policy logging `dstate.rho[label]` at the same event. One mechanism serves gates A1 and B-D2.

### Mutex / stream discipline (constraint 4 — explicit disavowal)
ALL device work (hooks + compose) stays on the NULL stream under the ONE shared `GpuEamWindowState::mu`. No new streams, no events in default builds, buffers WITHOUT streams. **This is NOT a reincarnation of the rolled-back E5c-concurrency**: the 3b win is launched-work REMOVAL (each density pair computed once instead of intra×3/cross×2), which the existing `TDMD_EAM_RING_TIMERS` split must show as a compose `timer_kernel_s` drop ≈ the wall drop (h2d/rest flat). Null-stream ordering + per-op mutex holds make the shared grid arrays and concat scratch race-free across z node threads; every device op is self-contained under one lock: grow → fill → launch.

---

## 3. DATA STRUCTURES + LIFECYCLE + MEMORY BUDGET

### 3.1 `GpuEamDonationNodeStore` (NEW, in `eam_window_force_gpu.cuh`; handle = `shared_ptr<Impl>` so the ring's vector stores it trivially)

```cpp
struct GpuEamDonationNodeStoreImpl {
  long token = 0;                                // begin_pass(h) sets it (host-only, no device work)
  struct ZoneLane {                              // index == zone LABEL (fsm.id), NEVER slot
    double *x=nullptr,*y=nullptr,*z=nullptr;     // position slab mirror, 24 B/atom (POST-drift bytes)
    long long* rho=nullptr;                      // persistent per-zone int64 ρ lane (fb-agnostic raw)
    int cap=0, n=0;                              // grow-only capacity / members this pass
    long stamp=-1;                               // == token ⇔ uploaded this pass  [PR-4: rebuild_epoch seat]
  };
  std::vector<ZoneLane> zone;                    // size n_zones
  double *cxx=nullptr,*cxy=nullptr,*cxz=nullptr; int ccap=0;   // cross-batch concat scratch (grow-only)
  int* d_of_dn=nullptr;                          // sticky donation overflow flag (bit 1 = quantize),
                                                 // zeroed ONCE at creation — HALT is terminal for run();
                                                 // BANNER: any future pass-retry/in-flight-rescue MUST add
                                                 // a pass-boundary reset (R5)
};
```
- **No device key mirrors** (donations never touch keys — WContract §6). **No per-zone grids** (J4).
- CPU policies define `struct NodeState { void begin_pass(long) {} };` + `NodeState make_node_state(int) const { return {}; }` — inert; CPU ring byte-identical (F-NOOP gate = CPU suite bitwise).

### 3.2 Shared `GpuEamWindowState` — deltas
Window scratch, grid, min_r2 sentinel (bits-of-1e300, F5), ctor-frozen `dens_scale` from **runtime** `density_fracbits()` (constraint 7), rho_cap: ALL unchanged. Added: donation counters (J11), capture log (test-only), `GpuDonationPoison` knobs (test-only, default off — never a kernel default; P-k precedent).

### 3.3 Lifecycle table

| event | ring (unchanged logic) | device side (policy, knob-ON / donated) |
|---|---|---|
| `run()` | creates `node_stores_` (z entries) | lazy allocs on first use; `d_of_dn` alloc+zero |
| pass head | `ns.begin_pass(h)` in `run_pass_impl` | `token = h` (host only) |
| RECV/arrival | `dstate.reset_zone(label)` (host mirror, unchanged) | nothing (stamp is the staleness fence) |
| self(L) batch | ledger exactly-once THROW; hook call | lock; `DA::kScale==dens_scale` assert (T-12); lane grow; `n=blk.n`; `stamp=token`; if n==0 return (stamp set, ledger closes vacuously — WContract §5); H2D slab (post-drift bytes — `donate_position` runs after `ensure_drift`, eam_ring.hpp:549-553; positions immutable for the rest of the pass); memsetAsync ρ (defensive); cull: `ensure_grid_geometry(hint=3·blk.n)` + population refresh over the slab; launch SELF kernel (`d_rho=lane.rho`, `overflow=d_of_dn`); ++counter. NO sync, NO D2H |
| cross(la,lb) | ledger close (edge-order `close_cross_batch` unchanged); hook call | lock; stamp+n check both labels (schedule guarantees CoRes ⇒ both selfs done); if either n==0 return; grow concat; D2D concat slabs [A|B]; cull: population refresh over concat; launch CROSS kernel; ++counter |
| compose(j) | host gather (still runs unconditionally — policy-agnostic; GPU ignores coords→no: uses them; ignores rho_w); builds `wb`; calls compose | §4.3 / §4.4 |
| mid-pass HALT | pass stack frame discarded (dstate unwinds) | device lanes retain garbage; HALT terminal for `run()`; next arrival's stamp bump + H2D + SELF overwrite resets — no partial-replay API exists (SPEC(6), WContract §5) |

### 3.4 Memory budget @ N=1e7, z=4, n=16 (fp64; s=625k/zone; max_m=1.875e6)

| item | per node | ×z=4 |
|---|---|---|
| slab positions 24 B/atom (Σ zones = N) | 240 MB | 960 MB |
| ρ lanes 8 B/atom | 80 MB | 320 MB |
| cross-concat scratch 2·s_max·24 B | 30 MB | 120 MB |
| **NEW total** | **350 MB** | **1.40 GiB** |
| existing shared window scratch 76 B×max_m + grid per-atom 8 B×max_m | — | ~158 MB (one) |
| existing shared whole-box grid (ncells≈4e6 × 12 B + CUB) | — | ~50 MB (one) |

Grand total ≈ **1.62 GiB** device — comfortably inside 15.5 GiB; below the audit ΔMem formula because key mirrors and per-zone grids are avoided. Scaling: new term ≈ z·N·32 B + z·lanes — linear in z; a 1e7 z-sweep bench should watch VRAM (R9). Host ZoneMsg arrays unchanged (B5: zone packed exactly once; no ρ in ZoneMsg).

---

## 4. EXACT CALL FLOW

### 4.1 Concept v2 (PR-3a; `eam_donation.hpp`; base `WindowForcePolicy` BYTE-UNTOUCHED)

```cpp
struct WindowBlocks {              // filled by the RING in finalize_owned from wslots (single source of
  int nb = 0;                      // slot→label mapping stays ring-side): label[t]=slot[wslots[t]].fsm.id,
  int label[3] = {-1,-1,-1};       // n[t]=w.msg.n(), center = t where wslots[t]==j. Block order ==
  int n[3]     = {0,0,0};          // gather order [pred][center][succ]. pbc n=1 emits the {0,0,0} triple
  int center   = -1;               // — NO dedup (mirrors the ring gather; K7 dedup is serial-path-only).
};

template <class WF>
concept DonatingWindowForcePolicy = WindowForcePolicy<WF> &&
  requires(const WF wf, typename WF::NodeState& ns, potentials::EamDonationState<core::fixed::FixedAccum<44>>& st,
           int i, const ZoneBlockView& blk, const core::PairGeom& g, const WindowBlocks& wb, /*…as today…*/) {
    { wf.make_node_state(i) } -> std::same_as<typename WF::NodeState>;
    { ns.begin_pass(long{}) } -> std::same_as<void>;
    { wf.on_zone_arrival(ns, st, i, blk, g) } -> std::same_as<void>;
    { wf.on_edge(ns, st, i, i, blk, blk, g) } -> std::same_as<void>;
    { wf.compose(ns, d,d,d, k, i, wb, ip, i, g, rc, rho_w, f,f,f, pe, mr, i) } -> std::same_as<void>;
  };
```
- Concept still checks DA=`FixedAccum<44>` only; BOTH 44/40 instantiations compile via the runtime fb-dispatch (`run_pass` keyed ONLY off `density_fracbits()` — unchanged trap, documented; kill fixture A3/B5).
- `compose` gains `ns` + `WindowBlocks` — closes the PR-2-flagged gap "compose receives no labels and no rotation" WITHOUT letting the shared policy cache cross-pass mutable state (labels arrive per-call from the ring).
- CPU policies (`CpuEamWindowForce` + test policies `CpuEamRecomputeWinForce`/`CpuEamPoisonWinForce`): mechanical shims — `NodeState = struct{ void begin_pass(long){} }`, accept+ignore `ns`/`wb`; bodies (executors / `eam_window_force_from_rho`) byte-unchanged. CPU trajectories byte-gated by the existing 21-test CPU suite.

### 4.2 Ring plumbing (`eam_ring.hpp`, PR-3a ONLY; byte-frozen in 3b)
1. member `std::vector<typename WinForce::NodeState> node_stores_;` — filled in `run()` after the three firewall asserts + `ZoneDecomposition::build` (n_ known), before jthreads spawn: `node_stores_.push_back(winforce_.make_node_state(n_))` ×z.
2. `run_pass(k,h)` → `run_pass_impl<DA>(k,h)`: `auto& ns = node_stores_[k]; ns.begin_pass(h);` at the pass head.
3. `donate_position`: the two hook calls gain `ns` as arg 1 (ledger logic — exactly-once THROWs, `close_cross_batch` edge order, G10 knob — VERBATIM).
4. `finalize_owned`: builds `wb` from `wslots` (data it already iterates); compose call gains `ns` + `wb`. Host gather loop (coords/key/rho_w, label-keyed, ownedloc contiguity) untouched.
END/SEND order, batched sends, positional Λ-chain, §7.4 parity, defer_head tail, ledger-vs-want check point, membership_ok (g = 0.5·(width−2rcut), `g_override` NOT plumbed), INV-4: **not touched** (hard constraint 2; the diff is the four plumbing sites above + the policy shims).

### 4.3 PR-3a GPU hook flow (`donate_device=false` default)
Exactly the lifecycle table §3.3. Key points:
- Knob-off: hooks return before ANY device call (`make_node_state` allocates nothing) ⇒ default byte-identical to HEAD.
- Knob-ON is purely additive/observational in 3a: donation kernels write ONLY NodeStore lanes + `d_of_dn`; compose ignores both and delegates to `compute()` (recompute — verbatim today) ⇒ trajectory bitwise == knob-off (gate A0, the A7-dashboard pattern).
- pbc seam cross(n−1,0): NO special device handling — coordinates are global/unwrapped, `geom.reduce` min-images over full Lz, the whole-box periodic-z grid wraps candidates (F3); the ring fires the seam IN-SCAN at j=n−1 strictly before finalize(n−1) (eam_ring.hpp:550-553) — the frozen schedule is INHERITED by the hooks, never re-implemented (a device batch queue deferring the seam to a tail is the verifier-refuted variant; poison-gated in B2).
- Overflow coverage: every donation batch precedes ≥1 subsequent compose sync in the same pass (free-z: `donation_layout(n−1)=∅`; pbc: finalize(0) is in the tail after the seam) ⇒ the sticky-flag read at compose covers all batches of the pass.

### 4.4 PR-3b donated compose (`donate_device=true` after c2; `false` = the preserved recompute reference leg — the G1-analog requirement)
1. lock; `m<=0` early-return leaving pe/min_r2 untouched (F5 parity).
2. stamp check ∀t<wb.nb: `lane[wb.label[t]].stamp == token && lane.n == wb.n[t]` else `logic_error` (J6).
3. `grow(m)`; H2D window positions/keys/owned + zero pe / sentinel min_r2 / zero d_of — **verbatim today** (force path untouched).
4. cull: window grid population refresh — verbatim today (the force cells kernel needs it regardless).
5. **density**: ≤3 `cudaMemcpyAsync` D2D: `d_rho + off[t] ← lane[wb.label[t]].rho`, off = prefix sums of wb.n, [pred][center][succ] order (lane member i == msg index i == the host rho_w gather order, eam_ring.hpp:485-495 — order-only, no arithmetic). **NO density kernel.**
6. `eam_embedding_nocap_kernel(m)` (= `eam_embedding_kernel` minus the all-m cap atomicOr; arithmetic identical).
7. `eam_owned_cap_kernel(owned, n_owned)` — К3: `if (double(d_rho[owned[o]])/dens_scale > rho_cap) atomicOr(d_of, 2)`.
8. force kernel (`eam_force_cells_kernel` / `eam_force_kernel`) — **BYTE-VERBATIM** (min_r2-before-cutoff, φ-once on GLOBAL keys, embedding-PE per owned, int64 tree-reduce PE unchanged).
9. D2H raws + scalars + `d_of` + read `d_of_dn`; `cudaDeviceSynchronize`; throw on cudaError / either flag — DISTINCT messages ("donation density overflow" vs "density/rho-cap overflow"), both → node_main catch → `Halt::Internal` (HALT symmetry, constraint 11).
10. decode owned lanes / `pe.raw += h_pe` / min_r2 min-merge — verbatim; `++donated_composes_`.

**Why the cap MUST move (К3, not optional):** donated persistent ρ is STRICTLY FULLER than window-recomputed ρ on the deep halo (PR-1 class-3) ⇒ the verbatim all-m cap would spuriously HALT on halo atoms. Trigger-set equivalence at pass granularity rests on D3 (ρ_a ≥ 0 load guards: full ≥ partial ⇒ any old-path partial-fire implies a donated owned-full fire in the same pass). `eval_F` on fuller deep-halo ρ is safe (same spline-clamp code CPU `eam_window_force_from_rho` pass-2 already runs); its `d_fp[deep]` output is never read (open cutoff excludes deep halo from owned sweeps). Read-halo lanes under the residence guard g equal window-recomputed ρ (L-CLOSE: their out-of-window edges contribute literal zero via open cutoff) — **the guard g is LOAD-BEARING for the donated≡recompute equivalence itself** (class-3) ⇒ gate B3 is mandatory.

---

## 5. KERNEL SPECS (NEW header `include/tdmd/cuda/zone_eam_donation.cuh`; its TU MUST link `tdmd_eam_cuda_flags` — `--fmad=false` is load-bearing for per-pair `rint ≡ std::rint ≡ cvt.rni`)

`zone_eam.cuh` / `zone_eam_cells.cuh` / `zone_cells.cuh` **BYTE-UNTOUCHED** in both PRs.

### 5.1 SELF batch — ZERO new kernels (PR-3a)
`eam_density_cells_kernel` (culled) / `eam_density_kernel` (plain) **verbatim** over the slab (m = s, positions = lane arrays, output = ρ lane, overflow = `d_of_dn`). Bitwise vs CPU `eam_donate_self` (one eval, add v to both ends): W-1 — dx negation exact IEEE, min-image round odd, operand order identical ⇒ r² bitwise-equal from either direction ⇒ each gather-side v equals the CPU's single v ⇒ identical quanta multiset per lane ⇒ B1 int64 associativity makes the lane sum equal for ANY enumeration/culling order. Acceptance predicate = the SAME `geom.reduce` (HOST_DEVICE single source, incl. the 1e-18 lower bound); the kernel is the SAME code already proven bitwise == CPU pass-1 per window (E5/E5c) — only the population differs.

### 5.2 CROSS batch — the ONE new kernel pair (PR-3a)
```cpp
// concat window [A|B] (na=|A|, m=na+nb); shared whole-box grid, population refreshed over the concat;
// one thread per concat atom; candidates restricted to the OTHER side; gather += into the OWN zone lane.
// NO min_r2, NO keys, NO PE pointers in the signature — donations move ρ ONLY (WContract §6; SPEC eam_zone.hpp:116).
__global__ void eam_donate_cross_cells_kernel(
    const double* cx, const double* cy, const double* cz, int na, int m,
    core::PairGeom geom, EamSetflView eam, double dens_scale,
    CellGrid g, const int* starts, const int* counts, const int* order,
    long long* rho_a, long long* rho_b, int* overflow);
__global__ void eam_donate_cross_kernel(/* same minus grid args; inner loop over all m — the cull=false leg */);
```
Body = `eam_density_cells_kernel`'s stencil walk **verbatim** with exactly two deltas: (i) same-side skip `if ((bb<na) == (aa<na)) continue;` replacing the `bb==aa` self-skip (zones are disjoint ⇒ no self pair possible); (ii) epilogue `if (aa<na) rho_a[aa] += q; else rho_b[aa-na] += q;` — read-modify-write is race-free: each lane element is owned by exactly ONE thread per launch, and launches are null-stream-ordered after the SELF kernel. Bitwise: same W-1 negation argument per direction; culling restricts CANDIDATES only over a superset stencil (L-SUP) + exact `geom.reduce` retest ⇒ accepted-pair multiset == CPU `eam_donate_cross`'s ⇒ lane sums raw-equal by B1.

### 5.3 PR-3b compose kernels (both ~8 lines, same header)
```cpp
__global__ void eam_embedding_nocap_kernel(int m, EamSetflView eam, double dens_scale,
                                           const long long* d_rho, double* d_fp);
// == eam_embedding_kernel minus the atomicOr(rho>cap, 2) line; arithmetic identical.
__global__ void eam_owned_cap_kernel(const int* owned, int n_owned, double dens_scale,
                                     double rho_cap, const long long* d_rho, int* overflow);
// К3: cap on OWNED-full ρ only; FP compare, no contraction sites — bitwise-inert for non-halting runs.
```
There is deliberately NO in-kernel view-concat: the D2D scatter materializes ρ into the existing contiguous `d_rho` window scratch in exactly the host gather's block/member order, so the frozen embedding/force kernels index `d_rho[window_local]` unchanged. This keeps the force path byte-frozen and the bitwise proof one-hop.

---

## 6. BITWISE PROOF SKETCHES

### PR-3a
- **By construction (default)**: knob-off ⇒ hooks return before any device call; compose byte-identical ⇒ trajectory/PE == HEAD bit-for-bit. Witness: existing 14 `Test_CUDA_EAM_Ring` + 21 CPU tests unchanged-green + wall check A8.
- **By construction (knob-ON, observational)**: donation kernels write ONLY NodeStore lanes + `d_of_dn`; compose ignores both ⇒ trajectory bitwise == knob-off (gate A0).
- **By gate (device ρ)**: device lanes raw-int64 == CPU donated `dstate.rho[label]` — same post-drift double inputs (slab H2D = the very bytes the finalize gather reads; drift idempotent, once per pass; positions immutable after drift), same `geom.reduce`/`eval_rhoa` under `--fmad=false` (rint ≡ std::rint, proven E5), same quanta by W-1 negation, B1 order-freedom over identical accepted-pair multisets (L-SUP superset culling + exact retest). Comparison domain: per-(pass, window) lane snapshots at compose, like-for-like CPU-donated vs GPU-donated — both hold the SAME executed-batch subset at that event ⇒ even legitimately-partial halo lanes compare EXACTLY EQUAL. This is NOT the forbidden full-ρ-vs-window memcmp (WContract §8) — both sides are donated states, not window recomputes.

### PR-3b (chain; each link frozen or gated)
1. device ρ lanes == CPU donated ρ raw (PR-3a gates) ⇒
2. `d_rho` after D2D scatter == host-gathered `rho_w` raws (same lanes, same [pred][center][succ] member order — order-only, no arithmetic) ⇒
3. `embedding_nocap(d_rho)` == CPU pass-2 `eval_F` all-m (identical arithmetic, identical inputs); cap trigger == CPU К3 owned-full check (same D3-backed trigger set, owner attribution via zone_j — same text the CPU throw names) ⇒
4. force kernel BYTE-VERBATIM on identical (positions, keys, d_rho, d_fp) ⇒ owned force raws / PE / min_r2 bitwise == CPU `eam_window_force_from_rho` ⇒
5. GPU-donated ring trajectory bitwise == CPU donated ring == (PR-2 L1) serial `zone_eam_pass` — **L1 transitivity preserved** (hard constraint 5).
Additionally donated-GPU == recompute-GPU bitwise on non-capping fixtures (gate B1): owned + read-halo ρ identical under the residence guard g (L-CLOSE); deep-halo divergence downstream-invisible (unused `d_fp`, К3, PE owned-only). Beyond g the ring HALTs StaleZone before any output is claimed — gate B3.
min_r2: donated and recompute legs run the SAME force kernel over the SAME window ⇒ min_r2 bitwise-equal between legs; across DIFFERENT candidate sets (CPU brute vs GPU culled) only the Overlap-HALT boolean is ever gated (standing rule).

---

## 7. R_W BENCH PROTOCOL (frozen decision procedure; pre-registration honored)

Harness (`tools/bench_eam_ring.cu`) additions: `--axisW`, `--nzones Z` (default 5), `--donate {on|off}` (ctor knob), `--rect nx,ny,nz` / `--apz A` (atoms-per-zone via existing `make_fcc_rect` growth), `--reps` (R_W runs use 8), `--cellk` **pinned EXPLICITLY in BOTH legs** (never per-leg AUTO — AUTO m_hint differs between donated/recompute first-calls). Estimator UNCHANGED: SI2 (min per leg over reps, THEN ratio), warmup-difference t_seg, fresh ring + fresh policy per run, `cudaFree(0)` spin-up.

**Frozen grid**: Al_zhou setfl, `--cellk 3`, fixed dt, S=20 steps, W=4 warmup, **reps=8**, idle GPU (single process, nvidia-smi-verified — Me5b contention lesson: parallel agents halve throughput):
- **Primary (gated): pbc-z, n_zones=5, z=1** — full ×3 redundancy every zone; seam + slot-rotation exercised. atoms/zone ∈ {**~2k, ~8k, ~21k, ~42k**} ⇒ N ∈ {~10k, ~40k, ~105k, ~210k}. Includes the pre-registered small-N underload points AND the saturating point ≥ ~21k atoms/zone.
- Corroboration (reported, ungated): free-z n_zones=4, z=1 @ ~21k/zone; Axis B rerun (z=5) sanity.
- Axis A (n_zones=1) is **NULL BY CONSTRUCTION** (density computed once per pass — nothing to remove) and is NOT an R_W leg (Reader-3 trap, honored).
- Legs: `donate=on` vs `donate=off`, BOTH `cull=true`, same cellk. R_W(point) = atom_steps/s(donated) / atom_steps/s(recompute).
- **Attribution requirement (constraint-4 evidence)**: TDMD_EAM_RING_TIMERS split must show the donated leg's compose `timer_kernel_s` drop ≈ the wall drop, h2d/rest ≈ flat — work REMOVAL, not overlap. Donation host-wall reported separately; no new syncs added for timing.

**GO/NULL rule (frozen BEFORE measurement)**: GO ⟺ R_W ≥ **1.15** at EVERY primary-grid point with ≥ 21k atoms/zone AND R_W ≥ **0.97** at every smaller point (no material small-N pessimization beyond ±3% noise). Otherwise NULL.

**NULL ⇒ rollback mechanics (J8)**: revert commit c2 ONLY (default flip). Production returns to the recompute leg (today's proven path); the donated path stays knob-gated + fully test-covered (A/B-gates keep instantiating it — not structurally dead); PR-3a substrate stays (value to PR-4/M5b independent); NULL numbers recorded in `TD_MD_Core_Bench_v1_0.md` + CLAUDE.md. Re-open trigger: M5b (≥2 GPU — donation as useful work while waiting on upstream, WContract §9 ARRIVAL rationale, preserved because donation execution stays a standalone scan step, never re-anchored inside finalize).

**PR-3a wall gate (±3%)**: same harness, HEAD binary vs PR-3a binary (default knob-off), configs = Axis A sizes {864..16384} + primary grid + Axis B; |Δ| ≤ 3% per config, SI2, reps=8, idle. Expected ~0 (knob-off does zero device work in hooks). Recorded in Bench §PR-3a; failure ⇒ investigate before landing, never silent.

---

## 8. GATES + KILL-MUTATIONS (every guard has a named kill; poisons are policy-level test-only fields, NEVER kernel defaults — P-k precedent; every donated gate carries a non-vacuity `EXPECT_GT` on the counters)

New CUDA target `tests/test_cuda_eam_donation.cu` (links `tdmd_core`, `GTest::gtest_main`, `tdmd_eam_cuda_flags`, `Threads::Threads`; LABELS cuda; matched by gpu_gate glob `test_cuda_*`). Test-only policy surface: `GpuDonationPoison { int stencil_override=-1; bool one_sided=false; bool defer_seam_to_next_compose=false; }` + `capture_lanes` log (J11) + counter accessors.

### 8.1 PR-3a gates

| # | gate | fixture | expectation | kill-mutation |
|---|---|---|---|---|
| A0 | KnobOnIsObservational | free n=4 z∈{1,2,3}; pbc n=5 z∈{2,3}; 12 steps | knob-ON trajectory + per-pass PE bitwise == knob-OFF; `EXPECT_GT(self+cross counters, 0)` | make a hook write into `st->d_rho` window scratch → red |
| A1 | DeviceRhoBitwiseVsCpu (T-1 analog) | same grid + free n∈{1,2}; capture-log per-(pass, window) lane snapshots at compose on BOTH rings (CPU wrapper policy logs `dstate.rho[label]`; GPU logs D2H lanes); all ≤3 window lanes compared | raw int64 `EXPECT_EQ` per lane element | (i) `stencil_override=1` → red; (ii) `one_sided` → red; (iii) index lanes by SLOT instead of label → red ONLY in the pbc n=5 ≥2-rotation fixture (G5/G-ROT class) |
| A2 | SeamNonVacuity (T-2 analog) | pbc n=5, ≥12 steps | `batches_cross == passes·n` (interior-only would give passes·(n−1) — the +1/pass IS the seam, rotation-proof); zone n−1/0 lanes bitwise (subsumed by A1 pbc) | skip the j=n−1 cross launch → count arithmetic red + A1 red on labels n−1,0 |
| A3 | Fb40DualFormat (T-12 analog) | β=3.3 setfl (asserts `density_fracbits()==40`) through A1 | lanes bitwise | hardwire kScale 44 in hooks → `DA::kScale` runtime assert throws / A1 red |
| A4 | VacuumEmptyZones | VacuumGap fixture (zones 1..4 empty), knob-ON | no launches for n==0; stamp still set; trajectory == knob-off; ledger closes ring-side (WContract §5) | remove the n==0 early-return → memcheck red / launch-failure; remove stamp-set on empty zone → A7 red |
| A5 | DriftedDonorAcrossZoneBoundary (Tier-0 G-B) | donor drifts +0.6·g across its zone boundary over ≥2 passes (G8-style mild-vz fixture) | lanes bitwise == CPU (whole-box binning bins at TRUE cell) | swap the self-batch grid to the slab-extent `make_zone_grid(n_zones>1)` clamp path → red |
| A6 | OverflowSticky | steep analytic setfl (rho_amp sized so one quantum overflows int64) | `d_of_dn` accessor nonzero after the pass | remove the sticky write → accessor zero (and 3b's B6 loses its HALT) |
| A7 | StaleStampThrows (B-import) | unit: `on_edge`/donated-compose with a label not stamped this pass → `logic_error` → `Halt::Internal` | direct | remove the stamp check → red |
| A8 | WallWithin3pct | §7 protocol, HEAD vs PR, default build, idle, SI2 | \|Δ\| ≤ 3% per config (expected ~0); recorded in Bench, not ctest | — (measurement gate) |
| A9 | Suites + surface | clean rebuild (Me2 stale-binary lesson): CPU 45/45 bitwise (concept-v2 shims F-NOOP); existing 14 GPU-ring gates unchanged; `cuda_compile_check.cu` += include `zone_eam_donation.cuh` + `static_assert(DonatingWindowForcePolicy<GpuEamWindowForce>)` (recompiles v2) + member-scoped `run()` re-instantiation (recompiles plumbing; whole-class instantiation ill-formed); gpu_gate memcheck+racecheck (gates ≤20 steps ⇒ no RACECHECK_FILTER additions); ONE manual initcheck run recorded (seam stays manual, J10) | | any CPU byte-drift / missing tdmd_eam_cuda_flags link fails |

### 8.2 PR-3b gates

| # | gate | fixture | expectation | kill-mutation |
|---|---|---|---|---|
| B0 | L1 pack (zero new code) | ALL 14 existing `Test_CUDA_EAM_Ring` gates run THROUGH the donated default: bitwise vs CPU ring free+pbc+per-pass PE, OneVsZ, Q2340 (fb=40 live), Culled≡AllWindow (both legs donated ⇒ donation-cull bitwise gated), anti-deadlock, firewall, **FP64 oracle D free+pbc = the dropped-donor witness THROUGH donated+culled** (constraint 9; 1-vs-z and run-to-run are blind) | + 2-line `EXPECT_GT` donation counters in the culled gate | B1's poisons make oracle D red (non-vacuity of the oracle through the new path) |
| B1 | DualPathGpu (G1-analog) | donate=true vs donate=false rings; free n=4 z∈{1,3}; pbc n=5 z∈{2,3}; fixed+auto dt | trajectory + per-pass PE bitwise; `EXPECT_GT` counters | `one_sided` → bitwise red + FP64 red >1e-6 while ledger stays CLEAN (T-7 honest-boundary analog: ledger blind to intra-batch loss) |
| B2 | SeamBeforeFinalize (T-6 analog) | pbc n=5; poison `defer_seam_to_next_compose` buffers the seam launch past finalize(n−1) | owned(n−1) trajectory diverges BITWISE vs correct leg AND ledger stays clean — pins the in-scan seam deadline as load-bearing physics ON DEVICE | the poison IS the mutation; gate asserts divergence |
| B3 | DriftBandHaltGpu (G8-analog; constraint 6 — g LOAD-BEARING, class-3) | G8 fixture verbatim (atom0 top of zone-0 slab, vz=1500, C_buf=30) on the donated GPU ring | `Halt::StaleZone`; acceptance replicates PR-2 M3's measured half once (guard bypassed in a scratch build → supra-tol FP64 divergence WELL beyond g, recorded in the test comment — band-edge fixtures are sub-tol and vacuous) | guard-off scratch run → different halt + measured divergence documented |
| B4 | RhoCapOwnedOnlyK3 (G7/T-10 analog) | loose cap (rho_max·1.5) → no throw + bitwise == recompute leg; tight (·0.5) → `Halt::Internal` w/ donation-path message; PLUS hand chain fixture with deep-halo-full > cap > owned-full | donated must NOT throw on the halo-fuller fixture | restore the all-m cap (use `eam_embedding_kernel`) → spurious halo throw → red |
| B5 | Fb40Donated | Q2340 (in B0) + A3 rerun donated | bitwise | as A3 |
| B6 | OverflowHalt | A6 fixture, donated default | `Halt::Internal`, message names DONATION overflow (distinct from density/cap) | drop the `d_of_dn` read in compose → no HALT → red |
| B7 | LedgerKnobGpu (G10) | `test_drop_first_self_ledger_` + GPU policy, z=1 | `Halt::StaleZone`, "donation ledger" in halt_msg (ring-side check, instantiation proof with the GPU policy) | remove the ring-side check in a scratch build → green run → gate red |
| B8 | RotationLabelKeyedRho (G5) | covered by B0/B1 pbc n=5 ≥12 steps (≥2 full rotations, slot≠label) + A1(iii) | bitwise across rotations | slot-keyed lanes → red only here |
| B9 | VacuumDonated | VacuumGap fixture, donated default | ring == serial bitwise; no launches; ledger closes | as A4 |
| B10 | HaltDiscardsPass | mid-pass HALT (overlap fixture) donated | run terminal, no partial replay (no API); a fresh ring is unaffected | — (by-construction + assert) |
| B11 | R_W bench gate | §7 protocol | GO/NULL per frozen rule | — |
| B12 | FreezeWitness | `git diff --stat` EMPTY on `eam_ring.hpp`, `eam_donation.hpp`, `eam_zone.hpp`, `zone_eam.cuh`, `zone_eam_cells.cuh`, `zone_cells.cuh`, all CPU code; CPU 45/45 | | any byte-drift fails |
| B13 | Sanitizers | gpu_gate memcheck+racecheck full (short-step gates — racecheck budget safe); manual initcheck over the donated path recorded | | — |

---

## 9. PR SPLIT + FILE INVENTORY + F-NOOP BYTE-FROZEN LISTS

### PR-3a — "device-donation substrate on per-zone slabs, default-inert (enabling)"
MODIFIED:
- `include/tdmd/potentials/eam_donation.hpp` — +`WindowBlocks` POD, concept v2 (NodeState / make_node_state / begin_pass / ns-first hooks / compose(ns,…,wb,…)). Executors, `eam_window_force_from_rho`, `EamDonationState`, `EamDonationPass`, `check_donation_residence`, serial drivers **byte-untouched within the file** (additive-only diff).
- `include/tdmd/potentials/eam_ring.hpp` — `node_stores_` member + creation in run(); begin_pass at pass head; ns at the 2 hook call sites; wb build + ns at the compose call; `CpuEamWindowForce` shim. Ledger/schedule/END/SEND/Λ/parity/defer_head logic untouched.
- `include/tdmd/cuda/eam_window_force_gpu.cuh` — `GpuEamDonationNodeStore`, `make_node_state`, real hooks behind `donate_device=false`, stamp/token, shared-state counters, capture knob, poison struct, `DA::kScale` assert.
- `tests/test_eam_ring.cpp` — mechanical shims for `CpuEamRecomputeWinForce`/`CpuEamPoisonWinForce` + the A1 CPU capture wrapper.
- `tools/cuda_compile_check.cu`, `CMakeLists.txt` (new target), `include/tdmd/cuda/zone_nbr_store.cuh` (**APPEND-ONLY** Tier-0-realized / Tier-1-DEFER annotation, §0 pt.5), docs (`TD_MD_Core_WContract_v1_0.md` §13 append pt.1: NodeStore ownership, stamp = PR-4 epoch seat, knob semantics; `TD_MD_Core_Bench_v1_0.md` §PR-3a wall table).
NEW: `include/tdmd/cuda/zone_eam_donation.cuh` (cross kernel pair); `tests/test_cuda_eam_donation.cu` (A-gates).

### PR-3b — "donated density on GPU + R_W gate" (two commits: c1 = code+gates default-off; c2 = default flip + numbers)
MODIFIED: `include/tdmd/cuda/eam_window_force_gpu.cuh` (donated compose branch; default flip in c2); `include/tdmd/cuda/zone_eam_donation.cuh` (+`eam_embedding_nocap_kernel`, `eam_owned_cap_kernel`); `tests/test_cuda_eam_donation.cu` (+B-gates); `tests/test_cuda_eam_ring.cu` (2-line non-vacuity counters in the culled gate); `tools/bench_eam_ring.cu` (Axis W + flags); `tools/cuda_compile_check.cu` if needed; docs (Bench §PR-3b + R_W verdict; WContract §13 pt.2 + §10 row "EAM Density GPU SHIPPED/NULL"; CLAUDE.md status).
**BYTE-FROZEN in 3b: `eam_ring.hpp` AND `eam_donation.hpp` entirely** — 3b is policy + kernels + bench + tests + docs only (B12 witnesses).

### F-NOOP BYTE-FROZEN in BOTH PRs (any diff = design violation; verified by empty `git diff` + suites)
`include/tdmd/potentials/eam_zone.hpp` (frozen oracle + `donation_layout` + `eam_window_layout` + `want_closure_mask`), `include/tdmd/cuda/zone_eam.cuh`, `include/tdmd/cuda/zone_eam_cells.cuh`, `include/tdmd/cuda/zone_cells.cuh`, `include/tdmd/cuda/eam_conveyor_gpu.cuh` (z=1 driver), `include/tdmd/potentials/eam.hpp`, `include/tdmd/potentials/many_body.hpp` (base concept), `core/fsm.hpp`, `core/conveyor.hpp`, `cuda/conveyor_gpu.cuh`, `core/skin_budget.hpp`, `core/rebuild_epoch.hpp`, transport/MPI edges, ALL sw/tersoff/meam headers + rings + GPU policies, `tests/test_eam_donation.cpp` (serial oracle gates), the serial `zone_eam_pass_donated` path. `zone_nbr_store.cuh` frozen EXCEPT the append-only annotation.

---

## 10. PR-4 / M5b FORWARD-COMPAT

- **PR-4**: `ZoneLane.stamp` (pass-token, LABEL-keyed) is the exact seat for `rebuild_epoch` — PR-4 replaces token-compare with `cuda::nbr_store::advance/stale` and makes population refresh lazy; Tier-2 per-(zone,role) Verlet-CSR slots land inside `GpuEamDonationNodeStore` beside the lanes, mem-gated per the frozen contract (`neighbor.mem_budget_gib` + VRAM probe). The per-batch grid-population refresh is exactly what epoch-gated reuse replaces. Nothing precludes epoch WRITE at first-send / READ at arrival-0 (ring-side, PR-4). All structures zone_id-keyed (WContract §2 names "device mirrors" in the label-keying clause — honored).
- **M5b**: the per-zone device slab (+ρ lane) is the natural D2D transport payload unit; the NodeStore boundary is the cross-GPU edge seam. Donation execution remains a standalone scan step fired from `donate_position` (ARRIVAL-variant §9(iv-v) freedom preserved — never re-anchored inside finalize) ⇒ the M5b try_recv useful-work scheduler stays reachable. Window-position D2D-concat (the deferred Tier-1 consumption) is a ~10-line `compose()` addition once slabs + `WindowBlocks` labels exist (both land in 3a) and M5b measures a consumer.

---

## 11. HARD-CONSTRAINT COMPLIANCE MAP
1. No ρ in ZoneMsg — donation state per-node device; header/msg untouched ✓. 2. END/SEND, batched END-sync, positional Λ-chain untouched — only hook-call arguments grow; no sends moved; no blocking recv/send added ✓. 3. No global reductions ✓. 4. Work REMOVAL, mutex + null-stream kept, E5c-concurrency explicitly disavowed (§2 discipline; TIMERS attribution in §7) ✓. 5. Bitwise INV-9 — §6 chain, raw-int64 gates, L1 transitivity preserved ✓. 6. g load-bearing (class-3) — B3 mandatory, `g_override` never plumbed ✓. 7. fb from runtime `density_fracbits()` only — raw int64 lanes + `DA::kScale` assert + A3/B5 ✓. 8. Kill-mutations named per gate (§8) ✓. 9. FP64 oracle non-vacuous THROUGH donated+culled (B0 oracle D + B1/B2 poisons; ledger honestly blind to intra-batch loss) ✓. 10. Ledger ring-side, hooks ρ-only; empty zones close vacuously ✓. 11. К3 cap on owned-full ρ at finalize + HALT symmetry (device flag → host throw → `Halt::Internal`, owner attribution) ✓. 12. pbc n=1 slot handling + defer_head ring-side; hooks agnostic; NO dedup added to device paths (pbc n=1 live-ring domain NOT extended) ✓.

---

## 12. RISKS (honestly costed)
- **R1 AUTO cell_div resolution point** can shift when the first culled device op is a donation batch: hooks pass hint 3·blk.n; benches pin `--cellk` both legs; residual = different-but-bitwise k, wall-visible only, covered by A8/§7.
- **R2 Small-N NULL**: donation kernels launch at zone granularity s (not 3s) + ~2n extra launches + ~3n grid refreshes/pass — pre-registered underload risk; GO rule + two-commit rollback bound the damage; realized saving < 1.36 ceiling (per-cell-visit overhead does not shrink; cross saved only ×2→×1).
- **R3 Concept-v2 churn** is wide but mechanical; a silent drift is caught only by CPU byte-gates ⇒ acceptance MUST run CLEAN rebuilds (Me2 stale-binary precedent) before declaring F-NOOP.
- **R4 Capture seam (A1/D2)** could false-red or vacuously compare zeros if mis-anchored ⇒ the `EXPECT_GT` counters + the three A1 kill-mutations are mandatory teeth.
- **R5 `d_of_dn` sticky-without-per-pass-reset** relies on HALT being terminal for run(); any future in-flight rescue/pass-retry MUST add a pass-boundary reset — banner comment in the NodeStore.
- **R6 Deep-halo `eval_F` on donated-fuller ρ** relies on spline clamping (same code CPU pass-2 runs); pathological F-spline tails would surface here first — D3 load guards + B4 loose/tight fixtures are the witnesses.
- **R7 Shared grid arrays + concat scratch** reused across hooks and compose under the mutex on the null stream; correctness = per-op self-containment (grow → fill → launch under one lock); a future edit that splits streams without events reintroduces the E5c hazard — pinned by comments + racecheck.
- **R8 The 14 existing GPU-ring gates silently become donated-path gates at the c2 flip** — any latent recompute-only assumption (none found; all are bitwise/oracle gates) surfaces as a red only at 3b; B1 dual-path isolates donation-side vs consumption-side.
- **R9 Memory** +1.4 GiB @1e7/z=4 fits 15.5 GiB but scales linearly in z (slabs dominate); watch VRAM on z-sweeps; M5b repartitions per-device N.
- **R10 pbc n=1** stays out of the live-ring domain (triple-window Overlap HALT; ring-side no-dedup) — the device D2D faithfully replicates the host triple-gather; domain NOT extended.

---

# ВЕРИФИКАЦИЯ ДИЗАЙНА — ВШИТЫЕ ПОПРАВКИ (4 линзы, все ACCEPT-WITH-FIXES; workflow wf_e3923bb3, 11 агентов)

Линзы: bitwise-b1 / vacuity / feasibility / fnoop-scope. Ниже — обязательные поправки,
имеющие СИЛУ ДИЗАЙНА (при конфликте с телом выше действует эта секция).

## MUST-FIX (M1–M9)

- **M1 (A8 неисполним в 3a как заморожено).** РЕШЕНИЕ: донационно-агностичные геометрические
  флаги харнесса `--nzones Z` / `--apz A` / `--rect nx,ny,nz` / `--reps` переезжают в **PR-3a**
  (правка `tools/bench_eam_ring.cu` входит в 3a-инвентарь); `--donate {on|off}` и `--axisW`
  остаются в 3b. §9-инвентарь и фриз-формулировка «3b = policy+kernels+bench+tests+docs»
  скорректированы: bench-харнесс правится в ОБОИХ PR (3a: геометрия; 3b: axis-W/donate).
- **M2 (форма бокса R_W не запинена — post-hoc степень свободы).** ЗАМОРОЖЕНО ДО ЗАМЕРА:
  primary-грид использует `make_fcc_rect` **nz=26** (ширина зоны 21.06 Å при n_zones=5,
  g=0.43 Å — достаточно для W+S=24 шагов при 300 K), точные триплеты:
  `--rect 12,8,26` → apz≈2.0k; `--rect 20,19,26` → apz≈7.9k; `--rect 32,32,26` → apz≈21.3k;
  `--rect 45,45,26` → apz≈42.1k. Пин `dt=5e-4`, maxwell-seed = дефолт харнесса, ЯВНО
  зафиксированный в выводе бенча. Отклонение от триплетов = нарушение пре-регистрации.
- **M3 (таймер-атрибуция ложно-фейлит по построению).** Донационные ядра запускаются из хуков
  БЕЗ sync ⇒ их GPU-время видит первый блокирующий H2D следующего compose (null-stream) и
  попадает в бакет h2d. ФИКС: compile-gated (TDMD_EAM_RING_TIMERS) event-пара вокруг
  донационных запусков в хуках, аккумулятор `t_donation_kernel_ms` в shared-state + акцессор.
  §7-ожидание ПЕРЕФОРМУЛИРОВАНО: (compose-ядра + донационные ядра) суммарное device-время
  донационной ноги падает ≈ падению wall; H2D-байты неизменны; h2d-бакет donated-ноги
  ЛЕГИТИМНО содержит донационное kernel-время — не «флэт» и НЕ признак overlap.
- **M4 (kill A5 структурно мёртв — clamp не срабатывает на легальных дрейфах).** Kill
  ре-анкорен: test-only poison `drop_outside_nominal_slab` (поле GpuDonationPoison) — при
  slab-upload дропает атомы, чья z вне номинального [zone_lo, zone_hi] (имитация РЕАЛЬНОЙ
  опасности mis-binning slab-CSR-формы). На дрейф-фикстуре A5: лейны EXPECT_EQ красные +
  FP64-оракул красный. Красноту ПРОВЕРИТЬ до заморозки (дисциплина Me2). Slab-clamp-мутация
  из тела ВЫЧЕРКНУТА как вакуумная.
- **M5 (A1 stencil-poison вакуумен при k=1).** На poison-ноге A1 пин `cell_div=3` и rcut=3
  (~1 Å ячейки ⇒ override-к-1 покрывает 2 Å < nn 2.86 Å — генуинно красный); в гейт-таблице
  записано «override-poison бессмыслен при k=1» (AUTO на малых фикстурах даёт k=1).
- **M6 (B4 halo-fuller противоречив на уровне кольца).** Эквивалентность триггер-сетов K3 —
  ровно причина, почему ring-фикстура «deep-halo>cap>owned» невозможна. РЕСПЕК: halo-fuller
  суб-гейт = ПРЯМОЙ policy-level unit-тест compose (рукостроенные NodeState-лейны +
  WindowBlocks: pred-лейн > cap, center-owned ≤ cap ⇒ donated compose no-throw; мутация
  «вернуть all-m cap» (eam_embedding_kernel) ⇒ throw ⇒ красный). Ring-level дополнение:
  tight-cap фикстура ассертит owner-атрибуцию в halt_msg (donation-path message).
- **M7 (kill A0 мёртв — запись хука в per-call scratch перезаписывается по построению).**
  Kill ре-анкорен: мутация «хук возмущает ПЕРСИСТЕНТНОЕ compose-читаемое состояние»
  (st->dens_scale на 1 ulp) ⇒ knob-ON траектория красная vs knob-OFF. Красноту проверить.
  A0 переклассифицирован: observational-by-construction для per-call scratch (записано),
  зуб живёт на персистентном состоянии.
- **M8 (незаспецифицированный grow разделяемых grid-массивов — класс Me5b silent-OOB).**
  Хуки ОБЯЗАНЫ `s.grow(needed)` (slab: n; cross: na+nb) ДО population-refresh — первая
  донация прохода идёт ДО первого compose при initial cap=64. Санитайзерная фикстура,
  несущая этот груз, ЗАКРЕПЛЕНА: первый донационный батч > 64 атомов до любого compose
  (free n=1: self-батч 96 атомов; и/или pbc-wide 288: concat ~115) — прогоняется под
  memcheck в A9.
- **M9 (use-after-free при grow: поднято из SHOULD — корректность).** Сегодня compute()
  полностью синкает до отпускания mutex; хуки 3a отпускают mutex с ЗАКВЬЮЕННЫМИ ядрами,
  ссылающимися на grid_.d_cell_of/d_order (и лейны). Последующий s.grow(m) делает cudaFree
  этих буферов ⇒ UAF. ФИКС: `grow()` при cap_m>0 выполняет `cudaDeviceSynchronize()` ДО
  free_scratch (grow-only, редкое событие — цена ноль); правило «каждый device-оп
  самодостаточен под одним lock: grow → fill → launch» дополнено «grow синкает перед free».

## Принятые SHOULD (S1–S13)

- **S1**: B0-инвентарь исправлен: через donated-путь идут **10 из 14** существующих гейтов
  (4 SingleNode*-гейта гоняют байт-замороженный z=1 драйвер eam_gpu_run_singlenode и
  donated-путь не трогают — честно записано).
- **S2**: A1 CPU-capture-обёртка определяется в НОВОМ CUDA-тест-TU (тест-политики TU-локальны;
  общего тест-хедера нет и не вводится).
- **S3**: append-нота zone_nbr_store.cuh обязана ПРОЦИТИРОВАТЬ и явно супersede оба
  устаревших упоминания «CONSUMER: PR-3a» (строки ~44-45 и DEFERRED-блок) — иначе файл
  внутренне противоречив.
- **S4**: семантика счётчиков ЗАПИНЕНА: donation_self/cross_batches_ инкрементятся для
  каждого ИСПОЛНЕННОГО (в т.ч. вакуумного n==0) батча ДО early-return; A4 проверяет
  отсутствие KERNEL-запусков при n==0, не отсутствие инкремента; арифметика A2
  (batches_cross == passes·n) держится и с пустыми зонами.
- **S5**: A6 толерантен к HALT-нувшему прогону (читать акцессор d_of_dn ПОСЛЕ возможно
  упавшего run(); knob-ON compose на крутом setfl тоже переполняется в recompute-ядре);
  rho_amp фикстуры размерять против фактического fb ПОСЛЕ auto-fallback крутых таблиц к
  Q23.40 (fixed_accum.hpp:57-62).
- **S6**: B10 переклассифицирован из гейта в задокументированное by-construction
  НЕ-гейт-свойство (replay-API не существует; halt терминален) — без строки в гейт-таблице,
  с явной записью, чтобы не выглядел зубом без kill.
- **S7**: WindowBlocks.center получает ПОТРЕБЛЯЮЩИЙ assert в donated compose:
  `wb.center>=0 && wb.label[wb.center]==zone_j` ⇒ throw при рассинхроне (kill: ring-side
  scratch-мутация построения wb) — поле не инертно (прецедент T-SCAFFOLD).
- **S8**: бюджет памяти в GiB: NEW total 1.40 GB = **1.30 GiB**; гранд-тотал ≈ **1.51 GiB**
  (направление консервативное, вывод «влезает в 15.5 GiB» неизменен).
- **S9**: B2-poison анкорен ТОЧНО: отложенный seam-запуск стреляет при входе в ХВОСТОВОЙ
  finalize(0)-compose (ПОСЛЕ чтения лейнов finalize(n−1)) — анкоровка на входе finalize(n−1)
  завершает лейны вовремя и дивергенции не даёт.
- **S10**: kill A1(iii) (slot-vs-label) реализуем ТОЛЬКО на ring-стороне (хуки слотов не
  видят): именованный сайт мутации = построение wb.label в finalize_owned (scratch-build);
  красный ТОЛЬКО на pbc n=5 ≥2-ротации ноге — если pbc-ногу когда-либо срежут по бюджету,
  ротационный зуб умирает молча (класс G5-рецидива; записано).
- **S11**: enumeration-overhead cross-ядра (полный (2·s_d+1)³-стенсил на каждый concat-тред
  даже при нуле cross-пар на широких зонах) записан как ПЕРВЫЙ ПОДОЗРЕВАЕМЫЙ при NULL R_W.
- **S12**: защитный memsetAsync лейна на arrival аннотирован «defensive, NOT load-bearing;
  reset-механизм = overwrite SELF-ядра; staleness-страж = stamp-fence» (иначе будущий
  ревьюер примет memset за механизм сброса).
- **S13**: CLAUDE.md добавлен в doc-инвентарь PR-3a (хаус-конвенция летописи на каждый PR).

## Резолюции открытых вопросов судьи (OQ1–OQ7)

- **OQ1**: primary-гейт остаётся pbc-only (free-z n=4 — corroboration, ungated): промоция
  free-z в гейт ПОСЛЕ заморозки подняла бы планку post-hoc; pbc — честная максимальная
  ×3-редундантность заявки. Free-z числа публикуются рядом.
- **OQ2**: capture-seam — mutation-proof обязателен в приёмке (три kill A1 — обязательные
  зубы; вакуумное сравнение нулей исключается EXPECT_GT-счётчиками).
- **OQ3**: AUTO-hint 3·blk.n заморожен; при флаге ±3% на конфиге — пин cell_div явно, без
  пере-деривации эвристики.
- **OQ4/J10**: owned-range D2H (initcheck-шов) — отдельный микро-PR со своим initcheck-свидетельством.
- **OQ5**: d_of_dn sticky-без-reset — баннер в NodeStore обязателен (условие пересмотра:
  in-flight rescue M7-бэклога).
- **OQ6**: приёмка подтверждает git-байт-целостность base WindowForcePolicy + siblings в
  ОБОИХ PR (B12/A9).
- **OQ7**: арифметический seam-свидетель (passes·n vs passes·(n−1)) ПРИНЯТ как T-2-аналог
  (ротационно-устойчив; label-keyed seam-счётчик не требуется — слоты хукам недоступны
  by design).

---

## РЕАЛИЗАЦИЯ PR-3a + ПРИЁМКА `wf_1c8cf3af` — APPEND-ONLY REALIZATION NOTES (2026-07-09)

Замороженный текст выше НЕ правится; ниже — отклонения реализации, найденные и
санкционированные приёмкой (4 линзы: bitwise ACCEPT-WITH-FIXES / teeth ACCEPT /
fnoop ACCEPT / spec ACCEPT-WITH-FIXES; все MUST-FIX вшиты до коммита).

1. **A6 ДВАЖДЫ respec-нут (замер реализации, супersede-ит строку §8.1):** дизайн-фикстура
   «steep analytic setfl, rho_amp sized so one quantum overflows int64» НЕДОСТИЖИМА —
   такой setfl вообще не загружается: страж `density_fracbits()` (rhoa_floor·kMaxCoord·kSafety
   < 2^23, иначе throw «density bound exceeds even Q23.40») делает single-quantum overflow
   структурно недостижимым для ЛЮБОГО загружаемого setfl ⇒ бит-1 sticky-флаг — defense-in-depth.
   A6 переспецифицирован в plumbing-зуб: тест-only порча сплайна rhoaspl ×1e7 ПОСЛЕ легальной
   загрузки (fb/dens_scale остаются консистентными — density_fracbits() ключуется от
   сохранённого rhoa_floor). Негативная полоса (born-set флаг) добавлена приёмкой в A0:
   `EXPECT_EQ(donation_overflow(), 0)` после легального knob-ON прогона.
2. **A1-ноги (MUST-FIX приёмки закрыл сужение И дыру):** реализация изначально гоняла
   {free n=4 z=1; pbc n=5 z=1; pbc n=5 z=3} — УЖЕ §8.1 («same grid + free n∈{1,2}») и,
   главное, ВСЕ фикстуры были cull=true ⇒ plain-ядро `eam_donate_cross_kernel` шло без
   единого свидетеля (ревьюер ВЫПОТРОШИЛ его тело — 8/8 зелёные; 7-й рецидив
   structurally-dead). Пост-фикс: + {free n=4 z=1 cull=false; pbc n=5 z=1 cull=false;
   free n=1; free n=2}; kill re-verified (гат plain-ядра ⇒ A1 красный).
3. **M4-глагол:** «slab-clamp-мутация измерена мёртвой» исправлено на «вычеркнута
   СТРУКТУРНЫМ аргументом верификатора» (WContract §13.4 + баннер A5) — замера не было.
4. **A8 primary-grid:** HEAD-бинарь не умеет Axis G (гео-флаги — M1, приземлены в 3a) ⇒
   HEAD-нога исполнена HEAD-worktree-бинарём с PR-харнессом на apz∈{1997,7904} back-to-back;
   apz∈{21299,42120} исключены ИЗМЕРЕННО (t0-стенка ~25 мин/~2.5 ч на ногу + межбутовая
   вариативность сегмента ±25% ⇒ 3%-гейт не well-posed кросс-процессно) — числа и
   обоснование в Bench §PR-3a.
5. **J10 initcheck:** один записанный прогон нашёл РОВНО один новый класс — RMW `+=`
   cross-ядра читает неинициализированный discard-sink one_sided-пуазона (тест-only путь);
   зачинен `cudaMemset` в `discard_lane()`; остаток — унаследованный uninitialized-D2H
   класс recompute-`compute()` (задокументирован ещё SW T5, PR-3a его не расширяет).
6. **Capture-fence (гигиена приёмки):** stamp/n-фенс + проверка кода возврата cudaMemcpy
   в capture-блоке compose (slot-vs-label умирает чистым logic_error, не Causality-каскадом).
7. **Axis G cellk-пин:** харнесс дефолтит cellk=3 на Axis G и громко warn-ит при явном
   отклонении (протокол исполнялся только дисциплиной оператора).
