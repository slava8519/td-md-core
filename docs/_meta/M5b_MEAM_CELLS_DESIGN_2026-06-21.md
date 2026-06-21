I now have the decisive code-grounded facts that resolve the disagreement among the four lenses. The key finding: **the canonical sort already runs on the host (`std::sort` by global key, line 198), and the all-window kernels then scan the window in sorted slot order.** A cell-walk visits `k` in `b_order[]` (cell-scatter) order, which is NOT sorted-slot order — and `sij` is an FP product (line 608) + `dscrfcn` an FP sum (line 643). So cells-vs-all-window bitwise on the FP screening requires the cell traversal to reproduce sorted-slot order. The shipped `cell_scatter_kernel` (line 158) uses `atomicAdd` cursors → nondeterministic intra-cell order. This is the single biggest correctness risk, and the lenses split on how to resolve it. Let me check the partial-screen-slab generator and confirm the screening k-loop centers on `c` (it does — lines 586, 592 use `xc=wx[c]`).

I have everything. Here is the implementation plan.

---

# Me5b — Cells-Culling for MEAM-GPU: Implementation Plan (Design of Record)

## 0. The one fact that drives the whole design (code-grounded, resolves the lens disagreement)

The MEAM screening reach is **sub-cell**, so one shared `rc`-padded whole-window grid culls density, force, AND screening-k — but the screening accumulators are **FP-order-sensitive and NOT protected by B1**, and the all-window path is **already key-sorted**. These two facts together decide everything.

**Fact A — reach is sub-cell.** `meam_getscreen_d_device` (meam.hpp:584,596-597) bounds every screening-k by `rik2 > rbound` and `rjk2 > rbound`, `rbound = ebound·rij²`. With Si `Cmax=2.8 ⇒ ebound = Cmax²/(4(Cmax−1)) ≈ 1.089`. The k-loop's own cut is `geom.reduce ⇒ rik2 < rc²` (meam.hpp:593), centered on **`c` = `wx[c]`** (meam.hpp:586,592), so every screening-k is an in-`rc` neighbour of the bond center. `MeamParams::recompute` *asserts* `√ebound·rc < 2·rc`. The `±s_d` stencil (`s_d = ceil((rc+pad)/c)`, zone_cells.cuh:95-99) spans `≥rc` per direction ⇒ the cell-neighbourhood of `c` is a **superset of every screening-k**. **No second, wider grid for screening; one grid covers density donors + force neighbours + screening-k.**

**Fact B — the screening accumulators are NOT B1-protected, and the all-window path is already sorted.** `sij = Π_k fcut(cikj)` is an **FP64 running product** (meam.hpp:608); `dscrfcn = Σ_k coef1·dCikj` is an **FP64 running sum** (meam.hpp:643). Both reassociate under k-reordering. The window arrives at the kernels **already `std::sort`-ed by global key on the host** (meam_window_force_gpu.cuh:198), so the all-window k-loop walks k in sorted-slot order. A cell-walk visits k in `b_order[]` (cell-scatter) order. **These differ ⇒ `sij`/`dscrfcn` reassociate ⇒ cells is NOT bitwise to all-window** — *unless the cell build reproduces sorted-slot order*. And the shipped `cell_scatter_kernel` (zone_cells.cuh:154-158) uses `atomicAdd` cursors → **nondeterministic intra-cell order** (explicitly blessed for EAM because EAM density is int64-order-free, zone_cells.cuh:10-13). **For MEAM this is the load-bearing escalation, not a footnote.**

This is the project's recurring "deterministic-but-wrong/non-bitwise" hazard: a nondeterministic cell scatter would make MEAM-cells *deterministic-per-run but non-bitwise to serial*, invisible to cells-vs-cells and 1-vs-z, caught only by the cells≡all-window raw-int64 gate and the canonical-sort discriminator. The CPU canonical-sort is **already recorded LOAD-BEARING for MEAM** (Me3b memory: "canonical ζ-sort LOAD-BEARING (screening-product Π_k order-sensitive)"). Me5b must carry that property onto the device grid.

**Decision:** the screening `sij`/`dscrfcn` k-loop must be reduced in **global-key order**, not cell-scatter order. Cheapest robust mechanism: the cells-aware screening helper **collects k-candidates from the stencil into a small local buffer, then sorts that buffer by `key[]` (≤~16 entries, register/insertion sort) before the product/derivative passes.** This makes cells `sij` bit-identical to the sorted all-window `sij` for *any* cell-scatter order — so the shipped nondeterministic `cell_scatter_kernel` can be reused **unchanged** (no new deterministic-scatter kernel needed). This is strictly simpler and more robust than depending on a key-stable scatter, and it is the Me5b analogue of Te3b's canonical-ζ-sort. Document it as the **"canonical-k cull."**

---

## 1. Culled kernel structure

New header `include/tdmd/cuda/zone_meam_cells.cuh` (mirrors `zone_eam_cells.cuh`). `zone_meam.cuh`, `zone_cells.cuh` stay **byte-untouched** (F-NOOP). New `MeamCellGrid`/`meam_build_window_grid`/`meam_cells_free` = structural copies of `EamCellGrid`/`eam_build_window_grid` (zone_eam_cells.cuh:182-234), just renamed — geometry (`make_zone_grid`, `cell_count_kernel`, CUB scan, `cell_scatter_kernel`) **reused verbatim from `zone_cells.cuh`**.

**K2 `meam_embed_kernel` — NOT culled, reused verbatim** (zone_meam.cuh:122-148). No neighbour scan.

### K1 → `meam_density_cells_kernel` (fork of zone_meam.cuh:69-117)

Replace the outer `for(jl<m)` (zone_meam.cuh:79) with the `±s_d` stencil **copied character-for-character from `eam_density_cells_kernel`** (zone_eam_cells.cuh:57-89: the `g.coords(xc,yc,zc,cxi,cyi,czi)`, the `(g.nz==1)?0:-g.sz` degenerate guards, the `wrapz/y/x` folds, the inner `for(t=beg;t<beg+cnt) → jl=b_order[t]; if(jl==c)continue`). For each accepted `j` (after `geom.reduce`), the body (zone_meam.cuh:83-113: `meam_getscreen_d_*`, the 27-lane `quantize` accumulation) is **byte-identical** except the screening call becomes the cells variant (§2). Density lanes are int64 → order-free B1 (the *lane sums* stay bitwise regardless of j-order; only `sij` inside the helper needs the canonical-k cull).

### K3 → `meam_force_cells_kernel` (fork of zone_meam.cuh:154-311) — four sub-culls

1. **o's own nbr-build** (`for(b<m)`, zone_meam.cuh:180-188) → `±s_d` stencil **centered on o**. Fills `nb[]/ndx[]/…` exactly as before. **kMaxNbr interplay (low risk, verified):** the cull reduces *enumerated candidates* (~68 cell candidates vs m), not the *in-rc count* (same superset + same `geom.reduce`) ⇒ `nb[]` contents bit-identical, the cap-64 HALT condition (sticky bit 4, zone_meam.cuh:187, thrown before writeback) unchanged. Cap stays 64.
2. **Role A k-loop** (zone_meam.cuh:215-223): iterates o's **cached `nb[]`** (`for kk<cnt`) — **already culled** the moment (1) is culled. The buffer IS the cell-neighbourhood. *Cleanest case, no grid traversal here.* Its `meam_getscreen_d_*` call (zone_meam.cuh:199, centered on o) → cells variant.
3. **Role B re-scan** (`for(b<m)`, zone_meam.cuh:249-259): scans **i's** window neighbours (i = a neighbour of o; i is the center, o the endpoint) → `±s_d` stencil **centered on i** (`g.coords(wx[i],…)` fresh per role-iteration), `if(b==i||b==o)continue` (zone_meam.cuh:250) preserved. Its `meam_getscreen_d_*` (zone_meam.cuh:234, centered on i) → cells variant.
4. **Role C double scan** (zone_meam.cuh:263-292): outer over o's `nb[]`; inner `for(b<m)` over **i's** neighbours j (zone_meam.cuh:270) → `±s_d` stencil **centered on i**; the `key[i]<key[b]` owner gate (zone_meam.cuh:272) + `if(b==i||b==o)` (zone_meam.cuh:271) preserved. Its `meam_getscreen_d_*` (zone_meam.cuh:277, centered on i) → cells variant; the `meam_screen_k_device` here (zone_meam.cuh:286) takes the **single fixed k=o** (no loop) — no inner cull needed.

The MEAM escalation over EAM: K3 re-centers the grid query on a **different atom `i` per role-iteration** (Roles B/C), vs EAM's single query-center per kernel. Same grid; only the stencil-center coords change. Cost: one extra `g.coords` per (role, neighbour) — negligible (~16 neighbours × 2 roles). Force output is **int64 register write-once (zone_meam.cuh:295) ⇒ bitwise to all-window by B1** *provided every role re-scan visits the identical in-rc set AND every embedded `sij` uses the canonical-k cull* (so the `bf`/`sk` values feeding the int64 quantize are themselves bit-identical).

**Why the inner screening k-scan MUST be culled (the quantitative crux, not an afterthought).** All-window K1 = O(m · m_inrc · m): the screening k-scan runs once per (c,j) pair *over the whole window*. Culling only the outer j-scan gives O(m · nbr · m) — still linear-in-m inside; **the inner screening dominates.** Culling the inner k-scan too gives O(m · nbr · nbr) — fully local. The cubic screening term is the reason angular potentials have no flagship number; the inner cull is the win.

---

## 2. The cells-aware screening helper (the canonical-k cull — the one genuine new machinery)

Add `meam_getscreen_d_cells_device` as a sibling of `meam_getscreen_d_device` (meam.hpp:578-652), `TDMD_HOST_DEVICE`, taking the grid + CSR + the center's precomputed cell coords:

```cpp
TDMD_HOST_DEVICE MeamScreenD meam_getscreen_d_cells_device(
    const double* wx, wy, wz, const long* key, int m, int c, int jl,
    djx,djy,djz, rij2, rij, const core::PairGeom& geom, const MeamScreenCParams& p,
    CellGrid g, const int* b_starts, const int* b_counts, const int* b_order,
    int cxi, int cyi, int czi);   // c's cell coords, computed ONCE by the caller
```

It replaces **both** `for(k=0;k<m;++k)` loops (meam.hpp:590, 621) with a single **gather-then-canonical-reduce**:
1. Walk the `±s_d` stencil **centered on c** (verbatim from zone_eam_cells.cuh:61-89), apply the *unchanged* skips/cuts (`k==c||k==jl`, `geom.reduce`, `rik2>rbound`, `rjk2>rbound`, `a<=0`, meam.hpp:591-601) — collecting surviving k into a small local array `kcand[KMAX]` with `KMAX = kMeamMaxNbr`.
2. **Sort `kcand` by `key[]`** (insertion sort, ≤~16 entries — global-key order = the sorted all-window slot order).
3. Run the product pass (`sij *= fcut`, meam.hpp:602-609) and the dscrfcn pass (meam.hpp:629-644) over `kcand` **in sorted order** — bodies character-identical.

Result: `sij`/`dscrfcn` are computed over the **same surviving-k set in the same global-key order** as the sorted all-window helper ⇒ **bitwise identical** ⇒ the shipped nondeterministic `cell_scatter_kernel` is reusable unchanged. The all-window `meam_getscreen_d_device` stays as the oracle's enumeration / `cull=false` reference (untouched). K1 and all three K3 roles call the `_cells` variant with their respective center's cell coords.

> Note: the early-`break` on `cikj <= Cmin` (meam.hpp:604, `sij=0; break`) is order-dependent only in *which* k triggers it, not in the result (`sij=0` regardless); sorting makes even that deterministic. Keep the `break`.

---

## 3. GpuMeamWinForce integration (cull_ flag + cell_div AUTO)

Mirror the EAM E5c block (eam_window_force_gpu.cuh:92-118, 168-199, 334-358) in `meam_window_force_gpu.cuh`:

- **`GpuMeamWindowState`** (meam_window_force_gpu.cuh:45): add `box_lo[3]/box_len[3]/periodic[3]/rcut`, `bool cull=true`, `int cell_div=0`, `unsigned long long cells_passes=0`, `MeamCellGrid grid_{}`, `bool grid_built_=false`. Capture box in the ctor (it already takes `const core::Box& box` via `GpuMeamWinForce` ctor, meam_window_force_gpu.cuh:145) — thread box into `GpuMeamWindowState`.
- **`ensure_grid_geometry(m_hint)`** — copy verbatim from eam_window_force_gpu.cuh:173-199 **including the AUTO `cell_div` heuristic** `k=clamp(round(cbrt((m/ncells_k1)/2.5)),1,4)`. **Measure, don't port EAM's k=3:** MEAM-Si rc=4.0 is far denser per cell than Al_zhou rc=10.1, so AUTO will likely pick **k=1 or k=2** — let the heuristic resolve it (it is rc/density-independent by construction).
- **`compute()` cull branch** (after the canonical sort + H2D, before the kernel launches at meam_window_force_gpu.cuh:226): branch on `s.cull` exactly like eam:334-358 — `ensure_grid_geometry(m)` once; per-pass `memset counts → cell_count_kernel → CUB ExclusiveSum → cudaMemcpyAsync cursor → cell_scatter_kernel`; then launch `meam_density_cells_kernel` / `meam_embed_kernel` (unchanged) / `meam_force_cells_kernel`; `++cells_passes`. `cull=false` keeps the all-window K1/K2/K3 (meam_window_force_gpu.cuh:227-236) as the **in-process bitwise reference**.
- **The canonical sort STAYS** (meam_window_force_gpu.cuh:196-205) — it orders the window for the FP density partial sums and is the reference order the canonical-k cull (§2) matches.
- **Firewall untouched** (`assert_supported`, meam_window_force_gpu.cuh:157-176): cells don't touch the transpose contract. F-NOOP.
- **Wire `cull`/`cell_div`** through `GpuMeamWinForce` ctor (and `GpuMeamRing`, meam_window_force_gpu.cuh:265) plus a `MeamConveyorOptions.cell_lists`-style flag.

**Default: OFF until G-B + G-POISON green on partial fixtures, then ON (AUTO).** Memory-gated to 10⁶ (§6).

---

## 4. Gates — `tests/test_cuda_meam_cells.cu`

Mirror `test_cuda_eam_cells.cu` A/B/P-k/W-k, escalated. Reuse the existing fixtures (`slab_win(pbc)` test_cuda_meam.cu:97, `taper_win()` :110, `dia_win()` :102) and the wired oracle `meam_direct_fp64` (:81-86). Add a `cell_div` knob + cells path to `gpu_window` (:65).

- **G-A — cells ≡ all-window, RAW int64 BITWISE** (all 27 `d_dens` lanes + `d_fx/fy/fz` + `d_pe_pair/embed` + `d_npartial/d_nzero`), on **diamond + partial slab (free) + partial slab (PBC seam) + taper**, ∀ `cell_div∈{1,2,3,4}`. Mirror `check_self_equiv` (test_cuda_eam_cells.cu:305). **This is the gate that forces the canonical-k cull (§2):** without it, `sij` reassociates and `d_dens` diverges. The `n_partial/n_zero` counters are a sharp tell (a dropped-k reclassifies a bond's screening band).
- **G-B — cells vs `meam_direct_fp64` oracle (BLOCKING, the SOLE dropped-screening-k witness):** force <1e-9, PE <1e-6, `n_screened_partial`/`n_screened_zero` **exactly equal**, on **partial slab (free + PBC) + taper-band**, asserting **`n_partial > 0`** (non-vacuous Role C / partial screening, test_cuda_meam.cu:194). ∀ `cell_div∈{1,2,3,4}`. The oracle uses a different enumeration (scatter-per-center, no grid) ⇒ it is the only witness to a *uniformly* dropped screening-k (to which G-A could be blind if both paths shared a build bug). **The diamond is binary-S (Role C dead) ⇒ NOT a cull-completeness witness; partial slab/taper are.** Mirror `check_vs_oracle` (test_cuda_eam_cells.cu:353).
- **G-POISON — teeth (two independent knobs, both MUST diverge):** (1) **stencil-too-small** — force `g.sx=g.sy=g.sz=1` while cells are sized `rc/k` (k≥3) so a screening-k at ~`1.04·rc` is dropped; assert G-B's force gap >1e-2 AND `n_partial` collapses (mirror `SubRcutPoisonHasTeeth`, test_cuda_eam_cells.cu:461). (2) **`drop_class>0`** (existing knob, meam_window_force_gpu.cuh:143 → zone_meam.cuh:214,264) → Role-C-dropped diverges from the oracle (test_cuda_meam.cu:224). If either fails to diverge, the gate is blind.
- **G-SORT (the load-bearing discriminator, no E5c analogue):** run cells K1 on a **reversed/shuffled** window with `skip_sort=true` vs canonical-sort → assert the raw `d_dens` flips iff the cell-walk order changed, and that with the canonical-k cull the result is **order-invariant** (un-permuted bitwise to the reference). Mirror the existing G3 sort-verdict (test_cuda_meam.cu:171-185). **Predicted load-bearing for MEAM** (unlike Tersoff/SW where measure-first found it defensive) — measure `|delta|`, don't assume.
- **G-W — tight-PBC boundary:** `n_k=2k+1` exact, the `n<2*cell_div+1 ⇒ n=1` off-by-one (zone_cells.cuh:91), verified by the FP64 oracle (mirror `SubRcutTightPbcBoundary`, test_cuda_eam_cells.cu:472). Catches a screening-k at `1.04·rc` poking past `s_d` at the tightest k=4.
- **G-kMaxNbr** under cells on the bench fixtures (cull doesn't change in-rc count; verify the wrap didn't double-count past 64 → the correct loud HALT).
- **G-momentum carried forward** (test_cuda_meam.cu:234): slab GPU `Σf∈(0,1e-9)` (screening transpose floor), diamond `Σf` int64-exact — cells must preserve it (a cell bug that symmetrized a write would zero `Σf` on the slab).

Run under `compute-sanitizer --tool memcheck/racecheck/initcheck` (`scripts/gpu_gate.sh`). The cells `d_fx/fy/fz` D2H reads all m but K3 writes only owned ⇒ keep the `cudaMemsetAsync` of the force buffers (meam_window_force_gpu.cuh:222-224).

---

## 5. The bench — `tools/bench_meam_cells.cu`

Fork `bench_eam.cu --free` / `bench_eam_ring.cu`. Fixture = jittered diamond-Si (`make_diamond_si` scaled to N∈{864, 4000, 16384, …, 10⁶}); diamond is binary-S (Role C mostly dead *in the bench*) but the bench measures **throughput**, not screening correctness (that is G-B's job on the partial slab). Report per N:
- **R_cull = t(all-window)/t(cells)** for K1, K3, total. E5c-EAM: 8.3×(864)→31×(4k)→117×(16k); MEAM's cubic screening term ⇒ **R_cull should be LARGER and grow steeper** (the inner k-cull removes an extra O(m) factor). Headline number.
- **Per-kernel split** (`#ifdef TDMD_MEAM_RING_TIMERS`, byte-identical default, mirror `TDMD_EAM_RING_TIMERS`, eam_window_force_gpu.cuh:81-90): confirm all-window K1 screening-scan dominates and the cull moves the bottleneck to K3 transpose-replay.
- **atom-steps/s** at the largest N that fits — **the first flagship-scale ANGULAR number.** Quote it INTERNAL (free-z, z=1, fp64), like E5c-ring's caveat — NOT the EAM 2.06e7 flagship comparison without the caveat.
- **realized cell_div (AUTO)** + per-cell occupancy + the 27-cell over-fetch (E5c-EAM ~6.4×; MEAM's screening ellipse `√ebound·rc < rc` ⇒ lower true hit-rate ⇒ **sub-rcut `cell_div` lever matters MORE** — report whether k>1 helps).
- **realized kMaxNbr** under cells across N (assert <64).
- **Live-ring translation** (mirror E5c-ring `bench_eam_ring.cu`, R_ring): does the isolated cull survive the host-orchestrated mutex-serialized `GpuMeamRing`? E5c found the ring 99% kernel-bound ⇒ expect it to translate.

---

## 6. THE FLAGSHIP plan (part A) — MEAM-cells at 10⁶ + SW/Tersoff sequencing

**(i) MEAM-cells enables the first angular 10⁶ number — this is the milestone's reason to exist.** All-window MEAM is O(m·nbr²·…) — hopeless at 10⁶ (why no angular flagship exists). Cells make it tractable. EAM hit 2.06e7 atom-steps/s at 10⁶; MEAM per-atom work is ~3–4× (4 densities + screening + 3-role transpose) ⇒ credible single-GPU MEAM ≈ **3e6–7e6 atom-steps/s at 10⁶**, reachable **only with cells.**
- **Run it as `bench_meam_cells --n 1000000` standalone** (isolated kernel throughput), **NOT the live ring** — E5c-ring measured the ring 99% kernel-bound but z=1 concurrency-dead (`SPEEDUP_z=0.99`), so the flagship is the isolated number, exactly as EAM's 2.06e7 was measured.
- **Memory-gate the claim to 10⁶.** Per-atom: 27 int64 density lanes (216 B) + `MeamEmbedDeriv` + force + window + grid CSR (`ncells` int + `m` int order). The 27 lanes dominate; budget **~6–8 GiB at 10⁶** (fits 15.5 GiB RTX 5080), **but not 10⁷** — gate the flagship claim to 10⁶ (EAM gated 10⁷ at 3.08 GiB; MEAM's 27 lanes are 27× EAM's single ρ). The cells add only the grid CSR (tiny vs 27 lanes) — **no cross-role CSR like verlet ⇒ cells-default is memory-safe**, gate the *flagship claim* (10⁶), not the default.

**(ii) SW/Tersoff: reuse infra, fork the kernel.** The grid is **potential-agnostic** (`make_zone_grid`/`CellGrid`/`cell_count`/`cell_scatter`/CUB — none knows about MEAM) ⇒ **100% infra reuse.** What differs is the per-kernel cull: SW's φ₃ triplet enumeration (center + two wings), Tersoff's 2-pass directed-bond ζ. Each needs its own `zone_{sw,tersoff}_cells.cuh` forking its force kernel's scans onto the `±s_d` stencil, importing `zone_cells.cuh` unchanged. **SW/Tersoff are strictly SIMPLER than MEAM** — no nested screening k-cull, no FP-product canonical-k order hazard (SW int64-order-free; Tersoff ζ is FP but single-window, already canonical-ζ-sorted on the host).

**Recommended sequencing:** **MEAM-cells first (this PR)** — it de-risks the hardest case (the nested doubly-cull + the canonical-k FP-product). Then **SW/Tersoff cells (T5b/Te5b) as a mechanical fast-follow** once the stencil-fork pattern is proven on MEAM. **Do NOT claim a SW/Tersoff flagship until their cells land** — the single angular 10⁶ number Me5b delivers is the MEAM one. A "флагман углов" table ultimately carries all three.

---

## 7. Ordered risk list (each with the gate that catches it)

| # | Risk | Why invisible to weak gates | Discriminator |
|---|---|---|---|
| **1 ★ (biggest correctness risk)** | **Nested-screening `sij`/`dscrfcn` FP-order**: cell-walk order ≠ sorted-slot order ⇒ FP product/sum reassociate; B1 does NOT cover them; shipped scatter nondeterministic | deterministic-per-run, non-bitwise-to-serial; blind to cells-vs-cells + 1-vs-z + run-to-run | **G-A raw `d_dens` cells≡all-window on partial slab + G-SORT.** Mitigation: §2 **canonical-k cull** (gather→key-sort→reduce inside the helper) — carries the Me3b LOAD-BEARING sort onto the device grid |
| 2 | **Screening k-loop stencil centered on `jl` instead of `c`** (easy copy bug) | k near c but far from jl dropped; deterministic; cells-vs-cells agree | **G-B FP64 oracle on partial slab** (all-window centers on c correctly) |
| 3 | **Role B/C use o's stencil instead of i's** (forget per-neighbour re-center) | i's far neighbours dropped; Role C f_k wrong; diamond binary-S hides it | **G-A on partial slab (Role C live) + G-B + G-POISON** |
| 4 | **Role B/C reach truncation** (window <2·rc or s_d too small) | wrong screened force; deterministic ⇒ invisible to cells-vs-cells | **G-B oracle (force <1e-9 + count==) + G-P.** Guarded by `effective_range_for("meam")={2,true}` + ctor min-image (meam_window_force_gpu.cuh:147-150) |
| 5 | **Screening-k at `1.04·rc` pokes past `s_d` at tight k=4** | off-by-one in `s_d`; drops a screening-k | **G-B/G-W at k=4 on partial slab** — `s_d=ceil((rc+pad)/c)` from realized c (zone_cells.cuh:95-99), never from k |
| 6 | **`--fmad=false` not linked** | exp/pow/Horner fuse to fma ⇒ ~1 ulp ⇒ G-A flips, looks like a cull bug | link `tdmd_eam_cuda_flags` (mandatory, zone_eam_cells.cuh:5-8); CPU↔GPU stays tolerance |
| 7 (low, verified) | **kMaxNbr=64 overflow** under denser cull fixtures | silent truncation = dropped k | already guarded HALT (zone_meam.cuh:185-187, sticky bit 4, thrown before writeback); G-kMaxNbr |

**The single biggest correctness risk: #1 — the nested screening `sij` FP-order.** EAM density is a *sum* (cells trivially bitwise by B1); MEAM screening is an FP64 *product* over k, FP-non-associative in cell-walk order, NOT B1-protected. Resolution = the **canonical-k cull** (§2: gather the stencil k-candidates, sort by global key, reduce in that order), gated by **G-A** (raw `d_dens` cells≡all-window on the partial slab) and **G-SORT**. This is the Me5b analogue of Te3b's canonical-ζ-sort, and per the Me3b memory it is genuinely load-bearing for MEAM (not defensive).

**The single biggest scope risk: claiming the angular flagship too broadly.** Me5b delivers exactly ONE angular 10⁶ number — **MEAM** — and it is an *internal* (free-z, z=1, fp64) isolated-kernel number, NOT the live ring (concurrency-dead) and NOT yet SW/Tersoff (their cells are deferred). Memory-gate the claim to 10⁶ (the 27 density lanes blow the budget at 10⁷). Default-ON only after G-B + G-POISON are green on the **partial** fixtures (the binary-S diamond cannot witness screening-cull completeness — the recurring "structurally-dead trap" that bit Me1/Me3/Me5; the load-bearing acceptance check is verifying `n_partial>0` actually fires *through the cull*, on the cells path, not merely present in the fixture).

### Files mirrored (citations)
- Cull kernels: `include/tdmd/cuda/zone_eam_cells.cuh:46-91` (density), `:98-171` (force), `:182-234` (grid/CSR) → `include/tdmd/cuda/zone_meam_cells.cuh`.
- Shared grid infra: `include/tdmd/cuda/zone_cells.cuh:54-138` (`make_zone_grid`, `coords`, `srad`/`s_d`), `:142-159` (count/**scatter — reused as-is; canonical-k cull removes the order dependence**).
- Kernels to cull: `include/tdmd/cuda/zone_meam.cuh:79` (K1 outer) `:85` (K1 nested screen), `:180-188` (K3 nbr-build), `:215,249,270` (Role A/B/C), `:295` (write-once int64).
- Nested screening to fork: `include/tdmd/potentials/meam.hpp:578-652` (`meam_getscreen_d_device`, k-loops `:590,:621`; bound `:584,:596-597`; product `:608`; sum `:643`; reach guard via `recompute`).
- Integration: `include/tdmd/cuda/meam_window_force_gpu.cuh:45` (state), `:178-262` (compute), `:196-205` (canonical sort — the reference k-order); pattern `include/tdmd/cuda/eam_window_force_gpu.cuh:92-118,173-199,334-358`.
- Gates: `tests/test_cuda_eam_cells.cu:305` (A), `:353` (B), `:461` (P-k), `:472` (W-k); oracle + fixtures + sort-verdict `tests/test_cuda_meam.cu:81-86,97-116,171-185,224,234`; generator `include/tdmd/gen/partial_screen_slab.hpp`.
- Bench: `tools/bench_eam.cu`, `tools/bench_eam_ring.cu` → `tools/bench_meam_cells.cu`.

**Bottom line:** Me5b = "fork `zone_eam_cells.cuh` for K1 + K3 (K2 reused), one shared `rc`-padded whole-window grid (`make_zone_grid` n_zones=1), `cull_`+`cell_div` AUTO wired into `GpuMeamWinForce` exactly like EAM." The one genuine escalation over EAM: the **nested screening k-scan is the cubic-cost dominant term and must itself be cell-culled**, and because `sij=Π_k fcut`/`dscrfcn=Σ_k` are FP-order-sensitive and B1 does NOT protect them while the all-window path is already key-sorted, the cells screening helper needs a **canonical-k cull** (gather stencil candidates → key-sort → reduce) — the device analogue of the LOAD-BEARING MEAM canonical sort. The reach is sub-cell (`√ebound·rc ≈ 1.04·rc`), so one grid covers density + force + screening-k. The FP64 oracle on the **partial slab/taper** (never the binary-S diamond) is the sole dropped-screening-k witness; G-A (raw int64 cells≡all-window) is the canonical-k forcing test.