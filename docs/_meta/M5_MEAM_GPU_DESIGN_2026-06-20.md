I now have the complete, precise grounding for every load-bearing structural claim. `meam_getscreen_d` and `meam_getscreen` take only a `MeamNbr` + `const std::vector<MeamNbr>&` + POD `MeamParams` — the device port replicates their bodies over register arrays (no vector). `meam_dens_final_deriv` operates on the POD `MeamDensity`. The screening k-loop in phase (3) is self-contained (recomputes cikj/dfcut/dCfunc2 from positions, reads only the bond-level `dscrfcn_ij`). I have everything needed to synthesize the implementation plan.

---

# Me5 — GPU MEAM: Implementation Plan (synthesized, code-grounded, implementation-ready)

This plan reconciles the three design lenses against the **actual code** I read (`meam_zone.hpp:60–366`, `meam_ring.hpp:91–222`, `meam.hpp:457–584,674,811`). Where the lenses disagreed on a number, the verified value is used and flagged.

## CORRECTIONS to the lenses (verified against source — do NOT propagate the lens errors)

- **Density lane count is 27 int64, not 14/16/23.** `FixedDens` (meam_zone.hpp:91–94) = rho0(1) + arho2b(1) + arho1(3) + arho2(6) + arho3(10) + arho3b(3) + t_ave(3) = **27 `ForceAccum` lanes**. Lay out as SoA of 27·m `long long`.
- **The phase-(3) screening k-loop does NOT read a cached per-k `scr`.** It recomputes `cikj/dfcut/dCfunc2` from positions every k (meam_zone.hpp:329–351) and reads only the **bond-level** `dscrfcn_ij`/`scrfcn_ij`/`fcpair` (lines 170–172). ⇒ On device, K3 needs only the per-bond `{scrfcn,fcpair,dscrfcn}` for the bond it owns, recomputed by one `meam_getscreen_d_device` call; the k-loop's per-k terms are self-contained. **This kills the "ragged scr cache" worry — no device-side variable-stride bond cache is needed at all.**
- **`id == index+1` bijection (meam_ring.hpp:107):** sorting by `key` == sorting by window index. The canonical sort is a `key`-sort exactly as the CPU `MeamWinForce` does it.

---

## (1) Device kernel structure — THREE kernels (density → embedding → force)

**Verdict: three kernels, the EAM `zone_eam.cuh` shape with screening folded in.** Confirmed by reading `meam_window_force`: it is internally three phases (meam_zone.hpp:97–136 density, :143–156 embedding-deriv for **all** window atoms incl. halo, :165–365 force transpose-replay). The single-kernel "recompute each center's screened density on the fly" alternative is **rejected**: Role B/C need `frhop`/`ed`/`dens` of `o`'s neighbours' centers (halo), and recomputing a center's full 27-lane *screened* density (itself an inner screening product) per (role,bond,k) is ~O(nbr⁴) per owned vs the three-kernel O(m·nbr·nbr_screen) once. EAM split density/embedding/force for exactly this reason.

New file `include/tdmd/cuda/zone_meam.cuh`, one null-stream sequence per `compute()` (mirror `eam_window_force_gpu.cuh:334–369`):

| Kernel | Threads | Reads | Writes | CPU mirror |
|---|---|---|---|---|
| **K1 `meam_density_kernel`** | one per **window** atom `c` (m, owned+halo) | `wx/wy/wz`, screening over c's nbrs | `d_dens[c]` = 27 int64 SoA lanes | meam_zone.hpp:97–136 |
| **K2 `meam_embed_kernel`** | one per **window** atom `c` (m, owned+halo) | `d_dens[c]` decoded | `d_ed[c]` = `MeamEmbedDeriv` POD | meam_zone.hpp:143–156 |
| **K3 `meam_force_kernel`** | one per **OWNED** atom `o` (n_owned) | `wx/wy/wz`, `d_dens`, `d_ed` (all window) | `d_fx/fy/fz[o]` int64 write-once, `d_pe_embed`, `d_pe_pair`, counts, min_r2 | meam_zone.hpp:165–365, **gathered per owned** |

K2 runs over **all m** (not n_owned) — Role B/C read `ed[i]`/`dens[i]` of a **halo** center `i` (meam_zone.hpp:188, 255–256). This is the non-negotiable reason embedding is a separate kernel over the full window.

### Device buffers (in `GpuMeamWindowState`, growable like `GpuEamWindowState::grow`, eam_window_force_gpu.cuh:146–166)
```
double*    d_wx, d_wy, d_wz;        // [m] host-key-sorted window positions
long*      d_key;                   // [m]
int*       d_owned;                 // [n_owned] window-local owned indices
// 27-lane int64 density SoA, per-atom-contiguous (atom c at d_dens[27*c + lane]):
long long* d_dens;                  // [27*m]   Q24.40 (ForceAccum::kScale)
MeamEdDev* d_ed;                    // [m]      decoded POD (13 doubles)
long long* d_fx, d_fy, d_fz;        // [m]      int64 Q24.40 force (write-once per owned)
long long* d_pe_embed, d_pe_pair;   // [1] each (block-reduced + atomicAdd)
unsigned long long* d_min_r2;       // [1] atomicMin(double-bits)  (zone_force.cuh:58)
long long* d_n_partial, d_n_zero;   // [1] each — TEST-ONLY witnesses
int*       d_overflow;              // [1] sticky HALT flag
```
27·8 = 216 B/atom density + 13·8 = 104 B/atom `ed`. **Per-atom contiguous** density (stride 27, not lane-major) so K2 reads one atom's 27 lanes stride-1.

---

## (2) On-device screening + density + force replication

### HOST_DEVICE helpers — ALREADY done (verified meam.hpp:38–209)
`fcut, dfcut, G_gam, dG_gam, dCfunc, dCfunc2, embedding, zbl, erose` are **already `TDMD_HOST_DEVICE`** (the comment at meam.hpp:33 confirms "for the future GPU rung (Me5)"). No work.

### New `TDMD_HOST_DEVICE` annotations (POD-only bodies — confirmed)
- **`meam_dens_final_deriv`** (meam.hpp:674) — operates on POD `MeamDensity`, calls only already-device leaf helpers + reads `p.v2D/v3D/shp`. Mark `TDMD_HOST_DEVICE`; K2 calls it directly with a stack `MeamDensity` filled from decoded int64. Add `static_assert(std::is_trivially_copyable_v<MeamDensity>)` and `<MeamEmbedDeriv>`.

### New device functions (replicate the vector-taking bodies over register/window arrays)
The two big routines take `const std::vector<MeamNbr>&` → cannot go to device as-is. **Replicate their bodies** as device free functions operating on the gathered window (the CPU `meam_window_force` already re-inlines `calc_rho1`, so this is transcription, not new derivation):

- **`meam_getscreen_d_device(wx,wy,wz, m, c, ej_local, geom, pv, &scrfcn,&fcpair,&dscrfcn)`** — replicate meam.hpp:458–529 **verbatim**, the inner `for(const auto& ek : nbr_i)` becoming `for(int k=0;k<m;++k)` with an in-rc `geom.reduce` test (k is a screening candidate iff `rik2<rc²`). **TWO trap lines to replicate verbatim** (silent-drop class):
  - meam.hpp:479 / 553 — `if (a <= 0.0) continue;` is the **ONLY** negative-C rejection; a `cikj<0` early-reject would silently drop screening-k (the colinear-non-rejection trap, meam.hpp:557).
  - meam.hpp:525 — `dscrfcn = dscrfcn*coef1 − coef2` — the **MINUS** on the radial taper (a `+` is silent on diamond, only the slab/cluster oracle catches).
- **K1 density inner** = the `calc_rho1` body already re-inlined at meam_zone.hpp:108–134, accumulated into **27 thread-local int64 registers** via `quantize(value, ForceAccum::kScale, overflow)` (zone_force.cuh:47), written once. Order-free by B1; no atomics.
- **`meam_bond_force_device(...)`** — factor the ~200-line bond body (meam_zone.hpp:179–319: `arg1i*`/`drho*dr*`/`drho*drm*`/`dt*dr*`/`drhodr*`/`drhods*`/`dUdrij`/`dUdsij`/`fm0..2`) + the screening k-loop (meam_zone.hpp:329–362) into ONE `TDMD_HOST_DEVICE` helper shared by K3, CPU `meam_window_force`, and `meam_run_fixed_force` (the SW/Tersoff single-source discipline). **This refactor MUST be byte-identical on the CPU side** (CPU suite 36/36 green = F-NOOP house rule) — do it FIRST, prove F-NOOP, then call from K3.

### Screening cache decision: **RECOMPUTE in K3, do NOT materialize on device**
The CPU caches `scr[c][jn]` only to avoid a second screen pass; K3 re-derives `meam_getscreen_d_device` for its one owned bond (it has the center's window-neighbours by re-scan). This is the GPU idiom (Te5 recomputes `zeta_center` per role rather than caching, zone_tersoff.cuh:104/130/164) and — verified above — the per-k k-loop terms never read the cache anyway. **Fallback carried, default off:** if K3 register pressure tanks occupancy, write a flat `d_dscrfcn[m·kMaxNbr]` in K1; gate-identical. Default = recompute.

### K3 — the force transpose-replay (THREE roles, int64 registers, write-once)
One thread per OWNED `o`. Cache `o`'s own in-rc neighbour list once into a `kMaxNbr` local buffer (zone_tersoff.cuh:78–88). Accumulate `qfx/qfy/qfz` int64 registers across three receiver roles, decoding each via `quantize(value, ForceAccum::kScale, overflow)`; `dx = wx[o] − wx[b]` (PROJECT convention, matches `MeamNbr` x_j−x_i? — **NB meam_window_force builds nbr with `dx = wx[b]−wx[a]`, meam_zone.hpp:82; replicate THAT sign exactly**, not the Tersoff `xo−wx[b]`):

- **Role A (`o == i`, lower key):** loop o's neighbours `j`, `key[o]<key[j]`; run `meam_bond_force_device` in **o's frame**; take `+fm` (meam_zone.hpp:321) + `+force1·d_ik` (k-loop, :356). Drives the screening k-loop with `o` as center.
- **Role B (`o == j`, higher key):** loop o's neighbours `i`, `key[i]<key[o]`; **re-scan the window for `i`'s neighbours** (Te5 Role-3/4 brute re-scan, zone_tersoff.cuh:138–171 — no second local buffer, unbounded, cannot overflow); run the bond body for `(i,o)` in **i's frame** reading `d_ed[i]`/`d_dens[i]` (i possibly **halo**); take `−fm` (:322) + `+force2·d_jk` (:357). **THE LOAD-BEARING LINE (meam_zone.hpp:35–37):** computing S_ij / the chain from `o`'s frame instead of the halo center `i`'s frame diverges uniformly across z → blind to 1-vs-z + run-to-run, **only the FP64 oracle catches**. This is correctness risk #2.
- **Role C (`o == k`, screening third atom):** for each neighbour `i` of `o`, re-scan `i`'s neighbours `j` forming a **partial** bond `(i,j)` (`key[i]<key[j]`, `0<sij<1`) where `o` is a screening-k; take `−(force1·d_ik + force2·d_jk)` (:359–361). The `needs_transpose` write `q(k)≠−q(i)` — the firewall's whole purpose.

**Write-once:** `d_fx[o]=qfx` once (no atomics, no cross-atom force write → B1 order-free). PE: `qpe_embed += quantize(ed[o].F)` once per owned (meam_zone.hpp:155); `qpe_pair += quantize(phi·sij0)` when `o` is the lower-key center (:177) — block tree-reduce + one atomicAdd each (zone_tersoff.cuh:178–189). Counts gated `is_owned ∧ key[c]<key[j]` (meam_zone.hpp:104–106), reduced to device scalars (test-only).

---

## (3) `GpuMeamWinForce` policy + the canonical-sort posture

New file `include/tdmd/cuda/meam_window_force_gpu.cuh`, structure copied from `tersoff_window_force_gpu.cuh` (transpose + canonical sort + un-permute) fused with `eam_window_force_gpu.cuh` (spline upload + multi-buffer growth + mutex).

### `MeamParams` is NOT trivially copyable — the #1 plumbing item
Verified: it holds `std::vector<double> phir, phirar..phirar6` (meam.hpp:254–255). **Do NOT pass by value** (slices the spline). Build a POD `MeamParamsView`:
```
struct MeamParamsView {
  // all scalars by value: alpha, beta0..3, re, t1_eff, t2, t3, t0, Cmin, Cmax, ebound,
  // rho0, A, Ec, rc, delr, gsmooth, ibar, v2D[6], v3D[10], vind2D[3][3], vind3D[3][3][3], shp[3], ...
  const double *phirar, *phirar1, *phirar2, *phirar3, *phirar4, *phirar5, *phirar6;  // device ptrs
  int nr; double rdrar;
  TDMD_HOST_DEVICE double phi_spline(double r) const;   // Horner over device ptrs (meam.hpp:330)
  TDMD_HOST_DEVICE double phip_spline(double r) const;  //                          (meam.hpp:341)
};
static_assert(std::is_trivially_copyable_v<MeamParamsView>);
```
Upload the 7 spline arrays to device pointers in the state ctor (mirror `init_setfl`, eam_window_force_gpu.cuh:120–128). The spline is pure Horner + int knot (transcendental-free → that part is even bitwise), but K3 overall is tolerance because of exp/log elsewhere. Make `phi_spline`/`phip_spline` `TDMD_HOST_DEVICE` methods on the view.

### `compute()` — exact 14-arg const signature (meam_ring.hpp:96–102), so the policy-ctor (meam_ring.hpp:193) just works
1. mutex-lock; `grow(m)`.
2. **Canonical-sort the window by `key`** (mirror tersoff_window_force_gpu.cuh:147–155 verbatim: perm/inv/sx/sy/sz/sk/sowned), with a `skip_sort` test hook (tersoff_window_force_gpu.cuh:100).
3. Upload sorted window; `cudaMemsetAsync(d_fx/fy/fz, 0, m·8)` (the **initcheck fix** — D2H reads all m, K3 writes only owned); seed scalars (pe=0, min_r2=sentinel, overflow=0, counts=0).
4. Launch K1 → K2 → K3 on the null stream.
5. D2H int64 forces + `d_pe_embed` + `d_pe_pair` + min_r2 + overflow + counts; sync.
6. **Throw on overflow** (HALT, before writeback — eam_window_force_gpu.cuh:402).
7. **Un-permute** `wF[perm[t]].raw = hf[t]` (tersoff_window_force_gpu.cuh:193).
8. **Fold both PE accumulators via int64** `pe.raw += h_pe_embed + h_pe_pair` (the order-free fold, meam_ring.hpp:124).

### Canonical sort — LOAD-BEARING or DEFENSIVE? **State the expected measurement.**
Te5/SW measured it **defensive** (GPU single-eval, block window z-independent). **For MEAM, prediction = LOAD-BEARING for ring-GPU≡serial-GPU bitwise** (the first GPU rung where it genuinely bites), because:
- **1-vs-z and run-to-run: DEFENSIVE** — the block window `[pred][center][succ]` is the same multiset in the same order for any z (`eam_window_layout`); K1 walks it in block order → z-independent without the sort. (Same as Te5/SW.)
- **ring-GPU ≡ serial-GPU: LOAD-BEARING** — the serial `meam_zone_pass` window is **sorted** (`zone_eam_window`); the ring window is **unsorted block order**. The density int64 quantization is order-free (B1), BUT the screening **product** `sij = Π_k fcut(cikj)` is an FP64 multiply chain (meam.hpp:486, non-associative); a different k-order flips a ULP in `sij` → flips a Q24.40 quantum in the density input → breaks ring≡serial bitwise. MEAM has the FP screening/density sums Te5/SW lacked.

**MEASURE it (the RE-MEASURE the prompt demands):** run K1 with `skip_sort=true` on the **partial-screening slab** (where screening is active) and compare raw int64 to the sorted run. **Predicted: flips ≥1 quantum on the slab, bitwise on the diamond (binary S, product trivial).** If it stays bitwise even on the slab → defensive like Te5/SW (annotate the corrected banner, the Me3b/Te3 "measured-false" honesty pattern). **Carry the sort regardless** (correctness).

### Firewall — copy `MeamWinForce::assert_supported` VERBATIM (meam_ring.hpp:153–177)
Accept `[Density, Embedding, Force(needs_transpose=true)]`; reject symmetric Force pass 2 (the EAM trap), `[BondOrder,Force]` pass 0 (Tersoff), iterative (QEq/ReaxFF). This **INVERTS** `GpuEamWindowForce::assert_supported` (which throws on `needs_transpose`). Wire via `if constexpr requires` in `MeamRing::run()` (meam_ring.hpp:221 — already present). The screening 3rd-atom-k write IS the legitimate non-symmetric write; the firewall (built in `35cf18b`) forces K3 to build it rather than silently run the symmetric EAM accumulator.

---

## (4) kMaxNbr bound + HALT + init/memset discipline

- **`kMaxNbr = 64`** (the Te5/SW value, zone_tersoff.cuh:34). MEAM Si at **rc=4.0** (verified meam.hpp:230) sees ~16 in-rc pre-screening (4×1NN@2.35 + 12×2NN@3.84<4.0); the screening k-loop scans the **full pre-screening** list, so the cached buffer must hold all ~16. 64 = ~3–4× headroom. **Only `o`'s OWN cached list hits kMaxNbr;** the Role-B/C window re-scans are unbounded (no buffer → cannot overflow). **MEASURE** the realized `max(cnt)` on diamond + thermalized + slab fixtures (G1 gate); keep 64 unless a fixture exceeds ~32; record in the Bench. rc=4.0 is LARGER relative to lattice than Tersoff's 3.2 — re-confirm, don't assume.
- **Overflow = guarded HALT, NEVER silent truncation** (drops a screening-k → wrong force, invisible to 1-vs-z): `atomicOr(d_overflow, bit)` in K1/K3; the wrapper throws AFTER sync, BEFORE writeback (zone_tersoff.cuh:87) → `Halt::Internal`.
- **init/memset (initcheck-clean):** `cudaMemsetAsync(d_fx/fy/fz,0,m·8)` before launch (K3 writes only owned slots, D2H reads all m — the SW-T5 uninit-D2H carry-forward, tersoff_window_force_gpu.cuh:168–170). K1 writes all 27·m density lanes (every window atom a thread, guard `if(c>=m) return`); K2 writes all m `d_ed`. Seed scalars (pe/min_r2/overflow/counts). **G-initcheck gate** in `gpu_gate.sh`.
- **Three-leg min-image ctor guard** (meam_ring.hpp:204–206): MEAM's j–k leg is a **difference of two min-imaged vectors** (worse than Tersoff) → `box.len(d) < 2·rc → throw`. `GpuMeamWinForce(const MeamParams&, const Box&)` ctor mirrors `GpuSwWinForce` ctor.

---

## (5) `GpuMeamRing` via the policy-ctor — ZERO orchestration change
```
using GpuMeamRing = potentials::MeamRing<double, GpuMeamWinForce<double>>;
```
The policy-injected ctor (meam_ring.hpp:193) wires it; `MeamRing` calls `winforce_.compute(...)` at meam_ring.hpp:445. GpuMeamRing inherits the **proven Me3b ring** (FSM, Λ-chain, defer_head, second-forward-hop, PBC cyclic window) by construction — no new orchestration. The firewall fires at `run()` (meam_ring.hpp:221).

---

## (6) THE GATES — `test_cuda_meam.cu` (per-window kernel) + `test_cuda_meam_ring.cu` (live ring)

**Fixtures:** `gen/diamond_si.hpp` (perturbed diamond — **binary S, Role C structurally DEAD**, the bulk case) AND the partial-screening fixtures `reference_data/meam_si/meam_tri3.data` (3-atom, S≈0.50) + `partial_slab.data` (24-atom, PBC seam, the new files in git status). **The partial fixture is LOAD-BEARING: on the diamond Role C never fires** (the recurring Te1-`fc_d` / Me1-partial-screening structurally-dead trap, meam.hpp:407) — a dropped Role-C write is invisible on diamond. Run every screening gate on the slab/cluster, not just diamond.

### `test_cuda_meam.cu`
- **G1 — kMaxNbr margin** (MEASURE): instrument K1, assert `max(cnt) ≤ ~24` on diamond + slab; record the 64 headroom.
- **G2 — CPU↔GPU TOLERANCE** (NOT bitwise — the inverse of EAM; exp/log/pow in `G_gam`/`embedding`/the four `rhoa*` diverge ~1 ulp): GPU vs CPU `meam_run_fixed_force` (meam.hpp:1198), `max|ΔF| < 1e-9`, `|ΔPE| < 1e-6`. On diamond AND the partial cluster.
- **G3 — GPU-INTERNAL bitwise:** run-to-run + window-permutation raw int64 `memcmp` EXACT. **This is the RE-MEASURE of the sort:** with `skip_sort=true`, predict it passes on diamond, **flips on the slab** (→ load-bearing); with `skip_sort=false` passes everywhere. Document the verdict in the banner.
- **G4 ⭐ — FP64 ORACLE (free), the LOAD-BEARING completeness gate:** GPU vs `meam_direct_fp64` (meam.hpp:811, scatter-per-center, **different enumeration**) `< 1e-9` force / `< 1e-6` PE, on the **partial cluster + slab**. The ONLY witness to a uniformly-dropped screening-k (invisible to 1-vs-z + run-to-run + CPU↔GPU-internal-bitwise).
- **G5 ⭐ — FP64 ORACLE (PBC seam):** same on `partial_slab.data` (two-wing PBC, the j–k difference-leg). Catches a wrong-image screening triple.
- **G6 ⭐ — POISON teeth:** K3 with a `drop_class>0` device flag (drop the Role-C k-scatter, meam_zone.hpp:328) → must DIVERGE from the oracle by `>1e-3` on the partial slab (atom-k force collapses). On diamond it's a no-op (binary S) → teeth bite ONLY on the slab — exactly why the slab is load-bearing.
- **G7 ⭐ — MOMENTUM discriminator** (the Me4 finding): on the slab, GPU int64 `Σf ∈ (0, 1e-9)` (the screening transpose quantizes f_i,f_j,f_k independently → `rint(f_i)+rint(f_j)+rint(f_k)≠0`); FP64 oracle `Σf < 1e-12` (round-off). On the diamond, Σf is int64-exact (binary S → symmetric `q(j)=−q(i)` pairs cancel). **A symmetric GPU shortcut zeros atom-k AND makes both Σf exact** → the gap proves Role C is live. Mirror `MomentumFloorIsInt64Quantization` (test_meam.cpp).
- **G8 — overflow HALT:** compile a `kMaxNbr=8` kernel variant or use a dense cluster → expect the throw (sticky bit set, before writeback).

### `test_cuda_meam_ring.cu`
- **G9 — ring(z=1) ≡ serial-VV GPU-internal bitwise** (raw int64 / x,v).
- **G10 — ring 1-vs-z bitwise:** z∈{2,3,5} free + PBC{5,6} (n_zones≥5 for PBC, meam_ring.hpp:208), fixed AND auto-dt, x/v bitwise. (1-vs-z is blind to dropped-k — G4 covers that.)
- **G11 ⭐ — ring-vs-FP64-oracle trajectory:** ~1000 steps, ring within 1e-6 of a serial-VV driven by `meam_direct_fp64`, on a partial-screening-active config (the MB2 witness the ring can't self-check).
- **G12 — firewall:** symmetric-Force descriptor → throw; real MEAM → run.
- **G13 — anti-deadlock:** z=1..5 complete without hang.
- **memcheck + racecheck + initcheck CLEAN** (`scripts/gpu_gate.sh build-cuda`).

**Build:** both `.cu` link `tdmd_eam_cuda_flags`, compile **`--fmad=false`** (CMakeLists.txt:392–411 pattern). `--fmad=false` is REQUIRED for GPU-INTERNAL structural determinism (NOT for CPU↔GPU — impossible with transcendentals). No `--use_fast_math`/`--ftz`/`--prec-div=false`. Register both with the `cuda` ctest label.

---

## (7) Ordered risk list (deterministic-but-wrong — green to consistency gates)

| # | Risk | Why invisible | Discriminator (the catching gate) |
|---|---|---|---|
| **1 🔴 BIGGEST CORRECTNESS** | **Uniformly-dropped screening-k** (K1's k-loop or K3's Role-C re-scan drops a candidate everywhere — an off-by-one window bound, the `a≤0` colinear-reject trap meam.hpp:479/553, or the `−coef2` sign meam.hpp:525) | 1-vs-z, run-to-run, CPU↔GPU-internal-bitwise ALL green (drop is uniform) | **G4/G5 FP64 oracle** (`meam_direct_fp64`, different enumeration) — the SOLE witness; **G6 POISON** proves the gate has teeth; the `{2,true}` reach proof guarantees the k IS resident |
| **2 🔴** | **Role B reading the bond from `o`'s frame, not the halo center `i`'s** (meam_zone.hpp:35–37) | Diverges uniformly across z → oracle-only | **G4/G5 free + PBC** |
| **3** | **Wrong `ed`/`dens` index in K3** (reading `d_ed[o]` where CPU reads halo `d_ed[i]`) | Uniform divergence | **G4/G5 oracle** |
| **4** | **Role C never built** (symmetric shortcut — drops the 3rd-atom-k write) | Σf becomes exact, forces deterministic + stable | **G7 momentum discriminator** (int64 Σf≠0 vs oracle round-off) + **firewall** forces the transpose build + **G6 poison** |
| **5** | **`MeamParams` slicing** (φ-spline vector passed by value → zeroed spline) | Compile-silent garbage | **`MeamParamsView` POD + `static_assert(trivially_copyable)`** + device spline upload |
| **6** | **Canonical-sort wrong verdict** (assume defensive, but MEAM screening product is load-bearing → ring≠serial) | run-to-run green (same order each run) | **G3 window-permutation with `skip_sort`** — RE-MEASURE; predicted load-bearing on slab |
| **7** | **j–k wrong-image** (difference of two min-imaged vectors) | Deterministic | **Three-leg min-image ctor guard** (throw) + **G5 PBC-seam oracle** |
| **8** | **Uninit-D2H** (K3 writes only owned slots) | Undefined but benign-looking | **memset forces + initcheck gate** |
| **9** | **FMA fusion** (GPU-internal nondeterminism) | Passes once, flips on recompile | **`--fmad=false`** + **G3 run-to-run** |

---

## THE TWO FLAGS (explicit, as required)

- **🔴 SINGLE BIGGEST CORRECTNESS RISK: the uniformly-dropped screening-k (Risk #1).** It is invisible to *every* consistency gate (1-vs-z, run-to-run, even CPU↔GPU-internal-bitwise) because the drop is systematic. The **only** witness is the independent FP64 oracle `meam_direct_fp64` (a different enumeration) on a **partial-screening fixture** (`meam_tri3` / `partial_slab`) — because on the diamond, screening is binary and Role C is structurally dead, so the diamond cannot witness the screening force at all. **Acceptance MUST run G4/G5/G6/G7 on the partial fixtures, not the diamond.** This is the load-bearing acceptance item.

- **🟠 SINGLE BIGGEST SCOPE RISK: the K3 force kernel's three-role gather + the `meam_bond_force_device` factoring.** The ~200-line bond body (meam_zone.hpp:179–319) plus the screening k-loop must become ONE `TDMD_HOST_DEVICE` helper shared with the CPU path, and that refactor **must be byte-identical on the CPU** (the 36/36 suite stays green — the F-NOOP house rule) BEFORE any device code is written. Role B/C require brute-force window re-scans for a *neighbour's* center (the Te5 Role-3/4 idiom), and getting the frame (i's, not o's) and the `dx = wx[b]−wx[a]` sign right is the bulk of the risk. **De-risk by sequencing (the implementation order below); do not write K3 before the CPU refactor is proven F-NOOP and K1/K2 are bitwise-validated against `fd[c]`/`ed[c]`.**

## Implementation order (hand to the agent)
1. **Refactor `meam_bond_force_device` + `meam_getscreen_d_device` + `meam_dens_final_deriv` into `TDMD_HOST_DEVICE` single-source helpers in `meam.hpp`** — prove CPU 36/36 byte-identical (F-NOOP) FIRST. Load-bearing prerequisite.
2. **`MeamParamsView`** (POD + device spline upload) — unit-check `phi_spline`/`phip_spline` view == CPU on a grid.
3. **`zone_meam.cuh`** K1 → bitwise-check raw int64 vs CPU `fd[c]` on a window; K2 → check `ed[c]`; K3 (Roles A/B/C, `--fmad=false`).
4. **`meam_window_force_gpu.cuh`** — state (spline + 27-lane scratch + mutex), canonical-sort wrapper (+ `skip_sort` hook), firewall (verbatim), memset discipline, `GpuMeamRing` alias.
5. **`test_cuda_meam` G1–G8** → **`test_cuda_meam_ring` G9–G13** → `gpu_gate.sh` (memcheck/racecheck/initcheck).
6. **MEASURE** (G1 kMaxNbr realized, G3 sort verdict, ncu occupancy — expect latency-bound/low-occupancy like Te5) → record in `TD_MD_Core_Bench_v1_0.md §Me5`. Defer cells-culling (Me5b), device-resident D2D, multi-element — measure-first.

**Files:** NEW `include/tdmd/cuda/zone_meam.cuh`, `include/tdmd/cuda/meam_window_force_gpu.cuh`, `tests/test_cuda_meam.cu`, `tests/test_cuda_meam_ring.cu`. EDIT `include/tdmd/potentials/meam.hpp` (HOST_DEVICE annotations + static_asserts + the shared bond-force helper, F-NOOP on CPU). Mirrored line-precise: K3 ← `zone_tersoff.cuh:59–190`; pipeline ← `zone_eam.cuh:44–126`; wrapper ← `eam_window_force_gpu.cuh:228–420` + `tersoff_window_force_gpu.cuh:130–203`; firewall ← `meam_ring.hpp:153–177` (verbatim); CPU body transposed ← `meam_zone.hpp:97–365`; oracle gate ← `meam.hpp:811` (drop_class poison).