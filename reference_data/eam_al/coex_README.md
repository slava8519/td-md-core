# M6(c) — Al melting point T_m via two-phase coexistence (Al_zhou EAM)

Frozen-golden reference for the M6 physics-acceptance criterion (c): our NVE engine
reproduces LAMMPS's **two-phase solid–liquid coexistence** melting point for the
`Al_zhou.eam.alloy` setfl. Adversarial design: `_meta` workflow `wf_3a0aa75f-4f3`
(2026-06-18). CI stays LAMMPS-free at run time — only the frozen artifacts below are read.

## Result

| quantity | value |
|---|---|
| **T_m^LAMMPS** (production NVE plateau) | **642.3 K ± 1.6 K** |
| ⟨P⟩ at coexistence | ≈ 0.00 GPa (NPH-relaxed, couple-xy) |
| box (lateral-symmetric) | 33.39 × 33.39 × 100.58 Å |
| N | 6144 (8×8×24 FCC, interface ⟂ z) |
| solid fraction f_sol | ≈ 0.75 (liquid-lean coexistence, ~25 % liquid) |
| interface order param Φ=(S_sol−S_liq)/(S_sol+S_liq) | ≈ 0.78 (clear bimodal S(z)) |

**Honest caveat (G2, non-blocking):** experimental Al melts at **933.47 K**; this setfl
melts at **~642 K (−31 %)**. EAM-Al potentials usually *over*-estimate T_m; the Zhou-2001
generalized EAM database *under*-estimates it here. This is a property of the **potential**,
not the engine. The deliverable validates that **our NVE engine reproduces LAMMPS's coexistence
behavior for this exact setfl** (forces already match to ~1e-12 eV/Å, PR-E2) — NOT the
experimental Al melting point, and never bitwise == LAMMPS (Lyapunov + coordinate precision
dominate a chaotic melt; VALIDATION_EXPERIMENT §7).

## Provenance (how the frozen config was built)

Coexistence prep is finicky: an existing s/l interface has **no nucleation barrier**, so any
thermostat above T_m sweeps the solid away (homogeneous one-phase melting needs ~20 % superheat;
an interface needs none). Two earlier recipes failed by exactly this — both went all-liquid:
- `coex_build.in` (v1): bare `aniso` NPT at fixed 1000 K > T_m → solid melted; also broke x↔y symmetry.
- `coex_build_v2.in` (v2): couple-xy + NPH, but NVT-thermostatted the two-phase system at 950 K > T_m first → melted.

Working recipe (two stages):
1. **`coex_bracket.in -var Tp 900 -var tag t900`** → `coex_t900.data`. Freeze the solid half
   (zero velocity + `setforce 0`) so it *cannot* melt while the lower half is blasted to 1800 K;
   quench the liquid; release; NPH (couple-xy, T floats). A T_m bracket over Tp∈{700,800,900}
   showed Tp=700/800 fully freeze (all-solid) and **Tp=900 yields a genuine bimodal interface**
   — pinning T_m to the Tp≈900 energy window.
2. **`coex_refine.in`** reads `coex_t900.data` and runs a long **NPH** (couple-xy, P=0, T floats):
   for a two-phase system NPH self-regulates T → T_m independent of the prep, erasing the
   perfect-lattice cooling artifact. Then freezes the box and **NVE-produces** the reference,
   writing `coex_al_6144.data` + `Tm_lammps.txt`. The interface **survived the full 200 ps**.

## Frozen artifacts (committed)
- `coex_al_6144.data` — the equilibrated two-phase launch state (positions+velocities, P≈0, 642.3 K).
- `Tm_lammps.txt` — production NVE temperature time series → T_m^LAMMPS = 642.3 ± 1.6 K.
- `coex_lammps_zprofile.txt` — z-resolved density + per-atom PE (interface baseline).
- `coex_bracket.in`, `coex_refine.in` — the reproducible LAMMPS recipe (run from an ASCII cwd).

## Our-engine validation (`tools/eam_coexist.cu`, RTX 5080)

Pre-registered falsifiable gates (frozen before the run; see the tool header):
- **G1 (primary)** |T_m^ours − T_m^LAMMPS| ≤ 25 K — our NVE reproduces LAMMPS on this setfl.
- **G1b** |T_m^ours − T_m^reseed| ≤ 10 K — the plateau is an attractor, not an inherited coast.
- **G3** Φ>0.5 AND f_sol∈[0.20,0.80] every sample — a real interface (not full-melt/full-freeze).
- **G4** max|ΔE|/|E0| ≤ 1e-6 ; **G5** |Δp_cm| ≤ 1e-9 ; **G6** |⟨P⟩| ≤ 0.05 GPa.
- **G2 (non-blocking)** vs experimental 933.47 K — reported, never gating (see caveat above).

**Result (RTX 5080, 40 ps production + 20 ps reseed, dt=1 fs):**

| gate | measured | band | verdict |
|---|---|---|---|
| **G1** T_m^ours vs LAMMPS | 636.6 K vs 642.3 K → **5.7 K** | ≤ 25 K | **PASS** |
| **G1b** reseed attractor | 639.5 K → 2.9 K | ≤ 10 K | **PASS** |
| **G3** interface | Φ_min 0.717, f_sol∈[0.67,0.79], bimodal every sample | Φ>0.5, f_sol∈[0.20,0.80] | **PASS** |
| **G4** energy | max|ΔE|/E0 = 2.0e-7 | ≤ 1e-6 | **PASS** |
| **G5** momentum | |Δp_cm| = 1.7e-10 | ≤ 1e-9 | **PASS** |
| **G6** pressure | ⟨P⟩ = +0.004 GPa | ≤ 0.05 GPa | **PASS** |
| G2 (non-blocking) | −297 K (−32 %) vs exp 933.5 K | reported | setfl underestimates |

**M6(c) engine-vs-LAMMPS gates: ALL PASS ✓** — our NVE engine reproduces LAMMPS's
two-phase coexistence melting point for the Al_zhou setfl (Δ=5.7 K), with a stable
interface held through the run, energy conserved to 2e-7, momentum to 1.7e-10, and the
coexistence pinned at P≈0. The reseed replica (velocities re-drawn at the plateau T from
an independent Maxwell microstate) returns to the same T_m, proving the plateau is a
microcanonical attractor — not a trajectory coasting on LAMMPS's prepared state.
