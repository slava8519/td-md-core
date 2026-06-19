# MEAM-Si frozen golden — `pair_style meam run 0` (ENERGY, Me1)

The LOAD-BEARING external witness for the MEAM bond-order/screened potential (Me1, energy-only).
For MEAM this golden is MORE load-bearing than for Tersoff: MEAM's **energy itself** runs the
screening + partial-density VALUE algebra that the force (Me2) differentiates, so FD-of-energy is
independent of force-assembly bugs but NOT of a shared-value bug (wrong `C_ijk`, wrong `v3D`,
wrong `Γ`) — those corrupt energy AND FD identically. ⟹ **this LAMMPS golden is the ONLY
value-algebra witness in Me1.** CI is LAMMPS-free (`Test_MEAM_LAMMPS` reads these frozen files).

## Files
- `meam_si_64.data` — diamond-cubic Si, 64 atoms (2×2×2, a₀=5.431 Å), perturbed
  (`gen::make_diamond_si(2,2,2,5.431,0.20,seed=12345)` — the SAME config as Tersoff/SW). Fully
  periodic; box ≈ 10.862³ Å (> 2·rc = 8 Å, min-image unambiguous).
- `meam_si_64.energy` — total `PE` (eV) from `pair_style meam run 0`.
- `library.meam` — the LAMMPS global MEAM library (the Si 'dia' entry is the source of the
  hardcoded params); `Si.meam` — the single-element parameter file (`rc=4.0, delr=0.1`; the rest
  default: Cmin=2.0, Cmax=2.8, augt1=1, ialloy=0, ibar=1).
- `gen_energy.in` — the committed LAMMPS regeneration script.

## Provenance
- LAMMPS stable 22Jul2025 (MEAM package), `pair_style meam`, `pair_coeff * * library.meam Si
  Si.meam Si`. Si params (library.meam): lat=dia, Z=4, ielt=14, atwt=28.086, alpha=4.87,
  b0..3=4.8, alat=5.431, esub(Ec)=4.63, asub(A)=1.0, t0=1.0, t1=3.30, t2=5.105, t3=−0.80,
  rozero=1.0, ibar=1.

## Measured agreement (the gate's basis, NOT pre-registered)
- total PE: ours **−243.388597227171** vs LAMMPS −243.388597227171 (Δ ≈ 3e-13 eV — on this config
  `std::exp`/`std::log` happened to agree with LAMMPS's Cephes `fm_exp` to round-off). The gate is
  kept at 1e-4 for the cross-geometry `std::exp`-vs-`fm_exp` tolerance — NOT a bitwise-to-LAMMPS
  claim (the SW/Morse-CPU situation, the inverse of EAM's transcendental-free spline).
- **SCREENING (the `Mask_triplet`) is non-vacuous but BINARY on this fixture:** of 438 i<j pairs,
  128 have S=1, **310 have S=0** (2NN pairs sit nearly colinear behind a 1NN atom ⇒ C<Cmin ⇒ the
  i–j bond vanishes), 0 partial (0<S<1). Screening is load-bearing — the `poison` discriminator
  (S≡1, no k-product) diverges by ~17.9 eV. **The partial band (0<S<1) is absent on the diamond
  FAMILY not by perturbation size but because every perturb ≥ 0.40 that would populate it ALSO
  drives a bond below r_zbl (the ZBL guard throws).** So the partial branch is structurally dead
  on the diamond fixture (acceptance MUST-FIX — corrupting `fcut(C)` there changes nothing). It is
  exercised by `meam_tri3.data/.energy`: a free 3-atom cluster (i–j screened by k, S≈0.50, all
  bonds above r_zbl) with its own frozen LAMMPS golden (PE = −0.935624255785264), so
  `Test_MEAM_LAMMPS.PartialScreeningClusterMatchesLammps` validates the partial multiply against
  LAMMPS independently. The two MUST-FIXes (augt1 `t1_eff=2.82`; ebound
  `=Cmax²/(4(Cmax−1))`) are both load-bearing (G-AUGT1: flipping augt1 shifts PE > 0.5 eV).
- **SCOPE:** Me1 is ENERGY-ONLY (the ~470-line analytic force is Me2). Single-element Si, hardcoded
  params (no library parser); all other branches (msmeam, nn2/2NN, ialloy∈{1,2}, non-DIA lattices,
  multi-element) `throw` in `recompute()`. ZBL blend region (r < re(1−1/α)≈1.87 Å) guarded
  (the diamond's shortest bond 1.95 Å stays above it; a compressed config throws).
