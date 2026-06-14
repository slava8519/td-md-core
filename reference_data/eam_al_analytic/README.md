# Analytic Finnis–Sinclair EAM-Al fixture (M6 PR-E1)

A **self-contained, closed-form** EAM potential used by the M6 many-body
correctness ladder. It is **not** a validated Al potential — it is an analytic
fixture chosen so the force is the *exact* gradient of a C¹ energy, making the
finite-difference self-consistency test (`Test_EAM_FD`) unambiguous with no
spline/interpolation error. External validation against a real tabulated Al
potential (LAMMPS `pair_style eam/alloy`, `run 0`) is **PR-E7** (LAMMPS-gated).

## Form (units: LAMMPS `metal` — eV, Å)

```
E      = Σ_{i<j} φ(r_ij) + Σ_i F(ρ_i),   ρ_i = Σ_{j≠i} ρ_a(r_ij)
φ(r)   = D[e^{−2α(r−r0)} − 2 e^{−α(r−r0)}]      (Morse pair)
ρ_a(r) = e^{−β(r−r0ρ)}                          (exp-like electron density)
F(ρ)   = −A √ρ,   F'(ρ) = −A/(2√ρ)             (Finnis–Sinclair embedding)
```

Both `φ` and `ρ_a` are **force-shifted** at `rcut` (value *and* derivative go to
zero there), so the total energy is C¹ and the analytic force matches central
differences to FP precision.

## Default parameters (`potentials/eam_analytic.hpp` → `make_analytic_al<double>()`)

| symbol | value | meaning |
|---|---|---|
| `D`, `α`, `r0` | 0.29614, 1.11892, 3.29692 | Morse pair (dissertation Al) |
| `ρ_amp`, `β`, `r0ρ` | 1.0, 1.5, 3.29692 | electron density |
| `A` | 1.0 | embedding strength |
| `rcut` | 4.0 Å | cutoff (force-shifted) |

## Geometry

Reuses `../al_fcc_72.data` (72-atom FCC Al, PBC) — the same golden lattice as the
Morse reference. With `rcut = 4.0 Å` each atom sees exactly its 12 nearest
neighbours (the 2nd FCC shell sits just beyond the cutoff), so the per-atom
density is homogeneous `ρ_i = 12·ρ_a(r_nn)` — the analytic anchor for
`Test_EAM_Density_Sanity`.

## What it validates (no LAMMPS)

- `Test_EAM_FD` — analytic force ≡ −dU/dx (central diff, `max|F_an−F_fd| < 1e-5`).
- `Test_EAM_Density_Sanity` — `ρ_i` ≡ independent brute-force sum, and ≡
  coordination·`ρ_a(r_nn)` on the perfect lattice.
- `Test_EAM_Guard` — the load-time density-quantum guard (`density_fracbits()`)
  picks Q19.44 / Q23.40 / throws as `ρ_a` steepens.
- `Test_EAM_Potential_Interface` — `EamPotential : IManyBodyPotential` fixed-point
  passes match the FP64 oracle within the quantization bound.
