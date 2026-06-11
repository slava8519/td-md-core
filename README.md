# TD-MD Core

GPU molecular-dynamics engine based on the **Time Decomposition (TD)** method
(Andreev V.V. dissertation, 2007). See [CLAUDE.md](CLAUDE.md) for project
invariants and [`docs/`](docs/) for the full specification set; the milestone
plan is [`docs/TD_MD_Core_Roadmap_v1_0.md`](docs/TD_MD_Core_Roadmap_v1_0.md).

## Status — M0–M2.5 done, next M3 (2026-06-11)

CPU-only engine, no LAMMPS dependency; CUDA/HAL arrive at M3+:

- **M0** — end-to-end run: config → geometry → naive O(N²) Morse (PBC,
  min-image, energy shift) → velocity-Verlet NVE (FP64) → `.lammpstrj`.
- **M1** — zone finite-state machine (`ZoneFSM`, pure logic, INV-3).
- **M2** — causality buffer (eq. 33) + automatic time step (C1/K2/C3,
  eq. 62) + HALT/rescue dump on INV-4 violation (`config/config_auto.yaml`).
- **M2.5** — analytical GPU-occupancy probe (Tier-1, CUDA-free):
  [`docs/_meta/occupancy_probe_2026-06-06.md`](docs/_meta/occupancy_probe_2026-06-06.md).

Units are LAMMPS `metal` (eV, Å, amu, ps) — see
[`docs/TD_MD_Core_Units_v1_0.md`](docs/TD_MD_Core_Units_v1_0.md).
Dissertation formulas were extracted to
[`source/time_decomposition.md`](source/time_decomposition.md) and verified
against the code:
[`docs/_meta/FORMULA_VERIFICATION_2026-06-11.md`](docs/_meta/FORMULA_VERIFICATION_2026-06-11.md).

## Build & run

Requirements: C++20 compiler (GCC or Clang), CMake ≥ 3.20, Ninja, and network
access on first configure (CMake `FetchContent` pulls yaml-cpp 0.8.0 and
GoogleTest v1.15.2).

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# run the engine (writes traj.lammpstrj)
./build/tdmd config/config_m0.yaml

# run the acceptance tests
ctest --test-dir build --output-on-failure
```

## Tests (milestone acceptance)

- **Test_0_Step** (M0) — Morse forces and total PE vs the golden Al/Morse dataset
  (`reference_data/`); criterion `max|F_engine − F_golden| ≤ 1e-6` eV/Å (FP64)
  and `PE = 14.7286803884` eV. Self-contained, no LAMMPS.
- **Test_NVE_Drift** (M0) — 100 NVE steps, no NaN, bounded total-energy drift.
- **Test_Zone_FSM** (M1) — all legal/forbidden transitions, INV-3, S₁ seed,
  odd/even anti-deadlock I/O order.
- **Test_Buffer** (M2) — R_buf (eq. 33), causality check (INV-4), auto-dt
  C1/C3 semantics (eq. 62, exact), rescue dump on violation.
- **Test_Occupancy** (M2.5) — analytical occupancy model invariants.

## Layout

Per [`docs/TD_MD_Core_Skeleton_Interfaces_v1_0.md`](docs/TD_MD_Core_Skeleton_Interfaces_v1_0.md):
`include/tdmd/{core,potentials,io}`, `src/`, `tests/`, `config/`,
`reference_data/` (golden data), `docs/` (source of truth).
