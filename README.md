# TD-MD Core

GPU molecular-dynamics engine based on the **Time Decomposition (TD)** method
(Andreev V.V. dissertation, 2007). See [CLAUDE.md](CLAUDE.md) for project
invariants and [`docs/`](docs/) for the full specification set; the milestone
plan is [`docs/TD_MD_Core_Roadmap_v1_0.md`](docs/TD_MD_Core_Roadmap_v1_0.md).

## Status — M0 (walking skeleton)

CPU-only end-to-end run: config → read geometry → naive O(N²) Morse (PBC,
min-image, energy shift) → velocity-Verlet NVE (FP64) → `.lammpstrj`. No GPU,
no TD pipeline, no LAMMPS dependency. CUDA/HAL arrive at M3+.

Units are LAMMPS `metal` (eV, Å, amu, ps) — see
[`docs/TD_MD_Core_Units_v1_0.md`](docs/TD_MD_Core_Units_v1_0.md).

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

## Tests (M0 acceptance)

- **Test_0_Step** — Morse forces and total PE vs the golden Al/Morse dataset
  (`reference_data/`); criterion `max|F_engine − F_golden| ≤ 1e-6` eV/Å (FP64)
  and `PE = 14.7286803884` eV. Self-contained, no LAMMPS.
- **Test_NVE_Drift** — 100 NVE steps, no NaN, bounded total-energy drift.

## Layout

Per [`docs/TD_MD_Core_Skeleton_Interfaces_v1_0.md`](docs/TD_MD_Core_Skeleton_Interfaces_v1_0.md):
`include/tdmd/{core,potentials,io}`, `src/`, `tests/`, `config/`,
`reference_data/` (golden data), `docs/` (source of truth).
