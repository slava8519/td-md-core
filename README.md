# TD-MD Core

GPU molecular-dynamics engine based on the **Time Decomposition (TD)** method
(Andreev V.V. dissertation, 2007): a ring of asynchronous nodes computes
*successive time steps* concurrently, with causality guaranteed by a buffer
layer (eq. 33) and bitwise determinism guaranteed by int64 fixed-point force
accumulation (B1 ≡ AMBER SPFP). See [CLAUDE.md](CLAUDE.md) for project
invariants and the full milestone chronicle, [`docs/`](docs/) for the
specification set; the milestone plan is
[`docs/TD_MD_Core_Roadmap_v1_0.md`](docs/TD_MD_Core_Roadmap_v1_0.md).

## Status — M0–M7 done (M5b awaits ≥2 GPUs) · 2026-07-02

- **Ring engine** (M3.5–M5a): CPU/GPU/MPI time-conveyor rings, all **bitwise**
  — run-to-run, 1-vs-z nodes, MPI-vs-single-process, and CPU↔GPU on LJ in the
  `--fmad=false` verify builds (Morse CPU↔GPU is tolerance-class: libm-vs-CUDA
  exp ulps); `production_mixed` (int32 fixed offsets + FP32 pair math) bitwise
  too. Dissertation §3.6 replica (Al-72, 25 900 auto-dt steps, 1 vs 4 nodes):
  zero deviation, on CPU, GPU and MPI rings.
- **Physics suite** (M6): pairwise Morse (FD-checked golden) / LJ (NIST
  SRSW-validated) + full many-body ladders — **EAM** (spline setfl ≡ LAMMPS
  ~1e-12), **SW**, **Tersoff**, **MEAM** — each end-to-end: CPU force core →
  serial zone → threaded TD ring → NVE → GPU kernels → frozen-LAMMPS golden
  forces + RDF. Two-phase coexistence melting point vs LAMMPS: ΔT_m = 5.7 K.
- **Performance** (RTX 5080, fp64): pairwise LJ flagship @10⁶ atoms, live GPU
  ring — **2.06e7 atom-steps/s** (2.50e7 mixed+FMA). Angular @10⁶ — *isolated
  kernels, z=1/free-z, NOT live-ring throughput, idle-GPU protocol*: SW 2.60e6 ·
  Tersoff 9.15e5 · MEAM 1.77e5 (FP64-transcendental-bound, measured). Cell-list
  culling default-on everywhere (live-ring R up to 17.5×). Numbers + method:
  [`docs/TD_MD_Core_Bench_v1_0.md`](docs/TD_MD_Core_Bench_v1_0.md).
- **DevOps** (M7): cloud CI (build+test, -Werror lint, CUDA compile-only),
  Apptainer/Docker/Spack under [`deploy/`](deploy/), live ANSI dashboard
  (`--dashboard`) with a bitwise-observational ring hook (gate A7).
- **M5b** (multi-GPU cluster) is a deferred/parallel milestone — external
  trigger: a node with ≥2 GPUs.

**CLI scope (recorded decision, 2026-07-02):** the `tdmd` binary is the
*pairwise demo* (Morse/LJ — direct, cluster, CPU reference ring). The many-body
suite and the GPU/MPI rings run via tests and `tools/` harnesses; wiring them
into the CLI is a separate track. See
[`docs/_meta/AUDIT_W_PHASE_NEIGHBORS_2026-07-02.md`](docs/_meta/AUDIT_W_PHASE_NEIGHBORS_2026-07-02.md) §7.1.

Units are LAMMPS `metal` (eV, Å, amu, ps) — see
[`docs/TD_MD_Core_Units_v1_0.md`](docs/TD_MD_Core_Units_v1_0.md).
Dissertation formulas were extracted to
[`source/time_decomposition.md`](source/time_decomposition.md) and verified
against the code:
[`docs/_meta/FORMULA_VERIFICATION_2026-06-11.md`](docs/_meta/FORMULA_VERIFICATION_2026-06-11.md).

## Build & run

Requirements: C++20 compiler (GCC or Clang), CMake ≥ 3.20, Ninja, and network
access on first configure (CMake `FetchContent` pulls yaml-cpp 0.8.0 and
GoogleTest v1.15.2). **No LAMMPS dependency** — golden references are frozen
under `reference_data/` (the LAMMPS-derived ones — eam_al/sw_si/tersoff_si/
meam_si — each carry a committed `gen_*.in` regeneration script; nist_lj/ is
NIST SRSW data verbatim, the Al/Morse golden regenerates via its own scripts).

```bash
# CPU build + the full CPU test suite (~41 tests; this is what cloud CI runs)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure

# run the engine (pairwise demo; writes traj.lammpstrj)
./build/tdmd config/config_m0.yaml
./build/tdmd config/config_ring.yaml --dashboard   # TD ring + live dashboard

# CUDA build (RTX 5080 / sm_120 default; override CMAKE_CUDA_ARCHITECTURES)
cmake -S . -B build-cuda -G Ninja -DTDMD_WITH_CUDA=ON
cmake --build build-cuda
./scripts/gpu_gate.sh build-cuda    # ctest -L cuda + compute-sanitizer

# MPI ring tests (needs CUDA + MPI)
cmake -S . -B build-cuda -G Ninja -DTDMD_WITH_CUDA=ON -DTDMD_WITH_MPI=ON
cmake --build build-cuda && ctest --test-dir build-cuda -L mpi
```

CMake options: `TDMD_BUILD_TESTS` (ON), `TDMD_WITH_CUDA` (OFF),
`TDMD_WITH_MPI` (OFF), `TDMD_WITH_NVTX` (ON). HPC container images:
[`deploy/README.md`](deploy/README.md).

## Tests

Self-contained (no LAMMPS, no GPU needed for the CPU suite): golden Al/Morse
forces (`Test_0_Step` ≤1e-6 eV/Å), NIST SRSW LJ energies/virials, frozen
LAMMPS goldens for EAM/SW/Tersoff/MEAM (forces ~1e-9…1e-12 + time-averaged
RDF overlap ≥0.9998), bitwise determinism gates (run-to-run, 1-vs-z,
zone-vs-monolith, ring-vs-serial), independent FP64 oracles per many-body
potential, NVE energy/momentum conservation, HALT/rescue paths. GPU tests
carry the `cuda` label and are gated locally via `scripts/gpu_gate.sh`
(memcheck+racecheck); cloud CI compiles CUDA but never executes it.

## Layout

`include/tdmd/{core,potentials,cuda,io,cli,gen,hal,metrics,mpi,probe}` +
`include/tdmd/units.hpp`,
`src/`, `tests/`, `tools/` (benches + physics harnesses), `config/`,
`reference_data/` (frozen goldens), `deploy/` (containers), `docs/` (specs;
`docs/_meta/` — decision journal). As-built map:
[`docs/TD_MD_Core_Skeleton_Interfaces_v1_0.md`](docs/TD_MD_Core_Skeleton_Interfaces_v1_0.md).
