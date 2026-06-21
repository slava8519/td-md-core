All anchors verified against the live tree. `PassStats` has exactly `{pe, ke, dt, v_max, a_max, k2cap}`; the rescue dump is already wired into both HALT paths (main.cpp:201, :274); the repo is public (`slava8519/td-md-core`) confirming the self-hosted-runner security constraint. The three lenses agree on substance; minor divergences (Containerfile naming `deploy/docker/Containerfile` vs `deploy/Apptainer.def`; whether container-build runs in CI) I'll resolve in the synthesis. Here is the plan.

---

# M7 (UX / I/O / DevOps / HPC-deploy) — Implementation Plan, TD-MD Core (branch `opus`)

**M7 criterion (Roadmap):** *"CI зелёный на чистой машине (без LAMMPS — golden-тесты); деплой контейнера на кластере проходит smoke-тест."* Two halves: **(A)** a *full* cloud CI that is green without LAMMPS and without a GPU; **(B)** a container that builds and passes an in-container smoke test. The user's explicit ask: **CI / containers / dashboard**.

**The orchestrator has one RTX 5080 and no cluster.** Every item below is tagged **NOW** (validatable on the single box / cloud today) or **LATER** (documented, runs at the M5b external trigger — same gate-rule exception as M5b in the Roadmap). I never claim "cluster-ready."

**Three load-bearing conventions, honored everywhere:** (1) cloud = CPU tests + compile-only CUDA; **no GPU, no LAMMPS in the cloud**; (2) the async ring's **bitwise determinism (INV-9) is sacred** — any dashboard hook is observational-only; (3) the repo is **public** ⇒ a self-hosted GPU runner is `workflow_dispatch`/push-to-`main` **only**, never `pull_request`.

---

## PART 1 — The full CI (`.github/workflows/ci.yml`)

Replace the current single-job file (`on: [push, pull_request]`, one `build-test` matrix, no caching, no `-LE cuda`, no CUDA/lint jobs) with **four jobs**. Jobs 1–3 run on GitHub-hosted runners (cloud-safe, CPU-only / compile-only). Job 4 is **defined but inert** (the documented self-hosted gate).

Pin `runs-on: ubuntu-24.04` (not `-latest`) to match the container base and stop silent image drift. Scope triggers and add concurrency:

```yaml
name: ci
on:
  push:        { branches: [main, opus] }
  pull_request:
  workflow_dispatch:
concurrency: { group: ci-${{ github.ref }}, cancel-in-progress: true }
```

### Job 1 — `build-test` (CPU; **this is the M7 criterion**)  — NOW
- Keep matrix `cxx: [g++, clang++]`, `fail-fast: false`.
- **apt:** `ninja-build ccache`. **Do NOT** install `libyaml-cpp-dev`/`libgtest-dev` — they are FetchContent-pinned on purpose (CMakeLists:43/205, yaml-cpp 0.8.0, googletest v1.15.2).
- **Caching** (the dominant wall-time win — every cold run rebuilds yaml-cpp + googletest):
  ```yaml
  - uses: actions/cache@v4
    with:
      path: |
        build/_deps
        ~/.cache/ccache
      key: deps-${{ matrix.cxx }}-${{ hashFiles('CMakeLists.txt') }}
      restore-keys: deps-${{ matrix.cxx }}-
  ```
  A `CMakeLists.txt` hash key auto-busts on a tag bump. Add `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache` — header-heavy C++20 (every `test_*` re-instantiates `TimeConveyor`/`run_simulation`) makes ccache a large win across the ~40 CPU test targets.
- **Configure:** `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release` (CUDA stays OFF by default — CMakeLists:20).
- **Test:** `ctest --test-dir build --output-on-failure -LE cuda`. The **`-LE cuda`** is a defensive belt: with `TDMD_WITH_CUDA=OFF` no `cuda`-labeled test is even registered (CMakeLists:381 guards them), so it's a no-op today — but it documents intent and survives a future default flip, guaranteeing the cloud never tries a GPU kernel. The two long tests carry ctest `TIMEOUT 300/600` (CMakeLists:373/379) ⇒ a ring deadlock fails deterministically instead of hanging the runner.

### Job 2 — `lint` — NOW (low-false-positive scope)
- **`clang-format --dry-run --Werror`** gated on a committed `.clang-format` (LLVM base, `IndentWidth: 2`, `ColumnLimit: 80` — matches the house style). **Ship a one-time `clang-format -i` normalization commit so the gate starts green on day one.**
- **Warnings-as-errors smoke:** reconfigure with `-DCMAKE_CXX_FLAGS=-Werror` and build only the core lib + CLI + one representative test (`-Wall -Wextra` is already on — CMakeLists:64/71/77 — this just promotes them). Highest-value, lowest-noise static check with zero new config.
- **NOT a blocking gate:** clang-tidy / cppcheck. On this codebase (transitive `.cu` headers, 580-line `meam.hpp`, the FP-determinism `int64`/`rint` idioms) they are slow and noisy without a curated `.clang-tidy`. Add later as `continue-on-error: true` advisory if wanted.

### Job 3 — `cuda-compile` (compile-only; **the convention's load-bearing job**) — NOW
Run inside the CUDA toolkit container on a normal (GPU-less) cloud runner:
```yaml
cuda-compile:
  runs-on: ubuntu-24.04
  container: nvidia/cuda:13.1.0-devel-ubuntu24.04
  strategy: { matrix: { mpi: [OFF, ON] } }
  steps:
    - uses: actions/checkout@v4
    - run: apt-get update && apt-get install -y cmake ninja-build git ca-certificates g++ libopenmpi-dev openmpi-bin
    - run: cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release
             -DTDMD_WITH_CUDA=ON -DTDMD_WITH_MPI=${{ matrix.mpi }}
             -DCMAKE_CUDA_ARCHITECTURES=70 -DTDMD_BUILD_TESTS=OFF
    - run: cmake --build build-cuda     # COMPILE ONLY — never ctest (no GPU)
```
- **Arch override `=70`, decisive:** CMakeLists:30 hardcodes `120-real` (5080 SASS, needs CUDA ≥12.8), but CMakeLists:29 honors a pre-set `CMAKE_CUDA_ARCHITECTURES` ⇒ `-DCMAKE_CUDA_ARCHITECTURES=70` is respected with **zero source change**. CI only needs nvcc to *compile*, not to emit 5080 SASS — this decouples the job from sm_120/CUDA-version availability, the single biggest CI fragility (see risks).
- **MPI is a matrix axis, not a 4th job:** `TDMD_WITH_MPI` nests inside the CUDA block (CMakeLists:520, `find_package(MPI)`), so `ON` adds `libopenmpi-dev` (OpenMPI 4.1.6, not CUDA-aware — matches `MpiRingEdge` host-staging). Still compile-only: `test_mpi_conveyor.cu` is `cuda;mpi`-labeled (CMakeLists:532), never run.
- **Explicitly NO `ctest`, NO `scripts/gpu_gate.sh`** here (gpu_gate.sh:4 says "NOT a cloud-CI job"; it `exit 2`s without a GPU and invokes `compute-sanitizer`).

### Job 4 — `gpu` (the self-hosted gate) — **DEFINED-BUT-INERT / LATER**
```yaml
gpu:
  if: github.event_name == 'workflow_dispatch' ||
      (github.event_name == 'push' && github.ref == 'refs/heads/main')
  runs-on: [self-hosted, gpu]
  steps:
    - uses: actions/checkout@v4
    - run: cmake -S . -B build-cuda -G Ninja -DTDMD_WITH_CUDA=ON && cmake --build build-cuda
    - run: ./scripts/gpu_gate.sh build-cuda    # ctest -L cuda + compute-sanitizer, verbatim
```
The `if:` is the **security control the convention demands** — no `pull_request` path reaches it, so untrusted fork code never executes on the dev box (the repo is public, confirmed `slava8519/td-md-core`). Native arch `120-real` here (real 5080). Inert until the user registers a `[self-hosted, gpu]` runner.

### What CI explicitly does NOT do (state in a comment block in `ci.yml`)
- No `cuda`-labeled ctest in the cloud (no GPU). No `scripts/gpu_gate.sh` in the cloud (needs device + compute-sanitizer).
- No `-DWITH_LAMMPS=ON` — golden tests are frozen in `reference_data/`, default build is LAMMPS-free. **This is precisely what makes "CI зелёный без LAMMPS" satisfiable today.**
- No apt `libyaml-cpp-dev`/`libgtest-dev` (FetchContent-pinned).
- **No container-build job in routine CI** (resolving the cross-lens disagreement — see Part 2 §smoke).

---

## PART 2 — Containers (`deploy/`)

Smallest-robust: **one Apptainer def + one Containerfile sharing one cmake invocation + one Spack env + a README**. Base image `nvidia/cuda:13.1.0-devel-ubuntu24.04` (devel ⇒ `nvcc`). **Verify the exact tag resolves on Docker Hub before committing** (NVIDIA's scheme is `cuda:13.1.x-devel-ubuntuXX.YY`; a wrong tag fails the whole build — top container risk).

### `deploy/apptainer/tdmd.def` — NOW (CPU smoke) + LATER (GPU/MPI)
- `%post`: `apt install cmake ninja-build git ca-certificates g++ libopenmpi-dev openmpi-bin`; `git clone --branch opus --depth 1`; `cmake -DTDMD_WITH_CUDA=ON -DTDMD_WITH_MPI=ON` (default arch `120-real` from CMakeLists:30; override via `TDMD_CUDA_ARCH` env for other clusters — A100 `80-real`, H100 `90-real`); `cmake --build`. yaml-cpp + googletest are FetchContent ⇒ `%post` needs network (Apptainer `%post` has it).
- `%runscript`: `exec /opt/tdmd/build/tdmd "${@:-/opt/tdmd/share/smoke.yaml}"`.
- **`%test` = the in-container smoke, runs at `apptainer build` time on the GPU-less build host ⇒ CPU-only:**
  ```sh
  ctest --test-dir build --output-on-failure -LE cuda   # golden-data, LAMMPS-free, label-excludes GPU
  ./build/tdmd share/smoke.yaml                          # the engine runs a step on a tiny golden config
  ```
  `-LE cuda` is the **inverse of gpu_gate.sh:16's `-L cuda`** — mandatory because the build host has no device.
- `%help`: documents `apptainer run --nv tdmd.sif config/your.yaml` (single-GPU), the `apptainer exec --nv tdmd.sif ctest -L cuda` GPU gate, the `TDMD_CUDA_ARCH` override, and the **multi-node** `mpirun -np N apptainer exec ... tdmd config/ring.yaml` recipe (LATER, needs ≥2 GPU = M5b). **Do NOT put `compute-sanitizer` in `%test`** — that's the dev-machine gate, not a deploy smoke.

### `deploy/docker/Containerfile` — NOW
Docker/podman twin (multi-stage). The one real difference from the def: **`COPY . /opt/tdmd`** (builds the PR's checkout) vs the def's **`git clone <tag>`** (reproducible cluster deploy) — both intentional. Bakes the same CPU smoke as a `RUN ctest ... -LE cuda && ./build/tdmd config/config_m0.yaml`. `ENTRYPOINT ["/opt/tdmd/build/tdmd"]`.

### `deploy/spack/spack.yaml` — NOW (`concretize`) / LATER (`install`)
Deliberately *short* — Spack provides only the **toolchain + system libs**, NOT the FetchContent deps (yaml-cpp/googletest stay pinned in CMakeLists, one source of version truth, no skew):
```yaml
spack:
  specs: [cmake@3.27:, ninja, cuda@13.1, "openmpi@4.1.6 ~cuda", git, gcc@13.3]
  concretizer: { unify: true }
```
**Why it exists:** the no-privileged-build / air-gapped escape hatch — on a real cluster you often can't run a privileged container build, and the **site** OpenMPI/UCX must win so MPI traverses the fabric (`~cuda` mirrors the "OpenMPI 4.1.6, NOT CUDA-aware → host-staging" constraint). `spack concretize` runs anywhere NOW; an actual `spack install` against a cluster's compilers is LATER.

### `deploy/README.md` — the validatable-now vs LATER table + all build/run/gpu-gate/multi-node commands.

### The smoke test + "can CI build the container?" — **RESOLVED: gate it, don't run on every PR**
The two lenses split here; the synthesis: **the in-container CPU smoke is the M7 deliverable and runs at `apptainer build` time** (NOW, single box). A **CI** container build is heavy (the CUDA-devel base is ~4–7 GB; the angular `.cu` TUs are slow to nvcc) and would dominate cloud wall-time — so it is **not** a routine-PR job. Add it only as an **optional gated job** (`if: workflow_dispatch || ref==main`) building the **Containerfile** (compile-only, nvcc present, no GPU run). The GPU-run half of the smoke stays on the self-hosted 5080 / documented. This keeps routine CI at CPU + compile-only CUDA, matching the existing `ci.yml`'s deliberate CPU-only posture.

---

## PART 3 — The live ANSI dashboard (the hard one)

### (a) The hook — observational callback in `ConveyorOptions`, fired post-reduce at the single stats-write point
The blocking-call constraint is **load-bearing and cannot be worked around**: `run_conveyor` needs all `o_.steps` passes in one invocation (the Λ-chain `n−1` light-cone pre-history + pass-order rotation are built across the whole horizon — chunking would reset the ring). So progress is **pulled out of the live jthreads** at the one natural seam: each node writes its per-pass record exactly once at **`conveyor.hpp:551`**, *after* the B1 int64 reduce.

**Add to `ConveyorOptions` (conveyor.hpp:91), empty-defaulted ⇒ every existing call site is byte-unchanged:**
```cpp
// M7 dashboard hook (OBSERVATIONAL ONLY). Fired exactly once per completed pass,
// from the node thread that wrote stats[h-1], AFTER the B1 int64 reduce. It reads
// a finished PassStats; it MUST NOT touch atoms/forces/dt. z node threads fire
// concurrently ⇒ the impl must be thread-safe (the shipped ProgressMonitor locks).
// Empty by default ⇒ zero overhead, bitwise no-op beyond one `if (on_pass)`.
std::function<void(long pass_h, const PassStats& st)> on_pass{};
```
**Fire at conveyor.hpp:551:**
```cpp
res_.stats[std::size_t(h - 1)] = {pass_pe, ke, dt, agg.v, agg.a, agg.k2cap};
if (o_.on_pass) o_.on_pass(h, res_.stats[std::size_t(h - 1)]);
```
**Determinism-safe by construction (the #1 M7 risk, mitigated):** the callback receives a `const PassStats&` + `long`, reads finished scalars, and never enters `do_pair`/`end_zone`/the FixedAccum reduce/`auto_dt`. The force order, int64 accumulation order, and Λ-chain are untouched ⇒ INV-9 (bitwise 1-vs-z, run-to-run) preserved. This is the *exact* discipline of the existing `TDMD_EAM_RING_TIMERS` per-phase timer (CMakeLists:186) — feature-gated observation that leaves the proven hot path byte-identical. The `run_simulation` direct path feeds the **same** monitor via its existing `on_frame` hook (simulation.hpp:46).

**Rejected alternatives:** (ii) a bare polling `atomic<long> passes_done` loses the per-pass scalars and still needs a mutex-protected snapshot → converges back to the callback; (iii) simulation-only would leave the dashboard **dark for the actual product** (the dissertation's headline path is the ring).

### (b) The renderer — CUDA-free, no ncurses, split for testability
```
include/tdmd/cli/dashboard.hpp        # Snapshot + Dashboard (TTY-aware, pure formatter)
src/cli/dashboard.cpp                  # ANSI rendering + plain-log fallback
include/tdmd/cli/progress_monitor.hpp  # on_pass→snapshot adapter + ~10 Hz render thread
```
**Surfaced (all already computed — no new physics):** progress `pass_h/steps`; **E(t) + NVE drift** `(E−e0)/|e0|` (from `pe+ke` vs `ConveyorResult::e0`, the headline invariant); **dt evolution** (`PassStats::dt`, surfaces auto-step/Λ-chain); **T** = `2·ke/(n_dof·kB)` (`n_dof=dof_thermal(N)`, thermal.hpp:62); **v_max** (`PassStats::v_max`); **the causality-buffer headroom** `v_max·dt / R_buf` (buffer.hpp:33/38 — the most TD-distinctive readout; `<1` healthy, `→1` an imminent Causality HALT; no other MD engine shows this); zones/nodes; **HALT status** (green running → red `halt_msg`); an **E(t) sparkline** (`▁▂▃▄▅▆▇█`, last ~60 snapshots).
**Concurrency discipline:** `on_pass` does the absolute minimum under one lock — copy scalars into `LatestSnapshot`, bump `atomic<long> max_pass`, push one float into a fixed-size sparkline ring — then returns. **Rendering (formatting, ANSI) happens on the separate monitor thread at ~10 Hz, never inside `on_pass`.** ANSI: `\033[H` home + per-line `\033[K` clear-to-eol (no full-clear flicker), 8/16-color SGR. No ncurses (keeps the minimal-dep posture).

### (c) TTY detection — non-TTY degrades to plain append-only log
```cpp
const bool tty = ::isatty(fileno(stdout)) && std::getenv("TERM")
              && !std::getenv("CI") && std::getenv("NO_COLOR") == nullptr;
```
TTY → live in-place ANSI. Pipe/file/CI/Slurm → one plain line every N passes (`pass 12000  E=…  rel=…  dt=…  T=…  Rbuf_hr=0.43`) — exactly what CI logs/`tee` want; the dashboard's final frame **subsumes** the existing `printf` summary (main.cpp:209–214). Auto-disables on non-TTY inside a container/batch job ⇒ no ANSI garbage in Slurm logs. CLI gets a `--dashboard` flag (ANSI to **stderr** so stdout stays the parseable summary).

### (d) The unit test — renderer tested with NO real run (`tests/test_dashboard.cpp` → `Test_Dashboard`, CPU, cloud-CI-able)
`Dashboard::render_plain(pass, stats)` is a **pure function of a Snapshot** (no terminal/thread/timing). Tests feed a synthetic deterministic `PassStats` stream and assert exact substrings: progress `50/100`, `E=…`, `rel=6.667e-02`, the T and Rbuf-headroom strings, a HALT-line red marker, sparkline `▁..█` ordering, and a `ProgressMonitor` TSan test hammering `on_pass` from z threads (covers thread-safety). Register via `add_test`.

---

## PART 4 — Acceptance criterion (what proves M7)

| # | Proof | Validatable |
|---|---|---|
| A1 | **Cloud CI green on a clean machine, no LAMMPS, no GPU.** Mirror locally: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build -LE cuda --output-on-failure` → all ~40 CPU tests pass. | **NOW** |
| A2 | `cuda-compile` job: `-DTDMD_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70` (±MPI) **compiles** in the CUDA container; no ctest. | **NOW** (cloud) |
| A3 | `lint`: `.clang-format` committed + repo `clang-format -i`-normalized ⇒ format gate green; `-Werror` build of core+CLI green. | **NOW** |
| A4 | **Container builds + in-container CPU smoke passes:** `apptainer build tdmd.sif deploy/apptainer/tdmd.def` (the `%test` `ctest -LE cuda` + `tdmd share/smoke.yaml` runs at build time, exit 0). | **NOW** (single box, has CUDA 13.1) |
| A5 | **GPU gate inside the container:** `apptainer exec --nv tdmd.sif ctest -L cuda` passes on the 5080. | **NOW** (single 5080) |
| A6 | **Dashboard renders + `Test_Dashboard` green** (synthetic stream, plain + HALT + sparkline + TSan). | **NOW** |
| A7 | **THE determinism gate:** `Test_Conveyor` "1-vs-z" + a 50k `Test_NVE_Invariant` run **with a non-trivial `on_pass` attached** are **bitwise-identical** to the detached run. This is the load-bearing assertion that lets the dashboard ship without touching the sacred ring. | **NOW** |
| A8 | Multi-node MPI ring across cluster nodes; `spack install` against a site toolchain. | **LATER** (M5b trigger) |

M7 is **DONE** when A1–A7 are green. A8 is documented, runs at the M5b external trigger (Roadmap gate-rule exception, identical class to M5b).

---

## PART 5 — Scope cut

- **Rescue dump: IN scope, already exists** — `write_rescue_xyz` is wired into both HALT paths (main.cpp:201 ring, :274 direct). M7 *delta* is tiny: the dashboard's HALT frame prints the rescue path. The **in-flight mid-ring dump** stays **deferred (documented)** — no consistent mid-ring geometry exists (zones of several passes in flight; conveyor.hpp:148–152), so a GPU-ring CUDA-error surfaces as `Halt::Internal` with **t0 state**. That's an honest, already-documented limitation, not new work.
- **Async-I/O (background writer thread): DEFER to a 2nd M7 sub-PR.** Today `TrajectoryWriter::write_frame` reopens the file in append mode every frame (synchronous; fine for the direct stepper's coarse `traj_every`; the ring path doesn't write trajectories at all — main.cpp:142). It has real value but is **orthogonal to the dashboard** and carries its own correctness surface (ordering, backpressure, flush-on-HALT). Land the dashboard first (self-contained, the headline ask); spec async-I/O separately as a single dedicated writer `jthread` consuming a bounded frame queue with deterministic flush-on-HALT. **Do not entangle it with the determinism-sensitive ring hook.**
- **Cluster smoke + multi-node MPI + `spack install`: documented-not-run** (M5b).
- **clang-tidy/cppcheck: not a gate** (advisory-only if added).
- **Container build in routine CI: cut** — gated/dispatch-only (heavy; routine CI stays CPU + compile-only CUDA).

---

## PART 6 — Risks (ranked) + what to MEASURE

1. **[HIGHEST] Determinism regression from the dashboard hook.** Green tests stay green (they run headless) ⇒ a silent break is invisible — exactly the project's standing "deterministic-but-wrong is invisible" rule. **Mitigation:** read-only callback, fired *after* the B1 reduce, empty-by-default. **MEASURE/gate (A7):** re-run `Test_Conveyor` 1-vs-z + `Test_NVE_Invariant` 50k **with** a non-trivial `on_pass` live → bits must be identical to detached. If they differ, the hook touched state. **This single assertion is the gate that lets M7 ship.** Also assert wall-time delta is within noise (a few %) to catch lock-contention regressions.
2. **[HIGH] The CUDA-toolkit CI coupling (sm_120 ↔ CUDA version).** `Jimver/cuda-toolkit` / the CUDA-devel image and `ubuntu-latest`'s gcc drift; a too-new host gcc → nvcc "unsupported GNU version"; `120-real` needs CUDA ≥12.8 so you can't pin an old stable CUDA. A green CI silently breaks on image roll. **Mitigation:** `-DCMAKE_CUDA_ARCHITECTURES=70` in CI (decouples from sm_120), pin `ubuntu-24.04` + the action/CUDA versions, keep `cuda-compile` required-but-isolated so its breakage never blocks the CPU/physics gates. **MEASURE:** print the nvcc version in the log (canary for silent toolkit drift); per-job wall time; `build/_deps` cache hit-rate.
3. **[MEDIUM] Container base-tag + driver/runtime skew.** (a) the exact `13.1.0-devel-ubuntu24.04` tag must exist on Docker Hub — verify before committing the def. (b) `sm_120` SASS built in the container needs the deploy node's driver to support CUDA 13.1; `--nv` injects the host driver, so an older cluster driver won't load the `120-real` cubin. **MEASURE:** `apptainer build` locally (tag resolves?) then `apptainer exec --nv tdmd.sif ctest -L cuda` on the 5080 (host-driver-injected runtime loads the cubin?).
4. **[MEDIUM] FetchContent network at `%post` / configure.** Air-gapped HPC build hosts fail `git clone yaml-cpp/googletest`. **Mitigation:** the Spack-mirror path / vendor the two deps / `-DFETCHCONTENT_SOURCE_DIR_*` overrides. **MEASURE:** build with `--no-net` to confirm it fails *loudly*, not silently.
5. **[LOW] ANSI in a non-TTY.** Mitigation: `isatty` + `CI`/`NO_COLOR` gate → plain-log fallback; tested via the `tty=false` `render_plain` path.
6. **[LOW] Concurrent `on_pass` thread-safety from z nodes.** Mitigation: minimal-work-under-lock snapshot, render off-thread. Gate: the `ProgressMonitor` TSan test; run the dashboard test build under TSan in CI (cheap, CPU-only).

**THE single biggest risk is #1, the dashboard determinism hook** — fully mitigated *by construction* (read-only, post-B1-reduce, empty-by-default) and *proven* by the A7 bitwise net. Among DevOps items, #2 (the CUDA-toolkit CI coupling) is the top operational risk, neutralized by the `=70` arch override.

---

## Files to create / modify
- **Modify** `.github/workflows/ci.yml` → 4 jobs (harden Job 1: cache + `-LE cuda`; add `lint`, `cuda-compile` ±MPI matrix, inert `gpu`); scope triggers + concurrency.
- **Create** `.clang-format` + a one-time `clang-format -i` normalization commit.
- **Create** `deploy/apptainer/tdmd.def`, `deploy/docker/Containerfile`, `deploy/spack/spack.yaml`, `deploy/README.md`.
- **Modify** `include/tdmd/core/conveyor.hpp` — `on_pass` in `ConveyorOptions` (:91) + fire at :551 (one `if`).
- **Create** `include/tdmd/cli/dashboard.hpp`, `src/cli/dashboard.cpp`, `include/tdmd/cli/progress_monitor.hpp`, `tests/test_dashboard.cpp` (+ `add_test Test_Dashboard`).
- **Modify** `src/main.cpp` — `--dashboard` flag wires `co.on_pass`/the monitor on the ring path and the monitor on the direct path's existing `on_frame`; HALT frame prints the rescue path.
- **No CMakeLists source change for CI knobs** — `-DTDMD_WITH_CUDA`/`-DTDMD_WITH_MPI`/`-DCMAKE_CUDA_ARCHITECTURES=70`/`-DCMAKE_CXX_FLAGS=-Werror`/`-DTDMD_BUILD_TESTS=OFF` are all already respected command-line options (CMakeLists:16/20/22/29). Only `add_test(Test_Dashboard)` + the `src/cli/` sources are new build entries.

**Smallest-robust outcome:** the build system is already CI-ready (only the workflow file + format config are new); the container is one def + one Containerfile sharing one cmake invocation; the smoke is the existing `ctest -LE cuda` (no new test code); the dashboard is one atomic-guarded callback at the single existing stats-write site (conveyor.hpp:551), empty-by-default, with zero change to the proven bitwise hot path.