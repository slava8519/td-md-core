# TD-MD Core — deployment (M7)

Containers + Spack env for HPC deploy. The M7 deploy criterion (Roadmap): *the
container builds and passes an in-container smoke test*. The smoke is **CPU-only
and LAMMPS-free** (golden references are frozen under `reference_data/`), so it
runs at container-build time on a GPU-less build host. The GPU correctness gate
(`ctest -L cuda` + `compute-sanitizer`) runs on a node **with** an NVIDIA GPU.

## What is validatable NOW vs LATER

| Capability | How | Status |
|---|---|---|
| CPU build + ~40 golden/determinism tests | `ctest -LE cuda` (no GPU, no LAMMPS) | **NOW** (cloud CI + any box) |
| In-container CPU smoke | `%test` / build-time `RUN ctest -LE cuda` + `tdmd config_m0.yaml` | **NOW** (single box) |
| GPU correctness gate in the container | `apptainer exec --nv tdmd.sif ctest -L cuda` (+ `scripts/gpu_gate.sh`) | **NOW** (one RTX 5080) |
| `spack concretize` of the toolchain env | `spack env activate -d deploy/spack && spack concretize` | **NOW** (anywhere) |
| `spack install` against a site toolchain | activate env, `spack install`, then `cmake … && cmake --build` | **LATER** (cluster) |
| Multi-node MPI ring across nodes | `mpirun -np N apptainer exec --nv tdmd.sif …` | **LATER** (M5b: ≥2 GPUs) |

"NOW" = doable on the single dev box / cloud today. "LATER" = needs the external
M5b trigger (a node with ≥2 GPUs); documented here, run when the resource exists.
This mirrors the Roadmap gate-rule exception for M5b — nothing here claims
"cluster-validated."

## Files

- `apptainer/tdmd.def` — Apptainer/Singularity image. `%post` `git clone`s a
  fixed tag (`opus`) and builds with CUDA+MPI; `%test` is the build-time CPU
  smoke; `%runscript` runs the engine.
- `docker/Containerfile` — the Docker/Podman twin. The one intentional
  difference: it `COPY . /opt/tdmd` (builds **this** checkout, e.g. a PR) instead
  of cloning a tag. Multi-stage (devel build → runtime image).
- `spack/spack.yaml` — a Spack env providing only the **toolchain + system libs**
  (cmake, cuda, openmpi, gcc). yaml-cpp/googletest stay FetchContent-pinned in
  `CMakeLists.txt` — NOT Spack deps (one source of version truth).

## Build & run

### Apptainer (HPC)

```sh
# Build (needs network in %post for the FetchContent deps); the %test CPU smoke
# runs at build time. The orchestrator/build host runs this — it is heavy.
apptainer build tdmd.sif deploy/apptainer/tdmd.def

# Single-GPU run
apptainer run --nv tdmd.sif config/config_m0.yaml
apptainer run --nv tdmd.sif config/config_ring.yaml --dashboard   # live dashboard

# GPU correctness gate (needs a device)
apptainer exec --nv tdmd.sif ctest --test-dir /opt/tdmd/build -L cuda

# Non-5080 cluster: set the CUDA arch before building (A100=80-real, H100=90-real)
TDMD_CUDA_ARCH=80-real apptainer build tdmd.sif deploy/apptainer/tdmd.def
```

### Docker / Podman

```sh
docker build -f deploy/docker/Containerfile -t tdmd .         # from the repo root
docker run --gpus all tdmd config/config_m0.yaml
podman build -f deploy/docker/Containerfile -t tdmd .
docker build --build-arg TDMD_CUDA_ARCH=80-real -f deploy/docker/Containerfile -t tdmd .
```

### Spack (no-privileged-build / air-gapped)

```sh
spack env activate -d deploy/spack
spack concretize            # NOW: resolves the toolchain
spack install               # LATER: builds it against the site compilers
# then, in the activated env:
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DTDMD_WITH_CUDA=ON -DTDMD_WITH_MPI=ON \
    -DCMAKE_CUDA_ARCHITECTURES=<cluster-arch>
cmake --build build
```

### Multi-node ring (LATER — M5b)

```sh
# Needs ≥2 GPUs across nodes + a host MPI launcher; OpenMPI is host-staging
# (not CUDA-aware), matching the M5a MpiRingEdge transport.
mpirun -np N apptainer exec --nv tdmd.sif \
    /opt/tdmd/build/tdmd config/config_ring.yaml
```

## Notes / gotchas

- **Base image tag.** `nvidia/cuda:13.1.0-devel-ubuntu24.04` must resolve on the
  registry; a wrong tag fails the whole build. The runtime stage of the
  Containerfile uses the matching `…-runtime-…` tag.
- **FetchContent needs network at build/configure time.** Air-gapped build hosts
  must vendor yaml-cpp/googletest or pass `-DFETCHCONTENT_SOURCE_DIR_*` overrides
  (or build via the Spack env + a local mirror).
- **`--nv` injects the host driver.** SASS built for `sm_120` (the 5080 default)
  needs the deploy node's driver to support CUDA 13.1; older cluster drivers
  require an arch rebuild (`TDMD_CUDA_ARCH`).
- **No `compute-sanitizer` in the smoke.** That is the dev-machine gate
  (`scripts/gpu_gate.sh`), not a deploy smoke.
