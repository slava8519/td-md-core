#!/usr/bin/env bash
# Local GPU gate (Roadmap "Сквозные требования", M2.7+): runs every ctest with
# the `cuda` label, then compute-sanitizer (memcheck + racecheck) over the CUDA
# test binaries. NOT a cloud-CI job — requires an NVIDIA GPU.
#   ./scripts/gpu_gate.sh [build-dir]   (default: build-cuda)
set -euo pipefail
BUILD="${1:-build-cuda}"

if [[ ! -d "$BUILD" ]]; then
  echo "gpu_gate: build dir '$BUILD' not found." >&2
  echo "  cmake -S . -B $BUILD -G Ninja -DTDMD_WITH_CUDA=ON && cmake --build $BUILD" >&2
  exit 2
fi

echo "=== gpu_gate: ctest -L cuda ==="
ctest --test-dir "$BUILD" -L cuda --output-on-failure

SANITIZER="${COMPUTE_SANITIZER:-compute-sanitizer}"
# Long-horizon tests (26k/50k passes) are the same code paths as the short
# ones — excluded from sanitizer runs for time. racecheck additionally skips
# Mixed* (pathological slowdown, see Bench doc; pack/unpack kernels write
# disjoint per-atom elements — race-free by construction, memcheck-covered).
MEMCHECK_FILTER='-*Replica36*:*NveInvariant50k*'
RACECHECK_FILTER='-*Replica36*:*NveInvariant50k*:*Mixed*'
for bin in "$BUILD"/test_cuda_*; do
  [[ -x "$bin" ]] || continue
  echo "=== gpu_gate: $SANITIZER --tool memcheck $(basename "$bin") ==="
  "$SANITIZER" --tool memcheck --error-exitcode 1 "$bin" --gtest_filter="$MEMCHECK_FILTER"
  echo "=== gpu_gate: $SANITIZER --tool racecheck $(basename "$bin") ==="
  "$SANITIZER" --tool racecheck --error-exitcode 1 "$bin" --gtest_filter="$RACECHECK_FILTER"
done
echo "=== gpu_gate: OK ==="
