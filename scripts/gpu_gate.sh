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
for bin in "$BUILD"/test_cuda_smoke; do
  [[ -x "$bin" ]] || continue
  for tool in memcheck racecheck; do
    echo "=== gpu_gate: $SANITIZER --tool $tool $(basename "$bin") ==="
    "$SANITIZER" --tool "$tool" --error-exitcode 1 "$bin"
  done
done
echo "=== gpu_gate: OK ==="
