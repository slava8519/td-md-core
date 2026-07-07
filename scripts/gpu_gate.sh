#!/usr/bin/env bash
# Local GPU gate (Roadmap "Сквозные требования", M2.7+): runs every ctest with
# the `cuda` label, then compute-sanitizer (memcheck + racecheck) over the CUDA
# test binaries. NOT a cloud-CI job — requires an NVIDIA GPU.
#
#   ./scripts/gpu_gate.sh [build-dir] [binary-glob ...]
#
# build-dir default: build-cuda. Optional binary globs (relative to build-dir,
# e.g. 'test_cuda_eam_ring' 'test_cuda_meam*') narrow the SANITIZER loop only —
# ctest -L cuda ALWAYS runs in full. Scoped runs are for PR iteration on the
# binaries a change actually touches; the FULL sweep (no glob) remains the
# pre-merge gate. A scoped run prints a SCOPED banner so its log cannot be
# mistaken for the full gate.
#
# The sanitizer jobs (binary × tool) are independent and run in parallel
# (GPU_GATE_JOBS, default 4): sanitizer overhead is mostly host-side
# instrumentation of tiny kernel launches, and the verdict is
# presence-of-errors, not timing — GPU contention can skew speed, never the
# result (this is NOT a perf harness; perf numbers still require an idle GPU).
# Per-job logs land in a temp dir; failures dump their tail.
set -euo pipefail
BUILD="${1:-build-cuda}"
if [[ $# -gt 0 ]]; then shift; fi
GLOBS=("$@")
if [[ ${#GLOBS[@]} -eq 0 ]]; then GLOBS=('test_cuda_*'); fi

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
# disjoint per-atom elements — race-free by construction, memcheck-covered)
# and *OffEquilibrium* (M4-S 3000-step shock stress: measured 2h+ under
# racecheck 2026-07-02 while the SHORT Verlet tests — same code paths — take
# ~20 s each and STAY racechecked; the stress test remains memcheck-covered,
# matching its M4-S acceptance record).
MEMCHECK_FILTER='-*Replica36*:*NveInvariant50k*'
RACECHECK_FILTER='-*Replica36*:*NveInvariant50k*:*Mixed*:*OffEquilibrium*'

# Resolve binaries (dedup, preserve order); an empty match is an ERROR, not a
# vacuous pass.
BINS=()
for pat in "${GLOBS[@]}"; do
  for bin in "$BUILD"/$pat; do
    [[ -x "$bin" ]] && BINS+=("$bin")
  done
done
if [[ ${#BINS[@]} -eq 0 ]]; then
  echo "gpu_gate: no executable matches '${GLOBS[*]}' in $BUILD — refusing a vacuous pass." >&2
  exit 2
fi
mapfile -t BINS < <(printf '%s\n' "${BINS[@]}" | awk '!seen[$0]++')

if [[ "${GLOBS[*]}" != 'test_cuda_*' ]]; then
  echo "=== gpu_gate: SCOPED sanitizer run (${GLOBS[*]} -> ${#BINS[@]} binaries) — NOT the full pre-merge gate ==="
fi

JOBS="${GPU_GATE_JOBS:-4}"
LOGDIR="$(mktemp -d "${TMPDIR:-/tmp}/gpu_gate.XXXXXX")"

run_one() {
  local tool="$1" bin="$2" filter="$3"
  local base log
  base="$(basename "$bin")"
  log="$LOGDIR/$base.$tool.log"
  if "$SANITIZER" --tool "$tool" --error-exitcode 1 "$bin" --gtest_filter="$filter" >"$log" 2>&1; then
    echo "=== gpu_gate: PASS $tool $base ==="
  else
    echo "=== gpu_gate: FAIL $tool $base — log tail ($log): ==="
    tail -n 60 "$log"
    : >"$LOGDIR/FAILED"
  fi
}

for bin in "${BINS[@]}"; do
  for tool in memcheck racecheck; do
    filter="$MEMCHECK_FILTER"
    [[ "$tool" == racecheck ]] && filter="$RACECHECK_FILTER"
    while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || true; done
    run_one "$tool" "$bin" "$filter" &
  done
done
wait

if [[ -e "$LOGDIR/FAILED" ]]; then
  echo "=== gpu_gate: FAILED (logs: $LOGDIR) ===" >&2
  exit 1
fi
echo "=== gpu_gate: OK ==="
