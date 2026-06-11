#pragma once
// Minimal HAL layer (Skeleton_Interfaces §2). Deliberately tiny: only the
// host/device markers needed by shared pair-math headers (M3). Streams,
// events, accumulate() and transport land at M4 — do not grow this early.
#if defined(__CUDACC__)
#define TDMD_HOST_DEVICE __host__ __device__
#else
#define TDMD_HOST_DEVICE
#endif
