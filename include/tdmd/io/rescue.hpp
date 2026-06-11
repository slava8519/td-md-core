#pragma once
#include <string>
#include "tdmd/core/soa.hpp"

namespace tdmd::io {

// Writes the last valid state to an extended-XYZ rescue dump (ZoneFSM §9):
// element x y z vx vy vz — restart-capable for NVE (B9, ARCH_ROADMAP_REVIEW
// 2026-06-11). Used on a causality violation (INV-4), CUDA error, or other
// HALT. Returns false on I/O failure. `reason` goes into the comment line —
// the caller should include step and dt there for full restart context.
bool write_rescue_xyz(const std::string& path,
                      const core::AtomSoA<double>& atoms,
                      const core::Box& box,
                      const std::string& reason);

} // namespace tdmd::io
