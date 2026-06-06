#pragma once
#include <string>
#include "tdmd/core/soa.hpp"

namespace tdmd::io {

// Writes the last valid geometry to an XYZ rescue dump (ZoneFSM §9). Used on a
// causality violation (INV-4), CUDA error, or other HALT. Returns false on I/O
// failure. `reason` is recorded in the XYZ comment line.
bool write_rescue_xyz(const std::string& path,
                      const core::AtomSoA<double>& atoms,
                      const core::Box& box,
                      const std::string& reason);

} // namespace tdmd::io
