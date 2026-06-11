#pragma once
#include <string>
#include "tdmd/core/soa.hpp"

namespace tdmd::io {

// Reads a LAMMPS read_data file (atom_style atomic, metal units).
// Fills atom positions/types/masses, velocities (Velocities section, if
// present; zero otherwise) and box bounds. `box.periodic` is left untouched
// (caller sets it from config). Returns false on failure.
// Atoms are stored sorted by atom id ascending (index i == id i+1).
bool read_lammps_data(const std::string& path,
                      core::AtomSoA<double>& atoms,
                      core::Box& box);

} // namespace tdmd::io
