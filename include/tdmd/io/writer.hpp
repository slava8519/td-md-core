#pragma once
#include <string>
#include "tdmd/core/soa.hpp"

namespace tdmd::io {

// Appends frames to a LAMMPS dump (.lammpstrj). Truncates the file on construction.
class TrajectoryWriter {
 public:
  explicit TrajectoryWriter(const std::string& path);
  void write_frame(long timestep, const core::AtomSoA<double>& atoms,
                   const core::Box& box);
  bool ok() const { return ok_; }

 private:
  std::string path_;
  bool ok_;
};

} // namespace tdmd::io
