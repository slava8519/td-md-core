#include "tdmd/io/writer.hpp"

#include <fstream>
#include <iomanip>

namespace tdmd::io {

TrajectoryWriter::TrajectoryWriter(const std::string& path)
    : path_(path), ok_(true) {
  std::ofstream f(path_, std::ios::trunc);  // create / clear
  ok_ = static_cast<bool>(f);
}

void TrajectoryWriter::write_frame(long timestep,
                                   const core::AtomSoA<double>& a,
                                   const core::Box& box) {
  std::ofstream f(path_, std::ios::app);
  if (!f) { ok_ = false; return; }

  std::string bb;
  for (int d = 0; d < 3; ++d) {
    bb += box.periodic[d] ? "pp" : "ff";
    if (d < 2) bb += ' ';
  }

  f << std::setprecision(10);
  f << "ITEM: TIMESTEP\n" << timestep << "\n";
  f << "ITEM: NUMBER OF ATOMS\n" << a.n << "\n";
  f << "ITEM: BOX BOUNDS " << bb << "\n";
  for (int d = 0; d < 3; ++d) f << box.lo[d] << ' ' << box.hi[d] << "\n";
  // xu yu zu: global coordinates are never wrapped back into the box (B10) —
  // 'x y z' would make OVITO/LAMMPS treat them as wrapped and mis-render PBC.
  f << "ITEM: ATOMS id type xu yu zu\n";
  for (int i = 0; i < a.n; ++i)
    f << (i + 1) << ' ' << a.type[i] << ' '
      << a.x[i] << ' ' << a.y[i] << ' ' << a.z[i] << "\n";
}

} // namespace tdmd::io
