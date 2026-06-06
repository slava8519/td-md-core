#include "tdmd/io/rescue.hpp"

#include <fstream>
#include <iomanip>

namespace tdmd::io {

static std::string element_of(int type) {
  return (type == 1) ? "Al" : ("Type" + std::to_string(type));
}

bool write_rescue_xyz(const std::string& path,
                      const core::AtomSoA<double>& a,
                      const core::Box& box,
                      const std::string& reason) {
  std::ofstream f(path, std::ios::trunc);
  if (!f) return false;
  f << std::setprecision(10);
  f << a.n << "\n";
  f << "TD-MD rescue dump — " << reason
    << "  box=[" << box.len(0) << " " << box.len(1) << " " << box.len(2) << "]\n";
  for (int i = 0; i < a.n; ++i)
    f << element_of(a.type[i]) << ' '
      << a.x[i] << ' ' << a.y[i] << ' ' << a.z[i] << "\n";
  return static_cast<bool>(f);
}

} // namespace tdmd::io
