#pragma once
#include <vector>
#include <array>

// SoA atom container and simulation box (ТЗ §4: strict SoA; global coords FP64,
// force accumulation FP64 for determinism INV-9). Kept template<typename Real>
// per the single-codebase precision rule; M0 instantiates Real=double.
namespace tdmd::core {

struct Box {
  std::array<double, 3> lo{0, 0, 0};
  std::array<double, 3> hi{0, 0, 0};
  std::array<bool, 3>   periodic{true, true, true};
  double len(int d) const { return hi[d] - lo[d]; }
};

template <typename Real>
struct AtomSoA {
  std::vector<double> x, y, z;     // global positions (Å), FP64
  std::vector<Real>   vx, vy, vz;  // velocities (Å/ps)
  std::vector<double> fx, fy, fz;  // forces (eV/Å), FP64 accumulation
  std::vector<int>    type;        // atom type id
  std::vector<double> mass;        // per-atom mass (amu)
  int n = 0;

  void resize(int N) {
    n = N;
    x.assign(N, 0.0);  y.assign(N, 0.0);  z.assign(N, 0.0);
    vx.assign(N, Real(0)); vy.assign(N, Real(0)); vz.assign(N, Real(0));
    fx.assign(N, 0.0); fy.assign(N, 0.0); fz.assign(N, 0.0);
    type.assign(N, 0); mass.assign(N, 0.0);
  }
};

} // namespace tdmd::core
