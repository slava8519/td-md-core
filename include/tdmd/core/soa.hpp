#pragma once
#include <vector>
#include <array>
#include <algorithm>

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
  std::vector<int>    id;          // original atom id — survives Z-order sort (M3)
  int n = 0;

  void resize(int N) {
    n = N;
    x.assign(N, 0.0);  y.assign(N, 0.0);  z.assign(N, 0.0);
    vx.assign(N, Real(0)); vy.assign(N, Real(0)); vz.assign(N, Real(0));
    fx.assign(N, 0.0); fy.assign(N, 0.0); fz.assign(N, 0.0);
    type.assign(N, 0); mass.assign(N, 0.0);
    id.resize(N);
    for (int i = 0; i < N; ++i) id[i] = i + 1;
  }
};

// Force buffers are zeroed by the CALLER, not inside potential::compute (M3
// split): the w-mechanism of zones accumulates partial contributions from
// several compute passes into one buffer (INV-3/INV-8), so zeroing is loop
// policy, not potential policy.
template <typename Real>
void zero_forces(AtomSoA<Real>& a) {
  std::fill(a.fx.begin(), a.fx.end(), 0.0);
  std::fill(a.fy.begin(), a.fy.end(), 0.0);
  std::fill(a.fz.begin(), a.fz.end(), 0.0);
}

} // namespace tdmd::core
