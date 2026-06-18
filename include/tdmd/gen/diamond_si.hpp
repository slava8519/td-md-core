#pragma once
#include <array>
#include <random>
#include <vector>

#include "tdmd/core/soa.hpp"

// Diamond-cubic Si generator (8-atom basis, a0≈5.431 Å), DETERMINISTICALLY perturbed.
// Shared by test_sw.cpp (the SW fixture) and tools/sw_lammps_check.cpp (the LAMMPS
// cross-check) so the two cannot silently drift (SW PR-1 acceptance SHOULD-improve).
// A perfect lattice sits at the tetrahedral angle (cosθ=−1/3=cos0 ⇒ Δ=0 ⇒ φ₃ AND its
// force vanish — vacuous); the perturbation breaks the angle so the angular machinery
// is exercised. Box > 2·rcut so min-image is unambiguous. FCC would be vacuous.
namespace tdmd::gen {

inline core::AtomSoA<double> make_diamond_si(int nx, int ny, int nz, double a0,
                                             double perturb, core::Box& box,
                                             unsigned seed = 12345) {
  box.lo = {0, 0, 0};
  box.hi = {nx * a0, ny * a0, nz * a0};
  box.periodic = {true, true, true};
  const double b[8][3] = {{0, 0, 0},         {0, 0.5, 0.5},      {0.5, 0, 0.5},
                          {0.5, 0.5, 0},      {0.25, 0.25, 0.25}, {0.25, 0.75, 0.75},
                          {0.75, 0.25, 0.75}, {0.75, 0.75, 0.25}};
  std::vector<std::array<double, 3>> pos;
  for (int ix = 0; ix < nx; ++ix)
    for (int iy = 0; iy < ny; ++iy)
      for (int iz = 0; iz < nz; ++iz)
        for (auto& bb : b)
          pos.push_back({(ix + bb[0]) * a0, (iy + bb[1]) * a0, (iz + bb[2]) * a0});
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-perturb, perturb);
  core::AtomSoA<double> a;
  a.resize(int(pos.size()));
  for (int i = 0; i < a.n; ++i) {
    a.x[i] = pos[i][0] + u(rng);
    a.y[i] = pos[i][1] + u(rng);
    a.z[i] = pos[i][2] + u(rng);
    a.type[i] = 1;
    a.mass[i] = 28.0855;
  }
  return a;
}

}  // namespace tdmd::gen
