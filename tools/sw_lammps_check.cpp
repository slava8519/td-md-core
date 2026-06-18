// One-off SW validation vs LAMMPS pair_style sw (NOT the frozen golden — that is T7).
// Emits a LAMMPS data file from a perturbed diamond-Si config + this engine's PE and
// per-atom forces (sw_direct_fp64). A companion .in runs `sw; run 0`; a script diffs.
// Confirms the SW FORM + PARAMETERS are right (FD only proves force=−dU/dx for our U).
#include <cstdio>

#include "tdmd/core/soa.hpp"
#include "tdmd/gen/diamond_si.hpp"  // SAME fixture as test_sw.cpp (no drift)
#include "tdmd/potentials/sw.hpp"

using namespace tdmd;
namespace pot = tdmd::potentials;

int main() {
  core::Box box;
  core::AtomSoA<double> a = gen::make_diamond_si(2, 2, 2, 5.431, 0.20, box);

  // LAMMPS data file (metal units, atomic style).
  FILE* d = std::fopen("/tmp/sw_si.data", "w");
  std::fprintf(d, "SW diamond Si cross-check\n\n%d atoms\n1 atom types\n\n", a.n);
  std::fprintf(d, "%.16g %.16g xlo xhi\n%.16g %.16g ylo yhi\n%.16g %.16g zlo zhi\n\n",
               box.lo[0], box.hi[0], box.lo[1], box.hi[1], box.lo[2], box.hi[2]);
  std::fprintf(d, "Masses\n\n1 28.0855\n\nAtoms\n\n");
  for (int i = 0; i < a.n; ++i)
    std::fprintf(d, "%d 1 %.16g %.16g %.16g\n", i + 1, a.x[i], a.y[i], a.z[i]);
  std::fclose(d);

  pot::SwParams sp;
  core::AtomSoA<double> g = a;
  core::zero_forces(g);
  const auto acc = pot::sw_direct_fp64(g, box, sp, true);
  FILE* m = std::fopen("/tmp/sw_mine.txt", "w");
  std::fprintf(m, "PE %.16g\n", acc.pe);
  for (int i = 0; i < a.n; ++i)
    std::fprintf(m, "%d %.16g %.16g %.16g\n", i + 1, g.fx[i], g.fy[i], g.fz[i]);
  std::fclose(m);
  std::printf("wrote /tmp/sw_si.data (%d atoms) + /tmp/sw_mine.txt; my PE=%.10g eV\n",
              a.n, acc.pe);
  return 0;
}
