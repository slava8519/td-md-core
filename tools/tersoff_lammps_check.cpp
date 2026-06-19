// One-off Tersoff validation vs LAMMPS pair_style tersoff (regenerates the Te1 golden).
// Emits a LAMMPS data file from a perturbed diamond-Si config + our PE/forces (the FP64
// oracle). The LAMMPS golden is the LOAD-BEARING external witness of the b_ij derivative
// algebra + the parameter traps (T-a h-in-g, T-b m=3 odd lam3^3) that FD-of-energy is blind to.
#include <cstdio>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/gen/diamond_si.hpp"
#include "tdmd/potentials/tersoff.hpp"

using namespace tdmd;
namespace pot = tdmd::potentials;

int main() {
  pot::TersoffParams p;
  core::Box box;
  core::AtomSoA<double> a = gen::make_diamond_si(2, 2, 2, 5.431, 0.20, box);
  box.periodic = {true, true, true};

  FILE* d = std::fopen("/tmp/ters_si.data", "w");
  std::fprintf(d, "Tersoff diamond Si cross-check\n\n%d atoms\n1 atom types\n\n", a.n);
  std::fprintf(d, "%.16g %.16g xlo xhi\n%.16g %.16g ylo yhi\n%.16g %.16g zlo zhi\n\n",
               box.lo[0], box.hi[0], box.lo[1], box.hi[1], box.lo[2], box.hi[2]);
  std::fprintf(d, "Masses\n\n1 28.0855\n\nAtoms\n\n");
  for (int i = 0; i < a.n; ++i)
    std::fprintf(d, "%d 1 %.16g %.16g %.16g\n", i + 1, a.x[i], a.y[i], a.z[i]);
  std::fclose(d);

  const core::PairGeom geom(box, p.rcut());
  core::AtomSoA<double> g = a;
  core::zero_forces(g);
  const auto acc = pot::tersoff_direct_fp64(g, geom, p, true);
  FILE* m = std::fopen("/tmp/ters_mine.txt", "w");
  std::fprintf(m, "PE %.16g\n", acc.pe);
  for (int i = 0; i < a.n; ++i)
    std::fprintf(m, "%d %.16g %.16g %.16g\n", i + 1, g.fx[i], g.fy[i], g.fz[i]);
  std::fclose(m);
  std::printf("wrote /tmp/ters_si.data (%d atoms) + /tmp/ters_mine.txt; my PE=%.10g eV\n",
              a.n, acc.pe);
  return 0;
}
