#!/usr/bin/env python3
"""Parametric FCC supercell generator (M3): writes LAMMPS data files
(atom_style atomic, metal units) for arbitrary nx*ny*nz cells.

Generalizes the geometry path of reference_data/generate_reference.py:
same basis, same seeded uniform perturbation, same output format — so
`--cells 3 3 2` reproduces the golden al_fcc_72.data geometry bit-for-bit.

Reference forces for large N are NOT generated or stored (audit C2): the
oracle for the clustered path is the internal cross test against the direct
O(N^2) path (<=1e-12), not a golden CSV.

Examples:
  tools/gen_fcc.py --cells 3 3 4 -o reference_data/al_fcc_144.data   # >=4 zones
  tools/gen_fcc.py --cells 14 14 14 -o /tmp/al_11k.data              # ~10^4 atoms
  tools/gen_fcc.py --cells 30 30 30 -o /tmp/al_108k.data             # ~10^5 atoms
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--cells", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                required=True, help="conventional FCC cells per axis (4 atoms/cell)")
ap.add_argument("--lattice", type=float, default=4.05, help="lattice constant, Å (Al)")
ap.add_argument("--mass", type=float, default=26.9815, help="atomic mass, amu (Al)")
ap.add_argument("--perturb", type=float, default=0.10,
                help="uniform displacement amplitude, Å (0 = perfect lattice)")
ap.add_argument("--seed", type=int, default=42, help="perturbation RNG seed")
ap.add_argument("-o", "--out", required=True, help="output LAMMPS data file")
args = ap.parse_args()

NX, NY, NZ = args.cells
A = args.lattice

basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * A
pos = []
for i in range(NX):
    for j in range(NY):
        for k in range(NZ):
            for b in basis:
                pos.append(b + np.array([i, j, k]) * A)
pos = np.array(pos)
N = len(pos)
box = np.array([NX * A, NY * A, NZ * A])

if args.perturb > 0:
    rng = np.random.default_rng(args.seed)
    pos = (pos + rng.uniform(-args.perturb, args.perturb, pos.shape)) % box

with open(args.out, "w") as f:
    f.write(f"Al FCC {N}-atom supercell {NX}x{NY}x{NZ} a={A} "
            f"(seed={args.seed}, amp={args.perturb}) — metal units, tools/gen_fcc.py\n\n")
    f.write(f"{N} atoms\n1 atom types\n\n")
    f.write(f"0.0 {box[0]:.6f} xlo xhi\n0.0 {box[1]:.6f} ylo yhi\n0.0 {box[2]:.6f} zlo zhi\n\n")
    f.write(f"Masses\n\n1 {args.mass}\n\n")
    f.write("Atoms # atomic\n\n")
    for i, (x, y, z) in enumerate(pos, 1):
        f.write(f"{i} 1 {x:.10f} {y:.10f} {z:.10f}\n")

print(f"Written: {args.out}  (N={N}, box={box[0]:.3f}x{box[1]:.3f}x{box[2]:.3f} Å)")
