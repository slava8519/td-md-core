#!/usr/bin/env python3
"""Golden reference generator: FCC Al + Morse (params from Andreev 2007 dissertation).
Builds a 3x3x2 FCC supercell (72 atoms) with PBC, applies a small seeded random
displacement to break symmetry, computes per-atom Morse forces and total PE,
and verifies forces against central finite differences."""
import numpy as np, json, csv

# --- Morse parameters (dissertation Гл.3.5, стр.1187-1190) ---
D, ALPHA, R0 = 0.29614, 1.11892, 3.29692   # eV, 1/Å, Å
RCUT = 4.0                                   # Å (<= half min box for min-image)
A = 4.05                                      # Å, FCC Al lattice constant
NX, NY, NZ = 3, 3, 2                          # conventional cells -> 4*18 = 72 atoms
MASS = 26.9815                                # amu
SEED, AMP = 42, 0.10                          # displacement seed, amplitude Å

# --- build perfect FCC ---
basis = np.array([[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]])*A
pos = []
for i in range(NX):
    for j in range(NY):
        for k in range(NZ):
            for b in basis:
                pos.append(b + np.array([i,j,k])*A)
pos = np.array(pos)
N = len(pos)
box = np.array([NX*A, NY*A, NZ*A])
assert N == 72, N
# perturb
rng = np.random.default_rng(SEED)
pos = (pos + rng.uniform(-AMP, AMP, pos.shape)) % box

# --- Morse energy & force, minimum image, energy-shifted at RCUT ---
def Ushift():
    e = D*(np.exp(-2*ALPHA*(RCUT-R0)) - 2*np.exp(-ALPHA*(RCUT-R0)))
    return e
USH = Ushift()

def energy_forces(p):
    F = np.zeros_like(p); PE = 0.0
    for i in range(N):
        d = p[i] - p          # i - j
        d -= box*np.round(d/box)   # minimum image
        r = np.linalg.norm(d, axis=1)
        m = (r < RCUT) & (r > 1e-9)
        rr = r[m]
        u = D*(np.exp(-2*ALPHA*(rr-R0)) - 2*np.exp(-ALPHA*(rr-R0))) - USH
        PE += 0.5*u.sum()
        dUdr = 2*ALPHA*D*(np.exp(-ALPHA*(rr-R0)) - np.exp(-2*ALPHA*(rr-R0)))
        fmag = -dUdr          # F = -dU/dr along (r_i-r_j)/r
        F[i] += ((fmag/rr)[:,None]*d[m]).sum(axis=0)
    return PE, F

PE, F = energy_forces(pos)

# --- verify forces via central finite differences ---
h = 1e-5; maxerr = 0.0
for i in rng.choice(N, 6, replace=False):
    for c in range(3):
        p2 = pos.copy(); p2[i,c]+=h; Ep,_=energy_forces(p2)
        p2 = pos.copy(); p2[i,c]-=h; Em,_=energy_forces(p2)
        ffd = -(Ep-Em)/(2*h)
        maxerr = max(maxerr, abs(ffd - F[i,c]))

print(f"N={N}  PE_total={PE:.10f} eV  max|F|={np.abs(F).max():.10f} eV/Å")
print(f"FD force check: max|F_analytic - F_fd| = {maxerr:.3e} eV/Å  -> {'OK' if maxerr<1e-5 else 'FAIL'}")

# --- write outputs ---
import os; os.makedirs('reference_data', exist_ok=True)
# LAMMPS data file (metal units)
with open('reference_data/al_fcc_72.data','w') as f:
    f.write("Al FCC 72-atom Morse reference (Andreev 2007 params) — metal units\n\n")
    f.write(f"{N} atoms\n1 atom types\n\n")
    f.write(f"0.0 {box[0]:.6f} xlo xhi\n0.0 {box[1]:.6f} ylo yhi\n0.0 {box[2]:.6f} zlo zhi\n\n")
    f.write(f"Masses\n\n1 {MASS}\n\n")
    f.write("Atoms # atomic\n\n")
    for i,(x,y,z) in enumerate(pos,1):
        f.write(f"{i} 1 {x:.10f} {y:.10f} {z:.10f}\n")
# XYZ
with open('reference_data/al_fcc_72.xyz','w') as f:
    f.write(f"{N}\nAl FCC 72-atom perturbed (seed={SEED},amp={AMP})\n")
    for x,y,z in pos: f.write(f"Al {x:.10f} {y:.10f} {z:.10f}\n")
# reference forces
with open('reference_data/reference_forces.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['id','fx','fy','fz'])
    for i,(fx,fy,fz) in enumerate(F,1): w.writerow([i,f"{fx:.10e}",f"{fy:.10e}",f"{fz:.10e}"])
# summary
json.dump({'n_atoms':N,'lattice_const_A':A,'supercell':[NX,NY,NZ],'box_A':box.tolist(),
    'potential':'morse','D_eV':D,'alpha_inv_A':ALPHA,'r0_A':R0,'rcut_A':RCUT,'pair_modify':'shift yes',
    'mass_amu':MASS,'pbc':[True,True,True],'seed':SEED,'displace_amp_A':AMP,
    'PE_total_eV':PE,'max_abs_force_eV_per_A':float(np.abs(F).max()),
    'fd_check_max_err':maxerr,'units':'metal'},
    open('reference_data/reference_summary.json','w'), indent=2)
# LAMMPS reproduction input
with open('reference_data/in.al_morse','w') as f:
    f.write(f"""# Reproduce golden forces in LAMMPS (metal units)
units metal
atom_style atomic
boundary p p p
read_data al_fcc_72.data
pair_style morse {RCUT}
pair_coeff 1 1 {D} {ALPHA} {R0}
pair_modify shift yes
run 0
dump d all custom 1 forces.dump id fx fy fz
run 0
""")
print("Written: reference_data/{al_fcc_72.data, .xyz, reference_forces.csv, reference_summary.json, in.al_morse}")
