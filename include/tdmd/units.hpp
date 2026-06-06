#pragma once
// Units & constants — LAMMPS `metal` system (eV, Å, amu, ps).
// Source of truth: docs/TD_MD_Core_Units_v1_0.md.
namespace tdmd::units {

inline constexpr double kB    = 8.617333262e-5;  // Boltzmann constant, eV/K
inline constexpr double mvv2e = 1.0364269e-4;    // mass*velocity^2 -> energy
// Units doc gives ftm2v = 1/mvv2e (rounded display 9648.5336). We use the exact
// reciprocal so that mvv2e*ftm2v == 1 — required for velocity-Verlet energy
// conservation (KE = ½·mvv2e·m·v² must be consistent with a = ftm2v·F/m).
inline constexpr double ftm2v = 1.0 / mvv2e;     // force/mass -> acceleration

// KE[eV]    = 0.5 * mvv2e * m[amu] * v^2[(Å/ps)^2]
// a[Å/ps^2] = ftm2v * F[eV/Å] / m[amu]
// T[K]      = 2*KE / (N_dof * kB)

} // namespace tdmd::units
