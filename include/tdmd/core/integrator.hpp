#pragma once
#include "tdmd/core/soa.hpp"
#include "tdmd/units.hpp"

// Velocity-Verlet integrator (Гл.1.2), NVE. Two half-steps (kick-drift-kick):
//   first_half : v(t+½) = v(t) + ½dt·a(t);  x(t+dt) = x(t) + dt·v(t+½)
//   (recompute forces at x(t+dt))
//   second_half: v(t+dt) = v(t+½) + ½dt·a(t+dt)
// with a = ftm2v·F/m (metal units, see tdmd/units.hpp).
namespace tdmd::core {

template <typename Real>
struct VelocityVerlet {
  static void first_half(AtomSoA<Real>& a, double dt) {
    for (int i = 0; i < a.n; ++i) {
      const double inv_m = units::ftm2v / a.mass[i];
      a.vx[i] += Real(0.5 * dt * inv_m * a.fx[i]);
      a.vy[i] += Real(0.5 * dt * inv_m * a.fy[i]);
      a.vz[i] += Real(0.5 * dt * inv_m * a.fz[i]);
      a.x[i] += dt * double(a.vx[i]);
      a.y[i] += dt * double(a.vy[i]);
      a.z[i] += dt * double(a.vz[i]);
    }
  }

  static void second_half(AtomSoA<Real>& a, double dt) {
    for (int i = 0; i < a.n; ++i) {
      const double inv_m = units::ftm2v / a.mass[i];
      a.vx[i] += Real(0.5 * dt * inv_m * a.fx[i]);
      a.vy[i] += Real(0.5 * dt * inv_m * a.fy[i]);
      a.vz[i] += Real(0.5 * dt * inv_m * a.fz[i]);
    }
  }
};

template <typename Real>
double kinetic_energy(const AtomSoA<Real>& a) {
  double ke = 0.0;
  for (int i = 0; i < a.n; ++i) {
    const double v2 = double(a.vx[i]) * a.vx[i] +
                      double(a.vy[i]) * a.vy[i] +
                      double(a.vz[i]) * a.vz[i];
    ke += 0.5 * units::mvv2e * a.mass[i] * v2;
  }
  return ke;
}

} // namespace tdmd::core
