// M4-N — K-cadence sweep (perf track, "measure-first"). Measures the realized
// neighbour-list rebuild cadence K_eff vs the predicted K_pred = skin/(2·R_buf)
// on an equilibrium→shock sweep, the F.4 "K_pred/K_eff/cadence" line. CPU NVE
// velocity-Verlet over the FP64 Morse-Al reference potential — no CUDA, no GPU.
//
// conservatism = K_eff/K_pred quantifies how much the conservative causality
// envelope over-rebuilds vs the real motion:
//   - thermal (diffusive): the max-displacement atom decorrelates → K_eff >> K_pred
//     ⇒ conservatism well above C_buf. A PersistentVerlet list amortizes here.
//   - coherent drift-shock (ballistic): displacement = v·dt·k exactly ⇒
//     conservatism → C_buf (≈1.5), the pure envelope factor. No amortization gap.
//   build: plain CPU; run: ./bench_kcadence [--cells 5] [--steps 2000] [--skin 1.0]
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "tdmd/core/buffer.hpp"
#include "tdmd/core/integrator.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/metrics/k_cadence.hpp"
#include "tdmd/potentials/morse.hpp"
#include "tdmd/core/thermal.hpp"

using namespace tdmd;
using core::AtomSoA;
using core::Box;

namespace {
constexpr double kAlMass = 26.9815385;  // amu
constexpr double kA0 = 4.05;            // Å, FCC-Al lattice constant

AtomSoA<double> make_fcc(Box& box, int nc) {
  box.lo = {0, 0, 0};
  box.hi = {nc * kA0, nc * kA0, nc * kA0};
  box.periodic = {true, true, true};
  const double b[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  std::vector<std::array<double, 3>> p;
  for (int ix = 0; ix < nc; ++ix)
    for (int iy = 0; iy < nc; ++iy)
      for (int iz = 0; iz < nc; ++iz)
        for (auto& bb : b)
          p.push_back({(ix + bb[0]) * kA0, (iy + bb[1]) * kA0, (iz + bb[2]) * kA0});
  AtomSoA<double> a;
  a.resize(int(p.size()));
  for (int i = 0; i < a.n; ++i) {
    a.x[i] = p[i][0]; a.y[i] = p[i][1]; a.z[i] = p[i][2];
    a.type[i] = 1; a.mass[i] = kAlMass;
  }
  return a;
}

double temperature(const AtomSoA<double>& a) {
  const double ke = core::kinetic_energy(a);  // eV
  const int dof = 3 * a.n - 3;
  return dof > 0 ? 2.0 * ke / (double(dof) * units::kB) : 0.0;
}

// One NVE step (fixed dt). Returns nothing; forces left in a.f for the caller.
void nve_step(AtomSoA<double>& a, const Box& box, potentials::MorsePotential<double>& pot,
              double dt) {
  core::VelocityVerlet<double>::first_half(a, dt);
  core::zero_forces(a);
  pot.compute(a, box);
  core::VelocityVerlet<double>::second_half(a, dt);
}

// Run one regime: thermalize (+ optional coherent drift), equilibrate, then
// measure K-cadence over `steps`. drift>0 adds a uniform +x velocity to every
// atom AFTER thermalization (the ballistic "shock" reference).
void run_regime(const char* tag, AtomSoA<double> a, const Box& box, double T,
                double drift, double dt, double skin, double C_buf, long steps) {
  potentials::MorsePotential<double> pot;  // Al-Morse defaults, rcut 4.0, Shift
  if (T > 0.0) core::thermal::maxwell_init(a, T, /*seed=*/12345u);
  if (drift > 0.0)
    for (int i = 0; i < a.n; ++i) a.vx[i] += drift;

  // prime forces, then equilibrate so the cadence is measured at steady state
  core::zero_forces(a);
  pot.compute(a, box);
  const long equil = steps / 2;
  for (long s = 0; s < equil; ++s) nve_step(a, box, pot, dt);

  const double T_meas = temperature(a);
  metrics::KCadenceMeter meter;
  meter.begin(a.x, a.y, a.z, skin);
  double sum_vmax = 0.0;
  for (long s = 0; s < steps; ++s) {
    nve_step(a, box, pot, dt);
    // global v_max/a_max with single-node lag=1 — the z=1 reference envelope.
    // The conveyor's per-zone Λ-chain R_buf (n−1 lag) is ≥ this, so the global
    // choice errs conservative (never optimistic) vs the realized cadence.
    const double v_max = core::buffer::max_speed(a);
    const double a_max = core::buffer::max_accel(a);
    const double v_pred = v_max + a_max * dt;
    const double R_buf = core::buffer::compute_R_buf(v_pred, dt, C_buf);
    meter.step(a.x, a.y, a.z, R_buf);  // meter aggregates R_buf (skin budget)
    sum_vmax += v_max;
  }
  // for the coherent shock the COM drift dominates KE, so a "temperature" label
  // would be meaningless — report the drift speed instead.
  if (drift > 0.0) std::printf("drift=%4.0f ", drift);
  else             std::printf("T~%6.0fK ", T_meas);
  meter.acc.report(tag);
  std::printf("           v_max~%.2f Å/ps, dt=%.4f ps, C_buf=%.2f\n",
              sum_vmax / double(steps), dt, C_buf);
}
}  // namespace

int main(int argc, char** argv) {
  int nc = 5;
  long steps = 2000;
  double skin = 1.0, dt = 0.001, C_buf = 1.5;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    if (s == "--cells") nc = std::stoi(argv[++i]);
    else if (s == "--steps") steps = std::stol(argv[++i]);
    else if (s == "--skin") skin = std::stod(argv[++i]);
    else if (s == "--dt") dt = std::stod(argv[++i]);
    else if (s == "--cbuf") C_buf = std::stod(argv[++i]);
  }
  Box box;
  const AtomSoA<double> base = make_fcc(box, nc);
  std::printf("bench_kcadence: FCC-Al N=%d, Morse rcut=4.0, skin=%.2f, dt=%.4f ps\n",
              base.n, skin, dt);
  std::printf("  (T_meas = steady-state temperature after equilibration; Maxwell init "
              "loses ~½ KE→PE)\n");

  // equilibrium→hot thermal sweep
  for (double T : {200.0, 600.0, 1200.0, 2400.0})
    run_regime("thermal", base, box, T, /*drift=*/0.0, dt, skin, C_buf, steps);
  // ballistic reference: coherent drift-shock (30 Å/ps ≈ several km/s)
  run_regime("drift-shock", base, box, /*T=*/300.0, /*drift=*/30.0, dt, skin, C_buf, steps);

  std::printf("\nNOTE: conservatism = K_eff/K_pred. thermal ≫ C_buf (diffusive "
              "decorrelation ⇒ a PersistentVerlet list amortizes); drift-shock → "
              "C_buf=%.1f (ballistic, no amortization gap). The M4-B bake-off uses "
              "K_eff as the build-cost amortization factor.\n", C_buf);
  return 0;
}
