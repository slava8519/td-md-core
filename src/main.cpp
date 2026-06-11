#include <cstdio>
#include <string>
#include <cmath>

#include "tdmd/io/config.hpp"
#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/io/writer.hpp"
#include "tdmd/io/rescue.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/simulation.hpp"
#include "tdmd/core/thermal.hpp"
#include "tdmd/potentials/morse.hpp"
#include "tdmd/units.hpp"

using namespace tdmd;

int main(int argc, char** argv) {
  const std::string cfg_path = (argc > 1) ? argv[1] : "config/config_m0.yaml";

  io::Config cfg;
  try {
    cfg = io::load_config(cfg_path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[fatal] config (%s): %s\n", cfg_path.c_str(), e.what());
    return 1;
  }
  // enum/range guards (ensemble, potential, C_buf>=1, ...) live in load_config
  // since B10 — a Config that reaches this point is valid.

  // --- pre-start checklist (Units doc §5, ConfigSchema §3) ---
  std::printf("=== TD-MD Core ===\n");
  std::printf("config       : %s\n", cfg_path.c_str());
  std::printf("units        : metal (eV, Å, amu, ps)\n");
  std::printf("constants    : kB=%.9e eV/K  ftm2v=%.6f  mvv2e=%.7e\n",
              units::kB, units::ftm2v, units::mvv2e);
  std::printf("precision    : %s\n", cfg.precision_mode.c_str());
  std::printf("ensemble     : %s   steps=%ld   dt=%g ps (%s)\n",
              cfg.ensemble.c_str(), cfg.steps, cfg.dt, cfg.ts_mode.c_str());
  std::printf("potential    : %s  r_cut=%g  shift=%s  D=%g alpha=%g r0=%g\n",
              cfg.pot_type.c_str(), cfg.rcut, cfg.shift ? "yes" : "no",
              cfg.D, cfg.alpha, cfg.r0);

  core::AtomSoA<double> atoms;
  core::Box box;
  box.periodic = {cfg.periodic[0], cfg.periodic[1], cfg.periodic[2]};
  if (!io::read_lammps_data(cfg.geom_file, atoms, box)) {
    std::fprintf(stderr, "[fatal] cannot read geometry: %s\n", cfg.geom_file.c_str());
    return 1;
  }
  std::printf("geometry     : %s  N=%d  box=[%.4f %.4f %.4f]  pbc=[%d%d%d]\n",
              cfg.geom_file.c_str(), atoms.n, box.len(0), box.len(1), box.len(2),
              box.periodic[0], box.periodic[1], box.periodic[2]);

  // min-image validity (ConfigSchema): r_cut <= ½·min box edge
  const double min_edge = std::min({box.len(0), box.len(1), box.len(2)});
  if (cfg.rcut > 0.5 * min_edge + 1e-12) {
    std::fprintf(stderr,
                 "[fatal] r_cut=%g > half min box edge %g — min-image invalid\n",
                 cfg.rcut, 0.5 * min_edge);
    return 1;
  }

  // M2.6 (B8): init_temperature > 0 — Maxwell velocities from run.seed
  // (overrides any Velocities from the data file); 0 — keep file velocities.
  if (cfg.init_temperature > 0.0) {
    core::thermal::maxwell_init(atoms, cfg.init_temperature,
                                static_cast<uint64_t>(cfg.seed));
  }
  {
    const double T0 =
        core::thermal::temperature(atoms, core::thermal::dof_thermal(atoms.n));
    const auto p = core::thermal::momentum(atoms);
    std::printf("thermal init : %s  T(0)=%.4f K  |p|=(%.3e %.3e %.3e) amu·Å/ps\n",
                cfg.init_temperature > 0.0 ? "maxwell(seed)" : "from data file",
                T0, p[0], p[1], p[2]);
  }

  potentials::MorsePotential<double> morse{cfg.D, cfg.alpha, cfg.r0, cfg.rcut,
                                           cfg.shift};

  core::SimOptions opt;
  opt.steps       = cfg.steps;
  opt.dt          = cfg.dt;
  opt.auto_step   = (cfg.ts_mode == "auto");
  opt.ts          = {cfg.C1, cfg.K2, cfg.C3, cfg.C_buf,
                     cfg.cell_size, cfg.dt_max, 1e-6};
  opt.frame_every = cfg.traj_every;
  if (opt.auto_step)
    std::printf("auto-step    : C1=%g K2=%g C3=%g C_buf=%g cell=%g dt_max=%g\n",
                opt.ts.C1, opt.ts.K2, opt.ts.C3, opt.ts.C_buf,
                opt.ts.cell_size, opt.ts.dt_max);

  io::TrajectoryWriter writer(cfg.traj_file);
  auto res = core::run_simulation(
      atoms, box, morse, opt,
      [&](long step) { writer.write_frame(step, atoms, box); });

  if (res.halt != core::Halt::None) {
    std::fprintf(stderr, "[HALT] %s\n", res.halt_msg.c_str());
    // NonFiniteEnergy state is garbage — no rescue (restartable dump needs the
    // last finite state, backlog B9/minor-13); Overlap/Causality dump as-is.
    if (cfg.rescue_enabled && res.halt != core::Halt::NonFiniteEnergy) {
      io::write_rescue_xyz(cfg.rescue_file, atoms, box, res.halt_msg);
      std::fprintf(stderr, "[HALT] rescue dump: %s\n", cfg.rescue_file.c_str());
    }
    switch (res.halt) {
      case core::Halt::NonFiniteEnergy: return 2;
      case core::Halt::Causality:       return 3;
      case core::Halt::Overlap:         return 4;
      default:                          return 2;
    }
  }

  std::printf("step 0       : E=%.10f eV\n", res.e0);
  std::printf("after %ld    : E=%.10f  drift(max-min)=%.3e eV  rel=%.3e\n",
              res.steps_done, res.e_final, res.drift,
              res.drift / std::fabs(res.e0));
  if (opt.auto_step)
    std::printf("final dt     : %.5g ps  (v_max=%.4g Å/ps)\n",
                res.dt_final, res.v_max);
  std::printf("trajectory   : %s\n", cfg.traj_file.c_str());
  return 0;
}
