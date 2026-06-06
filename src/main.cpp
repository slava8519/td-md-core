#include <cstdio>
#include <string>
#include <cmath>
#include <algorithm>

#include "tdmd/io/config.hpp"
#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/io/writer.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/integrator.hpp"
#include "tdmd/core/buffer.hpp"
#include "tdmd/potentials/morse.hpp"
#include "tdmd/io/rescue.hpp"
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

  // --- pre-start checklist (Units doc §5, ConfigSchema §3) ---
  std::printf("=== TD-MD Core — M0 walking skeleton ===\n");
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

  // M0 scope guards
  if (cfg.ensemble != "nve") {
    std::fprintf(stderr, "[fatal] M0 supports only ensemble: nve\n"); return 1;
  }
  if (cfg.pot_type != "morse") {
    std::fprintf(stderr, "[fatal] M0 supports only potential: morse\n"); return 1;
  }
  if (cfg.ts_mode == "auto" && cfg.C_buf < 1.0) {  // ConfigSchema invariant
    std::fprintf(stderr,
        "[fatal] timestep.C_buf=%g < 1.0 — buffer cannot guarantee causality\n",
        cfg.C_buf);
    return 1;
  }

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

  potentials::MorsePotential<double> morse{cfg.D, cfg.alpha, cfg.r0, cfg.rcut, cfg.shift};

  double pe = morse.compute(atoms, box);
  double ke = core::kinetic_energy(atoms);
  const double e0 = pe + ke;
  std::printf("step 0       : PE=%.10f  KE=%.10f  E=%.10f eV\n", pe, ke, e0);

  io::TrajectoryWriter writer(cfg.traj_file);
  if (cfg.traj_every > 0) writer.write_frame(0, atoms, box);

  const bool auto_step = (cfg.ts_mode == "auto");
  core::buffer::TimeStepCfg ts{cfg.C1, cfg.K2, cfg.C3, cfg.C_buf,
                               cfg.cell_size, cfg.dt_max, 1e-6};
  if (auto_step)
    std::printf("auto-step    : C1=%g K2=%g C3=%g C_buf=%g cell=%g dt_max=%g\n",
                ts.C1, ts.K2, ts.C3, ts.C_buf, ts.cell_size, ts.dt_max);

  double dt = cfg.dt;
  double v_max_prev = core::buffer::max_speed(atoms);  // local v_max (A8)
  double emin = e0, emax = e0;
  for (long step = 1; step <= cfg.steps; ++step) {
    // Buffer for this step uses the previous pass's (conservative) v_max (A8).
    const double R_buf = core::buffer::compute_R_buf(v_max_prev, dt, ts.C_buf);

    core::VelocityVerlet<double>::first_half(atoms, dt);
    pe = morse.compute(atoms, box);
    core::VelocityVerlet<double>::second_half(atoms, dt);
    ke = core::kinetic_energy(atoms);

    const double e = pe + ke;
    if (std::isnan(e) || std::isinf(e)) {
      std::fprintf(stderr, "[fatal] non-finite energy at step %ld\n", step);
      return 2;
    }

    const double v_max_now = core::buffer::max_speed(atoms);
    // INV-4: no atom may cross the buffer. step>1 skips the rest-start transient
    // where the stale v_max (=0) would give a meaningless R_buf.
    if (auto_step && step > 1 &&
        !core::buffer::causality_ok(v_max_now, dt, R_buf)) {
      std::fprintf(stderr,
          "[HALT] causality (INV-4) at step %ld: v_max·dt=%.4g > R_buf=%.4g\n",
          step, v_max_now * dt, R_buf);
      if (cfg.rescue_enabled) {
        io::write_rescue_xyz(cfg.rescue_file, atoms, box,
                             "INV-4 causality violation");
        std::fprintf(stderr, "[HALT] rescue dump: %s\n", cfg.rescue_file.c_str());
      }
      return 3;
    }

    emin = std::min(emin, e);
    emax = std::max(emax, e);
    if (cfg.traj_every > 0 && step % cfg.traj_every == 0)
      writer.write_frame(step, atoms, box);

    v_max_prev = v_max_now;
    if (auto_step)
      dt = core::buffer::auto_dt(v_max_now, dt, ts,
                                 core::buffer::temperature_limited_dt(atoms, ts.K2));
  }

  const double drift = emax - emin;
  std::printf("after %ld    : E=%.10f  drift(max-min)=%.3e eV  rel=%.3e\n",
              cfg.steps, pe + ke, drift, drift / std::fabs(e0));
  if (auto_step)
    std::printf("final dt     : %.5g ps  (v_max=%.4g Å/ps)\n", dt, v_max_prev);
  std::printf("trajectory   : %s\n", cfg.traj_file.c_str());
  return 0;
}
