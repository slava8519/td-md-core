#pragma once
#include <string>
#include <array>

// Minimal config for the M0 walking skeleton. Schema: docs/TD_MD_Core_ConfigSchema_v1_0.md.
// Only the fields M0 needs are parsed; unknown fields are ignored.
namespace tdmd::io {

struct Config {
  // run
  long        steps    = 100;
  std::string ensemble = "nve";
  long        seed     = 1;
  double      init_temperature = 0.0;  // K; >0 — Maxwell init from seed (M2.6/B8)
  std::string precision_mode = "deterministic_fp64";
  // geometry
  std::string geom_file;
  std::string geom_format = "lammps_data";
  std::array<bool, 3> periodic{true, true, true};
  // potential (morse | lj)
  std::string pot_type = "morse";
  double rcut = 4.0;
  // cutoff scheme: cut | shift | force_shift (cutoff.hpp). Resolved here:
  // legacy `potential.shift: bool` maps to shift/cut, `potential.truncation`
  // wins when both are given.
  std::string truncation = "shift";
  double D = 0.29614, alpha = 1.11892, r0 = 3.29692;
  double lj_epsilon = 1.0, lj_sigma = 1.0;  // reduced-unit defaults
  // timestep
  std::string ts_mode = "fixed";  // fixed | auto
  double dt = 0.005;              // ps (dt_initial)
  double dt_max = 0.02;          // ps
  double C1 = 0.1, K2 = 50.0, C3 = 0.5, C_buf = 1.5;  // auto-step coeffs (M2)
  double cell_size = 2.33;       // Å, spatial cell (decomposition.cell_size)
  // neighbor (M3)
  std::string neighbor_mode = "direct";  // direct (O(N²) reference) | cluster
  double skin = 1.0;                     // Å, pair-list skin (neighbor.skin)
  // io
  std::string traj_file  = "traj.lammpstrj";
  long        traj_every = 50;
  bool        rescue_enabled = true;
  std::string rescue_file = "rescue.xyz";
};

// Loads and validates a config.yaml. Throws std::runtime_error on fatal problems.
Config load_config(const std::string& path);

} // namespace tdmd::io
