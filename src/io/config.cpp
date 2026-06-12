#include "tdmd/io/config.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace tdmd::io {
namespace {

// B10 (M3 first PR): strict validation per ConfigSchema §2 — bad enum values
// and out-of-range numbers are FATAL (collected, then thrown together);
// unknown keys are WARNINGS with the key name (typo protection).

class Validator {
 public:
  void fail(const std::string& msg) { errors_.push_back(msg); }

  void check_enum(const std::string& val, const char* field,
                  std::initializer_list<const char*> allowed) {
    for (const char* a : allowed)
      if (val == a) return;
    std::string list;
    for (const char* a : allowed) {
      if (!list.empty()) list += "|";
      list += a;
    }
    fail(std::string(field) + ": '" + val + "' not in {" + list + "}");
  }

  void check(bool ok, const std::string& msg) {
    if (!ok) fail(msg);
  }

  void throw_if_failed() const {
    if (errors_.empty()) return;
    std::string all = "config validation failed:";
    for (const auto& e : errors_) all += "\n  - " + e;
    throw std::runtime_error(all);
  }

 private:
  std::vector<std::string> errors_;
};

void warn_unknown_keys(const YAML::Node& node, const char* section,
                       std::initializer_list<const char*> known) {
  if (!node || !node.IsMap()) return;
  for (const auto& kv : node) {
    const std::string k = kv.first.as<std::string>();
    const bool ok = std::any_of(known.begin(), known.end(),
                                [&](const char* s) { return k == s; });
    if (!ok)
      std::fprintf(stderr,
                   "[config] warning: unknown key '%s%s%s' — typo? (see "
                   "TD_MD_Core_ConfigSchema_v1_0.md)\n",
                   section, *section ? "." : "", k.c_str());
  }
}

bool parse_boundary(const YAML::Node& n, const char* axis, Validator& v) {
  if (!n) return true;  // default: periodic
  const std::string s = n.as<std::string>();
  if (s == "periodic") return true;
  if (s == "free") return false;
  // was: anything != "free" silently meant periodic ('fre' -> periodic). Fatal now.
  v.check_enum(s, (std::string("boundary.") + axis).c_str(),
               {"periodic", "free"});
  return true;
}

}  // namespace

Config load_config(const std::string& path) {
  YAML::Node root = YAML::LoadFile(path);
  Config c;
  Validator v;

  warn_unknown_keys(root, "",
                    {"run", "units", "precision", "geometry", "boundary",
                     "decomposition", "neighbor", "potential", "timestep",
                     "integrator", "io", "verify"});

  if (auto u = root["units"]) {
    const auto s = u.as<std::string>();
    v.check(s == "metal", "units: '" + s + "' — only 'metal' is supported");
  }

  if (auto run = root["run"]) {
    warn_unknown_keys(run, "run", {"steps", "ensemble", "seed", "init_temperature"});
    if (run["steps"])    c.steps    = run["steps"].as<long>();
    if (run["ensemble"]) c.ensemble = run["ensemble"].as<std::string>();
    if (run["seed"])     c.seed     = run["seed"].as<long>();
    if (run["init_temperature"])
      c.init_temperature = run["init_temperature"].as<double>();
  }
  v.check(c.steps >= 0, "run.steps must be >= 0");
  v.check_enum(c.ensemble, "run.ensemble", {"nve"});  // v1: NVT/NPT — backlog
  v.check(c.init_temperature >= 0.0, "run.init_temperature must be >= 0 K");

  if (auto pr = root["precision"]) {
    warn_unknown_keys(pr, "precision", {"mode", "real_type"});
    if (pr["mode"]) c.precision_mode = pr["mode"].as<std::string>();
  }
  v.check_enum(c.precision_mode, "precision.mode",
               {"production_mixed", "deterministic_fp64"});

  if (auto g = root["geometry"]) {
    warn_unknown_keys(g, "geometry", {"file", "format"});
    if (g["file"])   c.geom_file   = g["file"].as<std::string>();
    if (g["format"]) c.geom_format = g["format"].as<std::string>();
  }
  v.check(!c.geom_file.empty(), "geometry.file is required");
  v.check_enum(c.geom_format, "geometry.format", {"lammps_data"});

  if (auto b = root["boundary"]) {
    warn_unknown_keys(b, "boundary", {"x", "y", "z"});
    c.periodic[0] = parse_boundary(b["x"], "x", v);
    c.periodic[1] = parse_boundary(b["y"], "y", v);
    c.periodic[2] = parse_boundary(b["z"], "z", v);
  }

  if (auto p = root["potential"]) {
    warn_unknown_keys(p, "potential",
                      {"type", "r_cut", "shift", "morse", "eam", "table"});
    if (p["type"])  c.pot_type = p["type"].as<std::string>();
    if (p["r_cut"]) c.rcut     = p["r_cut"].as<double>();
    if (p["shift"]) c.shift    = p["shift"].as<bool>();
    if (auto m = p["morse"]) {
      warn_unknown_keys(m, "potential.morse", {"D", "alpha", "r0"});
      if (m["D"])     c.D     = m["D"].as<double>();
      if (m["alpha"]) c.alpha = m["alpha"].as<double>();
      if (m["r0"])    c.r0    = m["r0"].as<double>();
    }
  }
  v.check_enum(c.pot_type, "potential.type", {"morse"});  // lj — M3 core
  v.check(c.rcut > 0.0, "potential.r_cut must be > 0");
  v.check(c.D > 0.0 && c.alpha > 0.0 && c.r0 > 0.0,
          "potential.morse: D, alpha, r0 must all be > 0");

  if (auto t = root["timestep"]) {
    warn_unknown_keys(t, "timestep",
                      {"mode", "dt_initial", "dt_max", "C1", "K2", "C3", "C_buf"});
    if (t["mode"])       c.ts_mode = t["mode"].as<std::string>();
    if (t["dt_initial"]) c.dt      = t["dt_initial"].as<double>();
    if (t["dt_max"])     c.dt_max  = t["dt_max"].as<double>();
    if (t["C1"])         c.C1      = t["C1"].as<double>();
    if (t["K2"])         c.K2      = t["K2"].as<double>();
    if (t["C3"])         c.C3      = t["C3"].as<double>();
    if (t["C_buf"])      c.C_buf   = t["C_buf"].as<double>();
  }
  // ConfigSchema §2: unknown timestep.mode is FATAL, not a silent 'fixed'
  v.check_enum(c.ts_mode, "timestep.mode", {"fixed", "auto"});
  v.check(c.dt > 0.0, "timestep.dt_initial must be > 0");
  v.check(c.dt_max >= c.dt, "timestep.dt_max must be >= dt_initial");
  v.check(c.C1 > 0.0 && c.C1 <= 1.0, "timestep.C1 must be in (0, 1]");
  v.check(c.K2 > 0.0, "timestep.K2 must be > 0");
  v.check(c.C3 > 0.0, "timestep.C3 must be > 0");
  v.check(c.C_buf >= 1.0,
          "timestep.C_buf must be >= 1.0 — buffer cannot guarantee causality (eq.33)");

  if (auto d = root["decomposition"]) {
    warn_unknown_keys(d, "decomposition",
                      {"axis", "mode", "zone_width", "n_zones", "cell_size", "ring"});
    if (auto r = d["ring"])
      warn_unknown_keys(r, "decomposition.ring",
                        {"backend", "n_nodes", "steps_per_node", "transport"});
    if (d["cell_size"]) c.cell_size = d["cell_size"].as<double>();
  }
  v.check(c.cell_size > 0.0, "decomposition.cell_size must be > 0");

  if (auto nb = root["neighbor"]) {
    warn_unknown_keys(nb, "neighbor", {"mode", "skin"});
    if (nb["mode"]) c.neighbor_mode = nb["mode"].as<std::string>();
    if (nb["skin"]) c.skin          = nb["skin"].as<double>();
  }
  v.check_enum(c.neighbor_mode, "neighbor.mode", {"direct", "cluster"});
  v.check(c.skin > 0.0, "neighbor.skin must be > 0");

  if (auto io = root["io"]) {
    warn_unknown_keys(io, "io", {"trajectory", "rescue"});
    if (auto tr = io["trajectory"]) {
      warn_unknown_keys(tr, "io.trajectory", {"file", "every"});
      if (tr["file"])  c.traj_file  = tr["file"].as<std::string>();
      if (tr["every"]) c.traj_every = tr["every"].as<long>();
    }
    if (auto rs = io["rescue"]) {
      warn_unknown_keys(rs, "io.rescue", {"enabled", "file", "format"});
      if (rs["enabled"]) c.rescue_enabled = rs["enabled"].as<bool>();
      if (rs["file"])    c.rescue_file    = rs["file"].as<std::string>();
    }
  }
  v.check(c.traj_every >= 0, "io.trajectory.every must be >= 0");

  if (auto vf = root["verify"])
    warn_unknown_keys(vf, "verify", {"enabled", "golden", "tests"});

  v.throw_if_failed();
  return c;
}

} // namespace tdmd::io
