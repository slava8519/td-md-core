#include "tdmd/io/config.hpp"

#include <yaml-cpp/yaml.h>
#include <stdexcept>

namespace tdmd::io {

static bool periodic_of(const YAML::Node& n, bool def) {
  if (!n) return def;
  return n.as<std::string>("periodic") != "free";
}

Config load_config(const std::string& path) {
  YAML::Node root = YAML::LoadFile(path);
  Config c;

  if (auto run = root["run"]) {
    if (run["steps"])    c.steps    = run["steps"].as<long>();
    if (run["ensemble"]) c.ensemble = run["ensemble"].as<std::string>();
    if (run["seed"])     c.seed     = run["seed"].as<long>();
  }
  if (auto pr = root["precision"]) {
    if (pr["mode"]) c.precision_mode = pr["mode"].as<std::string>();
  }
  if (auto g = root["geometry"]) {
    if (g["file"])   c.geom_file   = g["file"].as<std::string>();
    if (g["format"]) c.geom_format = g["format"].as<std::string>();
  }
  if (auto b = root["boundary"]) {
    c.periodic[0] = periodic_of(b["x"], true);
    c.periodic[1] = periodic_of(b["y"], true);
    c.periodic[2] = periodic_of(b["z"], true);
  }
  if (auto p = root["potential"]) {
    if (p["type"])  c.pot_type = p["type"].as<std::string>();
    if (p["r_cut"]) c.rcut     = p["r_cut"].as<double>();
    if (p["shift"]) c.shift    = p["shift"].as<bool>();
    if (auto m = p["morse"]) {
      if (m["D"])     c.D     = m["D"].as<double>();
      if (m["alpha"]) c.alpha = m["alpha"].as<double>();
      if (m["r0"])    c.r0    = m["r0"].as<double>();
    }
  }
  if (auto t = root["timestep"]) {
    if (t["mode"])       c.ts_mode = t["mode"].as<std::string>();
    if (t["dt_initial"]) c.dt      = t["dt_initial"].as<double>();
  }
  if (auto io = root["io"]) {
    if (auto tr = io["trajectory"]) {
      if (tr["file"])  c.traj_file  = tr["file"].as<std::string>();
      if (tr["every"]) c.traj_every = tr["every"].as<long>();
    }
  }

  if (c.geom_file.empty())
    throw std::runtime_error("config: geometry.file is required");
  return c;
}

} // namespace tdmd::io
