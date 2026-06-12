#include "tdmd/io/reader_lammps.hpp"

#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <array>
#include <algorithm>
#include <numeric>
#include <cctype>
#include <cstdio>
#include <exception>

namespace tdmd::io {

namespace {

std::vector<std::string> tokenize(const std::string& s) {
  std::vector<std::string> v;
  std::istringstream is(s);
  std::string t;
  while (is >> t) v.push_back(t);
  return v;
}

void complain(const std::string& path, const std::string& what) {
  std::fprintf(stderr, "[reader] %s: %s\n", path.c_str(), what.c_str());
}

// B10: stoi/stod on a malformed token throw — caught in read_lammps_data, so a
// broken data file reports an error instead of std::terminate.
bool parse(std::ifstream& f, const std::string& path,
           core::AtomSoA<double>& atoms, core::Box& box) {
  std::string line;
  int natoms = 0;
  std::map<int, double> type_mass;
  std::vector<int> ids, types;
  std::vector<std::array<double, 3>> pos;
  std::map<int, std::array<double, 3>> vel;  // id -> velocity (Å/ps), M2.6/B8

  std::getline(f, line);  // first line: title/comment

  enum Mode { NONE, MASSES, ATOMS, VELOCITIES } mode = NONE;
  while (std::getline(f, line)) {
    auto t = tokenize(line);
    if (t.empty()) continue;

    // --- header / section detection ---
    if (t.size() >= 2 && t[1] == "atoms") { natoms = std::stoi(t[0]); continue; }
    if (t.size() >= 3 && t[1] == "atom" && t[2] == "types") { continue; }
    if (t.size() >= 4 && t[2] == "xlo" && t[3] == "xhi") { box.lo[0]=std::stod(t[0]); box.hi[0]=std::stod(t[1]); continue; }
    if (t.size() >= 4 && t[2] == "ylo" && t[3] == "yhi") { box.lo[1]=std::stod(t[0]); box.hi[1]=std::stod(t[1]); continue; }
    if (t.size() >= 4 && t[2] == "zlo" && t[3] == "zhi") { box.lo[2]=std::stod(t[0]); box.hi[2]=std::stod(t[1]); continue; }
    if (t[0] == "Masses")     { mode = MASSES;     continue; }
    if (t[0] == "Atoms")      { mode = ATOMS;      continue; }
    if (t[0] == "Velocities") { mode = VELOCITIES; continue; }
    // any other capitalised section header resets the mode
    if (std::isalpha(static_cast<unsigned char>(t[0][0]))) { mode = NONE; continue; }

    // --- data rows ---
    if (mode == MASSES) {
      if (t.size() >= 2) type_mass[std::stoi(t[0])] = std::stod(t[1]);
    } else if (mode == ATOMS) {
      // atom_style atomic: id type x y z
      if (t.size() >= 5) {
        ids.push_back(std::stoi(t[0]));
        types.push_back(std::stoi(t[1]));
        pos.push_back({std::stod(t[2]), std::stod(t[3]), std::stod(t[4])});
      }
    } else if (mode == VELOCITIES) {
      // id vx vy vz (metal units: Å/ps)
      if (t.size() >= 4)
        vel[std::stoi(t[0])] = {std::stod(t[1]), std::stod(t[2]), std::stod(t[3])};
    }
  }

  if (natoms == 0) {
    complain(path, "no '<N> atoms' header line (or N == 0)");
    return false;
  }
  if (static_cast<int>(pos.size()) != natoms) {
    complain(path, "Atoms section has " + std::to_string(pos.size()) +
                       " rows, header declares " + std::to_string(natoms));
    return false;
  }

  // sort by atom id so index i corresponds to id i+1 (matches reference CSV order)
  std::vector<int> order(ids.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return ids[a] < ids[b]; });

  // B10: duplicate atom ids corrupt the id->index mapping — reject
  for (int k = 1; k < natoms; ++k) {
    if (ids[order[k]] == ids[order[k - 1]]) {
      complain(path, "duplicate atom id " + std::to_string(ids[order[k]]));
      return false;
    }
  }

  atoms.resize(natoms);
  for (int k = 0; k < natoms; ++k) {
    const int s = order[k];
    atoms.id[k] = ids[s];
    atoms.x[k] = pos[s][0];
    atoms.y[k] = pos[s][1];
    atoms.z[k] = pos[s][2];
    atoms.type[k] = types[s];
    auto it = type_mass.find(types[s]);
    // B10: a missing/non-positive mass would propagate to a division in the
    // integrator and silently NaN the whole run — reject at the source.
    if (it == type_mass.end() || it->second <= 0.0) {
      complain(path, "atom id " + std::to_string(ids[s]) + " has type " +
                         std::to_string(types[s]) +
                         " with missing or non-positive mass (Masses section)");
      return false;
    }
    atoms.mass[k] = it->second;
    auto vit = vel.find(ids[s]);
    if (vit != vel.end()) {
      atoms.vx[k] = vit->second[0];
      atoms.vy[k] = vit->second[1];
      atoms.vz[k] = vit->second[2];
    }
  }
  return true;
}

}  // namespace

bool read_lammps_data(const std::string& path,
                      core::AtomSoA<double>& atoms,
                      core::Box& box) {
  std::ifstream f(path);
  if (!f) {
    complain(path, "cannot open file");
    return false;
  }
  try {
    return parse(f, path, atoms, box);
  } catch (const std::exception& e) {
    complain(path, std::string("parse error: ") + e.what());
    return false;
  }
}

} // namespace tdmd::io
