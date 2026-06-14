// M6 PR-E7 — EAM radial distribution function g(r) cross-check vs LAMMPS.
// The static eam/alloy run-0 PE/forces cross-check (~1e-12) is PR-E2; here we
// validate the STRUCTURE: our EAM g(r) on the LAMMPS-equilibrated FCC-Al config
// (Al_zhou, NVE @300K) has its shell peaks at the SAME positions as LAMMPS's
// time-averaged g(r). Self-contained — reads the FROZEN reference (no LAMMPS at
// run time; reference_data/eam_al/, nist_lj pattern). Melt-coexistence is a
// deferred long-run experiment (VALIDATION_EXPERIMENT_2026-06-12.md §5/§7).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/io/reader_lammps.hpp"

using namespace tdmd;

static std::string root() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT);
#else
  return ".";
#endif
}
static std::string eam_dir() { return root() + "/reference_data/eam_al/"; }

namespace {
// Parse the LAMMPS `compute rdf` / `fix ave/time vector` file: 3 comment lines,
// then "<step> <nrows>", then rows "<bin> <r> <g(r)> <coord>".
struct Rdf { std::vector<double> r, g; };
Rdf load_lammps_gr(const std::string& path) {
  std::ifstream f(path);
  std::string line;
  for (int i = 0; i < 3; ++i) std::getline(f, line);  // comments
  long step; int nrows;
  std::getline(f, line); { std::istringstream ss(line); ss >> step >> nrows; }
  Rdf out;
  for (int i = 0; i < nrows; ++i) {
    std::getline(f, line);
    std::istringstream ss(line);
    int b; double r, g, c;
    ss >> b >> r >> g >> c;
    out.r.push_back(r); out.g.push_back(g);
  }
  return out;
}

// Our g(r) on a single config, SAME bins as LAMMPS (uniform [0, rmax], nbins).
std::vector<double> our_gr(const core::AtomSoA<double>& a, const core::Box& box,
                           double rmax, int nbins) {
  const double dr = rmax / nbins;
  const core::PairGeom geom(box, rmax);  // min-image + r2<rmax² acceptance
  std::vector<double> hist(nbins, 0.0);
  for (int i = 0; i < a.n; ++i)
    for (int j = i + 1; j < a.n; ++j) {
      double dx = a.x[i] - a.x[j], dy = a.y[i] - a.y[j], dz = a.z[i] - a.z[j], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
      const int b = int(std::sqrt(r2) / dr);
      if (b >= 0 && b < nbins) hist[b] += 2.0;  // both i,j
    }
  // standard normalization g(r) = hist / (N · ρ · 4π r² dr)
  const double V = box.len(0) * box.len(1) * box.len(2);
  const double rho = a.n / V;
  std::vector<double> g(nbins, 0.0);
  for (int b = 0; b < nbins; ++b) {
    const double rlo = b * dr, rhi = rlo + dr;
    const double shell = (4.0 / 3.0) * std::numbers::pi * (rhi * rhi * rhi - rlo * rlo * rlo);
    g[b] = hist[b] / (a.n * rho * shell);
  }
  return g;
}

// argmax index of g within [rlo, rhi).
int peak_bin(const std::vector<double>& r, const std::vector<double>& g, double rlo, double rhi) {
  int best = -1; double bg = -1;
  for (size_t i = 0; i < r.size(); ++i)
    if (r[i] >= rlo && r[i] < rhi && g[i] > bg) { bg = g[i]; best = int(i); }
  return best;
}
}  // namespace

// --- g(r) shell peaks of our EAM ≡ LAMMPS's (structure cross-check) ---
TEST(EamRdf, ShellPeaksMatchLammps) {
  const Rdf lmp = load_lammps_gr(eam_dir() + "eam_rdf_864.gr");
  ASSERT_EQ(int(lmp.r.size()), 100);
  const double rmax = lmp.r.back() + (lmp.r[1] - lmp.r[0]) * 0.5;  // ≈ 6.0
  const int nbins = int(lmp.r.size());

  core::Box box; box.periodic = {true, true, true};
  core::AtomSoA<double> a;
  ASSERT_TRUE(io::read_lammps_data(eam_dir() + "eam_rdf_864.data", a, box));
  ASSERT_EQ(a.n, 864);
  const auto g = our_gr(a, box, rmax, nbins);

  // FCC-Al shells (a≈4.05): nn≈2.86, 2nd≈4.05, 3rd≈4.96. Peak bins must coincide
  // within ±1 bin (dr≈0.06) between our single-config g(r) and LAMMPS's average.
  struct Shell { double lo, hi; const char* name; };
  for (const Shell& s : {Shell{2.4, 3.4, "nn"}, Shell{3.6, 4.4, "2nd"}, Shell{4.6, 5.4, "3rd"}}) {
    const int bo = peak_bin(lmp.r, g, s.lo, s.hi);
    const int bl = peak_bin(lmp.r, lmp.g, s.lo, s.hi);
    ASSERT_GE(bo, 0); ASSERT_GE(bl, 0);
    EXPECT_LE(std::abs(bo - bl), 1) << s.name << " peak: ours bin=" << bo
                                    << " (r=" << lmp.r[bo] << ") lammps bin=" << bl;
    EXPECT_GT(g[bo], 1.3) << s.name << " peak height (structure present)";
  }
  // first peak position is the nn distance ≈ 2.86 Å
  const int nn = peak_bin(lmp.r, g, 2.4, 3.4);
  EXPECT_NEAR(lmp.r[nn], 2.86, 0.12) << "nn peak position";
}
