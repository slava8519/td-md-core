// M6 physics acceptance (b) — STATISTICAL RDF cross-check vs LAMMPS.
// The E7 CPU test (ShellPeaksMatchLammps) checks a SINGLE config's peak BINS; this
// time-averages g(r) over an NVE trajectory started from the LAMMPS-equilibrated
// config (eam_rdf_864.data, with velocities = the LAMMPS step-1000 state) and
// compares the WHOLE curve to LAMMPS's time-averaged g(r) via cosine overlap.
//
// g(r) is an equilibrium ensemble average: although our trajectory Lyapunov-diverges
// from LAMMPS's, the SAME initial state samples the SAME equilibrium ⇒ the averaged
// curves coincide. Short run (a few hundred steps) — the FCC g(r) converges fast, so
// even 12 frames overlap to >0.9999; kept short to stay fast under compute-sanitizer.
// Heavy/long version: tools/eam_rdf_stat.cu (4000 steps, 100 frames → 0.999967).
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/eam_conveyor_gpu.cuh"
#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/potentials/eam_spline.hpp"

using namespace tdmd;

namespace {
constexpr double kRmax = 6.0;
constexpr int kNbins = 100;

std::string root() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT);
#else
  return ".";
#endif
}
std::string eam_dir() { return root() + "/reference_data/eam_al/"; }

std::vector<double> load_lammps_gr(const std::string& path) {
  std::ifstream f(path);
  std::string line;
  for (int i = 0; i < 3; ++i) std::getline(f, line);
  long step; int nrows;
  std::getline(f, line); { std::istringstream ss(line); ss >> step >> nrows; }
  std::vector<double> g;
  for (int i = 0; i < nrows; ++i) {
    std::getline(f, line);
    std::istringstream ss(line);
    int b; double r, gg, c; ss >> b >> r >> gg >> c;
    g.push_back(gg);
  }
  return g;
}

void accum_hist(const core::AtomSoA<double>& a, const core::Box& box, std::vector<double>& hist) {
  const double dr = kRmax / kNbins;
  const core::PairGeom geom(box, kRmax);
  for (int i = 0; i < a.n; ++i)
    for (int j = i + 1; j < a.n; ++j) {
      double dx = a.x[i] - a.x[j], dy = a.y[i] - a.y[j], dz = a.z[i] - a.z[j], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
      const int b = int(std::sqrt(r2) / dr);
      if (b >= 0 && b < kNbins) hist[b] += 2.0;
    }
}

std::vector<double> normalize_gr(const std::vector<double>& hist, int frames,
                                 const core::AtomSoA<double>& a, const core::Box& box) {
  const double dr = kRmax / kNbins;
  const double V = box.len(0) * box.len(1) * box.len(2);
  const double rho = a.n / V;
  std::vector<double> g(kNbins, 0.0);
  for (int b = 0; b < kNbins; ++b) {
    const double rlo = b * dr, rhi = rlo + dr;
    const double shell = (4.0 / 3.0) * std::numbers::pi * (rhi * rhi * rhi - rlo * rlo * rlo);
    g[b] = hist[b] / (double(frames) * a.n * rho * shell);
  }
  return g;
}

double cosine_overlap(const std::vector<double>& u, const std::vector<double>& v) {
  double uv = 0, uu = 0, vv = 0;
  for (size_t i = 0; i < u.size(); ++i) { uv += u[i] * v[i]; uu += u[i] * u[i]; vv += v[i] * v[i]; }
  return uv / std::sqrt(uu * vv);
}
}  // namespace

// time-averaged g(r) over our NVE trajectory ≡ LAMMPS's averaged g(r) (full curve).
TEST(CudaEamRdf, StatisticalOverlapMatchesLammps) {
  core::Box box; box.periodic = {true, true, true};
  core::AtomSoA<double> a;
  ASSERT_TRUE(io::read_lammps_data(eam_dir() + "eam_rdf_864.data", a, box));
  ASSERT_EQ(a.n, 864);
  const auto setfl = potentials::EamSetfl<double>::from_setfl(eam_dir() + "Al_zhou.eam.alloy");
  const auto zd = core::ZoneDecomposition::build(a, box, 1, setfl.rcut, 2);

  const int frames = 12;
  const long per = 50;          // 600 steps total at dt=1 fs
  const double dt = 1e-3;       // ps
  std::vector<double> hist(kNbins, 0.0);
  accum_hist(a, box, hist);
  for (int f = 0; f < frames; ++f) {
    cuda::eam_gpu_run_singlenode(a, box, zd, setfl, per, dt);
    accum_hist(a, box, hist);
  }
  const auto g_our = normalize_gr(hist, frames + 1, a, box);
  const auto g_lmp = load_lammps_gr(eam_dir() + "eam_rdf_864.gr");
  ASSERT_EQ(g_our.size(), g_lmp.size());

  const double cos = cosine_overlap(g_our, g_lmp);
  EXPECT_GE(cos, 0.995) << "cosine overlap of time-averaged g(r) vs LAMMPS = " << cos;

  // nn-shell peak height also agrees (structure amplitude, not just position)
  double pk_our = 0, pk_lmp = 0;
  for (int b = 40; b < 55; ++b) { pk_our = std::max(pk_our, g_our[b]); pk_lmp = std::max(pk_lmp, g_lmp[b]); }
  EXPECT_NEAR(pk_our, pk_lmp, 0.5) << "nn-peak height ours=" << pk_our << " lammps=" << pk_lmp;
}
