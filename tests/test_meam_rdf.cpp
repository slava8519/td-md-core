// M6 / MEAM-ladder Me7: statistical RDF — the time-averaged g(r) from our MEAM NVE trajectory
// (started from a LAMMPS-equilibrated 600 K diamond-Si config) cosine-overlaps LAMMPS's time-
// averaged g(r) (the WHOLE curve, both the 1NN and 2NN shells). g(r) is an equilibrium ensemble
// average: though our trajectory Lyapunov-diverges from LAMMPS's, the AVERAGED curves coincide. CI
// is LAMMPS-free (frozen reference_data/meam_si/). Closes the MEAM suite (forces golden = Me1/Me2
// Test_MEAM_LAMMPS; structure = here). Mirrors Test_Tersoff_RDF / Test_SW_RDF / EAM E7.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/potentials/meam.hpp"
#include "tdmd/units.hpp"

namespace core = tdmd::core;
namespace io = tdmd::io;
namespace pot = tdmd::potentials;

namespace {
constexpr int kNbins = 80;  // matches LAMMPS `compute rdf 80`
std::string meam_dir() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT) + "/reference_data/meam_si/";
#else
  return "reference_data/meam_si/";
#endif
}
void accum_hist(const core::AtomSoA<double>& a, const core::Box& box, double rmax,
                std::vector<double>& hist) {
  const double dr = rmax / kNbins;
  const core::PairGeom geom(box, rmax);
  for (int i = 0; i < a.n; ++i)
    for (int j = i + 1; j < a.n; ++j) {
      double dx = a.x[i] - a.x[j], dy = a.y[i] - a.y[j], dz = a.z[i] - a.z[j], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
      const int b = int(std::sqrt(r2) / dr);
      if (b >= 0 && b < kNbins) hist[b] += 2.0;
    }
}
std::vector<double> normalize_gr(const std::vector<double>& hist, int frames,
                                 const core::AtomSoA<double>& a, const core::Box& box, double rmax) {
  const double dr = rmax / kNbins;
  const double rho = a.n / (box.len(0) * box.len(1) * box.len(2));
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
std::vector<double> load_lammps_gr(const std::string& path) {
  std::ifstream in(path);
  std::vector<double> g; std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::vector<double> v; double x;
    while (ss >> x) v.push_back(x);
    if (v.size() == 2) continue;            // the "timestep n_rows" line
    if (v.size() >= 3) g.push_back(v[2]);   // bin r g(r) coord ⇒ g(r) = v[2]
  }
  return g;
}
void serial_vv_meam(core::AtomSoA<double>& a, const core::PairGeom& geom,
                    const pot::MeamParams& p, long steps, double dt) {
  core::zero_forces(a); pot::meam_run_fixed_force(a, geom, p);
  for (long h = 0; h < steps; ++h) {
    for (int i = 0; i < a.n; ++i) { double im = tdmd::units::ftm2v / a.mass[i];
      a.vx[i]+=0.5*dt*im*a.fx[i]; a.vy[i]+=0.5*dt*im*a.fy[i]; a.vz[i]+=0.5*dt*im*a.fz[i];
      a.x[i]+=dt*a.vx[i]; a.y[i]+=dt*a.vy[i]; a.z[i]+=dt*a.vz[i]; }
    core::zero_forces(a); pot::meam_run_fixed_force(a, geom, p);
    for (int i = 0; i < a.n; ++i) { double im = tdmd::units::ftm2v / a.mass[i];
      a.vx[i]+=0.5*dt*im*a.fx[i]; a.vy[i]+=0.5*dt*im*a.fy[i]; a.vz[i]+=0.5*dt*im*a.fz[i]; }
  }
}
}  // namespace

TEST(MeamRdf, StatisticalOverlapMatchesLammps) {
  core::Box box; box.periodic = {true, true, true};
  core::AtomSoA<double> a;
  ASSERT_TRUE(io::read_lammps_data(meam_dir() + "meam_rdf_216.data", a, box));
  ASSERT_EQ(a.n, 216);
  // The .data carries LAMMPS's 600 K velocities (Velocities section) — use them directly so our NVE
  // samples the SAME thermal state (g(r) is an ensemble average; our trajectory Lyapunov-diverges
  // but the averaged curve coincides). Zeroing them would run a COLDER ensemble (sharper peaks).

  pot::MeamParams p;
  const double rmax = p.rc;  // LAMMPS compute-rdf cutoff = the MEAM cutoff rc = 4.0
  const core::PairGeom geom(box, rmax);
  const int frames = 16;
  const long per = 40;
  const double dt = 1e-3;  // ps

  std::vector<double> hist(kNbins, 0.0);
  accum_hist(a, box, rmax, hist);
  for (int f = 0; f < frames; ++f) {
    serial_vv_meam(a, geom, p, per, dt);
    accum_hist(a, box, rmax, hist);
  }
  const auto g_our = normalize_gr(hist, frames + 1, a, box, rmax);
  const auto g_lmp = load_lammps_gr(meam_dir() + "meam_rdf_lammps.gr");
  ASSERT_EQ(int(g_lmp.size()), kNbins) << "LAMMPS g(r) bin count mismatch";

  const double cos = cosine_overlap(g_our, g_lmp);
  double pk_our = 0, pk_lmp = 0;
  for (int b = 0; b < kNbins; ++b) { pk_our = std::max(pk_our, g_our[b]); pk_lmp = std::max(pk_lmp, g_lmp[b]); }
  std::printf("[MEAM RDF] cosine overlap = %.6f | nn-peak ours=%.3f lammps=%.3f\n", cos, pk_our, pk_lmp);

  EXPECT_GE(cos, 0.99) << "cosine overlap of time-averaged g(r) vs LAMMPS = " << cos;
  EXPECT_GT(pk_our, 1.5) << "no first-neighbour peak — structure lost (melted?)";
  EXPECT_NEAR(pk_our, pk_lmp, std::max(0.8, 0.20 * pk_lmp)) << "nn-peak ours=" << pk_our << " lammps=" << pk_lmp;

  // teeth: a scrambled (uniform-gas) g(r) must NOT overlap — the gate discriminates structure.
  std::vector<double> flat(kNbins, 1.0);
  EXPECT_LT(cosine_overlap(g_our, flat), 0.95) << "g(r) indistinguishable from a flat gas — no structure";
}
