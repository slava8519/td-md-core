// M6 / SW-ladder T7: statistical RDF — the time-averaged g(r) from our SW NVE trajectory
// (started from a LAMMPS-equilibrated 600 K diamond-Si config) cosine-overlaps LAMMPS's
// time-averaged g(r) (the WHOLE curve, not just peaks). g(r) is an equilibrium ensemble
// average: though our trajectory Lyapunov-diverges from LAMMPS's, the AVERAGED curves
// coincide. CI is LAMMPS-free (frozen reference_data/sw_si/). Mirrors Test_CUDA_EAM_RDF.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include "tdmd/core/integrator.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/potentials/sw.hpp"
#include "tdmd/units.hpp"

namespace core = tdmd::core;
namespace io = tdmd::io;
namespace pot = tdmd::potentials;

namespace {
constexpr int kNbins = 80;  // matches LAMMPS `compute rdf 80`
std::string sw_dir() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT) + "/reference_data/sw_si/";
#else
  return "reference_data/sw_si/";
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
// LAMMPS fix ave/time vector file: rows "bin r g(r) coord"; g(r) is the 3rd field.
std::vector<double> load_lammps_gr(const std::string& path) {
  std::ifstream in(path);
  std::vector<double> g; std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::vector<double> v; double x;
    while (ss >> x) v.push_back(x);
    if (v.size() == 2) continue;       // the "timestep n_rows" line
    if (v.size() >= 3) g.push_back(v[2]);  // bin r g(r) coord ⇒ g(r) = v[2]
  }
  return g;
}
void serial_vv_sw(core::AtomSoA<double>& a, const core::Box& box, const pot::SwParams& sp,
                  long steps, double dt) {
  core::zero_forces(a); pot::sw_run_fixed(a, box, sp);
  for (long h = 0; h < steps; ++h) {
    for (int i = 0; i < a.n; ++i) { double im = tdmd::units::ftm2v / a.mass[i];
      a.vx[i]+=0.5*dt*im*a.fx[i]; a.vy[i]+=0.5*dt*im*a.fy[i]; a.vz[i]+=0.5*dt*im*a.fz[i];
      a.x[i]+=dt*a.vx[i]; a.y[i]+=dt*a.vy[i]; a.z[i]+=dt*a.vz[i]; }
    core::zero_forces(a); pot::sw_run_fixed(a, box, sp);
    for (int i = 0; i < a.n; ++i) { double im = tdmd::units::ftm2v / a.mass[i];
      a.vx[i]+=0.5*dt*im*a.fx[i]; a.vy[i]+=0.5*dt*im*a.fy[i]; a.vz[i]+=0.5*dt*im*a.fz[i]; }
  }
}
}  // namespace

TEST(SwRdf, StatisticalOverlapMatchesLammps) {
  core::Box box; box.periodic = {true, true, true};
  core::AtomSoA<double> a;
  ASSERT_TRUE(io::read_lammps_data(sw_dir() + "sw_rdf_216.data", a, box));
  ASSERT_EQ(a.n, 216);

  pot::SwParams sp;
  const double rmax = sp.rcut();  // LAMMPS compute-rdf default cutoff = the SW cutoff a·σ
  const int frames = 16;
  const long per = 40;
  const double dt = 1e-3;  // ps

  std::vector<double> hist(kNbins, 0.0);
  accum_hist(a, box, rmax, hist);
  for (int f = 0; f < frames; ++f) {
    serial_vv_sw(a, box, sp, per, dt);
    accum_hist(a, box, rmax, hist);
  }
  const auto g_our = normalize_gr(hist, frames + 1, a, box, rmax);
  const auto g_lmp = load_lammps_gr(sw_dir() + "sw_rdf_lammps.gr");
  ASSERT_EQ(int(g_lmp.size()), kNbins) << "LAMMPS g(r) bin count mismatch";

  const double cos = cosine_overlap(g_our, g_lmp);
  double pk_our = 0, pk_lmp = 0;
  for (int b = 0; b < kNbins; ++b) { pk_our = std::max(pk_our, g_our[b]); pk_lmp = std::max(pk_lmp, g_lmp[b]); }
  std::printf("[SW RDF] cosine overlap = %.6f | nn-peak ours=%.3f lammps=%.3f\n", cos, pk_our, pk_lmp);

  EXPECT_GE(cos, 0.99) << "cosine overlap of time-averaged g(r) vs LAMMPS = " << cos;
  // the nn-shell peak (structure amplitude) also agrees, not just position.
  EXPECT_GT(pk_our, 1.5) << "no first-neighbour peak — structure lost (melted?)";
  // amplitude tooth (cosine is scale-invariant ⇒ this is the ONLY amplitude check). Measured
  // |Δpk|≈0.11; max(0.5, 0.15·pk_lmp)≈1.0 is ~9× margin (tightened from a 18× band).
  EXPECT_NEAR(pk_our, pk_lmp, std::max(0.5, 0.15 * pk_lmp)) << "nn-peak ours=" << pk_our << " lammps=" << pk_lmp;
}
