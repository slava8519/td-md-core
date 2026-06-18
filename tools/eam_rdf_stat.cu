// M6 physics acceptance (b) — STATISTICAL RDF cross-check vs LAMMPS (full-curve
// cosine overlap, not just peak bins). The E7 test (ShellPeaksMatchLammps) checks
// a SINGLE config's peak positions; here we time-average g(r) over a long NVE
// trajectory and compare the WHOLE curve to LAMMPS's time-averaged g(r).
//
// g(r) is an equilibrium ensemble average, so although our trajectory and LAMMPS's
// Lyapunov-diverge, starting from the SAME config+velocities (eam_rdf_864.data, the
// LAMMPS step-1000 state) samples the SAME equilibrium ⇒ the time-averaged curves
// must coincide. LAMMPS reference (gen_rdf.in): 50 frames over 500 steps at dt=1 fs;
// we average more frames for a smoother curve (more averaging only raises overlap).
//
// PRE-REGISTERED (M6-physics design 2026-06-18): cosine overlap ≥ 0.995.
//   build: -DTDMD_WITH_CUDA=ON, --fmad=false ; run: ./eam_rdf_stat [--steps S] [--frames F]
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
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

namespace core = tdmd::core;
namespace potentials = tdmd::potentials;
namespace io = tdmd::io;
namespace tdcu = tdmd::cuda;

namespace {
constexpr double kRmax = 6.0;
constexpr int kNbins = 100;

// LAMMPS `fix ave/time ... mode vector` over `compute rdf`: 3 comments, then
// "<step> <nrows>", then rows "<bin> <r> <g(r)> <coord>".
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

// accumulate the raw pair histogram of one config into `hist` (uniform [0,rmax)).
void accum_hist(const core::AtomSoA<double>& a, const core::Box& box,
                std::vector<double>& hist) {
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

// time-averaged g(r) = hist / (frames · N · ρ · ideal-gas shell).
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

int main(int argc, char** argv) {
  std::string data = "reference_data/eam_al/eam_rdf_864.data";
  std::string gr = "reference_data/eam_al/eam_rdf_864.gr";
  std::string setfl_path = "reference_data/eam_al/Al_zhou.eam.alloy";
  long steps = 4000;   // total NVE steps (dt = 1 fs, LAMMPS metal default)
  int frames = 100;    // g(r) samples (time-average window)
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    if (s == "--steps") steps = std::stol(argv[++i]);
    else if (s == "--frames") frames = std::stoi(argv[++i]);
    else if (s == "--data") data = argv[++i];
    else if (s == "--gr") gr = argv[++i];
  }
  cudaDeviceProp pr{}; cudaGetDeviceProperties(&pr, 0);

  core::Box box; box.periodic = {true, true, true};
  core::AtomSoA<double> a;
  if (!io::read_lammps_data(data, a, box)) { std::printf("read failed: %s\n", data.c_str()); return 2; }
  const auto setfl = potentials::EamSetfl<double>::from_setfl(setfl_path);
  const auto zd = core::ZoneDecomposition::build(a, box, 1, setfl.rcut, 2);  // z=1 whole-system EAM
  const double dt = 1e-3;  // ps (1 fs)
  const long per = steps / frames;

  std::printf("eam_rdf_stat: %s | N=%d, %ld steps (dt=1 fs), %d frames | overlap target >= 0.995\n",
              pr.name, a.n, steps, frames);

  std::vector<double> hist(kNbins, 0.0);
  accum_hist(a, box, hist);  // frame 0 = the LAMMPS step-1000 config
  for (int fme = 0; fme < frames; ++fme) {
    tdcu::eam_gpu_run_singlenode(a, box, zd, setfl, per, dt);
    accum_hist(a, box, hist);
  }
  const auto g_our = normalize_gr(hist, frames + 1, a, box);
  const auto g_lmp = load_lammps_gr(gr);
  const double cos = cosine_overlap(g_our, g_lmp);

  // also report the peak-height agreement (the nn peak ~ bin 47, r≈2.85)
  double pk_our = 0, pk_lmp = 0;
  for (int b = 40; b < 55; ++b) { pk_our = std::max(pk_our, g_our[b]); pk_lmp = std::max(pk_lmp, g_lmp[b]); }

  const bool pass = cos >= 0.995;
  std::printf("  cosine overlap = %.6f | nn-peak ours=%.3f lammps=%.3f | %s\n",
              cos, pk_our, pk_lmp, pass ? "PASS" : "FAIL");
  std::printf("M6-(b) statistical RDF: %s\n", pass ? "PASS ✓" : "FAIL ✗");
  return pass ? 0 : 1;
}
