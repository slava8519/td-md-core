// M3 acceptance: "таймингом подтверждён уход от O(N²)" — coarse chrono bench
// of the direct O(N²) driver vs the clustered driver on generated FCC systems.
// Not a CI gate (google/benchmark + the Bench doc land at M4); run manually:
//   ./build/bench_pair            # N ≈ 1e4 and 1e5
// Expected: direct scales ~x100 per decade of N, clustered ~x10 (linear).
#include <chrono>
#include <cstdio>
#include <cstdint>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/thermal.hpp"
#include "tdmd/potentials/morse.hpp"
#include "tdmd/potentials/clustered_morse.hpp"

using namespace tdmd;
using clk = std::chrono::steady_clock;

static core::AtomSoA<double> make_fcc(int nc, core::Box& box) {
  const double a0 = 4.05;
  core::AtomSoA<double> a;
  a.resize(4 * nc * nc * nc);
  box.lo = {0, 0, 0};
  box.hi = {nc * a0, nc * a0, nc * a0};
  box.periodic = {true, true, true};
  const double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  core::thermal::SplitMix64 rng(7);
  int k = 0;
  for (int i = 0; i < nc; ++i)
    for (int j = 0; j < nc; ++j)
      for (int l = 0; l < nc; ++l)
        for (auto& b : basis) {
          a.x[k] = (i + b[0]) * a0 + 0.1 * (2 * rng.uniform() - 1);
          a.y[k] = (j + b[1]) * a0 + 0.1 * (2 * rng.uniform() - 1);
          a.z[k] = (l + b[2]) * a0 + 0.1 * (2 * rng.uniform() - 1);
          a.mass[k] = 26.9815;
          a.type[k] = 1;
          ++k;
        }
  return a;
}

template <typename Pot>
static double time_ms(core::AtomSoA<double>& a, const core::Box& box, Pot& pot,
                      int iters) {
  core::zero_forces(a);
  pot.compute(a, box);  // warm-up (and pair-list build for the clustered path)
  const auto t0 = clk::now();
  double sink = 0;
  for (int it = 0; it < iters; ++it) {
    core::zero_forces(a);
    sink += pot.compute(a, box);
  }
  const auto t1 = clk::now();
  if (sink == 12345.6789) std::printf("#");  // defeat optimizer
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main() {
  std::printf("| N | direct O(N²), ms/step | clustered, ms/step | speedup |\n");
  std::printf("|--:|--:|--:|--:|\n");
  double prev_d = 0, prev_c = 0;
  long prev_n = 0;
  for (int nc : {14, 30}) {  // 10 976 and 108 000 atoms
    core::Box box;
    auto atoms = make_fcc(nc, box);
    auto a2 = atoms;

    potentials::MorsePotential<double> direct;
    potentials::ClusteredMorse<double> clustered;
    const int it_d = nc > 20 ? 1 : 3, it_c = nc > 20 ? 5 : 20;
    const double td = time_ms(atoms, box, direct, it_d);
    const double tc = time_ms(a2, box, clustered, it_c);
    std::printf("| %d | %.1f | %.2f | x%.0f |\n", atoms.n, td, tc, td / tc);
    if (prev_n) {
      std::printf("scaling %ld -> %d (x%.1f atoms): direct x%.1f, clustered x%.1f\n",
                  prev_n, atoms.n, double(atoms.n) / prev_n, td / prev_d, tc / prev_c);
    }
    prev_d = td; prev_c = tc; prev_n = atoms.n;
  }
  return 0;
}
