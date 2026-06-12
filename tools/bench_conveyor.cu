// M4 bench baseline (Roadmap): "TD-конвейер vs наивный single-kernel путь".
// NOT a CI gate — записывает коэффициент накладных расходов оркестровки (A2)
// и условия измерения. Методика: docs/TD_MD_Core_Bench_v1_0.md.
//
// Both paths run the SAME kernels (zone_force / zone_integrate), so the
// conveyor/baseline ratio isolates pure orchestration cost (transport,
// events, per-zone host syncs). The force path is the tile-O(m²) zone
// kernel — the pair-list port to the GPU ring is a separate (post-M4)
// lever, so absolute atom-steps/s here are NOT the project's flagship
// numbers; the RATIO is.
//
// Usage:
//   bench_conveyor [--cells C] [--steps S] [--zones N] [--nodes Z]
//                  [--mode fp64|mixed] [--mem-probe]
// Defaults: C=14 (10976 atoms), S=30, N=8, Z=4, fp64.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "tdmd/core/conveyor.hpp"
#include "tdmd/core/thermal.hpp"
#include "tdmd/cuda/conveyor_gpu.cuh"
#include "tdmd/potentials/cutoff.hpp"
#include "tdmd/potentials/pair_lj.hpp"

using namespace tdmd;

namespace {

constexpr double kRcut = 4.0;

cuda::LJDev make_lj64() {
  potentials::LJParams<double> p{0.4, 2.55};
  return {p, potentials::CutoffScheme::make(
                 potentials::Truncation::Shift, kRcut,
                 [&](double r, double& u, double& f) {
                   potentials::pair_lj(r, p, u, f);
                 })};
}
cuda::LJDevF32 make_lj32() {
  potentials::LJParams<float> p{0.4f, 2.55f};
  return {p, potentials::CutoffScheme::make(
                 potentials::Truncation::Shift, kRcut,
                 [&](double r, double& u, double& f) {
                   float uf, ff;
                   potentials::pair_lj(float(r), p, uf, ff);
                   u = double(uf);
                   f = double(ff);
                 })};
}

core::AtomSoA<double> make_fcc(core::Box& box, int c, double a0 = 4.05) {
  core::AtomSoA<double> at;
  at.resize(4 * c * c * c);
  box.lo = {0.0, 0.0, 0.0};
  box.hi = {c * a0, c * a0, c * a0};
  box.periodic = {true, true, true};
  static const double basis[4][3] = {
      {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  int k = 0;
  for (int ix = 0; ix < c; ++ix)
    for (int iy = 0; iy < c; ++iy)
      for (int iz = 0; iz < c; ++iz)
        for (int b = 0; b < 4; ++b, ++k) {
          at.x[k] = (ix + basis[b][0]) * a0;
          at.y[k] = (iy + basis[b][1]) * a0;
          at.z[k] = (iz + basis[b][2]) * a0 + 0.25 * a0;
          at.type[k] = 1;
          at.mass[k] = 26.9815;
        }
  return at;
}

double now_s() {
  using clk = std::chrono::steady_clock;
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// Naive single-kernel path: whole system = one zone, one stream, the same
// kernels, one halt-check sync per step — what a straightforward GPU MD
// loop does. No transport, no events, no per-zone bookkeeping.
template <typename PairF>
double run_baseline(const core::AtomSoA<double>& init, const core::Box& box,
                    const PairF& pot, long steps, double dt) {
  const int n = init.n;
  const core::PairGeom geom(box, kRcut);
  auto up = [&](const std::vector<double>& v) {
    double* d = nullptr;
    cudaMalloc(&d, v.size() * 8);
    cudaMemcpy(d, v.data(), v.size() * 8, cudaMemcpyHostToDevice);
    return d;
  };
  double *x = up(init.x), *y = up(init.y), *z = up(init.z);
  double *vx = up(init.vx), *vy = up(init.vy), *vz = up(init.vz);
  double *fx = up(init.fx), *fy = up(init.fy), *fz = up(init.fz);
  double* m = up(init.mass);
  long long* raw = nullptr;
  cudaMalloc(&raw, 3LL * 8 * n);
  unsigned long long *dv2, *da2, *dkc, *dmr;
  long long *dke, *dpe;
  int* dfl;
  cudaMalloc(&dv2, 8); cudaMalloc(&da2, 8); cudaMalloc(&dkc, 8);
  cudaMalloc(&dmr, 8); cudaMalloc(&dke, 8); cudaMalloc(&dpe, 8);
  cudaMalloc(&dfl, 4);
  cuda::EndScalars* hs = nullptr;
  cudaHostAlloc(&hs, sizeof(cuda::EndScalars), 0);

  const int gi = (n + cuda::kIntBlock - 1) / cuda::kIntBlock;
  const int gf = (n + cuda::kZoneBlock - 1) / cuda::kZoneBlock;
  const unsigned long long inf = 0x7FF0000000000000ULL;

  const double t0 = now_s();
  for (long h = 1; h <= steps; ++h) {
    cuda::zone_drift_kernel<<<gi, cuda::kIntBlock>>>(x, y, z, vx, vy, vz, fx,
                                                     fy, fz, m, n, dt);
    cudaMemset(raw, 0, 3LL * 8 * n);
    cudaMemcpy(dmr, &inf, 8, cudaMemcpyHostToDevice);
    cudaMemset(dpe, 0, 8);
    cudaMemset(dfl, 0, 4);
    cuda::ZoneForceArgs args{x,   y,   z,   n,   x,   y,  z,  n,
                             raw, raw + n, raw + 2LL * n, dpe,
                             dmr, dfl, true, true};
    cuda::zone_pair_kernel<PairF><<<gf, cuda::kZoneBlock>>>(args, geom, pot);
    cuda::reset_zone_scalars_kernel<<<1, 1>>>(dv2, da2, dkc);
    cudaMemset(dke, 0, 8);
    cuda::zone_end_kernel<<<gi, cuda::kIntBlock>>>(
        vx, vy, vz, fx, fy, fz, raw, raw + n, raw + 2LL * n, m, n, dt, 50.0,
        dv2, da2, dkc, dke, dfl);
    cudaMemcpyAsync(&hs->min_r2, dmr, 8, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(&hs->flags, dfl, 4, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();  // per-step halt check, like the engine
    if (hs->flags) { std::printf("baseline: flag halt\n"); break; }
  }
  const double dt_wall = now_s() - t0;
  for (auto* p : {x, y, z, vx, vy, vz, fx, fy, fz, m}) cudaFree(p);
  cudaFree(raw);
  for (auto* p : {dv2, da2, dkc, dmr}) cudaFree(p);
  cudaFree(dke); cudaFree(dpe); cudaFree(dfl);
  cudaFreeHost(hs);
  return dt_wall;
}

template <typename PairF>
double run_ring(const core::AtomSoA<double>& init, const core::Box& box,
                const PairF& pot, long steps, double dt, int zones, int nodes,
                bool mixed) {
  core::ConveyorOptions o;
  o.steps = steps;
  o.n_zones = zones;
  o.n_nodes = nodes;
  o.auto_step = false;
  o.dt_initial = dt;
  o.mixed_transport = mixed;
  core::AtomSoA<double> a = init;
  const double t0 = now_s();
  auto r = cuda::run_conveyor_gpu(a, box, kRcut, pot, o);
  const double t = now_s() - t0;
  if (r.halt != core::Halt::None)
    std::printf("ring HALT: %s\n", r.halt_msg.c_str());
  return t;
}

// Memory checkpoint (Roadmap M4): the device-buffer budget of the ring at
// N=1e7 — allocate exactly the conveyor's slot-pool layout and report.
void mem_probe(int zones, int nodes, bool mixed) {
  const long N = 10'000'000;
  const int cap = int((N + zones - 1) / zones);
  const int S = zones + 2;
  std::size_t total = 0;
  std::vector<void*> blocks;
  auto grab = [&](std::size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
      std::printf("mem-probe: cudaMalloc FAILED at %.2f GiB\n",
                  double(total) / (1ULL << 30));
      std::exit(1);
    }
    blocks.push_back(p);
    total += bytes;
  };
  for (int k = 0; k < nodes; ++k)
    for (int s = 0; s < S; ++s) {
      grab(10ULL * 8 * cap);       // packed double arrays
      grab(3ULL * 8 * cap);        // raw accumulators
      if (mixed) grab(8ULL * cap + 36ULL * cap);  // int32 wire image
    }
  std::size_t freeb = 0, totb = 0;
  cudaMemGetInfo(&freeb, &totb);
  std::printf("mem-probe: N=%ld zones=%d nodes=%d slots/node=%d mixed=%d\n"
              "  ring buffers: %.2f GiB; device used (total-free): %.2f GiB "
              "of %.2f GiB\n",
              N, zones, nodes, S, int(mixed), double(total) / (1ULL << 30),
              double(totb - freeb) / (1ULL << 30),
              double(totb) / (1ULL << 30));
  for (void* p : blocks) cudaFree(p);
}

}  // namespace

int main(int argc, char** argv) {
  int cells = 14, zones = 8, nodes = 4;
  long steps = 30;
  bool mixed = false, probe = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&] { return std::stol(argv[++i]); };
    if (a == "--cells") cells = int(next());
    else if (a == "--steps") steps = next();
    else if (a == "--zones") zones = int(next());
    else if (a == "--nodes") nodes = int(next());
    else if (a == "--mode") mixed = (std::string(argv[++i]) == "mixed");
    else if (a == "--mem-probe") probe = true;
  }

  cudaDeviceProp p{};
  cudaGetDeviceProperties(&p, 0);
  int drv = 0, rtv = 0;
  cudaDriverGetVersion(&drv);
  cudaRuntimeGetVersion(&rtv);
  std::printf("bench_conveyor: %s, driver %d, runtime %d, %d SMs\n", p.name,
              drv, rtv, p.multiProcessorCount);

  if (probe) {
    mem_probe(zones, nodes, mixed);
    return 0;
  }

  core::Box box;
  auto atoms = make_fcc(box, cells);
  core::thermal::maxwell_init(atoms, 300.0, 7);
  // t0 forces for the baseline path (the ring computes its own)
  core::AtomSoA<double> base_init = atoms;
  {
    const auto zd = core::ZoneDecomposition::build(atoms, box, 1, kRcut);
    core::zero_forces(base_init);
    const auto lj = make_lj64();
    core::zone_force_pass(base_init, box, zd, kRcut, lj);
  }
  const double dt = 0.002;
  const char* mode = mixed ? "production_mixed" : "deterministic_fp64";
  std::printf("system: N=%d, box %.1f^3, zones=%d (width %.2f), nodes=%d, "
              "steps=%ld, dt=%g, mode=%s\n",
              atoms.n, box.len(0), zones, box.len(2) / zones, nodes, steps,
              dt, mode);

  auto bench = [&](auto pot) {
    // warmup + difference timing: t(W+S) - t(W) cancels init/t0 cost
    const long W = 5;
    run_baseline(base_init, box, pot, W, dt);  // warmup
    const double tb = run_baseline(base_init, box, pot, W + steps, dt) -
                      run_baseline(base_init, box, pot, W, dt);
    const double tr = run_ring(atoms, box, pot, W + steps, dt, zones, nodes,
                               mixed) -
                      run_ring(atoms, box, pot, W, dt, zones, nodes, mixed);
    const double as_b = double(atoms.n) * steps / tb;
    const double as_r = double(atoms.n) * steps / tr;
    std::printf("  baseline (single-kernel): %8.3f s  -> %.3e atom-steps/s\n",
                tb, as_b);
    std::printf("  TD ring  (z=%d, n=%d):     %8.3f s  -> %.3e atom-steps/s\n",
                nodes, zones, tr, as_r);
    std::printf("  ring/baseline overhead ratio: %.3f  [%s]\n", tb / tr, mode);
  };
  if (mixed) bench(make_lj32());
  else bench(make_lj64());
  return 0;
}
