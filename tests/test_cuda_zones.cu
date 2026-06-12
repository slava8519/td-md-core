// M4 (label: cuda) — B1 fixed-point zone forces on GPU (cuda/zone_force.cuh).
//
// The headline assertion: with --fmad=false (this TU; CPU is project-wide
// -ffp-contract=off) and a transcendental-free potential (LJ), the GPU raw
// int64 force/energy accumulators equal the CPU zone path BIT-FOR-BIT —
// quantization per pair contribution + integer associativity make the sum
// independent of tiles/blocks/launch order on either platform. Morse (exp:
// libm vs CUDA differ in ulps) is held to a 1e-12 tolerance vs CPU plus
// GPU-internal bitwise checks (run-to-run, launch-order independence).
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/thermal.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/zone_force.cuh"
#include "tdmd/potentials/cutoff.hpp"
#include "tdmd/potentials/morse.hpp"
#include "tdmd/potentials/pair_lj.hpp"

using namespace tdmd;

namespace {

// --- shared single-source pair functors (host + device, zone_force.cuh) ---

using cuda::LJDev;
using cuda::MorseDev;

LJDev make_lj(double eps, double sigma, double rcut) {
  potentials::LJParams<double> p{eps, sigma};
  return {p, potentials::CutoffScheme::make(
                 potentials::Truncation::Shift, rcut,
                 [&](double r, double& u, double& f) {
                   potentials::pair_lj(r, p, u, f);
                 })};
}
MorseDev make_morse(double rcut) {
  potentials::MorseParams<double> p{0.29614, 1.11892, 3.29692};
  return {p, potentials::CutoffScheme::make(
                 potentials::Truncation::Shift, rcut,
                 [&](double r, double& u, double& f) {
                   potentials::pair_morse(r, p, u, f);
                 })};
}

constexpr double kRcut = 4.0;

core::AtomSoA<double> make_fcc(core::Box& box, int cx, int cy, int cz,
                               double a0 = 4.05) {
  core::AtomSoA<double> at;
  at.resize(4 * cx * cy * cz);
  box.lo = {0.0, 0.0, 0.0};
  box.hi = {cx * a0, cy * a0, cz * a0};
  box.periodic = {true, true, true};
  static const double basis[4][3] = {
      {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  int k = 0;
  for (int ix = 0; ix < cx; ++ix)
    for (int iy = 0; iy < cy; ++iy)
      for (int iz = 0; iz < cz; ++iz)
        for (int b = 0; b < 4; ++b, ++k) {
          at.x[k] = (ix + basis[b][0]) * a0;
          at.y[k] = (iy + basis[b][1]) * a0;
          at.z[k] = (iz + basis[b][2]) * a0 + 0.25 * a0;
          at.type[k] = 1;
          at.mass[k] = 26.9815;
        }
  return at;
}

// --- CPU mirror with RAW accumulators (the exact zones.hpp pass structure,
// fixed::FixedAccum exposed) — also tied to the canonical zone_force_pass
// below by a bitwise value() comparison in each test. ---

struct RawForces {
  std::vector<std::vector<long long>> fx, fy, fz;  // [zone][member]
  long long pe = 0;
};

template <typename Pair>
RawForces cpu_raw_pass(const core::AtomSoA<double>& a, const core::Box& box,
                       const core::ZoneDecomposition& zd, double rcut,
                       const Pair& pair) {
  const core::PairGeom geom(box, rcut);
  const int n = zd.n_zones;
  RawForces out;
  std::vector<std::vector<core::fixed::ForceAccum>> ax(n), ay(n), az(n);
  for (int zi = 0; zi < n; ++zi) {
    ax[zi].assign(zd.members[zi].size(), {});
    ay[zi].assign(zd.members[zi].size(), {});
    az[zi].assign(zd.members[zi].size(), {});
  }
  core::fixed::EnergyAccum pe;
  auto do_pair = [&](int za, int s, int zb, int t) {
    const int i = zd.members[za][s], j = zd.members[zb][t];
    double dx = a.x[i] - a.x[j];
    double dy = a.y[i] - a.y[j];
    double dz = a.z[i] - a.z[j];
    double r2;
    if (!geom.reduce(dx, dy, dz, r2)) return;
    double u, f_over_r;
    pair(std::sqrt(r2), u, f_over_r);
    pe.add(u);
    ax[za][s].add(f_over_r * dx);
    ay[za][s].add(f_over_r * dy);
    az[za][s].add(f_over_r * dz);
    ax[zb][t].add(-f_over_r * dx);
    ay[zb][t].add(-f_over_r * dy);
    az[zb][t].add(-f_over_r * dz);
  };
  for (int zi = 0; zi < n; ++zi) {
    const int m = int(zd.members[zi].size());
    for (int s = 0; s < m; ++s)
      for (int t = s + 1; t < m; ++t) do_pair(zi, s, zi, t);
    if (n > 1) {
      const int zn = (zi + 1) % n;
      if (zn > zi || box.periodic[2])
        for (int s = 0; s < m; ++s)
          for (int t = 0; t < int(zd.members[zn].size()); ++t)
            do_pair(zi, s, zn, t);
    }
  }
  out.fx.resize(n);
  out.fy.resize(n);
  out.fz.resize(n);
  for (int zi = 0; zi < n; ++zi) {
    for (auto& v : ax[zi]) out.fx[zi].push_back(v.raw);
    for (auto& v : ay[zi]) out.fy[zi].push_back(v.raw);
    for (auto& v : az[zi]) out.fz[zi].push_back(v.raw);
  }
  out.pe = pe.raw;
  return out;
}

// --- GPU runner: per-zone buffers, internal + two-sided cross launches in
// the SAME pass structure as the CPU. ---

template <typename T>
T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  EXPECT_EQ(cudaMalloc(&d, v.size() * sizeof(T)), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(d, v.data(), v.size() * sizeof(T),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  return d;
}

struct GpuPassResult {
  RawForces raw;
  double min_r2 = 0.0;
  int overflow = 0;
};

template <typename PairF>
GpuPassResult gpu_raw_pass(const core::AtomSoA<double>& a,
                           const core::Box& box,
                           const core::ZoneDecomposition& zd, double rcut,
                           const PairF& pot, bool reverse_zone_order = false) {
  const core::PairGeom geom(box, rcut);
  const int n = zd.n_zones;
  struct ZBuf {
    double *x, *y, *z;
    long long *fx, *fy, *fz;
    int m;
  };
  std::vector<ZBuf> zb(n);
  for (int zi = 0; zi < n; ++zi) {
    const auto& mem = zd.members[zi];
    std::vector<double> hx, hy, hz;
    for (int i : mem) {
      hx.push_back(a.x[i]);
      hy.push_back(a.y[i]);
      hz.push_back(a.z[i]);
    }
    std::vector<long long> zero(mem.size(), 0);
    zb[zi] = {upload(hx), upload(hy), upload(hz),
              upload(zero), upload(zero), upload(zero), int(mem.size())};
  }
  std::vector<long long> pe0{0};
  std::vector<unsigned long long> mr0{0x7FF0000000000000ULL};  // +inf bits
  std::vector<int> of0{0};
  long long* dpe = upload(pe0);
  unsigned long long* dmr = upload(mr0);
  int* dof = upload(of0);

  auto launch = [&](int za, int zo, bool same, bool energy) {
    if (zb[za].m == 0) return;
    cuda::ZoneForceArgs args{
        zb[za].x,  zb[za].y,  zb[za].z,  zb[za].m,
        zb[zo].x,  zb[zo].y,  zb[zo].z,  zb[zo].m,
        zb[za].fx, zb[za].fy, zb[za].fz, dpe, dmr, dof, same, energy};
    const int grid = (zb[za].m + cuda::kZoneBlock - 1) / cuda::kZoneBlock;
    cuda::zone_pair_kernel<PairF><<<grid, cuda::kZoneBlock>>>(args, geom, pot);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  };

  for (int q = 0; q < n; ++q) {
    const int zi = reverse_zone_order ? n - 1 - q : q;
    launch(zi, zi, /*same=*/true, /*energy=*/true);
    if (n > 1) {
      const int zn = (zi + 1) % n;
      if (zn > zi || box.periodic[2]) {
        launch(zi, zn, false, true);   // A-side: forces to zi, energy once
        launch(zn, zi, false, false);  // B-side: Newton-3 partner forces
      }
    }
  }
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  GpuPassResult res;
  res.raw.fx.resize(n);
  res.raw.fy.resize(n);
  res.raw.fz.resize(n);
  for (int zi = 0; zi < n; ++zi) {
    const int m = zb[zi].m;
    res.raw.fx[zi].resize(m);
    res.raw.fy[zi].resize(m);
    res.raw.fz[zi].resize(m);
    EXPECT_EQ(cudaMemcpy(res.raw.fx[zi].data(), zb[zi].fx, m * 8,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(cudaMemcpy(res.raw.fy[zi].data(), zb[zi].fy, m * 8,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(cudaMemcpy(res.raw.fz[zi].data(), zb[zi].fz, m * 8,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (auto* p : {zb[zi].x, zb[zi].y, zb[zi].z}) cudaFree(p);
    for (auto* p : {zb[zi].fx, zb[zi].fy, zb[zi].fz}) cudaFree(p);
  }
  unsigned long long mr = 0;
  EXPECT_EQ(cudaMemcpy(&res.raw.pe, dpe, 8, cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(&mr, dmr, 8, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(&res.overflow, dof, 4, cudaMemcpyDeviceToHost),
            cudaSuccess);
  std::memcpy(&res.min_r2, &mr, 8);
  cudaFree(dpe);
  cudaFree(dmr);
  cudaFree(dof);
  return res;
}

::testing::AssertionResult raw_eq(const RawForces& a, const RawForces& b) {
  if (a.pe != b.pe)
    return ::testing::AssertionFailure()
           << "pe raw mismatch: " << a.pe << " vs " << b.pe;
  for (std::size_t z = 0; z < a.fx.size(); ++z)
    for (std::size_t i = 0; i < a.fx[z].size(); ++i)
      if (a.fx[z][i] != b.fx[z][i] || a.fy[z][i] != b.fy[z][i] ||
          a.fz[z][i] != b.fz[z][i])
        return ::testing::AssertionFailure()
               << "raw force mismatch at zone " << z << " member " << i;
  return ::testing::AssertionSuccess();
}

}  // namespace

// --- the headline: LJ raw accumulators, CPU == GPU bit-for-bit ---

TEST(CudaZones, LjRawAccumulatorsBitwiseCpuVsGpu) {
  core::Box box;
  auto a = make_fcc(box, 2, 2, 6);
  core::thermal::maxwell_init(a, 300.0, 31);
  const auto zd = core::ZoneDecomposition::build(a, box, 4, kRcut);
  const auto lj = make_lj(0.4, 2.55, kRcut);

  const auto cpu = cpu_raw_pass(a, box, zd, kRcut, lj);
  const auto gpu = gpu_raw_pass(a, box, zd, kRcut, lj);
  EXPECT_EQ(gpu.overflow, 0);
  EXPECT_TRUE(raw_eq(cpu, gpu.raw));

  // Tie the raw mirror to the canonical CPU path: value() of the raws must
  // reproduce zone_force_pass output doubles bitwise.
  core::AtomSoA<double> b = a;
  core::zero_forces(b);
  const double pe = core::zone_force_pass(b, box, zd, kRcut, lj);
  EXPECT_EQ(pe, double(cpu.pe) / core::fixed::EnergyAccum::kScale);
  for (int zi = 0; zi < zd.n_zones; ++zi)
    for (std::size_t s = 0; s < zd.members[zi].size(); ++s) {
      const int i = zd.members[zi][s];
      EXPECT_EQ(b.fx[i],
                double(cpu.fx[zi][s]) / core::fixed::ForceAccum::kScale);
      EXPECT_EQ(b.fz[i],
                double(cpu.fz[zi][s]) / core::fixed::ForceAccum::kScale);
    }
}

// --- Morse: tolerance vs CPU (exp ulps), bitwise GPU-internal ---

TEST(CudaZones, MorseToleranceAndGpuBitwiseInternals) {
  core::Box box;
  auto a = make_fcc(box, 4, 4, 8);  // 512 atoms, 4 zones of 8.1 Å
  core::thermal::maxwell_init(a, 300.0, 37);
  const auto zd = core::ZoneDecomposition::build(a, box, 4, kRcut);
  const auto mo = make_morse(kRcut);

  const auto g1 = gpu_raw_pass(a, box, zd, kRcut, mo);
  EXPECT_EQ(g1.overflow, 0);

  // run-to-run and launch-order independence: bitwise on raw accumulators
  const auto g2 = gpu_raw_pass(a, box, zd, kRcut, mo);
  const auto g3 = gpu_raw_pass(a, box, zd, kRcut, mo, /*reverse=*/true);
  EXPECT_TRUE(raw_eq(g1.raw, g2.raw));
  EXPECT_TRUE(raw_eq(g1.raw, g3.raw));

  // vs CPU: transcendental-limited tolerance
  core::AtomSoA<double> b = a;
  core::zero_forces(b);
  const double pe_cpu = core::zone_force_pass(b, box, zd, kRcut, mo);
  double maxd = 0.0;
  for (int zi = 0; zi < zd.n_zones; ++zi)
    for (std::size_t s = 0; s < zd.members[zi].size(); ++s) {
      const int i = zd.members[zi][s];
      maxd = std::max(
          {maxd,
           std::fabs(b.fx[i] - double(g1.raw.fx[zi][s]) /
                                   core::fixed::ForceAccum::kScale),
           std::fabs(b.fy[i] - double(g1.raw.fy[zi][s]) /
                                   core::fixed::ForceAccum::kScale),
           std::fabs(b.fz[i] - double(g1.raw.fz[zi][s]) /
                                   core::fixed::ForceAccum::kScale)});
    }
  const double pe_gpu = double(g1.raw.pe) / core::fixed::EnergyAccum::kScale;
  std::printf("Test_CUDA_Zones[morse]: GPU-vs-CPU max|dF|=%.3e eV/A  "
              "|dPE|=%.3e eV\n",
              maxd, std::fabs(pe_gpu - pe_cpu));
  EXPECT_LT(maxd, 1e-12);
  EXPECT_LT(std::fabs(pe_gpu - pe_cpu), 1e-10);
}

// --- B1 overflow: steep LJ at sub-Å distance must set the sticky flag,
// never silently wrap (the CPU twin throws — review M3.5). The coincident-
// pair min_r2 channel is checked alongside. ---

TEST(CudaZones, OverflowFlagAndMinR2) {
  core::Box box;
  box.lo = {0.0, 0.0, 0.0};
  box.hi = {20.0, 20.0, 20.0};
  box.periodic = {false, false, false};
  core::AtomSoA<double> a;
  a.resize(2);
  a.x = {10.0, 10.5};  // r = 0.5 Å: |F_LJ| >> 2^23 eV/Å
  a.y = {10.0, 10.0};
  a.z = {10.0, 10.0};
  a.type = {1, 1};
  a.mass = {1.0, 1.0};
  const auto zd = core::ZoneDecomposition::build(a, box, 1, kRcut);
  const auto lj = make_lj(0.0104, 3.4, kRcut);  // Ar-like, metal units

  const auto g = gpu_raw_pass(a, box, zd, kRcut, lj);
  EXPECT_EQ(g.overflow, 1);
  EXPECT_NEAR(g.min_r2, 0.25, 1e-12);

  // CPU twin throws on the same contribution (FixedAccum range guard)
  core::AtomSoA<double> b = a;
  core::zero_forces(b);
  EXPECT_THROW(core::zone_force_pass(b, box, zd, kRcut, lj),
               std::overflow_error);
}

// --- Tier-2 bookkeeping for the new kernel (regs / theoretical occupancy);
// achieved occupancy via ncu is recorded in _meta/ncu_occupancy (M4 bench). ---

TEST(CudaZones, RegistersAndOccupancy) {
  cudaDeviceProp p;
  ASSERT_EQ(cudaGetDeviceProperties(&p, 0), cudaSuccess);
  for (auto [name, fn] :
       {std::pair{"lj", (const void*)cuda::zone_pair_kernel<LJDev>},
        std::pair{"morse", (const void*)cuda::zone_pair_kernel<MorseDev>}}) {
    cudaFuncAttributes attr;
    ASSERT_EQ(cudaFuncGetAttributes(&attr, fn), cudaSuccess);
    int blocks = 0;
    ASSERT_EQ(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &blocks, fn, cuda::kZoneBlock, attr.sharedSizeBytes),
              cudaSuccess);
    const double occ =
        100.0 * blocks * cuda::kZoneBlock / p.maxThreadsPerMultiProcessor;
    std::printf("Test_CUDA_Zones[%s]: regs/thread=%d smem=%zu B blocks/SM=%d "
                "occupancy=%.0f%% [fmad=false verify build]\n",
                name, attr.numRegs, attr.sharedSizeBytes, blocks, occ);
    EXPECT_GE(occ, 50.0);  // A2 red-flag threshold (M2.5)
  }
}
