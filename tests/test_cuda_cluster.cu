// M3 Tier-2 (label: cuda): prototype clustered force kernel over the SAME
// pair-list format and the SAME pair_morse functor as the CPU drivers — run
// BEFORE freezing the cluster-pair list format (audit M2): measures real
// regs/thread (cudaFuncGetAttributes) and theoretical occupancy
// (cudaOccupancyMaxActiveBlocksPerMultiprocessor) and cross-checks forces
// against the CPU clustered driver.
//
// Mapping (design rule from the measured 24-blocks/SM cap): 1 warp = 1 cluster,
// block 128 = 4 clusters => 12 blocks/SM, 100% occupancy if regs allow.
// FULL-neighbour list => each thread owns atom i and writes its force once —
// no atomics in this kernel at all (int64 fixed-point accumulation B1 is for
// cross-pass zone-boundary contributions at M3.5/M4, not intra-pass writes).
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <string>
#include <vector>

#include "tdmd/core/cluster.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/thermal.hpp"
#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/potentials/clustered_morse.hpp"
#include "tdmd/potentials/pair_morse.hpp"

using namespace tdmd;

static std::string project_root() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT);
#else
  return ".";
#endif
}

namespace {

constexpr int kBlock = 128;                       // 4 warp-clusters per block
constexpr int kWarps = kBlock / 32;

template <typename Real>
__global__ void cluster_force_kernel(
    const double* __restrict__ x, const double* __restrict__ y,
    const double* __restrict__ z, double* __restrict__ fx,
    double* __restrict__ fy, double* __restrict__ fz,
    const int2* __restrict__ crange, const int* __restrict__ nbr_off,
    const int* __restrict__ nbr, int n_clusters, double Lx, double Ly,
    double Lz, potentials::MorseParams<Real> prm, double rcut2) {
  __shared__ double sx[kWarps][32], sy[kWarps][32], sz[kWarps][32];
  const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int A = blockIdx.x * kWarps + w;
  if (A >= n_clusters) return;
  const int2 ra = crange[A];
  const int i = ra.x + lane;
  const bool active = i < ra.y;
  const double xi = active ? x[i] : 0.0;
  const double yi = active ? y[i] : 0.0;
  const double zi = active ? z[i] : 0.0;
  double fxi = 0.0, fyi = 0.0, fzi = 0.0;

  for (int e = nbr_off[A]; e < nbr_off[A + 1]; ++e) {
    const int2 rb = crange[nbr[e]];
    const int nb = rb.y - rb.x;
    if (lane < nb) {  // warp-cooperative stage of cluster B (coalesced)
      sx[w][lane] = x[rb.x + lane];
      sy[w][lane] = y[rb.x + lane];
      sz[w][lane] = z[rb.x + lane];
    }
    __syncwarp();
    if (active) {
      for (int t = 0; t < nb; ++t) {
        const int j = rb.x + t;
        if (j == i) continue;
        double dx = xi - sx[w][t];
        double dy = yi - sy[w][t];
        double dz = zi - sz[w][t];
        dx -= Lx * rint(dx / Lx);  // PBC min-image (prototype: all-periodic)
        dy -= Ly * rint(dy / Ly);
        dz -= Lz * rint(dz / Lz);
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= rcut2 || r2 < 1e-18) continue;
        Real u, f_over_r;
        potentials::pair_morse<Real>(Real(sqrt(r2)), prm, u, f_over_r);
        fxi += double(f_over_r) * dx;
        fyi += double(f_over_r) * dy;
        fzi += double(f_over_r) * dz;
      }
    }
    __syncwarp();
  }
  if (active) { fx[i] = fxi; fy[i] = fyi; fz[i] = fzi; }  // owner write
}

template <typename T>
T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  EXPECT_EQ(cudaMalloc(&d, v.size() * sizeof(T)), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice),
            cudaSuccess);
  return d;
}

// CPU clustered reference (sorts atoms, builds the list), then the GPU kernel
// on the SAME sorted layout and CSR-flattened list. Returns max |F_gpu - F_cpu|.
template <typename Real>
double gpu_vs_cpu(core::AtomSoA<double>& atoms, const core::Box& box) {
  potentials::ClusteredMorse<double> cpu;
  core::zero_forces(atoms);
  cpu.compute(atoms, box);

  const auto& cs = cpu.cluster_set();
  const int nc = int(cs.clusters.size());
  std::vector<int2> crange(nc);
  std::vector<int> off(nc + 1, 0), flat;
  for (int c = 0; c < nc; ++c) {
    crange[c] = {cs.clusters[c].begin, cs.clusters[c].end};
    off[c + 1] = off[c] + int(cs.nbr[c].size());
    flat.insert(flat.end(), cs.nbr[c].begin(), cs.nbr[c].end());
  }

  double *dx = upload(atoms.x), *dy = upload(atoms.y), *dz = upload(atoms.z);
  std::vector<double> zero(atoms.n, 0.0);
  double *dfx = upload(zero), *dfy = upload(zero), *dfz = upload(zero);
  int2* dcr = upload(crange);
  int *doff = upload(off), *dnbr = upload(flat);

  const potentials::MorseParams<Real> prm{Real(cpu.D), Real(cpu.alpha), Real(cpu.r0)};
  const int grid = (nc + kWarps - 1) / kWarps;
  cluster_force_kernel<Real><<<grid, kBlock>>>(
      dx, dy, dz, dfx, dfy, dfz, dcr, doff, dnbr, nc, box.len(0), box.len(1),
      box.len(2), prm, cpu.rcut * cpu.rcut);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<double> gfx(atoms.n), gfy(atoms.n), gfz(atoms.n);
  EXPECT_EQ(cudaMemcpy(gfx.data(), dfx, atoms.n * 8, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(gfy.data(), dfy, atoms.n * 8, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(gfz.data(), dfz, atoms.n * 8, cudaMemcpyDeviceToHost), cudaSuccess);
  for (auto* p : {dx, dy, dz, dfx, dfy, dfz}) cudaFree(p);
  cudaFree(dcr); cudaFree(doff); cudaFree(dnbr);

  double maxd = 0.0;
  for (int i = 0; i < atoms.n; ++i)
    maxd = std::max({maxd, std::fabs(gfx[i] - atoms.fx[i]),
                     std::fabs(gfy[i] - atoms.fy[i]),
                     std::fabs(gfz[i] - atoms.fz[i])});
  return maxd;
}

core::AtomSoA<double> fcc(int nc, core::Box& box) {
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

}  // namespace

// Tier-2 deliverable: REAL regs/thread + theoretical occupancy of the kernel
// (replaces the 40-regs guess in the occupancy-probe inputs).
TEST(CudaCluster, RegistersAndOccupancy) {
  cudaDeviceProp p;
  ASSERT_EQ(cudaGetDeviceProperties(&p, 0), cudaSuccess);
  for (auto [name, fn] :
       {std::pair{"fp32-pair", (const void*)cluster_force_kernel<float>},
        std::pair{"fp64-pair", (const void*)cluster_force_kernel<double>}}) {
    cudaFuncAttributes attr;
    ASSERT_EQ(cudaFuncGetAttributes(&attr, fn), cudaSuccess);
    int blocks = 0;
    ASSERT_EQ(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, kBlock,
                                                            attr.sharedSizeBytes),
              cudaSuccess);
    const double occ =
        100.0 * blocks * kBlock / p.maxThreadsPerMultiProcessor;
    std::printf("Test_CUDA_Cluster[%s]: regs/thread=%d  smem(static)=%zu B  "
                "blocks/SM=%d  occupancy=%.0f%%\n",
                name, attr.numRegs, attr.sharedSizeBytes, blocks, occ);
    EXPECT_GE(occ, 50.0);  // A2 red-flag threshold from M2.5
  }
}

TEST(CudaCluster, MatchesCpuClusteredGolden72) {
  core::AtomSoA<double> atoms;
  core::Box box;
  box.periodic = {true, true, true};
  ASSERT_TRUE(io::read_lammps_data(
      project_root() + "/reference_data/al_fcc_72.data", atoms, box));
  const double maxd = gpu_vs_cpu<double>(atoms, box);
  std::printf("Test_CUDA_Cluster: golden72 GPU-vs-CPU max|dF|=%.3e eV/Å\n", maxd);
  EXPECT_LT(maxd, 1e-12);
}

TEST(CudaCluster, MatchesCpuClustered11k) {
  core::Box box;
  auto atoms = fcc(14, box);  // 10976 atoms
  const double maxd = gpu_vs_cpu<double>(atoms, box);
  std::printf("Test_CUDA_Cluster: N=10976 GPU-vs-CPU max|dF|=%.3e eV/Å\n", maxd);
  EXPECT_LT(maxd, 1e-12);
}

TEST(CudaCluster, Fp32PairMathOnGpu) {
  core::Box box;
  auto atoms = fcc(7, box);  // 1372 atoms
  const double maxd = gpu_vs_cpu<float>(atoms, box);
  std::printf("Test_CUDA_Cluster: fp32 GPU-vs-fp64-CPU max|dF|=%.3e eV/Å\n", maxd);
  EXPECT_LT(maxd, 1e-4);
}
