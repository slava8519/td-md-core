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

#include <climits>
#include <cmath>
#include <cstring>
#include <set>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/thermal.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/zone_force.cuh"
#include "tdmd/cuda/zone_integrate.cuh"
#include "tdmd/cuda/zone_verlet.cuh"
#include <cub/device/device_scan.cuh>
#include "tdmd/potentials/cutoff.hpp"
#include "tdmd/potentials/morse.hpp"
#include "tdmd/potentials/pair_lj.hpp"

// CUB/libcu++ exposes a global ::cuda namespace, and nvcc-generated host
// stubs reference cuda::std unqualified — `using namespace tdmd` would make
// `cuda` ambiguous there. Targeted aliases instead:
namespace core = tdmd::core;
namespace potentials = tdmd::potentials;
namespace units = tdmd::units;
namespace tdcu = tdmd::cuda;

namespace {

// --- shared single-source pair functors (host + device, zone_force.cuh) ---

using tdcu::LJDev;
using tdcu::MorseDev;

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
    tdcu::ZoneForceArgs args{
        zb[za].x,  zb[za].y,  zb[za].z,  zb[za].m,
        zb[zo].x,  zb[zo].y,  zb[zo].z,  zb[zo].m,
        zb[za].fx, zb[za].fy, zb[za].fz, dpe, dmr, dof, same, energy};
    const int grid = (zb[za].m + tdcu::kZoneBlock - 1) / tdcu::kZoneBlock;
    tdcu::zone_pair_kernel<PairF><<<grid, tdcu::kZoneBlock>>>(args, geom, pot);
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

// --- GPU runner over the PERSISTENT Verlet path (zone_verlet.cuh): build a
// per-zone cell grid at (rcut+skin), materialise a CSR partner list per (A,B)
// launch, then force from the stored list. Same pass structure as
// gpu_raw_pass — its raw accumulators MUST equal the tile path bit-for-bit
// (single pass => the list is an exact geometric superset of within-rcut). ---

template <typename PairF>
GpuPassResult gpu_verlet_pass(const core::AtomSoA<double>& a,
                              const core::Box& box,
                              const core::ZoneDecomposition& zd, double rcut,
                              double skin, const PairF& pot) {
  const core::PairGeom geom(box, rcut);
  const core::PairGeom geom_list(box, rcut + skin);
  const int n = zd.n_zones;
  const double len[3] = {box.len(0), box.len(1), box.len(2)};
  const bool per[3] = {box.periodic[0], box.periodic[1], box.periodic[2]};

  auto excl_scan = [](const int* in, int* out, int num) {
    void* tmp = nullptr;
    std::size_t bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, bytes, in, out, num);
    EXPECT_EQ(cudaMalloc(&tmp, bytes), cudaSuccess);
    cub::DeviceScan::ExclusiveSum(tmp, bytes, in, out, num);
    cudaFree(tmp);
  };

  struct ZBuf {
    double *x, *y, *z;
    long long *fx, *fy, *fz;
    int m, nc;
    tdcu::CellGrid grid;
    int *cell_of, *order, *counts, *starts, *cursor;
  };
  std::vector<ZBuf> zb(n);
  std::vector<int*> scratch;  // freed after the final sync (async kernels)

  for (int zi = 0; zi < n; ++zi) {
    const auto& mem = zd.members[zi];
    std::vector<double> hx, hy, hz;
    for (int i : mem) { hx.push_back(a.x[i]); hy.push_back(a.y[i]); hz.push_back(a.z[i]); }
    std::vector<long long> zero(mem.size(), 0);
    ZBuf b{};
    b.m = int(mem.size());
    b.x = upload(hx); b.y = upload(hy); b.z = upload(hz);
    b.fx = upload(zero); b.fy = upload(zero); b.fz = upload(zero);
    b.grid = tdcu::make_zone_grid(box.lo.data(), len, per, rcut + skin, n, zi);
    b.nc = b.grid.ncells();
    EXPECT_EQ(cudaMalloc(&b.cell_of, sizeof(int) * std::max(b.m, 1)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&b.order, sizeof(int) * std::max(b.m, 1)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&b.counts, sizeof(int) * b.nc), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&b.starts, sizeof(int) * b.nc), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&b.cursor, sizeof(int) * b.nc), cudaSuccess);
    cudaMemset(b.counts, 0, sizeof(int) * b.nc);
    if (b.m > 0) {
      const int g = (b.m + tdcu::kVerletBlock - 1) / tdcu::kVerletBlock;
      tdcu::cell_count_kernel<<<g, tdcu::kVerletBlock>>>(b.x, b.y, b.z, b.m,
                                                         b.grid, b.cell_of, b.counts);
    }
    excl_scan(b.counts, b.starts, b.nc);
    cudaMemcpy(b.cursor, b.starts, sizeof(int) * b.nc, cudaMemcpyDeviceToDevice);
    if (b.m > 0) {
      const int g = (b.m + tdcu::kVerletBlock - 1) / tdcu::kVerletBlock;
      tdcu::cell_scatter_kernel<<<g, tdcu::kVerletBlock>>>(b.cell_of, b.m, b.cursor, b.order);
    }
    zb[zi] = b;
  }

  std::vector<long long> pe0{0};
  std::vector<unsigned long long> mr0{0x7FF0000000000000ULL};
  std::vector<int> of0{0};
  long long* dpe = upload(pe0);
  unsigned long long* dmr = upload(mr0);
  int* dof = upload(of0);

  auto launch = [&](int za, int zo, bool same, bool energy) {
    const int na = zb[za].m;
    if (na == 0) return;
    int *dcounts = nullptr, *doffsets = nullptr;
    EXPECT_EQ(cudaMalloc(&dcounts, sizeof(int) * (na + 1)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&doffsets, sizeof(int) * (na + 1)), cudaSuccess);
    cudaMemset(dcounts, 0, sizeof(int) * (na + 1));
    const int g = (na + tdcu::kVerletBlock - 1) / tdcu::kVerletBlock;
    tdcu::verlet_count_kernel<<<g, tdcu::kVerletBlock>>>(
        zb[za].x, zb[za].y, zb[za].z, na, zb[zo].x, zb[zo].y, zb[zo].z,
        geom_list, zb[zo].grid, zb[zo].starts, zb[zo].counts, zb[zo].order,
        same, INT_MAX, nullptr, dcounts);
    excl_scan(dcounts, doffsets, na + 1);
    int total = 0;
    cudaMemcpy(&total, doffsets + na, sizeof(int), cudaMemcpyDeviceToHost);
    int* didx = nullptr;
    if (total > 0) EXPECT_EQ(cudaMalloc(&didx, sizeof(int) * total), cudaSuccess);
    tdcu::verlet_fill_kernel<<<g, tdcu::kVerletBlock>>>(
        zb[za].x, zb[za].y, zb[za].z, na, zb[zo].x, zb[zo].y, zb[zo].z,
        geom_list, zb[zo].grid, zb[zo].starts, zb[zo].counts, zb[zo].order,
        same, INT_MAX, doffsets, didx);
    tdcu::ZoneForceArgs args{zb[za].x,  zb[za].y,  zb[za].z,  na,
                             zb[zo].x,  zb[zo].y,  zb[zo].z,  zb[zo].m,
                             zb[za].fx, zb[za].fy, zb[za].fz, dpe, dmr, dof,
                             same,      energy};
    tdcu::zone_pair_verlet_kernel<PairF><<<g, tdcu::kVerletBlock>>>(
        args, geom, pot, doffsets, didx);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    scratch.push_back(dcounts);
    scratch.push_back(doffsets);
    if (didx) scratch.push_back(didx);
  };

  for (int q = 0; q < n; ++q) {
    const int zi = q;
    launch(zi, zi, /*same=*/true, /*energy=*/true);
    if (n > 1) {
      const int zn = (zi + 1) % n;
      if (zn > zi || box.periodic[2]) {
        launch(zi, zn, false, true);
        launch(zn, zi, false, false);
      }
    }
  }
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  GpuPassResult res;
  res.raw.fx.resize(n); res.raw.fy.resize(n); res.raw.fz.resize(n);
  for (int zi = 0; zi < n; ++zi) {
    const int m = zb[zi].m;
    res.raw.fx[zi].resize(m); res.raw.fy[zi].resize(m); res.raw.fz[zi].resize(m);
    if (m > 0) {
      cudaMemcpy(res.raw.fx[zi].data(), zb[zi].fx, m * 8, cudaMemcpyDeviceToHost);
      cudaMemcpy(res.raw.fy[zi].data(), zb[zi].fy, m * 8, cudaMemcpyDeviceToHost);
      cudaMemcpy(res.raw.fz[zi].data(), zb[zi].fz, m * 8, cudaMemcpyDeviceToHost);
    }
    for (auto* p : {zb[zi].x, zb[zi].y, zb[zi].z}) cudaFree(p);
    for (auto* p : {zb[zi].fx, zb[zi].fy, zb[zi].fz}) cudaFree(p);
    for (auto* p : {zb[zi].cell_of, zb[zi].order, zb[zi].counts, zb[zi].starts,
                    zb[zi].cursor})
      cudaFree(p);
  }
  unsigned long long mr = 0;
  cudaMemcpy(&res.raw.pe, dpe, 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(&mr, dmr, 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(&res.overflow, dof, 4, cudaMemcpyDeviceToHost);
  std::memcpy(&res.min_r2, &mr, 8);
  for (int* p : scratch) cudaFree(p);
  cudaFree(dpe); cudaFree(dmr); cudaFree(dof);
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

// GATE-01 helper: materialise the single-zone SELF Verlet list (radius
// rcut+skin) at `a`'s positions and return it to the host as per-atom partner
// index lists.
std::vector<std::vector<int>> materialize_self_list(const core::AtomSoA<double>& a,
                                                    const core::Box& box,
                                                    double rcut, double skin) {
  const int n = a.n;
  const core::PairGeom geom_list(box, rcut + skin);
  const double len[3] = {box.len(0), box.len(1), box.len(2)};
  const bool per[3] = {box.periodic[0], box.periodic[1], box.periodic[2]};
  const tdcu::CellGrid grid =
      tdcu::make_zone_grid(box.lo.data(), len, per, rcut + skin, 1, 0);
  const int nc = grid.ncells();
  auto scan = [](const int* in, int* out, int num) {
    void* tmp = nullptr; std::size_t b = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, b, in, out, num);
    cudaMalloc(&tmp, b); cub::DeviceScan::ExclusiveSum(tmp, b, in, out, num);
    cudaFree(tmp);
  };
  double *dx = upload(a.x), *dy = upload(a.y), *dz = upload(a.z);
  int *cell_of, *order, *counts, *starts, *cursor;
  cudaMalloc(&cell_of, n * 4); cudaMalloc(&order, n * 4);
  cudaMalloc(&counts, nc * 4); cudaMalloc(&starts, nc * 4); cudaMalloc(&cursor, nc * 4);
  cudaMemset(counts, 0, nc * 4);
  const int g = (n + tdcu::kVerletBlock - 1) / tdcu::kVerletBlock;
  tdcu::cell_count_kernel<<<g, tdcu::kVerletBlock>>>(dx, dy, dz, n, grid, cell_of, counts);
  scan(counts, starts, nc);
  cudaMemcpy(cursor, starts, nc * 4, cudaMemcpyDeviceToDevice);
  tdcu::cell_scatter_kernel<<<g, tdcu::kVerletBlock>>>(cell_of, n, cursor, order);
  int *vc, *voff;
  cudaMalloc(&vc, (n + 1) * 4); cudaMalloc(&voff, (n + 1) * 4); cudaMemset(vc, 0, (n + 1) * 4);
  tdcu::verlet_count_kernel<<<g, tdcu::kVerletBlock>>>(
      dx, dy, dz, n, dx, dy, dz, geom_list, grid, starts, counts, order,
      /*same=*/true, INT_MAX, nullptr, vc);
  scan(vc, voff, n + 1);
  int total = 0; cudaMemcpy(&total, voff + n, 4, cudaMemcpyDeviceToHost);
  int* vidx = nullptr; if (total > 0) cudaMalloc(&vidx, total * 4);
  tdcu::verlet_fill_kernel<<<g, tdcu::kVerletBlock>>>(
      dx, dy, dz, n, dx, dy, dz, geom_list, grid, starts, counts, order,
      /*same=*/true, INT_MAX, voff, vidx);
  cudaDeviceSynchronize();
  std::vector<int> hoff(n + 1), hidx(std::max(total, 1));
  cudaMemcpy(hoff.data(), voff, (n + 1) * 4, cudaMemcpyDeviceToHost);
  if (total > 0) cudaMemcpy(hidx.data(), vidx, total * 4, cudaMemcpyDeviceToHost);
  for (auto* p : {dx, dy, dz}) cudaFree(p);
  for (auto* p : {cell_of, order, counts, starts, cursor, vc, voff}) cudaFree(p);
  if (vidx) cudaFree(vidx);
  std::vector<std::vector<int>> out(n);
  for (int i = 0; i < n; ++i)
    for (int t = hoff[i]; t < hoff[i + 1]; ++t) out[i].push_back(hidx[t]);
  return out;
}

}  // namespace

// --- GATE-01: standalone GEOMETRIC oracle for the Verlet list (NL-INV-1),
// independent of the cell-path reference. (1) The materialised list is EXACTLY
// the set of within-(rcut+skin) partners (the 27-cell sweep misses none).
// (2) After any per-atom displacement < skin/2, every within-rcut pair is
// still in the build-time list — the superset guarantee the skin budget rests
// on. ---

TEST(CudaZones, VerletListGeometricOracle) {
  core::Box box;
  auto a = make_fcc(box, 3, 3, 6);  // 216 atoms, single zone, PBC
  core::thermal::maxwell_init(a, 300.0, 19);
  const double skin = 1.0;
  const auto list = materialize_self_list(a, box, kRcut, skin);
  ASSERT_EQ(int(list.size()), a.n);

  // (1) exact coverage at build positions
  core::PairGeom gl(box, kRcut + skin);
  for (int i = 0; i < a.n; ++i) {
    std::set<int> listed(list[i].begin(), list[i].end());
    for (int j = 0; j < a.n; ++j) {
      if (j == i) continue;
      double dx = a.x[i] - a.x[j], dy = a.y[i] - a.y[j], dz = a.z[i] - a.z[j], r2;
      const bool in_skin = gl.reduce(dx, dy, dz, r2);
      ASSERT_EQ(in_skin, listed.count(j) > 0) << "coverage i=" << i << " j=" << j;
    }
  }

  // (2) superset survives a bounded displacement (|d_i| < skin/2)
  core::AtomSoA<double> moved = a;
  unsigned s = 12345;
  auto rnd = [&] { s = s * 1664525u + 1013904223u; return (s >> 8) * (1.0 / 16777216.0); };
  const double dmax = (skin / 2.0) * 0.99 / std::sqrt(3.0);  // => |d| < skin/2
  for (int i = 0; i < a.n; ++i) {
    moved.x[i] += (2 * rnd() - 1) * dmax;
    moved.y[i] += (2 * rnd() - 1) * dmax;
    moved.z[i] += (2 * rnd() - 1) * dmax;
  }
  core::PairGeom gc(box, kRcut);
  long checked = 0;
  for (int i = 0; i < a.n; ++i) {
    std::set<int> listed(list[i].begin(), list[i].end());
    for (int j = 0; j < a.n; ++j) {
      if (j == i) continue;
      double dx = moved.x[i] - moved.x[j], dy = moved.y[i] - moved.y[j],
             dz = moved.z[i] - moved.z[j], r2;
      if (gc.reduce(dx, dy, dz, r2)) {
        ASSERT_TRUE(listed.count(j) > 0)
            << "missed within-rcut pair after drift: i=" << i << " j=" << j;
        ++checked;
      }
    }
  }
  EXPECT_GT(checked, 0);  // the test actually exercised within-rcut pairs
  std::printf("Test_CUDA_Zones[oracle]: %ld within-rcut pairs all covered "
              "after |d|<skin/2 drift\n", checked);
}

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

// --- PR-1: the Verlet-list force path == the cell/tile path bit-for-bit ---
// (zone_verlet.cuh). Single pass at equal positions => the materialised list
// is an exact geometric superset of the within-rcut pairs, so re-testing
// r2<rcut2 over the list reproduces the tile accumulators exactly (NL-INV-5).

TEST(CudaZones, VerletListBitwiseVsTiles) {
  struct Case { int cx, cy, cz, nz; bool pbc_z; double skin; const char* name; };
  const Case cases[] = {
      {2, 2, 6, 4, true,  1.0, "pbc 4 zones"},
      {2, 2, 6, 4, false, 1.0, "free-z 4 zones"},
      {3, 3, 6, 1, true,  1.0, "single zone pbc"},
      {2, 2, 8, 5, true,  2.0, "5 zones wide skin"},
  };
  for (const auto& c : cases) {
    core::Box box;
    auto a = make_fcc(box, c.cx, c.cy, c.cz);
    box.periodic = {true, true, c.pbc_z};
    core::thermal::maxwell_init(a, 300.0, 31);
    const auto zd = core::ZoneDecomposition::build(a, box, c.nz, kRcut);
    const auto lj = make_lj(0.4, 2.55, kRcut);

    const auto tiles = gpu_raw_pass(a, box, zd, kRcut, lj);
    const auto verlet = gpu_verlet_pass(a, box, zd, kRcut, c.skin, lj);
    EXPECT_EQ(verlet.overflow, 0) << c.name;
    EXPECT_TRUE(raw_eq(tiles.raw, verlet.raw)) << c.name;
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

// --- integration + zone-local reductions (zone_integrate.cuh): drift, the
// END kick with raw->FP64 conversion, v_max²/a_max²/k2cap — all bitwise vs
// the CPU conveyor expressions; KE — fixed-point vs the CPU FP64 sum within
// the quantization scale. ---

TEST(CudaZones, DriftEndKickAndReductionsBitwise) {
  core::Box box;
  auto a = make_fcc(box, 3, 3, 6);  // 648 atoms = one "zone"
  core::thermal::maxwell_init(a, 300.0, 41);
  const int n = a.n;
  const double dt = 0.002, K2 = 50.0;
  const auto zd = core::ZoneDecomposition::build(a, box, 1, kRcut);
  const auto lj = make_lj(0.4, 2.55, kRcut);

  // f(t0) doubles + raw force accumulators standing in for the next pass
  const auto raw = cpu_raw_pass(a, box, zd, kRcut, lj);
  std::vector<double> f0x(n), f0y(n), f0z(n);
  for (int i = 0; i < n; ++i) {
    f0x[i] = double(raw.fx[0][i]) / core::fixed::ForceAccum::kScale;
    f0y[i] = double(raw.fy[0][i]) / core::fixed::ForceAccum::kScale;
    f0z[i] = double(raw.fz[0][i]) / core::fixed::ForceAccum::kScale;
  }

  // CPU reference — verbatim conveyor expressions (ensure_drift / end_zone)
  core::AtomSoA<double> c = a;
  double v2m = 0.0, a2m = 0.0, kcap = std::numeric_limits<double>::infinity();
  double ke = 0.0;
  for (int i = 0; i < n; ++i) {
    const double inv_m = units::ftm2v / c.mass[i];
    c.vx[i] += 0.5 * dt * inv_m * f0x[i];
    c.vy[i] += 0.5 * dt * inv_m * f0y[i];
    c.vz[i] += 0.5 * dt * inv_m * f0z[i];
    c.x[i] += dt * c.vx[i];
    c.y[i] += dt * c.vy[i];
    c.z[i] += dt * c.vz[i];
  }
  for (int i = 0; i < n; ++i) {
    c.fx[i] = double(raw.fx[0][i]) / core::fixed::ForceAccum::kScale;
    c.fy[i] = double(raw.fy[0][i]) / core::fixed::ForceAccum::kScale;
    c.fz[i] = double(raw.fz[0][i]) / core::fixed::ForceAccum::kScale;
    const double inv_m = units::ftm2v / c.mass[i];
    c.vx[i] += 0.5 * dt * inv_m * c.fx[i];
    c.vy[i] += 0.5 * dt * inv_m * c.fy[i];
    c.vz[i] += 0.5 * dt * inv_m * c.fz[i];
    const double vi2 = core::buffer::speed2(c.vx[i], c.vy[i], c.vz[i]);
    v2m = std::max(v2m, vi2);
    a2m = std::max(a2m, core::buffer::accel2(c.fx[i], c.fy[i], c.fz[i],
                                             c.mass[i]));
    kcap = std::min(kcap, core::buffer::k2_limited_dt_atom(
                              c.fx[i], c.fy[i], c.fz[i], c.vx[i], c.vy[i],
                              c.vz[i], c.mass[i], K2));
    ke += 0.5 * units::mvv2e * c.mass[i] * vi2;
  }

  // GPU
  double *dx = upload(a.x), *dy = upload(a.y), *dz = upload(a.z);
  double *dvx = upload(a.vx), *dvy = upload(a.vy), *dvz = upload(a.vz);
  double *dfx = upload(f0x), *dfy = upload(f0y), *dfz = upload(f0z);
  double* dm = upload(a.mass);
  long long *drx = upload(raw.fx[0]), *dry = upload(raw.fy[0]),
            *drz = upload(raw.fz[0]);
  std::vector<unsigned long long> z0{0}, zinf{0x7FF0000000000000ULL};
  std::vector<long long> l0{0};
  std::vector<int> i0{0};
  unsigned long long *dv2 = upload(z0), *da2 = upload(z0), *dkc = upload(zinf);
  long long* dke = upload(l0);
  int* dof = upload(i0);

  const int grid = (n + tdcu::kIntBlock - 1) / tdcu::kIntBlock;
  tdcu::zone_drift_kernel<<<grid, tdcu::kIntBlock>>>(dx, dy, dz, dvx, dvy,
                                                     dvz, dfx, dfy, dfz, dm,
                                                     n, dt);
  tdcu::zone_end_kernel<<<grid, tdcu::kIntBlock>>>(
      dvx, dvy, dvz, dfx, dfy, dfz, drx, dry, drz, dm, n, dt, K2, dv2, da2,
      dkc, dke, dof);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<double> gx(n), gvx(n), gfx(n), gz(n), gvz(n), gfz(n);
  ASSERT_EQ(cudaMemcpy(gx.data(), dx, n * 8, cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(gz.data(), dz, n * 8, cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(gvx.data(), dvx, n * 8, cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(gvz.data(), dvz, n * 8, cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(gfx.data(), dfx, n * 8, cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(gfz.data(), dfz, n * 8, cudaMemcpyDeviceToHost), cudaSuccess);
  unsigned long long hv2 = 0, ha2 = 0, hkc = 0;
  long long hke = 0;
  int hof = 0;
  ASSERT_EQ(cudaMemcpy(&hv2, dv2, 8, cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&ha2, da2, 8, cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&hkc, dkc, 8, cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&hke, dke, 8, cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&hof, dof, 4, cudaMemcpyDeviceToHost), cudaSuccess);
  for (auto* p : {dx, dy, dz, dvx, dvy, dvz, dfx, dfy, dfz, dm}) cudaFree(p);
  for (auto* p : {drx, dry, drz, dke}) cudaFree(p);
  for (auto* p : {dv2, da2, dkc}) cudaFree(p);
  cudaFree(dof);

  EXPECT_EQ(hof, 0);
  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(gx[i], c.x[i]) << i;     // drift bitwise
    ASSERT_EQ(gz[i], c.z[i]) << i;
    ASSERT_EQ(gvx[i], c.vx[i]) << i;   // both kicks bitwise
    ASSERT_EQ(gvz[i], c.vz[i]) << i;
    ASSERT_EQ(gfx[i], c.fx[i]) << i;   // raw->FP64 conversion bitwise
    ASSERT_EQ(gfz[i], c.fz[i]) << i;
  }
  double gv2, ga2, gkc;
  std::memcpy(&gv2, &hv2, 8);
  std::memcpy(&ga2, &ha2, 8);
  std::memcpy(&gkc, &hkc, 8);
  EXPECT_EQ(gv2, v2m);   // INV-4 / Λ-chain inputs: bitwise
  EXPECT_EQ(ga2, a2m);
  EXPECT_EQ(gkc, kcap);
  const double gke = double(hke) / core::fixed::EnergyAccum::kScale;
  EXPECT_NEAR(gke, ke, n * 1.0 / core::fixed::EnergyAccum::kScale);
}

// --- Tier-2 bookkeeping for the new kernel (regs / theoretical occupancy);
// achieved occupancy via ncu is recorded in _meta/ncu_occupancy (M4 bench). ---

TEST(CudaZones, RegistersAndOccupancy) {
  cudaDeviceProp p;
  ASSERT_EQ(cudaGetDeviceProperties(&p, 0), cudaSuccess);
  for (auto [name, fn] :
       {std::pair{"lj", (const void*)tdcu::zone_pair_kernel<LJDev>},
        std::pair{"morse", (const void*)tdcu::zone_pair_kernel<MorseDev>}}) {
    cudaFuncAttributes attr;
    ASSERT_EQ(cudaFuncGetAttributes(&attr, fn), cudaSuccess);
    int blocks = 0;
    ASSERT_EQ(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &blocks, fn, tdcu::kZoneBlock, attr.sharedSizeBytes),
              cudaSuccess);
    const double occ =
        100.0 * blocks * tdcu::kZoneBlock / p.maxThreadsPerMultiProcessor;
    std::printf("Test_CUDA_Zones[%s]: regs/thread=%d smem=%zu B blocks/SM=%d "
                "occupancy=%.0f%% [fmad=false verify build]\n",
                name, attr.numRegs, attr.sharedSizeBytes, blocks, occ);
    EXPECT_GE(occ, 50.0);  // A2 red-flag threshold (M2.5)
  }
}
