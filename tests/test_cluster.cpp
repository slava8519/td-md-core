// M3 core: Z-order clustering + cluster-pair list + clustered force driver.
// Acceptance (Tier-1): clustered path ≡ direct O(N²) path to <=1e-12 (internal
// cross test, the oracle for all generated sizes) and ≡ golden to <=1e-6.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "tdmd/core/cluster.hpp"
#include "tdmd/core/simulation.hpp"
#include "tdmd/core/thermal.hpp"
#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/potentials/morse.hpp"
#include "tdmd/potentials/clustered_morse.hpp"

using namespace tdmd;

static std::string project_root() {
#ifdef TDMD_PROJECT_ROOT
  return std::string(TDMD_PROJECT_ROOT);
#else
  return ".";
#endif
}

namespace {

// In-test FCC builder (no Python/numpy in CI): SplitMix64-perturbed lattice.
core::AtomSoA<double> make_fcc(int nx, int ny, int nz, core::Box& box,
                               double amp = 0.10, uint64_t seed = 7) {
  const double a0 = 4.05;
  core::AtomSoA<double> a;
  a.resize(4 * nx * ny * nz);
  box.lo = {0, 0, 0};
  box.hi = {nx * a0, ny * a0, nz * a0};
  box.periodic = {true, true, true};
  const double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  core::thermal::SplitMix64 rng(seed);
  auto uni = [&] { return amp * (2.0 * rng.uniform() - 1.0); };
  int k = 0;
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int l = 0; l < nz; ++l)
        for (auto& b : basis) {
          a.x[k] = (i + b[0]) * a0 + uni();
          a.y[k] = (j + b[1]) * a0 + uni();
          a.z[k] = (l + b[2]) * a0 + uni();
          a.mass[k] = 26.9815;
          a.type[k] = 1;
          ++k;
        }
  return a;
}

// Max |F_clustered - F_direct| over atoms, matched by atom id (the clustered
// driver sorts atoms in place).
template <typename RealC>
double cross_test_max_diff(core::AtomSoA<double> atoms, const core::Box& box,
                           double* pe_diff = nullptr) {
  core::AtomSoA<double> ref = atoms;  // direct path on the original ordering
  potentials::MorsePotential<double> direct;
  core::zero_forces(ref);
  const double pe_ref = direct.compute(ref, box);
  std::map<int, int> ref_of_id;
  for (int i = 0; i < ref.n; ++i) ref_of_id[ref.id[i]] = i;

  core::AtomSoA<double> clu;
  double pe_clu;
  double maxd = 0.0;
  if constexpr (std::is_same_v<RealC, double>) {
    potentials::ClusteredMorse<double> clustered;
    clu = atoms;
    core::zero_forces(clu);
    pe_clu = clustered.compute(clu, box);
  } else {
    potentials::ClusteredMorse<float> clustered;
    core::AtomSoA<float> cf;
    cf.resize(atoms.n);
    cf.x = atoms.x; cf.y = atoms.y; cf.z = atoms.z;
    cf.mass = atoms.mass; cf.type = atoms.type; cf.id = atoms.id;
    core::zero_forces(cf);
    pe_clu = clustered.compute(cf, box);
    clu.resize(cf.n);
    clu.x = cf.x; clu.y = cf.y; clu.z = cf.z;
    clu.fx = cf.fx; clu.fy = cf.fy; clu.fz = cf.fz;
    clu.id = cf.id;
  }
  for (int i = 0; i < clu.n; ++i) {
    const int r = ref_of_id.at(clu.id[i]);
    maxd = std::max({maxd, std::fabs(clu.fx[i] - ref.fx[r]),
                     std::fabs(clu.fy[i] - ref.fy[r]),
                     std::fabs(clu.fz[i] - ref.fz[r])});
  }
  if (pe_diff) *pe_diff = std::fabs(pe_clu - pe_ref);
  return maxd;
}

}  // namespace

TEST(Cluster, PartitionAndSizes) {
  core::Box box;
  auto atoms = make_fcc(3, 3, 2, box);  // 72 atoms -> 32+32+8
  core::ClusterSet cs;
  cs.build(atoms, box, 2.33, 5.0);
  ASSERT_EQ(cs.clusters.size(), 3u);
  EXPECT_EQ(cs.clusters[0].begin, 0);
  EXPECT_EQ(cs.clusters[2].end, 72);
  for (size_t c = 0; c + 1 < cs.clusters.size(); ++c)
    EXPECT_EQ(cs.clusters[c].end, cs.clusters[c + 1].begin);
  // ids are a permutation of 1..72
  std::vector<int> ids = atoms.id;
  std::sort(ids.begin(), ids.end());
  for (int i = 0; i < 72; ++i) EXPECT_EQ(ids[i], i + 1);
}

TEST(Cluster, BuildIsDeterministic) {
  core::Box box;
  auto a1 = make_fcc(4, 4, 4, box);
  auto a2 = a1;
  core::ClusterSet c1, c2;
  c1.build(a1, box, 2.33, 5.0);
  c2.build(a2, box, 2.33, 5.0);
  ASSERT_EQ(a1.id, a2.id);  // identical permutation
  ASSERT_EQ(c1.nbr.size(), c2.nbr.size());
  for (size_t i = 0; i < c1.nbr.size(); ++i) EXPECT_EQ(c1.nbr[i], c2.nbr[i]);
}

// Every atom pair within r_cut must be covered by the cluster-pair list.
TEST(Cluster, PairListIsComplete) {
  core::Box box;
  auto atoms = make_fcc(4, 3, 3, box);  // 144 atoms
  core::ClusterSet cs;
  const double rcut = 4.0, skin = 1.0;
  cs.build(atoms, box, 2.33, rcut + skin);

  std::vector<int> cluster_of(atoms.n);
  for (size_t c = 0; c < cs.clusters.size(); ++c)
    for (int i = cs.clusters[c].begin; i < cs.clusters[c].end; ++i)
      cluster_of[i] = int(c);

  const double L[3] = {box.len(0), box.len(1), box.len(2)};
  int pairs_checked = 0;
  for (int i = 0; i < atoms.n; ++i) {
    for (int j = 0; j < atoms.n; ++j) {
      if (j == i) continue;
      double dx = atoms.x[i] - atoms.x[j];
      double dy = atoms.y[i] - atoms.y[j];
      double dz = atoms.z[i] - atoms.z[j];
      dx -= L[0] * std::round(dx / L[0]);
      dy -= L[1] * std::round(dy / L[1]);
      dz -= L[2] * std::round(dz / L[2]);
      if (dx * dx + dy * dy + dz * dz >= rcut * rcut) continue;
      const auto& nb = cs.nbr[cluster_of[i]];
      ASSERT_NE(std::find(nb.begin(), nb.end(), cluster_of[j]), nb.end())
          << "pair (" << i << "," << j << ") not covered";
      ++pairs_checked;
    }
  }
  EXPECT_GT(pairs_checked, 1000);  // sanity: the system actually has pairs
}

TEST(Cluster, MatchesDirectOnGolden72) {
  core::AtomSoA<double> atoms;
  core::Box box;
  box.periodic = {true, true, true};
  ASSERT_TRUE(io::read_lammps_data(
      project_root() + "/reference_data/al_fcc_72.data", atoms, box));
  double pe_diff;
  const double maxd = cross_test_max_diff<double>(atoms, box, &pe_diff);
  std::printf("Test_Cluster: golden72 max|dF|=%.3e eV/Å  |dPE|=%.3e eV\n",
              maxd, pe_diff);
  EXPECT_LT(maxd, 1e-12);
  EXPECT_LT(pe_diff, 1e-12);
}

TEST(Cluster, MatchesDirectOn144) {
  core::AtomSoA<double> atoms;
  core::Box box;
  box.periodic = {true, true, true};
  ASSERT_TRUE(io::read_lammps_data(
      project_root() + "/reference_data/al_fcc_144.data", atoms, box));
  EXPECT_LT(cross_test_max_diff<double>(atoms, box), 1e-12);
}

// The CI-scale cross test (roadmap: N up to 1e4 in CI): ~11k atoms.
TEST(Cluster, MatchesDirectOn11k) {
  core::Box box;
  auto atoms = make_fcc(14, 14, 14, box);  // 10976 atoms
  double pe_diff;
  const double maxd = cross_test_max_diff<double>(atoms, box, &pe_diff);
  std::printf("Test_Cluster: N=10976 max|dF|=%.3e eV/Å  |dPE|=%.3e eV\n",
              maxd, pe_diff);
  EXPECT_LT(maxd, 1e-12);
  EXPECT_LT(pe_diff, 1e-9);  // PE is a global sum — looser absolute bound
}

// Real=float instantiation: pair math in FP32, coordinates/accumulation FP64.
TEST(Cluster, Fp32PairMathVsFp64Direct) {
  core::AtomSoA<double> atoms;
  core::Box box;
  box.periodic = {true, true, true};
  ASSERT_TRUE(io::read_lammps_data(
      project_root() + "/reference_data/al_fcc_72.data", atoms, box));
  const double maxd = cross_test_max_diff<float>(atoms, box);
  std::printf("Test_Cluster: fp32 pair math max|dF|=%.3e eV/Å (vs fp64 direct)\n",
              maxd);
  EXPECT_LT(maxd, 1e-4);  // roadmap tolerance for the fp32 instantiation
}

// Skin semantics: small displacements reuse the list (no rebuild) and the
// forces still match direct; displacement > skin/2 triggers a rebuild.
TEST(Cluster, SkinRebuildPolicy) {
  core::Box box;
  auto atoms = make_fcc(3, 3, 2, box);
  potentials::ClusteredMorse<double> pot;  // skin = 1.0

  core::zero_forces(atoms);
  pot.compute(atoms, box);
  ASSERT_EQ(pot.rebuild_count, 1);

  // move everyone a little (< skin/2): list must be reused AND stay exact
  for (int i = 0; i < atoms.n; ++i) { atoms.x[i] += 0.3; atoms.y[i] -= 0.2; }
  core::AtomSoA<double> ref = atoms;
  potentials::MorsePotential<double> direct;
  core::zero_forces(ref);
  direct.compute(ref, box);
  std::map<int, int> ref_of_id;
  for (int i = 0; i < ref.n; ++i) ref_of_id[ref.id[i]] = i;

  core::zero_forces(atoms);
  pot.compute(atoms, box);
  EXPECT_EQ(pot.rebuild_count, 1);  // reused
  double maxd = 0.0;
  for (int i = 0; i < atoms.n; ++i) {
    const int r = ref_of_id.at(atoms.id[i]);
    maxd = std::max({maxd, std::fabs(atoms.fx[i] - ref.fx[r]),
                     std::fabs(atoms.fy[i] - ref.fy[r]),
                     std::fabs(atoms.fz[i] - ref.fz[r])});
  }
  EXPECT_LT(maxd, 1e-12);

  // one atom beyond skin/2 => rebuild
  atoms.x[0] += 0.6;
  core::zero_forces(atoms);
  pot.compute(atoms, box);
  EXPECT_EQ(pot.rebuild_count, 2);
}

// Full NVE run on the clustered path stays bitwise-deterministic run-to-run
// (sorting permutation is keyed on (morton, id) — unique and reproducible).
TEST(Cluster, ClusteredRunIsBitwiseDeterministic) {
  auto run = [&] {
    core::AtomSoA<double> atoms;
    core::Box box;
    box.periodic = {true, true, true};
    EXPECT_TRUE(io::read_lammps_data(
        project_root() + "/reference_data/al_fcc_144.data", atoms, box));
    core::thermal::maxwell_init(atoms, 300.0, 20070101u);
    potentials::ClusteredMorse<double> pot;
    core::SimOptions o;
    o.steps = 100;
    o.dt = 0.001;
    auto res = core::run_simulation(atoms, box, pot, o, [](long) {});
    EXPECT_EQ(res.halt, core::Halt::None) << res.halt_msg;
    return atoms;
  };
  auto a = run(), b = run();
  ASSERT_EQ(a.id, b.id);
  EXPECT_TRUE(std::memcmp(a.x.data(), b.x.data(), a.n * sizeof(double)) == 0);
  EXPECT_TRUE(std::memcmp(a.vx.data(), b.vx.data(), a.n * sizeof(double)) == 0);
}
