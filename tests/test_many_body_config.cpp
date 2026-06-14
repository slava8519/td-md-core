// M6 PR-E0: descriptor + IManyBodyPotential interface + many-body geometry
// guards (reach_mult / symmetric_reach). This PR adds NO force math — it lands
// the plumbing and proves it is a BITWISE no-op for the pair path (reach_mult=1
// reproduces the legacy zone guards exactly). The 2*rcut effective-force-range
// geometry (M6_EAM_MANYBODY_DESIGN §1) is exercised through the throw matrix.
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/io/config.hpp"
#include "tdmd/potentials/many_body.hpp"

using namespace tdmd;
namespace fs = std::filesystem;

namespace {

// A periodic box of height Lz with a handful of atoms spread along z — enough
// for ZoneDecomposition::build (it only bins z) to exercise the zone-count and
// width guards at the EAM (reach_mult=2) boundary, which needs Lz >= 40 for the
// >=5 periodic zones a real fcc reference fixture cannot host.
core::AtomSoA<double> tall_box(core::Box& box, double Lz, bool pbc_z) {
  box.lo = {0, 0, 0};
  box.hi = {10, 10, Lz};
  box.periodic = {true, true, pbc_z};
  core::AtomSoA<double> a;
  a.resize(20);
  for (int i = 0; i < a.n; ++i) {
    a.x[i] = 1.0; a.y[i] = 1.0;
    a.z[i] = (Lz * i) / a.n;  // spread across [0, Lz)
  }
  return a;
}

// Writes a config and returns its path (RAII cleanup). Mirrors test_config.cpp.
class TempConfig {
 public:
  explicit TempConfig(const std::string& yaml) {
    static int n = 0;
    path_ = fs::temp_directory_path() /
            ("tdmd_mb_cfg_" + std::to_string(n++) + ".yaml");
    std::ofstream(path_) << yaml;
  }
  ~TempConfig() { std::error_code ec; fs::remove(path_, ec); }
  std::string path() const { return path_.string(); }

 private:
  fs::path path_;
};

}  // namespace

// --- the descriptor: EffectiveRange map + pass graph shape ---

TEST(ManyBodyConfig, EffectiveRangeMap) {
  // Pair types are inert (reach 1, asymmetric) — the no-op default.
  EXPECT_EQ(potentials::effective_range_for("morse").cutoff_multiplier, 1);
  EXPECT_FALSE(potentials::effective_range_for("morse").symmetric_reach);
  EXPECT_EQ(potentials::effective_range_for("lj").cutoff_multiplier, 1);
  EXPECT_FALSE(potentials::effective_range_for("lj").symmetric_reach);
  // EAM: force range 2*rcut, both immediate neighbours resident (§1.2).
  EXPECT_EQ(potentials::effective_range_for("eam").cutoff_multiplier, 2);
  EXPECT_TRUE(potentials::effective_range_for("eam").symmetric_reach);

  // The EAM pass graph: density -> embedding -> force, with the documented
  // fixed-point accumulators (§3). This just pins the descriptor contract.
  using potentials::PassDecl;
  using potentials::PassKind;
  const PassDecl eam[] = {
      {PassKind::Density, true, false, false, 44},
      {PassKind::Embedding, false, false, false, 30},
      {PassKind::Force, true, false, false, 40},
  };
  EXPECT_EQ(eam[0].accum_fracbits, 44);  // DensityAccum Q19.44
  EXPECT_EQ(eam[2].accum_fracbits, 40);  // ForceAccum Q24.40
  EXPECT_TRUE(eam[0].reuses_candidates);
  EXPECT_FALSE(eam[1].reuses_candidates);  // embedding is a local per-atom map
}

// DensityAccum is integer-associative just like ForceAccum (B1): ρ is order-
// free regardless of neighbour-walk / zone / thread order (and the redundant
// halo recompute on a neighbouring node) — the determinism half of §4.1.
TEST(ManyBodyConfig, DensityAccumOrderFree) {
  const double v[4] = {0.7, 1.3e-3, 42.0, 9.1e-5};
  core::fixed::DensityAccum fwd, rev;
  for (int i = 0; i < 4; ++i) fwd.add(v[i]);
  for (int i = 3; i >= 0; --i) rev.add(v[i]);
  EXPECT_EQ(fwd.raw, rev.raw);  // bitwise, not just ==value()
  EXPECT_NEAR(fwd.value(), 0.7 + 1.3e-3 + 42.0 + 9.1e-5, 1e-10);
}

// --- config: `potential.type: eam` parses (valid), pair types still parse ---

TEST(ManyBodyConfig, EamConfigParses) {
  TempConfig eam(R"(
run: { steps: 10, ensemble: nve, seed: 1 }
units: metal
precision: { mode: deterministic_fp64 }
geometry: { file: reference_data/al_fcc_72.data, format: lammps_data }
boundary: { x: periodic, y: periodic, z: free }
potential: { type: eam, r_cut: 4.0 }
timestep: { mode: fixed, dt_initial: 0.001 }
)");
  io::Config c = io::load_config(eam.path());
  EXPECT_EQ(c.pot_type, "eam");
  EXPECT_DOUBLE_EQ(c.rcut, 4.0);

  // regression: a morse config is unaffected by the new enum entry.
  TempConfig morse(R"(
run: { steps: 10, ensemble: nve, seed: 1 }
units: metal
geometry: { file: reference_data/al_fcc_72.data, format: lammps_data }
boundary: { x: periodic, y: periodic, z: free }
potential: { type: morse, r_cut: 4.0, shift: true, morse: { D: 0.29614, alpha: 1.11892, r0: 3.29692 } }
)");
  EXPECT_EQ(io::load_config(morse.path()).pot_type, "morse");
}

// --- the geometry guards: reach_mult=1 is a verbatim no-op; reach_mult=2 is EAM ---

TEST(ManyBodyConfig, ReachMult1IsBitwiseNoOp) {
  core::Box box;
  auto a = tall_box(box, /*Lz=*/50.0, /*pbc_z=*/true);
  const double rcut = 4.0;
  for (int nz : {1, 5, 6, 10}) {
    const auto legacy = core::ZoneDecomposition::build(a, box, nz, rcut);
    const auto m1 = core::ZoneDecomposition::build(a, box, nz, rcut, 1);
    EXPECT_EQ(m1.n_zones, legacy.n_zones);
    EXPECT_DOUBLE_EQ(m1.width, legacy.width);  // bitwise-equal width
    ASSERT_EQ(m1.members.size(), legacy.members.size());
    for (size_t z = 0; z < m1.members.size(); ++z)
      EXPECT_EQ(m1.members[z], legacy.members[z]);  // identical partition
  }
  // legacy throw conditions reproduced under explicit reach_mult=1.
  EXPECT_THROW(core::ZoneDecomposition::build(a, box, 2, rcut, 1),
               std::invalid_argument);  // periodic 2-zone double-count
}

TEST(ManyBodyConfig, ManyBodyZoneGuards) {
  const double rcut = 4.0;  // EAM needs width >= 2*rcut = 8.0, >= 5 periodic zones
  core::Box box;
  auto tall = tall_box(box, /*Lz=*/50.0, /*pbc_z=*/true);

  // width guard: 10 zones of 5.0 Å < 8.0 — fatal for reach_mult=2 (fine for pair).
  EXPECT_NO_THROW(core::ZoneDecomposition::build(tall, box, 10, rcut, 1));
  EXPECT_THROW(core::ZoneDecomposition::build(tall, box, 10, rcut, 2),
               std::invalid_argument);

  // zone-count guard: width OK (>=8.0) but < 5 periodic zones — fatal for EAM
  // even though the same geometry is legal for pair (reach_mult=1).
  EXPECT_NO_THROW(core::ZoneDecomposition::build(tall, box, 4, rcut, 1));  // pair: fine
  EXPECT_THROW(core::ZoneDecomposition::build(tall, box, 4, rcut, 2),
               std::invalid_argument);  // 12.5 Å wide but only 4 periodic zones
  EXPECT_THROW(core::ZoneDecomposition::build(tall, box, 2, rcut, 2),
               std::invalid_argument);  // 25 Å wide but only 2 periodic zones

  // valid EAM decompositions: >= 5 zones each >= 8.0 Å.
  EXPECT_NO_THROW(core::ZoneDecomposition::build(tall, box, 5, rcut, 2));  // 10.0 Å
  EXPECT_NO_THROW(core::ZoneDecomposition::build(tall, box, 6, rcut, 2));  // 8.33 Å
  EXPECT_NO_THROW(core::ZoneDecomposition::build(tall, box, 1, rcut, 2));  // monolith always OK

  // free z drops the periodic zone-count rule: 2 zones of 25 Å is fine for EAM.
  core::Box freebox;
  auto tall_free = tall_box(freebox, /*Lz=*/50.0, /*pbc_z=*/false);
  EXPECT_NO_THROW(core::ZoneDecomposition::build(tall_free, freebox, 2, rcut, 2));
  // ...but width still binds: 8 zones of 6.25 Å < 8.0 — fatal even free.
  EXPECT_THROW(core::ZoneDecomposition::build(tall_free, freebox, 8, rcut, 2),
               std::invalid_argument);

  // reach_mult must be >= 1.
  EXPECT_THROW(core::ZoneDecomposition::build(tall, box, 5, rcut, 0),
               std::invalid_argument);
}
