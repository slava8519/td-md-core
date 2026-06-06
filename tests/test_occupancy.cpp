#include <gtest/gtest.h>
#include "tdmd/probe/occupancy.hpp"

using namespace tdmd::probe;

// M2.5 Tier-1: validate the analytical occupancy model.

TEST(Occupancy, PerSmFullOccupancy) {
  DeviceProfile d{"t", 1, 32, 64, 32, 65536, 200000, true};
  KernelProfile k{256, 32, 0};  // 8 warps/block, light regs, no smem
  auto o = per_sm_occupancy(d, k);
  EXPECT_EQ(o.active_blocks_per_sm, 8);   // min(warps=8, blocks=32, regs=8)
  EXPECT_EQ(o.active_warps_per_sm, 64);
  EXPECT_DOUBLE_EQ(o.occupancy, 1.0);
}

TEST(Occupancy, RegisterLimited) {
  DeviceProfile d{"t", 1, 32, 64, 32, 65536, 200000, true};
  KernelProfile k{256, 64, 0};  // regs_per_block=16384 -> 4 blocks
  auto o = per_sm_occupancy(d, k);
  EXPECT_EQ(o.active_blocks_per_sm, 4);
  EXPECT_EQ(o.active_warps_per_sm, 32);
  EXPECT_DOUBLE_EQ(o.occupancy, 0.5);
  EXPECT_EQ(o.limiter, Limiter::Registers);
}

TEST(Occupancy, SharedMemLimited) {
  DeviceProfile d{"t", 1, 32, 64, 32, 1 << 30, 49152, true};
  KernelProfile k{128, 16, 16384};  // 49152/16384 = 3 blocks
  auto o = per_sm_occupancy(d, k);
  EXPECT_EQ(o.active_blocks_per_sm, 3);
  EXPECT_EQ(o.limiter, Limiter::SharedMem);
}

TEST(Occupancy, AtomsPerZoneAndBox) {
  const double rho = 0.06022;
  EXPECT_NEAR(box_side(1000000, rho), 255.5, 1.0);
  EXPECT_NEAR(atoms_per_zone(1000000, rho, 4.0), 15700.0, 400.0);
  // tiny golden system: a single thin zone is ~one warp's worth of atoms
  EXPECT_LT(atoms_per_zone(72, rho, 4.0), 40.0);
}

TEST(Occupancy, DeviceFillAndSaturation) {
  DeviceProfile d{"t", 84, 32, 48, 32, 65536, 102400, true};
  KernelProfile k{128, 40, 0};
  Material al{"Al", 0.06022};
  auto o = per_sm_occupancy(d, k);

  auto small = device_fill(10000, al, 4.0, d, o);
  auto big   = device_fill(1000000, al, 4.0, d, o);
  EXPECT_LT(small.single_zone_fill, big.single_zone_fill);  // bigger fills more
  EXPECT_GT(big.zones_to_saturate, 1);                       // still needs batching
  // zones_to_saturate · warps_per_zone covers the device capacity
  EXPECT_GE(big.zones_to_saturate * big.warps_per_zone, big.device_warp_cap);
  EXPECT_LE(big.single_zone_fill, 1.0);
}
