#pragma once
#include <string>
#include <cmath>
#include <algorithm>
#include <climits>

// Analytical GPU-occupancy model (M2.5 Tier-1). Pure C++, NO CUDA — runs on any
// host (incl. the Mac mini M4 Pro). It answers risk A2: does a thin TD zone
// (~R_cut wide) supply enough warps to keep a GPU busy, and if not, how many
// zones must run concurrently to recover utilization?
//
// IMPORTANT: device specs are *inputs*, not baked-in truth. On the target GPU
// feed real values from cudaGetDeviceProperties / `deviceQuery` (Tier-2). The
// fields below map 1:1 to cudaDeviceProp.
namespace tdmd::probe {

struct DeviceProfile {
  std::string name;
  int  sm_count          = 0;   // cudaDeviceProp::multiProcessorCount
  int  warp_size         = 32;  // ::warpSize
  int  max_warps_per_sm  = 0;   // maxThreadsPerMultiProcessor / warpSize
  int  max_blocks_per_sm = 0;   // ::maxBlocksPerMultiProcessor
  long regs_per_sm       = 0;   // ::regsPerMultiprocessor
  long smem_per_sm       = 0;   // ::sharedMemPerMultiprocessor (bytes)
  bool provisional       = true;// true => specs NOT confirmed from deviceQuery
};

// Force-kernel resource estimate. No compiled kernel exists yet (M3); refine
// with `nvcc -Xptxas -v` / Nsight Compute on the target.
struct KernelProfile {
  int  block_size      = 128;  // threads per block
  int  regs_per_thread = 40;   // estimate for a pair-force kernel
  long smem_per_block  = 0;    // bytes (0 = no shared-mem neighbour cache)
};

enum class Limiter { Warps, Blocks, Registers, SharedMem };
inline const char* to_string(Limiter l) {
  switch (l) {
    case Limiter::Warps:     return "warps";
    case Limiter::Blocks:    return "blocks";
    case Limiter::Registers: return "registers";
    case Limiter::SharedMem: return "shared-mem";
  }
  return "?";
}

struct OccupancyResult {
  int     active_blocks_per_sm = 0;
  int     active_warps_per_sm  = 0;
  double  occupancy            = 0.0;  // active_warps / max_warps_per_sm
  Limiter limiter              = Limiter::Warps;
};

inline long ceil_div(long a, long b) { return (a + b - 1) / b; }

// Standard NVIDIA per-SM theoretical occupancy. Depends on the kernel's
// resources and the device's per-SM limits — NOT on problem size.
inline OccupancyResult per_sm_occupancy(const DeviceProfile& d,
                                        const KernelProfile& k) {
  const int warps_per_block = (int)ceil_div(k.block_size, d.warp_size);
  const int by_warps  = d.max_warps_per_sm / warps_per_block;
  const int by_blocks = d.max_blocks_per_sm;
  const long regs_per_block = (long)k.regs_per_thread * k.block_size;
  const int by_regs = regs_per_block > 0 ? (int)(d.regs_per_sm / regs_per_block) : INT_MAX;
  const int by_smem = k.smem_per_block > 0 ? (int)(d.smem_per_sm / k.smem_per_block) : INT_MAX;

  int active_blocks = std::min(std::min(by_warps, by_blocks), std::min(by_regs, by_smem));
  if (active_blocks < 0) active_blocks = 0;

  OccupancyResult r;
  r.active_blocks_per_sm = active_blocks;
  r.active_warps_per_sm  = active_blocks * warps_per_block;
  r.occupancy = d.max_warps_per_sm > 0
      ? double(r.active_warps_per_sm) / d.max_warps_per_sm : 0.0;
  if      (active_blocks == by_warps)  r.limiter = Limiter::Warps;
  else if (active_blocks == by_blocks) r.limiter = Limiter::Blocks;
  else if (active_blocks == by_regs)   r.limiter = Limiter::Registers;
  else                                 r.limiter = Limiter::SharedMem;
  return r;
}

// --- problem-size model ---
struct Material { std::string name; double number_density; };  // atoms / Å^3

// Cubic-box side (Å) for N atoms at density rho.
inline double box_side(long n_atoms, double rho) {
  return std::pow(double(n_atoms) / rho, 1.0 / 3.0);
}

// Atoms in a slab zone of width `zone_width` (Å) across a cubic box of N atoms.
// atoms = rho · L² · zone_width,  L = (N/rho)^{1/3}.
inline double atoms_per_zone(long n_atoms, double rho, double zone_width) {
  const double cross = std::pow(double(n_atoms) / rho, 2.0 / 3.0);  // L²
  return rho * cross * zone_width;
}

struct FillResult {
  double atoms_per_zone     = 0.0;
  long   warps_per_zone     = 0;
  long   n_zones_in_system  = 0;   // L / zone_width along the decomposition axis
  long   device_warp_cap    = 0;   // sm_count · active_warps_per_sm
  double single_zone_fill   = 0.0; // min(1, warps_per_zone / capacity)
  long   zones_to_saturate  = 0;   // ceil(capacity / warps_per_zone)
};

// Device utilization for one thin zone, plus how many zones must run together.
inline FillResult device_fill(long n_atoms, const Material& mat, double zone_width,
                              const DeviceProfile& d, const OccupancyResult& occ) {
  FillResult f;
  f.atoms_per_zone = atoms_per_zone(n_atoms, mat.number_density, zone_width);
  f.warps_per_zone = std::max<long>(1, (long)std::ceil(f.atoms_per_zone / d.warp_size));
  f.n_zones_in_system =
      std::max<long>(1, (long)(box_side(n_atoms, mat.number_density) / zone_width));
  f.device_warp_cap = (long)d.sm_count * occ.active_warps_per_sm;
  f.single_zone_fill = f.device_warp_cap > 0
      ? std::min(1.0, double(f.warps_per_zone) / f.device_warp_cap) : 0.0;
  f.zones_to_saturate = f.warps_per_zone > 0
      ? std::max<long>(1, ceil_div(f.device_warp_cap, f.warps_per_zone)) : 0;
  return f;
}

} // namespace tdmd::probe
