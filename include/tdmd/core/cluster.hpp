#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <type_traits>

#include "tdmd/core/soa.hpp"

// M3 [ENG]: spatial sorting (Z-order/Morton) + warp-aligned 32-atom clusters +
// cluster-pair list with a skin layer. Replaces the dissertation's "cell with
// <=1 atom" grid (Гл. 3.1) — the grid's ROLE (spatial locality, A9) is kept,
// the layout is GPU-shaped: a cluster of 32 consecutive sorted atoms = 1 warp.
//
// Determinism (INV-9): the sort key is (morton, atom id) — unique => the
// permutation, the clusters and the pair list are bit-identical run-to-run.
//
// The pair list is FULL-neighbor (nbr[A] contains A itself and every B whose
// AABB is within r_cut+skin; both (A,B) and (B,A) appear): each atom
// accumulates only its own force — no cross-thread writes on the GPU, and the
// summation structure mirrors the O(N^2) reference driver (per-i sums).
// Decision recorded in TD_MD_Core_Rationale (full vs half neighbour).
namespace tdmd::core {

// --- Morton encoding (21 bits per axis -> 63-bit code) ---
inline uint64_t expand_bits3(uint64_t v) {
  v &= 0x1fffffULL;
  v = (v | v << 32) & 0x001f00000000ffffULL;
  v = (v | v << 16) & 0x001f0000ff0000ffULL;
  v = (v | v << 8)  & 0x100f00f00f00f00fULL;
  v = (v | v << 4)  & 0x10c30c30c30c30c3ULL;
  v = (v | v << 2)  & 0x1249249249249249ULL;
  return v;
}
inline uint64_t morton3(uint32_t x, uint32_t y, uint32_t z) {
  return expand_bits3(x) | (expand_bits3(y) << 1) | (expand_bits3(z) << 2);
}

struct Cluster {
  int begin = 0, end = 0;     // [begin, end) range in the sorted SoA
  double lo[3]{}, hi[3]{};    // AABB in binning space (wrapped for PBC dims)
};

// Permutes every per-atom array of the SoA (positions, velocities, forces,
// type, mass, id) so that new index k holds old atom perm[k].
template <typename Real>
void apply_permutation(AtomSoA<Real>& a, const std::vector<int>& perm) {
  const int n = a.n;
  auto gather = [&](auto& vec) {
    using V = std::remove_reference_t<decltype(vec)>;
    V tmp(n);
    for (int k = 0; k < n; ++k) tmp[k] = vec[perm[k]];
    vec.swap(tmp);
  };
  gather(a.x);  gather(a.y);  gather(a.z);
  gather(a.vx); gather(a.vy); gather(a.vz);
  gather(a.fx); gather(a.fy); gather(a.fz);
  gather(a.type); gather(a.mass); gather(a.id);
}

class ClusterSet {
 public:
  static constexpr int kClusterSize = 32;  // 1 warp (A9); >=2 clusters per
                                           // block on sm_120 (24 blocks/SM cap)

  // Sorts `a` IN PLACE by (morton, id) and rebuilds clusters + the pair list.
  // `cell` — binning cell size (Å); `cutoff` — pair-list radius = r_cut + skin.
  template <typename Real>
  void build(AtomSoA<Real>& a, const Box& box, double cell, double cutoff) {
    const int n = a.n;

    // binning-space coordinate: wrapped into [lo, lo+L) on periodic dims,
    // raw on free dims (binned relative to the running minimum).
    double bmin[3], span[3];
    auto bcoord = [&](int d, double v) -> double {
      if (box.periodic[d]) {
        const double L = box.len(d);
        return v - L * std::floor((v - box.lo[d]) / L);
      }
      return v;
    };
    for (int d = 0; d < 3; ++d) {
      if (box.periodic[d]) {
        bmin[d] = box.lo[d];
        span[d] = box.len(d);
      } else {
        double mn = 1e300, mx = -1e300;
        const auto& c = (d == 0) ? a.x : (d == 1) ? a.y : a.z;
        for (int i = 0; i < n; ++i) { mn = std::min(mn, c[i]); mx = std::max(mx, c[i]); }
        bmin[d] = mn;
        span[d] = std::max(mx - mn, 1e-12);
      }
    }
    int ncell[3];
    for (int d = 0; d < 3; ++d)
      ncell[d] = std::max(1, std::min(int(span[d] / cell), (1 << 21) - 1));

    std::vector<uint64_t> code(n);
    for (int i = 0; i < n; ++i) {
      uint32_t ci[3];
      const double bc[3] = {bcoord(0, a.x[i]), bcoord(1, a.y[i]), bcoord(2, a.z[i])};
      for (int d = 0; d < 3; ++d) {
        int c = int((bc[d] - bmin[d]) / span[d] * ncell[d]);
        ci[d] = uint32_t(std::clamp(c, 0, ncell[d] - 1));
      }
      code[i] = morton3(ci[0], ci[1], ci[2]);
    }

    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int p, int q) {
      return code[p] != code[q] ? code[p] < code[q] : a.id[p] < a.id[q];
    });
    apply_permutation(a, perm);

    // clusters: consecutive 32-atom chunks of the sorted array
    const int nc = (n + kClusterSize - 1) / kClusterSize;
    clusters.assign(nc, {});
    for (int c = 0; c < nc; ++c) {
      Cluster& cl = clusters[c];
      cl.begin = c * kClusterSize;
      cl.end = std::min(n, cl.begin + kClusterSize);
      for (int d = 0; d < 3; ++d) { cl.lo[d] = 1e300; cl.hi[d] = -1e300; }
      for (int i = cl.begin; i < cl.end; ++i) {
        const double bc[3] = {bcoord(0, a.x[i]), bcoord(1, a.y[i]), bcoord(2, a.z[i])};
        for (int d = 0; d < 3; ++d) {
          cl.lo[d] = std::min(cl.lo[d], bc[d]);
          cl.hi[d] = std::max(cl.hi[d], bc[d]);
        }
      }
    }

    // pair list: B in nbr[A] iff min-image AABB gap <= cutoff (full list, incl.
    // self). O(C^2) — fine through ~10^5 atoms; grid-cull lands with M4 scale.
    nbr.assign(nc, {});
    const double cut2 = cutoff * cutoff;
    for (int p = 0; p < nc; ++p) {
      for (int q = 0; q < nc; ++q) {
        double gap2 = 0.0;
        for (int d = 0; d < 3; ++d) {
          const double ca = 0.5 * (clusters[p].lo[d] + clusters[p].hi[d]);
          const double cb = 0.5 * (clusters[q].lo[d] + clusters[q].hi[d]);
          const double ha = 0.5 * (clusters[p].hi[d] - clusters[p].lo[d]);
          const double hb = 0.5 * (clusters[q].hi[d] - clusters[q].lo[d]);
          double dc = ca - cb;
          if (box.periodic[d]) {
            const double L = box.len(d);
            dc -= L * std::round(dc / L);
            // an AABB pair wider than the box can always touch through PBC
            if (ha + hb >= 0.5 * L) continue;
          }
          const double g = std::max(0.0, std::fabs(dc) - ha - hb);
          gap2 += g * g;
        }
        if (gap2 <= cut2) nbr[p].push_back(q);
      }
    }
  }

  std::vector<Cluster> clusters;
  std::vector<std::vector<int>> nbr;  // full neighbour lists (self included)
};

} // namespace tdmd::core
