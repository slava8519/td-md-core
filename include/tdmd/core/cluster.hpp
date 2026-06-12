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

    // flattened AABB mirrors (centre/half per dim): the narrow phase below
    // touches random clusters — SoA mirrors keep it to two tight loads per
    // dim instead of striding through the 72-byte Cluster structs.
    for (int d = 0; d < 3; ++d) {
      cc_[d].resize(nc);
      hh_[d].resize(nc);
      for (int q = 0; q < nc; ++q) {
        cc_[d][q] = 0.5 * (clusters[q].lo[d] + clusters[q].hi[d]);
        hh_[d][q] = 0.5 * (clusters[q].hi[d] - clusters[q].lo[d]);
      }
    }

    // pair list: B in nbr[A] iff min-image AABB gap <= cutoff (full list,
    // incl. self). LOOSE GRID broad-phase (object rasterization) + the exact
    // AABB test as narrow phase => the pair SET is IDENTICAL to the O(C²)
    // reference (build_pairs_bruteforce; Test_Cluster cross-checks list
    // equality), but the build is O(C) — needed at the flagship 10⁶–10⁷
    // scale (M4).
    //
    // Why rasterization: Z-order chunking leaves MANY clusters straddling
    // curve discontinuities with AABBs of 2×..box-size (measured 10⁶: 12% of
    // clusters exceed 2·median) — any centre-binned grid sized to contain
    // them collapses or pays a fat search radius for everyone. Registering a
    // cluster in EVERY cell its AABB overlaps makes the cost adapt to each
    // cluster's own extent: if two AABBs are within `cutoff` per dim, p's
    // padded interval intersects q's interval, the intersection lies in some
    // cell, and q is registered there — no misses for arbitrarily degenerate
    // straddlers. Decision recorded in Rationale.
    nbr.assign(nc, {});
    const double cut2 = cutoff * cutoff;

    int g[3];
    double edge[3];
    for (int d = 0; d < 3; ++d) {
      g[d] = std::max(1, std::min(int(span[d] / cutoff), 128));
      edge[d] = span[d] / g[d];
    }
    const size_t ncells = size_t(g[0]) * g[1] * g[2];

    // absolute cell range [c0, c0+cnt) (mod g on PBC dims) of interval [lo,hi].
    // Index math in long long with the double clamped BEFORE conversion: a
    // degenerate free dim (flat monolayer => span floored to 1e-12) makes the
    // raw index ~ +-cutoff/1e-12 ~ 5e12 — converting that straight to int is
    // UB ([conv.fpint]; caught by UBSan in review).
    auto cell_idx = [&](int d, double v) -> long long {
      const double t = std::floor((v - bmin[d]) / edge[d]);
      return (long long)(std::clamp(t, -4.0e18, 4.0e18));
    };
    auto range_of = [&](int d, double lo, double hi, int& c0, int& cnt) {
      const long long a = cell_idx(d, lo);
      const long long b = cell_idx(d, hi);
      if (box.periodic[d]) {
        cnt = int(std::min<long long>(b - a + 1, g[d]));
        c0 = int((a % g[d] + g[d]) % g[d]);
      } else {
        const long long ca = std::clamp<long long>(a, 0, g[d] - 1);
        const long long cb = std::clamp<long long>(b, 0, g[d] - 1);
        c0 = int(ca);
        cnt = int(cb - ca + 1);
      }
    };
    auto wrap = [&](int d, int c) { return box.periodic[d] ? c % g[d] : c; };

    // CSR cell table, two passes (deterministic: entries ascending within a
    // cell because pass 2 iterates q ascending)
    std::vector<int> r0(size_t(nc) * 3), rn(size_t(nc) * 3);
    std::vector<int> off(ncells + 1, 0);
    for (int q = 0; q < nc; ++q)
      for (int d = 0; d < 3; ++d)
        range_of(d, clusters[q].lo[d], clusters[q].hi[d], r0[size_t(q) * 3 + d],
                 rn[size_t(q) * 3 + d]);
    auto for_each_cell = [&](int q, auto&& fn) {
      const int* c0 = &r0[size_t(q) * 3];
      const int* cn = &rn[size_t(q) * 3];
      for (int i = 0; i < cn[0]; ++i) {
        const size_t ax = wrap(0, c0[0] + i);
        for (int j = 0; j < cn[1]; ++j) {
          const size_t ay = wrap(1, c0[1] + j);
          for (int k = 0; k < cn[2]; ++k)
            fn((ax * g[1] + ay) * g[2] + wrap(2, c0[2] + k));
        }
      }
    };
    for (int q = 0; q < nc; ++q)
      for_each_cell(q, [&](size_t cell) { ++off[cell + 1]; });
    for (size_t c = 0; c < ncells; ++c) off[c + 1] += off[c];
    std::vector<int> entries(off[ncells]);
    {
      std::vector<int> cur(off.begin(), off.end() - 1);
      for (int q = 0; q < nc; ++q)
        for_each_cell(q, [&](size_t cell) { entries[cur[cell]++] = q; });
    }

    // queries: visit the cells of the cutoff-padded AABB, stamp-dedupe
    // (a cluster registered in several cells must be tested once)
    std::vector<int> stamp(nc, -1);
    std::vector<int> qc0(3), qcn(3);
    for (int p = 0; p < nc; ++p) {
      for (int d = 0; d < 3; ++d) {
        // pad beyond cutoff by >> 1 ulp: the narrow phase computes the gap
        // from rounded centre/half mirrors (cc, hh), the broad phase from raw
        // lo/hi — for a gap within a few ulp of EXACTLY cutoff the two could
        // disagree and the grid would miss a pair the bruteforce keeps
        // (review counterexample). The pad keeps coverage strictly
        // conservative; extra candidates are rejected by aabb_close anyway.
        const double qpad = cutoff + 1e-9 * (span[d] + cutoff);
        range_of(d, clusters[p].lo[d] - qpad, clusters[p].hi[d] + qpad,
                 qc0[d], qcn[d]);
      }
      auto& out = nbr[p];
      for (int i = 0; i < qcn[0]; ++i) {
        const size_t ax = wrap(0, qc0[0] + i);
        for (int j = 0; j < qcn[1]; ++j) {
          const size_t ay = wrap(1, qc0[1] + j);
          for (int k = 0; k < qcn[2]; ++k) {
            const size_t cell = (ax * g[1] + ay) * g[2] + wrap(2, qc0[2] + k);
            for (int e = off[cell]; e < off[cell + 1]; ++e) {
              const int q = entries[e];
              if (stamp[q] == p) continue;
              stamp[q] = p;
              if (aabb_close(p, q, box, cut2)) out.push_back(q);
            }
          }
        }
      }
      // ascending, like the brute-force reference — keeps the traversal (and
      // thus the FP64 accumulation order) independent of the grid layout
      std::sort(out.begin(), out.end());
    }
  }

  // Exact min-image AABB gap test (narrow phase) — single source for the grid
  // path and the brute-force reference. Early exit once gap² > cut² cannot
  // change the verdict (pure rejection shortcut).
  bool aabb_close(int p, int q, const Box& box, double cut2) const {
    double gap2 = 0.0;
    for (int d = 0; d < 3; ++d) {
      const double ha = hh_[d][p], hb = hh_[d][q];
      double dc = cc_[d][p] - cc_[d][q];
      if (box.periodic[d]) {
        const double L = box.len(d);
        dc -= L * std::round(dc / L);
        // an AABB pair wider than the box can always touch through PBC
        if (ha + hb >= 0.5 * L) continue;
      }
      const double g = std::fabs(dc) - ha - hb;
      if (g > 0.0) {
        gap2 += g * g;
        if (gap2 > cut2) return false;
      }
    }
    return gap2 <= cut2;
  }

  // Reference O(C²) pair list over the CURRENT clusters — the cross-test
  // oracle for the grid cull (and nothing else: not used by production paths).
  std::vector<std::vector<int>> build_pairs_bruteforce(const Box& box,
                                                       double cutoff) const {
    const int nc = int(clusters.size());
    const double cut2 = cutoff * cutoff;
    std::vector<std::vector<int>> out(nc);
    for (int p = 0; p < nc; ++p)
      for (int q = 0; q < nc; ++q)
        if (aabb_close(p, q, box, cut2)) out[p].push_back(q);
    return out;
  }

  std::vector<Cluster> clusters;
  std::vector<std::vector<int>> nbr;  // full neighbour lists (self included)

 private:
  std::vector<double> cc_[3], hh_[3];  // AABB centre/half mirrors (SoA)
};

} // namespace tdmd::core
