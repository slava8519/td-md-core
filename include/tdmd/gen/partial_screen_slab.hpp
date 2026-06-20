#pragma once
#include <array>
#include <cmath>
#include <vector>

#include "tdmd/core/soa.hpp"

// M6 / MEAM-ladder Me3 — the CONSTRUCTED partial-screening z-slab fixture. THE CRUX of Me3:
// the diamond is binary-S (every fcut(C)∈{0,1} ⇒ the screening FORCE is structurally dead — a
// dropped screening-k is invisible), and MEAM's ZBL guard throws before any diamond perturb can
// populate the partial band (measured, README §"SCREENING ... BINARY"). So the screening
// candidate-completeness — the entire reason Me3 is harder than EAM/Te3 — is UNTESTABLE without a
// fixture that has genuine 0<S<1 pairs AND is large enough to host z>1 zones (width ≥ 2·rc = 8 Å).
//
// CONSTRUCTION. The partial-screening motif is the meam_tri3 cluster's RIGID triple
//   i, j with |ij| = 3.2 Å  and  k with |ik| = |jk| = 2.868 Å  (S ≈ 0.50, C ≈ 2.21 ∈ (Cmin,Cmax)).
// The screening C_ikj is a function of the THREE inter-atom distances only ⇒ ROTATION-INVARIANT.
// So the triple is laid out with the i→j BOND ALONG z (i at z0, j at z0+3.2) and k off-axis in x
// at the bond midpoint (z0+1.6). Placing z0 just below a zone boundary B = (zi+1)·width makes the
// owned center i fall in zone zi while k (and j) fall in the ADJACENT zone zi+1 — i.e. a
// partial-screening triple whose screening-k lives one zone away from the bond's owned center
// (the §3.2 non-vacuity precondition #2: the residence loss of the forward-only window is a
// SCREENING loss, not merely a density-donor loss).
//
// Triples are spaced ≥ ~spacing Å apart along z and offset transversely so DISTINCT triples never
// interact (each atom's only in-rc neighbours are its own two partners): the fixture is then a
// disjoint union of identical partial-screening triples, every bond above r_zbl ≈ 1.87 Å, and the
// per-triple screening force fires across a zone boundary. Periodic in all three dims, with x,y ≥
// 2·rc and Lz = n_triples · spacing chosen so each of z∈{5,6,8} zones is ≥ 2·rc wide.
//
// VALIDATE in the probe/test (do NOT assume): meam_direct_fp64 must report n_screened_partial > 0,
// no ZBL throw, and ≥1 partial triple with its screening-k in a zone adjacent to the bond center.
namespace tdmd::gen {

struct PartialScreenSlab {
  core::AtomSoA<double> atoms;
  core::Box box;
  int n_triples = 0;
  double bond = 0.0;       // |i→j| (along z)
  double k_off = 0.0;      // k transverse (x) offset from the bond axis
  double k_z = 0.0;        // k z-offset from i (bond midpoint)
  double boundary_z = 0.0; // the zone boundary the lowest triple straddles (for n_zones0)
};

// n_triples triples stacked along z. spacing = the z-period each triple occupies (≥ ~7.4 ⇒ even at
// z=8 zones the width Lz/8 = n_triples·spacing/8 ≥ 2·rc when n_triples ≥ 8). transverse_jitter
// alternates the (x,y) anchor of successive triples so no two triples come within rc of each other.
inline PartialScreenSlab make_partial_screen_slab(int n_triples = 8, double spacing = 8.0,
                                                  double lx = 16.0, double ly = 16.0,
                                                  int n_zones0 = 8) {
  PartialScreenSlab s;
  s.n_triples = n_triples;
  // The rigid partial-S triple (meam_tri3 geometry), bond rotated onto z, k off-axis in x.
  const double bond = 3.2;                                   // |i→j|
  const double rik = 2.8678214728256712;                    // = |i→k| = |j→k|
  const double kz = bond * 0.5;                             // k at the bond midpoint in z
  const double kx = std::sqrt(rik * rik - kz * kz);        // k transverse offset (≈ 2.380 Å)
  s.bond = bond;
  s.k_off = kx;
  s.k_z = kz;

  const double Lz = n_triples * spacing;
  s.box.lo = {0, 0, 0};
  s.box.hi = {lx, ly, Lz};
  s.box.periodic = {true, true, true};
  const double width0 = Lz / n_zones0;  // zone width at the design decomposition

  core::AtomSoA<double> a;
  a.resize(3 * n_triples);
  for (int t = 0; t < n_triples; ++t) {
    // Straddle a zone boundary: put i just below boundary (t+? )·width0 so k (at i_z + kz) and j
    // (at i_z + bond) land in the NEXT zone. Anchor the lowest triple so the boundary it straddles
    // is interior (not the wrap seam) for the n_zones0 decomposition; higher triples follow with
    // an extra full-zone step so each occupies its own z-band and sits below a distinct boundary.
    const double boundary = (t + 1) * width0;
    const double iz = boundary - 0.30;  // i just below the boundary ⇒ k,j cross into zone t+1
    if (t == 0) s.boundary_z = boundary;
    // Transverse anchor: stagger so triple t's atoms are far (> rc) from triple t±1 in x,y. The
    // triples are already ≥ spacing − bond ≈ 4.8 Å apart in z and we add a transverse shift too.
    const double ax = (t % 2 == 0) ? 2.0 : (lx - 2.0 - kx);  // keep k within [0,lx)
    const double ay = (t % 2 == 0) ? 2.0 : (ly - 4.0);
    const int b = 3 * t;
    // i (lower-key endpoint, owned center of the bond)
    a.x[b + 0] = ax;          a.y[b + 0] = ay;          a.z[b + 0] = iz;
    // j (higher-key endpoint), bond along +z
    a.x[b + 1] = ax;          a.y[b + 1] = ay;          a.z[b + 1] = iz + bond;
    // k (the screening third atom), off-axis in x, at the bond midpoint in z (crosses boundary)
    a.x[b + 2] = ax + kx;     a.y[b + 2] = ay;          a.z[b + 2] = iz + kz;
    for (int u = 0; u < 3; ++u) { a.type[b + u] = 1; a.mass[b + u] = 28.0855; a.id[b + u] = b + u + 1; }
  }
  s.atoms = std::move(a);
  return s;
}

}  // namespace tdmd::gen
