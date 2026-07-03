#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"  // ZoneDecomposition, PairGeom
#include "tdmd/potentials/eam.hpp"  // EamPotential, PassDecl, validate_pass_decls

// M6 PR-E3 — serial multi-pass zone EAM with SYMMETRIC THREE-ZONE residence.
// The bitwise reference for the EAM ring (analog of zones.hpp::zone_force_pass
// for pairs); the threaded TimeConveyor wiring lands in PR-E3b.
//
// Per owned zone S_zi the node holds the window {S_{zi-1}, S_zi, S_{zi+1}}
// (width ≥ 2·rcut ⇒ every density donor of any neighbour of an owned atom is
// resident — M6_EAM_MANYBODY_DESIGN §1.2) and runs three passes:
//   1 density   ρ_j (fixed-point) for every WINDOW atom — complete for any j
//               that is a neighbour of an owned atom;
//   2 embedding F'_j = F'(ρ_j) for window atoms;
//   3 force     FULL-NEIGHBOUR per OWNED atom: f_i = Σ_j (...). No Newton-3
//               sharing across zones — each owned atom's force is written ONCE
//               by its own zone. Bitwise ≡ the Newton-3 monolith because the
//               quantizer rint is ODD: rint(−x) = −rint(x), so the i-side and
//               j-side contributions of a pair quantize to negatives of each
//               other on either accumulation scheme.
//
// Determinism (B1/INV-9): fixed-point ρ and force ⇒ the per-atom result is
// bit-identical for ANY zone count / processing order. DUAL-FORMAT density
// dispatch on density_fracbits() (Q19.44 / Q23.40) — closes the PR-E2 P0 stopgap
// (eam_run_fixed only handled Q19.44).
namespace tdmd::potentials {

// Window atom indices for zone zi. symmetric=true ⇒ {S_{zi-1}, S_zi, S_{zi+1}}
// (the correct EAM residence). symmetric=false ⇒ {S_zi, S_{zi+1}} — the
// forward-only window of the current pair conveyor; EAM-incorrect, kept ONLY so
// the test can demonstrate that it diverges from the monolith (the adversarial
// halo finding, M6 §9). PBC wraps zone indices cyclically.
inline std::vector<int> zone_eam_window(const core::ZoneDecomposition& zd, int zi,
                                        bool pbc_z, bool symmetric) {
  std::vector<int> w;
  const int nz = zd.n_zones;
  auto add_zone = [&](int z) {
    if (pbc_z) z = (z % nz + nz) % nz;
    if (z < 0 || z >= nz) return;
    for (int at : zd.members[z]) w.push_back(at);
  };
  if (symmetric) add_zone(zi - 1);
  add_zone(zi);
  add_zone(zi + 1);
  std::sort(w.begin(), w.end());
  w.erase(std::unique(w.begin(), w.end()), w.end());
  return w;
}

// M6 E5b (F4) — window-slot layout for the 3-zone EAM residence {j-1, j, j+1}:
// the SINGLE source of truth shared by the CPU EamRing AND the GPU EamGpuConveyor
// gather, so the center-vs-edge / PBC-cyclic / free-z-drop slot selection is
// IDENTICAL by construction. A divergence here silently drops a donor ⇒ ρ_j
// truncated ⇒ force wrong on ALL z (invisible to 1-vs-z) — eliminating that
// off-by-one class is exactly why this is one function. Returns nw ∈ {2,3} and
// fills wslots[0..nw) in increasing window order; the CENTER zone j is always
// present (its owned atoms are the ones finalized).
//   PBC (z periodic): cyclic {(j-1+n)%n, j, (j+1)%n} — n≥5 ⇒ all distinct
//     (ZoneDecomposition reach_mult=2 guard). free-z: the in-range subset of
//     {j-1, j, j+1}, edges DROP (j=0→{0,1}, j=n-1→{n-2,n-1}); never clamp
//     (clamping would scoop S_{j-2} or duplicate a slot).
inline int eam_window_layout(int j, int n, bool pbc, int wslots[3]) {
  int nw = 0;
  if (pbc) {
    wslots[nw++] = (j - 1 + n) % n;
    wslots[nw++] = j;
    wslots[nw++] = (j + 1) % n;
  } else {
    for (int d = -1; d <= 1; ++d) {
      const int p = j + d;
      if (p >= 0 && p < n) wslots[nw++] = p;
    }
  }
  return nw;
}

// === PR-0a (W-contract, audit §3.1) — the SINGLE source of truth for the donation
// schedule. Sits next to eam_window_layout (its E5b-F4 precedent — one place, no
// center-vs-edge off-by-one). PURE function; in PR-0a NO ring consumes it (consumers
// are PR-1/2). Contract (SPEC; violating any point = a regression of an audit MUST-FIX):
//  (1) PASS SLOT-space: slot = arrival position; label=(r+slot)%n, r=(h-1)%n
//      (ZoneFSM §7.2; helpers pass_rotation/slot_zone_id below — the SINGLE source of
//      the formulas). The schedule is PASS-INVARIANT. ALL future PERSISTENT structures
//      (ρ-accumulators, CSR, epochs, device mirrors) key by zone_id, NOT by slot;
//      PR-2's tooth is a PBC run of ≥2 full rotations, n≥5, bitwise.
//  (2) self(k): precondition — the false→true edge of ensure_drift(k) with pass h's dt
//      (NOT "on RECV": dt arrives with arrival 0, drift is lazy). drift idempotence is
//      load-bearing (finalize self-drifts wrap-members again).
//  (3) cross(a,b): at CoRes(a,b) — both slots arrived AND drifted.
//  (4) pbc seam cross(n-1,0): available from CoRes(slot n-1, slot 0) = arrival of the
//      LAST zone (ready=n-2), MUST execute STRICTLY BEFORE finalize_owned(n-1), which
//      runs IN-SCAN (the tail's defer_head defers ONLY finalize(0)+sends). Encoded
//      STRUCTURALLY here: the seam is emitted at j=n-1, and by the call contract the
//      batches of position j execute BEFORE finalize_owned(j). The "in the pass tail"
//      variant was REFUTED by the audit verifier — do not resurrect it. The micro-point
//      WITHIN a position's window (arrival vs inside finalize(k-1) after START — audit
//      open question §8.1) — DECIDED (PR-1): the ARRIVAL variant; see
//      TD_MD_Core_WContract_v1_0.md §9. PR-2 schedules donations as a standalone
//      scan-loop step between arrival/drift and finalize, NOT baked into finalize;
//      the finalize(k-1)-after-START alternative (the dissertation letter) lies in
//      the same window and is W-1-equivalent — rejected for the M5b freedom.
//  (5) Exactly one batch per unordered pair of adjacent zones per pass + exactly one
//      self per slot (a generalization of INV-8).
//  (6) Donation accumulators + ledger are strictly pass-scoped (reset on RECV); a
//      mid-pass HALT discards the WHOLE pass, partial batch replay is forbidden.
//      Donations do NOT touch min_r2 / PE / φ-once.
//  (7) LOAD-BEARING "only adjacent edges" completeness premise: the membership_ok guard
//      with g = 0.5*(width - 2*rcut). A PR that relaxes g or changes membership MUST
//      re-derive this function (mutation tooth with a drift fixture — PR-1).
//  (8) Domain: free — any n≥1; pbc — n==1 (degenerates to the free path, no seam) or
//      n≥5 (ZoneDecomposition::build rejects periodic 2..4 at reach_mult=2). pbc n in
//      [2,4] → throw — an EXPLICIT guard (eam_window_layout silently duplicates slots).
struct DonationBatches {
  int n_self = 0, n_cross = 0;
  int self[2] = {-1, -1};                  // self(k) slots; executed FIRST
  int cross[2][2] = {{-1, -1}, {-1, -1}};  // unordered pairs {a,b}; after self
};

inline DonationBatches donation_layout(int j, int n, bool pbc) {
  if (n < 1 || j < 0 || j >= n)
    throw std::invalid_argument("donation_layout: j out of [0,n)");
  if (pbc && n >= 2 && n <= 4)
    throw std::invalid_argument("donation_layout: periodic n_zones 2..4 rejected (reach_mult=2)");
  DonationBatches b;
  if (n == 1) { b.self[b.n_self++] = 0; return b; }   // free path; no seam
  if (j == 0) {                                       // scan 0: slots 0,1 arrived+drifted
    b.self[b.n_self++] = 0;
    b.self[b.n_self++] = 1;
    b.cross[0][0] = 0; b.cross[0][1] = 1; b.n_cross = 1;
    return b;
  }
  if (j + 1 < n) {                                    // 1 <= j <= n-2
    b.self[b.n_self++] = j + 1;                       // the new drift of this scan
    b.cross[0][0] = j; b.cross[0][1] = j + 1; b.n_cross = 1;
    return b;
  }
  if (pbc) { b.cross[0][0] = n - 1; b.cross[0][1] = 0; b.n_cross = 1; }  // (4): seam BEFORE finalize(n-1)
  return b;
}

// Helpers slot<->label — the SINGLE source of the formulas for PR-1/2's persistent
// structures (formulas = the ring's frozen conventions; tooth T-ROT).
inline int pass_rotation(long h, int n, bool pbc) { return pbc ? int((h - 1) % n) : 0; }
inline int slot_zone_id(int r, int slot, int n) { return (r + slot) % n; }

// END-wait mask of the zone in slot j (audit §3.3 MUST-FIX). VACUOUS-BATCH convention:
// a ledger bit means "the batch EXECUTED (possibly vacuously)", NOT "contributed ≥1
// pair". An empty zone (n()==0 is legally resident) sets bits vacuously — else END
// starves. The want side has NO population parameter by construction.
// РАЗГРАНИЧЕНИЕ (PR-2: do not conflate): the PLANNING want-set of finalize(j) — selves of
// ALL window slots + the interior edges (j-1,j),(j,j+1) — is closed by deadline scheduling
// (teeth T8/T-SIM); want_closure_mask(j) is the END BOOKKEEPING of zone j ITSELF (only its
// own self/lo/hi roles) — the pair-conveyor contrib_mask precedent. The future check point
// (PR-2, NOT here): immediately before ZoneFSM::apply(END) in the ring's end path.
// All shipped descriptors (uniform kCOnly) → 0 everywhere = the F-NOOP anchor.
inline uint32_t want_closure_mask(std::span<const potentials::PassDecl> passes,
                                  int j, int n, bool pbc) {
  potentials::validate_pass_decls(passes);
  (void)donation_layout(j, n, pbc);   // shared domain guard (throws on pbc 2..4 / j out)
  const bool cyc    = pbc && n > 1;
  const bool has_lo = cyc || j > 0;
  const bool has_hi = cyc || j + 1 < n;    // edges drop on free-z borders, exactly as
  uint32_t w = 0;                          // eam_window_layout drops slots
  for (std::size_t p = 0; p < passes.size(); ++p) {
    if (passes[p].w_class != potentials::WClass::kAccumB1) continue;
    const uint8_t dr = passes[p].donor_roles;
    using potentials::DonorRole;
    if (dr & potentials::donor_bit(DonorRole::kSelf))
      w |= potentials::closure_bit(int(p), DonorRole::kSelf);
    if (has_lo && (dr & potentials::donor_bit(DonorRole::kCrossLo)))
      w |= potentials::closure_bit(int(p), DonorRole::kCrossLo);
    if (has_hi && (dr & potentials::donor_bit(DonorRole::kCrossHi)))
      w |= potentials::closure_bit(int(p), DonorRole::kCrossHi);
  }
  return w;   // all shipped descriptors (uniform kCOnly) -> 0 everywhere = F-NOOP anchor
}

// M6 PR-E3b-1 — the 3-pass EAM force on ONE owned zone over a CONTIGUOUS window,
// addressed by local index. Extracted from the per-zone body so BOTH the serial
// oracle (global AtomSoA) and the threaded EamRing (fragmented per-zone payloads
// gathered into a window) call the SAME code (the ring has no global AtomSoA
// mid-pass). Bit-exact to the pre-refactor oracle by construction (same multiset
// of int64 contributions). `key` is the canonical φ-once order (serial: global
// atom index; ring: atom id) — either gives φ exactly once per undirected pair,
// PE order-invariant. Forces written into wF{x,y,z}[owned] (local); pe/min_r2
// accumulated. Caller zeroes the accumulators.
template <typename Math, typename DensAccum>
void eam_window_force(const double* wx, const double* wy, const double* wz,
                      const long* key, int m, const int* owned, int n_owned,
                      const Math& math, const core::PairGeom& geom, double rho_cap,
                      std::vector<core::fixed::ForceAccum>& wFx,
                      std::vector<core::fixed::ForceAccum>& wFy,
                      std::vector<core::fixed::ForceAccum>& wFz,
                      core::fixed::EnergyAccum& pe, double& min_r2) {
  // pass 1: density for every window atom
  std::vector<DensAccum> rho(m);
  for (int aa = 0; aa < m; ++aa)
    for (int bb = 0; bb < m; ++bb) {
      if (bb == aa) continue;
      double dx = wx[aa] - wx[bb], dy = wy[aa] - wy[bb], dz = wz[aa] - wz[bb], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
      double v, dv;
      math.eval_rhoa(std::sqrt(r2), v, dv);
      rho[aa].add(v);
    }
  // pass 2: embedding F'(ρ) for window atoms (+ density-range HALT, P1)
  std::vector<double> fp(m);
  for (int aa = 0; aa < m; ++aa) {
    const double r = rho[aa].value();
    if (r > rho_cap)
      throw std::runtime_error("eam_window_force: ρ exceeds the F(ρ) grid");
    double F, Fpv;
    math.eval_F(r, F, Fpv);
    fp[aa] = Fpv;
  }
  // pass 3: FULL-NEIGHBOUR force for OWNED atoms only
  for (int o = 0; o < n_owned; ++o) {
    const int ii = owned[o];
    double Fi, Fpi;
    math.eval_F(rho[ii].value(), Fi, Fpi);
    pe.add(Fi);  // embedding energy, once per owned atom
    for (int bb = 0; bb < m; ++bb) {
      if (bb == ii) continue;
      double dx = wx[ii] - wx[bb], dy = wy[ii] - wy[bb], dz = wz[ii] - wz[bb], r2;
      const bool ok = geom.reduce(dx, dy, dz, r2);
      min_r2 = std::min(min_r2, r2);  // before cutoff — captures overlaps too
      if (!ok) continue;
      const double r = std::sqrt(r2);
      double phi, dphi, ra, dra;
      math.eval_phi(r, phi, dphi);
      math.eval_rhoa(r, ra, dra);
      const double f_over_r = -(dphi + (fp[ii] + fp[bb]) * dra) / r;
      wFx[ii].add(f_over_r * dx);
      wFy[ii].add(f_over_r * dy);
      wFz[ii].add(f_over_r * dz);
      if (key[ii] < key[bb]) pe.add(phi);  // φ once per undirected pair (PE bitwise)
    }
  }
}

template <typename Real, typename Math, typename DensAccum>
EamAccum zone_eam_pass_impl(core::AtomSoA<Real>& a, const core::Box& box,
                            const core::ZoneDecomposition& zd, const Math& math,
                            const std::vector<int>& order, bool symmetric) {
  const core::PairGeom geom(box, math.rcut);
  const int n = a.n;
  core::fixed::EnergyAccum pe;
  double min_r2 = 1e300;  // overlap probe (B10) — system min over visited pairs
  const double rho_cap = math.density_grid_max();
  std::vector<int> pos(n, -1);  // global atom -> window-local index

  for (int zi : order) {
    const auto win = zone_eam_window(zd, zi, box.periodic[2], symmetric);
    const int m = int(win.size());
    // gather the window into contiguous arrays (key = global atom index)
    std::vector<double> wx(m), wy(m), wz(m);
    std::vector<long> key(m);
    for (int aa = 0; aa < m; ++aa) {
      const int g = win[aa];
      wx[aa] = a.x[g]; wy[aa] = a.y[g]; wz[aa] = a.z[g]; key[aa] = g;
      pos[g] = aa;
    }
    std::vector<int> owned;
    owned.reserve(zd.members[zi].size());
    for (int g : zd.members[zi]) owned.push_back(pos[g]);

    std::vector<core::fixed::ForceAccum> wFx(m), wFy(m), wFz(m);
    eam_window_force<Math, DensAccum>(wx.data(), wy.data(), wz.data(), key.data(),
                                      m, owned.data(), int(owned.size()), math,
                                      geom, rho_cap, wFx, wFy, wFz, pe, min_r2);
    // scatter owned forces back (each atom owned by exactly one zone ⇒ once)
    for (int o = 0; o < int(owned.size()); ++o) {
      const int loc = owned[o], g = win[loc];
      a.fx[g] += Real(wFx[loc].value());
      a.fy[g] += Real(wFy[loc].value());
      a.fz[g] += Real(wFz[loc].value());
    }
    for (int aa = 0; aa < m; ++aa) pos[win[aa]] = -1;  // reset for next zone
  }

  EamAccum acc;
  acc.pe = pe.value();
  acc.min_r2 = min_r2;
  return acc;
}

// Dual-format dispatch on the load-time density guard. Writes forces into a.f
// (ACCUMULATED, caller zeroes), returns PE. order = zone processing order
// (permutation of 0..n_zones-1). symmetric=true is the correct EAM residence.
template <typename Real, typename Math>
EamAccum zone_eam_pass(core::AtomSoA<Real>& a, const core::Box& box,
                       const core::ZoneDecomposition& zd,
                       const EamPotential<Real, Math>& pot,
                       const std::vector<int>& order, bool symmetric = true) {
  core::validate_zone_order(order, zd.n_zones);  // permutation guard (UB / silent corruption)
  // Residence guard: the symmetric three-zone window covers ±2·rcut around an
  // owned atom only if the zone is at least 2·rcut wide. Catches a potential/
  // decomposition rcut mismatch (the window would otherwise silently miss
  // donors — deterministically, so 1-vs-z would NOT flag it).
  if (symmetric && zd.n_zones > 1 && zd.width < 2.0 * pot.math.rcut)
    throw std::runtime_error(
        "zone_eam_pass: zone width < 2·rcut — three-zone window insufficient for "
        "the EAM force range (potential.rcut and the zone decomposition must "
        "use the same rcut)");
  const int fb = pot.math.density_fracbits();  // 44 (Q19.44) or 40 (Q23.40)
  if (fb == 44)
    return zone_eam_pass_impl<Real, Math, core::fixed::FixedAccum<44>>(
        a, box, zd, pot.math, order, symmetric);
  if (fb == 40)
    return zone_eam_pass_impl<Real, Math, core::fixed::FixedAccum<40>>(
        a, box, zd, pot.math, order, symmetric);
  throw std::runtime_error("zone_eam_pass: unexpected density_fracbits (not 44/40)");
}

// Convenience overload: identity zone order.
template <typename Real, typename Math>
EamAccum zone_eam_pass(core::AtomSoA<Real>& a, const core::Box& box,
                       const core::ZoneDecomposition& zd,
                       const EamPotential<Real, Math>& pot, bool symmetric = true) {
  std::vector<int> order(zd.n_zones);
  std::iota(order.begin(), order.end(), 0);
  return zone_eam_pass(a, box, zd, pot, order, symmetric);
}

}  // namespace tdmd::potentials
