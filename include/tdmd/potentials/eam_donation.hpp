#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "tdmd/core/fixed_accum.hpp"
#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"        // ZoneDecomposition, PairGeom
#include "tdmd/potentials/eam.hpp"    // EamPotential, kEamPassDecls, assert_eam_symmetric_passes
#include "tdmd/potentials/eam_zone.hpp"  // donation_layout, want_closure_mask, eam_window_layout

// PR-1 (W-contract ladder, step 2) — the SERIAL DONATION ORACLE for EAM.
// Design of record: docs/_meta/PR1_DONATION_ORACLE_DESIGN_2026-07-02.md.
//
// WHAT: the density ρ is assembled by DONATION BATCHES per donation_layout(j,n,pbc)
// (eam_zone.hpp) — self(k) intra-zone pairs + cross(k,k+1) boundary pairs — into
// PERSISTENT per-zone int64 DensAccum arrays keyed by ZONE (not by window), instead
// of the per-window 3-zone recompute. The force/PE/min_r2/φ-once pass stays C-phase
// VERBATIM (donations move ONLY the density pass — audit §3.3); this removes the
// ×3 density redundancy the audit diagnosed, on the serial path first.
//
// BITWISE CONTRACT (G-A domain — EXPLICIT):
//   * forces of ALL atoms (written owned-only, once)  ≡ zone_eam_pass BITWISE;
//   * PE and min_r2                                    ≡ zone_eam_pass EXACTLY;
//   * raw int64 ρ of OWNED atoms at their finalize     ≡ the window multiset (W-1).
//   * halo-ρ in the persistent accumulators LEGITIMATELY DIVERGES from the window-
//     local ρ (the persistent state holds edges the {j-1,j,j+1} window never
//     enumerates, e.g. (j-2,j-1)). A FULL-ρ MEMCMP ACROSS HALO IS FALSE-RED AND
//     FORBIDDEN AS A GATE. (Design §3.6; measured.)
//
// LEMMA W-1 (why donation == window, bitwise): the window pass-1 visits every
// unordered pair TWICE (directed); wx[a]-wx[b] and wx[b]-wx[a] are exact IEEE
// negations; std::round in min-image is odd, so the reduction yields the exact
// negation; squares and the dx²+dy²+dz² sum keep the same operand order ⇒ r2 is
// bitwise equal from either side ⇒ sqrt/eval_rhoa/rint identical, the acceptance
// predicate identical ⇒ donating ONE v into both ends reproduces the directed
// walk's quantum multiset on every per-atom accumulator, and int64 addition is
// associative (B1) ⇒ ρ is bitwise ≡ the window recompute for ANY batch partition
// and ANY batch order.
//
// L-CLOSE (why reads at finalize(j) are bit-final — LOAD-BEARING premise: the
// residence guard g = 0.5*(width - 2*rcut)): a halo neighbour read by the force
// pass lies within rcut of the j slab; its donors lie within 2*rcut ⇒ resident;
// the not-yet-donated edge (j+1,j+2) contributes EXACTLY ZERO to it (the OPEN
// cutoff r2 < rc2 rejects — literally no contribution, not a quantized +0).
// MEASURED (design verification): beyond the guard, on the drift band
// (g_eam, g_pair], G-A goes RED via divergence class 3 — the persistent ρ of a
// read halo atom is STRICTLY FULLER than the window's (it holds already-donated
// out-of-window adjacent edges) ⇒ g_eam is load-bearing for the equivalence
// ITSELF, not merely for read-stability.
//
// STATE SCOPE: "persistent" = survives finalize calls WITHIN one pass, NOT across
// passes: pass-scoped, reset on RECV (PR-2) / reset_pass (serial). A mid-pass
// HALT discards the whole pass; partial batch replay is FORBIDDEN (no API for it
// on purpose — the next RECV does reset_zone).
//
// fb TRAP (MUST): the fixed-point format keys ONLY off pot.math.density_fracbits()
// (44 = Q19.44 / 40 = Q23.40), NEVER off the static kEamPassDecls[0].accum_fracbits
// — a Q23.40 potential would silently break otherwise (tooth Т-12).
namespace tdmd::potentials {

// PR-1 (Т-13 seam — design-verification MUST-FIX): the donation-descriptor
// requirement lives in a FREE span-taking function so the rollback tooth can call
// it directly with a kCOnly copy (EamPotential is `final` with passes() hardwired
// to kEamPassDecls — no descriptor can be injected through the ctor).
inline void assert_eam_donation_descriptor(std::span<const PassDecl> passes) {
  if (passes.empty() || passes[0].w_class != WClass::kAccumB1 ||
      passes[0].donor_roles !=
          (donor_bit(DonorRole::kSelf) | donor_bit(DonorRole::kCrossLo) |
           donor_bit(DonorRole::kCrossHi)))
    throw std::runtime_error(
        "eam_donation: the Density pass must be declared kAccumB1 with donor_roles "
        "self|lo|hi (the PR-1 D5 flip in kEamPassDecls) — without it the ledger "
        "wants nothing and the donation driver would be structurally dead");
}

// Test-only poison knobs; the default is INERT (the MEAM/Tersoff drop_class
// precedent). They express the INTRA-batch faults the ledger is honestly blind to.
struct DonationPoison {
  bool skip_first_pair = false;  // lose ONE accepted pair inside the first non-empty batch
  bool one_sided = false;        // add(v) into the lower end only (the classic half-bug)
  bool ledger_off = false;       // disable the finalize ledger assert (for FP64/G-A witnesses)
};

// A zone's coordinate block in members order (== the ring's msg-payload order).
// key = global atom indices (the φ-once canonical order of the serial oracle).
struct ZoneBlockView {
  const double *x = nullptr, *y = nullptr, *z = nullptr;
  const long* key = nullptr;
  int n = 0;
};

namespace donation_detail {
struct NullPairHook {
  void operator()(long, long) const {}
};
}  // namespace donation_detail

// --- batch executors -------------------------------------------------------
// Unordered enumeration, ONCE per pair; the orientation is fixed dx = a - b.
// The pair body is the ONLY math: geom.reduce (accept iff 1e-18 <= r2 < rc2,
// OPEN cutoff) → ONE eval_rhoa(sqrt(r2)) → the SAME v added into both ends
// (dv computed and discarded — keeps the FP expression shapes of the oracle's
// pass-1). on_pair(g_a, g_b) fires once per ACCEPTED pair (default no-op).

template <typename Math, typename DensAccum, typename Hook = donation_detail::NullPairHook>
void eam_donate_self(const ZoneBlockView& k, const Math& math, const core::PairGeom& geom,
                     std::vector<DensAccum>& rho_k, Hook&& on_pair = {},
                     const DonationPoison* poison = nullptr, bool* skip_pending = nullptr) {
  for (int t = 0; t < k.n; ++t)
    for (int u = t + 1; u < k.n; ++u) {
      double dx = k.x[t] - k.x[u], dy = k.y[t] - k.y[u], dz = k.z[t] - k.z[u], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
      if (poison && poison->skip_first_pair && skip_pending && *skip_pending) {
        *skip_pending = false;  // lose exactly this one accepted pair
        continue;
      }
      double v, dv;
      math.eval_rhoa(std::sqrt(r2), v, dv);
      rho_k[std::size_t(t)].add(v);
      if (!(poison && poison->one_sided)) rho_k[std::size_t(u)].add(v);
      on_pair(k.key[t], k.key[u]);
    }
}

template <typename Math, typename DensAccum, typename Hook = donation_detail::NullPairHook>
void eam_donate_cross(const ZoneBlockView& a, const ZoneBlockView& b, const Math& math,
                      const core::PairGeom& geom, std::vector<DensAccum>& rho_a,
                      std::vector<DensAccum>& rho_b, Hook&& on_pair = {},
                      const DonationPoison* poison = nullptr, bool* skip_pending = nullptr) {
  for (int i = 0; i < a.n; ++i)
    for (int j = 0; j < b.n; ++j) {
      double dx = a.x[i] - b.x[j], dy = a.y[i] - b.y[j], dz = a.z[i] - b.z[j], r2;
      if (!geom.reduce(dx, dy, dz, r2)) continue;
      if (poison && poison->skip_first_pair && skip_pending && *skip_pending) {
        *skip_pending = false;
        continue;
      }
      double v, dv;
      math.eval_rhoa(std::sqrt(r2), v, dv);
      rho_a[std::size_t(i)].add(v);
      if (!(poison && poison->one_sided)) rho_b[std::size_t(j)].add(v);
      on_pair(a.key[i], b.key[j]);
    }
}

// --- the serial residence guard (the L-CLOSE load-bearing premise) ----------
// Formula-mirror of EamRing::membership_ok (the formula, wrap, cyclic min, free-edge
// exemptions): g = 0.5*(width - 2*rcut) — the MANY-BODY reach tolerance, NOT the
// pair (width-rcut)/2. Serial coordinates are static during the assembly, so the
// guard runs once in the ctor (the ring's per-pass END check is the PR-2 twin).
// Derivation: two atoms of zones k and k+2 are separated by >= width - 2g =
// 2*rcut > rcut ⇒ no density pair exists ⇒ a non-donated edge contributes a
// LITERAL zero (open cutoff) — no quantized +0.
// g_override is a TEST-ONLY lever for the D7 mutation teeth; PR-2 must NOT
// forward it into the ring.
template <typename Real>
void check_donation_residence(const core::AtomSoA<Real>& a, const core::Box& box,
                              const core::ZoneDecomposition& zd, double g) {
  const double w = zd.width;
  const double lo_box = box.lo[2], Lz = box.len(2);
  for (int zi = 0; zi < zd.n_zones; ++zi) {
    const double lo = lo_box + zi * w, hi = lo + w;
    const bool first = (zi == 0), last = (zi == zd.n_zones - 1);
    for (int gidx : zd.members[zi]) {
      double zw = double(a.z[gidx]);
      if (box.periodic[2]) zw -= Lz * std::floor((zw - lo_box) / Lz);  // wrap
      double excess = 0.0;
      if (zw < lo) {
        excess = lo - zw;
        if (box.periodic[2]) excess = std::min(excess, zw + Lz - hi);  // cyclic min
        else if (first) excess = 0.0;
      } else if (zw > hi) {
        excess = zw - hi;
        if (box.periodic[2]) excess = std::min(excess, lo + Lz - zw);
        else if (last) excess = 0.0;
      }
      if (excess > g)
        throw std::runtime_error(
            "donation residence: atom beyond g=0.5*(width-2*rcut) of its zone slab "
            "— donation completeness premise broken");
    }
  }
}

// --- the donated finalize: passes 2/3 of the C-phase, verbatim --------------
// Body = eam_window_force (eam_zone.hpp) SYMBOL-FOR-SYMBOL with THREE controlled
// deltas: (1) pass-1 replaced by the rho_w INPUT (raw copies of the donated ρ in
// window gather order; values read via .value() — the same int64→double path);
// (2) the rho_cap check MOVED from pass-2 to the pass-3 OWNED loop (К3): eval_F
// runs for ALL m as today (kWrapup), but the cap fires only on an owned atom's
// FULL ρ — the trigger set is pass-equivalent (see the driver header §4 note);
// (3) trace.on_neighbor_read fires per ACCEPTED pass-3 neighbour (G-LCLOSE).
// HONEST parity caveat: this is a ~30-line COPY, not a call (eam_window_force
// fuses pass-1 and is frozen as the oracle); the anti-drift guard for the copy
// IS the G-A gate itself.
struct NullDonationTrace {
  void on_neighbor_read(int, long, long long) {}
};

template <typename Math, typename DensAccum, typename Trace = NullDonationTrace>
void eam_window_force_from_rho(const double* wx, const double* wy, const double* wz,
                               const long* key, int m, const int* owned, int n_owned,
                               const Math& math, const core::PairGeom& geom,
                               double rho_cap, const DensAccum* rho_w,
                               std::vector<core::fixed::ForceAccum>& wFx,
                               std::vector<core::fixed::ForceAccum>& wFy,
                               std::vector<core::fixed::ForceAccum>& wFz,
                               core::fixed::EnergyAccum& pe, double& min_r2,
                               Trace&& trace = {}, int zone_j = -1) {
  // pass 2: embedding F'(ρ) for window atoms (kWrapup — ALL m; no cap here, К3)
  std::vector<double> fp(m);
  for (int aa = 0; aa < m; ++aa) {
    const double r = rho_w[aa].value();
    double F, Fpv;
    math.eval_F(r, F, Fpv);
    fp[aa] = Fpv;
  }
  // pass 3: FULL-NEIGHBOUR force for OWNED atoms only
  for (int o = 0; o < n_owned; ++o) {
    const int ii = owned[o];
    // К3: the cap fires ONLY on an owned atom's FULL ρ, at its owner finalize
    if (rho_w[ii].value() > rho_cap)
      throw std::runtime_error(
          "donated finalize: full ρ of OWNED atom exceeds the F(ρ) grid (zone " +
          std::to_string(zone_j) + ")");
    double Fi, Fpi;
    math.eval_F(rho_w[ii].value(), Fi, Fpi);
    pe.add(Fi);  // embedding energy, once per owned atom
    for (int bb = 0; bb < m; ++bb) {
      if (bb == ii) continue;
      double dx = wx[ii] - wx[bb], dy = wy[ii] - wy[bb], dz = wz[ii] - wz[bb], r2;
      const bool ok = geom.reduce(dx, dy, dz, r2);
      min_r2 = std::min(min_r2, r2);  // before cutoff — captures overlaps too
      if (!ok) continue;
      trace.on_neighbor_read(zone_j, key[bb], rho_w[bb].raw);  // G-LCLOSE witness
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

// --- pass-scoped donation state (keyed by ZONE, not window) ------------------
// PR-2 keys the SAME layout by zone_id under PBC rotation (slot != label); the
// serial path has slot ≡ label (pass_rotation == 0 semantics).
template <typename DensAccum>
struct EamDonationState {
  std::vector<std::vector<DensAccum>> rho;  // [zone][member_idx] — members order
  std::vector<uint32_t> ledger;             // [zone] closure bits (closure_bit)
  void reset_pass(const core::ZoneDecomposition& zd) {
    rho.assign(std::size_t(zd.n_zones), {});
    ledger.assign(std::size_t(zd.n_zones), 0u);
    for (int z = 0; z < zd.n_zones; ++z)
      rho[std::size_t(z)].assign(zd.members[std::size_t(z)].size(), DensAccum{});
  }
  void reset_zone(int zid, std::size_t n_members) {  // PR-2: on RECV (pass-scoped)
    rho[std::size_t(zid)].assign(n_members, DensAccum{});
    ledger[std::size_t(zid)] = 0u;
  }
};

// --- the step-wise donation pass ---------------------------------------------
// The seams PR-2 splices one-to-one: donate_self on the drift front, donate_cross
// on CoRes, finalize inside finalize_owned, ledger check before END.
template <typename Real, typename Math, typename DensAccum, typename Trace = NullDonationTrace>
class EamDonationPass {
 public:
  EamDonationPass(core::AtomSoA<Real>& a, const core::Box& box,
                  const core::ZoneDecomposition& zd, const EamPotential<Real, Math>& pot,
                  Trace* trace = nullptr, double g_override = -1.0,
                  const DonationPoison& poison = {})
      : a_(a), box_(box), zd_(zd), pot_(pot), geom_(box, pot.math.rcut),
        rho_cap_(pot.math.density_grid_max()), n_(zd.n_zones),
        pbc_(box.periodic[2]), trace_(trace), poison_(poison),
        skip_pending_(poison.skip_first_pair) {
    validate_pass_decls(pot_.passes());
    assert_eam_symmetric_passes(pot_.passes(), "EamDonationPass");
    assert_eam_donation_descriptor(pot_.passes());
    // width guard — same text/condition as zone_eam_pass (the symmetric residence)
    if (zd_.n_zones > 1 && zd_.width < 2.0 * pot_.math.rcut)
      throw std::runtime_error(
          "zone_eam_pass: zone width < 2·rcut — three-zone window insufficient for "
          "the EAM force range (potential.rcut and the zone decomposition must "
          "use the same rcut)");
    // the L-CLOSE residence premise (n>=3 mirrors the ring's guard condition;
    // n<3 ⇒ donations cover all pairs trivially)
    if (n_ >= 3) {
      const double g = (g_override < 0.0) ? 0.5 * (zd_.width - 2.0 * pot_.math.rcut)
                                          : g_override;
      check_donation_residence(a_, box_, zd_, g);
    }
    // copy per-zone coordinate blocks by value (bit-preserving; == the ring's
    // ZoneMsg payload; serial coords are static during the pass)
    zx_.resize(std::size_t(n_)); zy_.resize(std::size_t(n_)); zz_.resize(std::size_t(n_));
    gkey_.resize(std::size_t(n_));
    for (int z = 0; z < n_; ++z) {
      const auto& mem = zd_.members[std::size_t(z)];
      zx_[z].reserve(mem.size()); zy_[z].reserve(mem.size()); zz_[z].reserve(mem.size());
      gkey_[z].reserve(mem.size());
      for (int g : mem) {
        zx_[z].push_back(double(a_.x[g]));
        zy_[z].push_back(double(a_.y[g]));
        zz_[z].push_back(double(a_.z[g]));
        gkey_[z].push_back(long(g));
      }
    }
    st_.reset_pass(zd_);
  }

  // self(k): ledger bit set even for an EMPTY zone (the vacuous-batch convention
  // — a bit means "the batch EXECUTED, possibly vacuously"; else END starves).
  // Exactly-once: a repeated batch is a runtime INV-8 violation ⇒ THROW.
  void donate_self(int k) {
    const uint32_t bit = closure_bit(0, DonorRole::kSelf);
    if (st_.ledger[std::size_t(k)] & bit)
      throw std::runtime_error("eam_donation: self batch executed twice (zone " +
                               std::to_string(k) + ") — INV-8 violation");
    eam_donate_self<Math, DensAccum>(block_(k), pot_.math, geom_, st_.rho[std::size_t(k)],
                                     donation_detail::NullPairHook{},
                                     &poison_, &skip_pending_);
    st_.ledger[std::size_t(k)] |= bit;
  }
  template <typename Hook>
  void donate_self(int k, Hook&& h) {
    const uint32_t bit = closure_bit(0, DonorRole::kSelf);
    if (st_.ledger[std::size_t(k)] & bit)
      throw std::runtime_error("eam_donation: self batch executed twice (zone " +
                               std::to_string(k) + ") — INV-8 violation");
    eam_donate_self<Math, DensAccum>(block_(k), pot_.math, geom_, st_.rho[std::size_t(k)],
                                     std::forward<Hook>(h), &poison_, &skip_pending_);
    st_.ledger[std::size_t(k)] |= bit;
  }

  // cross(a,b): b MUST be the cyclic successor of a. EDGE-ORDER WARNING (the spot
  // PR-2 would get wrong by hand): close_cross_batch takes the EDGE ENDS, not the
  // numeric order — a is the zone whose UPPER boundary is the edge (CrossHi → a,
  // CrossLo → b). The seam donate_cross(n-1, 0) marks CrossHi in zone n-1 and
  // CrossLo in zone 0. A numeric sort (lo=0) would invert the roles; the want-mask
  // assert in finalize self-catches it, but the convention is named here on purpose.
  void donate_cross(int za, int zb) { donate_cross(za, zb, donation_detail::NullPairHook{}); }
  template <typename Hook>
  void donate_cross(int za, int zb, Hook&& h) {
    if (zb != (za + 1) % std::max(n_, 1) && !(n_ == 1 && za == 0 && zb == 0))
      throw std::runtime_error("eam_donation: cross(a,b) requires b == cyclic successor of a");
    const uint32_t hi_bit = closure_bit(0, DonorRole::kCrossHi);
    if (st_.ledger[std::size_t(za)] & hi_bit)
      throw std::runtime_error("eam_donation: cross batch executed twice (edge " +
                               std::to_string(za) + "," + std::to_string(zb) +
                               ") — INV-8 violation");
    eam_donate_cross<Math, DensAccum>(block_(za), block_(zb), pot_.math, geom_,
                                      st_.rho[std::size_t(za)], st_.rho[std::size_t(zb)],
                                      std::forward<Hook>(h), &poison_, &skip_pending_);
    close_cross_batch(st_.ledger[std::size_t(za)], st_.ledger[std::size_t(zb)], 0);
  }

  // finalize(j): (а) ledger[j] == want_closure_mask (unless poison.ledger_off);
  // (б) window gather per eam_window_layout in BLOCK order [pred][center][succ]
  //     with slot DEDUP by zone_id (pbc n=1 emits duplicates — К7); key = global
  //     atom index; raw-gather of the donated ρ;
  // (в) eam_window_force_from_rho; (г) scatter owned forces into a.f (+=).
  void finalize(int j) {
    if (!poison_.ledger_off) {
      const uint32_t want = want_closure_mask(pot_.passes(), j, n_, pbc_);
      if (st_.ledger[std::size_t(j)] != want)
        throw std::runtime_error(
            "eam_donation: finalize(" + std::to_string(j) + ") ledger " +
            std::to_string(st_.ledger[std::size_t(j)]) + " != want " +
            std::to_string(want) + " — a donation batch is missing or late");
    }
    int wslots[3];
    const int nw = eam_window_layout(j, n_, pbc_, wslots);
    // dedup by zone id, preserving first occurrence (block order) — pbc n=1 emits
    // {0,0,0}; the oracle's sort+unique dedup is load-bearing exactly there (К7)
    int slots[3], ns = 0;
    for (int s = 0; s < nw; ++s) {
      bool dup = false;
      for (int t = 0; t < ns; ++t) dup = dup || (slots[t] == wslots[s]);
      if (!dup) slots[ns++] = wslots[s];
    }
    int m = 0;
    for (int s = 0; s < ns; ++s) m += int(gkey_[std::size_t(slots[s])].size());
    std::vector<double> wx(m), wy(m), wz(m);
    std::vector<long> key(m);
    std::vector<DensAccum> rho_w(m);
    std::vector<int> owned;
    int at = 0;
    for (int s = 0; s < ns; ++s) {
      const int z = slots[s];
      const auto& kx = zx_[std::size_t(z)];
      for (std::size_t t = 0; t < kx.size(); ++t, ++at) {
        wx[at] = kx[t];
        wy[at] = zy_[std::size_t(z)][t];
        wz[at] = zz_[std::size_t(z)][t];
        key[at] = gkey_[std::size_t(z)][t];
        rho_w[at] = st_.rho[std::size_t(z)][t];  // raw copy — bit-trivial
        if (z == j) owned.push_back(at);
      }
    }
    std::vector<core::fixed::ForceAccum> wFx(m), wFy(m), wFz(m);
    if (trace_)
      eam_window_force_from_rho<Math, DensAccum, Trace&>(
          wx.data(), wy.data(), wz.data(), key.data(), m, owned.data(),
          int(owned.size()), pot_.math, geom_, rho_cap_, rho_w.data(), wFx, wFy, wFz,
          pe_, min_r2_, *trace_, j);
    else
      eam_window_force_from_rho<Math, DensAccum>(
          wx.data(), wy.data(), wz.data(), key.data(), m, owned.data(),
          int(owned.size()), pot_.math, geom_, rho_cap_, rho_w.data(), wFx, wFy, wFz,
          pe_, min_r2_, NullDonationTrace{}, j);
    // scatter owned forces back (each atom owned by exactly one zone ⇒ once)
    for (int o = 0; o < int(owned.size()); ++o) {
      const int loc = owned[o];
      const long g = key[loc];
      a_.fx[g] += Real(wFx[loc].value());
      a_.fy[g] += Real(wFy[loc].value());
      a_.fz[g] += Real(wFz[loc].value());
    }
  }

  EamAccum finish() {
    EamAccum acc;
    acc.pe = pe_.value();
    acc.min_r2 = min_r2_;
    acc.virial = 0.0;
    return acc;
  }

  const EamDonationState<DensAccum>& state() const { return st_; }

 private:
  ZoneBlockView block_(int z) const {
    ZoneBlockView v;
    v.x = zx_[std::size_t(z)].data();
    v.y = zy_[std::size_t(z)].data();
    v.z = zz_[std::size_t(z)].data();
    v.key = gkey_[std::size_t(z)].data();
    v.n = int(gkey_[std::size_t(z)].size());
    return v;
  }

  core::AtomSoA<Real>& a_;
  const core::Box& box_;
  const core::ZoneDecomposition& zd_;
  const EamPotential<Real, Math>& pot_;
  core::PairGeom geom_;
  double rho_cap_;
  int n_;
  bool pbc_;
  Trace* trace_;
  DonationPoison poison_;
  bool skip_pending_;
  std::vector<std::vector<double>> zx_, zy_, zz_;
  std::vector<std::vector<long>> gkey_;
  EamDonationState<DensAccum> st_;
  core::fixed::EnergyAccum pe_;
  double min_r2_ = 1e300;
};

// --- the canonical serial driver ---------------------------------------------
// The scan mirrors the T-SIM model: for each position j execute the batches of
// donation_layout(j,n,pbc) (self FIRST, then cross), then finalize(j) IN-SCAN —
// except pbc j=0, whose finalize is deferred to the tail (the defer_head mirror).
// §8.1 CONTRACT (DECIDED: the ARRIVAL variant — see TD_MD_Core_WContract_v1_0.md
// §9): serial execution glues both candidates (batches run between "arrival" and
// finalize of the same position); the wording that binds PR-2 lives in the doc.
template <typename Real, typename Math, typename DensAccum,
          typename Trace = NullDonationTrace>
EamAccum zone_eam_pass_donated_impl(core::AtomSoA<Real>& a, const core::Box& box,
                                    const core::ZoneDecomposition& zd,
                                    const EamPotential<Real, Math>& pot,
                                    const DonationPoison& poison, Trace* trace,
                                    double g_override) {
  EamDonationPass<Real, Math, DensAccum, Trace> pass(a, box, zd, pot, trace,
                                                     g_override, poison);
  const int n = zd.n_zones;
  const bool pbc = box.periodic[2];
  for (int j = 0; j < n; ++j) {
    const DonationBatches b = donation_layout(j, n, pbc);
    for (int i = 0; i < b.n_self; ++i) pass.donate_self(b.self[i]);
    for (int i = 0; i < b.n_cross; ++i) pass.donate_cross(b.cross[i][0], b.cross[i][1]);
    if (!(pbc && n > 1 && j == 0)) pass.finalize(j);
  }
  if (pbc && n > 1) pass.finalize(0);  // defer_head mirror (the ring's tail)
  return pass.finish();
}

// fb dispatch (44 = Q19.44 / 40 = Q23.40) off the RUNTIME density_fracbits() —
// mirrors zone_eam_pass; the static descriptor fb is documentation (Т-12).
template <typename Real, typename Math>
EamAccum zone_eam_pass_donated(core::AtomSoA<Real>& a, const core::Box& box,
                               const core::ZoneDecomposition& zd,
                               const EamPotential<Real, Math>& pot,
                               const DonationPoison& poison = {},
                               double g_override = -1.0) {
  const int fb = pot.math.density_fracbits();
  if (fb == 44)
    return zone_eam_pass_donated_impl<Real, Math, core::fixed::FixedAccum<44>>(
        a, box, zd, pot, poison, static_cast<NullDonationTrace*>(nullptr), g_override);
  if (fb == 40)
    return zone_eam_pass_donated_impl<Real, Math, core::fixed::FixedAccum<40>>(
        a, box, zd, pot, poison, static_cast<NullDonationTrace*>(nullptr), g_override);
  throw std::runtime_error("zone_eam_pass_donated: unexpected density_fracbits (not 44/40)");
}

}  // namespace tdmd::potentials
