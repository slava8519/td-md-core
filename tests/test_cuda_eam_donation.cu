// PR-3a (W-contract ladder — device-donation substrate; design of record:
// docs/_meta/PR3AB_GPU_DONATION_DESIGN_2026-07-07.md §8.1) — the A-gates.
//
// The substrate is DEFAULT-INERT (donate_device=false ⇒ byte-identical to HEAD;
// witnessed by the untouched 14-gate test_cuda_eam_ring suite). Knob-ON is
// OBSERVATIONAL in PR-3a: the hooks run the donation kernels into per-node
// LABEL-keyed device lanes, compose still RECOMPUTES density (the donated
// compose is PR-3b, gated R_W>=1.15). These gates prove the device lanes are
// raw-int64 BITWISE equal to the CPU-donated rho (the L1 chain's first link for
// PR-3b) and that every guard has a live kill.
//
// Counter semantics (design amendment S4): donation_self/cross_batches_ count
// every EXECUTED batch INCLUDING vacuous n==0 ones; A4 checks the absence of
// kernel LAUNCHES for n==0 (memcheck leg), not of the increment.
//
// Kill-mutations that live OUTSIDE this TU (scratch-build, acceptance —
// design amendments M7/S10): (A0) a hook perturbing persistent compose-read
// state (st->dens_scale by 1 ulp) must turn A0 red; (A1-iii) building
// WindowBlocks.label from the SLOT index instead of slot[].fsm.id in
// finalize_owned must turn the pbc >=2-rotation leg of A1 red (free-z has
// rotation r=0 so slot==label — the pbc leg carries the rotation tooth; if it
// is ever trimmed, that tooth dies silently — the G5 recidive class).
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

#include "tdmd/core/soa.hpp"
#include "tdmd/core/zones.hpp"
#include "tdmd/cuda/eam_conveyor_gpu.cuh"        // eam_sn_detail (include-order contract)
#include "tdmd/cuda/eam_window_force_gpu.cuh"    // GpuEamWindowForce + NodeStore + poison
#include "tdmd/potentials/eam.hpp"
#include "tdmd/potentials/eam_analytic.hpp"
#include "tdmd/potentials/eam_ring.hpp"
#include "tdmd/potentials/eam_spline.hpp"
#include "tdmd/potentials/eam_zone.hpp"
#include "tdmd/units.hpp"

namespace core = tdmd::core;
namespace potentials = tdmd::potentials;
namespace tdcu = tdmd::cuda;

namespace {
constexpr double kRcut = 3.0;

core::AtomSoA<double> make_fcc(int nx, int ny, int nz, double alat, core::Box& box) {
  box.lo = {0, 0, 0};
  box.hi = {nx * alat, ny * alat, nz * alat};
  box.periodic = {true, true, false};  // free z (3-zone EAM residence)
  const double b[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  std::vector<std::array<double, 3>> pos;
  for (int ix = 0; ix < nx; ++ix)
    for (int iy = 0; iy < ny; ++iy)
      for (int iz = 0; iz < nz; ++iz)
        for (auto& bb : b)
          pos.push_back({(ix + bb[0]) * alat, (iy + bb[1]) * alat, (iz + bb[2]) * alat});
  core::AtomSoA<double> a;
  a.resize(int(pos.size()));
  for (int i = 0; i < a.n; ++i) {
    a.x[i] = pos[i][0]; a.y[i] = pos[i][1]; a.z[i] = pos[i][2];
    a.type[i] = 1; a.mass[i] = 26.98;
    a.vx[i] = 0.001 * ((i * 7) % 5 - 2);
    a.vy[i] = 0.001 * ((i * 3) % 5 - 2);
    a.vz[i] = 0.001 * ((i * 11) % 5 - 2);
  }
  return a;
}

potentials::EamSetfl<double> make_setfl() {
  potentials::AnalyticEam<double> m; m.rcut = kRcut; m.finalize();
  return potentials::EamSetfl<double>::from_analytic(m, 4000, 4000, 60.0);
}
potentials::EamSetfl<double> make_setfl_q23() {  // steep beta => Q23.40 (fb=40)
  potentials::AnalyticEam<double> m; m.rcut = kRcut; m.beta = 3.3; m.finalize();
  return potentials::EamSetfl<double>::from_analytic(m, 4000, 4000, 60.0);
}

// PBC WIDE slab (the M8 sanitizer-load fixture — named on purpose): 288 atoms,
// n_zones=5, ~57-58 atoms/zone; the FIRST cross batch concat (~115 atoms) exceeds
// the initial cap_m=64 of the shared grid per-atom arrays BEFORE any compose has
// grown them — the missing-grow silent-OOB (Me5b class) is memcheck-visible here.
core::AtomSoA<double> make_fcc_pbc_wide(core::Box& box) {
  auto a = make_fcc(3, 3, 8, 4.05, box);  // Lx=Ly=12.15, Lz=32.4, width 6.48
  box.periodic[2] = true;
  return a;
}

using SetflPot = potentials::EamPotential<double, potentials::EamSetfl<double>>;

core::ConveyorOptions ring_opts(long steps, int n_zones, int n_nodes, double dt) {
  core::ConveyorOptions o;
  o.steps = steps; o.n_zones = n_zones; o.n_nodes = n_nodes;
  o.auto_step = false; o.dt_initial = dt;
  o.reach_mult = 2; o.symmetric_reach = true;
  return o;
}

::testing::AssertionResult bitwise_eq(const core::AtomSoA<double>& a,
                                      const core::AtomSoA<double>& b) {
  if (a.n != b.n) return ::testing::AssertionFailure() << "size";
  for (int i = 0; i < a.n; ++i)
    if (a.x[i] != b.x[i] || a.y[i] != b.y[i] || a.z[i] != b.z[i] ||
        a.vx[i] != b.vx[i] || a.vy[i] != b.vy[i] || a.vz[i] != b.vz[i])
      return ::testing::AssertionFailure() << "mismatch at atom " << i;
  return ::testing::AssertionSuccess();
}

// --- the CPU capture twin (A1/B-D2 seam; defined in THIS TU — design amendment S2:
// test policies are TU-local, no shared test header exists or is introduced). It is
// the PRODUCTION CpuEamWindowForce behavior + a compose-time log of the gathered
// rho_w raws split per WindowBlocks — exactly the same lane segments, in the same
// [pred][center][succ] member order, the GPU capture snapshots from its device lanes.
template <typename Math>
struct CpuCaptureWinForce {
  const Math* math = nullptr;
  int fb = 44;
  std::shared_ptr<std::vector<tdcu::EamLaneCapture>> log;
  std::shared_ptr<std::mutex> mu;  // z jthreads share the policy => guard the log
  explicit CpuCaptureWinForce(const Math& m)
      : math(&m), fb(m.density_fracbits()),
        log(std::make_shared<std::vector<tdcu::EamLaneCapture>>()),
        mu(std::make_shared<std::mutex>()) {}

  void compute(const double* wx, const double* wy, const double* wz, const long* key, int m_,
               const int* owned, int n_owned, const core::PairGeom& geom, double rho_cap,
               std::vector<core::fixed::ForceAccum>& wFx, std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz, core::fixed::EnergyAccum& pe,
               double& min_r2) const {
    if (fb == 44)
      potentials::eam_window_force<Math, core::fixed::FixedAccum<44>>(
          wx, wy, wz, key, m_, owned, n_owned, *math, geom, rho_cap, wFx, wFy, wFz, pe, min_r2);
    else
      potentials::eam_window_force<Math, core::fixed::FixedAccum<40>>(
          wx, wy, wz, key, m_, owned, n_owned, *math, geom, rho_cap, wFx, wFy, wFz, pe, min_r2);
  }
  static void assert_supported(std::span<const potentials::PassDecl> p) {
    potentials::assert_eam_symmetric_passes(p, "CpuCaptureWinForce");
  }
  struct NodeState {
    long token = 0;
    void begin_pass(long h) { token = h; }
  };
  NodeState make_node_state(int) const { return {}; }
  template <class DA>
  void on_zone_arrival(NodeState&, potentials::EamDonationState<DA>& st, int label,
                       const potentials::ZoneBlockView& blk, const core::PairGeom& geom) const {
    potentials::eam_donate_self<Math, DA>(blk, *math, geom, st.rho[std::size_t(label)]);
  }
  template <class DA>
  void on_edge(NodeState&, potentials::EamDonationState<DA>& st, int la, int lb,
               const potentials::ZoneBlockView& a, const potentials::ZoneBlockView& b,
               const core::PairGeom& geom) const {
    potentials::eam_donate_cross<Math, DA>(a, b, *math, geom, st.rho[std::size_t(la)],
                                           st.rho[std::size_t(lb)]);
  }
  template <class DA>
  void compose(NodeState& ns, const double* wx, const double* wy, const double* wz,
               const long* key, int m_, const potentials::WindowBlocks& wb, const int* owned,
               int n_owned, const core::PairGeom& geom, double rho_cap, const DA* rho_w,
               std::vector<core::fixed::ForceAccum>& wFx, std::vector<core::fixed::ForceAccum>& wFy,
               std::vector<core::fixed::ForceAccum>& wFz, core::fixed::EnergyAccum& pe,
               double& min_r2, int zone_j) const {
    {
      std::lock_guard<std::mutex> lk(*mu);
      tdcu::EamLaneCapture rec;
      rec.pass = ns.token; rec.zone_j = zone_j; rec.nb = wb.nb;
      int off = 0;
      for (int t = 0; t < wb.nb; ++t) {
        rec.label[t] = wb.label[t];
        rec.lane[t].resize(std::size_t(wb.n[t]));
        for (int i = 0; i < wb.n[t]; ++i) rec.lane[t][std::size_t(i)] = rho_w[off + i].raw;
        off += wb.n[t];
      }
      log->push_back(std::move(rec));
    }
    potentials::eam_window_force_from_rho<Math, DA>(wx, wy, wz, key, m_, owned, n_owned, *math,
        geom, rho_cap, rho_w, wFx, wFy, wFz, pe, min_r2, potentials::NullDonationTrace{}, zone_j);
  }
};

// map keyed by (pass, zone_j) — capture ORDER differs across z (nodes interleave),
// but each (pass, zone) is finalized exactly once.
using LaneMap = std::map<std::pair<long, int>, tdcu::EamLaneCapture>;
LaneMap to_map(const std::vector<tdcu::EamLaneCapture>& v) {
  LaneMap m;
  for (const auto& r : v) {
    const bool inserted = m.emplace(std::make_pair(r.pass, r.zone_j), r).second;
    EXPECT_TRUE(inserted) << "duplicate capture (pass=" << r.pass << ", zone=" << r.zone_j << ")";
  }
  return m;
}

// compare two capture maps; returns the number of differing lane ELEMENTS.
long compare_lanes(const LaneMap& cpu, const LaneMap& gpu) {
  EXPECT_EQ(cpu.size(), gpu.size());
  long diffs = 0;
  for (const auto& [k, cr] : cpu) {
    auto it = gpu.find(k);
    if (it == gpu.end()) { ++diffs; continue; }
    const auto& gr = it->second;
    EXPECT_EQ(cr.nb, gr.nb) << "pass=" << k.first << " zone=" << k.second;
    for (int t = 0; t < cr.nb; ++t) {
      EXPECT_EQ(cr.label[t], gr.label[t]) << "pass=" << k.first << " zone=" << k.second;
      if (cr.lane[t].size() != gr.lane[t].size()) { ++diffs; continue; }
      for (std::size_t i = 0; i < cr.lane[t].size(); ++i)
        if (cr.lane[t][i] != gr.lane[t][i]) ++diffs;
    }
  }
  return diffs;
}

// run the GPU ring with a donation-configured policy; returns the result and keeps
// the policy handle (shared state) alive for the observability accessors.
struct GpuRunOut {
  core::ConveyorResult res;
  tdcu::GpuEamWindowForce policy;
};
GpuRunOut run_gpu_donation_ring(core::AtomSoA<double>& a, const core::Box& box,
                                const SetflPot& pot,
                                const potentials::EamSetfl<double>& setfl,
                                const core::ConveyorOptions& o, bool donate, bool capture,
                                int cell_div = 1,
                                tdcu::GpuDonationPoison poison = {}, bool cull = true) {
  tdcu::GpuEamWindowForce wf(setfl, box, cull, cell_div, donate);
  wf.st->capture_lanes = capture;
  wf.st->poison = poison;
  potentials::EamRing<double, potentials::EamSetfl<double>, tdcu::GpuEamWindowForce> ring(
      a, box, pot, o, wf);
  GpuRunOut out{ring.run(), wf};
  return out;
}
}  // namespace

// A0 — KnobOnIsObservational: the knob-ON trajectory + per-pass PE are BITWISE equal
// to knob-OFF (the donation kernels write ONLY NodeStore lanes + d_of_dn; compose
// ignores both — recompute). Non-vacuity: the counters prove donations actually ran.
// KILL (scratch-build, amendment M7): a hook perturbing st->dens_scale by 1 ulp
// (persistent compose-READ state) turns this red; per-call scratch writes are dead
// by construction (overwritten before compose consumes them) — recorded honestly.
TEST(CudaEamDonation, A0KnobOnIsObservational) {
  const auto setfl = make_setfl();
  const SetflPot pot(setfl);
  const double dt = 0.001;
  const long steps = 12;

  struct Cfg { bool pbc; int n_zones; int z; };
  for (const Cfg& c : {Cfg{false, 4, 1}, Cfg{false, 4, 2}, Cfg{false, 4, 3},
                       Cfg{true, 5, 2}, Cfg{true, 5, 3}}) {
    core::Box box;
    auto init = c.pbc ? make_fcc_pbc_wide(box) : make_fcc(2, 2, 6, 4.05, box);
    const auto o = ring_opts(steps, c.n_zones, c.z, dt);

    core::AtomSoA<double> off_a = init;
    auto off = run_gpu_donation_ring(off_a, box, pot, setfl, o, /*donate=*/false, false);
    ASSERT_EQ(int(off.res.halt), int(core::Halt::None)) << off.res.halt_msg;

    core::AtomSoA<double> on_a = init;
    auto on = run_gpu_donation_ring(on_a, box, pot, setfl, o, /*donate=*/true, false);
    ASSERT_EQ(int(on.res.halt), int(core::Halt::None)) << on.res.halt_msg;

    EXPECT_TRUE(bitwise_eq(off_a, on_a)) << "pbc=" << c.pbc << " z=" << c.z;
    for (long s = 0; s < steps; ++s)
      EXPECT_EQ(off.res.stats[std::size_t(s)].pe, on.res.stats[std::size_t(s)].pe)
          << "per-pass PE, step " << s;
    EXPECT_GT(on.policy.donation_self_batches(), 0u);
    EXPECT_GT(on.policy.donation_cross_batches(), 0u);
    EXPECT_EQ(off.policy.donation_self_batches(), 0u);  // knob-off: hooks never execute
    // A6's NEGATIVE band (acceptance wf_1c8cf3af): a LEGAL knob-ON run leaves the
    // donation overflow flag clear — a born-set/spurious flag (the false-positive
    // fault class) is red HERE; A6 itself proves only the positive plumbing.
    EXPECT_EQ(on.policy.donation_overflow(), 0);
  }
}

// A1 — DeviceRhoBitwiseVsCpu (the T-1 analog on the live ring): per-(pass, window)
// lane snapshots at compose, GPU device lanes (D2H) vs the CPU-donated rho_w raws,
// raw int64 EXPECT_EQ per element. Kills (in-gate): (i) stencil_override=1 with
// cell_div=3 — the poison is MEANINGLESS at k=1 (AUTO resolves k=1 on these small
// boxes ⇒ override-to-1 is the identity; amendment M5), so the poison leg pins k=3
// (~1 Å cells; override covers ~2 Å < nn 2.86 Å ⇒ genuinely red); (ii) one_sided —
// the B-side write goes to a discard sink ⇒ red. (iii) slot-vs-label is a
// scratch-build ring mutation (see the TU banner).
// ACCEPTANCE MUST-FIX (wf_1c8cf3af): the cull=false legs exercise the PLAIN kernel
// pair (eam_donate_cross_kernel + eam_density_kernel-over-slab) — the acceptance
// reviewer gutted the ENTIRE plain cross kernel and all 8 gates stayed GREEN (the
// 7th recidive of the structurally-dead class); with these legs the same mutation
// is A1-red (kill re-verified after the fix). free n∈{1,2} legs restore the frozen
// design §8.1 fixture row verbatim (n=1 = self-only lane, no cross; n=2 = minimal
// single-edge cross).
TEST(CudaEamDonation, A1DeviceRhoBitwiseVsCpu) {
  const auto setfl = make_setfl();
  const SetflPot pot(setfl);
  const double dt = 0.001;
  const long steps = 12;

  struct Cfg { bool pbc; int n_zones; int z; bool cull; };
  for (const Cfg& c : {Cfg{false, 4, 1, true}, Cfg{true, 5, 1, true}, Cfg{true, 5, 3, true},
                       Cfg{false, 4, 1, false}, Cfg{true, 5, 1, false},
                       Cfg{false, 1, 1, true}, Cfg{false, 2, 1, true}}) {
    core::Box box;
    auto init = c.pbc ? make_fcc_pbc_wide(box) : make_fcc(2, 2, 6, 4.05, box);
    const auto o = ring_opts(steps, c.n_zones, c.z, dt);

    // CPU twin (production CPU donation + compose-time rho_w log)
    core::AtomSoA<double> ca = init;
    CpuCaptureWinForce<potentials::EamSetfl<double>> cwf(setfl);
    {
      potentials::EamRing<double, potentials::EamSetfl<double>,
                          CpuCaptureWinForce<potentials::EamSetfl<double>>> ring(
          ca, box, pot, ring_opts(steps, c.n_zones, 1, dt), cwf);
      const auto r = ring.run();
      ASSERT_EQ(int(r.halt), int(core::Halt::None)) << r.halt_msg;
    }

    // GPU knob-ON with lane capture
    core::AtomSoA<double> ga = init;
    auto g = run_gpu_donation_ring(ga, box, pot, setfl, o, /*donate=*/true, /*capture=*/true,
                                   /*cell_div=*/1, {}, c.cull);
    ASSERT_EQ(int(g.res.halt), int(core::Halt::None)) << g.res.halt_msg;

    const auto cm = to_map(*cwf.log);
    const auto gm = to_map(g.policy.capture_log());
    ASSERT_GT(cm.size(), 0u);  // the capture seam is not vacuously comparing nothing
    EXPECT_EQ(compare_lanes(cm, gm), 0)
        << "pbc=" << c.pbc << " z=" << c.z << " cull=" << c.cull
        << " device lanes != CPU-donated rho";
  }

  // Kill (i): stencil poison, MEANINGFUL ONLY at cell_div>=2 (pinned k=3 — M5).
  {
    core::Box box;
    auto init = make_fcc(2, 2, 6, 4.05, box);
    const auto o = ring_opts(4, 4, 1, dt);
    core::AtomSoA<double> ca = init;
    CpuCaptureWinForce<potentials::EamSetfl<double>> cwf(setfl);
    {
      potentials::EamRing<double, potentials::EamSetfl<double>,
                          CpuCaptureWinForce<potentials::EamSetfl<double>>> ring(
          ca, box, pot, o, cwf);
      (void)ring.run();
    }
    tdcu::GpuDonationPoison p; p.stencil_override = 1;
    core::AtomSoA<double> ga = init;
    auto g = run_gpu_donation_ring(ga, box, pot, setfl, o, true, true, /*cell_div=*/3, p);
    EXPECT_GT(compare_lanes(to_map(*cwf.log), to_map(g.policy.capture_log())), 0)
        << "stencil_override=1 at k=3 did NOT diverge — the A1 teeth are dead";
  }
  // Kill (ii): one_sided — the cross batch's B-side lane writes are discarded.
  {
    core::Box box;
    auto init = make_fcc(2, 2, 6, 4.05, box);
    const auto o = ring_opts(4, 4, 1, dt);
    core::AtomSoA<double> ca = init;
    CpuCaptureWinForce<potentials::EamSetfl<double>> cwf(setfl);
    {
      potentials::EamRing<double, potentials::EamSetfl<double>,
                          CpuCaptureWinForce<potentials::EamSetfl<double>>> ring(
          ca, box, pot, o, cwf);
      (void)ring.run();
    }
    tdcu::GpuDonationPoison p; p.one_sided = true;
    core::AtomSoA<double> ga = init;
    auto g = run_gpu_donation_ring(ga, box, pot, setfl, o, true, true, 1, p);
    EXPECT_GT(compare_lanes(to_map(*cwf.log), to_map(g.policy.capture_log())), 0)
        << "one_sided did NOT diverge — the A1 teeth are dead";
  }
}

// A2 — SeamNonVacuity (the T-2 analog): under pbc the per-pass cross-batch count is
// exactly n (the +1 over the free-z n-1 IS the seam (n-1,0), rotation-proof — the
// arithmetic witness, design OQ7/J11). KILL: skipping the ENTIRE on_edge body for
// the seam (launch AND counter — amendment: a launch-only skip that still counts
// stays green on the arithmetic and red on A1's pbc labels n-1/0) breaks the count.
TEST(CudaEamDonation, A2SeamNonVacuityArithmetic) {
  const auto setfl = make_setfl();
  const SetflPot pot(setfl);
  const long steps = 12;

  {  // pbc n=5: cross == steps*n, self == steps*n
    core::Box box;
    auto init = make_fcc_pbc_wide(box);
    core::AtomSoA<double> a = init;
    auto g = run_gpu_donation_ring(a, box, pot, setfl, ring_opts(steps, 5, 2, 0.001), true, false);
    ASSERT_EQ(int(g.res.halt), int(core::Halt::None)) << g.res.halt_msg;
    EXPECT_EQ(g.policy.donation_cross_batches(), (unsigned long long)(steps * 5));
    EXPECT_EQ(g.policy.donation_self_batches(), (unsigned long long)(steps * 5));
  }
  {  // free n=4: cross == steps*(n-1) — the missing edge is the free boundary
    core::Box box;
    auto init = make_fcc(2, 2, 6, 4.05, box);
    core::AtomSoA<double> a = init;
    auto g = run_gpu_donation_ring(a, box, pot, setfl, ring_opts(steps, 4, 2, 0.001), true, false);
    ASSERT_EQ(int(g.res.halt), int(core::Halt::None)) << g.res.halt_msg;
    EXPECT_EQ(g.policy.donation_cross_batches(), (unsigned long long)(steps * 3));
    EXPECT_EQ(g.policy.donation_self_batches(), (unsigned long long)(steps * 4));
  }
}

// A3 — Fb40DualFormat (the T-12 analog): a steep-beta setfl selects Q23.40 (fb=40);
// the lanes must STILL be bitwise vs CPU (raw int64 in the 40-bit format). The
// in-code tooth is the DA::kScale runtime assert in the hooks (hardwiring kScale
// 44 in the hooks throws / turns this red).
TEST(CudaEamDonation, A3Fb40DualFormatLanesBitwise) {
  const auto setfl = make_setfl_q23();
  ASSERT_EQ(setfl.density_fracbits(), 40);
  const SetflPot pot(setfl);
  core::Box box;
  auto init = make_fcc(2, 2, 6, 4.05, box);
  const auto o = ring_opts(8, 4, 1, 0.001);

  core::AtomSoA<double> ca = init;
  CpuCaptureWinForce<potentials::EamSetfl<double>> cwf(setfl);
  {
    potentials::EamRing<double, potentials::EamSetfl<double>,
                        CpuCaptureWinForce<potentials::EamSetfl<double>>> ring(ca, box, pot, o, cwf);
    const auto r = ring.run();
    ASSERT_EQ(int(r.halt), int(core::Halt::None)) << r.halt_msg;
  }
  core::AtomSoA<double> ga = init;
  auto g = run_gpu_donation_ring(ga, box, pot, setfl, o, true, true);
  ASSERT_EQ(int(g.res.halt), int(core::Halt::None)) << g.res.halt_msg;
  EXPECT_EQ(compare_lanes(to_map(*cwf.log), to_map(g.policy.capture_log())), 0);
}

// A4 — VacuumEmptyZones (D8 on the device path): a free-z box with EMPTY upper
// zones; knob-ON must stay bitwise == knob-OFF, complete without a stamp throw
// (the vacuous batch SETS the stamp), and count the vacuous batches (S4).
// KILLS (scratch-build): removing the n==0 early-return launches a 0-block grid
// (CUDA error under memcheck); removing the stamp-set on the empty zone turns the
// next on_edge into an A7 stale-lane throw.
TEST(CudaEamDonation, A4VacuumEmptyZones) {
  const auto setfl = make_setfl();
  const SetflPot pot(setfl);
  core::Box box;
  auto init = make_fcc(2, 2, 2, 4.05, box);  // atoms in z in [0, 8.1)
  box.hi[2] = 24.3;                          // free-z span => zones 2,3 EMPTY (width 6.075)
  const auto o = ring_opts(8, 4, 2, 0.001);

  core::AtomSoA<double> off_a = init;
  auto off = run_gpu_donation_ring(off_a, box, pot, setfl, o, false, false);
  ASSERT_EQ(int(off.res.halt), int(core::Halt::None)) << off.res.halt_msg;

  core::AtomSoA<double> on_a = init;
  auto on = run_gpu_donation_ring(on_a, box, pot, setfl, o, true, false);
  ASSERT_EQ(int(on.res.halt), int(core::Halt::None)) << on.res.halt_msg;

  EXPECT_TRUE(bitwise_eq(off_a, on_a));
  // S4: vacuous batches COUNT (8 passes * 4 self, * 3 cross — including empty zones)
  EXPECT_EQ(on.policy.donation_self_batches(), 8ull * 4ull);
  EXPECT_EQ(on.policy.donation_cross_batches(), 8ull * 3ull);
}

// A5 — DriftedDonorAcrossZoneBoundary (the Tier-0 G-B fixture): a donor atom on a
// zone-boundary lattice plane drifts ACROSS its nominal slab boundary (excess > 0,
// within the residence guard g) — the whole-box grid bins it at its TRUE cell, so
// the lanes stay bitwise vs CPU. KILL (amendment M4 — the slab-clamp mutation was
// struck as structurally dead by the design verifier's STRUCTURAL argument, not by
// an executed measurement): the drop_outside_nominal_slab poison parks the
// drifted donor in the slab MIRROR only (mimicking a slab-extent CSR losing it —
// the real Tier-0 hazard) => lanes red. In PR-3a the poison is lane-visible only
// (compose recomputes); the trajectory-level witness arrives with the PR-3b flip.
TEST(CudaEamDonation, A5DriftedDonorAcrossZoneBoundary) {
  const auto setfl = make_setfl();
  const SetflPot pot(setfl);
  core::Box box;
  auto init = make_fcc(2, 2, 6, 4.05, box);  // width 6.075, g = 0.0375
  // pick an atom on the z=6.075 plane (zone-1 member at its lower boundary) and
  // drift it DOWN into zone 0: v_z = -10 A/ps, dt = 1 fs => 0.01 A/pass; 3 passes
  // => excess 0.03 < g = 0.0375 (membership guard green the whole run).
  int drift_atom = -1;
  for (int i = 0; i < init.n; ++i)
    if (std::abs(init.z[i] - 6.075) < 1e-9) { drift_atom = i; break; }
  ASSERT_GE(drift_atom, 0);
  init.vz[drift_atom] = -10.0;
  const auto o = ring_opts(3, 4, 1, 0.001);

  // positive: lanes bitwise vs CPU (whole-box binning bins the drifted donor at
  // its true cell — Tier-0 drift-binning discharged by construction)
  core::AtomSoA<double> ca = init;
  CpuCaptureWinForce<potentials::EamSetfl<double>> cwf(setfl);
  {
    potentials::EamRing<double, potentials::EamSetfl<double>,
                        CpuCaptureWinForce<potentials::EamSetfl<double>>> ring(ca, box, pot, o, cwf);
    const auto r = ring.run();
    ASSERT_EQ(int(r.halt), int(core::Halt::None)) << r.halt_msg;
  }
  core::AtomSoA<double> ga = init;
  auto g = run_gpu_donation_ring(ga, box, pot, setfl, o, true, true);
  ASSERT_EQ(int(g.res.halt), int(core::Halt::None)) << g.res.halt_msg;
  EXPECT_EQ(compare_lanes(to_map(*cwf.log), to_map(g.policy.capture_log())), 0)
      << "drifted donor: whole-box binning must keep lanes bitwise";

  // kill: park the out-of-nominal-slab donor in the mirror => its pairs vanish
  tdcu::GpuDonationPoison p; p.drop_outside_nominal_slab = true;
  core::AtomSoA<double> pa = init;
  auto gp = run_gpu_donation_ring(pa, box, pot, setfl, o, true, true, 1, p);
  EXPECT_GT(compare_lanes(to_map(*cwf.log), to_map(gp.policy.capture_log())), 0)
      << "the dropped-drifted-donor poison did NOT diverge — A5 has no teeth";
}

// A6 — OverflowSticky (policy-level unit; respecified TWICE, both honest records):
// (1) ring-level is impossible — a single-quantum-overflow setfl cannot pass the
// ring's CPU t0 zone_eam_pass (design amendment S5); (2) MEASURED while building
// this gate: a single-quantum-overflow setfl cannot even be LOADED — the fb-
// fallback ladder throws "density bound exceeds even Q23.40" (density_fracbits():
// rhoa_floor*kMaxCoord*kSafety < 2^23 required), and for a LOADABLE table the
// single-pair quantum v <= rhoa_floor << 2^(63-fb) by that same guard => the
// donation-kernel bit-1 quantize overflow is UNREACHABLE for legal inputs —
// defense-in-depth, exactly like the frozen E5 density kernel's flag. So this
// tooth checks the PLUMBING (write -> sticky -> accessor) via a test-only spline
// corruption AFTER a legal load (x1e7 on rhoaspl; density_fracbits() keys off the
// stored rhoa_floor, so the fb/dens_scale stay consistent — only the evaluated
// quantum explodes). KILL: removing the sticky d_of_dn write => accessor returns 0.
TEST(CudaEamDonation, A6OverflowStickyFlag) {
  auto setfl = make_setfl();
  for (auto& c : setfl.rhoaspl) c *= 1e7;  // test-only corruption (see banner)
  core::Box box;
  auto init = make_fcc(2, 2, 6, 4.05, box);
  const core::PairGeom geom(box, kRcut);

  tdcu::GpuEamWindowForce wf(setfl, box, /*cull=*/true, /*cell_div=*/1, /*donate=*/true);
  auto ns = wf.make_node_state(4);
  ns.begin_pass(1);

  const double hx[2] = {0.0, 2.0}, hy[2] = {0.0, 0.0}, hz[2] = {1.0, 1.0};
  const long hk[2] = {0, 1};
  potentials::ZoneBlockView blk{hx, hy, hz, hk, 2};
  potentials::EamDonationState<core::fixed::FixedAccum<44>> st44;
  potentials::EamDonationState<core::fixed::FixedAccum<40>> st40;
  st44.rho.assign(4, {}); st44.ledger.assign(4, 0u);
  st40.rho.assign(4, {}); st40.ledger.assign(4, 0u);

  EXPECT_EQ(wf.donation_overflow(), 0);
  // fb dispatch mirror: call with the DA the runtime fb selects (the other throws
  // the kScale assert — itself a tooth, checked below).
  if (setfl.density_fracbits() == 44)
    wf.on_zone_arrival(ns, st44, 0, blk, geom);
  else
    wf.on_zone_arrival(ns, st40, 0, blk, geom);
  cudaDeviceSynchronize();
  EXPECT_NE(wf.donation_overflow() & 1, 0)
      << "single-pair quantum overflow did not stick in d_of_dn";
  // the kScale trap (A3's in-code tooth): the WRONG DA format must throw
  if (setfl.density_fracbits() == 44)
    EXPECT_THROW(wf.on_zone_arrival(ns, st40, 1, blk, geom), std::logic_error);
  else
    EXPECT_THROW(wf.on_zone_arrival(ns, st44, 1, blk, geom), std::logic_error);
}

// A7 — StaleStampThrows (the J6 fence): a cross batch touching a lane NOT stamped
// this pass is a label-keying / missed-arrival bug => deterministic logic_error
// (node_main maps it to Halt::Internal in-ring). KILL: removing the stamp check
// in on_edge makes both EXPECT_THROWs fail.
TEST(CudaEamDonation, A7StaleStampThrows) {
  const auto setfl = make_setfl();
  core::Box box;
  auto init = make_fcc(2, 2, 6, 4.05, box);
  const core::PairGeom geom(box, kRcut);

  tdcu::GpuEamWindowForce wf(setfl, box, true, 1, /*donate=*/true);
  auto ns = wf.make_node_state(4);
  ns.begin_pass(1);

  const double hx[2] = {0.0, 2.0}, hy[2] = {0.0, 0.0}, hz[2] = {1.0, 1.0};
  const long hk[2] = {0, 1};
  potentials::ZoneBlockView blk{hx, hy, hz, hk, 2};
  potentials::EamDonationState<core::fixed::FixedAccum<44>> st;
  st.rho.assign(4, {}); st.ledger.assign(4, 0u);

  // no self at all: both lanes unstamped
  EXPECT_THROW(wf.on_edge(ns, st, 0, 1, blk, blk, geom), std::logic_error);
  // self on label 0 only: lane 1 still unstamped
  wf.on_zone_arrival(ns, st, 0, blk, geom);
  EXPECT_THROW(wf.on_edge(ns, st, 0, 1, blk, blk, geom), std::logic_error);
  // stale PASS: stamp lane 1 in pass 1, then advance to pass 2 without re-arrival
  wf.on_zone_arrival(ns, st, 1, blk, geom);
  ns.begin_pass(2);
  EXPECT_THROW(wf.on_edge(ns, st, 0, 1, blk, blk, geom), std::logic_error);
  cudaDeviceSynchronize();
}
