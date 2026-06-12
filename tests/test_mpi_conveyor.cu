// M5a (labels: cuda, mpi) — the TD ring across MPI ranks on ONE GPU.
// Run under mpirun -np {2,4}; every rank drives one logical ring node
// (z_local=1) on its own CUDA context/stream; zones cross rank boundaries as
// host-staged MPI messages (MpiRingEdge), intra-rank life is unchanged.
//
// The M4 determinism criterion on >= 2 ranks: the MPI ring must be BITWISE
// equal to the single-process GPU conveyor with the same logical node count
// (transport is a bit-exact byte move, so rank partitioning cannot change
// trajectories) — fixed and auto dt, deterministic_fp64 and production_mixed
// (the latter's int32 wire image also rides MPI — the B5 traffic win).
// Each rank recomputes the single-process reference locally (deterministic),
// the final-pass owner compares, the verdict is allreduced.
#include <cuda_runtime.h>
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#include "tdmd/core/conveyor.hpp"
#include "tdmd/core/thermal.hpp"
#include "tdmd/cuda/conveyor_gpu.cuh"
#include "tdmd/mpi/mpi_ring_edge.hpp"
#include "tdmd/io/reader_lammps.hpp"
#include "tdmd/potentials/cutoff.hpp"
#include "tdmd/potentials/morse.hpp"
#include "tdmd/potentials/pair_lj.hpp"

// CUB/libcu++ exposes a global ::cuda namespace, and nvcc-generated host
// stubs reference cuda::std unqualified — `using namespace tdmd` would make
// `cuda` ambiguous there. Targeted aliases instead:
namespace core = tdmd::core;
namespace potentials = tdmd::potentials;
namespace io = tdmd::io;
namespace units = tdmd::units;
namespace tdcu = tdmd::cuda;
namespace mpi = tdmd::mpi;

namespace {

constexpr double kRcut = 4.0;
int g_failures = 0;
int g_rank = 0;

#define MPI_CHECK(cond, msg)                                              \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "[rank %d] FAIL: %s (%s:%d)\n", g_rank, msg,  \
                   __FILE__, __LINE__);                                   \
      ++g_failures;                                                       \
    }                                                                     \
  } while (0)

tdcu::LJDev make_lj64() {
  potentials::LJParams<double> p{0.4, 2.55};
  return {p, potentials::CutoffScheme::make(
                 potentials::Truncation::Shift, kRcut,
                 [&](double r, double& u, double& f) {
                   potentials::pair_lj(r, p, u, f);
                 })};
}
tdcu::MorseDev make_morse() {
  potentials::MorseParams<double> p{0.29614, 1.11892, 3.29692};
  return {p, potentials::CutoffScheme::make(
                 potentials::Truncation::Shift, kRcut,
                 [&](double r, double& u, double& f) {
                   potentials::pair_morse(r, p, u, f);
                 })};
}
tdcu::LJDevF32 make_lj32() {
  potentials::LJParams<float> p{0.4f, 2.55f};
  return {p, potentials::CutoffScheme::make(
                 potentials::Truncation::Shift, kRcut,
                 [&](double r, double& u, double& f) {
                   float uf, ff;
                   potentials::pair_lj(float(r), p, uf, ff);
                   u = double(uf);
                   f = double(ff);
                 })};
}

core::AtomSoA<double> make_fcc(core::Box& box, int cx, int cy, int cz,
                               double a0 = 4.05) {
  core::AtomSoA<double> at;
  at.resize(4 * cx * cy * cz);
  box.lo = {0.0, 0.0, 0.0};
  box.hi = {cx * a0, cy * a0, cz * a0};
  box.periodic = {true, true, false};
  static const double basis[4][3] = {
      {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  int k = 0;
  for (int ix = 0; ix < cx; ++ix)
    for (int iy = 0; iy < cy; ++iy)
      for (int iz = 0; iz < cz; ++iz)
        for (int b = 0; b < 4; ++b, ++k) {
          at.x[k] = (ix + basis[b][0]) * a0;
          at.y[k] = (iy + basis[b][1]) * a0;
          at.z[k] = (iz + basis[b][2]) * a0 + 0.25 * a0;
          at.type[k] = 1;
          at.mass[k] = 26.9815;
        }
  return at;
}

bool bitwise_eq(const core::AtomSoA<double>& a, const core::AtomSoA<double>& b) {
  auto eq = [&](const std::vector<double>& u, const std::vector<double>& v) {
    return std::memcmp(u.data(), v.data(), u.size() * sizeof(double)) == 0;
  };
  return a.n == b.n && eq(a.x, b.x) && eq(a.y, b.y) && eq(a.z, b.z) &&
         eq(a.vx, b.vx) && eq(a.vy, b.vy) && eq(a.vz, b.vz) &&
         eq(a.fx, b.fx) && eq(a.fy, b.fy) && eq(a.fz, b.fz);
}

// One scenario: single-process reference vs the rank-partitioned ring.
// k > 1 = "k шагов на узел" (Гл. 3.4/ур. 51): k logical nodes per rank — a
// zone makes k cheap intra-rank hops per ONE MPI hop (traffic /k), and the
// trajectory must stay bitwise identical (the protocol never depends on the
// node count). k>1 also exercises MPI_THREAD_MULTIPLE for real: the in- and
// out-boundary edges live on different node threads of the same rank.
template <typename PairF>
void run_case(const char* name, const core::AtomSoA<double>& init,
              const core::Box& box, core::ConveyorOptions o, const PairF& pot,
              int rank, int nranks, int k = 1) {
  const int Z = nranks * k;
  // reference (every rank, locally — deterministic and identical)
  o.n_nodes = Z;
  core::AtomSoA<double> ref = init;
  auto rref = tdcu::run_conveyor_gpu(ref, box, kRcut, pot, o);
  MPI_CHECK(rref.halt == core::Halt::None, "reference halted");

  // the wire image size must agree on both sides of every boundary: it is a
  // pure function of (mixed, cap) — recompute cap exactly like the conveyor
  const auto zd = core::ZoneDecomposition::build(
      const_cast<core::AtomSoA<double>&>(init), box, o.n_zones, kRcut);
  int cap = 1;
  for (const auto& m : zd.members) cap = std::max(cap, int(m.size()));
  const std::size_t wire =
      o.mixed_transport
          ? sizeof(double) * cap + 9 * sizeof(int) * std::size_t(cap)
          : 10 * sizeof(double) * std::size_t(cap);

  mpi::MpiRingEdge edge(MPI_COMM_WORLD, rank, nranks, wire,
                        o.n_zones + 2);
  tdcu::RingPart part;
  part.z_global = Z;
  part.node0 = rank * k;
  part.z_local = k;
  part.in = &edge;
  part.out = &edge;
  auto om = o;
  om.n_nodes = k;
  core::AtomSoA<double> mine = init;
  auto rmpi = tdcu::run_conveyor_gpu(mine, box, kRcut, pot, om, part);
  MPI_CHECK(rmpi.halt == core::Halt::None, "mpi ring halted");

  const int final_owner = int(((o.steps - 1) % Z) / k);
  MPI_CHECK(rmpi.has_final == (rank == final_owner), "final ownership");
  if (rmpi.has_final)
    MPI_CHECK(bitwise_eq(ref, mine), "final state != single-process, bitwise");

  // Δt-handoff across MPI: my owned passes must carry the reference dt
  for (long h = 1; h <= o.steps; ++h) {
    if (int(((h - 1) % Z) / k) != rank) continue;
    const auto& a = rmpi.stats[std::size_t(h - 1)];
    const auto& b = rref.stats[std::size_t(h - 1)];
    if (!(a.dt == b.dt && a.pe == b.pe && a.v_max == b.v_max))
      std::fprintf(stderr,
                   "[rank %d] pass %ld: dt %.17g vs %.17g  pe %.17g vs %.17g"
                   "  vmax %.17g vs %.17g\n",
                   rank, h, a.dt, b.dt, a.pe, b.pe, a.v_max, b.v_max);
    MPI_CHECK(a.dt == b.dt && a.pe == b.pe && a.v_max == b.v_max,
              "owned-pass stats mismatch");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
    std::printf("[mpi ring] %-22s np=%d k=%d : ok\n", name, nranks, k);
}

// Host-staging overhead probe (M5a criterion): time the MPI ring vs the
// single-process D2D ring with the SAME logical node count, same kernels.
template <typename PairF>
void run_bench(const core::AtomSoA<double>& init, const core::Box& box,
               core::ConveyorOptions o, const PairF& pot, int rank,
               int nranks, int k) {
  const int Z = nranks * k;
  const auto zd = core::ZoneDecomposition::build(
      const_cast<core::AtomSoA<double>&>(init), box, o.n_zones, kRcut);
  int cap = 1;
  for (const auto& m : zd.members) cap = std::max(cap, int(m.size()));
  const std::size_t wire = 10 * sizeof(double) * std::size_t(cap);

  // difference timing t(W+S) - t(W) (review M5a: single-shot carried a
  // ~10-15% one-time setup bias — t0 force pass + slot-pool allocation)
  auto timed_mpi = [&](long steps) {
    mpi::MpiRingEdge edge(MPI_COMM_WORLD, rank, nranks, wire, o.n_zones + 2);
    tdcu::RingPart part{Z, rank * k, k, &edge, &edge};
    auto om = o;
    om.n_nodes = k;
    om.steps = steps;
    core::AtomSoA<double> mine = init;
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    auto r = tdcu::run_conveyor_gpu(mine, box, kRcut, pot, om, part);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_CHECK(r.halt == core::Halt::None, "bench mpi ring halted");
    return MPI_Wtime() - t0;
  };
  const long W = 5;
  const double t_mpi = timed_mpi(W + o.steps) - timed_mpi(W);

  double t_sp = 0.0;
  if (rank == 0) {  // single-process D2D ring, same Z, GPU otherwise idle
    auto timed_sp = [&](long steps) {
      auto os = o;
      os.n_nodes = Z;
      os.steps = steps;
      core::AtomSoA<double> sp = init;
      const double s0 = MPI_Wtime();
      auto rs = tdcu::run_conveyor_gpu(sp, box, kRcut, pot, os);
      MPI_CHECK(rs.halt == core::Halt::None, "bench single halted");
      return MPI_Wtime() - s0;
    };
    t_sp = timed_sp(W + o.steps) - timed_sp(W);
    std::printf("[mpi bench] N=%d steps=%ld zones=%d np=%d k=%d (Z=%d)\n"
                "  mpi ring (host-staging): %7.3f s -> %.3e atom-steps/s\n"
                "  single-process (D2D)   : %7.3f s -> %.3e atom-steps/s\n"
                "  mpi/single time ratio  : x%.3f  (UPPER bound of staging\n"
                "  cost: np contexts time-slice ONE GPU — see Bench doc)\n",
                init.n, o.steps, o.n_zones, nranks, k, Z, t_mpi,
                double(init.n) * o.steps / t_mpi, t_sp,
                double(init.n) * o.steps / t_sp, t_mpi / t_sp);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace

int main(int argc, char** argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  int rank = 0, nranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  g_rank = rank;
  if (provided < MPI_THREAD_MULTIPLE && rank == 0)
    std::fprintf(stderr, "warning: MPI_THREAD_MULTIPLE not provided (%d)\n",
                 provided);

  core::Box box;
  auto init = make_fcc(box, 2, 2, 6);
  core::thermal::maxwell_init(init, 300.0, 71);

  core::ConveyorOptions o;
  o.n_zones = 4;
  o.auto_step = false;
  o.dt_initial = 0.002;
  o.steps = 60;
  run_case("lj fixed fp64", init, box, o, make_lj64(), rank, nranks);

  auto oa = o;
  oa.steps = 100;
  oa.auto_step = true;
  oa.dt_initial = 0.001;
  oa.ts.C1 = 0.01;
  oa.ts.C3 = 1.0;
  run_case("lj auto fp64", init, box, oa, make_lj64(), rank, nranks);

  auto om = o;
  om.mixed_transport = true;
  run_case("lj fixed mixed", init, box, om, make_lj32(), rank, nranks);

  // PBC along the decomposition axis over MPI: the rotation closure's send
  // order crosses rank boundaries (the partition must be transparent to it)
  core::Box pbox;
  auto pinit = make_fcc(pbox, 2, 2, 8);
  pbox.periodic = {true, true, true};
  core::thermal::maxwell_init(pinit, 300.0, 77);
  run_case("lj fixed pbc-z", pinit, pbox, o, make_lj64(), rank, nranks);
  run_case("lj fixed pbc-z", pinit, pbox, o, make_lj64(), rank, nranks, 2);

  // INV-7 reduced slot pool over MPI: n_zones=16 > S=8 per logical node
  core::Box sb;
  auto sinit = make_fcc(sb, 2, 2, 32);
  core::thermal::maxwell_init(sinit, 300.0, 91);
  auto os = o;
  os.n_zones = 16;
  run_case("lj fixed n16 S8", sinit, sb, os, make_lj64(), rank, nranks, 2);

  // k steps per node (M5a, Гл. 3.4): bitwise vs Z = np*k single-process
  run_case("lj fixed fp64", init, box, o, make_lj64(), rank, nranks, 2);
  run_case("lj auto fp64", init, box, oa, make_lj64(), rank, nranks, 3);
  run_case("lj fixed mixed", init, box, om, make_lj32(), rank, nranks, 2);
  auto oma = oa;
  oma.mixed_transport = true;
  run_case("lj auto mixed", init, box, oma, make_lj32(), rank, nranks, 2);

  // HALT propagation across ranks: a causality halt on the owning rank must
  // poison the ring; every rank reports a halt (origin keeps its kind).
  {
    core::Box hb;
    hb.lo = {0.0, 0.0, 0.0};
    hb.hi = {40.0, 12.0, 12.0};
    hb.periodic = {false, false, false};
    core::AtomSoA<double> ha;
    ha.resize(2);
    ha.x = {17.9, 22.1};
    ha.y = {6.0, 6.0};
    ha.z = {6.0, 6.0};
    ha.vx = {20.0, -20.0};
    ha.type = {1, 1};
    ha.mass = {1.0, 1.0};
    core::ConveyorOptions oh;
    oh.steps = 10;
    oh.n_zones = 1;
    oh.dt_initial = 0.02;
    const std::size_t wire = 10 * sizeof(double) * 2;  // cap = 2 atoms
    mpi::MpiRingEdge edge(MPI_COMM_WORLD, rank, nranks, wire, 3);
    tdcu::RingPart part{nranks, rank, 1, &edge, &edge};
    oh.n_nodes = 1;
    auto r = tdcu::run_conveyor_gpu(ha, hb, kRcut, make_morse(), oh, part);
    MPI_CHECK(r.halt != core::Halt::None, "halt did not reach this rank");
    int causality = (r.halt == core::Halt::Causality) ? 1 : 0, any = 0;
    MPI_Allreduce(&causality, &any, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_CHECK(any >= 1, "no rank reported the original Causality kind");
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
      std::printf("[mpi ring] %-22s np=%d k=1 : ok\n", "halt propagation",
                  nranks);
    // z_local=2: the poisoned rank must also unwind its INTRA node threads
    // (review M5a: this variant deadlocked before the propagation fix)
    mpi::MpiRingEdge edge2(MPI_COMM_WORLD, rank, nranks, wire, 3);
    tdcu::RingPart part2{2 * nranks, 2 * rank, 2, &edge2, &edge2};
    oh.n_nodes = 2;
    core::AtomSoA<double> hb2 = ha;
    auto r2 = tdcu::run_conveyor_gpu(hb2, hb, kRcut, make_morse(), oh, part2);
    MPI_CHECK(r2.halt != core::Halt::None, "k=2 halt did not reach this rank");
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
      std::printf("[mpi ring] %-22s np=%d k=2 : ok\n", "halt propagation",
                  nranks);
  }


  if (argc > 1 && std::string(argv[1]) == "--replica") {
    // §3.6 replica over the MPI ring (M5a criterion: the M4 determinism/NVE
    // tests on >= 2 ranks): Al-72, free boundaries, auto-step, 25 900 steps,
    // k=2 logical nodes per rank — bitwise vs the single-process GPU ring.
    core::Box rb;
    core::AtomSoA<double> r72;
    const std::string root =
#ifdef TDMD_PROJECT_ROOT
        TDMD_PROJECT_ROOT;
#else
        ".";
#endif
    if (!io::read_lammps_data(root + "/reference_data/al_fcc_72.data", r72, rb)) {
      std::fprintf(stderr, "[rank %d] cannot read golden data\n", rank);
      ++g_failures;
    } else {
      rb.periodic = {false, false, false};
      core::thermal::maxwell_init(r72, 300.0, 1);
      core::ConveyorOptions orr;
      orr.steps = 25900;
      orr.n_zones = 2;
      orr.auto_step = true;
      orr.dt_initial = 0.001;
      run_case("replica36 morse auto", r72, rb, orr, make_morse(), rank,
               nranks, 2);
    }
  }

  if (argc > 1 && std::string(argv[1]) == "--bench") {
    const int k = (argc > 2) ? std::atoi(argv[2]) : 1;
    core::Box bb;
    auto big = make_fcc(bb, 14, 14, 14);
    core::thermal::maxwell_init(big, 300.0, 7);
    core::ConveyorOptions ob;
    ob.n_zones = 8;
    ob.dt_initial = 0.002;
    ob.steps = 60;
    run_bench(big, bb, ob, make_lj64(), rank, nranks, k);
  }

  int local = g_failures, total = 0;
  MPI_Allreduce(&local, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf(total == 0 ? "[mpi ring] ALL PASS\n"
                           : "[mpi ring] FAILURES: %d\n",
                total);
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
