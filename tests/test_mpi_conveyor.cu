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

using namespace tdmd;

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

cuda::LJDev make_lj64() {
  potentials::LJParams<double> p{0.4, 2.55};
  return {p, potentials::CutoffScheme::make(
                 potentials::Truncation::Shift, kRcut,
                 [&](double r, double& u, double& f) {
                   potentials::pair_lj(r, p, u, f);
                 })};
}
cuda::MorseDev make_morse() {
  potentials::MorseParams<double> p{0.29614, 1.11892, 3.29692};
  return {p, potentials::CutoffScheme::make(
                 potentials::Truncation::Shift, kRcut,
                 [&](double r, double& u, double& f) {
                   potentials::pair_morse(r, p, u, f);
                 })};
}
cuda::LJDevF32 make_lj32() {
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
         eq(a.vx, b.vx) && eq(a.vy, b.vy) && eq(a.vz, b.vz);
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
  auto rref = cuda::run_conveyor_gpu(ref, box, kRcut, pot, o);
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

  mpi::MpiRingEdge edge(MPI_COMM_WORLD, rank, nranks, wire);
  cuda::RingPart part;
  part.z_global = Z;
  part.node0 = rank * k;
  part.z_local = k;
  part.in = &edge;
  part.out = &edge;
  auto om = o;
  om.n_nodes = k;
  core::AtomSoA<double> mine = init;
  auto rmpi = cuda::run_conveyor_gpu(mine, box, kRcut, pot, om, part);
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

  mpi::MpiRingEdge edge(MPI_COMM_WORLD, rank, nranks, wire);
  cuda::RingPart part{Z, rank * k, k, &edge, &edge};
  auto om = o;
  om.n_nodes = k;
  core::AtomSoA<double> mine = init;
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();
  auto r = cuda::run_conveyor_gpu(mine, box, kRcut, pot, om, part);
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_mpi = MPI_Wtime() - t0;
  MPI_CHECK(r.halt == core::Halt::None, "bench mpi ring halted");

  if (rank == 0) {  // single-process D2D ring, same Z, GPU otherwise idle
    auto os = o;
    os.n_nodes = Z;
    core::AtomSoA<double> sp = init;
    const double s0 = MPI_Wtime();
    auto rs = cuda::run_conveyor_gpu(sp, box, kRcut, pot, os);
    const double t_sp = MPI_Wtime() - s0;
    MPI_CHECK(rs.halt == core::Halt::None, "bench single halted");
    std::printf("[mpi bench] N=%d steps=%ld zones=%d np=%d k=%d (Z=%d)\n"
                "  mpi ring (host-staging): %7.3f s -> %.3e atom-steps/s\n"
                "  single-process (D2D)   : %7.3f s -> %.3e atom-steps/s\n"
                "  host-staging overhead  : x%.3f\n",
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

  // k steps per node (M5a, Гл. 3.4): bitwise vs Z = np*k single-process
  run_case("lj fixed fp64", init, box, o, make_lj64(), rank, nranks, 2);
  run_case("lj auto fp64", init, box, oa, make_lj64(), rank, nranks, 3);
  run_case("lj fixed mixed", init, box, om, make_lj32(), rank, nranks, 2);

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
