#pragma once
#include <cstdint>
#include <cstdio>

// M4-N — measurement harness (perf track, "measure-first"). A structured
// ms/step phase breakdown + hit-rate for the GPU EAM force path, so the
// neighbour-backend bake-off (ClusterFull vs PersistentVerlet vs TileMask) is
// DATA-driven, not asserted (v2 design §F.4, И-F). Host-only POD; the bench
// (tools/bench_eam.cu) fills the timings from CUDA events and the pair counts.
namespace tdmd::metrics {

struct EamPhaseBreakdown {
  // wall time accumulated over `steps` iterations (ms)
  double density_ms = 0.0;
  double embedding_ms = 0.0;
  double force_ms = 0.0;
  long n_atoms = 0;
  long steps = 0;
  // pair counts (one representative step): examined = the candidate set the
  // kernel walked; real = pairs that passed r2<rcut2. hit_rate = real/examined
  // is the cell-list opportunity (low ⇒ a culled backend wins big).
  long long examined_pairs = 0;
  long long real_pairs = 0;

  double total_ms() const { return density_ms + embedding_ms + force_ms; }
  double ms_per_step() const { return steps > 0 ? total_ms() / double(steps) : 0.0; }
  // atom-steps/s = (atoms · steps) / total_seconds
  double atom_steps_per_s() const {
    const double s = total_ms() / 1e3;
    return s > 0 ? double(n_atoms) * double(steps) / s : 0.0;
  }
  double hit_rate() const {
    return examined_pairs > 0 ? double(real_pairs) / double(examined_pairs) : 0.0;
  }
  // fraction of ms/step in each phase (the "stacked bar")
  double density_frac() const { return total_ms() > 0 ? density_ms / total_ms() : 0.0; }
  double embedding_frac() const { return total_ms() > 0 ? embedding_ms / total_ms() : 0.0; }
  double force_frac() const { return total_ms() > 0 ? force_ms / total_ms() : 0.0; }

  void report(const char* tag) const {
    std::printf(
        "[%s] N=%ld steps=%ld | ms/step=%.4f (density %.1f%% / embed %.1f%% / "
        "force %.1f%%) | %.3e atom-steps/s | hit-rate %.4f (%lld real / %lld "
        "examined)\n",
        tag, n_atoms, steps, ms_per_step(), 100.0 * density_frac(),
        100.0 * embedding_frac(), 100.0 * force_frac(), atom_steps_per_s(),
        hit_rate(), real_pairs, examined_pairs);
  }
};

}  // namespace tdmd::metrics
