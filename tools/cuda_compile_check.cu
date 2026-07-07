// tools/cuda_compile_check.cu — PR-0a, audit §7 p.2 (CI blind spot): the cloud
// cuda-compile job configures -DTDMD_BUILD_TESTS=OFF, so the angular GPU window-force
// policies + the Gpu*Ring instantiations were compiled ONLY by test TUs — i.e. NEVER
// in the cloud. This TU compiles that surface in the cloud. It NEVER runs device code.
//
// FORM (MUST-FIX from the design verification): a whole-class `template class
// tp::SwRing<...>;` explicit instantiation is statically ILL-FORMED for all four rings —
// it instantiates the default-policy convenience ctor `Ring(...): Ring(..., WinForce(pot.
// x))`, but every GPU policy ctor requires (params, box[, cull, cell_div]) with no single-
// arg form. So we explicitly instantiate the MEMBER `run()` instead: that implicitly
// instantiates the class specialization (⇒ the class-scope WindowForcePolicy static_assert
// fires) and compiles the whole run/orchestration surface WITHOUT touching the ill-formed
// ctor. The 4 concept static_asserts and the free-function eam_gpu_run_singlenode
// explicit instantiation instantiate no ring and are fine as-is.
//
// INCLUDE ORDER (MUST-FIX): eam_conveyor_gpu.cuh MUST precede eam_window_force_gpu.cuh —
// the latter's non-template GpuEamWindowForce::compute does `using namespace eam_sn_detail;`,
// a namespace defined only in eam_conveyor_gpu.cuh (both existing consumers depend on this
// order). eam_window_force_gpu.cuh includes neither eam_conveyor_gpu.cuh nor eam_ring.hpp,
// and sw_window_force_gpu.cuh includes no ring header ⇒ EamRing/SwRing are pulled explicitly.
//
// TU SPLIT (measured): zone_sw.cuh and zone_tersoff.cuh BOTH define `tdmd::cuda::kMaxNbr`
// ⇒ they cannot coexist in one TU (why the test suite compiles SW/Tersoff separately). This
// TU carries {EAM, MEAM, SW} (no symbol clash among them: zone_meam uses kMeamMaxNbr, zone_eam
// its own); Tersoff lives in cuda_compile_check_tersoff.cu. zone_*.cuh stay byte-untouched
// (F-NOOP witnesses) — the split is the correct fix, not renaming the constant.
#include <cstdio>

#include "tdmd/cuda/eam_conveyor_gpu.cuh"       // eam_sn_detail + eam_gpu_run_singlenode
#include "tdmd/cuda/eam_window_force_gpu.cuh"    // GpuEamWindowForce (needs eam_sn_detail above)
#include "tdmd/potentials/eam_ring.hpp"          // EamRing
#include "tdmd/cuda/sw_window_force_gpu.cuh"     // GpuSwWinForce
#include "tdmd/potentials/sw_ring.hpp"           // SwRing (sw policy header pulls no ring header)
#include "tdmd/cuda/meam_window_force_gpu.cuh"   // GpuMeamWinForce + GpuMeamRing alias

namespace tp = tdmd::potentials;
namespace tc = tdmd::cuda;

// Concept teeth on the GPU policies — verified IN THE CLOUD (test TUs are not compiled there).
static_assert(tp::WindowForcePolicy<tc::GpuSwWinForce<double>>);
static_assert(tp::WindowForcePolicy<tc::GpuMeamWinForce<double>>);
static_assert(tp::WindowForcePolicy<tc::GpuEamWindowForce>);
static_assert(tp::DonatingWindowForcePolicy<tc::GpuEamWindowForce>);  // PR-2: donation hooks

// Member-scoped explicit instantiation of run() — implicitly instantiates each ring class
// (firing its static_assert) + compiles the orchestration surface, without the ill-formed ctor.
template tdmd::core::ConveyorResult tp::SwRing<double, tc::GpuSwWinForce<double>>::run();
template tdmd::core::ConveyorResult tp::MeamRing<double, tc::GpuMeamWinForce<double>>::run();
template tdmd::core::ConveyorResult
    tp::EamRing<double, tp::EamSetfl<double>, tc::GpuEamWindowForce>::run();

// Free-function driver (with the PR-0a trailing passes param) — the FIREWALL GAP entry point.
template void tc::eam_gpu_run_singlenode<double>(
    tdmd::core::AtomSoA<double>&, const tdmd::core::Box&, const tdmd::core::ZoneDecomposition&,
    const tp::EamSetfl<double>&, long, double, bool, std::vector<double>*, std::vector<double>*,
    std::vector<double>*, std::span<const tp::PassDecl>);

int main() {
  std::puts("tdmd cuda compile surface (eam+meam+sw): OK");
  return 0;
}
