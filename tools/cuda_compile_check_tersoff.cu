// tools/cuda_compile_check_tersoff.cu — PR-0a, audit §7 p.2 (CI blind spot), TERSOFF half.
// Split from cuda_compile_check.cu because zone_sw.cuh and zone_tersoff.cuh both define
// tdmd::cuda::kMaxNbr ⇒ the SW and Tersoff GPU headers cannot share a TU (the test suite
// compiles them separately for the same reason). Never runs device code. See the sibling
// TU for the form rationale (member-scoped run() explicit instantiation, not `template class`).
#include <cstdio>

#include "tdmd/cuda/tersoff_window_force_gpu.cuh"  // GpuTersoffWinForce + GpuTersoffRing alias
#include "tdmd/potentials/tersoff_ring.hpp"        // TersoffRing

namespace tp = tdmd::potentials;
namespace tc = tdmd::cuda;

static_assert(tp::WindowForcePolicy<tc::GpuTersoffWinForce<double>>);

template tdmd::core::ConveyorResult tp::TersoffRing<double, tc::GpuTersoffWinForce<double>>::run();

int main() {
  std::puts("tdmd cuda compile surface (tersoff): OK");
  return 0;
}
