// M2.7 GPU bring-up smoke test (label: cuda; not a cloud-CI gate).
// Validates the CUDA toolchain end-to-end: compile for the target arch, launch,
// and check that device math matches host math for the Morse pair force —
// FP64 to 1e-12 (deterministic path sanity), FP32 to 1e-5 (production path).
// This is NOT the M3 prototype force kernel: no SoA, no neighbours — toolchain only.
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

namespace {

// Morse pair force magnitude -dU/dr (same formula as tdmd/potentials/morse.hpp;
// duplicated here so host and device run the literally identical expression).
template <typename Real>
__host__ __device__ Real morse_fpair(Real r, Real D, Real alpha, Real r0) {
  const Real ex = exp(-alpha * (r - r0));
  return Real(2) * alpha * D * (ex * ex - ex);
}

template <typename Real>
__global__ void eval_fpair(const Real* r, Real* out, int n,
                           Real D, Real alpha, Real r0) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = morse_fpair(r[i], D, alpha, r0);
}

template <typename Real>
void run_smoke(double tol) {
  // golden Al/Morse parameters (CLAUDE.md / config defaults)
  const Real D = Real(0.29614), alpha = Real(1.11892), r0 = Real(3.29692);

  const int n = 1024;
  std::vector<Real> r(n), gpu(n);
  for (int i = 0; i < n; ++i) r[i] = Real(2.0) + Real(2.5) * i / (n - 1);  // 2.0..4.5 Å

  Real *d_r = nullptr, *d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_r, n * sizeof(Real)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_out, n * sizeof(Real)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_r, r.data(), n * sizeof(Real), cudaMemcpyHostToDevice),
            cudaSuccess);

  eval_fpair<<<(n + 127) / 128, 128>>>(d_r, d_out, n, D, alpha, r0);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(gpu.data(), d_out, n * sizeof(Real), cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(d_r); cudaFree(d_out);

  double max_err = 0.0;
  for (int i = 0; i < n; ++i) {
    const double ref = morse_fpair(double(r[i]), 0.29614, 1.11892, 3.29692);
    max_err = std::max(max_err, std::fabs(double(gpu[i]) - ref));
  }
  std::printf("Test_CUDA_Smoke<%s>: max |device - host(fp64)| = %.3e eV/Å\n",
              sizeof(Real) == 8 ? "double" : "float", max_err);
  EXPECT_LT(max_err, tol);
}

}  // namespace

TEST(CudaSmoke, DeviceIsUsable) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  ASSERT_GE(count, 1);
  cudaDeviceProp p;
  ASSERT_EQ(cudaGetDeviceProperties(&p, 0), cudaSuccess);
  std::printf("Test_CUDA_Smoke: %s, CC %d.%d, %d SMs\n",
              p.name, p.major, p.minor, p.multiProcessorCount);
}

TEST(CudaSmoke, MorsePairForceFp64MatchesHost) { run_smoke<double>(1e-12); }
TEST(CudaSmoke, MorsePairForceFp32MatchesHost) { run_smoke<float>(1e-5); }
