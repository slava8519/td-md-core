// M2.7 GPU bring-up: device properties + microbenchmarks for the Tier-1/Tier-2
// occupancy model (M2.5/A2) and the B1 fixed-point decision.
//   ./build-cuda/devprobe > docs/_meta/gpu_baseline_<date>.md   (markdown-ish)
// Measures: FP32/FP64 FMA throughput (the deterministic_fp64-vs-production_mixed
// gap), contended atomicAdd double vs unsigned long long (B1 evidence), D2D
// bandwidth, and the Tier-2 occupancy API on a reference kernel.
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

template <typename T>
__global__ void fma_bench(T* out, int iters) {
  T a = T(1.000001) + T(1e-7) * threadIdx.x;
  T b = T(0.999999);
  T c = T(0.0000001);
  for (int i = 0; i < iters; ++i) { a = a * b + c; b = b * a + c; }
  out[blockIdx.x * blockDim.x + threadIdx.x] = a + b;
}

__global__ void atom_f64(double* acc, int iters) {
  for (int i = 0; i < iters; ++i) atomicAdd(acc, 1e-9);
}
__global__ void atom_i64(unsigned long long* acc, int iters) {
  for (int i = 0; i < iters; ++i) atomicAdd(acc, 1ULL);
}

template <typename Kernel, typename... Args>
static float time_kernel(Kernel k, dim3 g, dim3 b, Args... args) {
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0); cudaEventCreate(&t1);
  k<<<g, b>>>(args...);  // warmup
  cudaEventRecord(t0);
  k<<<g, b>>>(args...);
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0); cudaEventDestroy(t1);
  return ms;
}

int main() {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  // NOTE (CUDA 13): clockRate/memoryClockRate were removed from cudaDeviceProp;
  // use cudaDeviceGetAttribute if ever needed.
  printf("name                  : %s\n", p.name);
  printf("compute capability    : %d.%d\n", p.major, p.minor);
  printf("SMs                   : %d\n", p.multiProcessorCount);
  printf("global mem            : %.1f GiB\n", p.totalGlobalMem / (1024.0*1024*1024));
  printf("mem bus               : %d-bit\n", p.memoryBusWidth);
  printf("L2                    : %.1f MiB\n", p.l2CacheSize / (1024.0*1024));
  printf("shared mem / SM       : %zu KiB\n", p.sharedMemPerMultiprocessor / 1024);
  printf("shared mem / block max: %zu KiB\n", p.sharedMemPerBlockOptin / 1024);
  printf("regs / SM             : %d\n", p.regsPerMultiprocessor);
  printf("max threads / SM      : %d  (warps/SM: %d)\n",
         p.maxThreadsPerMultiProcessor, p.maxThreadsPerMultiProcessor / p.warpSize);
  printf("max blocks / SM       : %d\n", p.maxBlocksPerMultiProcessor);
  printf("warp size             : %d\n", p.warpSize);
  printf("async engines         : %d\n", p.asyncEngineCount);
  printf("unified addressing    : %d, managed: %d\n", p.unifiedAddressing, p.managedMemory);
  printf("cooperative launch    : %d\n", p.cooperativeLaunch);

  // probe flags for the Tier-1 occupancy model (occupancy_probe):
  printf("\noccupancy_probe flags : --sm %d --warps-per-sm %d --blocks-per-sm %d --confirmed \"%s\"\n",
         p.multiProcessorCount, p.maxThreadsPerMultiProcessor / p.warpSize,
         p.maxBlocksPerMultiProcessor, p.name);

  // Tier-2 occupancy API on a reference kernel (the real force kernel lands in M3)
  int blocks128 = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks128, fma_bench<float>, 128, 0));
  printf("occupancy API (ref)   : fma_bench<float>, block=128 -> %d blocks/SM (%d warps, %.0f%%)\n",
         blocks128, blocks128 * 128 / p.warpSize,
         100.0 * blocks128 * 128 / p.maxThreadsPerMultiProcessor);

  const int blocks = p.multiProcessorCount * 8, threads = 256, iters = 200000;
  const double flops = 2.0 /*kernel chains*/ * 2.0 /*fma=2flop*/ * iters * blocks * threads;

  float* o32; double* o64;
  CK(cudaMalloc(&o32, blocks*threads*sizeof(float)));
  CK(cudaMalloc(&o64, blocks*threads*sizeof(double)));
  float ms32 = time_kernel(fma_bench<float>,  dim3(blocks), dim3(threads), o32, iters);
  float ms64 = time_kernel(fma_bench<double>, dim3(blocks), dim3(threads), o64, iters);
  printf("\nFP32 FMA              : %.2f TFLOP/s (%.2f ms)\n", flops/ms32/1e9, ms32);
  printf("FP64 FMA              : %.2f TFLOP/s (%.2f ms)\n", flops/ms64/1e9, ms64);
  printf("FP64:FP32 ratio       : 1:%.1f\n", ms64/ms32);

  double* accD; unsigned long long* accU;
  CK(cudaMalloc(&accD, 8)); CK(cudaMemset(accD, 0, 8));
  CK(cudaMalloc(&accU, 8)); CK(cudaMemset(accU, 0, 8));
  // contended single-address atomics, 1 block (worst-case shape; relative compare
  // is what matters — B1: integer accumulation vs FP64 atomicAdd)
  float msAD = time_kernel(atom_f64, dim3(1), dim3(256), accD, 20000);
  float msAU = time_kernel(atom_i64, dim3(1), dim3(256), accU, 20000);
  printf("atomicAdd f64 (contended): %.2f ms;  atomicAdd u64: %.2f ms;  ratio %.2f\n",
         msAD, msAU, msAD/msAU);

  size_t NB = 1ull << 30;
  void *a, *b;
  CK(cudaMalloc(&a, NB)); CK(cudaMalloc(&b, NB));
  cudaMemcpy(b, a, NB, cudaMemcpyDeviceToDevice);  // warm
  cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
  cudaMemcpy(b, a, NB, cudaMemcpyDeviceToDevice);
  cudaEventRecord(t1); cudaEventSynchronize(t1);
  float msbw; cudaEventElapsedTime(&msbw, t0, t1);
  printf("D2D bandwidth         : %.0f GB/s (copy counts r+w)\n", 2.0*NB/msbw/1e6);
  return 0;
}
