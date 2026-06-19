cmake_minimum_required(VERSION 3.20)
project(tdmd
  VERSION 0.0.1
  DESCRIPTION "TD-MD Core — Time Decomposition molecular dynamics engine"
  LANGUAGES CXX)
# NOTE: CUDA is intentionally NOT enabled at M0 (CPU walking skeleton, builds on
# any C++20 host). CUDA/HAL come at M3+ behind an option on a GPU machine.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(TDMD_BUILD_TESTS "Build unit tests (Test_0_Step, NVE drift)" ON)
# M2.7 GPU bring-up: devprobe + CUDA smoke test. OFF by default — the CPU build
# and cloud CI must not require a CUDA toolkit. GPU tests carry the `cuda` ctest
# label and are gated locally via scripts/gpu_gate.sh (compute-sanitizer).
option(TDMD_WITH_CUDA "Build CUDA components (tools/devprobe, Test_CUDA_Smoke)" OFF)
# M5a: the MPI ring test (mpirun, ranks share one GPU). Needs TDMD_WITH_CUDA.
option(TDMD_WITH_MPI "Build the M5a MPI ring test (needs CUDA + MPI)" OFF)
# NVTX phase markers in the GPU ring (header-only NVTX3; nsys range profiling)
option(TDMD_WITH_NVTX "Annotate the GPU conveyor with NVTX ranges" ON)

if(TDMD_WITH_CUDA)
  # 120-real: SASS for the RTX 5080 dev box, no PTX (bare "120" would embed PTX
  # that does NOT JIT on datacenter sm_100 — for clusters list archs explicitly).
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 120-real)
  endif()
  # M4: the GPU conveyor headers include the M3.5 host conveyor (std::jthread)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  enable_language(CUDA)
endif()

include(FetchContent)
# Old deps (yaml-cpp 0.8.0) declare a pre-3.5 minimum; allow them under CMake 4.x.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# --- yaml-cpp (config parser) ---
FetchContent_Declare(yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG        0.8.0
  GIT_SHALLOW    TRUE)
set(YAML_CPP_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL       OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

# --- core library (no external deps: pure C++20) ---
add_library(tdmd_core
  src/io/reader_lammps.cpp
  src/io/writer.cpp
  src/io/rescue.cpp)
target_include_directories(tdmd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(tdmd_core PUBLIC
  TDMD_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
# Disable FP contraction (FMA) for cross-compiler determinism — the project's
# determinism invariant (INV-9) needs reproducible FP64; propagated to consumers.
target_compile_options(tdmd_core PUBLIC -ffp-contract=off)
target_compile_options(tdmd_core PRIVATE -Wall -Wextra)

# --- config library (B10: separate target so it is unit-testable; keeps
#     tdmd_core and the golden-data tests YAML-free) ---
add_library(tdmd_config src/io/config.cpp)
target_include_directories(tdmd_config PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tdmd_config PUBLIC yaml-cpp)
target_compile_options(tdmd_config PRIVATE -Wall -Wextra)

# --- engine executable ---
find_package(Threads REQUIRED)   # M4: the CLI ring path runs on std::jthread
add_executable(tdmd src/main.cpp)
target_link_libraries(tdmd PRIVATE tdmd_core tdmd_config Threads::Threads)
target_compile_options(tdmd PRIVATE -Wall -Wextra)

# --- M2.5 Tier-1 occupancy probe (CUDA-free analytical tool) ---
add_executable(occupancy_probe src/probe/occupancy_probe.cpp)
target_include_directories(occupancy_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(occupancy_probe PRIVATE -Wall -Wextra)

# --- M2.7 GPU bring-up: device probe (Tier-2 of the occupancy model) ---
if(TDMD_WITH_CUDA)
  add_executable(devprobe tools/devprobe.cu)

  # M4: conveyor-vs-baseline bench + 1e7 memory probe (manual tool, not CI)
  find_package(Threads REQUIRED)
  add_executable(bench_conveyor tools/bench_conveyor.cu)
  target_link_libraries(bench_conveyor PRIVATE tdmd_core Threads::Threads)
  target_compile_options(bench_conveyor PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

  # production-FMA twin (post-M5a lever): same source, FMA contraction ON.
  # GPU-internal bitwise determinism survives FMA (B1); CPU<->GPU bitwise is
  # a verify-build-only claim. Check with: bench_conveyor_fma --verify-det
  add_executable(bench_conveyor_fma tools/bench_conveyor.cu)
  target_link_libraries(bench_conveyor_fma PRIVATE tdmd_core Threads::Threads)

  if(TDMD_WITH_NVTX)
    foreach(tgt bench_conveyor bench_conveyor_fma)
      target_compile_definitions(${tgt} PRIVATE TDMD_WITH_NVTX)
    endforeach()
  endif()

  # M4-N (perf track): GPU EAM ms/step phase breakdown + hit-rate (measure-first).
  # --fmad=false to keep the EAM kernels on the deterministic_fp64 contract.
  add_executable(bench_eam tools/bench_eam.cu)
  target_link_libraries(bench_eam PRIVATE tdmd_core)
  target_compile_options(bench_eam PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

  # E5c FMA twin: same kernels, FMA contraction ON (the GPU default). On the
  # FP64-1:64 RTX 5080 a fused mul-add is 1 FP64 op vs 2 ⇒ ~2× the FP64
  # throughput of the --fmad=false determinism build. GPU-INTERNAL determinism
  # survives FMA (1-vs-z bitwise, B1 empirical — like bench_conveyor_fma); only
  # the CPU↔GPU bitwise contract is given up. Measures the determinism tax on EAM.
  add_executable(bench_eam_fma tools/bench_eam.cu)
  target_link_libraries(bench_eam_fma PRIVATE tdmd_core)

  # M6 physics (a): NVE energy+momentum drift on a system set (kT/(ns·dof)).
  add_executable(eam_drift tools/eam_drift.cu)
  target_link_libraries(eam_drift PRIVATE tdmd_core)
  target_compile_options(eam_drift PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

  # M6 physics (b): statistical RDF cosine overlap vs frozen LAMMPS g(r).
  add_executable(eam_rdf_stat tools/eam_rdf_stat.cu)
  target_link_libraries(eam_rdf_stat PRIVATE tdmd_core)
  target_compile_options(eam_rdf_stat PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

  # M6 physics (c): two-phase melt-coexistence T_m (NVE) vs LAMMPS reference.
  add_executable(eam_coexist tools/eam_coexist.cu)
  target_link_libraries(eam_coexist PRIVATE tdmd_core)
  target_compile_options(eam_coexist PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

  # E5c device-resident PROBE: does z>1 EAM kernel concurrency help on one GPU?
  add_executable(probe_eam_concurrency tools/probe_eam_concurrency.cu)
  target_link_libraries(probe_eam_concurrency PRIVATE tdmd_core)
  target_compile_options(probe_eam_concurrency PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

  # E5c-ring: throughput of the LIVE streaming GPU EAM ring (does culling translate?).
  # Links Threads — run_eam_ring spawns one std::jthread per node. TDMD_EAM_RING_TIMERS
  # turns on the compile-gated per-phase timers in GpuEamWindowForce::compute() (the
  # DEFAULT engine/test build leaves them OFF ⇒ the proven bitwise hot path is byte-identical).
  add_executable(bench_eam_ring tools/bench_eam_ring.cu)
  target_link_libraries(bench_eam_ring PRIVATE tdmd_core Threads::Threads)
  target_compile_definitions(bench_eam_ring PRIVATE TDMD_EAM_RING_TIMERS)
  target_compile_options(bench_eam_ring PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
endif()

# --- M3 timing evidence: direct vs clustered (manual tool, not a CI gate) ---
add_executable(bench_pair tools/bench_pair.cpp)
target_include_directories(bench_pair PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(bench_pair PRIVATE -Wall -Wextra -O3)

# M4-N: K-cadence sweep (CPU NVE; realized K_eff vs predicted K_pred). No CUDA.
add_executable(bench_kcadence tools/bench_kcadence.cpp)
target_include_directories(bench_kcadence PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(bench_kcadence PRIVATE -Wall -Wextra -O3)
target_link_libraries(bench_kcadence PRIVATE Threads::Threads)

# --- tests (do NOT depend on yaml-cpp; golden-data driven, no LAMMPS) ---
if(TDMD_BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(test_0_step tests/test_0_step.cpp)
  target_link_libraries(test_0_step PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_0_Step COMMAND test_0_step)

  add_executable(test_nve_drift tests/test_nve_drift.cpp)
  target_link_libraries(test_nve_drift PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_NVE_Drift COMMAND test_nve_drift)

  add_executable(test_fsm tests/test_fsm.cpp)          # M1: zone FSM
  target_link_libraries(test_fsm PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_Zone_FSM COMMAND test_fsm)

  add_executable(test_buffer tests/test_buffer.cpp)    # M2: buffer / auto-step
  target_link_libraries(test_buffer PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_Buffer COMMAND test_buffer)

  add_executable(test_occupancy tests/test_occupancy.cpp)  # M2.5: occupancy model
  target_link_libraries(test_occupancy PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_Occupancy COMMAND test_occupancy)

  add_executable(test_thermal tests/test_thermal.cpp)      # M2.6: Maxwell init
  target_link_libraries(test_thermal PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_Thermal COMMAND test_thermal)

  add_executable(test_config tests/test_config.cpp)        # M3/B10: strict config
  target_link_libraries(test_config PRIVATE tdmd_config tdmd_core GTest::gtest_main)
  add_test(NAME Test_Config COMMAND test_config)

  add_executable(test_reader tests/test_reader.cpp)        # M3/B10: reader hardening
  target_link_libraries(test_reader PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_Reader COMMAND test_reader)

  add_executable(test_determinism tests/test_determinism.cpp)  # M3: bitwise net
  target_link_libraries(test_determinism PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_Determinism COMMAND test_determinism)

  add_executable(test_cluster tests/test_cluster.cpp)      # M3: clustering core
  target_link_libraries(test_cluster PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_Cluster COMMAND test_cluster)

  add_executable(test_lj_nist tests/test_lj_nist.cpp)      # M3: LJ vs NIST SRSW
  target_link_libraries(test_lj_nist PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_LJ_NIST COMMAND test_lj_nist)

  add_executable(test_zones tests/test_zones.cpp)          # M3.5: B1 + w-assembly
  target_link_libraries(test_zones PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_Zones COMMAND test_zones)

  add_executable(test_many_body_config                     # M6 PR-E0: descriptor + guards
                 tests/test_many_body_config.cpp)
  target_link_libraries(test_many_body_config PRIVATE tdmd_config tdmd_core
                        GTest::gtest_main)
  add_test(NAME Test_ManyBody_Config COMMAND test_many_body_config)

  add_executable(test_eam tests/test_eam.cpp)              # M6 PR-E1: analytic EAM + FD
  target_link_libraries(test_eam PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_EAM COMMAND test_eam)

  add_executable(test_eam_spline tests/test_eam_spline.cpp)  # M6 PR-E2: setfl spline + LAMMPS
  target_link_libraries(test_eam_spline PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_EAM_Spline COMMAND test_eam_spline)

  add_executable(test_sw tests/test_sw.cpp)               # SW PR-1: angular φ₃ + FD + oracle
  target_link_libraries(test_sw PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_SW COMMAND test_sw)

  add_executable(test_sw_zone tests/test_sw_zone.cpp)     # SW T3: serial zone-SW + 1-vs-z + oracle
  target_link_libraries(test_sw_zone PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_SW_Zone COMMAND test_sw_zone)

  add_executable(test_sw_ring tests/test_sw_ring.cpp)     # SW T3b: threaded ring + global-key witness
  target_link_libraries(test_sw_ring PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_SW_Ring COMMAND test_sw_ring)

  add_executable(test_sw_nve tests/test_sw_nve.cpp)       # SW T4: NVE energy/momentum conservation
  target_link_libraries(test_sw_nve PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_SW_NVE COMMAND test_sw_nve)

  add_executable(test_eam_zone tests/test_eam_zone.cpp)    # M6 PR-E3: zone EAM (3-zone residence)
  target_link_libraries(test_eam_zone PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_EAM_Zone COMMAND test_eam_zone)

  add_executable(test_eam_rdf tests/test_eam_rdf.cpp)      # M6 PR-E7: RDF vs frozen LAMMPS
  target_link_libraries(test_eam_rdf PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_EAM_RDF COMMAND test_eam_rdf)

  add_executable(test_metrics tests/test_metrics.cpp)      # M4-N: perf-breakdown math
  target_link_libraries(test_metrics PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_Metrics COMMAND test_metrics)

  add_executable(test_oracle tests/test_oracle.cpp)        # M4-N Gate-02a: physical oracle
  target_link_libraries(test_oracle PRIVATE tdmd_core GTest::gtest_main)
  add_test(NAME Test_Physical_Oracle COMMAND test_oracle)

  add_executable(test_eam_ring tests/test_eam_ring.cpp)    # M6 PR-E3b: threaded EAM ring
  target_link_libraries(test_eam_ring PRIVATE tdmd_core GTest::gtest_main Threads::Threads)
  add_test(NAME Test_EAM_Ring COMMAND test_eam_ring)

  find_package(Threads REQUIRED)                           # M3.5: jthread ring

  add_executable(test_conveyor tests/test_conveyor.cpp)    # M3.5: TimeConveyor-CPU
  target_link_libraries(test_conveyor PRIVATE tdmd_core GTest::gtest_main
                        Threads::Threads)
  add_test(NAME Test_Conveyor COMMAND test_conveyor)
  # deadlock => hang; the timeout turns it into a failure, not a stuck CI
  set_tests_properties(Test_Conveyor PROPERTIES TIMEOUT 300)

  add_executable(test_nve_invariant tests/test_nve_invariant.cpp)  # M3.5: 50k NVE
  target_link_libraries(test_nve_invariant PRIVATE tdmd_core GTest::gtest_main
                        Threads::Threads)
  add_test(NAME Test_NVE_Invariant COMMAND test_nve_invariant)
  set_tests_properties(Test_NVE_Invariant PROPERTIES TIMEOUT 600)

  if(TDMD_WITH_CUDA)                                       # M2.7: toolchain smoke
    add_executable(test_cuda_smoke tests/test_cuda_smoke.cu)
    target_link_libraries(test_cuda_smoke PRIVATE GTest::gtest_main)
    add_test(NAME Test_CUDA_Smoke COMMAND test_cuda_smoke)
    set_tests_properties(Test_CUDA_Smoke PROPERTIES LABELS cuda)

    add_executable(test_cuda_cluster tests/test_cuda_cluster.cu)  # M3 Tier-2
    target_link_libraries(test_cuda_cluster PRIVATE tdmd_core GTest::gtest_main)
    # real regs/thread for the occupancy model (Tier-2): keep -Xptxas -v in the log
    target_compile_options(test_cuda_cluster PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
    add_test(NAME Test_CUDA_Cluster COMMAND test_cuda_cluster)
    set_tests_properties(Test_CUDA_Cluster PROPERTIES LABELS cuda)

    add_executable(test_cuda_zones tests/test_cuda_zones.cu)      # M4: B1 on GPU
    target_link_libraries(test_cuda_zones PRIVATE tdmd_core GTest::gtest_main)
    # verify build (Roadmap M4): bitwise CPU<->GPU parity of the fixed-point
    # accumulators needs FMA contraction OFF on the kernel TU (the CPU side is
    # -ffp-contract=off project-wide). Perf kernels may re-enable FMA later —
    # that is a bench-PR decision, annotated with the precision mode.
    target_compile_options(test_cuda_zones PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false -Xptxas=-v>)
    add_test(NAME Test_CUDA_Zones COMMAND test_cuda_zones)
    set_tests_properties(Test_CUDA_Zones PROPERTIES LABELS cuda)

    # M6 PR-E5: EAM on GPU. --fmad=false is the OQ1 bit-exact precondition; carry
    # it on a dedicated INTERFACE target so any EAM .cu (incl. the E5b ring) makes
    # it structural by LINKING (closes the E2-P2 host-only flag-audit finding).
    add_library(tdmd_eam_cuda_flags INTERFACE)
    target_compile_options(tdmd_eam_cuda_flags INTERFACE
      $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    add_executable(test_cuda_eam tests/test_cuda_eam.cu)
    target_link_libraries(test_cuda_eam PRIVATE tdmd_core GTest::gtest_main
                          tdmd_eam_cuda_flags)
    target_compile_options(test_cuda_eam PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
    add_test(NAME Test_CUDA_EAM COMMAND test_cuda_eam)
    set_tests_properties(Test_CUDA_EAM PROPERTIES LABELS cuda)

    add_executable(test_cuda_eam_ring tests/test_cuda_eam_ring.cu)  # M6 E5b-2/3b: EAM GPU ring (single + streaming z>1)
    target_link_libraries(test_cuda_eam_ring PRIVATE tdmd_core GTest::gtest_main
                          tdmd_eam_cuda_flags Threads::Threads)
    add_test(NAME Test_CUDA_EAM_Ring COMMAND test_cuda_eam_ring)
    set_tests_properties(Test_CUDA_EAM_Ring PROPERTIES LABELS cuda)

    # M6 E5c: cell-list culled EAM kernels. Test A (bitwise cells == all-window)
    # + Test B (the BLOCKING dropped-donor gate vs eam_direct_fp64). --fmad=false
    # via tdmd_eam_cuda_flags (the f_over_r/Horner fuse would break 1 ULP).
    add_executable(test_cuda_eam_cells tests/test_cuda_eam_cells.cu)
    target_link_libraries(test_cuda_eam_cells PRIVATE tdmd_core GTest::gtest_main
                          tdmd_eam_cuda_flags)
    target_compile_options(test_cuda_eam_cells PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
    add_test(NAME Test_CUDA_EAM_Cells COMMAND test_cuda_eam_cells)
    set_tests_properties(Test_CUDA_EAM_Cells PROPERTIES LABELS cuda)

    # M6 physics (b): statistical RDF — time-averaged g(r) over an NVE trajectory
    # started from the LAMMPS-equilibrated config, cosine overlap vs frozen LAMMPS.
    add_executable(test_cuda_eam_rdf tests/test_cuda_eam_rdf.cu)
    target_link_libraries(test_cuda_eam_rdf PRIVATE tdmd_core GTest::gtest_main
                          tdmd_eam_cuda_flags)
    target_compile_definitions(test_cuda_eam_rdf PRIVATE TDMD_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
    add_test(NAME Test_CUDA_EAM_RDF COMMAND test_cuda_eam_rdf)
    set_tests_properties(Test_CUDA_EAM_RDF PROPERTIES LABELS cuda)

    if(TDMD_WITH_MPI)                                            # M5a: MPI ring
      find_package(MPI REQUIRED COMPONENTS CXX)
      add_executable(test_mpi_conveyor tests/test_mpi_conveyor.cu)
      target_link_libraries(test_mpi_conveyor PRIVATE tdmd_core Threads::Threads
                            MPI::MPI_CXX)
      target_compile_options(test_mpi_conveyor PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
      foreach(NP 2 4)
        add_test(NAME Test_MPI_Conveyor_np${NP}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP}
                         $<TARGET_FILE:test_mpi_conveyor>)
        set_tests_properties(Test_MPI_Conveyor_np${NP}
                             PROPERTIES LABELS "cuda;mpi" TIMEOUT 300)
      endforeach()
    endif()

    add_executable(test_cuda_conveyor tests/test_cuda_conveyor.cu)  # M4: GPU ring
    if(TDMD_WITH_NVTX)
      target_compile_definitions(test_cuda_conveyor PRIVATE TDMD_WITH_NVTX)
      if(TDMD_WITH_MPI)
        target_compile_definitions(test_mpi_conveyor PRIVATE TDMD_WITH_NVTX)
      endif()
    endif()
    target_link_libraries(test_cuda_conveyor PRIVATE tdmd_core GTest::gtest_main
                          Threads::Threads)
    target_compile_options(test_cuda_conveyor PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false -Xptxas=-v>)
    add_test(NAME Test_CUDA_Conveyor COMMAND test_cuda_conveyor)
    set_tests_properties(Test_CUDA_Conveyor PROPERTIES LABELS cuda TIMEOUT 600)
  endif()
endif()
