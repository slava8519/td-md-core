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
add_executable(tdmd src/main.cpp)
target_link_libraries(tdmd PRIVATE tdmd_core tdmd_config)
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
endif()

# --- M3 timing evidence: direct vs clustered (manual tool, not a CI gate) ---
add_executable(bench_pair tools/bench_pair.cpp)
target_include_directories(bench_pair PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(bench_pair PRIVATE -Wall -Wextra -O3)

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

    add_executable(test_cuda_conveyor tests/test_cuda_conveyor.cu)  # M4: GPU ring
    target_link_libraries(test_cuda_conveyor PRIVATE tdmd_core GTest::gtest_main
                          Threads::Threads)
    target_compile_options(test_cuda_conveyor PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false -Xptxas=-v>)
    add_test(NAME Test_CUDA_Conveyor COMMAND test_cuda_conveyor)
    set_tests_properties(Test_CUDA_Conveyor PROPERTIES LABELS cuda TIMEOUT 600)
  endif()
endif()
