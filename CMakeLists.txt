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

# --- engine executable (config parser pulls yaml-cpp) ---
add_executable(tdmd src/main.cpp src/io/config.cpp)
target_link_libraries(tdmd PRIVATE tdmd_core yaml-cpp)
target_compile_options(tdmd PRIVATE -Wall -Wextra)

# --- M2.5 Tier-1 occupancy probe (CUDA-free analytical tool) ---
add_executable(occupancy_probe src/probe/occupancy_probe.cpp)
target_include_directories(occupancy_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(occupancy_probe PRIVATE -Wall -Wextra)

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
endif()
