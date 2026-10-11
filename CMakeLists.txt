cmake_minimum_required(VERSION 3.20)
project(knoop LANGUAGES CXX VERSION 0.4.0)

# C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  add_compile_options(-march=x86-64-v2 -mtune=haswell)
endif()

# Register knoop as a header-only library
add_library(knoop_lib INTERFACE)

# Shut up GCC's ABI-change notice
add_compile_options(-Wno-psabi)

# Add debugging options depending on CMAKE_BUILD_TYPE, and
# only add these to knoop_lib (not, e.g., HDF5)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(knoop_lib INTERFACE
    -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(knoop_lib INTERFACE
    -fsanitize=address,undefined -fno-omit-frame-pointer)
endif()

# Public headers for users of the lib
target_include_directories(knoop_lib
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# Require explicit includes on macOS
if (APPLE)
  target_compile_definitions(knoop_lib INTERFACE
      _LIBCPP_REMOVE_TRANSITIVE_INCLUDES)
endif()

# Use Eigen with BLAS/LAPACK
if (APPLE)
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  target_link_libraries(knoop_lib INTERFACE ${ACCELERATE_FRAMEWORK})
  target_compile_definitions(knoop_lib INTERFACE ACCELERATE_NEW_LAPACK)
else()
  set(BLA_STATIC ON)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  target_link_libraries(knoop_lib INTERFACE BLAS::BLAS LAPACK::LAPACK)
endif()

# Statically linking OpenMP, a bit involved since we need to
# (a) on macOS, help CMake find the correct installation path
# (b) resolve the library name (libomp / libgomp)
# (c) link against the statically built archive instead of dynamically, which
#     requires that we find the static library and check everything exists as
#     expected
if (APPLE AND NOT DEFINED OpenMP_ROOT)
  find_path(
    OMP_INCLUDE_HINT omp.h
    PATHS /opt/homebrew/opt/libomp
          /usr/local/opt/libomp
          /opt/local
    PATH_SUFFIXES include
  )
  if (OMP_INCLUDE_HINT)
    get_filename_component(OpenMP_ROOT "${OMP_INCLUDE_HINT}" DIRECTORY)
  endif()
endif()

find_package(OpenMP REQUIRED)

set(static_omp_libs "")
foreach(name IN LISTS OpenMP_CXX_LIB_NAMES)
  get_filename_component(dir "${OpenMP_${name}_LIBRARY}" DIRECTORY)
  get_filename_component(stem "${OpenMP_${name}_LIBRARY}" NAME_WE)
  if(NOT EXISTS "${dir}/${stem}.a")
    message(FATAL_ERROR "static ${stem}.a not found in ${dir}")
  endif()
  list(APPEND static_omp_libs "${dir}/${stem}.a")
endforeach()

set_property(
  TARGET OpenMP::OpenMP_CXX
  PROPERTY INTERFACE_LINK_LIBRARIES "${static_omp_libs}"
)

target_link_libraries(knoop_lib INTERFACE OpenMP::OpenMP_CXX)

# Fetch dependencies
include(FetchContent)
set(FETCHCONTENT_QUIET False)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# HDF5 – use FetchContent instead of relying on user install since e.g.
# Homebrew dynamically links against libszip and libaec which we don't want
# in the prebuilt binaries
FetchContent_Declare(
  HDF5
  GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
  GIT_TAG 2.2.0
  USES_TERMINAL_DOWNLOAD TRUE
  DOWNLOAD_EXTRACT_TIMESTAMP ON
  EXCLUDE_FROM_ALL
)

set(HDF5_ENABLE_SZIP_SUPPORT OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_Z_LIB_SUPPORT ON CACHE BOOL "" FORCE)
set(HDF5_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(HDF5)

target_link_libraries(knoop_lib INTERFACE hdf5-static)

# Boost
FetchContent_Declare(
  Boost
  URL https://github.com/boostorg/boost/releases/download/boost-1.91.0-1/boost-1.91.0-1-cmake.tar.gz
  USES_TERMINAL_DOWNLOAD TRUE
  DOWNLOAD_EXTRACT_TIMESTAMP ON
  DOWNLOAD_NO_EXTRACT FALSE
  GIT_PROGRESS TRUE
)

set(BOOST_INCLUDE_LIBRARIES algorithm math process random)
set(BOOST_ENABLE_CMAKE ON)
FetchContent_MakeAvailable(Boost)

target_link_libraries(knoop_lib INTERFACE
  Boost::math
  Boost::algorithm
  Boost::process
  Boost::random
)

# Eigen
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 5.0.1
  USES_TERMINAL_DOWNLOAD TRUE
  DOWNLOAD_EXTRACT_TIMESTAMP ON
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(Eigen)

target_link_libraries(knoop_lib INTERFACE Eigen3::Eigen)

# HighFive
FetchContent_Declare(
  HighFive
  GIT_REPOSITORY https://github.com/highfive-devs/highfive.git
  GIT_TAG v3.3.0
  USES_TERMINAL_DOWNLOAD TRUE
  DOWNLOAD_EXTRACT_TIMESTAMP ON
  EXCLUDE_FROM_ALL
)

set(HIGHFIVE_FIND_HDF5 OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_USE_EIGEN ON CACHE BOOL "" FORCE)
set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(HighFive)

target_link_libraries(knoop_lib INTERFACE HighFive::HighFive)

# tomlplusplus
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG v3.4.0
    USES_TERMINAL_DOWNLOAD TRUE
    DOWNLOAD_EXTRACT_TIMESTAMP ON
)
FetchContent_MakeAvailable(tomlplusplus)

target_link_libraries(knoop_lib INTERFACE tomlplusplus::tomlplusplus)

target_compile_definitions(knoop_lib INTERFACE EIGEN_USE_BLAS EIGEN_USE_LAPACK)
target_compile_definitions(knoop_lib INTERFACE HIGHFIVE_USE_EIGEN)

# use Google Benchmark
option(BUILD_BENCHMARKS "Build knoop benchmarking suite" OFF)

if (BUILD_BENCHMARKS)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark
    GIT_TAG v1.9.5
    USES_TERMINAL_DOWNLOAD TRUE
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(googlebenchmark)

  add_executable(benchmarks benchmarks/benchmarks.cc)
  target_link_libraries(benchmarks PRIVATE knoop_lib benchmark::benchmark)
endif()

# The CLI tool
option(BUILD_EXECUTABLE "Build knoop CLI tool" ON)

if (BUILD_EXECUTABLE)
  add_executable(knoop src/knoop.cc)
  target_link_libraries(knoop PRIVATE knoop_lib)
  target_compile_definitions(knoop PRIVATE
    knoop_OS="${CMAKE_SYSTEM_NAME}"
    knoop_VERSION="${PROJECT_VERSION}"
    knoop_ARCH="${CMAKE_SYSTEM_PROCESSOR}"
    knoop_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    knoop_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
  )
endif()
