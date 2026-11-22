cmake_minimum_required(VERSION 3.20)
project(knoop LANGUAGES CXX C VERSION 0.5.0)

# C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  add_compile_options(-march=x86-64-v2 -mtune=haswell)
endif()

# Bundle of compile options for knoop
add_library(knoop_lib INTERFACE)

# Shut up GCC's ABI-change notice
add_compile_options(-Wno-psabi)

# Options:
option(BUILD_LINK_OPENMP "Statically link against OpenMP" ON)
option(BUILD_BENCHMARKS "Build knoop benchmarking suite" OFF)
option(BUILD_TESTS "Build knoop test suite" OFF)
option(BUILD_EXECUTABLE "Build knoop CLI tool" ON)
option(BUILD_SANITIZE "Enable ASan + UBSan" OFF)

# Add debugging options depending on CMAKE_BUILD_TYPE, and only add these to
# knoop_lib (not, e.g., HDF5 or other dependencies – this leads to exponentially
# slower compile times in debug mode)
if (BUILD_SANITIZE)
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
#
# (a) on macOS, help CMake find the correct installation path
# (b) resolve the library name (libomp / libgomp)
# (c) link against the statically built archive instead of dynamically, which
#     requires that we find the static library and check everything exists as
#     expected
#
# There are some known issues with statically linking against OpenMP. One
# example is if we were sharing knoop as a shared library – it might enter an
# environment where OpenMP is being dynamically linked.
#
# Seeing as knoop is a stand-alone binary, this isn't an issue. But in the event
# that someone wants to interact with a compiled binary in a different
# environment, we'll use the BUILD_LINK_OPENMP to allow them to turn this option
# off, in which case OpenMP must be available dynamically.
if (BUILD_LINK_OPENMP)
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
else()
  find_package(OpenMP REQUIRED)
  target_link_libraries(knoop_lib INTERFACE OpenMP::OpenMP_CXX)
endif()

# Fetch dependencies
include(FetchContent)
set(FETCHCONTENT_QUIET False)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Configured the Lua scripting / configuration layer. Lua v5.4.9 is stored under
# external/lua
file(GLOB LUA_SOURCES external/lua/*.c)
list(FILTER LUA_SOURCES EXCLUDE REGEX "(lua|luac)\\.c$") # exclude runtime envs
add_library(lua_static STATIC "${LUA_SOURCES}")
target_include_directories(lua_static INTERFACE external/lua)
set_target_properties(lua_static PROPERTIES C_STANDARD 99)

set(SOL2_LUA_VERSION "5.4" CACHE STRING "" FORCE)
set(SOL2_BUILD_LUA FALSE CACHE BOOL "" FORCE)

FetchContent_Declare(
  sol2
  GIT_REPOSITORY https://github.com/ThePhD/sol2
  GIT_TAG v3.5.0
  USES_TERMINAL_DOWNLOAD TRUE
  DOWNLOAD_EXTRACT_TIMESTAMP ON
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(sol2)
target_link_libraries(knoop_lib INTERFACE sol2::sol2 lua_static)

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
SET(BOOST_VERSION "boost-1.91.0-1")
SET(BOOST_URL_PREFIX "https://github.com/boostorg/boost/releases/download/")

FetchContent_Declare(
  Boost
  URL "${BOOST_URL_PREFIX}/${BOOST_VERSION}/${BOOST_VERSION}-cmake.tar.gz"
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
target_compile_definitions(knoop_lib INTERFACE EIGEN_USE_BLAS EIGEN_USE_LAPACK)

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

target_compile_definitions(knoop_lib INTERFACE HIGHFIVE_USE_EIGEN)
target_link_libraries(knoop_lib INTERFACE HighFive::HighFive)

# tskit – found under external/tskit, requires external/kastore as well
file(GLOB TSKIT_SOURCES external/tskit/*.c external/kastore/kastore.c)

add_library(tskit_lib STATIC "${TSKIT_SOURCES}")
target_include_directories(tskit_lib PUBLIC external external/kastore)
set_target_properties(tskit_lib PROPERTIES C_STANDARD 99)

target_link_libraries(knoop_lib INTERFACE tskit_lib)

# use Google Benchmark
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

# Test suite
if (BUILD_TESTS)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest
    GIT_TAG v1.16.0
    USES_TERMINAL_DOWNLOAD TRUE
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(knoop_tests tests/tests.cc)
  target_link_libraries(knoop_tests PRIVATE knoop_lib GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(knoop_tests)
endif()

# The CLI tool
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
