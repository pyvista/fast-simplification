cmake_minimum_required(VERSION 3.15...3.26)

project(fast_simplification LANGUAGES CXX)

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

# OpenMP is optional: the upstream Fast-Quadric-Mesh-Simplification core does
# not currently emit OpenMP pragmas, but we keep the wiring so the toolchain
# stays enabled if the core adds parallel regions.
find_package(OpenMP)

function(add_fs_module name source)
  nanobind_add_module(
    ${name}
    # Target the stable ABI for Python 3.12+, which reduces the number of
    # binary wheels that must be built. This does nothing on older Python
    # versions.
    STABLE_ABI
    # Conserve space by reusing a shared libnanobind across libraries.
    NB_STATIC
    ${source}
  )

  if(OpenMP_CXX_FOUND)
    target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
  endif()

  # Let nanobind_add_module set the C++ standard (C++17). Only add
  # optimization and warning-suppression flags here.
  if(MSVC)
    target_compile_options(${name} PRIVATE /O2 /w)
  else()
    target_compile_options(${name} PRIVATE -O3 -w)
  endif()

  install(TARGETS ${name} LIBRARY DESTINATION fast_simplification)
endfunction()

add_fs_module(_simplify fast_simplification/_simplify.cpp)
add_fs_module(_replay fast_simplification/_replay.cpp)
