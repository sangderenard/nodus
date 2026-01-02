cmake_minimum_required(VERSION 3.15)
project(canvas_tables LANGUAGES CXX)

## Dependencies are built in isolation under dev_bundle/deps/*
## The main project only consumes already-installed packages.

# ---------------------------------------------------------------------------
# Project options
# ---------------------------------------------------------------------------
option(NODUS_ENABLE_TORCH "Enable Torch/LibTorch build and Torch-dependent targets" ON)
option(NODUS_DEPS_ONLY "Fetch/populate dependencies/dev_bundle and skip nodus targets" OFF)
option(NODUS_NO_DEPS "Do not fetch external dependencies; require preinstalled packages" OFF)

if(NODUS_NO_DEPS)
  message(STATUS "NODUS_NO_DEPS=ON: main project will not fetch/build any dependencies.")
endif()

if(NODUS_DEPS_ONLY AND NODUS_NO_DEPS)
  message(FATAL_ERROR "NODUS_DEPS_ONLY and NODUS_NO_DEPS are mutually exclusive.")
endif()

# ---------------------------------------------------------------------------
# Dependency pins (reasonable defaults; override via -D...)
# ---------------------------------------------------------------------------
set(NODUS_SDL3_GIT_TAG "release-3.2.4" CACHE STRING "SDL3 git tag")
set(NODUS_SDLTTF_GIT_TAG "release-3.2.2" CACHE STRING "SDL_ttf git tag")
set(NODUS_FREETYPE_GIT_TAG "VER-2-13-2" CACHE STRING "FreeType git tag")

# Python/torch probe removed — prefer explicit repo venv or user-specified python.
set(NODUS_PYTHON_EXECUTABLE "" CACHE FILEPATH "Explicit python executable to use for repo tasks (optional)")
set(NODUS_TORCH_VERSION "main" CACHE STRING "Torch version to fetch (tag or branch, e.g. v2.4.1 or main)")

# ---------------------------------------------------------------------------
# Developer bundle paths
# ---------------------------------------------------------------------------
set(DEV_BUNDLE_DIR "${CMAKE_SOURCE_DIR}/dev_bundle")
set(DEV_BUNDLE_BIN "${DEV_BUNDLE_DIR}/bin")
set(DEV_BUNDLE_INCLUDE "${DEV_BUNDLE_DIR}/include")
set(DEV_BUNDLE_LIB "${DEV_BUNDLE_DIR}/lib")
set(DEV_BUNDLE_LIBTORCH "${DEV_BUNDLE_DIR}/libtorch")
set(DEV_BUNDLE_LIBTORCH_BIN "${DEV_BUNDLE_LIBTORCH}/bin")
set(DEV_BUNDLE_LIBTORCH_LIB "${DEV_BUNDLE_LIBTORCH}/lib")
set(DEV_BUNDLE_LIBTORCH_SHARE "${DEV_BUNDLE_LIBTORCH}/share/cmake/Torch")
set(DEV_BUNDLE_LIBTORCH_INCLUDE "${DEV_BUNDLE_LIBTORCH}/include")

# ---------------------------------------------------------------------------
# Dependency consumption (NO building/fetching here)
#
# Build/install deps in isolation using:
#   dev_bundle/deps/README.md
# ---------------------------------------------------------------------------
list(PREPEND CMAKE_PREFIX_PATH "${DEV_BUNDLE_DIR}")

# NVTX / nvToolsExt (prebuilt in dev_bundle/nvtoolsext)
if(WIN32)
  set(_NVTX_PERSISTENT_ROOT "${CMAKE_SOURCE_DIR}/dev_bundle/nvtoolsext")
  set(_NVTX_PERSISTENT_INCLUDE "${_NVTX_PERSISTENT_ROOT}/include")
  set(_NVTX_PERSISTENT_LIB_X64 "${_NVTX_PERSISTENT_ROOT}/lib/x64")

  set(NVTOOLSEXT_INCLUDE_DIR "${_NVTX_PERSISTENT_INCLUDE}" CACHE PATH "NVTX include dir" FORCE)
  set(NVTOOLSEXT_LIBRARY     "${_NVTX_PERSISTENT_LIB_X64}/nvToolsExt64_1.lib" CACHE FILEPATH "NVTX import library" FORCE)
  set(NVTOOLSEXT_PATH        "${_NVTX_PERSISTENT_ROOT}" CACHE PATH "NVTX prefix (has include/)" FORCE)
  set(ENV{NVTOOLSEXT_PATH} "${_NVTX_PERSISTENT_ROOT}")

  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
    set_target_properties(CUDA::nvToolsExt PROPERTIES
      IMPORTED_LOCATION "${NVTOOLSEXT_LIBRARY}"
      IMPORTED_LOCATION_RELEASE "${NVTOOLSEXT_LIBRARY}"
      IMPORTED_LOCATION_DEBUG "${NVTOOLSEXT_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${NVTOOLSEXT_INCLUDE_DIR}"
    )
  endif()
endif()

# SDL3
find_package(SDL3 CONFIG QUIET)
if(NOT SDL3_FOUND)
  message(FATAL_ERROR
    "SDL3 not found. Build/install it into dev_bundle first using dev_bundle/deps/README.md (SDL3 section).\n"
    "Expected CMake package under: ${DEV_BUNDLE_DIR}"
  )
endif()

# SDL3_ttf (exports SDL3_ttf and/or SDL_TTF::SDL_TTF depending on build)
find_package(SDL3_ttf CONFIG QUIET)
if(NOT SDL3_ttf_FOUND)
  message(FATAL_ERROR
    "SDL3_ttf not found. Build/install it into dev_bundle first using dev_bundle/deps/README.md (SDL3_ttf section).\n"
    "Expected CMake package under: ${DEV_BUNDLE_DIR}"
  )
endif()

# Ensure SDL_TTF::SDL_TTF alias exists for downstream compatibility
if(NOT TARGET SDL_TTF::SDL_TTF)
  if(TARGET SDL3_ttf::SDL3_ttf)
    add_library(SDL_TTF::SDL_TTF ALIAS SDL3_ttf::SDL3_ttf)
  elseif(TARGET SDL3_ttf-shared)
    add_library(SDL_TTF::SDL_TTF ALIAS SDL3_ttf-shared)
  elseif(TARGET SDL3_ttf-static)
    add_library(SDL_TTF::SDL_TTF ALIAS SDL3_ttf-static)
  endif()
endif()

# OpenMP
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/openmp>)
endif()
find_package(OpenMP QUIET)

# Torch
if(NODUS_ENABLE_TORCH)
  set(_NODUS_TORCH_PACKAGE_DIR "${DEV_BUNDLE_LIBTORCH}/share/cmake/Torch")
  if(EXISTS "${_NODUS_TORCH_PACKAGE_DIR}/TorchConfig.cmake")
    find_package(Torch REQUIRED PATHS "${_NODUS_TORCH_PACKAGE_DIR}" NO_DEFAULT_PATH)
  else()
    message(FATAL_ERROR
      "Torch requested (NODUS_ENABLE_TORCH=ON) but no Torch package was found in dev_bundle.\n"
      "Build/stage it first using dev_bundle/deps/README.md (Torch section).\n"
      "Expected: ${_NODUS_TORCH_PACKAGE_DIR}/TorchConfig.cmake"
    )
  endif()
endif()

add_custom_target(prepare_dev_bundle
  COMMENT "Prepare developer bundle (copies built DLLs and headers into ${DEV_BUNDLE_DIR})"
)

add_custom_command(TARGET prepare_dev_bundle PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DEV_BUNDLE_BIN}"
)
# ---------------------------------------------------------------------------
# Windows DLL exports convenience
# ---------------------------------------------------------------------------
if(WIN32)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
if(NODUS_DEPS_ONLY)
  message(STATUS "NODUS_DEPS_ONLY=ON: skipping nodus targets; use prepare_dev_bundle to stage fetched sources/binaries.")
  return()
endif()

add_library(canvas_tables SHARED
  # sources are now gathered from src/ via cmake/auto_sources.cmake
  # placeholder to keep CMake syntax valid; real sources appended by include(cmake/auto_sources.cmake)
)

# Small helper executable to run the module actualizer from the built project
add_executable(modlib_actualize tools/modlib_actualize_main.cpp)
target_include_directories(modlib_actualize PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(modlib_actualize PRIVATE canvas_tables)
target_compile_features(modlib_actualize PUBLIC cxx_std_17)

# (sdlttf_demo removed)

# sdlttf_demo has been removed; use `frontend_shell` and `canvas_tables` for runtime.



# Small helper to build/load modules at runtime for development
add_executable(modloader tools/modloader_main.cpp)
target_include_directories(modloader PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(modloader PRIVATE canvas_tables)
target_compile_features(modloader PUBLIC cxx_std_17)

find_package(OpenGL REQUIRED)

if(NODUS_ENABLE_TORCH)
  add_executable(frontend_shell tools/frontend_shell.cpp)
  target_include_directories(frontend_shell PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(frontend_shell PRIVATE
    canvas_tables
    Torch::Torch
    OpenGL::GL
  )
  if(TARGET nodus_torch_includes)
    target_link_libraries(frontend_shell PRIVATE nodus_torch_includes)
  endif()
  if(TARGET SDL3::SDL3)
    target_link_libraries(frontend_shell PRIVATE SDL3::SDL3)
  else()
    message(FATAL_ERROR "SDL3 is required (SDL2 support removed). Please install SDL3 or enable network fetch to fetch it.")
  endif()
  target_compile_features(frontend_shell PUBLIC cxx_std_17)

  if(TARGET SDL3::SDL3)
    add_custom_command(TARGET frontend_shell POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:SDL3::SDL3>"
        "$<TARGET_FILE_DIR:frontend_shell>/$<TARGET_FILE_NAME:SDL3::SDL3>"
      COMMENT "Copy SDL3 runtime DLL beside frontend_shell"
      VERBATIM
    )
  else()
    message(FATAL_ERROR "SDL3 runtime target not available to copy DLL. Ensure SDL3 is found/built.")
  endif()

  if(DEFINED Torch_DIR)
    get_filename_component(_TorchInstallPrefix "${Torch_DIR}/../../.." ABSOLUTE)
    set(_TorchDllDir "${_TorchInstallPrefix}/lib")
    if(EXISTS "${_TorchDllDir}")
      file(GLOB _TorchDlls "${_TorchDllDir}/*.dll")
      foreach(_TorchDll IN LISTS _TorchDlls)
        get_filename_component(_TorchDllName "${_TorchDll}" NAME)
        add_custom_command(TARGET frontend_shell POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_TorchDll}"
            "$<TARGET_FILE_DIR:frontend_shell>/${_TorchDllName}"
          COMMENT "Copy ${_TorchDllName} beside frontend_shell"
          VERBATIM
        )
      endforeach()
    endif()
  endif()
else()
  message(STATUS "frontend_shell skipped because NODUS_ENABLE_TORCH=OFF")
endif()

find_package(Eigen3 REQUIRED)

target_include_directories(canvas_tables PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(canvas_tables PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(canvas_tables PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(canvas_tables PUBLIC Eigen3::Eigen)
if(NODUS_ENABLE_TORCH AND TARGET Torch::Torch)
  target_link_libraries(canvas_tables PUBLIC Torch::Torch)
endif()
if(NODUS_ENABLE_TORCH AND TARGET nodus_persisted_torch_includes)
  target_link_libraries(canvas_tables PUBLIC nodus_persisted_torch_includes)
endif()
if(NODUS_ENABLE_TORCH AND TARGET nodus_torch_includes)
  target_link_libraries(canvas_tables PUBLIC nodus_torch_includes)
endif()
if(TARGET SDL_TTF::SDL_TTF)
  target_link_libraries(canvas_tables PRIVATE SDL_TTF::SDL_TTF)
endif()

if(TARGET SDL3::SDL3)
  target_include_directories(canvas_tables PUBLIC $<TARGET_PROPERTY:SDL3::SDL3,INTERFACE_INCLUDE_DIRECTORIES>)
else()
  message(FATAL_ERROR "SDL3 not available; SDL2 support removed. Install SDL3 or enable network fetch.")
endif()

if(TARGET SDL_TTF::SDL_TTF)
  target_include_directories(canvas_tables PUBLIC $<TARGET_PROPERTY:SDL_TTF::SDL_TTF,INTERFACE_INCLUDE_DIRECTORIES>)
endif()
target_compile_features(canvas_tables PUBLIC cxx_std_20)
target_compile_definitions(canvas_tables PUBLIC
  NODUS_ENABLE_TORCH=$<IF:$<BOOL:${NODUS_ENABLE_TORCH}>,1,0>
)

# Make prepare_dev_bundle depend on the main library so DLLs exist.
if(NOT NODUS_DEPS_ONLY)
  add_dependencies(prepare_dev_bundle canvas_tables)
else()
  message(STATUS "prepare_dev_bundle will not depend on canvas_tables because NODUS_DEPS_ONLY=ON")
endif()

enable_testing()

# Unit tests
add_executable(test_thread_manager
  tests/thread_manager_test.cpp
)
target_link_libraries(test_thread_manager PRIVATE canvas_tables)
add_test(NAME thread_manager_test COMMAND test_thread_manager)

add_executable(test_frame_serialization
  tests/frame_serialization_test.cpp
)
target_link_libraries(test_frame_serialization PRIVATE canvas_tables)
add_test(NAME frame_serialization_test COMMAND test_frame_serialization)

add_executable(test_edge_backend_coordination
  tests/edge_backend_coordination_test.cpp
)
target_link_libraries(test_edge_backend_coordination PRIVATE canvas_tables)
add_test(NAME edge_backend_coordination_test COMMAND test_edge_backend_coordination)

add_executable(test_meta_edge_snapshot_wrapper
  tests/meta_edge_snapshot_wrapper_test.cpp
)
target_link_libraries(test_meta_edge_snapshot_wrapper PRIVATE canvas_tables)
add_test(NAME meta_edge_snapshot_wrapper_test COMMAND test_meta_edge_snapshot_wrapper)

add_executable(test_mem_backend_manifest
  tests/mem_backend_manifest_test.cpp
)
target_link_libraries(test_mem_backend_manifest PRIVATE canvas_tables)
add_test(NAME mem_backend_manifest_test COMMAND test_mem_backend_manifest)

add_test(NAME tool_dll_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/tool_dll_test.py
)

option(ENABLE_EIGEN_SIM "Enable Eigen-accelerated rope simulation" OFF)
if(ENABLE_EIGEN_SIM)
  target_compile_definitions(canvas_tables PUBLIC EIGEN_SIM)
endif()
# include auto-discovered sources
include(cmake/auto_sources.cmake)

