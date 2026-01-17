cmake_minimum_required(VERSION 3.15)

# ---------------------------------------------------------------------------
# vcpkg bootstrap
#
# This repo relies on vcpkg (manifest mode) for core dependencies (e.g. SDL3).
# If you configure with plain `cmake -S . -B build` and delete the build
# directory, any previously-cached toolchain path is lost. Auto-detect vcpkg
# via environment variables so a clean build directory still configures.
# ---------------------------------------------------------------------------
set(_NODUS_NEEDS_TOOLCHAIN FALSE)
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(_NODUS_NEEDS_TOOLCHAIN TRUE)
elseif(CMAKE_TOOLCHAIN_FILE STREQUAL "")
  set(_NODUS_NEEDS_TOOLCHAIN TRUE)
elseif(NOT EXISTS "${CMAKE_TOOLCHAIN_FILE}")
  set(_NODUS_NEEDS_TOOLCHAIN TRUE)
endif()

if(_NODUS_NEEDS_TOOLCHAIN)
  if(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "vcpkg toolchain file" FORCE)
  elseif(DEFINED ENV{VCPKG_INSTALLATION_ROOT} AND EXISTS "$ENV{VCPKG_INSTALLATION_ROOT}/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_INSTALLATION_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "vcpkg toolchain file" FORCE)
  else()
    # Try PATH: if vcpkg is on PATH we can infer its root from the executable location.
    find_program(_NODUS_VCPKG_EXE NAMES vcpkg vcpkg.exe)
    if(_NODUS_VCPKG_EXE)
      get_filename_component(_NODUS_VCPKG_ROOT "${_NODUS_VCPKG_EXE}" DIRECTORY)
      if(EXISTS "${_NODUS_VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "${_NODUS_VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "vcpkg toolchain file" FORCE)
      endif()
    endif()

    # Common locations (best-effort; only applied if found).
    if(_NODUS_NEEDS_TOOLCHAIN AND (NOT DEFINED CMAKE_TOOLCHAIN_FILE OR CMAKE_TOOLCHAIN_FILE STREQUAL "" OR NOT EXISTS "${CMAKE_TOOLCHAIN_FILE}"))
      foreach(_NODUS_VCPKG_GUESS IN ITEMS
        "${CMAKE_SOURCE_DIR}/vcpkg"
        "${CMAKE_SOURCE_DIR}/../vcpkg"
        "C:/vcpkg"
        "C:/src/vcpkg"
      )
        if(EXISTS "${_NODUS_VCPKG_GUESS}/scripts/buildsystems/vcpkg.cmake")
          set(CMAKE_TOOLCHAIN_FILE "${_NODUS_VCPKG_GUESS}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "vcpkg toolchain file" FORCE)
          break()
        endif()
      endforeach()
    endif()
  endif()
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/vcpkg.json" AND (NOT DEFINED CMAKE_TOOLCHAIN_FILE OR CMAKE_TOOLCHAIN_FILE STREQUAL "" OR NOT EXISTS "${CMAKE_TOOLCHAIN_FILE}"))
  message(FATAL_ERROR
    "vcpkg manifest detected (vcpkg.json) but vcpkg toolchain was not found.\n"
    "Set VCPKG_ROOT (or VCPKG_INSTALLATION_ROOT) to your vcpkg install, or add vcpkg to PATH, then re-run:\n"
    "  cmake -S . -B build\n"
    "Or configure with an explicit toolchain:\n"
    "  cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake\n"
  )
endif()

if(NOT DEFINED ENV{VCPKG_FEATURE_FLAGS})
  # Helps older vcpkg builds default to manifest behavior.
  set(ENV{VCPKG_FEATURE_FLAGS} "manifests")
endif()

project(canvas_tables LANGUAGES C CXX)

## Dependencies are built in isolation under dev_bundle/deps/*
## The main project only consumes already-installed packages.

# ---------------------------------------------------------------------------
# Project options
# ---------------------------------------------------------------------------
option(NODUS_ENABLE_TORCH "Enable Torch/LibTorch build and Torch-dependent targets" ON)
option(NODUS_DEPS_ONLY "Fetch/populate dependencies/dev_bundle and skip nodus targets" OFF)
option(NODUS_NO_DEPS "Do not fetch external dependencies; require preinstalled packages" OFF)
option(NODUS_DEBUG_PLUGIN_DEPLOY "Enable verbose plugin deployment logging" OFF)
option(NODUS_DEBUG_MOUSE_LISTENER "Enable verbose mouse listener logging" OFF)
option(NODUS_DEBUG_SERIALIZE "Enable verbose serialize/deserialize logging" OFF)

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

# Python/torch probe removed - prefer explicit repo venv or user-specified python.
set(NODUS_PYTHON_EXECUTABLE "C:/Users/alber/AppData/Local/Programs/Python/Python311/python.exe" CACHE FILEPATH "Explicit python executable to use for repo tasks (optional)")
set(NODUS_TORCH_ROOT "C:/Users/alber/AppData/Local/Programs/Python/Python311/Lib/site-packages/torch" CACHE PATH "Root of a Torch install (e.g. Python site-packages/torch or a libtorch unpack)")
set(NODUS_TORCH_PYTHON_INCLUDE_DIRS
  ""
  CACHE STRING
  "Torch include dirs from python site-packages (semicolon-separated); leave empty to auto-derive from NODUS_TORCH_ROOT if possible"
)
set(NODUS_TORCH_VERSION "v2.9.1" CACHE STRING "Torch version to fetch (tag or branch, e.g. v2.4.1 or main)")
set(NODUS_CUDA_TOOLKIT_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4" CACHE PATH "CUDA toolkit root (set when using CUDA-enabled Torch)")
set(NODUS_CUDA_BIN_DIR "" CACHE PATH "CUDA bin dir override (optional; defaults to <root>/bin)")
set(NODUS_CUDA_LIB_DIR "" CACHE PATH "CUDA lib dir override (optional; defaults to <root>/lib/x64 on Windows)")
set(NODUS_CUDA_INCLUDE_DIR "" CACHE PATH "CUDA include dir override (optional; defaults to <root>/include)")

# ---------------------------------------------------------------------------
# Developer bundle paths
# ---------------------------------------------------------------------------
set(DEV_BUNDLE_DIR "${CMAKE_SOURCE_DIR}/dev_bundle")
set(DEV_BUNDLE_BIN "${DEV_BUNDLE_DIR}/bin")
set(DEV_BUNDLE_INCLUDE "${DEV_BUNDLE_DIR}/include")
set(DEV_BUNDLE_LIB "${DEV_BUNDLE_DIR}/lib")

# ---------------------------------------------------------------------------
# Dependency consumption
# ---------------------------------------------------------------------------

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

# SDL3 (vcpkg)
find_package(SDL3 CONFIG REQUIRED)

## SDL3_ttf removed (custom font renderer in progress)

# OpenMP
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/openmp>)
endif()
find_package(OpenMP QUIET)

# Torch
if(NODUS_ENABLE_TORCH)
  # If the cache was left empty, reset CUDA root to the default path (quoted) so CMake sees it.
  if(NODUS_CUDA_TOOLKIT_ROOT STREQUAL "")
    set(NODUS_CUDA_TOOLKIT_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4" CACHE PATH "CUDA toolkit root (set when using CUDA-enabled Torch)" FORCE)
  endif()

  # Surface CUDA toolkit root for CUDA-enabled Torch wheels (fixed default path unless user overrides).
  if(EXISTS "${NODUS_CUDA_TOOLKIT_ROOT}")
    # Modern CMake prefers CUDAToolkit_ROOT; older configs look at CUDA_TOOLKIT_ROOT_DIR / CUDA_PATH / CUDA_NVCC_EXECUTABLE.
    set(CUDAToolkit_ROOT "${NODUS_CUDA_TOOLKIT_ROOT}" CACHE PATH "" FORCE)
    set(CUDA_TOOLKIT_ROOT_DIR "${NODUS_CUDA_TOOLKIT_ROOT}" CACHE PATH "" FORCE)
    set(ENV{CUDA_PATH} "${NODUS_CUDA_TOOLKIT_ROOT}")
    if(EXISTS "${NODUS_CUDA_TOOLKIT_ROOT}/bin/nvcc.exe")
      set(CMAKE_CUDA_COMPILER "${NODUS_CUDA_TOOLKIT_ROOT}/bin/nvcc.exe" CACHE FILEPATH "" FORCE)
      set(CUDA_NVCC_EXECUTABLE "${NODUS_CUDA_TOOLKIT_ROOT}/bin/nvcc.exe" CACHE FILEPATH "" FORCE)
    endif()
    message(STATUS "CUDA toolkit root: ${NODUS_CUDA_TOOLKIT_ROOT}")
  else()
    message(STATUS "CUDA toolkit not found at ${NODUS_CUDA_TOOLKIT_ROOT}; if your Torch wheel is CUDA-enabled, install CUDA there or override NODUS_CUDA_TOOLKIT_ROOT.")
  endif()

  # Prefer a user-specified Torch root (Python wheel or libtorch); fall back to dev_bundle.
  set(_NODUS_TORCH_SEARCH_DIRS "")
  if(NOT NODUS_TORCH_ROOT STREQUAL "")
    list(APPEND _NODUS_TORCH_SEARCH_DIRS
      "${NODUS_TORCH_ROOT}/share/cmake/Torch"
      "${NODUS_TORCH_ROOT}/cmake"
    )
  endif()

  set(_NODUS_TORCH_CONFIG "")
  foreach(_NODUS_TORCH_PACKAGE_DIR IN LISTS _NODUS_TORCH_SEARCH_DIRS)
    if(EXISTS "${_NODUS_TORCH_PACKAGE_DIR}/TorchConfig.cmake")
      set(_NODUS_TORCH_CONFIG "${_NODUS_TORCH_PACKAGE_DIR}/TorchConfig.cmake")
      break()
    endif()
  endforeach()

  set(_NODUS_TORCH_FOUND FALSE)
  if(_NODUS_TORCH_CONFIG)
    get_filename_component(_NODUS_TORCH_PACKAGE_DIR "${_NODUS_TORCH_CONFIG}" DIRECTORY)
    find_package(Torch REQUIRED PATHS "${_NODUS_TORCH_PACKAGE_DIR}" NO_DEFAULT_PATH)
    set(_NODUS_TORCH_FOUND TRUE)
  else()
    find_package(Torch QUIET)
    if(Torch_FOUND)
      set(_NODUS_TORCH_FOUND TRUE)
    endif()
  endif()

  if(NOT _NODUS_TORCH_FOUND)
    message(FATAL_ERROR
      "Torch requested (NODUS_ENABLE_TORCH=ON) but TorchConfig.cmake was not found.\n"
      "Hint: set -DNODUS_TORCH_ROOT=<python site-packages>/torch or point at a libtorch unpack.\n"
      "Searched: ${_NODUS_TORCH_SEARCH_DIRS} and CMake defaults (Torch_DIR/CMAKE_PREFIX_PATH)."
    )
  endif()

  if(Torch_FOUND AND NOT TARGET Torch::Torch)
    if(TARGET torch)
      add_library(Torch::Torch ALIAS torch)
    elseif(DEFINED TORCH_LIBRARIES)
      add_library(Torch::Torch INTERFACE IMPORTED)
      set_target_properties(Torch::Torch PROPERTIES INTERFACE_LINK_LIBRARIES "${TORCH_LIBRARIES}")
      if(DEFINED TORCH_INCLUDE_DIRS)
        set_target_properties(Torch::Torch PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}")
      endif()
    endif()
  endif()
endif()

# Optional torch headers from python site-packages (no linking).
if(NODUS_TORCH_PYTHON_INCLUDE_DIRS STREQUAL "" AND NOT NODUS_TORCH_ROOT STREQUAL "")
  set(_NODUS_TORCH_PYTHON_INCLUDE_DIRS_GUESS
    "${NODUS_TORCH_ROOT}/include;${NODUS_TORCH_ROOT}/include/torch/csrc/api/include;${NODUS_TORCH_ROOT}/include/TH;${NODUS_TORCH_ROOT}/include/THC"
  )
  set(NODUS_TORCH_PYTHON_INCLUDE_DIRS "${_NODUS_TORCH_PYTHON_INCLUDE_DIRS_GUESS}" CACHE STRING
    "Torch include dirs from python site-packages (semicolon-separated); auto-derived from NODUS_TORCH_ROOT when unset"
    FORCE)
endif()

set(_NODUS_TORCH_INCLUDE_DIRS "")
foreach(_NODUS_TORCH_INC_DIR IN LISTS NODUS_TORCH_PYTHON_INCLUDE_DIRS)
  if(EXISTS "${_NODUS_TORCH_INC_DIR}")
    list(APPEND _NODUS_TORCH_INCLUDE_DIRS "${_NODUS_TORCH_INC_DIR}")
  endif()
endforeach()
if(_NODUS_TORCH_INCLUDE_DIRS)
  add_library(nodus_torch_includes INTERFACE)
  target_include_directories(nodus_torch_includes INTERFACE ${_NODUS_TORCH_INCLUDE_DIRS})
endif()

# Optional CUDA toolkit verification (for CUDA-enabled Torch). Only runs when a CUDA path is supplied.
set(_NODUS_CUDA_WANTS_CHECK FALSE)
if(NOT NODUS_CUDA_TOOLKIT_ROOT STREQUAL "")
  set(_NODUS_CUDA_WANTS_CHECK TRUE)
endif()
if(NOT NODUS_CUDA_BIN_DIR STREQUAL "")
  set(_NODUS_CUDA_WANTS_CHECK TRUE)
endif()
if(NOT NODUS_CUDA_LIB_DIR STREQUAL "")
  set(_NODUS_CUDA_WANTS_CHECK TRUE)
endif()
if(NOT NODUS_CUDA_INCLUDE_DIR STREQUAL "")
  set(_NODUS_CUDA_WANTS_CHECK TRUE)
endif()

if(_NODUS_CUDA_WANTS_CHECK)
  set(_NODUS_CUDA_ROOT "${NODUS_CUDA_TOOLKIT_ROOT}")
  if(_NODUS_CUDA_ROOT STREQUAL "" AND NOT NODUS_CUDA_BIN_DIR STREQUAL "")
    get_filename_component(_NODUS_CUDA_ROOT "${NODUS_CUDA_BIN_DIR}/.." ABSOLUTE)
  endif()
  if(_NODUS_CUDA_ROOT STREQUAL "" AND NOT NODUS_CUDA_LIB_DIR STREQUAL "")
    get_filename_component(_NODUS_CUDA_ROOT "${NODUS_CUDA_LIB_DIR}/.." ABSOLUTE)
  endif()
  if(_NODUS_CUDA_ROOT STREQUAL "" AND NOT NODUS_CUDA_INCLUDE_DIR STREQUAL "")
    get_filename_component(_NODUS_CUDA_ROOT "${NODUS_CUDA_INCLUDE_DIR}/.." ABSOLUTE)
  endif()

  if(_NODUS_CUDA_ROOT STREQUAL "")
    message(FATAL_ERROR "CUDA check requested but no CUDA root could be inferred. Set -DNODUS_CUDA_TOOLKIT_ROOT=<path>.")
  endif()

  set(_NODUS_CUDA_BIN "${NODUS_CUDA_BIN_DIR}")
  if(_NODUS_CUDA_BIN STREQUAL "")
    set(_NODUS_CUDA_BIN "${_NODUS_CUDA_ROOT}/bin")
  endif()

  set(_NODUS_CUDA_LIB "${NODUS_CUDA_LIB_DIR}")
  if(_NODUS_CUDA_LIB STREQUAL "")
    if(WIN32)
      set(_NODUS_CUDA_LIB "${_NODUS_CUDA_ROOT}/lib/x64")
    else()
      set(_NODUS_CUDA_LIB "${_NODUS_CUDA_ROOT}/lib64")
    endif()
  endif()

  set(_NODUS_CUDA_INCLUDE "${NODUS_CUDA_INCLUDE_DIR}")
  if(_NODUS_CUDA_INCLUDE STREQUAL "")
    set(_NODUS_CUDA_INCLUDE "${_NODUS_CUDA_ROOT}/include")
  endif()

  if(NOT EXISTS "${_NODUS_CUDA_INCLUDE}/cuda.h")
    message(FATAL_ERROR "cuda.h not found at ${_NODUS_CUDA_INCLUDE}/cuda.h. Check NODUS_CUDA_TOOLKIT_ROOT or NODUS_CUDA_INCLUDE_DIR.")
  endif()

  find_library(_NODUS_CUDA_CUDART_LIBRARY
    NAMES cudart cudart_static cudart64_12 cudart64_11 cudart64_10
    PATHS "${_NODUS_CUDA_LIB}"
    NO_DEFAULT_PATH
  )
  if(NOT _NODUS_CUDA_CUDART_LIBRARY)
    message(FATAL_ERROR "cudart library not found in ${_NODUS_CUDA_LIB}. Set NODUS_CUDA_LIB_DIR or NODUS_CUDA_TOOLKIT_ROOT.")
  endif()

  if(NOT EXISTS "${_NODUS_CUDA_BIN}")
    message(FATAL_ERROR "CUDA bin dir not found: ${_NODUS_CUDA_BIN}.")
  endif()

  message(STATUS "CUDA toolkit check: root=${_NODUS_CUDA_ROOT} bin=${_NODUS_CUDA_BIN} lib=${_NODUS_CUDA_LIB} include=${_NODUS_CUDA_INCLUDE} cudart=${_NODUS_CUDA_CUDART_LIBRARY}")
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

add_library(canvas_tables_static STATIC)

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

# Developer helper: interrogate shaping and outline metadata without running the full canvas.
add_executable(kpath_shaper_tool tools/kpath_shaper_tool.cpp)
target_include_directories(kpath_shaper_tool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kpath_shaper_tool PRIVATE canvas_tables_static)
target_compile_features(kpath_shaper_tool PUBLIC cxx_std_20)

# Demo: compile relational geometry into kpath outlines and rasterize a composed scene to PNG.
add_executable(kpath_relgeo_scene_demo tools/kpath_relgeo_scene_demo.cpp)
target_include_directories(kpath_relgeo_scene_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kpath_relgeo_scene_demo PRIVATE canvas_tables_static)
target_compile_features(kpath_relgeo_scene_demo PUBLIC cxx_std_20)

# Demo: tensor-backed rasterization with live SDL3 output.
add_executable(kpath_raster_tensor_demo tools/kpath_raster_tensor_demo.cpp)
target_include_directories(kpath_raster_tensor_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kpath_raster_tensor_demo PRIVATE canvas_tables_static)
if(TARGET SDL3::SDL3)
  target_link_libraries(kpath_raster_tensor_demo PRIVATE SDL3::SDL3)
else()
  message(FATAL_ERROR "SDL3 is required for kpath_raster_tensor_demo.")
endif()
target_compile_features(kpath_raster_tensor_demo PUBLIC cxx_std_20)

find_package(OpenGL REQUIRED)

if(NODUS_ENABLE_TORCH)
  add_executable(frontend_shell tools/frontend_shell.cpp)
  target_include_directories(frontend_shell PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(frontend_shell PRIVATE
    canvas_tables
    canvas_tables_static
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
find_package(PNG CONFIG REQUIRED)
find_package(Freetype REQUIRED)
find_package(HarfBuzz CONFIG REQUIRED)
        # vcpkg provides a lowercase harfbuzz target; alias the canonical HarfBuzz::HarfBuzz name for compatibility.
        if(TARGET harfbuzz AND NOT TARGET HarfBuzz::HarfBuzz)
          add_library(HarfBuzz::HarfBuzz ALIAS harfbuzz)
        endif()
# vcpkg currently publishes the imported target harfbuzz (lower-case); alias HarfBuzz::HarfBuzz to that target.
if(TARGET harfbuzz AND NOT TARGET HarfBuzz::HarfBuzz)
  add_library(HarfBuzz::HarfBuzz ALIAS harfbuzz)
endif()

target_include_directories(canvas_tables PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(canvas_tables PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(canvas_tables PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(canvas_tables PUBLIC Eigen3::Eigen PNG::PNG Freetype::Freetype HarfBuzz::HarfBuzz)
if(NODUS_ENABLE_TORCH AND TARGET Torch::Torch)
  target_link_libraries(canvas_tables PUBLIC Torch::Torch)
endif()
if(NODUS_ENABLE_TORCH AND TARGET nodus_persisted_torch_includes)
  target_link_libraries(canvas_tables PUBLIC nodus_persisted_torch_includes)
endif()
if(NODUS_ENABLE_TORCH AND TARGET nodus_torch_includes)
  target_link_libraries(canvas_tables PUBLIC nodus_torch_includes)
endif()
if(TARGET SDL3::SDL3)
  target_include_directories(canvas_tables PUBLIC $<TARGET_PROPERTY:SDL3::SDL3,INTERFACE_INCLUDE_DIRECTORIES>)
else()
  message(FATAL_ERROR "SDL3 not available; SDL2 support removed. Install SDL3 or enable network fetch.")
endif()

target_compile_features(canvas_tables PUBLIC cxx_std_20)
target_compile_definitions(canvas_tables PUBLIC
  NODUS_ENABLE_TORCH=$<IF:$<BOOL:${NODUS_ENABLE_TORCH}>,1,0>
  NODUS_DEBUG_PLUGIN_DEPLOY=$<IF:$<BOOL:${NODUS_DEBUG_PLUGIN_DEPLOY}>,1,0>
  NODUS_DEBUG_MOUSE_LISTENER=$<IF:$<BOOL:${NODUS_DEBUG_MOUSE_LISTENER}>,1,0>
  NODUS_DEBUG_SERIALIZE=$<IF:$<BOOL:${NODUS_DEBUG_SERIALIZE}>,1,0>
)

target_include_directories(canvas_tables_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(canvas_tables_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(canvas_tables_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(canvas_tables_static PUBLIC Eigen3::Eigen PNG::PNG Freetype::Freetype HarfBuzz::HarfBuzz)
if(NODUS_ENABLE_TORCH AND TARGET Torch::Torch)
  target_link_libraries(canvas_tables_static PUBLIC Torch::Torch)
endif()
if(NODUS_ENABLE_TORCH AND TARGET nodus_persisted_torch_includes)
  target_link_libraries(canvas_tables_static PUBLIC nodus_persisted_torch_includes)
endif()
if(NODUS_ENABLE_TORCH AND TARGET nodus_torch_includes)
  target_link_libraries(canvas_tables_static PUBLIC nodus_torch_includes)
endif()
if(TARGET SDL3::SDL3)
  target_include_directories(canvas_tables_static PUBLIC $<TARGET_PROPERTY:SDL3::SDL3,INTERFACE_INCLUDE_DIRECTORIES>)
endif()
target_compile_features(canvas_tables_static PUBLIC cxx_std_20)
target_compile_definitions(canvas_tables_static PUBLIC
  NODUS_ENABLE_TORCH=$<IF:$<BOOL:${NODUS_ENABLE_TORCH}>,1,0>
  NODUS_DEBUG_PLUGIN_DEPLOY=$<IF:$<BOOL:${NODUS_DEBUG_PLUGIN_DEPLOY}>,1,0>
  NODUS_DEBUG_MOUSE_LISTENER=$<IF:$<BOOL:${NODUS_DEBUG_MOUSE_LISTENER}>,1,0>
  NODUS_DEBUG_SERIALIZE=$<IF:$<BOOL:${NODUS_DEBUG_SERIALIZE}>,1,0>
)

if(WIN32)
  set_target_properties(canvas_tables PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)
  target_link_options(canvas_tables PRIVATE "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/cmake/canvas_tables.def")
endif()

# Make prepare_dev_bundle depend on the main library so DLLs exist.
if(NOT NODUS_DEPS_ONLY)
  add_dependencies(prepare_dev_bundle canvas_tables)
else()
  message(STATUS "prepare_dev_bundle will not depend on canvas_tables because NODUS_DEPS_ONLY=ON")
endif()

enable_testing()

# Python interpreter for tests/tools.
if(NODUS_PYTHON_EXECUTABLE)
  set(Python3_EXECUTABLE "${NODUS_PYTHON_EXECUTABLE}" CACHE FILEPATH "Python3 interpreter" FORCE)
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Unit tests
add_executable(test_thread_manager
  tests/thread_manager_test.cpp
)
target_link_libraries(test_thread_manager PRIVATE canvas_tables_static)
add_test(NAME thread_manager_test COMMAND test_thread_manager)

add_executable(test_frame_serialization
  tests/frame_serialization_test.cpp
)
target_link_libraries(test_frame_serialization PRIVATE canvas_tables_static)
add_test(NAME frame_serialization_test COMMAND test_frame_serialization)

add_executable(test_edge_backend_coordination
  tests/edge_backend_coordination_test.cpp
)
target_link_libraries(test_edge_backend_coordination PRIVATE canvas_tables_static)
add_test(NAME edge_backend_coordination_test COMMAND test_edge_backend_coordination)

add_executable(test_meta_edge_snapshot_wrapper
  tests/meta_edge_snapshot_wrapper_test.cpp
)
target_link_libraries(test_meta_edge_snapshot_wrapper PRIVATE canvas_tables_static)
add_test(NAME meta_edge_snapshot_wrapper_test COMMAND test_meta_edge_snapshot_wrapper)

add_executable(test_mem_backend_manifest
  tests/mem_backend_manifest_test.cpp
)
target_link_libraries(test_mem_backend_manifest PRIVATE canvas_tables_static)
add_test(NAME mem_backend_manifest_test COMMAND test_mem_backend_manifest)

add_executable(test_tensor_backend_smoke
  tests/tensor_backend_smoke_test.cpp
)
target_link_libraries(test_tensor_backend_smoke PRIVATE canvas_tables_static)
add_test(NAME tensor_backend_smoke_test COMMAND test_tensor_backend_smoke)

add_executable(test_tensor_coo_matrix
  tests/tensor_coo_matrix_test.cpp
)
target_link_libraries(test_tensor_coo_matrix PRIVATE canvas_tables_static)
add_test(NAME tensor_coo_matrix_test COMMAND test_tensor_coo_matrix)

add_executable(test_tensor_registry
  tests/tensor_registry_test.cpp
)
target_link_libraries(test_tensor_registry PRIVATE canvas_tables_static)
add_test(NAME tensor_registry_test COMMAND test_tensor_registry)

add_executable(test_graph_sparse
  tests/graph_sparse_test.cpp
)
target_link_libraries(test_graph_sparse PRIVATE canvas_tables_static)
add_test(NAME graph_sparse_test COMMAND test_graph_sparse)

add_executable(test_graph_ir
  tests/graph_ir_test.cpp
)
target_link_libraries(test_graph_ir PRIVATE canvas_tables_static)
target_compile_features(test_graph_ir PRIVATE cxx_std_20)
add_test(NAME graph_ir_test COMMAND test_graph_ir)

add_executable(test_relgeo_esat_smoke
  tests/relgeo_esat_smoke_test.cpp
)
target_link_libraries(test_relgeo_esat_smoke PRIVATE canvas_tables_static)
target_compile_features(test_relgeo_esat_smoke PRIVATE cxx_std_20)
add_test(NAME relgeo_esat_smoke_test COMMAND test_relgeo_esat_smoke)

add_executable(test_relgeo_stencil_grid_render
  tests/relgeo_stencil_grid_render_test.cpp
)
target_link_libraries(test_relgeo_stencil_grid_render PRIVATE canvas_tables_static)
target_compile_features(test_relgeo_stencil_grid_render PRIVATE cxx_std_20)
add_test(NAME relgeo_stencil_grid_render_test COMMAND test_relgeo_stencil_grid_render)

add_executable(test_graph_sparse_builder
  tests/graph_sparse_builder_test.cpp
)
target_link_libraries(test_graph_sparse_builder PRIVATE canvas_tables_static)
target_compile_features(test_graph_sparse_builder PRIVATE cxx_std_20)
add_test(NAME graph_sparse_builder_test COMMAND test_graph_sparse_builder)

add_executable(test_kpath_pipeline
  tests/kpath_pipeline_test.cpp
)
target_link_libraries(test_kpath_pipeline PRIVATE canvas_tables_static)
target_compile_features(test_kpath_pipeline PRIVATE cxx_std_20)
add_test(NAME kpath_pipeline_test COMMAND test_kpath_pipeline)

add_executable(test_kpath_toolpath_facilities
  tests/kpath_toolpath_facilities_test.cpp
)
target_link_libraries(test_kpath_toolpath_facilities PRIVATE canvas_tables_static)
target_compile_features(test_kpath_toolpath_facilities PRIVATE cxx_std_20)
add_test(NAME kpath_toolpath_facilities_test COMMAND test_kpath_toolpath_facilities)

add_executable(test_kpath_film
  tests/kpath_film_test.cpp
)
target_link_libraries(test_kpath_film PRIVATE canvas_tables_static)
target_compile_features(test_kpath_film PRIVATE cxx_std_20)
add_test(NAME kpath_film_test COMMAND test_kpath_film)

add_executable(test_kpath_api_fill_token
  tests/kpath_api_fill_token_test.cpp
)
target_link_libraries(test_kpath_api_fill_token PRIVATE canvas_tables_static)
target_compile_features(test_kpath_api_fill_token PRIVATE cxx_std_20)
add_test(NAME kpath_api_fill_token_test COMMAND test_kpath_api_fill_token)

add_executable(test_kpath_functional_intersection
  tests/kpath_functional_intersection_test.cpp
)
target_link_libraries(test_kpath_functional_intersection PRIVATE canvas_tables_static)
target_compile_features(test_kpath_functional_intersection PRIVATE cxx_std_20)
add_test(NAME kpath_functional_intersection_test COMMAND test_kpath_functional_intersection)

add_executable(test_kpath_media_raycast
  tests/kpath_media_raycast_test.cpp
)
target_link_libraries(test_kpath_media_raycast PRIVATE canvas_tables_static)
target_compile_features(test_kpath_media_raycast PRIVATE cxx_std_20)
add_test(NAME kpath_media_raycast_test COMMAND test_kpath_media_raycast)

add_executable(test_thread_pool
  tests/thread_pool_test.cpp
)
target_link_libraries(test_thread_pool PRIVATE canvas_tables_static)
add_test(NAME thread_pool_test COMMAND test_thread_pool)

add_executable(test_tensor_compare
  tests/tensor_compare_test.cpp
)
target_link_libraries(test_tensor_compare PRIVATE canvas_tables_static)
add_test(NAME tensor_compare_test COMMAND test_tensor_compare)

add_executable(test_scatter_indices
  tests/scatter_indices_test.cpp
)
target_link_libraries(test_scatter_indices PRIVATE canvas_tables_static)
add_test(NAME scatter_indices_test COMMAND test_scatter_indices)

add_executable(test_dyadic_bins_bench
  tests/dyadic_bins_bench_test.cpp
)
target_link_libraries(test_dyadic_bins_bench PRIVATE canvas_tables_static)
add_test(NAME dyadic_bins_bench_test COMMAND test_dyadic_bins_bench)

add_executable(test_edge_fifo_sparse_delta
  tests/edge_fifo_sparse_delta_test.cpp
)
target_link_libraries(test_edge_fifo_sparse_delta PRIVATE canvas_tables_static)
add_test(NAME edge_fifo_sparse_delta_test COMMAND test_edge_fifo_sparse_delta)

add_executable(test_tensor_ops_lowering
  tests/tensor_ops_lowering_test.cpp
)
target_link_libraries(test_tensor_ops_lowering PRIVATE canvas_tables_static)
add_test(NAME tensor_ops_lowering_test COMMAND test_tensor_ops_lowering)

add_executable(test_tensor_tiling_int8_int16_timing
  tests/tensor_tiling_int8_int16_timing_test.cpp
)
target_link_libraries(test_tensor_tiling_int8_int16_timing PRIVATE canvas_tables_static)
add_test(NAME tensor_tiling_int8_int16_timing_test COMMAND test_tensor_tiling_int8_int16_timing)

add_executable(test_tool_dll
  tests/tool_dll_test.cpp
)
target_link_libraries(test_tool_dll PRIVATE canvas_tables_static)
file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" NODUS_REPO_ROOT_PATH)
file(TO_CMAKE_PATH "${CMAKE_BINARY_DIR}" NODUS_BUILD_DIR_PATH)
target_compile_definitions(test_tool_dll PRIVATE
  NODUS_REPO_ROOT="${NODUS_REPO_ROOT_PATH}"
  NODUS_BUILD_DIR="${NODUS_BUILD_DIR_PATH}"
)
add_test(NAME tool_dll_test COMMAND test_tool_dll)

# Many unit tests use assert() for both validation and required side effects
# (e.g., assert(backend.map(...)) to populate output pointers). In non-Debug
# configurations the standard NDEBUG definition compiles those expressions out,
# which can leave pointers uninitialized and lead to crashes.
#
# Ensure asserts remain enabled for test translation units in non-Debug builds.
set(NODUS_TEST_TARGETS
  test_thread_manager
  test_frame_serialization
  test_edge_backend_coordination
  test_meta_edge_snapshot_wrapper
  test_mem_backend_manifest
  test_tensor_backend_smoke
  test_tensor_coo_matrix
  test_tensor_registry
  test_graph_sparse
  test_thread_pool
  test_tensor_compare
  test_scatter_indices
  test_edge_fifo_sparse_delta
  test_tensor_ops_lowering
  test_tensor_tiling_int8_int16_timing
  test_tool_dll
  test_kpath_toolpath_facilities
  test_kpath_functional_intersection
  test_kpath_media_raycast
  test_relgeo_stencil_grid_render
)

foreach(_NODUS_TEST_TARGET IN LISTS NODUS_TEST_TARGETS)
  if(TARGET ${_NODUS_TEST_TARGET})
    if(MSVC)
      target_compile_options(${_NODUS_TEST_TARGET} PRIVATE $<$<NOT:$<CONFIG:Debug>>:/UNDEBUG>)
    else()
      target_compile_options(${_NODUS_TEST_TARGET} PRIVATE $<$<NOT:$<CONFIG:Debug>>:-UNDEBUG>)
    endif()
  endif()
endforeach()

option(ENABLE_EIGEN_SIM "Enable Eigen-accelerated rope simulation" OFF)
if(ENABLE_EIGEN_SIM)
  target_compile_definitions(canvas_tables PUBLIC EIGEN_SIM)
endif()
# include auto-discovered sources
include(cmake/auto_sources.cmake)

