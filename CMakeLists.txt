cmake_minimum_required(VERSION 3.15)
project(canvas_tables LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Policies (avoid noisy warnings on newer CMake)
# ---------------------------------------------------------------------------
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW) # honor DOWNLOAD_EXTRACT_TIMESTAMP when provided
endif()

# FetchContent_Populate() deprecation warning control on newer CMake.
# We intentionally use Populate for "download-only" archives (no CMakeLists).
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

# ---------------------------------------------------------------------------
# FetchContent persistence (avoid re-downloading across fresh build dirs)
#
# Default: a stable per-user cache folder. Override via -DFETCHCONTENT_BASE_DIR=...
# ---------------------------------------------------------------------------
if(NOT DEFINED FETCHCONTENT_BASE_DIR)
  if(WIN32)
    set(_NODUS_FC_DEFAULT "$ENV{LOCALAPPDATA}/nodus_fetchcontent")
  else()
    set(_NODUS_FC_DEFAULT "$ENV{HOME}/.cache/nodus_fetchcontent")
  endif()
  set(FETCHCONTENT_BASE_DIR "${_NODUS_FC_DEFAULT}" CACHE PATH "Where FetchContent stores downloaded/extracted dependencies")
endif()

include(FetchContent)

# Avoid FetchContent trying to update git deps every configure unless explicitly desired.
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Disable FetchContent updates after first download")

# ---------------------------------------------------------------------------
# Options / overrides for fully-automatic dependency acquisition
# ---------------------------------------------------------------------------
option(NODUS_ALLOW_NETWORK_FETCH "Allow CMake to download dependencies (Torch/SDL2/NVTX) when not found locally" ON)

# If Torch must be downloaded, these can be set to force the exact libtorch ZIP.
# Example:
#   -DNODUS_TORCH_VERSION=2.7.1 -DNODUS_TORCH_CUDA_TAG=cu126
set(NODUS_TORCH_VERSION "" CACHE STRING "PyTorch version to download for LibTorch (e.g. 2.7.1). If empty, tries to derive from Python torch.")
set(NODUS_TORCH_CUDA_TAG "" CACHE STRING "LibTorch CUDA tag (e.g. cu126, cu124, cpu). If empty, tries to derive from Python torch.")
set(NODUS_TORCH_URL "" CACHE STRING "Override LibTorch ZIP URL directly (takes precedence over version/tag).")

# NVTX (nvToolsExt) redist ZIP (Windows)
option(NODUS_ENABLE_NVTX "Fetch Nsight NVTX redistributable (nvToolsExt) so Torch CUDA CMake can find nvToolsExt on CUDA 12.x+" ON)
set(NODUS_NVTX_URL
  "https://developer.download.nvidia.com/compute/cuda/redist/nsight_nvtx/windows-x86_64/nsight_nvtx-windows-x86_64-1.21018621-archive.zip"
  CACHE STRING "Nsight NVTX (nvToolsExt) redist ZIP URL"
)

# ---------------------------------------------------------------------------
# Python interpreter (used for Torch autodetection and tests)
# ---------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter)

message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")

# ---------------------------------------------------------------------------
# Attempt to derive Torch info from Python (prefix path, version, CUDA tag)
# ---------------------------------------------------------------------------
set(PYTORCH_CMAKE_PREFIX "")
set(PYTORCH_VERSION_FROM_PY "")
set(PYTORCH_CUDA_TAG_FROM_PY "")

set(_PYTORCH_PREFIX_SCRIPT "")
string(APPEND _PYTORCH_PREFIX_SCRIPT
  "import sys\n"
  "try:\n"
  "  import torch\n"
  "  prefixes = torch.utils.cmake_prefix_path\n"
  "  if isinstance(prefixes, (list, tuple)):\n"
  "    print('\\n'.join(prefixes))\n"
  "  else:\n"
  "    print(prefixes or '')\n"
  "except Exception:\n"
  "  import traceback\n"
  "  traceback.print_exc()\n"
  "  sys.exit(1)\n"
)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" "-c" "${_PYTORCH_PREFIX_SCRIPT}"
  RESULT_VARIABLE _PYTORCH_PREFIX_RC
  OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_VARIABLE _PYTORCH_PREFIX_ERR
  ERROR_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
message(STATUS "Torch prefix probe stdout raw: '${PYTORCH_CMAKE_PREFIX}'")
string(LENGTH "${PYTORCH_CMAKE_PREFIX}" _PYTORCH_PREFIX_LEN)
message(STATUS "Torch prefix probe stdout length: ${_PYTORCH_PREFIX_LEN}")

set(_PYTORCH_VERSION_SCRIPT "")
string(APPEND _PYTORCH_VERSION_SCRIPT
  "import sys\n"
  "try:\n"
  "  import torch\n"
  "  ver = str(torch.__version__).split('+')[0]\n"
  "  cuda = torch.version.cuda\n"
  "  tag = ('cu' + str(cuda).replace('.','')) if cuda else 'cpu'\n"
  "  print(ver)\n"
  "  print(tag)\n"
  "except Exception:\n"
  "  import traceback\n"
  "  traceback.print_exc()\n"
  "  sys.exit(1)\n"
)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" "-c" "${_PYTORCH_VERSION_SCRIPT}"
  RESULT_VARIABLE _PYTORCH_VER_RC
  OUTPUT_VARIABLE _PYTORCH_VER_OUT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_VARIABLE _PYTORCH_VER_ERR
  ERROR_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
message(STATUS "Torch version probe stdout raw: '${_PYTORCH_VER_OUT}'")
string(LENGTH "${_PYTORCH_VER_OUT}" _PYTORCH_VER_OUT_LEN)
message(STATUS "Torch version probe stdout length: ${_PYTORCH_VER_OUT_LEN}")

if(_PYTORCH_PREFIX_RC EQUAL 0 AND PYTORCH_CMAKE_PREFIX)
  string(REPLACE "\r\n" "\n" _PYTORCH_CMAKE_PREFIX_TEMP "${PYTORCH_CMAKE_PREFIX}")
  string(REPLACE "\r" "\n" _PYTORCH_CMAKE_PREFIX_TEMP "${_PYTORCH_CMAKE_PREFIX_TEMP}")
  string(REGEX REPLACE "\n+$" "" _PYTORCH_CMAKE_PREFIX_TEMP "${_PYTORCH_CMAKE_PREFIX_TEMP}")
  string(REPLACE "\n" ";" _PYTORCH_CMAKE_PREFIX_LIST "${_PYTORCH_CMAKE_PREFIX_TEMP}")
  message(STATUS "Detected Python Torch cmake prefix path(s): ${_PYTORCH_CMAKE_PREFIX_LIST}")
  foreach(_prefix IN LISTS _PYTORCH_CMAKE_PREFIX_LIST)
    list(APPEND CMAKE_PREFIX_PATH "${_prefix}")
  endforeach()
else()
  message(STATUS "Python-based Torch cmake-prefix detection failed; will fall back to fetch-based resolution if needed.")
  message(STATUS "Torch prefix probe exit code: ${_PYTORCH_PREFIX_RC}")
  if(_PYTORCH_PREFIX_ERR)
    message(STATUS "Torch prefix probe stderr: ${_PYTORCH_PREFIX_ERR}")
  endif()
endif()

if(_PYTORCH_VER_RC EQUAL 0 AND _PYTORCH_VER_OUT)
  string(REPLACE "\r\n" "\n" _PYTORCH_VER_OUT "${_PYTORCH_VER_OUT}")
  string(REPLACE "\r" "\n" _PYTORCH_VER_OUT "${_PYTORCH_VER_OUT}")
  string(REGEX REPLACE "\n+$" "" _PYTORCH_VER_OUT "${_PYTORCH_VER_OUT}")
  string(REPLACE "\n" ";" _PYTORCH_VER_LIST "${_PYTORCH_VER_OUT}")
  list(LENGTH _PYTORCH_VER_LIST _PYTORCH_VER_LIST_LEN)
  if(_PYTORCH_VER_LIST_LEN GREATER_EQUAL 2)
    list(GET _PYTORCH_VER_LIST 0 PYTORCH_VERSION_FROM_PY)
    list(GET _PYTORCH_VER_LIST 1 PYTORCH_CUDA_TAG_FROM_PY)
    message(STATUS "Detected Python Torch version: ${PYTORCH_VERSION_FROM_PY}")
    message(STATUS "Detected Python Torch CUDA tag: ${PYTORCH_CUDA_TAG_FROM_PY}")
  endif()
else()
  message(STATUS "Torch version probe exit code: ${_PYTORCH_VER_RC}")
  if(_PYTORCH_VER_ERR)
    message(STATUS "Torch version probe stderr: ${_PYTORCH_VER_ERR}")
  endif()
endif()

# Honor explicit overrides (cache vars), otherwise derive from Python torch.
if(NOT NODUS_TORCH_VERSION AND PYTORCH_VERSION_FROM_PY)
  set(NODUS_TORCH_VERSION "${PYTORCH_VERSION_FROM_PY}")
endif()
if(NOT NODUS_TORCH_CUDA_TAG AND PYTORCH_CUDA_TAG_FROM_PY)
  set(NODUS_TORCH_CUDA_TAG "${PYTORCH_CUDA_TAG_FROM_PY}")
endif()

# ---------------------------------------------------------------------------
# Ensure nvToolsExt is available (CUDA 12.x+ often lacks the legacy stub lib)
# This must be done BEFORE find_package(Torch).
# ---------------------------------------------------------------------------
if(WIN32 AND NODUS_ENABLE_NVTX AND NODUS_ALLOW_NETWORK_FETCH)
  FetchContent_Declare(nsight_nvtx_redist
    URL "${NODUS_NVTX_URL}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_GetProperties(nsight_nvtx_redist)
  if(NOT nsight_nvtx_redist_POPULATED)
    message(STATUS "Fetching Nsight NVTX (nvToolsExt) redistributable to satisfy Torch CUDA CMake checks.")
    FetchContent_Populate(nsight_nvtx_redist)
  endif()

  set(_NVTX_ROOT "${nsight_nvtx_redist_SOURCE_DIR}")

  # Robust search compatible with CMake 3.15 (avoid '**' patterns).
  set(_NVTX_HDR "")
  set(_NVTX_LIB64 "")
  file(GLOB_RECURSE _NVTX_ALL_FILES LIST_DIRECTORIES false "${_NVTX_ROOT}/*")
  foreach(_f IN LISTS _NVTX_ALL_FILES)
    if(NOT _NVTX_HDR AND _f MATCHES "nvToolsExt\\.h$")
      set(_NVTX_HDR "${_f}")
    endif()
    if(NOT _NVTX_LIB64 AND _f MATCHES "nvToolsExt64_1\\.lib$")
      set(_NVTX_LIB64 "${_f}")
    endif()
  endforeach()

  if(NOT _NVTX_HDR OR NOT _NVTX_LIB64)
    message(FATAL_ERROR
      "Nsight NVTX redist fetched but nvToolsExt.h or nvToolsExt64_1.lib was not found under: ${_NVTX_ROOT}"
    )
  endif()

  get_filename_component(_NVTX_INCLUDE_DIR "${_NVTX_HDR}" DIRECTORY)
  get_filename_component(_NVTX_LIB_DIR "${_NVTX_LIB64}" DIRECTORY)

  # Torch's cuda.cmake searches for a library named 'nvToolsExt' (often expecting nvToolsExt.lib),
  # while the redistributable is typically named nvToolsExt64_1.lib. Provide an alias file name.
  set(_NVTX_ALIAS_LIB "${_NVTX_LIB_DIR}/nvToolsExt.lib")
  if(NOT EXISTS "${_NVTX_ALIAS_LIB}")
    configure_file("${_NVTX_LIB64}" "${_NVTX_ALIAS_LIB}" COPYONLY)
  endif()

  # Normalize include root (NVTX 3 puts headers in include/nvtx3/ sometimes)
  set(_NVTX_INCLUDE_ROOT "${_NVTX_INCLUDE_DIR}")
  if(_NVTX_INCLUDE_ROOT MATCHES "[/\\\\]include[/\\\\]nvtx3$")
    get_filename_component(_NVTX_INCLUDE_ROOT "${_NVTX_INCLUDE_ROOT}" DIRECTORY) # .../include
  endif()

  # Torch tends to form include path as ${NVTOOLSEXT_PATH}/include, so set PATH to the
  # directory whose child is "include".
  get_filename_component(_NVTX_PREFIX "${_NVTX_INCLUDE_ROOT}" DIRECTORY)

  # Safety: if layout is odd, fall back to _NVTX_ROOT but ensure an include/ exists.
  if(NOT EXISTS "${_NVTX_PREFIX}/include")
    set(_NVTX_PREFIX "${_NVTX_ROOT}")
    file(MAKE_DIRECTORY "${_NVTX_PREFIX}/include")
  endif()

  # Create Torch-expected lib/x64 layout so ${NVTOOLSEXT_PATH}/lib/x64/nvToolsExt64_1.lib exists.
  set(_NVTX_LIB_X64_DIR "${_NVTX_PREFIX}/lib/x64")
  file(MAKE_DIRECTORY "${_NVTX_LIB_X64_DIR}")
  if(NOT EXISTS "${_NVTX_LIB_X64_DIR}/nvToolsExt64_1.lib")
    configure_file("${_NVTX_LIB64}" "${_NVTX_LIB_X64_DIR}/nvToolsExt64_1.lib" COPYONLY)
  endif()
  if(EXISTS "${_NVTX_LIB_DIR}/nvToolsExt64_1.dll" AND NOT EXISTS "${_NVTX_LIB_X64_DIR}/nvToolsExt64_1.dll")
    configure_file("${_NVTX_LIB_DIR}/nvToolsExt64_1.dll" "${_NVTX_LIB_X64_DIR}/nvToolsExt64_1.dll" COPYONLY)
  endif()

  # These cache vars are what Torch's CMake expects to be set / discovered.
  set(NVTOOLSEXT_INCLUDE_DIR "${_NVTX_INCLUDE_ROOT}" CACHE PATH "NVTX include dir" FORCE)
  set(NVTOOLSEXT_LIBRARY     "${_NVTX_ALIAS_LIB}"   CACHE FILEPATH "NVTX import library" FORCE)
  set(NVTOOLSEXT_PATH        "${_NVTX_PREFIX}"      CACHE PATH "NVTX prefix (has include/)" FORCE)

  # Also helpful for any scripts/tooling that consult the environment.
  set(ENV{NVTOOLSEXT_PATH} "${_NVTX_PREFIX}")

  if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
    set_target_properties(CUDA::nvToolsExt PROPERTIES
      IMPORTED_LOCATION "${_NVTX_ALIAS_LIB}"
      IMPORTED_LOCATION_RELEASE "${_NVTX_ALIAS_LIB}"
      IMPORTED_LOCATION_DEBUG "${_NVTX_ALIAS_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${_NVTX_INCLUDE_ROOT}"
    )
  endif()
endif()


# ---------------------------------------------------------------------------
# SDL2
# ---------------------------------------------------------------------------
if(NOT TARGET SDL2::SDL2)
  find_package(SDL2 CONFIG QUIET)
  if(NOT SDL2_FOUND)
    if(NODUS_ALLOW_NETWORK_FETCH)
      message(STATUS "SDL2 not found; fetching SDL2 2.28.5 as a dependency.")
      FetchContent_Declare(SDL2
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG release-2.28.5
        GIT_SHALLOW TRUE
      )
      FetchContent_MakeAvailable(SDL2)
    else()
      message(FATAL_ERROR "SDL2 not found and NODUS_ALLOW_NETWORK_FETCH=OFF. Please install SDL2 (CONFIG package) or enable network fetch.")
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# Torch / LibTorch
# ---------------------------------------------------------------------------
find_package(Torch QUIET)

if(NOT Torch_FOUND)
  if(WIN32)
    if(NOT NODUS_ALLOW_NETWORK_FETCH)
      message(FATAL_ERROR "Torch not found and NODUS_ALLOW_NETWORK_FETCH=OFF. Please install Torch/LibTorch or enable network fetch.")
    endif()

    if(NODUS_TORCH_URL)
      set(_LIBTORCH_URL "${NODUS_TORCH_URL}")
    else()
      if(NOT NODUS_TORCH_VERSION OR NOT NODUS_TORCH_CUDA_TAG)
        message(FATAL_ERROR
          "Torch not found. Provide -DNODUS_TORCH_URL=... or set both "
          "-DNODUS_TORCH_VERSION=<x.y.z> and -DNODUS_TORCH_CUDA_TAG=<cu126|cu124|cpu>. "
          "If Python torch is installed, these are normally auto-derived."
        )
      endif()

      # Release-channel libtorch zips follow this pattern:
      #   https://download.pytorch.org/libtorch/<cuTAG>/libtorch-win-shared-with-deps-<ver>%2B<cuTAG>.zip
      set(_LIBTORCH_URL
        "https://download.pytorch.org/libtorch/${NODUS_TORCH_CUDA_TAG}/libtorch-win-shared-with-deps-${NODUS_TORCH_VERSION}%2B${NODUS_TORCH_CUDA_TAG}.zip"
      )
    endif()

    message(STATUS "Torch not found; downloading LibTorch from: ${_LIBTORCH_URL}")
    FetchContent_Declare(libtorch_prebuilt
      URL "${_LIBTORCH_URL}"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_GetProperties(libtorch_prebuilt)
    if(NOT libtorch_prebuilt_POPULATED)
      FetchContent_Populate(libtorch_prebuilt)
    endif()

    # Extracted ZIP contains a 'libtorch' root folder.
    set(LIBTORCH_ROOT "${libtorch_prebuilt_SOURCE_DIR}/libtorch")
    list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
    find_package(Torch REQUIRED PATHS "${LIBTORCH_ROOT}/share/cmake/Torch" NO_DEFAULT_PATH)
  else()
    message(FATAL_ERROR "Torch not found. Please set Torch_DIR/CMAKE_PREFIX_PATH or install LibTorch for your platform.")
  endif()
endif()

if(DEFINED TORCH_INCLUDE_DIRS)
  set(_TORCH_INCLUDE_DIRS_CLEAN "")
  foreach(_inc IN LISTS TORCH_INCLUDE_DIRS)
    if(EXISTS "${_inc}")
      list(APPEND _TORCH_INCLUDE_DIRS_CLEAN "${_inc}")
    else()
      message(STATUS "Pruned non-existent Torch include dir: ${_inc}")
    endif()
  endforeach()
  set(TORCH_INCLUDE_DIRS "${_TORCH_INCLUDE_DIRS_CLEAN}")
endif()

if(NOT TARGET Torch::Torch)
  message(STATUS "Patching CMake to expose Torch::Torch for compatibility.")
  add_library(Torch::Torch INTERFACE IMPORTED)
  set_target_properties(Torch::Torch PROPERTIES
    INTERFACE_LINK_LIBRARIES "${TORCH_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}"
  )
  if(TORCH_CXX_FLAGS)
    target_compile_options(Torch::Torch INTERFACE ${TORCH_CXX_FLAGS})
  endif()
endif()

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Windows DLL exports convenience
# ---------------------------------------------------------------------------
if(WIN32)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
add_library(canvas_tables SHARED
  canvas_abi.cpp
  module_preview.cpp
  module_library.cpp
  module_library_actualizer.cpp
  plugin_loader.cpp
  plugin_manager.cpp
  table_abi.cpp
  table_node_groups.cpp
  menu_waveform.cpp
  menu_waveform_abi.cpp
  rope_sim.cpp
  raytrace_2d.cpp
  raytrace_3d.cpp
  stage_abi.cpp
  subject_capture.cpp
  text_render_helper.cpp
  tool_registry.cpp
  thread_manager.cpp
)

# Small helper executable to run the module actualizer from the built project
add_executable(modlib_actualize tools/modlib_actualize_main.cpp)
target_include_directories(modlib_actualize PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(modlib_actualize PRIVATE canvas_tables)
target_compile_features(modlib_actualize PUBLIC cxx_std_17)

# Small helper to build/load modules at runtime for development
add_executable(modloader tools/modloader_main.cpp)
target_include_directories(modloader PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(modloader PRIVATE canvas_tables)
target_compile_features(modloader PUBLIC cxx_std_17)

find_package(OpenGL REQUIRED)

add_executable(frontend_shell tools/frontend_shell.cpp)
target_include_directories(frontend_shell PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(frontend_shell PRIVATE
  canvas_tables
  SDL2::SDL2
  SDL2::SDL2main
  Torch::Torch
  OpenGL::GL
)
target_compile_features(frontend_shell PUBLIC cxx_std_17)

if(TARGET SDL2::SDL2)
  add_custom_command(TARGET frontend_shell POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE_DIR:SDL2::SDL2>/SDL2.dll"
      "$<TARGET_FILE_DIR:frontend_shell>/SDL2.dll"
    COMMENT "Copy SDL2 runtime DLL beside frontend_shell"
    VERBATIM
  )
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

find_package(Eigen3 REQUIRED)

target_include_directories(canvas_tables PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(canvas_tables PUBLIC Eigen3::Eigen)
target_compile_features(canvas_tables PUBLIC cxx_std_17)

enable_testing()

# Unit tests
add_executable(test_thread_manager
  tests/thread_manager_test.cpp
)
target_link_libraries(test_thread_manager PRIVATE canvas_tables)
add_test(NAME thread_manager_test COMMAND test_thread_manager)

add_test(NAME tool_dll_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/tool_dll_test.py
)

option(ENABLE_EIGEN_SIM "Enable Eigen-accelerated rope simulation" OFF)
if(ENABLE_EIGEN_SIM)
  target_compile_definitions(canvas_tables PUBLIC EIGEN_SIM)
endif()
