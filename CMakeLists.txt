  # BitOps test
  add_executable(test_bitops
    tests/test_bitops.cpp
  )
  target_link_libraries(test_bitops PRIVATE canvas_tables)
  add_test(NAME test_bitops COMMAND test_bitops)

  # Translation matrix test
  add_executable(test_translation_matrix
    tests/test_translation_matrix.cpp
  )
  target_link_libraries(test_translation_matrix PRIVATE canvas_tables)
  add_test(NAME test_translation_matrix COMMAND test_translation_matrix)
  # Torch (standalone)
  if(EXISTS "${CMAKE_SOURCE_DIR}/dev_bundle/include/torch-src" AND EXISTS "${CMAKE_SOURCE_DIR}/dev_bundle/include/torch-src/CMakeLists.txt")
    message(STATUS "Using local torch-src from dev_bundle/include; adding as subdirectory.")
    add_subdirectory("${CMAKE_SOURCE_DIR}/dev_bundle/include/torch-src" "${CMAKE_BINARY_DIR}/torch-src-build")
    include_directories("${CMAKE_SOURCE_DIR}/dev_bundle/include/torch-src")
  endif()
  # Ensure SDL_TTF::SDL_TTF alias exists for downstream compatibility
  if(TARGET SDL3_ttf-shared)
    add_library(SDL_TTF::SDL_TTF ALIAS SDL3_ttf-shared)
  elseif(TARGET SDL3_ttf-static)
    add_library(SDL_TTF::SDL_TTF ALIAS SDL3_ttf-static)
  endif()
# Set local source directories for all dependencies (including Torch)
set(_SDL3_SRC      "${CMAKE_SOURCE_DIR}/dev_bundle/include/SDL3-src")
set(_SDLTTF_SRC    "${CMAKE_SOURCE_DIR}/dev_bundle/include/SDL_ttf-src")
set(_FREETYPE_SRC  "${CMAKE_SOURCE_DIR}/dev_bundle/include/freetype-src")
set(_HARFBUZZ_SRC  "${CMAKE_SOURCE_DIR}/dev_bundle/include/harfbuzz-src")


cmake_minimum_required(VERSION 3.15)
project(canvas_tables LANGUAGES CXX)
include(FetchContent)
set(_all_sources_found TRUE)
foreach(_dep_src IN ITEMS _SDL3_SRC _SDLTTF_SRC _FREETYPE_SRC _HARFBUZZ_SRC)
  if(NOT EXISTS ${${_dep_src}})
    set(_all_sources_found FALSE)
  endif()
endforeach()

  # SDL3
  if(EXISTS "${_SDL3_SRC}")
    message(STATUS "Using local SDL3-src from dev_bundle/include; adding as subdirectory.")
    add_subdirectory("${_SDL3_SRC}" "${CMAKE_BINARY_DIR}/SDL3-src-build")
    include_directories("${_SDL3_SRC}/include")
  else()
    find_package(SDL3 QUIET)
    if(NOT SDL3_FOUND)
      if(NODUS_ALLOW_NETWORK_FETCH)
        message(STATUS "SDL3 not found; fetching SDL 3.x as a dependency.")
        FetchContent_Declare(SDL3
          GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
          GIT_TAG ${NODUS_SDL3_GIT_TAG}
          GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(SDL3)
      else()
        message(FATAL_ERROR "SDL3 not found and NODUS_ALLOW_NETWORK_FETCH=OFF. Please install SDL3 or enable network fetch.")
      endif()
    endif()
  endif()

  # SDL_ttf
  if(EXISTS "${_SDLTTF_SRC}")
    message(STATUS "Using local SDL_ttf-src from dev_bundle/include; adding as subdirectory.")
    add_subdirectory("${_SDLTTF_SRC}" "${CMAKE_BINARY_DIR}/SDL_ttf-src-build")
    include_directories("${_SDLTTF_SRC}/include")
    # Ensure SDL_TTF::SDL_TTF alias exists for downstream compatibility
    if(TARGET SDL3_ttf-shared)
      add_library(SDL_TTF::SDL_TTF ALIAS SDL3_ttf-shared)
    elseif(TARGET SDL3_ttf-static)
      add_library(SDL_TTF::SDL_TTF ALIAS SDL3_ttf-static)
    endif()
  else()
    find_package(SDL3_ttf QUIET)
    if(NOT SDL3_ttf_FOUND)
      if(NODUS_ALLOW_NETWORK_FETCH)
        message(STATUS "SDL_ttf (SDL3-capable) not found; fetching SDL_ttf and HarfBuzz as dependencies.")
        message(STATUS "Also fetching FreeType so SDL_ttf builds against a known FreeType.")
        # FreeType (font rasterizer)
        if(EXISTS "${_FREETYPE_SRC}")
          add_subdirectory("${_FREETYPE_SRC}" "${CMAKE_BINARY_DIR}/freetype-src-build")
          include_directories("${_FREETYPE_SRC}/include")
        else()
          FetchContent_Declare(freetype
            GIT_REPOSITORY https://github.com/freetype/freetype.git
            GIT_TAG ${NODUS_FREETYPE_GIT_TAG}
            GIT_SHALLOW TRUE
          )
          FetchContent_MakeAvailable(freetype)
          list(APPEND CMAKE_PREFIX_PATH "${freetype_SOURCE_DIR}")
          if(DEFINED freetype_BINARY_DIR)
            list(APPEND CMAKE_PREFIX_PATH "${freetype_BINARY_DIR}")
          endif()
        endif()
        # HarfBuzz (used for shaping)
        if(EXISTS "${_HARFBUZZ_SRC}")
          add_subdirectory("${_HARFBUZZ_SRC}" "${CMAKE_BINARY_DIR}/harfbuzz-src-build")
          include_directories("${_HARFBUZZ_SRC}/src")
        else()
          FetchContent_Declare(harfbuzz
            GIT_REPOSITORY https://github.com/harfbuzz/harfbuzz.git
            GIT_TAG 8.2.0
            GIT_SHALLOW TRUE
          )
          FetchContent_MakeAvailable(harfbuzz)
          list(APPEND CMAKE_PREFIX_PATH "${harfbuzz_SOURCE_DIR}")
        endif()
        # SDL_ttf (source repo supports building against SDL3 when configured)
        FetchContent_Declare(SDL_ttf
          GIT_REPOSITORY https://github.com/libsdl-org/SDL_ttf.git
          GIT_TAG ${NODUS_SDLTTF_GIT_TAG}
          GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(SDL_ttf)
      else()
        message(FATAL_ERROR "SDL_ttf (SDL3) not found and NODUS_ALLOW_NETWORK_FETCH=OFF. Please install SDL3_ttf or enable network fetch.")
      endif()
    endif()
  endif()



  # SDL3
  if(EXISTS "${_SDL3_SRC}")
    message(STATUS "Using local SDL3-src from dev_bundle/include; skipping FetchContent.")
    list(APPEND CMAKE_PREFIX_PATH "${_SDL3_SRC}")
    include_directories("${_SDL3_SRC}")
  else()
    find_package(SDL3 QUIET)
    if(NOT SDL3_FOUND)
      if(NODUS_ALLOW_NETWORK_FETCH)
        message(STATUS "SDL3 not found; fetching SDL 3.x as a dependency.")
        FetchContent_Declare(SDL3
          GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
          GIT_TAG ${NODUS_SDL3_GIT_TAG}
          GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(SDL3)
      else()
        message(FATAL_ERROR "SDL3 not found and NODUS_ALLOW_NETWORK_FETCH=OFF. Please install SDL3 or enable network fetch.")
      endif()
    endif()
  endif()

  # FreeType
  if(EXISTS "${_FREETYPE_SRC}")
    message(STATUS "Using local freetype-src from dev_bundle/include; skipping FetchContent.")
    list(APPEND CMAKE_PREFIX_PATH "${_FREETYPE_SRC}")
    include_directories("${_FREETYPE_SRC}")
  else()
    set(_NEED_FETCH_FREETYPE TRUE)
  endif()

  # HarfBuzz
  if(EXISTS "${_HARFBUZZ_SRC}")
    message(STATUS "Using local harfbuzz-src from dev_bundle/include; skipping FetchContent.")
    list(APPEND CMAKE_PREFIX_PATH "${_HARFBUZZ_SRC}")
    include_directories("${_HARFBUZZ_SRC}")
  else()
    set(_NEED_FETCH_HARFBUZZ TRUE)
  endif()

  # SDL_ttf
  if(EXISTS "${_SDLTTF_SRC}")
    message(STATUS "Using local SDL_ttf-src from dev_bundle/include; skipping FetchContent.")
    list(APPEND CMAKE_PREFIX_PATH "${_SDLTTF_SRC}")
    include_directories("${_SDLTTF_SRC}")
  else()
    find_package(SDL3_ttf QUIET)
    if(NOT SDL3_ttf_FOUND)
      if(NODUS_ALLOW_NETWORK_FETCH)
        message(STATUS "SDL_ttf (SDL3-capable) not found; fetching SDL_ttf and HarfBuzz as dependencies.")
        message(STATUS "Also fetching FreeType so SDL_ttf builds against a known FreeType.")
        # FreeType (font rasterizer)
        if(_NEED_FETCH_FREETYPE)
          FetchContent_Declare(freetype
            GIT_REPOSITORY https://github.com/freetype/freetype.git
            GIT_TAG ${NODUS_FREETYPE_GIT_TAG}
            GIT_SHALLOW TRUE
          )
          FetchContent_MakeAvailable(freetype)
          list(APPEND CMAKE_PREFIX_PATH "${freetype_SOURCE_DIR}")
          if(DEFINED freetype_BINARY_DIR)
            list(APPEND CMAKE_PREFIX_PATH "${freetype_BINARY_DIR}")
          endif()
        endif()
        # HarfBuzz (used for shaping)
        if(_NEED_FETCH_HARFBUZZ)
          FetchContent_Declare(harfbuzz
            GIT_REPOSITORY https://github.com/harfbuzz/harfbuzz.git
            GIT_TAG 8.2.0
            GIT_SHALLOW TRUE
          )
          FetchContent_MakeAvailable(harfbuzz)
          list(APPEND CMAKE_PREFIX_PATH "${harfbuzz_SOURCE_DIR}")
        endif()
        # SDL_ttf (source repo supports building against SDL3 when configured)
        FetchContent_Declare(SDL_ttf
          GIT_REPOSITORY https://github.com/libsdl-org/SDL_ttf.git
          GIT_TAG ${NODUS_SDLTTF_GIT_TAG}
          GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(SDL_ttf)
      else()
        message(FATAL_ERROR "SDL_ttf (SDL3) not found and NODUS_ALLOW_NETWORK_FETCH=OFF. Please install SDL3_ttf or enable network fetch.")
      endif()
    endif()
  endif()


# --- Restored original Python probe logic for torch (with print statements) ---

# --- Strict Python torch probe: fail hard and print all output/errors ---
execute_process(
  COMMAND python -c "import torch; prefixes = getattr(torch.utils, 'cmake_prefix_path', '');\nif isinstance(prefixes, (list, tuple)): print('\\n'.join(prefixes))\nelse: print(prefixes or '')"
  RESULT_VARIABLE _PYTORCH_PREFIX_RC
  OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX
  ERROR_VARIABLE _PYTORCH_PREFIX_ERR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Torch prefix probe stdout raw: '${PYTORCH_CMAKE_PREFIX}'")
message(STATUS "Torch prefix probe stderr: '${_PYTORCH_PREFIX_ERR}'")
string(LENGTH "${PYTORCH_CMAKE_PREFIX}" _PYTORCH_PREFIX_LEN)
message(STATUS "Torch prefix probe stdout length: ${_PYTORCH_PREFIX_LEN}")
if(NOT (_PYTORCH_PREFIX_RC EQUAL 0 AND PYTORCH_CMAKE_PREFIX))
  message(FATAL_ERROR "Torch Python probe failed!\nReturn code: ${_PYTORCH_PREFIX_RC}\nStdout: ${PYTORCH_CMAKE_PREFIX}\nStderr: ${_PYTORCH_PREFIX_ERR}")
endif()
string(REPLACE "\r\n" "\n" _PYTORCH_CMAKE_PREFIX_TEMP "${PYTORCH_CMAKE_PREFIX}")
string(REPLACE "\r" "\n" _PYTORCH_CMAKE_PREFIX_TEMP "${_PYTORCH_CMAKE_PREFIX_TEMP}")
string(REPLACE "\n" ";" _PYTORCH_CMAKE_PREFIX_LIST "${_PYTORCH_CMAKE_PREFIX_TEMP}")
foreach(_prefix IN LISTS _PYTORCH_CMAKE_PREFIX_LIST)
  list(APPEND CMAKE_PREFIX_PATH "${_prefix}")
endforeach()

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

# If Torch_INCLUDE_DIRS is not set, use the first Python probe result, but always build from dev_bundle/include/torch-src
if(NOT DEFINED Torch_INCLUDE_DIRS AND DEFINED _PYTORCH_CMAKE_PREFIX_LIST)
  list(LENGTH _PYTORCH_CMAKE_PREFIX_LIST _torch_prefix_count)
  if(_torch_prefix_count GREATER 0)
    list(GET _PYTORCH_CMAKE_PREFIX_LIST 0 _first_torch_prefix)
    set(_TORCHSRC_DEST "${CMAKE_SOURCE_DIR}/dev_bundle/include/torch-src")
    file(MAKE_DIRECTORY "${_TORCHSRC_DEST}")
    if(EXISTS "${_first_torch_prefix}")
      file(COPY "${_first_torch_prefix}/" DESTINATION "${_TORCHSRC_DEST}")
    endif()
    set(Torch_INCLUDE_DIRS "${_TORCHSRC_DEST}")
  endif()
endif()


# ---------------------------------------------------------------------------
# Use persistent dev_bundle/nvtoolsext for nvToolsExt headers and libraries
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# SDL3
# ---------------------------------------------------------------------------
if(NOT TARGET SDL3::SDL3)
  find_package(SDL3 CONFIG QUIET)
  if(NOT SDL3_FOUND)
    if(NODUS_ALLOW_NETWORK_FETCH)
      message(STATUS "SDL3 not found; fetching SDL 3.x as a dependency.")
      FetchContent_Declare(SDL3
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG ${NODUS_SDL3_GIT_TAG}
        GIT_SHALLOW TRUE
      )
      FetchContent_MakeAvailable(SDL3)
    else()
      message(FATAL_ERROR "SDL3 not found and NODUS_ALLOW_NETWORK_FETCH=OFF. Please install SDL3 (CONFIG package) or enable network fetch.")
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# SDL_ttf (prefer SDL3_ttf) + HarfBuzz (required for shaping when building)
# ---------------------------------------------------------------------------
if(NOT TARGET SDL_TTF::SDL_TTF)
  # Require SDL3-aware SDL_ttf package (no SDL2 fallback)
  find_package(SDL3_ttf CONFIG QUIET)

  if(NOT SDL3_ttf_FOUND)
    if(NODUS_ALLOW_NETWORK_FETCH)
      message(STATUS "SDL_ttf (SDL3-capable) not found; fetching SDL_ttf and HarfBuzz as dependencies.")
      message(STATUS "Also fetching FreeType so SDL_ttf builds against a known FreeType.")
      # FreeType (font rasterizer)
      FetchContent_Declare(freetype
        GIT_REPOSITORY https://github.com/freetype/freetype.git
        GIT_TAG ${NODUS_FREETYPE_GIT_TAG}
        GIT_SHALLOW TRUE
      )
      
      FetchContent_GetProperties(freetype)
      if(NOT freetype_POPULATED)
        FetchContent_Populate(freetype)
      endif()
      # Expose freetype to the CMake prefix so SDL_ttf can find it
      list(APPEND CMAKE_PREFIX_PATH "${freetype_SOURCE_DIR}")
      if(DEFINED freetype_BINARY_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${freetype_BINARY_DIR}")
      endif()
      # HarfBuzz (used for shaping)
      FetchContent_Declare(harfbuzz
        GIT_REPOSITORY https://github.com/harfbuzz/harfbuzz.git
        GIT_TAG 8.2.0
        GIT_SHALLOW TRUE
      )
      # SDL_ttf (source repo supports building against SDL3 when configured)
      FetchContent_Declare(SDL_ttf
        GIT_REPOSITORY https://github.com/libsdl-org/SDL_ttf.git
        GIT_TAG ${NODUS_SDLTTF_GIT_TAG}
        GIT_SHALLOW TRUE
      )
      FetchContent_GetProperties(harfbuzz)
      if(NOT harfbuzz_POPULATED)
        FetchContent_Populate(harfbuzz)
      endif()
      # Expose harfbuzz to the CMake prefix so SDL_ttf can find it
      list(APPEND CMAKE_PREFIX_PATH "${harfbuzz_SOURCE_DIR}")

      FetchContent_GetProperties(SDL_ttf)
      if(NOT SDL_ttf_POPULATED)
        FetchContent_Populate(SDL_ttf)
      endif()
      # Make available (will configure and build if necessary)
      # Ensure freetype/harfbuzz are available before SDL_ttf so the SDL_ttf
      # build picks them up when configuring.
      FetchContent_MakeAvailable(freetype harfbuzz SDL_ttf)
    else()
      message(FATAL_ERROR "SDL_ttf (SDL3) not found and NODUS_ALLOW_NETWORK_FETCH=OFF. Please install SDL3_ttf or enable network fetch.")
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

# ---------------------------------------------------------------------------
# Developer bundle target: gather built DLLs and third-party headers
# into ${CMAKE_SOURCE_DIR}/dev_bundle for local developer builds.
# This target is intentionally opt-in: run `cmake --build . --target prepare_dev_bundle`.
# ---------------------------------------------------------------------------
set(DEV_BUNDLE_DIR "${CMAKE_SOURCE_DIR}/dev_bundle")
set(DEV_BUNDLE_BIN "${DEV_BUNDLE_DIR}/bin")
set(DEV_BUNDLE_INCLUDE "${DEV_BUNDLE_DIR}/include")

add_custom_target(prepare_dev_bundle
  COMMENT "Prepare developer bundle (copies built DLLs and headers into ${DEV_BUNDLE_DIR})"
)

# Ensure directories exist at build-time
add_custom_command(TARGET prepare_dev_bundle PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DEV_BUNDLE_BIN}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DEV_BUNDLE_INCLUDE}"
)

# Copy all DLLs from the canvas_tables target output directory into dev_bundle/bin
if(TARGET canvas_tables)
  add_custom_command(TARGET prepare_dev_bundle POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Copying runtime DLLs from $<TARGET_FILE_DIR:canvas_tables> to ${DEV_BUNDLE_BIN}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "$<TARGET_FILE_DIR:canvas_tables>" "${DEV_BUNDLE_BIN}"
  )
else()
  # Fallback: copy any DLLs from the build tree config dir
  add_custom_command(TARGET prepare_dev_bundle POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Copying runtime DLLs from ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR} to ${DEV_BUNDLE_BIN}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}" "${DEV_BUNDLE_BIN}"
  )
endif()



# Copy all major FetchContent source/include trees for dev completeness
set(_NODUS_FETCHCONTENT_DIR "$ENV{LOCALAPPDATA}/nodus_fetchcontent")
set(_NODUS_FETCHCONTENT_SUBDIRS
  "SDL_ttf-src"
  "SDL3-src"
  "freetype-src"
  "harfbuzz-src"
  "torch-src"
)
foreach(_subdir IN LISTS _NODUS_FETCHCONTENT_SUBDIRS)
  set(_srcdir "${_NODUS_FETCHCONTENT_DIR}/${_subdir}")
  if(EXISTS "${_srcdir}")
    add_custom_command(TARGET prepare_dev_bundle POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E echo "Copying ${_subdir} -> ${DEV_BUNDLE_INCLUDE}/${_subdir}"
      COMMAND ${CMAKE_COMMAND} -E copy_directory "${_srcdir}" "${DEV_BUNDLE_INCLUDE}/${_subdir}"
    )
  endif()
endforeach()


# Copy Torch headers into dev_bundle/include/torch-src for unified dev pack
if(DEFINED Torch_INCLUDE_DIRS)
  set(_TORCHSRC_DEST "${DEV_BUNDLE_INCLUDE}/torch-src")
  file(MAKE_DIRECTORY "${_TORCHSRC_DEST}")
  foreach(_inc IN LISTS Torch_INCLUDE_DIRS)
    if(EXISTS "${_inc}")
      add_custom_command(TARGET prepare_dev_bundle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Copying Torch headers from ${_inc} -> ${_TORCHSRC_DEST}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${_inc}" "${_TORCHSRC_DEST}"
      )
    endif()
  endforeach()
endif()

# Also allow copying explicit include vars if set (Torch_INCLUDE_DIRS etc.)
if(DEFINED Torch_INCLUDE_DIRS)
  foreach(_inc IN LISTS Torch_INCLUDE_DIRS)
    add_custom_command(TARGET prepare_dev_bundle POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E echo "Copying headers from ${_inc} to ${DEV_BUNDLE_INCLUDE}"
      COMMAND ${CMAKE_COMMAND} -E copy_directory "${_inc}" "${DEV_BUNDLE_INCLUDE}/torch_inc"
    )
  endforeach()
endif()

# Make prepare_dev_bundle depend on the main library so DLLs exist
add_dependencies(prepare_dev_bundle canvas_tables)


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

# Persist Torch headers so they survive build-dir cleanup and are available
# for plugin/module development. Copy headers into the FetchContent base dir
# (configurable via FETCHCONTENT_BASE_DIR) and expose them via an INTERFACE
# target `nodus_persisted_torch_includes` so other targets can consume them.
if(DEFINED TORCH_INCLUDE_DIRS)
  set(_NODUS_PERSISTENT_TORCH_INCLUDE "${FETCHCONTENT_BASE_DIR}/libtorch_include")
  if(NOT EXISTS "${_NODUS_PERSISTENT_TORCH_INCLUDE}")
    file(MAKE_DIRECTORY "${_NODUS_PERSISTENT_TORCH_INCLUDE}")
    foreach(_inc IN LISTS TORCH_INCLUDE_DIRS)
      if(EXISTS "${_inc}")
        message(STATUS "Persisting Torch headers from ${_inc} -> ${_NODUS_PERSISTENT_TORCH_INCLUDE}")
        file(COPY "${_inc}/" DESTINATION "${_NODUS_PERSISTENT_TORCH_INCLUDE}")
      endif()
    endforeach()
  endif()

    if(NOT NODUS_PERSIST_TORCH_IN_REPO)
      if(NOT TARGET nodus_persisted_torch_includes)
        add_library(nodus_persisted_torch_includes INTERFACE)
        target_include_directories(nodus_persisted_torch_includes INTERFACE "${_NODUS_PERSISTENT_TORCH_INCLUDE}")
      endif()
    else()
      message(STATUS "NODUS_PERSIST_TORCH_IN_REPO=ON; not creating nodus_persisted_torch_includes to avoid duplicate includes")
    endif()
endif()



# ---------------------------------------------------------------------------
# Prefer dev_bundle/include/*-src for dependency headers if present
set(_NODUS_DEV_BUNDLE_INCLUDE "${CMAKE_SOURCE_DIR}/dev_bundle/include")
foreach(_dep IN LISTS _NODUS_FETCHCONTENT_SUBDIRS)
  set(_depinc "${_NODUS_DEV_BUNDLE_INCLUDE}/${_dep}")
  if(EXISTS "${_depinc}")
    list(APPEND CMAKE_PREFIX_PATH "${_depinc}")
    include_directories("${_depinc}")
  endif()
endforeach()

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

# Fallback: if SDL_ttf/SDL3_ttf was populated via FetchContent, ensure its
# include dir is provided to the main library even if the imported target
# does not expose INTERFACE_INCLUDE_DIRECTORIES.
if(DEFINED SDL3_ttf_SOURCE_DIR AND EXISTS "${SDL3_ttf_SOURCE_DIR}/include")
  target_include_directories(canvas_tables PUBLIC "${SDL3_ttf_SOURCE_DIR}/include")
endif()

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
  Torch::Torch
  OpenGL::GL
)
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

find_package(Eigen3 REQUIRED)

target_include_directories(canvas_tables PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(canvas_tables PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(canvas_tables PUBLIC Eigen3::Eigen)
if(TARGET Torch::Torch)
  target_link_libraries(canvas_tables PUBLIC Torch::Torch)
endif()
if(TARGET nodus_persisted_torch_includes)
  target_link_libraries(canvas_tables PUBLIC nodus_persisted_torch_includes)
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

