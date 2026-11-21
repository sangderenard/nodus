# link_metrics_test: the link metrics row (module_library/source/tools/link_metrics.cpp), development case 2115fd107eba439fb0b5e65d6e6809e9 stage A1.
# A project of its own, like tools/exec_row: it compiles the rows and the real tool registry by themselves and links the
# tensor core of a Nodus build that already exists, read only, so it builds in seconds and touches no build in use:
#   cmake -S tools/link_metrics_test -B build-link-metrics -DNODUS_TENSOR_CORE_DIR=<a Nodus build>/Release
#   cmake --build build-link-metrics --config Release
#   (with that Release folder on PATH)  build-link-metrics/Release/link_metrics_test.exe
cmake_minimum_required(VERSION 3.20)
project(link_metrics_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(NODUS_TENSOR_CORE_DIR "" CACHE PATH "Folder that holds nodus_tensor_core.lib and nodus_tensor_core.dll")
if(NOT EXISTS "${NODUS_TENSOR_CORE_DIR}/nodus_tensor_core.lib")
  message(FATAL_ERROR "NODUS_TENSOR_CORE_DIR must hold nodus_tensor_core.lib (got '${NODUS_TENSOR_CORE_DIR}')")
endif()
if(NOT EXISTS "${NODUS_TENSOR_CORE_DIR}/canvas_tables.lib")
  message(FATAL_ERROR "NODUS_TENSOR_CORE_DIR must also hold canvas_tables.lib: the metrics row paints with gp_table_raster_rgba")
endif()
set(NODUS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_executable(link_metrics_test link_metrics_test.cpp "${NODUS_ROOT}/src/nodus_link_rows.cpp" "${NODUS_ROOT}/src/tool_registry.cpp")
target_include_directories(link_metrics_test PRIVATE "${NODUS_ROOT}" "${NODUS_ROOT}/include")
target_link_libraries(link_metrics_test PRIVATE "${NODUS_TENSOR_CORE_DIR}/nodus_tensor_core.lib" "${NODUS_TENSOR_CORE_DIR}/canvas_tables.lib" ws2_32 bcrypt winhttp crypt32)
target_compile_definitions(link_metrics_test PRIVATE NOMINMAX)
# The channel-group columns read the meta edges of a real headless canvas: give NODUS_HEADLESS_DIR the folder of a Nodus
# build that holds nodus_headless.lib (and, on PATH at run time, nodus_headless.dll) to include that part of the test.
set(NODUS_HEADLESS_DIR "" CACHE PATH "Optional: folder that holds nodus_headless.lib; adds the channel-group test")
if(EXISTS "${NODUS_HEADLESS_DIR}/nodus_headless.lib")
  target_link_libraries(link_metrics_test PRIVATE "${NODUS_HEADLESS_DIR}/nodus_headless.lib")
  target_compile_definitions(link_metrics_test PRIVATE LINK_METRICS_TEST_HEADLESS)
endif()
if(MSVC)
  target_compile_options(link_metrics_test PRIVATE /W4)
endif()
