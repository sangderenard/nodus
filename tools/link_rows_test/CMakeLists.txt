# link_rows_test: stage 2 of the Nodus link (include/nodus_link_rows.h), research case 98d49cb8520e44d092c5d134c37c3c32.
# A project of its own, like tools/exec_row: it compiles the rows and the real tool registry by themselves and links the
# tensor core of a Nodus build that already exists, read only, so it builds in seconds and touches no build in use:
#   cmake -S tools/link_rows_test -B build-link-rows -DNODUS_TENSOR_CORE_DIR=<a Nodus build>/Release
#   cmake --build build-link-rows --config Release
#   (with that Release folder on PATH)  build-link-rows/Release/link_rows_test.exe
cmake_minimum_required(VERSION 3.20)
project(link_rows_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(NODUS_TENSOR_CORE_DIR "" CACHE PATH "Folder that holds nodus_tensor_core.lib and nodus_tensor_core.dll")
# Linux (the Oracle A1 host): the build folder that holds libnodus_tensor_core.so and libnodus_headless.so (the canvas
# is linked into the headless image there; the link rows include the metrics row, which paints with gp_table_raster_rgba).
if(WIN32)
  if(NOT EXISTS "${NODUS_TENSOR_CORE_DIR}/nodus_tensor_core.lib")
    message(FATAL_ERROR "NODUS_TENSOR_CORE_DIR must hold nodus_tensor_core.lib (got '${NODUS_TENSOR_CORE_DIR}')")
  endif()
  if(NOT EXISTS "${NODUS_TENSOR_CORE_DIR}/canvas_tables.lib")
    message(FATAL_ERROR "NODUS_TENSOR_CORE_DIR must also hold canvas_tables.lib: the link rows include the metrics row, which paints with gp_table_raster_rgba")
  endif()
else()
  foreach(lib nodus_tensor_core nodus_headless)
    if(NOT EXISTS "${NODUS_TENSOR_CORE_DIR}/lib${lib}.so")
      message(FATAL_ERROR "NODUS_TENSOR_CORE_DIR must hold lib${lib}.so (got '${NODUS_TENSOR_CORE_DIR}')")
    endif()
  endforeach()
  find_package(OpenSSL REQUIRED)
  find_package(Threads REQUIRED)
endif()
set(NODUS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_executable(link_rows_test link_rows_test.cpp "${NODUS_ROOT}/src/nodus_link_rows.cpp" "${NODUS_ROOT}/src/tool_registry.cpp")
target_include_directories(link_rows_test PRIVATE "${NODUS_ROOT}" "${NODUS_ROOT}/include")
if(WIN32)
  target_link_libraries(link_rows_test PRIVATE "${NODUS_TENSOR_CORE_DIR}/nodus_tensor_core.lib" "${NODUS_TENSOR_CORE_DIR}/canvas_tables.lib" ws2_32 bcrypt winhttp crypt32)
else()
  target_link_libraries(link_rows_test PRIVATE "${NODUS_TENSOR_CORE_DIR}/libnodus_tensor_core.so"
    "${NODUS_TENSOR_CORE_DIR}/libnodus_headless.so" OpenSSL::SSL OpenSSL::Crypto Threads::Threads ${CMAKE_DL_LIBS})
  set_target_properties(link_rows_test PROPERTIES BUILD_RPATH "${NODUS_TENSOR_CORE_DIR}")
endif()
target_compile_definitions(link_rows_test PRIVATE NOMINMAX)
if(MSVC)
  target_compile_options(link_rows_test PRIVATE /W4)
endif()
