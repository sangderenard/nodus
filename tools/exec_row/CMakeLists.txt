# exec_row: the exec tool row (a plugin DLL) and the test that drives it the way the thread manager drives a row.
# A project of its own so it builds in seconds and never touches a Nodus build that is in use. It links the shared
# tensor core of a Nodus build that already exists, read only:
#   cmake -S tools/exec_row -B build-exec-row -DNODUS_TENSOR_CORE_DIR=<a Nodus build>/Release
#   cmake --build build-exec-row --config Release
#   (with that Release folder on PATH)
#   build-exec-row/Release/exec_row_test.exe build-exec-row/Release/exec_row_tool.dll <compressor> <scratch-dir>
cmake_minimum_required(VERSION 3.20)
project(exec_row LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(NODUS_TENSOR_CORE_DIR "" CACHE PATH "Folder that holds nodus_tensor_core.lib and nodus_tensor_core.dll")
if(NOT EXISTS "${NODUS_TENSOR_CORE_DIR}/nodus_tensor_core.lib")
  message(FATAL_ERROR "NODUS_TENSOR_CORE_DIR must hold nodus_tensor_core.lib (got '${NODUS_TENSOR_CORE_DIR}')")
endif()
set(NODUS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

add_library(exec_row_tool SHARED exec_row_tool.cpp)
add_executable(exec_row_test exec_row_test.cpp)
foreach(target exec_row_tool exec_row_test)
  target_include_directories(${target} PRIVATE "${NODUS_ROOT}" "${NODUS_ROOT}/include")
  target_link_libraries(${target} PRIVATE "${NODUS_TENSOR_CORE_DIR}/nodus_tensor_core.lib")
  target_compile_definitions(${target} PRIVATE NOMINMAX)
endforeach()

# The row as it sits in the tool registry: the registration source and the real registry, compiled here by
# themselves, so the registration is proven without building the canvas image.
#   exec_row_registry_test declare <file> <compressor> <scratch-dir>
#   (with NODUS_EXEC_ROWS=<file> and NODUS_EXEC_CORE_ROOTS=<core> set)  exec_row_registry_test check <file> <compressor> <scratch-dir>
add_executable(exec_row_registry_test exec_row_registry_test.cpp
  "${NODUS_ROOT}/src/tools/exec_row_registered.cpp" "${NODUS_ROOT}/src/tool_registry.cpp")
target_include_directories(exec_row_registry_test PRIVATE "${NODUS_ROOT}" "${NODUS_ROOT}/include")
target_link_libraries(exec_row_registry_test PRIVATE "${NODUS_TENSOR_CORE_DIR}/nodus_tensor_core.lib")
target_compile_definitions(exec_row_registry_test PRIVATE NOMINMAX)
