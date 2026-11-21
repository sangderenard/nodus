# fabric_state_test: the fabric row on the tensor state library against the row as it is (research case
# 2a1563063ca34358899d5cb971cdf521). Its own project against an existing Nodus build's tensor core:
#   cmake -S tools/fabric_state_test -B build-fabric -DNODUS_TENSOR_CORE_DIR=<a Nodus build>/Release
#   cmake --build build-fabric --config Release
cmake_minimum_required(VERSION 3.20)
project(fabric_state_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(NODUS_TENSOR_CORE_DIR "" CACHE PATH "Folder that holds nodus_tensor_core.lib and nodus_tensor_core.dll")
if(NOT EXISTS "${NODUS_TENSOR_CORE_DIR}/nodus_tensor_core.lib")
  message(FATAL_ERROR "NODUS_TENSOR_CORE_DIR must hold nodus_tensor_core.lib (got '${NODUS_TENSOR_CORE_DIR}')")
endif()
set(NODUS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_executable(fabric_state_test fabric_state_test.cpp
  "${NODUS_ROOT}/src/fabric_spring_state_tool.cpp" "${NODUS_ROOT}/src/tool_registry.cpp"
  "${NODUS_ROOT}/src/tensor_state_library_tool.cpp" "${NODUS_ROOT}/src/gpu_shader_job_tool.cpp" "${NODUS_ROOT}/src/gpu_work_row_tool.cpp"
  "${NODUS_ROOT}/src/kernel_pipeline_tool.cpp" "${NODUS_ROOT}/src/cpu_fabric_dispatch_tool.cpp"
  "${NODUS_ROOT}/src/gpu_fabric_dispatch_tool.cpp")
target_include_directories(fabric_state_test PRIVATE "${NODUS_ROOT}" "${NODUS_ROOT}/include")
target_link_libraries(fabric_state_test PRIVATE "${NODUS_TENSOR_CORE_DIR}/nodus_tensor_core.lib" ws2_32)
target_compile_definitions(fabric_state_test PRIVATE NOMINMAX)
