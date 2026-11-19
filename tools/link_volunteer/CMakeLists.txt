# link_volunteer: the volunteer as the whole system (volunteer case a11b5c9b1be74c9b9fb684790f7fd5a3, stage 3).
# A Torch-free build: the link, the contract market, the tensor state library and every library job row, against an
# existing build's tensor core and runtime:
#   cmake -S tools/link_volunteer -B build-volunteer -DNODUS_BUILD_DIR=<a Nodus build>/Release
cmake_minimum_required(VERSION 3.20)
project(link_volunteer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(NODUS_BUILD_DIR "" CACHE PATH "Folder that holds nodus_tensor_core.lib and nodus_runtime.lib")
if(NOT EXISTS "${NODUS_BUILD_DIR}/nodus_tensor_core.lib" OR NOT EXISTS "${NODUS_BUILD_DIR}/nodus_runtime.lib")
  message(FATAL_ERROR "NODUS_BUILD_DIR must hold nodus_tensor_core.lib and nodus_runtime.lib (got '${NODUS_BUILD_DIR}')")
endif()
set(NODUS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_executable(link_volunteer link_volunteer.cpp
  "${NODUS_ROOT}/src/tool_registry.cpp" "${NODUS_ROOT}/src/tensor_state_library_tool.cpp"
  "${NODUS_ROOT}/src/gpu_shader_job_tool.cpp" "${NODUS_ROOT}/src/kernel_pipeline_tool.cpp"
  "${NODUS_ROOT}/src/cpu_fabric_dispatch_tool.cpp" "${NODUS_ROOT}/src/gpu_fabric_dispatch_tool.cpp"
  "${NODUS_ROOT}/src/fftfree_tool.cpp" "${NODUS_ROOT}/src/wave_t4_state_tool.cpp"
  "${NODUS_ROOT}/src/fabric_spring_state_tool.cpp")
target_include_directories(link_volunteer PRIVATE "${NODUS_ROOT}" "${NODUS_ROOT}/include")
target_link_libraries(link_volunteer PRIVATE "${NODUS_BUILD_DIR}/nodus_tensor_core.lib" "${NODUS_BUILD_DIR}/nodus_runtime.lib"
  ws2_32 bcrypt winhttp crypt32)
target_compile_definitions(link_volunteer PRIVATE NOMINMAX)
if(MSVC)
  set_property(TARGET link_volunteer PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()
