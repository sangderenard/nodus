# link_fabric_worker: the volunteer Nodus GPU worker (research case a11b5c9b1be74c9b9fb684790f7fd5a3), built by itself
# against the tensor core of an existing Nodus build: no Torch, no canvas library, static MSVC runtime. It ships as
# link_fabric_worker.exe + nodus_tensor_core.dll; vulkan-1.dll comes with the GPU driver.
#   cmake -S tools/link_fabric_worker -B build-worker -DNODUS_TENSOR_CORE_DIR=<a Nodus build>/Release
#   cmake --build build-worker --config Release
cmake_minimum_required(VERSION 3.20)
project(link_fabric_worker LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(NODUS_TENSOR_CORE_DIR "" CACHE PATH "Folder that holds nodus_tensor_core.lib and nodus_tensor_core.dll")
if(NOT EXISTS "${NODUS_TENSOR_CORE_DIR}/nodus_tensor_core.lib")
  message(FATAL_ERROR "NODUS_TENSOR_CORE_DIR must hold nodus_tensor_core.lib (got '${NODUS_TENSOR_CORE_DIR}')")
endif()
set(NODUS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(WORKER_SOURCES ${WORKER_SOURCES_OVERRIDE})
if(NOT WORKER_SOURCES)
  set(WORKER_SOURCES "${NODUS_ROOT}/tools/link_gpu_exchange/link_gpu_exchange.cpp")
endif()
add_executable(link_fabric_worker ${WORKER_SOURCES}
  "${NODUS_ROOT}/src/nodus_link_rows.cpp" "${NODUS_ROOT}/src/tool_registry.cpp"
  "${NODUS_ROOT}/src/tensor_state_library_tool.cpp" "${NODUS_ROOT}/src/gpu_fabric_dispatch_tool.cpp")
target_include_directories(link_fabric_worker PRIVATE "${NODUS_ROOT}" "${NODUS_ROOT}/include")
target_link_libraries(link_fabric_worker PRIVATE "${NODUS_TENSOR_CORE_DIR}/nodus_tensor_core.lib" ws2_32 bcrypt winhttp crypt32)
target_compile_definitions(link_fabric_worker PRIVATE NOMINMAX)
if(MSVC)
  set_property(TARGET link_fabric_worker PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()
