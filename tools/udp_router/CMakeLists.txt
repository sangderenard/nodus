# udp_router: the service's UDP router. A project of its own so that it builds in seconds and without the rest of
# Nodus:   cmake -S tools/udp_router -B build-router && cmake --build build-router --config Release
cmake_minimum_required(VERSION 3.20)
project(udp_router LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(udp_router udp_router.cpp)
target_link_libraries(udp_router PRIVATE ws2_32)
if(MSVC)
  target_compile_options(udp_router PRIVATE /W4)
  set_property(TARGET udp_router PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")   # no runtime DLL beside the executable
endif()
