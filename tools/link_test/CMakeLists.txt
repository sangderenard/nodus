# link_test: stage 1 of the Nodus link (include/nodus_link.h), research case 98d49cb8520e44d092c5d134c37c3c32.
# A project of its own so it builds in seconds without the rest of Nodus:
#   cmake -S tools/link_test -B build-link && cmake --build build-link --config Release
#   build-link/Release/link_test.exe
cmake_minimum_required(VERSION 3.20)
project(link_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(link_test link_test.cpp)
target_include_directories(link_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(link_test PRIVATE ws2_32 bcrypt)
if(MSVC)
  target_compile_options(link_test PRIVATE /W4)
  set_property(TARGET link_test PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()
