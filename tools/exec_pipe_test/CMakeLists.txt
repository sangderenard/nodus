# exec_pipe_test: the pipe exchange an exec tool row uses, driven against a real native compressor. A project of its
# own so it builds in seconds without the rest of Nodus:
#   cmake -S tools/exec_pipe_test -B build-exec-pipe && cmake --build build-exec-pipe --config Release
#   build-exec-pipe/Release/exec_pipe_test.exe <compressor> <scratch-dir>
cmake_minimum_required(VERSION 3.20)
project(exec_pipe_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(exec_pipe_test exec_pipe_test.cpp)
target_include_directories(exec_pipe_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
if(MSVC)
  target_compile_options(exec_pipe_test PRIVATE /W4)
  set_property(TARGET exec_pipe_test PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()
