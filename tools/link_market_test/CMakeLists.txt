# link_market_test: the contract-negotiator market over the Nodus link (volunteer case a11b5c9b1be74c9b9fb684790f7fd5a3,
# stage 1). Its own project; the negotiator's result FIFO comes from an existing build's nodus_runtime:
#   cmake -S tools/link_market_test -B build-market -DNODUS_RUNTIME_DIR=<a Nodus build>/Release
cmake_minimum_required(VERSION 3.20)
project(link_market_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(NODUS_RUNTIME_DIR "" CACHE PATH "Folder that holds nodus_runtime.lib and nodus_runtime.dll")
if(NOT EXISTS "${NODUS_RUNTIME_DIR}/nodus_runtime.lib")
  message(FATAL_ERROR "NODUS_RUNTIME_DIR must hold nodus_runtime.lib (got '${NODUS_RUNTIME_DIR}')")
endif()
add_executable(link_market_test link_market_test.cpp)
target_include_directories(link_market_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(link_market_test PRIVATE "${NODUS_RUNTIME_DIR}/nodus_runtime.lib" ws2_32 bcrypt)
target_compile_definitions(link_market_test PRIVATE NOMINMAX)
if(MSVC)
  target_compile_options(link_market_test PRIVATE /W4)
endif()
