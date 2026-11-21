# link_settings_test: compiler settings and shader builds over the Nodus link (volunteer case
# a11b5c9b1be74c9b9fb684790f7fd5a3, stage 2). Its own project against an existing build's tensor core (the Vulkan
# inventory):   cmake -S tools/link_settings_test -B build-settings -DNODUS_TENSOR_CORE_DIR=<a Nodus build>/Release
cmake_minimum_required(VERSION 3.20)
project(link_settings_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(NODUS_TENSOR_CORE_DIR "" CACHE PATH "Folder that holds nodus_tensor_core.lib and nodus_tensor_core.dll")
add_executable(link_settings_test link_settings_test.cpp)
target_include_directories(link_settings_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(link_settings_test PRIVATE "${NODUS_TENSOR_CORE_DIR}/nodus_tensor_core.lib" ws2_32 bcrypt)
target_compile_definitions(link_settings_test PRIVATE NOMINMAX)
