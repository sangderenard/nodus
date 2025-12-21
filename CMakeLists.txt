cmake_minimum_required(VERSION 3.15)
project(canvas_tables LANGUAGES CXX)

if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

add_library(canvas_tables SHARED
    canvas_abi.cpp
    module_preview.cpp
    table_abi.cpp
    table_node_groups.cpp
    menu_waveform.cpp
    menu_waveform_abi.cpp
    rope_sim.cpp
    raytrace_2d.cpp
    raytrace_3d.cpp
    stage_abi.cpp
    subject_capture.cpp
    text_render_helper.cpp
    thread_manager.cpp
)

find_package(Eigen3 REQUIRED)

target_include_directories(canvas_tables PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(canvas_tables PUBLIC Eigen3::Eigen)
target_compile_features(canvas_tables PUBLIC cxx_std_17)

enable_testing()

# Unit tests
add_executable(test_thread_manager
    tests/thread_manager_test.cpp
)
target_link_libraries(test_thread_manager PRIVATE canvas_tables)
add_test(NAME thread_manager_test COMMAND test_thread_manager)

# Scaffold test target
# scaffold_test removed

option(ENABLE_EIGEN_SIM "Enable Eigen-accelerated rope simulation" OFF)
if(ENABLE_EIGEN_SIM)
    target_compile_definitions(canvas_tables PUBLIC EIGEN_SIM)
endif()
