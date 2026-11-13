# Pass with -DCMAKE_PROJECT_INCLUDE=<source>/tools/profiling/nodus_profile_targets.cmake
# to keep optional profiling targets out of the default application build.
add_executable(nodus_job_timing_profile
  "${CMAKE_SOURCE_DIR}/tools/profiling/nodus_job_timing_profile.cpp")
target_include_directories(nodus_job_timing_profile PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(nodus_job_timing_profile PRIVATE canvas_tables_static)
target_compile_features(nodus_job_timing_profile PRIVATE cxx_std_20)

add_executable(nodus_type_registry_profile
  "${CMAKE_SOURCE_DIR}/tools/profiling/type_registry_profile.cpp")
target_include_directories(nodus_type_registry_profile PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(nodus_type_registry_profile PRIVATE canvas_tables_static)
target_compile_features(nodus_type_registry_profile PRIVATE cxx_std_20)
