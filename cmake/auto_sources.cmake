if(NOT TARGET canvas_tables)
  message(FATAL_ERROR "auto_sources.cmake expected canvas_tables target to exist.")
endif()

file(GLOB_RECURSE NODUS_CANVAS_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
)

if(NODUS_CANVAS_SOURCES)
  target_sources(canvas_tables PRIVATE ${NODUS_CANVAS_SOURCES})
endif()
