if(NOT TARGET canvas_tables)
  message(FATAL_ERROR "auto_sources.cmake expected canvas_tables target to exist.")
endif()

file(GLOB_RECURSE NODUS_CANVAS_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
)

if(NODUS_CANVAS_SOURCES)
  # Runtime-only ABI sources have their own dependency-minimal targets and
  # must not be pulled back into the canvas amalgam by the broad legacy glob.
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]runtime[/\\\\]")
  target_sources(canvas_tables PRIVATE ${NODUS_CANVAS_SOURCES})
  if(TARGET canvas_tables_static)
    target_sources(canvas_tables_static PRIVATE ${NODUS_CANVAS_SOURCES})
  endif()
endif()
