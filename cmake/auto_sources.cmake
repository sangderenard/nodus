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
  # Headless graph-runtime ABI sources likewise get their own target
  # (nodus_headless) that links canvas_tables_static rather than being
  # folded back into the canvas amalgam itself.
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]headless[/\\\\]")
  # nodus_tensor_core sources: these hold the process's stateful tensor
  # singletons (backend registry, the InMemoryBackend instance, the default
  # tensor pool, GP_MemBackend handle maps). They must exist as exactly ONE
  # compiled copy shared by canvas_tables, canvas_tables_static-linked hosts,
  # and any plugin DLL -- duplicating them (the previous default via this
  # glob) meant a static-linked host and a canvas_tables.dll-linked plugin
  # each got their own registry/backend instance, so a tensor handle created
  # on one side silently failed to resolve on the other. See
  # research/12_substrate_blocker.md for the diagnosis.
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]common[/\\\\]tensors[/\\\\]abstraction[/\\\\](tensor_registry|in_memory_backend|abstract_tensor|abstract_tensor_pool|tensor_math)\\.cpp$")
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]common[/\\\\]thread_pool\\.cpp$")
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]mem_backend_host\\.cpp$")
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]spirv_translation\\.cpp$")
  target_sources(canvas_tables PRIVATE ${NODUS_CANVAS_SOURCES})
  if(TARGET canvas_tables_static)
    target_sources(canvas_tables_static PRIVATE ${NODUS_CANVAS_SOURCES})
  endif()
endif()
