if(NOT TARGET canvas_tables)
  message(FATAL_ERROR "auto_sources.cmake expected canvas_tables target to exist.")
endif()

file(GLOB_RECURSE NODUS_CANVAS_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
)

if(NODUS_CANVAS_SOURCES)
  # Dedicated protected-host and CLI targets must not contribute extra process
  # owners, WGL plumbing or entry points to the canvas amalgam.
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]opengl_link_once\\.cpp$")
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]kernels[/\\\\]opengl_link_worker_main\\.cpp$")
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]kernels[/\\\\]opengl_protected_(process|child|wgl)\\.cpp$")
  # Optional inventory-to-row registration links the native protected provider
  # as its own target; do not fold it back into the canvas dependency layer.
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]gpu_inventory_rows\\.cpp$")
  # Runtime-only ABI sources have their own dependency-minimal targets and
  # must not be pulled back into the canvas amalgam by the broad legacy glob.
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]runtime[/\\\\]")
  # Headless graph-runtime ABI sources likewise get their own target
  # (nodus_headless) that links canvas_tables_static rather than being
  # folded back into the canvas amalgam itself.
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]headless[/\\\\]")
  # Interpreted Python row support links the CPython development library and
  # is built as its own shared plugin dependency below.  Folding it into the
  # legacy canvas amalgam would duplicate its interpreter/GIL state and make
  # every canvas consumer require Python headers.
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]python_row_tool\\.cpp$")
  # nodus_tensor_core sources: these hold the process's stateful tensor
  # singletons (backend registry, the InMemoryBackend instance, the default
  # tensor pool, GP_MemBackend handle maps). They must exist as exactly ONE
  # compiled copy shared by canvas_tables, canvas_tables_static-linked hosts,
  # and any plugin DLL -- duplicating them (the previous default via this
  # glob) meant a static-linked host and a canvas_tables.dll-linked plugin
  # each got their own registry/backend instance, so a tensor handle created
  # on one side silently failed to resolve on the other. See
  # research/12_substrate_blocker.md for the diagnosis.
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]common[/\\\\]tensors[/\\\\]abstraction[/\\\\](tensor_registry|in_memory_backend|tensor_abi|abstract_tensor|abstract_tensor_pool|tensor_math|tensor_calculator_bridge)\\.cpp$")
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]common[/\\\\]thread_pool\\.cpp$")
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]mem_backend_(host|opengl)\\.cpp$")
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]spirv_translation\\.cpp$")
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]kernels[/\\\\](kernel_spirv|kernel_ir_text|kernel_vulkan|kernel_opengl|kernel_eigen)\\.cpp$")
  list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\](vocabulary_actualizer|artifact_importer|generic_object)\\.cpp$")
  # kernel_torch.cpp and mem_backend_torch.cpp include <torch/torch.h> unconditionally.
  # A NODUS_ENABLE_TORCH=OFF image (the Torch-free service build) leaves them out;
  # the Torch build keeps them. Both configurations must build.
  if(NOT NODUS_ENABLE_TORCH)
    list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]kernels[/\\\\]kernel_torch\\.cpp$")
    list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]mem_backend_torch\\.cpp$")
  endif()
  target_sources(canvas_tables PRIVATE ${NODUS_CANVAS_SOURCES})
  if(TARGET canvas_tables_static)
    target_sources(canvas_tables_static PRIVATE ${NODUS_CANVAS_SOURCES})
  endif()
endif()
