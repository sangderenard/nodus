# Auto-generated source gatherer for nodus
# Gathers top-level .cpp from src/ to avoid having sources at repo root.
file(GLOB_RECURSE NODUS_PROJECT_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cpp"
)
# Ensure include dir is added
if (TARGET canvas_tables)
    target_include_directories(canvas_tables PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/inl
    )
    # Exclude legacy or intentionally-ignored compilation units
    list(FILTER NODUS_PROJECT_SRC EXCLUDE REGEX ".*/mem_backend_cpu\\.cpp$")
    if(NOT NODUS_ENABLE_TORCH)
        # Drop Torch-dependent translation units when Torch is disabled
        list(FILTER NODUS_PROJECT_SRC EXCLUDE REGEX ".*/mem_backend_torch\\.cpp$")
        list(FILTER NODUS_PROJECT_SRC EXCLUDE REGEX ".*/kernel_torch\\.cpp$")
        list(FILTER NODUS_PROJECT_SRC EXCLUDE REGEX ".*/sdlttf_shim\\.cpp$")
    endif()
    target_sources(canvas_tables PRIVATE ${NODUS_PROJECT_SRC})
endif()
