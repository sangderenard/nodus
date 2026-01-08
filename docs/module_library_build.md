# Module Library Build (Agent + Human Notes)

This repo has two distinct but related build flows:

1) Repo build: builds `canvas_tables`, `modlib_actualize`, `modloader`, and other app binaries.
2) Module library build: generates tool/module sources from serialized `.gpmod` files and compiles tool DLLs.

These must stay separate. The module library build is a nested build rooted in `module_library/`
with its own outputs. Do not mix its output paths with the repo build output.

## What lives where

- `module_library/serialized/`
  - Source-of-truth `.gpmod` assets.
- `module_library/source/tools/`
  - Generated C++ tool sources from serialized modules.
- `module_library/source/modules/`
  - Generated module sources (if any).
- `module_library/build/`
  - Dedicated build root for compiled tool DLLs (nested build output).
- `module_library/test_build/`
  - Dedicated build root for tool DLL test outputs.
- `module_library/include/`
  - Future-proof headers for module library public APIs or shared tool headers.

## Canonical flow (real, not test-only)

1) Actualize sources from serialized modules:
   - C++ entrypoint: `gp_module_library_actualize_from_file(...)`
   - CLI tool: `modlib_actualize <module_library.txt> <module_library_root>`
2) Compile generated tool sources into DLLs:
   - Should occur under `module_library/build/` (not the repo `build/`).
   - Tests should compile into `module_library/test_build/` to keep artifacts isolated.
3) Load built tool DLLs:
   - Runtime loader: `PluginLoader` (via `modloader` or in-app wiring).

## Why this exists

The module library is essentially a nested subproject:
- `.gpmod` files define module composition.
- Actualization generates C++ sources.
- Those sources become dynamically loaded DLL tools.

This is intentional and should be treated like a submodule build in a monorepo.

## Test flow (tool DLL test)

The test should only:
- Use the actualizer to generate sources into `module_library/test_build/` (not `module_library/source/`).
- Build tool DLLs in a nested build root under `module_library/test_build/`.
- Load them with `PluginLoader` and exercise basic lifecycle methods.

Avoid Python harnesses or ad-hoc temp build paths that blur the repo vs module build boundary.

## Guardrails

Do:
- Keep module library build output under `module_library/build/`.
- Keep generated sources under `module_library/source/...`.
- Keep repo build output under `build/`.

Do not:
- Write nested build output into repo `build/`.
- Add global CMake output overrides in the repo build.
- Use OS-specific scripts for module library build orchestration.

## Relevant code

- Actualizer: `src/module_library_actualizer.cpp`
- Actualizer CLI: `tools/modlib_actualize_main.cpp`
- Loader: `src/plugin_loader.cpp`
- Loader CLI: `tools/modloader_main.cpp`
- Module library data: `include/module_library.h` (serialization layout)

## Agent note

When you need to build tools or load generated DLLs:
- Treat `module_library/` as its own build "home".
- Keep repo build and module library build separate.
- Prefer CMake-native or C++ orchestration, not OS-specific scripts.
