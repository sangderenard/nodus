# scripts

Small helpers for working with this repo on Windows.

## Build helpers

- `build_and_install.ps1`
  - Creates/uses a Python venv, runs a CMake configure/build, then installs any produced wheels.

- `wipe_torch_build.ps1`
  - Wipes the PyTorch-from-source CMake cache (and optionally the heavy `_deps` subtree) for a build directory like `build-torch-avx2`.
  - Useful when toggling flags like `USE_XPU`, `LIBKINETO_NOXPUPTI`, `XPU_ENABLE_KINETO`, etc.

Examples:

- Cache only:
  - `./scripts/wipe_torch_build.ps1`

- Cache + deps (recommended when switching Torch/XPU/Kineto options):
  - `./scripts/wipe_torch_build.ps1 -WipeDeps -Force`

- Entire build folder:
  - `./scripts/wipe_torch_build.ps1 -WipeAll -Force`

## Python env

- `create_venv.ps1`
  - Creates a repo venv.

- `install_wheels.ps1`
  - Installs wheels from `dev_bundle/wheels` into the repo venv.
