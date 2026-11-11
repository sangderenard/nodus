# Licensing

## Scope

The MIT grant in [LICENSE](LICENSE) applies to the first-party code and
documentation for which sangderenard holds licensing rights. Existing
copyright and permission notices must be retained. Third-party software,
submodules, dependencies, assets and independently authored material retain
their own terms. This grant does not claim ownership of those works or
supply missing permission for unverified material.

The owner selected this first-party MIT policy on 2026-11-11 to express the
projects' continuity with SpeakToMe. Repository history and author names were
checked as provenance evidence; they are not proof of ownership of every
imported file. This document is a scoped inventory, not a complete dependency
audit or a statement that every possible combined distribution is compatible.

## Preserved notices and dependencies

- `include/stb_easy_font.h` is Sean Barrett's third-party stb code.
  Its embedded full MIT/public-domain alternative terms remain unchanged.
  The root copyright line does not replace its attribution.
- `vcpkg.json` declares SDL3, FreeType, HarfBuzz and libpng dependencies.
  CMake also references Eigen, Torch, Vulkan, OpenGL and Python, depending
  on the selected build. Their terms remain separate.
- Packaging should collect the notices for the actual versions and options
  shipped. A dependency declaration is not a complete transitive notice
  inventory.

Nodus's canvas and contract-port runtime can be used by the other projects
without merging their repository identities or discarding component notices.
