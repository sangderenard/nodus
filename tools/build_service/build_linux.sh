#!/usr/bin/env bash
# Linux half of the Nodus compiler service (tools/build_service/nodus_build.py). Runs as root inside the service
# account's own WSL2 distro (nodus-build); the first run installs the toolchain and vcpkg there, later runs reuse them.
#   build_linux.sh <source.tar> <job key> <product.tar.gz> <vcpkg triplet>
# Paths are Linux paths (the Windows driver passes /mnt/c/...). The source never comes from git: it is the archive the
# request carried. Nothing here runs git on a core repository; vcpkg (a third-party checkout) is fetched once.
set -euo pipefail
SRC_TAR="$1"; KEY="$2"; OUT="$3"; TRIPLET="${4:-x64-linux}"
STATE=/var/lib/nodus-build
mkdir -p "$STATE"
export DEBIAN_FRONTEND=noninteractive

if [ ! -f "$STATE/apt.v2.done" ]; then   # bump the marker when the package list changes
  apt-get update -q
  apt-get install -y -q build-essential cmake ninja-build pkg-config git curl zip unzip tar python3 python3-venv python3-pip \
    autoconf autoconf-archive automake libtool bison flex \
    libssl-dev libeigen3-dev libvulkan-dev glslang-tools \
    libx11-dev libxext-dev libxrandr-dev libxcursor-dev libxi-dev libxss-dev libxfixes-dev libxtst-dev \
    libxkbcommon-dev libwayland-dev wayland-protocols libdecor-0-dev \
    libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev libdrm-dev libgbm-dev \
    libasound2-dev libpulse-dev libpipewire-0.3-dev libudev-dev libdbus-1-dev libibus-1.0-dev
  touch "$STATE/apt.v2.done"
fi
if [ ! -x /opt/vcpkg/vcpkg ]; then
  rm -rf /opt/vcpkg
  git clone --depth 1 https://github.com/microsoft/vcpkg /opt/vcpkg
  /opt/vcpkg/bootstrap-vcpkg.sh -disableMetrics
fi

WORK="$STATE/work/$KEY"
rm -rf "$WORK"; mkdir -p "$WORK/src" "$WORK/product"
tar -xf "$SRC_TAR" -C "$WORK/src"
export VCPKG_DEFAULT_BINARY_CACHE="$STATE/vcpkg-binary-cache"; mkdir -p "$VCPKG_DEFAULT_BINARY_CACHE"
cmake -S "$WORK/src" -B "$WORK/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET="$TRIPLET" \
  -DNODUS_ENABLE_TORCH=OFF
cmake --build "$WORK/build" --target nodus_headless frontend_shell

# The product: the native libraries the headless graph loads, the native host (frontend_shell --canvas, run with
# --offscreen where there is no display; docs/NODUS_NATIVE_HOST_PLAN_2026-11-23.md), and, until every host has moved to
# it, the Python side.
find "$WORK/build" -maxdepth 2 -name '*.so*' -exec cp -a {} "$WORK/product/" \;
cp -a "$WORK/build/frontend_shell" "$WORK/product/"
cp -a "$WORK/src/nodus_graph" "$WORK/src/nodus_headless_bridge.py" "$WORK/product/" 2>/dev/null || true
mkdir -p "$WORK/product/tools" && cp -a "$WORK/src/tools/run_canvas.py" "$WORK/product/tools/" 2>/dev/null || true
printf '{"triple": "%s", "key": "%s", "compiler": "%s"}\n' "$TRIPLET" "$KEY" "$(g++ --version | head -1)" > "$WORK/product/BUILD.json"
tar -czf "$OUT.tmp" -C "$WORK/product" .
mv "$OUT.tmp" "$OUT"
