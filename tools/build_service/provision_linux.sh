#!/usr/bin/env bash
# The machine half of a Linux Nodus host: packages, vcpkg and the service account, the same everywhere — the server's
# WSL2 build distro (nodus-build) and an Oracle Cloud VM. The workspace half (the repositories in their layout, the
# builds) is the outer repository's bootstrap_linux.sh, which calls this. Idempotent: a second run only does what the
# first did not.
#
#   sudo bash provision_linux.sh [--user NAME]
#
# Ubuntu 22.04/24.04 and Debian use apt (tested: Ubuntu 24.04 in WSL2). Oracle Linux 8/9 and RHEL-family use dnf with
# CodeReady Builder and EPEL (package names mapped; not yet run on an Oracle Linux machine).
# The vcpkg triplet follows the machine and is written to /etc/nodus/triplet: x86_64 -> x64-linux,
# aarch64 (Oracle Ampere A1) -> arm64-linux.
set -euo pipefail

USER_NAME=nodus
while [ $# -gt 0 ]; do
  case "$1" in
    --user) USER_NAME="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ "$(id -u)" -eq 0 ] || { echo "run as root (sudo)" >&2; exit 2; }

MARKERS=/var/lib/nodus-provision
mkdir -p "$MARKERS" /etc/nodus
. /etc/os-release

# ---- packages (bump the marker name when a list changes) ----
if [ ! -f "$MARKERS/packages.v3" ]; then
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -q
    apt-get install -y -q build-essential cmake ninja-build pkg-config git curl zip unzip tar ca-certificates \
      python3 python3-venv python3-pip python3-dev \
      autoconf autoconf-archive automake libtool bison flex \
      libssl-dev libeigen3-dev libvulkan-dev glslang-tools \
      libx11-dev libxext-dev libxrandr-dev libxcursor-dev libxi-dev libxss-dev libxfixes-dev libxtst-dev \
      libxkbcommon-dev libwayland-dev wayland-protocols libdecor-0-dev \
      libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev libdrm-dev libgbm-dev \
      libasound2-dev libpulse-dev libpipewire-0.3-dev libudev-dev libdbus-1-dev libibus-1.0-dev
  elif command -v dnf >/dev/null 2>&1; then
    major="${VERSION_ID%%.*}"
    if [ "${ID:-}" = "ol" ]; then
      dnf install -y -q "oracle-epel-release-el${major}" || true
      dnf config-manager --set-enabled "ol${major}_codeready_builder" || true
    else
      dnf install -y -q epel-release || true
      dnf config-manager --set-enabled crb || dnf config-manager --set-enabled powertools || true
    fi
    dnf install -y -q gcc gcc-c++ make cmake ninja-build pkgconf-pkg-config git curl zip unzip tar ca-certificates perl \
      python3 python3-pip python3-devel \
      autoconf automake libtool bison flex \
      openssl-devel eigen3-devel vulkan-loader-devel glslang \
      libX11-devel libXext-devel libXrandr-devel libXcursor-devel libXi-devel libXScrnSaver-devel libXfixes-devel libXtst-devel \
      libxkbcommon-devel wayland-devel wayland-protocols-devel libdecor-devel \
      mesa-libEGL-devel mesa-libGL-devel mesa-libGLES-devel libdrm-devel mesa-libgbm-devel \
      alsa-lib-devel pulseaudio-libs-devel pipewire-devel systemd-devel dbus-devel ibus-devel
  else
    echo "no apt-get or dnf: unsupported distribution ${PRETTY_NAME:-unknown}" >&2; exit 2
  fi
  touch "$MARKERS/packages.v3"
fi

# ---- the service account ----
if ! id "$USER_NAME" >/dev/null 2>&1; then
  useradd --system --create-home --home-dir "/home/$USER_NAME" --shell /bin/bash "$USER_NAME"
fi

# ---- vcpkg (owned by the service account: it writes its buildtrees, downloads and packages beside itself) ----
if [ ! -x /opt/vcpkg/vcpkg ]; then
  rm -rf /opt/vcpkg
  git clone --depth 1 https://github.com/microsoft/vcpkg /opt/vcpkg
  case "$(uname -m)" in aarch64|arm64) export VCPKG_FORCE_SYSTEM_BINARIES=1 ;; esac   # vcpkg's own tool downloads are x86
  /opt/vcpkg/bootstrap-vcpkg.sh -disableMetrics
fi
chown -R "$USER_NAME:$USER_NAME" /opt/vcpkg

case "$(uname -m)" in
  x86_64) echo x64-linux > /etc/nodus/triplet ;;
  aarch64|arm64) echo arm64-linux > /etc/nodus/triplet ;;
  *) echo "unsupported machine $(uname -m)" >&2; exit 2 ;;
esac
echo "machine ready: ${PRETTY_NAME:-linux}, $(uname -m), triplet $(cat /etc/nodus/triplet), user $USER_NAME"
