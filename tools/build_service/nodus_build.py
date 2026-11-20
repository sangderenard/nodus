"""Nodus compiler service: one exec row builds Nodus for a target triple with the system's own toolchain.

Owner, 2026-11-19: "let's just make the compiler service and nothing more, you put in the pipeline and we use it as an
exec call to get linux products". Not the Turing compiler: the native toolchain, straight to the binaries.

It is an exec row's program (nodus include/exec_row.h; declared in turing examples/meowdoku/deploy/exec_rows.txt): the
row hands it one request on standard input and takes the product from standard output.

  request   b"NBR1" | header length u32 | header JSON | source archive (tar of the Nodus tree; never fetched with git here)
            header: {"triple": "x64-linux" | "x64-windows"}
  product   a tar.gz: the native libraries the headless graph loads, nodus_graph, tools/run_canvas.py, BUILD.json

Toolchains: x64-windows is CMake + MSVC with vcpkg's x64-windows-static-md triplet on this machine; x64-linux is the same
CMake project inside the service account's own WSL2 distro (NODUS_BUILD_DISTRO, default nodus-build) with gcc and vcpkg's
x64-linux triplet (build_linux.sh, which installs the toolchain there on its first run).

Every product is cached by the digest of the source, the header and this driver's version: the same request returns the
cached archive without building. Work, cache and logs are in the working directory the row declares (outside the core).
A failed build writes nothing to standard output and exits 1; its log is <work>/jobs/<key>/build.log.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tarfile
from pathlib import Path

VERSION = 1
MAGIC = b"NBR1"
TRIPLETS = {"x64-linux": "x64-linux", "x64-windows": "x64-windows-static-md"}
HERE = Path(__file__).resolve().parent


def parse(request: bytes) -> tuple[dict, bytes]:
    if len(request) < 8 or request[:4] != MAGIC:
        raise ValueError("not a build request (NBR1)")
    (n,) = struct.unpack_from("<I", request, 4)
    header = json.loads(request[8:8 + n].decode("utf-8"))
    if header.get("triple") not in TRIPLETS:
        raise ValueError(f"triple is one of {sorted(TRIPLETS)}")
    return header, request[8 + n:]


def key_of(header: dict, source: bytes) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"driver": VERSION, **header}, sort_keys=True).encode("utf-8"))
    h.update(hashlib.sha256(source).digest())
    return h.hexdigest()[:32]


def wsl_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    return f"/mnt/{s[0].lower()}{s[2:]}" if len(s) > 1 and s[1] == ":" else s


def build_linux(job: Path, key: str, triplet: str, log) -> Path:
    out = job / "product.tar.gz"
    distro = os.environ.get("NODUS_BUILD_DISTRO", "nodus-build")
    script = wsl_path(HERE / "build_linux.sh")
    command = ["wsl.exe", "-d", distro, "-u", "root", "--", "bash", script,
               wsl_path(job / "source.tar"), key, wsl_path(out), triplet]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(f"the Linux build failed (exit {result.returncode})")
    return out


def build_windows(job: Path, key: str, triplet: str, log) -> Path:
    src, build, product = job / "src", job / "build", job / "product"
    for d in (src, product):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    with tarfile.open(job / "source.tar") as tar:
        tar.extractall(src, filter="data")
    env = dict(os.environ)
    env.setdefault("VCPKG_DEFAULT_BINARY_CACHE", str(job.parent.parent / "vcpkg-binary-cache"))
    Path(env["VCPKG_DEFAULT_BINARY_CACHE"]).mkdir(parents=True, exist_ok=True)
    vcpkg = os.environ.get("NODUS_BUILD_VCPKG", "C:/vcpkg")
    for step in (["cmake", "-S", str(src), "-B", str(build), "-G", "Visual Studio 17 2022",
                  f"-DCMAKE_TOOLCHAIN_FILE={vcpkg}/scripts/buildsystems/vcpkg.cmake", f"-DVCPKG_TARGET_TRIPLET={triplet}",
                  "-DNODUS_ENABLE_TORCH=OFF"],
                 ["cmake", "--build", str(build), "--config", "Release", "--target", "nodus_headless"]):
        if subprocess.run(step, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env).returncode != 0:
            raise RuntimeError(f"the Windows build failed at: {' '.join(step[:3])}")
    for f in (build / "Release").glob("*.dll"):
        shutil.copy2(f, product)
    for name in ("nodus_graph", "nodus_headless_bridge.py"):
        p = src / name
        if p.is_dir():
            shutil.copytree(p, product / name)
        elif p.exists():
            shutil.copy2(p, product)
    (product / "tools").mkdir(exist_ok=True)
    shutil.copy2(src / "tools" / "run_canvas.py", product / "tools")
    (product / "BUILD.json").write_text(json.dumps({"triple": "x64-windows", "key": key}) + "\n", encoding="utf-8")
    out = job / "product.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(product, arcname=".")
    return out


def run(request: bytes, work: Path) -> bytes:
    header, source = parse(request)
    key = key_of(header, source)
    cache = work / "cache" / f"{key}.tar.gz"
    if cache.exists():
        return cache.read_bytes()
    job = work / "jobs" / key
    job.mkdir(parents=True, exist_ok=True)
    (job / "source.tar").write_bytes(source)
    triplet = TRIPLETS[header["triple"]]
    with open(job / "build.log", "wb") as log:
        product = (build_linux if header["triple"].endswith("linux") else build_windows)(job, key, triplet, log)
    cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(product, cache)
    return cache.read_bytes()


def main() -> int:
    request = sys.stdin.buffer.read()
    try:
        product = run(request, Path.cwd())
    except Exception as error:  # noqa: BLE001 - the row only sees the exit code; the reason goes to the work folder
        (Path.cwd() / "last_error.txt").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        return 1
    sys.stdout.buffer.write(product)
    sys.stdout.buffer.flush()
    return 0


def request(triple: str, source_tar: bytes) -> bytes:
    """A build request (for callers and tests)."""
    header = json.dumps({"triple": triple}).encode("utf-8")
    return MAGIC + struct.pack("<I", len(header)) + header + source_tar


if __name__ == "__main__":
    sys.exit(main())
