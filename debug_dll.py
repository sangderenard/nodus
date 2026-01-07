import builtins
import glob
import os
import ctypes
import sys
import sysconfig

TORCH_INSTALL_BIN = r"C:\pytorch\build\bin\Release"

EXTERNAL_DLLS = [
    "cusparse64_12.dll",
    "cufft64_12.dll",
    "cusolver64_12.dll",
    "CUSPARSELT.dll",
    "cudss64_0.dll",
    "cudnn64_9.dll",
    "cublas64_13.dll",
    "cublasLt64_13.dll",
]

MKL_DLLS = [
    "mkl_rt.2.dll",
    "mkl_intel_thread.2.dll",
    "mkl_core.2.dll",
    "libiomp5md.dll",
]

def _get_env_path() -> str:
    raw = os.environ.get("Path") or os.environ.get("PATH") or ""
    if raw.upper().startswith("PATH="):
        raw = raw[5:]
    return raw


def describe_search_paths() -> list[str]:
    """Return the set of directories that torch would probe."""

    torch_root = os.path.dirname(os.path.dirname(__file__))
    torch_lib_dir = os.path.join(torch_root, "torch", "lib")
    search_paths = {
        TORCH_INSTALL_BIN,
        torch_lib_dir,
        os.path.join(sys.exec_prefix, "Library", "bin"),
        os.path.join(sys.exec_prefix, "bin"),
        os.path.join(sysconfig.get_config_var("userbase"), "Library", "bin"),
    }
    if sys.exec_prefix != sys.base_exec_prefix:
        search_paths.add(os.path.join(sys.base_exec_prefix, "Library", "bin"))

    cuda_version = ""
    try:
        from torch.version import cuda as cuda_version
    except Exception:
        cuda_version = ""
    if cuda_version:
        cuda_var = "CUDA_PATH_V" + cuda_version.replace(".", "_")
        default_path = os.path.join(
            os.getenv("ProgramFiles", r"C:\Program Files"),
            "NVIDIA GPU Computing Toolkit",
            "CUDA",
            f"v{cuda_version}",
        )
        search_paths.add(os.path.join(os.getenv(cuda_var, default_path), "bin"))

    nvtop = os.path.join(
        os.getenv("ProgramFiles", r"C:\Program Files"),
        "NVIDIA Corporation",
        "NvToolsExt",
        "bin",
        "x64",
    )
    if os.path.isdir(nvtop):
        search_paths.add(nvtop)

    raw_path = _get_env_path()
    path_dirs = [p.strip('"') for p in raw_path.split(os.pathsep) if p.strip('"')]
    print("\nDLL search path candidates:")
    for path in sorted(search_paths):
        if os.path.isdir(path):
            print("  ", path)
    print("\nPATH entries:")
    print(f"  (count: {len(path_dirs)})")
    for path in path_dirs:
        print("  ", path)
    return path_dirs


PATH_DIRS = describe_search_paths()

print("\n--- Probing Torch build binaries (PATH entries only) ---")
if os.path.exists(TORCH_INSTALL_BIN):
    bin_dlls = sorted(
        name for name in os.listdir(TORCH_INSTALL_BIN) if name.lower().endswith(".dll")
    )
else:
    print(f"MISSING build bin: {TORCH_INSTALL_BIN}")
    bin_dlls = []


def try_load(path: str) -> None:
    try:
        ctypes.CDLL(path)
        print(f"Attempting to load: {path}")
        print("  SUCCESS")
    except OSError:
        pass


def try_load_from_all_paths(dll_name: str) -> None:
    for directory in PATH_DIRS:
        candidate = os.path.join(directory, dll_name)
        exists = os.path.isfile(candidate)
        if exists:
            print(f"Path check: {candidate} -> FOUND")
    try_load(dll_name)


for dll_name in bin_dlls:
    try_load_from_all_paths(dll_name)

print("\n--- Probing external CUDA/CUDNN/CUSPARSELT/CUDSS DLLs via PATH entries ---")


def try_load_from_path(dll_name: str, check_all_paths: bool = False) -> None:
    for directory in PATH_DIRS:
        candidate = os.path.join(directory, dll_name)
        exists = os.path.isfile(candidate)
        if check_all_paths:
            if exists:
                print(f"Path check: {candidate} -> FOUND")
    try_load(dll_name)


for dll_name in EXTERNAL_DLLS:
    try_load_from_path(dll_name, check_all_paths=True)

print("\n--- Probing MKL-related DLLs via PATH entries ---")
for dll_name in MKL_DLLS:
    try_load_from_path(dll_name, check_all_paths=True)

print("\n--- Name-only PATH load check ---")
for dll_name in sorted(set(EXTERNAL_DLLS + MKL_DLLS + bin_dlls)):
    print(f"\nAttempting name-only load: {dll_name}")
    try:
        ctypes.CDLL(dll_name)
        print("  SUCCESS")
    except OSError:
        pass
