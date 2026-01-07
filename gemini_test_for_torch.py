import os
import ctypes

verified_paths = [
    r"C:\Program Files\NVIDIA\CUDNN\v9.17\bin\13.1",
    r"C:\Program Files\NVIDIA cuDSS\v0.7\bin\13",
    r"C:\Program Files\NVIDIA cuSPARSELt\v0.8\bin\13",
    r"C:\Program Files (x86)\Intel\oneAPI\2025.3\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64",
    r"C:\pytorch\build\lib\Release",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\extras\CUPTI\lib64",
]

for p in verified_paths:
    if os.path.isdir(p):
        os.add_dll_directory(p)

print("Loading aoti_custom_ops.dll directly...")
ctypes.WinDLL(r"C:\pytorch\torch\lib\aoti_custom_ops.dll")
print("aoti_custom_ops.dll loaded OK")

print("Now importing torch...")
import torch
print("torch import OK", torch.__version__)
