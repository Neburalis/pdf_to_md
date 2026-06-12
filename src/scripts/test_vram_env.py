"""
Minimal VRAM state check: does GGML_CUDA_NO_GRAPHS=1 fragment VRAM before model loading?
Run as:  python src/scripts/test_vram_env.py [with_env|without_env]
"""
import sys, os

mode = sys.argv[1] if len(sys.argv) > 1 else "with_env"
if mode == "with_env":
    os.environ["GGML_CUDA_NO_GRAPHS"] = "1"
    print("GGML_CUDA_NO_GRAPHS=1 is SET")
else:
    print("GGML_CUDA_NO_GRAPHS is NOT set")

import ctypes, site, pathlib

# Load DLLs the same way vision.py does
dll_dirs = []
for sp in site.getsitepackages():
    base = pathlib.Path(sp)
    nvidia_dir = base / "nvidia"
    if nvidia_dir.is_dir():
        for pkg in nvidia_dir.iterdir():
            for subdir in ("bin", "lib"):
                d = pkg / subdir
                if d.is_dir():
                    dll_dirs.append(d)
    llama_lib = base / "llama_cpp" / "lib"
    if llama_lib.is_dir():
        dll_dirs.append(llama_lib)

if dll_dirs:
    for d in dll_dirs:
        os.add_dll_directory(str(d))
    dll_map = {p.name: p for d in dll_dirs for p in d.glob("*.dll")}
    for name in ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll",
                 "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "ggml.dll"]:
        if name in dll_map:
            try:
                ctypes.CDLL(str(dll_map[name]))
                print(f"  Loaded {name}")
            except OSError as e:
                print(f"  Failed {name}: {e}")

# Now trigger CUDA device init by calling cudaMemGetInfo
cudart = ctypes.CDLL("cudart64_12.dll")
cudart.cudaMemGetInfo.restype = ctypes.c_int
cudart.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
cudart.cudaMalloc.restype = ctypes.c_int
cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cudart.cudaFree.restype = ctypes.c_int
cudart.cudaFree.argtypes = [ctypes.c_void_p]
cudart.cudaDeviceSynchronize.restype = ctypes.c_int

# Trigger CUDA context creation
ret = cudart.cudaDeviceSynchronize()
print(f"\ncudaDeviceSynchronize ret={ret}")

free = ctypes.c_size_t()
total = ctypes.c_size_t()
cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
free_mib = free.value / (1024 * 1024)
total_mib = total.value / (1024 * 1024)
print(f"VRAM after CUDA init: free={free_mib:.1f} MiB / {total_mib:.1f} MiB")

# Binary search max contiguous
lo, hi = 1, 8192
while lo < hi:
    mid = (lo + hi + 1) // 2
    ptr = ctypes.c_void_p()
    ret = cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(mid * 1024 * 1024))
    if ret == 0:
        cudart.cudaFree(ptr)
        lo = mid
    else:
        hi = mid - 1
print(f"Max contiguous block: {lo} MiB")
print(f"Model weights need:  4168 MiB  -> {'FITS' if lo >= 4168 else 'FAILS'}")
