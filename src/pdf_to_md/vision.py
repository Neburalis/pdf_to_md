import os

import base64
import gc
import sys
from pathlib import Path
from typing import NamedTuple

import requests

from . import logger as _logger_mod


class VisionResult(NamedTuple):
    text: str
    tokens_in: int
    tokens_out: int


def _register_cuda_dll_dirs() -> None:
    import ctypes
    import site

    dll_dirs: list[Path] = []
    for sp in site.getsitepackages():
        base = Path(sp)
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

    if not dll_dirs:
        return

    if hasattr(os, "add_dll_directory"):
        for d in dll_dirs:
            os.add_dll_directory(str(d))

    load_order = [
        "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll",
        "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "ggml.dll",
    ]
    dll_map = {p.name: p for d in dll_dirs for p in d.glob("*.dll")}
    for name in load_order:
        if name in dll_map:
            try:
                ctypes.CDLL(str(dll_map[name]))
            except OSError:
                pass


_PAGE_PROMPT = """Convert this PDF page image to Markdown.

Rules:
- Use # ## ### for headings based on visual size and weight
- Preserve lists (- or 1. or а), б), в)) exactly as structured in the image
- Preserve tables using Markdown table syntax (| col | col |)
- Preserve bold (**text**) and italic (*text*) formatting
- For mathematical formulas use LaTeX $ delimiters ONLY (NEVER \\( \\) or \\[ \\]): inline $formula$, display block $$formula$$
- Matrices inside display blocks: $$\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}$$
- Determinants inside display blocks: $$\\begin{vmatrix} a & b \\\\ c & d \\end{vmatrix}$$
- Numbered equations (right-side number N in the source): $$formula \\quad (N)$$
- For figures and diagrams write [Рисунок N] using the actual figure number from the text — do NOT generate image URLs
- Do NOT wrap the output in code fences
- Do NOT repeat the same content block multiple times or at different heading levels
- If the page contains only a page number or is blank, output an empty string
- Output ONLY the Markdown content, no explanations"""

_vision_llama = None
_vision_llama_model_path = None


def _log_vram_state(log, label: str = "") -> None:
    """Log free VRAM and max allocatable contiguous block via binary-search cudaMalloc."""
    import ctypes
    try:
        cudart = ctypes.CDLL("cudart64_12.dll")
        cudart.cudaMemGetInfo.restype = ctypes.c_int
        cudart.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        cudart.cudaMalloc.restype = ctypes.c_int
        cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        cudart.cudaFree.restype = ctypes.c_int
        cudart.cudaFree.argtypes = [ctypes.c_void_p]
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        free_mib = free.value / (1024 * 1024)
        total_mib = total.value / (1024 * 1024)
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
        tag = f" [{label}]" if label else ""
        log.info(f"VRAM{tag}: free={free_mib:.0f} MiB / {total_mib:.0f} MiB, max_contiguous={lo} MiB")
    except Exception as e:
        log.info(f"VRAM diagnostic failed: {e}")


def _strip_repeated_paragraphs(text: str) -> str:
    """Truncate at the first repeated paragraph block (hallucination guard)."""
    blocks = text.split('\n\n')
    seen: set[str] = set()
    result: list[str] = []
    for block in blocks:
        key = block.strip()
        if key and key in seen:
            break
        result.append(block)
        if key:
            seen.add(key)
    return '\n\n'.join(result).rstrip()


def unload_vision_model() -> None:
    global _vision_llama, _vision_llama_model_path
    if _vision_llama is not None:
        del _vision_llama
        _vision_llama = None
        _vision_llama_model_path = None
        gc.collect()
        _logger_mod.get().info("Vision model unloaded from VRAM.")


def unload_ollama_models(base_url: str = "http://localhost:11434") -> None:
    """Signal Ollama to release any loaded models from VRAM."""
    log = _logger_mod.get()
    try:
        r = requests.get(f"{base_url}/api/ps", timeout=5)
        if r.status_code != 200:
            return
        for m in r.json().get("models", []):
            requests.post(f"{base_url}/api/generate",
                          json={"model": m["name"], "keep_alive": 0}, timeout=10)
            log.info(f"Unloaded Ollama model from VRAM: {m['name']}")
    except Exception:
        pass


def _load_vision_model(model_path: str, mmproj_path: str, base_url: str) -> None:
    global _vision_llama, _vision_llama_model_path
    log = _logger_mod.get()

    if _vision_llama is not None and _vision_llama_model_path == model_path:
        return

    unload_ollama_models(base_url)
    _register_cuda_dll_dirs()

    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    except ImportError:
        log.error("llama-cpp-python is not installed.")
        raise SystemExit(1)
    except Exception as e:
        log.error(f"Failed to import llama-cpp-python: {e}")
        raise SystemExit(1)

    if not mmproj_path:
        log.error("--vision-mmproj is required for the llama-cpp-python backend.")
        raise SystemExit(1)

    log.info(f"Loading vision model: {model_path}")
    from llama_cpp._utils import suppress_stdout_stderr
    with suppress_stdout_stderr(disable=False):
        chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=True)
        _vision_llama = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=4096,  # было 8192; KV cache: 940 МБ → 470 МБ
            n_batch=64,
            n_gpu_layers=-1,
            flash_attn=True,
            verbose=True,
        )
    _vision_llama_model_path = model_path
    log.info("Vision model loaded.")


def page_to_md(
    png_bytes: bytes,
    model: str,
    mmproj: str,
    base_url: str = "http://localhost:11434",
) -> VisionResult:
    _load_vision_model(model, mmproj, base_url)

    img_b64 = base64.b64encode(png_bytes).decode()

    from llama_cpp._utils import suppress_stdout_stderr
    with suppress_stdout_stderr(disable=False):
        result = _vision_llama.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _PAGE_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }],
            max_tokens=2048,
            temperature=0.1,
            repeat_penalty=1.15,
        )
        _vision_llama.reset()
    usage = result.get("usage", {})
    text = _strip_repeated_paragraphs(result["choices"][0]["message"]["content"].strip())
    return VisionResult(
        text=text,
        tokens_in=usage.get("prompt_tokens", 0),
        tokens_out=usage.get("completion_tokens", 0),
    )
