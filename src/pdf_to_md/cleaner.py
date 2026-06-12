import re
import sys
from typing import NamedTuple

from . import logger as _logger_mod
from .vision import _register_cuda_dll_dirs


class CleanResult(NamedTuple):
    text: str
    tokens_in: int
    tokens_out: int


_CLEAN_PROMPT = """Clean up the following Markdown section converted from a PDF. Apply these fixes:

1. Remove page number artifacts (e.g. a lone "42" on its own line, "Page 42 of 100")
2. Remove running headers or footers that repeat across pages (book title, chapter name, etc.)
3. Fix words broken by hyphens at line boundaries only when clearly a split word, not a compound word
   Example: "exam-\\nple" -> "example", but "well-known" stays "well-known"
4. Merge paragraph sentences split across page breaks into continuous text
5. DO NOT remove, summarize, shorten, or rewrite any meaningful content
6. DO NOT change headings, lists, tables, or code blocks
7. Output ONLY the cleaned Markdown, no explanations
8. Do NOT wrap the output in ```markdown or any other code fences

Section:
{text}"""

_text_llama = None
_text_llama_model_path = None


def _load_text_model(model_path: str) -> None:
    global _text_llama, _text_llama_model_path
    log = _logger_mod.get()

    if _text_llama is not None and _text_llama_model_path == model_path:
        return

    _register_cuda_dll_dirs()

    try:
        from llama_cpp import Llama
    except ImportError:
        log.error("llama-cpp-python is not installed.")
        raise SystemExit(1)
    except Exception as e:
        log.error(f"Failed to import llama-cpp-python: {e}")
        raise SystemExit(1)

    log.info(f"Loading text model: {model_path}")
    from llama_cpp._utils import suppress_stdout_stderr
    with suppress_stdout_stderr():
        _text_llama = Llama(
            model_path=model_path,
            n_ctx=16384,
            n_gpu_layers=-1,   # full GPU: model 7.9 GB + Q4 KV cache ~1.4 GB = ~9.3 GB < 12 GB
            type_k=2,          # Q4_0: 4x smaller KV cache vs fp16
            type_v=2,          # Q4_0
            flash_attn=True,
            verbose=False,
        )
    _text_llama_model_path = model_path
    log.info("Text model loaded.")


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^```(?:markdown)?\s*\n', '', text)
    text = re.sub(r'\n```\s*$', '', text)
    text = re.sub(r'```markdown\s*\n(.*?)\n```', r'\1', text, flags=re.DOTALL)
    return text.strip()


def clean_chunk(text: str, model: str) -> CleanResult:
    _load_text_model(model)

    from llama_cpp._utils import suppress_stdout_stderr
    with suppress_stdout_stderr():
        result = _text_llama.create_chat_completion(
            messages=[{"role": "user", "content": _CLEAN_PROMPT.format(text=text)}],
            max_tokens=8192,
            temperature=0.0,
        )
    usage = result.get("usage", {})
    cleaned = _strip_code_fences(result["choices"][0]["message"]["content"].strip())
    return CleanResult(
        text=cleaned,
        tokens_in=usage.get("prompt_tokens", 0),
        tokens_out=usage.get("completion_tokens", 0),
    )
