import ctypes
import math
import os
import re
import site
from collections import Counter
from pathlib import Path

from . import logger


_MATH_UNICODE_RE = re.compile(
    r"[∈∉∀∃∂→←↔⇒⇔≤≥≠≈∞∑∏∫∬∭√∪∩⊂⊃⊆⊇∧∨⊕⊗±∓≡≃≅∝∠⊥∥⌊⌋⌈⌉⩽⩾"
    r"αβγδεζηθικλμνξπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ"
    r"¹²³⁰⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉]"
)

PAGE_SEP = "\n\n---\n\n"
PROMPT = """You analyze one page from a Russian mathematical analysis textbook extracted via vision OCR.

Is this a HALLUCINATION or OK?

HALLUCINATION if:
- Content is in English (not a Russian math textbook)
- Contains template placeholders like [Section X Title], [img-X], [Figure X]
- Same paragraph or block repeated 3+ times verbatim
- Fake/placeholder image URLs
- Topic completely unrelated to mathematics (nutrition, history, fiction, etc.)
- OCR prompt instructions appear in the output ("Convert this PDF page...", "Rules:", etc.)

OK if:
- Russian mathematical text (even imperfect OCR)
- Mathematical formulas (even not proper LaTeX)
- Title pages, blank pages, page numbers, chapter headers

Reply with exactly one line:
HALLUCINATION: <reason>
or
OK

Page content:
{text}"""

_SIZE_K = 2.0
_SIZE_CAP = 500


def _strip_latex(text: str) -> str:
    text = re.sub(r"\$\$[\s\S]*?\$\$", "", text)
    text = re.sub(r"\\\[[\s\S]*?\\\]", "", text)
    text = re.sub(r"\$[^\$\n]{1,300}\$", "", text)
    text = re.sub(r"\\\([^\)]*\\\)", "", text)
    return text


def _page_stats(pages: list[str]) -> tuple[float, float]:
    """Return the mean and sample stddev of pages containing at least 30 chars."""
    lengths = [len(page.strip()) for page in pages if len(page.strip()) >= 30]
    if len(lengths) < 2:
        return 0.0, 0.0
    mean = sum(lengths) / len(lengths)
    variance = sum((length - mean) ** 2 for length in lengths) / (
        len(lengths) - 1
    )
    return mean, math.sqrt(variance)


def size_deviation_check(
    page: str,
    mean_len: float,
    std_len: float,
) -> tuple[bool, str]:
    """Flag pages outside the per-file mean ± K*sigma window."""
    if mean_len < 100:
        return False, ""
    length = len(page.strip())
    sigma = min(std_len, _SIZE_CAP)
    low = mean_len - _SIZE_K * sigma
    high = mean_len + _SIZE_K * sigma
    if length < low:
        return True, (
            f"Too short: {length} chars "
            f"(mean {mean_len:.0f}, σ {sigma:.0f}, min {low:.0f})"
        )
    if length > high:
        return True, (
            f"Too long:  {length} chars "
            f"(mean {mean_len:.0f}, σ {sigma:.0f}, max {high:.0f})"
        )
    return False, ""


def _cyrillic_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 1.0
    return sum(1 for char in letters if "Ѐ" <= char <= "ӿ") / len(letters)


def _repetition_score(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 6:
        return 0.0
    top_count = Counter(lines).most_common(1)[0][1]
    return top_count / len(lines)


def heuristic_check(page: str) -> tuple[bool, str]:
    text = page.strip()
    if len(text) < 50:
        return False, ""
    if "i.imgur.com" in text:
        return True, "Fake imgur image URLs"
    if re.search(
        r"\[Section \d+[^\]]*\]|\[img-\d+\]|\[Figure \d+\]",
        text,
    ):
        return True, "Template placeholders [Section X] / [img-X]"
    if "Convert this PDF page" in text or "Output ONLY the Markdown" in text:
        return True, "OCR prompt text appeared in output"

    cyrillic_ratio = _cyrillic_ratio(text)
    if len(text) > 300 and cyrillic_ratio < 0.08:
        return True, f"English content (Cyrillic ratio {cyrillic_ratio:.1%})"
    if len(text) > 150 and cyrillic_ratio >= 0.10:
        no_latex = _strip_latex(text)
        math_symbols = len(_MATH_UNICODE_RE.findall(no_latex))
        dollar_signs = text.count("$")
        if math_symbols >= 8 and dollar_signs < 4:
            return True, (
                "Unicode math outside LaTeX "
                f"({math_symbols} symbols, {dollar_signs} $ signs)"
            )

    repetition = _repetition_score(text)
    if repetition > 0.62:
        return True, f"Highly repetitive ({repetition:.0%} of lines identical)"
    return False, ""


def _load_dlls() -> None:
    dll_dirs: list[Path] = []
    for site_package in site.getsitepackages():
        base = Path(site_package)
        nvidia_dir = base / "nvidia"
        if nvidia_dir.is_dir():
            for package_dir in nvidia_dir.iterdir():
                for subdir in ("bin", "lib"):
                    dll_dir = package_dir / subdir
                    if dll_dir.is_dir():
                        dll_dirs.append(dll_dir)
        llama_lib = base / "llama_cpp" / "lib"
        if llama_lib.is_dir():
            dll_dirs.append(llama_lib)

    if hasattr(os, "add_dll_directory"):
        for dll_dir in dll_dirs:
            os.add_dll_directory(str(dll_dir))

    dlls = {
        path.name: path
        for dll_dir in dll_dirs
        for path in dll_dir.glob("*.dll")
    }
    for name in (
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "ggml-base.dll",
        "ggml-cpu.dll",
        "ggml-cuda.dll",
        "ggml.dll",
    ):
        if name in dlls:
            try:
                ctypes.CDLL(str(dlls[name]))
            except OSError:
                pass


def load_model(model_path: str | Path):
    _load_dlls()
    from llama_cpp import Llama
    from llama_cpp._utils import suppress_stdout_stderr

    logger.get().info("Loading gemma...")
    with suppress_stdout_stderr():
        llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_gpu_layers=-1,
            type_k=2,
            type_v=2,
            flash_attn=True,
            verbose=False,
        )
    logger.get().info("Loaded.")
    return llm


def gemma_check(llm, text: str) -> tuple[bool, str]:
    from llama_cpp._utils import suppress_stdout_stderr

    with suppress_stdout_stderr():
        response = llm.create_chat_completion(
            messages=[
                {"role": "user", "content": PROMPT.format(text=text[:2000])}
            ],
            max_tokens=80,
            temperature=0.0,
        )
    verdict = response["choices"][0]["message"]["content"].strip()
    return verdict.upper().startswith("HALLUCINATION"), verdict
