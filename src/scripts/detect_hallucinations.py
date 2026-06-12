"""
Detect hallucinations in vision-extracted markdown pages.
Pass 1: fast heuristics (size deviation, English text, fake URLs, repetition).
Pass 2: gemma for pages that pass heuristics.

Usage:
    python detect_hallucinations.py                   # scans output/*.md
    python detect_hallucinations.py output/book.md    # specific file(s)
"""

import math
import os, re, site, ctypes, sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
from pathlib import Path

# Unicode math symbols that indicate OCR-text formulas (not LaTeX)
_MATH_UNICODE_RE = re.compile(
    r"[∈∉∀∃∂→←↔⇒⇔≤≥≠≈∞∑∏∫∬∭√∪∩⊂⊃⊆⊇∧∨⊕⊗±∓≡≃≅∝∠⊥∥⌊⌋⌈⌉⩽⩾"
    r"αβγδεζηθικλμνξπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ"
    r"¹²³⁰⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉]"
)

def _strip_latex(text: str) -> str:
    """Remove content inside LaTeX delimiters so we can count symbols outside them."""
    text = re.sub(r"\$\$[\s\S]*?\$\$", "", text)
    text = re.sub(r"\\\[[\s\S]*?\\\]", "", text)
    text = re.sub(r"\$[^\$\n]{1,300}\$", "", text)
    text = re.sub(r"\\\([^\)]*\\\)", "", text)
    return text

PAGE_SEP = "\n\n---\n\n"
MODEL = "models/gemma-3-12b-it-heretic-Q5_K_M.gguf"

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


# ── Helpers ───────────────────────────────────────────────────────────────────

_SIZE_K   = 2.0   # sigma multiplier: flag pages outside mean ± K*sigma
_SIZE_CAP = 500   # chars: cap on sigma so the window can't grow unboundedly

def _page_stats(pages: list[str]) -> tuple[float, float]:
    """Return (mean, std_dev) of non-trivial page lengths (≥30 chars)."""
    lens = [len(p.strip()) for p in pages if len(p.strip()) >= 30]
    if len(lens) < 2:
        return 0.0, 0.0
    mean = sum(lens) / len(lens)
    variance = sum((x - mean) ** 2 for x in lens) / (len(lens) - 1)
    return mean, math.sqrt(variance)


# ── Heuristics ────────────────────────────────────────────────────────────────

def size_deviation_check(page: str, mean_len: float, std_len: float) -> tuple[bool, str]:
    """Flag pages that deviate more than K sigma from the per-file mean.

    std_dev is capped at _SIZE_CAP so the window can't grow unboundedly when
    page lengths are naturally varied. With the defaults (K=2, cap=500):
      mean 1500, σ=300 → window [900, 2100]
      mean 1500, σ=500 → window [500, 2500]
      mean 1500, σ≥500 → window capped at [500, 2500]
    """
    if mean_len < 100:
        return False, ""
    n = len(page.strip())
    sigma = min(std_len, _SIZE_CAP)
    lo = mean_len - _SIZE_K * sigma
    hi = mean_len + _SIZE_K * sigma
    if n < lo:
        return True, (f"Too short: {n} chars "
                      f"(mean {mean_len:.0f}, σ {sigma:.0f}, min {lo:.0f})")
    if n > hi:
        return True, (f"Too long:  {n} chars "
                      f"(mean {mean_len:.0f}, σ {sigma:.0f}, max {hi:.0f})")
    return False, ""


def _cyrillic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum(1 for c in letters if "Ѐ" <= c <= "ӿ") / len(letters)


def _repetition_score(text: str) -> float:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 6:
        return 0.0
    top_count = Counter(lines).most_common(1)[0][1]
    return top_count / len(lines)


def heuristic_check(page: str) -> tuple[bool, str]:
    s = page.strip()
    if len(s) < 50:
        return False, ""

    if "i.imgur.com" in s:
        return True, "Fake imgur image URLs"

    if re.search(r"\[Section \d+[^\]]*\]|\[img-\d+\]|\[Figure \d+\]", s):
        return True, "Template placeholders [Section X] / [img-X]"

    # Prompt instructions leaked into output
    if "Convert this PDF page" in s or "Output ONLY the Markdown" in s:
        return True, "OCR prompt text appeared in output"

    # English content (very low Cyrillic)
    if len(s) > 300 and _cyrillic_ratio(s) < 0.08:
        return True, f"English content (Cyrillic ratio {_cyrillic_ratio(s):.1%})"

    # Math formulas written as Unicode text instead of LaTeX
    if len(s) > 150 and _cyrillic_ratio(s) >= 0.10:
        no_latex = _strip_latex(s)
        n_math = len(_MATH_UNICODE_RE.findall(no_latex))
        n_dollar = s.count("$")
        if n_math >= 8 and n_dollar < 4:
            return True, f"Unicode math outside LaTeX ({n_math} symbols, {n_dollar} $ signs)"

    # Repetitive blocks
    rep = _repetition_score(s)
    if rep > 0.62:
        return True, f"Highly repetitive ({rep:.0%} of lines identical)"

    return False, ""


# ── Gemma ─────────────────────────────────────────────────────────────────────

def _load_dlls() -> None:
    dll_dirs: list[Path] = []
    for sp in site.getsitepackages():
        base = Path(sp)
        nd = base / "nvidia"
        if nd.is_dir():
            for pkg in nd.iterdir():
                for sub in ("bin", "lib"):
                    d = pkg / sub
                    if d.is_dir():
                        dll_dirs.append(d)
        ll = base / "llama_cpp" / "lib"
        if ll.is_dir():
            dll_dirs.append(ll)
    if hasattr(os, "add_dll_directory"):
        for d in dll_dirs:
            os.add_dll_directory(str(d))
    dm = {p.name: p for d in dll_dirs for p in d.glob("*.dll")}
    for name in ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll",
                 "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "ggml.dll"]:
        if name in dm:
            try:
                ctypes.CDLL(str(dm[name]))
            except OSError:
                pass


def load_model():
    _load_dlls()
    from llama_cpp import Llama
    from llama_cpp._utils import suppress_stdout_stderr
    print("Loading gemma...", flush=True)
    with suppress_stdout_stderr():
        llm = Llama(
            model_path=MODEL,
            n_ctx=4096,
            n_gpu_layers=-1,
            type_k=2,
            type_v=2,
            flash_attn=True,
            verbose=False,
        )
    print("Loaded.", flush=True)
    return llm


def gemma_check(llm, text: str) -> tuple[bool, str]:
    from llama_cpp._utils import suppress_stdout_stderr
    with suppress_stdout_stderr():
        r = llm.create_chat_completion(
            messages=[{"role": "user", "content": PROMPT.format(text=text[:2000])}],
            max_tokens=80,
            temperature=0.0,
        )
    verdict = r["choices"][0]["message"]["content"].strip()
    return verdict.upper().startswith("HALLUCINATION"), verdict


# ── Main ──────────────────────────────────────────────────────────────────────

def page_line_starts(text: str, pages: list[str]) -> list[int]:
    starts, cur = [], 1
    for i, pg in enumerate(pages):
        starts.append(cur)
        cur += pg.count("\n") + (4 if i < len(pages) - 1 else 0)
    return starts


def main() -> None:
    # ── Resolve input files ───────────────────────────────────────────────────
    if len(sys.argv) > 1:
        paths = [Path(a) for a in sys.argv[1:]]
    else:
        paths = sorted(Path("output").glob("*.md"))

    if not paths:
        print("No .md files found in output/. Pass file paths as arguments.", flush=True)
        return

    llm = load_model()
    report_path = Path("output/hallucination_report.txt")
    all_flagged: list[tuple] = []

    with open(report_path, "w", encoding="utf-8") as rep:
        for path in paths:
            if not path.exists():
                print(f"SKIP: {path} not found", flush=True)
                continue

            label = path.stem
            text = path.read_text(encoding="utf-8")
            pages = text.split(PAGE_SEP)
            line_starts = page_line_starts(text, pages)

            # Per-file stats for size deviation check
            mean_len, std_len = _page_stats(pages)
            sigma_eff = min(std_len, _SIZE_CAP)
            lo = mean_len - _SIZE_K * sigma_eff
            hi = mean_len + _SIZE_K * sigma_eff

            print(f"\n[{label}] {len(pages)} pages  |  "
                  f"mean {mean_len:.0f}  std {std_len:.0f} (cap {sigma_eff:.0f})  "
                  f"window [{lo:.0f}, {hi:.0f}]", flush=True)
            rep.write(f"\n=== {label} ({len(pages)} pages, "
                      f"mean {mean_len:.0f}, σ {std_len:.0f}, "
                      f"window [{lo:.0f}, {hi:.0f}]) ===\n")
            part_count = 0

            for i, page in enumerate(pages):
                pnum = i + 1
                lnum = line_starts[i]
                stripped = page.strip()

                if len(stripped) < 30:
                    continue

                # Check order: size deviation (O(1)) → heuristics → gemma
                is_hall, reason = size_deviation_check(stripped, mean_len, std_len)
                method = "size    "
                if not is_hall:
                    is_hall, reason = heuristic_check(stripped)
                    method = "heuristic"
                if not is_hall:
                    is_hall, reason = gemma_check(llm, stripped)
                    method = "gemma   "

                tag = "HALL" if is_hall else "ok  "
                print(f"  p{pnum:4d} line~{lnum:5d}: {tag} [{method}] {reason[:65]}", flush=True)

                if is_hall:
                    entry = f"  page {pnum:4d}  line~{lnum:5d}  [{method.strip()}]  {reason}"
                    rep.write(entry + "\n")
                    rep.flush()
                    all_flagged.append((label, pnum, lnum, method.strip(), reason))
                    part_count += 1

            rep.write(f"  → {part_count} hallucination pages\n")
            print(f"  [{label}] flagged: {part_count}", flush=True)

        rep.write(f"\n\nTOTAL: {len(all_flagged)} hallucination pages\n\n")
        for label, pnum, lnum, method, reason in all_flagged:
            rep.write(f"  {label}  page {pnum:4d}  line~{lnum:5d}  [{method}]  {reason}\n")

    print(f"\nDone. Total flagged: {len(all_flagged)}", flush=True)
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
