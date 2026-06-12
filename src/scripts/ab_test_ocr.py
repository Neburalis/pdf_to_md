"""
A/B test: Qwen2.5-VL with vs without OCR text hint.
Tests 30 evenly-spaced pages from chast-2.
Metrics: LaTeX usage ($ count), Unicode math outside LaTeX, content length, hallucination flags.
"""

import base64, os, re, site, ctypes, sys, time, json
from collections import Counter
from pathlib import Path

import fitz

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VISION_MODEL = "models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
MMPROJ       = "models/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"
PDF          = "pdf/lektsii-po-matematicheskomu-analizu-v-3-ch-chast-2-mnogomernyj-analiz-integraly-i-rjady.pdf"
OUT          = Path("output/ab_test_ocr")
DPI          = 200
N_PAGES      = 30   # evenly spaced across the document

PROMPT_BASE = """\
Convert this PDF page image to Markdown.

Rules:
- Use # ## ### for headings based on visual size and weight
- Preserve lists (- or 1.) exactly as structured in the image
- Preserve tables using Markdown table syntax (| col | col |)
- Preserve bold (**text**) and italic (*text*) formatting
- For mathematical formulas use LaTeX with $ delimiters: inline $formula$, display block $$formula$$ — do NOT use \\( \\) or \\[ \\]
- For figures and diagrams write [Рисунок N] using the actual figure number from the text — do NOT generate image URLs
- Do NOT wrap the output in code fences
- Do NOT repeat the same content block multiple times or at different heading levels
- If the page contains only a page number or is blank, output an empty string
- Output ONLY the Markdown content, no explanations"""

PROMPT_OCR = PROMPT_BASE + """

OCR reference (use only for Cyrillic spelling and math term verification — rely on IMAGE for structure and layout):
{ocr_text}"""

# ── Math heuristics ───────────────────────────────────────────────────────────

_MATH_UNICODE_RE = re.compile(
    r"[∈∉∀∃∂→←↔⇒⇔≤≥≠≈∞∑∏∫∬∭√∪∩⊂⊃⊆⊇∧∨⊕⊗±∓≡≃≅∝∠⊥∥⌊⌋⌈⌉⩽⩾"
    r"αβγδεζηθικλμνξπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ"
    r"¹²³⁰⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉]"
)

def _strip_latex(text: str) -> str:
    text = re.sub(r"\$\$[\s\S]*?\$\$", "", text)
    text = re.sub(r"\\\[[\s\S]*?\\\]", "", text)
    text = re.sub(r"\$[^\$\n]{1,300}\$", "", text)
    return text

def _cyrillic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum(1 for c in letters if "Ѐ" <= c <= "ӿ") / len(letters)

def _repetition_score(text: str) -> float:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 6:
        return 0.0
    return Counter(lines).most_common(1)[0][1] / len(lines)

def score_page(text: str) -> dict:
    s = text.strip()
    no_latex = _strip_latex(s)
    n_math = len(_MATH_UNICODE_RE.findall(no_latex))
    n_dollar = s.count("$")
    rep = _repetition_score(s)
    cyr = _cyrillic_ratio(s)
    flags = []
    if "i.imgur.com" in s: flags.append("imgur")
    if cyr < 0.08 and len(s) > 300: flags.append("english")
    if n_math >= 8 and n_dollar < 4: flags.append("unicode_math")
    if rep > 0.62: flags.append("repetitive")
    if "Convert this PDF page" in s: flags.append("prompt_leak")
    return {
        "length": len(s),
        "dollar_count": n_dollar,
        "unicode_math_outside_latex": n_math,
        "repetition": round(rep, 3),
        "cyrillic_ratio": round(cyr, 3),
        "flags": flags,
        "has_backslash_parens": r"\(" in s,
    }

# ── Model helpers ─────────────────────────────────────────────────────────────

def _register_cuda_dll_dirs() -> None:
    dll_dirs = []
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
    for name in ["cudart64_12.dll","cublas64_12.dll","cublasLt64_12.dll",
                 "ggml-base.dll","ggml-cpu.dll","ggml-cuda.dll","ggml.dll"]:
        if name in dm:
            try: ctypes.CDLL(str(dm[name]))
            except OSError: pass


def load_model():
    _register_cuda_dll_dirs()
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    from llama_cpp._utils import suppress_stdout_stderr
    print("Loading Qwen2.5-VL...", flush=True)
    with suppress_stdout_stderr():
        handler = Qwen25VLChatHandler(clip_model_path=MMPROJ, verbose=False)
        llm = Llama(model_path=VISION_MODEL, chat_handler=handler,
                    n_ctx=8192, n_gpu_layers=-1, verbose=False)
    print("Loaded.", flush=True)
    return llm


def render_page(pdf_path: str, page_idx: int) -> bytes:
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def extract_text(pdf_path: str, page_idx: int) -> str:
    doc = fitz.open(pdf_path)
    return doc[page_idx].get_text("text").strip()


def run_vision(llm, png_bytes: bytes, prompt: str) -> tuple[str, float]:
    from llama_cpp._utils import suppress_stdout_stderr
    img_b64 = base64.b64encode(png_bytes).decode()
    t0 = time.time()
    with suppress_stdout_stderr():
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ]}],
            max_tokens=2048,
            temperature=0.1,
            repeat_penalty=1.15,
        )
        llm.reset()
    return result["choices"][0]["message"]["content"].strip(), time.time() - t0

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    llm = load_model()

    doc = fitz.open(PDF)
    total = len(doc)
    # Evenly spaced page indices (0-based)
    step = total / N_PAGES
    page_indices = [int(i * step) for i in range(N_PAGES)]
    print(f"Testing {N_PAGES} pages from {PDF}", flush=True)
    print(f"Page indices: {page_indices}", flush=True)

    results = []

    for page_idx in page_indices:
        label = f"p{page_idx+1:04d}"
        print(f"\n{'='*55}", flush=True)
        print(f"{label} (0-based {page_idx})", flush=True)

        png = render_page(PDF, page_idx)
        ocr_text = extract_text(PDF, page_idx)
        ocr_preview = ocr_text[:120].replace("\n", " ")
        print(f"  OCR: {ocr_preview}", flush=True)

        row = {"page": page_idx + 1, "page_idx": page_idx, "ocr_len": len(ocr_text)}

        for variant, prompt in [("no_ocr", PROMPT_BASE),
                                  ("with_ocr", PROMPT_OCR.format(ocr_text=ocr_text))]:
            out_file = OUT / f"{label}_{variant}.md"
            if out_file.exists():
                text = out_file.read_text(encoding="utf-8")
                elapsed = 0.0
                print(f"  [{variant}] cached", flush=True)
            else:
                print(f"  [{variant}] running...", flush=True)
                text, elapsed = run_vision(llm, png, prompt)
                out_file.write_text(text, encoding="utf-8")

            s = score_page(text)
            print(f"  [{variant}] {elapsed:.0f}s  {s['length']}ch  "
                  f"${s['dollar_count']}  unicode={s['unicode_math_outside_latex']}  "
                  f"flags={s['flags']}", flush=True)
            row[f"{variant}_len"] = s["length"]
            row[f"{variant}_dollars"] = s["dollar_count"]
            row[f"{variant}_unicode_math"] = s["unicode_math_outside_latex"]
            row[f"{variant}_flags"] = s["flags"]
            row[f"{variant}_backslash_parens"] = s["has_backslash_parens"]
            row[f"{variant}_elapsed"] = round(elapsed)

        results.append(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'='*55}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'page':>6}  {'no_ocr':>6} {'$':>4} {'Umath':>5} {'flags_no':>10}  |  "
          f"{'with_ocr':>8} {'$':>4} {'Umath':>5} {'flags_w':>10}", flush=True)

    no_ocr_wins = with_ocr_wins = ties = 0
    for r in results:
        n_bad_no  = len(r["no_ocr_flags"])
        n_bad_w   = len(r["with_ocr_flags"])
        winner = "TIE" if n_bad_no == n_bad_w else ("no_ocr" if n_bad_no < n_bad_w else "with_ocr")
        if winner == "no_ocr": no_ocr_wins += 1
        elif winner == "with_ocr": with_ocr_wins += 1
        else: ties += 1
        print(f"p{r['page']:04d}  {r['no_ocr_len']:>6} {r['no_ocr_dollars']:>4} "
              f"{r['no_ocr_unicode_math']:>5} {str(r['no_ocr_flags']):>10}  |  "
              f"{r['with_ocr_len']:>8} {r['with_ocr_dollars']:>4} "
              f"{r['with_ocr_unicode_math']:>5} {str(r['with_ocr_flags']):>10}  {winner}", flush=True)

    print(f"\nno_ocr wins: {no_ocr_wins}  with_ocr wins: {with_ocr_wins}  ties: {ties}", flush=True)

    # Aggregate metrics
    avg = lambda key: sum(r[key] for r in results) / len(results)
    print(f"\nAVERAGE no_ocr:   len={avg('no_ocr_len'):.0f}  $={avg('no_ocr_dollars'):.1f}  "
          f"unicode={avg('no_ocr_unicode_math'):.1f}  "
          f"flagged={sum(1 for r in results if r['no_ocr_flags'])}/{len(results)}", flush=True)
    print(f"AVERAGE with_ocr: len={avg('with_ocr_len'):.0f}  $={avg('with_ocr_dollars'):.1f}  "
          f"unicode={avg('with_ocr_unicode_math'):.1f}  "
          f"flagged={sum(1 for r in results if r['with_ocr_flags'])}/{len(results)}", flush=True)

    (OUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResults saved to {OUT}/results.json", flush=True)


if __name__ == "__main__":
    main()
