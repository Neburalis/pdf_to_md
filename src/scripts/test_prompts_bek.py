"""
Test prompt candidates for Beklemishev PDF (Analytical Geometry & Linear Algebra).

New challenges vs. prior Petrovich book:
- Matrices and determinants (\begin{pmatrix}, \begin{vmatrix})
- Numbered equations aligned right: (1), (2), (3)
- Running page headers to skip (Гл. N. ...)
- Bold vector notation (**r**, **n**) -> \\mathbf or \\vec

Tests v6 (current baseline) against three new candidates on 5 pages
covering: coordinates/figures, line equations, conic sections, orthogonal
transforms, and matrices/linear systems.

Usage:
    .venv\\Scripts\\python.exe src/scripts/test_prompts_bek.py
Results in output/prompt_test_bek/
"""

import base64
import ctypes
import os
import site
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import fitz  # PyMuPDF

PDF   = "pdf/Beklemishev_DV_Kurs_analiticheskoi_geometrii_i_lineinoi_algebry.pdf"
VISION_MODEL = "models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
MMPROJ       = "models/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"
OUT   = Path("output/prompt_test_bek")
DPI   = 200

# 0-based page indices
TEST_PAGES = {
    "p020": 19,   # cylindrical coords, figure, numbered eq (3)
    "p050": 49,   # perpendicular plane, line eq, numbered (12)(13)
    "p075": 74,   # ellipse/hyperbola, figures, numbered (11)(12)(13)
    "p100": 99,   # orthogonal transforms, rotation matrix
    "p150": 149,  # linear systems, matrix Ax=b, theorem
}

# ── Prompt candidates ─────────────────────────────────────────────────────────

PROMPTS = {

# ── v10: surgical hybrid — v6 structure + stronger LaTeX + matrix/eq rules ───
"v10_hybrid": """\
Convert this PDF page image to Markdown.

Rules:
- Use # ## ### for headings based on visual size and weight
- Preserve lists (- or 1. or а), б), в)) exactly as structured in the image
- Preserve tables using Markdown table syntax (| col | col |)
- Preserve bold (**text**) and italic (*text*) formatting
- For mathematical formulas use LaTeX $ delimiters ONLY (NEVER \\( \\) or \\[ \\]): inline $formula$, display block $$formula$$
- Matrices inside display blocks: $$\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}$$
- Determinants inside display blocks: $$\\begin{vmatrix} a & b \\\\ c & d \\end{vmatrix}$$
- Numbered equations (right-side number N in the source): $$formula \\quad (N)$$
- For figures and diagrams write [Рисунок N] using the actual figure number — do NOT generate image URLs
- Do NOT wrap the output in code fences
- Do NOT repeat the same content block multiple times or at different heading levels
- If the page contains only a page number or is blank, output an empty string
- Output ONLY the Markdown content, no explanations""",

# ── Baseline: current production prompt (v6) ─────────────────────────────────
"v6_baseline": """\
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
- Output ONLY the Markdown content, no explanations""",

# ── v7: v6 + matrix/det rules + numbered eq + header suppression + vectors ───
"v7_matrix_aware": """\
Convert this PDF page image to Markdown.

This is a page from a Russian university textbook on analytical geometry and linear algebra.

Rules:
- HEADINGS: # ## ### based on visual font size; SKIP the running page header (the chapter/section title printed at the very top edge of the page)
- LISTS: preserve exactly — dash (-), numbered (1.), or Russian (а), б), в))
- TABLES: Markdown table syntax | col | col |
- BOLD: **text**, ITALIC: *text*
- MATH — use ONLY $ and $$ delimiters, NEVER \\( \\) or \\[ \\]:
  - Inline: $x_1 + x_2 = 0$
  - Display: $$\\vec{r} = x\\vec{e}_1 + y\\vec{e}_2 + z\\vec{e}_3$$
  - Matrix: $$\\begin{pmatrix} a_{11} & a_{12} \\\\ a_{21} & a_{22} \\end{pmatrix}$$
  - Determinant: $$\\begin{vmatrix} a & b \\\\ c & d \\end{vmatrix}$$
  - Numbered equation (right-side number in book): $$formula \\quad (N)$$
  - Bold vectors in book (r, n, a) → \\mathbf{r}, \\mathbf{n}, \\mathbf{a} inside LaTeX
- FIGURES: write [Рисунок N.M] with the actual caption number — do NOT generate image URLs
- Do NOT wrap output in code fences
- Do NOT repeat the same content at multiple heading levels
- Blank page or page number only → empty string
- Output ONLY Markdown, no explanations""",

# ── v8: ✓/✗ examples framing to enforce LaTeX and matrix syntax ──────────────
"v8_examples": """\
Convert this PDF page image to Markdown.

DOCUMENT: Russian university textbook on analytical geometry and linear algebra.

MATH (strict — follow exactly):

LaTeX delimiters must be $ and $$ ONLY:
  CORRECT: $x^2 + y^2 = r^2$   and   $$Ax = \\mathbf{b}$$
  WRONG:   \\(x^2 + y^2\\)     and   \\[Ax = b\\]

Matrices inside $$...$$ only:
  CORRECT: $$\\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}$$
  WRONG:   [[1, 0], [0, 1]]   or any ASCII-art table

Determinants inside $$...$$ only:
  CORRECT: $$\\begin{vmatrix} a & b \\\\ c & d \\end{vmatrix} = ad - bc$$

Numbered equations — append the number:
  CORRECT: $$\\vec{r} = x\\vec{e}_1 + y\\vec{e}_2 \\quad (3)$$

STRUCTURE:
- Headings # ## ### by font size; skip the running page header (top edge of page)
- Lists: -, 1., or Russian а), б), в) exactly as shown
- Bold **text**, italic *text*
- Figures: [Рисунок N.M] with actual caption number — no URLs
- No code fences, no repeated content blocks
- Blank/page-number-only page → empty string
- Output ONLY Markdown, no explanations""",

# ── v9: math-priority framing, context-first ─────────────────────────────────
"v9_math_priority": """\
You are a precise OCR assistant converting a Russian university textbook on analytical geometry and linear algebra to Markdown.

PRIORITY 1 — MATHEMATICAL CONTENT:
Every formula must be in LaTeX. Use exclusively $ for inline and $$ for display — never \\( \\) or \\[ \\].
- Matrices: $$\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}$$
- Determinants: $$\\begin{vmatrix} a & b \\\\ c & d \\end{vmatrix}$$
- Numbered equations (the (N) shown right of a formula): $$formula \\quad (N)$$
- Bold vectors in the book (r, n, a): use \\mathbf{r}, \\mathbf{n}, \\mathbf{a} inside LaTeX

PRIORITY 2 — TEXT STRUCTURE:
- Headings # ## ### by visual font size; IGNORE the running header printed at the top of the page
- Lists exactly as in the image: -, 1., or а), б), в)
- Bold **text**, italic *text*
- Theorems/Lemmas/Propositions/Examples: keep the label as bold prefix as shown
- Figures: [Рисунок N.M] with actual number — no image URLs

OUTPUT: Markdown only. No code fences. No repeated content. No explanations.
Blank page or page-number-only → empty string.""",

}

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(text: str) -> dict:
    import re
    dollar_count   = text.count("$")
    backslash_paren = len(re.findall(r"\\\(|\\\)", text))
    backslash_brack = len(re.findall(r"\\\[|\\\]", text))
    unicode_math   = len(re.findall(r"[∈∀∃∂→⟶←⟵↔⟺∞∑∏∫√∝∥⊥≤≥≠≈∈∉⊂⊃∪∩∧∨¬αβγδεζηθλμνξπρστφχψω]", text))
    pmatrix_count  = text.count("\\begin{pmatrix}")
    vmatrix_count  = text.count("\\begin{vmatrix}")
    mathbf_count   = text.count("\\mathbf")
    vec_count      = text.count("\\vec{")
    quad_n_count   = len(re.findall(r"\\quad\s*\(\d+\)", text))
    code_fence     = "```" in text
    imgur_url      = "imgur.com" in text
    length         = len(text)
    return {
        "len": length,
        "dollars": dollar_count,
        "backslash_paren": backslash_paren,
        "backslash_brack": backslash_brack,
        "unicode_math": unicode_math,
        "pmatrix": pmatrix_count,
        "vmatrix": vmatrix_count,
        "mathbf": mathbf_count,
        "vec": vec_count,
        "quad_n": quad_n_count,
        "code_fence": int(code_fence),
        "imgur": int(imgur_url),
    }

# ── Helpers ───────────────────────────────────────────────────────────────────

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
    for name in ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll",
                 "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "ggml.dll"]:
        if name in dm:
            try:
                ctypes.CDLL(str(dm[name]))
            except OSError:
                pass


def render_page(pdf_path: str, page_idx: int, dpi: int = DPI) -> bytes:
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def load_vision_model():
    _register_cuda_dll_dirs()
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    from llama_cpp._utils import suppress_stdout_stderr

    print("Loading vision model...", flush=True)
    with suppress_stdout_stderr():
        handler = Qwen25VLChatHandler(clip_model_path=MMPROJ, verbose=False)
        llm = Llama(
            model_path=VISION_MODEL,
            chat_handler=handler,
            n_ctx=8192,
            n_gpu_layers=-1,
            verbose=False,
        )
    print("Loaded.", flush=True)
    return llm


def run_prompt(llm, png_bytes: bytes, prompt: str,
               temperature: float = 0.1, repeat_penalty: float = 1.15) -> tuple[str, float]:
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
            temperature=temperature,
            repeat_penalty=repeat_penalty,
        )
        llm.reset()
    elapsed = time.time() - t0
    return result["choices"][0]["message"]["content"].strip(), elapsed

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    llm = load_vision_model()

    rows = []  # (page, prompt, metrics, elapsed, preview)

    for page_label, page_idx in TEST_PAGES.items():
        png_path = OUT / f"{page_label}.png"
        if png_path.exists():
            png_bytes = png_path.read_bytes()
        else:
            print(f"\nRendering {page_label}...", flush=True)
            png_bytes = render_page(PDF, page_idx)
            png_path.write_bytes(png_bytes)

        for prompt_name, prompt_text in PROMPTS.items():
            out_file = OUT / f"{page_label}_{prompt_name}.md"
            print(f"  {page_label} [{prompt_name}] ", end="", flush=True)

            if out_file.exists():
                text = out_file.read_text(encoding="utf-8")
                elapsed = 0.0
                print("(cached)", flush=True)
            else:
                text, elapsed = run_prompt(llm, png_bytes, prompt_text)
                out_file.write_text(text, encoding="utf-8")
                print(f"{elapsed:.0f}s  {len(text)}ch", flush=True)

            m = compute_metrics(text)
            preview = (text[:90].replace("\n", "↵")) if text else "(empty)"
            rows.append((page_label, prompt_name, elapsed, m, preview))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'PAGE':<8} {'PROMPT':<22} {'s':>4} {'len':>5} {'$':>4} {'\\(\\)':>5} {'uni':>4} "
          f"{'pmat':>5} {'vmat':>5} {'quad_n':>7} {'fence':>6} {'preview'}")
    print("-" * 80)

    totals: dict[str, dict] = {p: {"dollars": 0, "bp": 0, "uni": 0, "pmat": 0,
                                    "vmat": 0, "qn": 0, "fence": 0, "pages": 0}
                                for p in PROMPTS}
    for page_label, prompt_name, elapsed, m, preview in rows:
        t = totals[prompt_name]
        t["dollars"] += m["dollars"]
        t["bp"]      += m["backslash_paren"] + m["backslash_brack"]
        t["uni"]     += m["unicode_math"]
        t["pmat"]    += m["pmatrix"]
        t["vmat"]    += m["vmatrix"]
        t["qn"]      += m["quad_n"]
        t["fence"]   += m["code_fence"]
        t["pages"]   += 1
        print(f"{page_label:<8} {prompt_name:<22} {elapsed:>4.0f} {m['len']:>5} "
              f"{m['dollars']:>4} {m['backslash_paren'] + m['backslash_brack']:>5} "
              f"{m['unicode_math']:>4} {m['pmatrix']:>5} {m['vmatrix']:>5} "
              f"{m['quad_n']:>7} {m['code_fence']:>6}   {preview[:50]}")

    print("\n" + "=" * 80)
    print("TOTALS (5 pages):")
    print(f"{'PROMPT':<22} {'$-sum':>6} {'\\(\\)-sum':>9} {'uni':>5} {'pmat':>5} {'vmat':>5} "
          f"{'quad_n':>7} {'fences':>7}")
    print("-" * 80)
    for pname, t in totals.items():
        print(f"{pname:<22} {t['dollars']:>6} {t['bp']:>9} {t['uni']:>5} "
              f"{t['pmat']:>5} {t['vmat']:>5} {t['qn']:>7} {t['fence']:>7}")

    # Write summary to file
    lines = []
    for page_label, prompt_name, elapsed, m, preview in rows:
        lines.append(
            f"{page_label}  {prompt_name:<22}  {elapsed:5.0f}s  len={m['len']}  "
            f"$={m['dollars']}  \\()={m['backslash_paren']+m['backslash_brack']}  "
            f"uni={m['unicode_math']}  pmat={m['pmatrix']}  vmat={m['vmatrix']}  "
            f"quad_n={m['quad_n']}  fence={m['code_fence']}  |  {preview[:80]}"
        )
    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults in {OUT}/")
    print("Read individual .md files for manual comparison.")


if __name__ == "__main__":
    main()
