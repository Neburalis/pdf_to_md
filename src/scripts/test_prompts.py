"""
Test different vision prompts on a fixed set of PDF pages.
Renders pages fresh from PDF (bypasses cache), runs each prompt variant,
saves results to output/prompt_test/ for comparison.
"""

import base64, os, site, ctypes, time, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import fitz  # PyMuPDF

VISION_MODEL = "models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
MMPROJ       = "models/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"
PDF          = "pdf/lektsii-po-matematicheskomu-analizu-v-3-ch-chast-2-mnogomernyj-analiz-integraly-i-rjady.pdf"
OUT          = Path("output/prompt_test")
DPI          = 200

# 0-based page indices: mix of early/mid/late pages
TEST_PAGES = {
    "p010": 9,    # early content
    "p025": 24,   # early-mid
    "p050": 49,   # mid
    "p080": 79,   # mid-late (area with known hallucinations)
    "p120": 119,  # late
}

# Additional pages known to hallucinate — from chast-1 (0-based)
TEST_PAGES_HALL = {
    "c1p014": 13,   # heading-depth repetition
    "c1p031": 30,   # content repetition
    "c1p129": 128,  # imgur on figure page
}

# ── Prompt variants ───────────────────────────────────────────────────────────

PROMPTS = {

"v1_current": """\
Convert this PDF page image to Markdown.

Rules:
- Use # ## ### for headings based on visual size and weight
- Preserve lists (- or 1.) exactly as structured in the image
- Preserve tables using Markdown table syntax (| col | col |)
- Preserve bold (**text**) and italic (*text*) formatting
- For mathematical formulas use LaTeX notation: inline $formula$, block $$formula$$
- If the page contains only a page number or is blank, output an empty string
- Output ONLY the Markdown content, no explanations, no meta-commentary
- Do NOT reproduce these instructions in the output""",

"v2_russian": """\
Ты ассистент, конвертирующий страницы PDF-учебника в Markdown.
На изображении — страница из русского учебника по математическому анализу.

Задача: точно воспроизведи содержимое страницы.

Правила:
- Выводи текст на русском языке, как написано в оригинале
- Математические формулы — в LaTeX: $инлайн$ и $$блок на отдельной строке$$
- Заголовки: # ## ### по визуальной иерархии
- Списки, таблицы — стандартный Markdown
- Если страница пустая или только номер страницы — выведи пустую строку
- Выводи ТОЛЬКО Markdown, без пояснений и обрамляющих ```""",

"v3_faithful": """\
Convert this PDF page image to Markdown.

This is a page from a Russian mathematical analysis textbook.

Critical rules:
- Transcribe ONLY what is visible in the image — do not invent or guess content
- Output in Russian (the source language of the document)
- Mathematical formulas: LaTeX syntax — $inline$ or $$display block$$
- Headings: # ## ### according to visual font size hierarchy
- If the page is blank or contains only a page number: output an empty string
- Output ONLY the Markdown content, no explanations, no code fences""",

"v4_minimal": """\
Convert this Russian mathematical analysis textbook page to Markdown.
Use LaTeX for all formulas: $inline$ or $$block$$.
Output only the Markdown content. If the page is blank or just a page number, output nothing.""",

"v5_figures": """\
Convert this PDF page image to Markdown.

Rules:
- Use # ## ### for headings based on visual size and weight
- Preserve lists (- or 1.) exactly as structured in the image
- Preserve tables using Markdown table syntax (| col | col |)
- Preserve bold (**text**) and italic (*text*) formatting
- For mathematical formulas use LaTeX: inline $formula$, display block $$formula$$
- For figures and diagrams write a placeholder like [Рисунок 1.2] — do NOT generate image URLs
- Do NOT wrap the output in code fences
- If the page contains only a page number or is blank, output an empty string
- Output ONLY the Markdown content, no explanations""",

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
    for name in ["cudart64_12.dll","cublas64_12.dll","cublasLt64_12.dll",
                 "ggml-base.dll","ggml-cpu.dll","ggml-cuda.dll","ggml.dll"]:
        if name in dm:
            try: ctypes.CDLL(str(dm[name]))
            except OSError: pass


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
               temperature: float = 0.0, repeat_penalty: float = 1.0) -> tuple[str, float]:
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

    summary_lines = []
    V5_PROMPT = PROMPTS["v5_figures"]

    # Test repeat_penalty on hallucination-prone pages
    pdf_c1 = "pdf/lektsii-po-matematicheskomu-analizu-v-3-ch-chast-1-vvedenie-v-matematicheskij-analiz.pdf"
    print("\n" + "="*60, flush=True)
    print("Testing repeat_penalty on known hallucination pages", flush=True)
    for page_label, page_idx in TEST_PAGES_HALL.items():
        png = render_page(pdf_c1, page_idx)
        (OUT / f"{page_label}.png").write_bytes(png)
        for name, temp, rp in [("v5_t0_rp1", 0.0, 1.0), ("v6_t01_rp115", 0.1, 1.15)]:
            out_file = OUT / f"{page_label}_{name}.md"
            if out_file.exists():
                text = out_file.read_text(encoding="utf-8"); elapsed = 0.0
                print(f"  {page_label} [{name}] skipped (cached)", flush=True)
            else:
                print(f"  {page_label} [{name}] running...", flush=True)
                text, elapsed = run_prompt(llm, png, V5_PROMPT, temperature=temp, repeat_penalty=rp)
                out_file.write_text(text, encoding="utf-8")
            preview = text[:100].replace("\n", "↵") if text else "(empty)"
            print(f"  {page_label} [{name}] {elapsed:.0f}s {len(text)}ch | {preview}", flush=True)
            summary_lines.append(f"{page_label}  {name:20s}  {elapsed:5.0f}s  {len(text):5d}ch  {preview[:70]}")

    for page_label, page_idx in TEST_PAGES.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Page: {page_label} (0-based idx {page_idx})", flush=True)

        png = render_page(PDF, page_idx)
        (OUT / f"{page_label}.png").write_bytes(png)
        print(f"  Rendered {len(png)//1024} KB PNG", flush=True)

        for prompt_name, prompt_text in PROMPTS.items():
            out_file = OUT / f"{page_label}_{prompt_name}.md"
            if out_file.exists():
                text = out_file.read_text(encoding="utf-8")
                elapsed = 0.0
                print(f"  [{prompt_name}] skipped (cached)", flush=True)
            else:
                print(f"  [{prompt_name}] running...", flush=True)
                text, elapsed = run_prompt(llm, png, prompt_text)
                out_file.write_text(text, encoding="utf-8")
            preview = text[:120].replace("\n", "↵") if text else "(empty)"
            print(f"  [{prompt_name}] {elapsed:.0f}s  {len(text)} chars  →  {preview}", flush=True)
            summary_lines.append(
                f"{page_label}  {prompt_name:20s}  {elapsed:5.0f}s  {len(text):5d} chars  {preview[:80]}"
            )

    summary = OUT / "summary.txt"
    summary.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\nDone. Results in {OUT}", flush=True)
    print(f"Summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
