import argparse
import json
import logging
import sys
from pathlib import Path

from . import logger


_LLM_HELP_TEXT = """# pdf-to-md — справка для LLM

## Что делает
Конвертирует PDF в Markdown через локальные LLM-модели (GGUF) с GPU inference (CUDA).
Pipeline: Render (PDF→PNG) → Vision (PNG→MD на страницу) → Merge → Clean (опционально).

## Модели (нужны GGUF-файлы)
- Vision: Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf + mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf
- Cleaning: gemma-3-12b-it-heretic-Q5_K_M.gguf (опционально, с --no-clean пропускается)

## Железо
RTX 4070 Ti (12 ГБ VRAM). Vision: ~6.2 ГБ. Cleaning: ~9.3 ГБ. Последовательно.
~60 сек/страницу vision inference.

## Субкоманды

### pdf-to-md convert INPUT.pdf [OUTPUT.md]
  --vision-model FILE       путь к GGUF vision-модели (обязателен)
  --vision-mmproj FILE      путь к mmproj GGUF (обязателен)
  --text-model FILE         путь к GGUF cleaning-модели (обязателен без --no-clean)
  --ollama-url URL          только для выгрузки Ollama-моделей (default: http://localhost:11434)
  --dpi N                   DPI рендеринга (default: 200, range: 72-600)
  --pages N-M               диапазон страниц (напр. 17-19, 1-indexed)
  --clean-chunk N           страниц в chunk для cleaning (default: 10)
  --clean-overlap K         overlap страниц (default: 2)
  --no-clean                пропустить фазу cleaning
  --cache-dir DIR           (default: .pdf_to_md_cache)
  --log-dir DIR             (default: <output-dir>/logs)
  --save-images DIR         сохранить PNG страниц после рендера
  --force                   перезаписать output если существует

### pdf-to-md detect [FILES...]
  --text-model FILE         GGUF для gemma-проверки (default: models/gemma-3-12b-...)
  --report FILE             путь к отчёту (default: output/hallucination_report.txt)
  --no-gemma                только эвристики, без gemma (быстро)
  FILES                     MD файлы для проверки (default: output/*.md)

### pdf-to-md clear-cache
  --cache-dir DIR           (default: .pdf_to_md_cache)

### pdf-to-md llm-help
  (эта справка)

## Кэш
Страницы кэшируются в .pdf_to_md_cache/<sha256(pdf)[:16]>/page_NNNN.md.
Повторный запуск пропускает закэшированные страницы.
Кэш не учитывает DPI — при смене DPI удалить page_NNNN.md вручную.

## Известные ограничения
- Cleaning временно отключён (--no-clean) — промпт gemma удаляет разделители страниц
- Detect hallucinations: MODEL захардкожен в скрипте, переопределяется через --text-model
"""


def _console_logger() -> logging.Logger:
    cli_logger = logger.get()
    if not cli_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        cli_logger.addHandler(handler)
        cli_logger.setLevel(logging.INFO)
    return cli_logger


def _parse_page_range(s: str) -> tuple[int, int]:
    try:
        if "-" in s:
            a, b = s.split("-", 1)
            start, end = int(a), int(b)
        else:
            start = end = int(s)
        if start < 1 or end < start:
            raise ValueError
        return start, end
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid page range {s!r}. Use a page number (5) or range (3-10)."
        )


def _existing_file(s: str) -> str:
    p = Path(s)
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {s}")
    return s


def _fail(message: str) -> None:
    _console_logger().info(message)
    raise SystemExit(1)


def _add_convert_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "convert",
        help="Convert a PDF to Markdown",
        description="Convert PDF to Markdown using local llama-cpp-python models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pdf-to-md convert paper.pdf
  pdf-to-md convert paper.pdf out.md --pages 1-20 --no-clean
  pdf-to-md convert book.pdf out.md \\
      --vision-model Qwen2-VL-7B.gguf --vision-mmproj mmproj.gguf \\
      --text-model gemma-3-12b.gguf
""",
    )
    p.set_defaults(handler=_cmd_convert)

    p.add_argument("input", type=Path, help="Input PDF file")
    p.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output Markdown file (default: input filename with .md extension)",
    )
    p.add_argument("--force", action="store_true", help="Overwrite output file if it exists")

    models = p.add_argument_group("models")
    models.add_argument(
        "--vision-model",
        default=None,
        type=_existing_file,
        metavar="FILE",
        help="Path to vision GGUF model file (required)",
    )
    models.add_argument(
        "--vision-mmproj",
        default=None,
        type=_existing_file,
        metavar="FILE",
        help="Path to mmproj GGUF file (required with --vision-model)",
    )
    models.add_argument(
        "--text-model",
        default=None,
        type=_existing_file,
        metavar="FILE",
        help="Path to text/cleaning GGUF model file",
    )
    models.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        metavar="URL",
        help="Ollama base URL used only to unload models from VRAM (default: http://localhost:11434)",
    )

    render = p.add_argument_group("rendering")
    render.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PDF render resolution in DPI (default: 200)",
    )
    render.add_argument(
        "--pages",
        type=_parse_page_range,
        metavar="N-M",
        help="Process only pages N through M (1-indexed, e.g. 5-20)",
    )

    cleaning = p.add_argument_group("cleaning")
    cleaning.add_argument(
        "--clean-chunk",
        type=int,
        default=10,
        metavar="N",
        help="Pages per cleaning chunk (default: 10)",
    )
    cleaning.add_argument(
        "--clean-overlap",
        type=int,
        default=2,
        metavar="K",
        help="Overlap pages between cleaning chunks (default: 2)",
    )
    cleaning.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip the cleaning step",
    )

    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".pdf_to_md_cache"),
        metavar="DIR",
        help="Per-page MD cache directory (default: .pdf_to_md_cache)",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Log and stats output directory (default: <output-dir>/logs)",
    )
    p.add_argument(
        "--save-images",
        type=Path,
        default=None,
        metavar="DIR",
        help="Save rendered page PNGs to this directory (Phase 0 output)",
    )


def _add_detect_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "detect",
        help="Detect hallucinations in Markdown files",
    )
    p.set_defaults(handler=_cmd_detect)
    p.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="MD files (default: output/*.md)",
    )
    p.add_argument(
        "--text-model",
        default="models/gemma-3-12b-it-heretic-Q5_K_M.gguf",
        metavar="FILE",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("output/hallucination_report.txt"),
    )
    p.add_argument(
        "--no-gemma",
        action="store_true",
        help="Heuristics only, without gemma",
    )


def _add_clear_cache_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "clear-cache",
        help="Delete hallucinated pages from the cache",
    )
    p.set_defaults(handler=_cmd_clear_cache)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".pdf_to_md_cache"),
    )


def _cmd_convert(args: argparse.Namespace) -> None:
    if not args.input.exists():
        _fail(f"Error: file not found: {args.input}")
    if args.input.suffix.lower() != ".pdf":
        _fail(f"Error: input must be a .pdf file, got: {args.input.suffix}")
    if args.vision_model is None:
        _fail("Error: --vision-model is required.")
    if args.vision_mmproj is None:
        _fail("Error: --vision-mmproj is required.")
    if not args.no_clean and args.text_model is None:
        _fail("Error: --text-model is required (or use --no-clean).")

    if args.output is None:
        args.output = args.input.with_suffix(".md")

    if args.output.exists() and not args.force:
        _console_logger().info(f"Error: output file already exists: {args.output}")
        _fail("Use --force to overwrite.")
    if args.dpi < 72 or args.dpi > 600:
        _fail("Error: --dpi must be between 72 and 600")
    if args.clean_chunk < 1:
        _fail("Error: --clean-chunk must be >= 1")
    if args.clean_overlap < 0 or args.clean_overlap >= args.clean_chunk:
        _fail("Error: --clean-overlap must be >= 0 and < --clean-chunk")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    from .pipeline import run

    run(
        pdf_path=args.input,
        output_path=args.output,
        vision_model=args.vision_model,
        vision_mmproj=args.vision_mmproj or "",
        text_model=args.text_model or "",
        base_url=args.ollama_url,
        dpi=args.dpi,
        clean_chunk_size=args.clean_chunk,
        clean_overlap=args.clean_overlap,
        cache_dir=args.cache_dir,
        skip_clean=args.no_clean,
        page_range=args.pages,
        log_dir=args.log_dir,
        save_images_dir=args.save_images,
    )


def _cmd_detect(args: argparse.Namespace) -> None:
    from .hallucinations import (
        PAGE_SEP,
        _SIZE_CAP,
        _SIZE_K,
        _page_stats,
        gemma_check,
        heuristic_check,
        load_model,
        size_deviation_check,
    )

    cli_logger = _console_logger()
    paths = args.files or sorted(Path("output").glob("*.md"))
    if not paths:
        cli_logger.info(
            "No .md files found in output/. Pass file paths as arguments."
        )
        return

    llm = None if args.no_gemma else load_model(args.text_model)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    all_flagged: list[tuple[str, int, int, str, str]] = []

    with args.report.open("w", encoding="utf-8") as rep:
        for path in paths:
            if not path.exists():
                cli_logger.info(f"SKIP: {path} not found")
                continue

            label = path.stem
            text = path.read_text(encoding="utf-8")
            pages = text.split(PAGE_SEP)
            line_starts: list[int] = []
            current_line = 1
            for i, page in enumerate(pages):
                line_starts.append(current_line)
                current_line += page.count("\n") + (4 if i < len(pages) - 1 else 0)

            mean_len, std_len = _page_stats(pages)
            sigma_eff = min(std_len, _SIZE_CAP)
            lo = mean_len - _SIZE_K * sigma_eff
            hi = mean_len + _SIZE_K * sigma_eff

            cli_logger.info(
                f"\n[{label}] {len(pages)} pages  |  "
                f"mean {mean_len:.0f}  std {std_len:.0f} (cap {sigma_eff:.0f})  "
                f"window [{lo:.0f}, {hi:.0f}]"
            )
            rep.write(
                f"\n=== {label} ({len(pages)} pages, "
                f"mean {mean_len:.0f}, σ {std_len:.0f}, "
                f"window [{lo:.0f}, {hi:.0f}]) ===\n"
            )
            part_count = 0

            for i, page in enumerate(pages):
                pnum = i + 1
                lnum = line_starts[i]
                stripped = page.strip()
                if len(stripped) < 30:
                    continue

                is_hall, reason = size_deviation_check(stripped, mean_len, std_len)
                method = "size    "
                if not is_hall:
                    is_hall, reason = heuristic_check(stripped)
                    method = "heuristic"
                if not is_hall and llm is not None:
                    is_hall, reason = gemma_check(llm, stripped)
                    method = "gemma   "

                tag = "HALL" if is_hall else "ok  "
                cli_logger.info(
                    f"  p{pnum:4d} line~{lnum:5d}: "
                    f"{tag} [{method}] {reason[:65]}"
                )

                if is_hall:
                    entry = (
                        f"  page {pnum:4d}  line~{lnum:5d}  "
                        f"[{method.strip()}]  {reason}"
                    )
                    rep.write(entry + "\n")
                    rep.flush()
                    all_flagged.append(
                        (label, pnum, lnum, method.strip(), reason)
                    )
                    part_count += 1

            rep.write(f"  → {part_count} hallucination pages\n")
            cli_logger.info(f"  [{label}] flagged: {part_count}")

        rep.write(f"\n\nTOTAL: {len(all_flagged)} hallucination pages\n\n")
        for label, pnum, lnum, method, reason in all_flagged:
            rep.write(
                f"  {label}  page {pnum:4d}  line~{lnum:5d}  "
                f"[{method}]  {reason}\n"
            )

    cli_logger.info(f"\nDone. Total flagged: {len(all_flagged)}")
    cli_logger.info(f"Report: {args.report}")


def _cmd_clear_cache(args: argparse.Namespace) -> None:
    from .cache_quality import is_bad

    cli_logger = _console_logger()
    if not args.cache_dir.exists():
        cli_logger.info("Cache directory not found.")
        return

    total_checked = 0
    total_deleted = 0

    for hash_dir in sorted(args.cache_dir.iterdir()):
        if not hash_dir.is_dir():
            continue

        meta_path = hash_dir / "meta.json"
        pdf_name = "?"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            pdf_name = Path(meta.get("pdf", "?")).name

        page_files = sorted(hash_dir.glob("page_*.md"))
        cli_logger.info(
            f"\n{hash_dir.name}  ({pdf_name})  {len(page_files)} cached pages"
        )

        deleted = 0
        for path in page_files:
            text = path.read_text(encoding="utf-8")
            bad, reason = is_bad(text)
            if bad:
                cli_logger.info(f"  DELETE {path.name}  {reason}")
                path.unlink()
                deleted += 1

        cli_logger.info(f"  => deleted {deleted}/{len(page_files)}")
        total_checked += len(page_files)
        total_deleted += deleted

    cli_logger.info(
        f"\nDone. Deleted {total_deleted}/{total_checked} cache files."
    )


def _cmd_llm_help(args: argparse.Namespace) -> None:
    del args
    _console_logger().info(_LLM_HELP_TEXT)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-to-md",
        description="Convert PDFs and inspect Markdown quality.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_convert_parser(subparsers)
    _add_detect_parser(subparsers)
    _add_clear_cache_parser(subparsers)

    llm_help = subparsers.add_parser(
        "llm-help",
        help="Show project usage notes for LLM agents",
    )
    llm_help.set_defaults(handler=_cmd_llm_help)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
