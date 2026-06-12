import argparse
import sys
from pathlib import Path

from .pipeline import run


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


def main() -> None:
    p = argparse.ArgumentParser(
        prog="pdf-to-md",
        description="Convert PDF to Markdown using local llama-cpp-python models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pdf-to-md paper.pdf
  pdf-to-md paper.pdf out.md --pages 1-20 --no-clean
  pdf-to-md book.pdf out.md \\
      --vision-model Qwen2-VL-7B.gguf --vision-mmproj mmproj.gguf \\
      --text-model gemma-3-12b.gguf
""",
    )

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
        "--dpi", type=int, default=200,
        help="PDF render resolution in DPI (default: 200)",
    )
    render.add_argument(
        "--pages", type=_parse_page_range, metavar="N-M",
        help="Process only pages N through M (1-indexed, e.g. 5-20)",
    )

    cleaning = p.add_argument_group("cleaning")
    cleaning.add_argument(
        "--clean-chunk", type=int, default=10, metavar="N",
        help="Pages per cleaning chunk (default: 10)",
    )
    cleaning.add_argument(
        "--clean-overlap", type=int, default=2, metavar="K",
        help="Overlap pages between cleaning chunks (default: 2)",
    )
    cleaning.add_argument(
        "--no-clean", action="store_true",
        help="Skip the cleaning step",
    )

    p.add_argument(
        "--cache-dir", type=Path, default=Path(".pdf_to_md_cache"), metavar="DIR",
        help="Per-page MD cache directory (default: .pdf_to_md_cache)",
    )
    p.add_argument(
        "--log-dir", type=Path, default=None, metavar="DIR",
        help="Log and stats output directory (default: <output-dir>/logs)",
    )
    p.add_argument(
        "--save-images", type=Path, default=None, metavar="DIR",
        help="Save rendered page PNGs to this directory (Phase 0 output)",
    )

    args = p.parse_args()

    # ── Validate input ────────────────────────────────────────────────────────
    if not args.input.exists():
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if args.input.suffix.lower() != ".pdf":
        print(f"Error: input must be a .pdf file, got: {args.input.suffix}", file=sys.stderr)
        sys.exit(1)

    if args.vision_model is None:
        print("Error: --vision-model is required.", file=sys.stderr)
        sys.exit(1)
    if args.vision_mmproj is None:
        print("Error: --vision-mmproj is required.", file=sys.stderr)
        sys.exit(1)
    if not args.no_clean and args.text_model is None:
        print("Error: --text-model is required (or use --no-clean).", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        args.output = args.input.with_suffix(".md")

    if args.output.exists() and not args.force:
        print(f"Error: output file already exists: {args.output}", file=sys.stderr)
        print("Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    if args.dpi < 72 or args.dpi > 600:
        print("Error: --dpi must be between 72 and 600", file=sys.stderr)
        sys.exit(1)
    if args.clean_chunk < 1:
        print("Error: --clean-chunk must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.clean_overlap < 0 or args.clean_overlap >= args.clean_chunk:
        print("Error: --clean-overlap must be >= 0 and < --clean-chunk", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    main()
