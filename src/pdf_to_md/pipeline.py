import hashlib
import json
import math
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from . import logger as _logger_mod
from .cleaner import clean_chunk
from .renderer import extract_page_text, page_count, render_pages
from .stats import RunStats
from .vision import page_to_md, unload_vision_model

PAGE_SEP = "\n\n---\n\n"


def _pdf_hash(pdf_path: Path) -> str:
    h = hashlib.sha256()
    h.update(pdf_path.read_bytes())
    return h.hexdigest()[:16]


def _phase_bar(desc: str, total: int, unit: str) -> tqdm:
    return tqdm(
        total=total,
        desc=f"{desc:<18}",
        unit=unit,
        position=1,
        leave=False,
        dynamic_ncols=True,
        bar_format="{desc} {bar} {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
    )


def run(
    pdf_path: Path,
    output_path: Path,
    vision_model: str,
    vision_mmproj: str,
    text_model: str,
    base_url: str,
    dpi: int,
    clean_chunk_size: int,
    clean_overlap: int,
    cache_dir: Path,
    skip_clean: bool,
    page_range: Optional[tuple[int, int]],
    log_dir: Optional[Path] = None,
    save_images_dir: Optional[Path] = None,
) -> None:
    pdf_path = pdf_path.resolve()

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_log_dir = log_dir or output_path.parent / "logs"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_path.stem
    log = _logger_mod.setup(run_log_dir / f"{prefix}_{ts}.log")

    run_stats = RunStats(
        stats_path=run_log_dir / f"{prefix}_{ts}_stats.csv",
        gpu_path=run_log_dir / f"{prefix}_{ts}_gpu.csv",
    )
    run_stats.start_gpu_monitor()

    log.info("=== pdf-to-md run started ===")
    log.info(f"Input:        {pdf_path}")
    log.info(f"Output:       {output_path}")
    log.info(f"Vision model: {vision_model}")
    log.info(f"Text model:   {text_model}")
    log.info(f"DPI:          {dpi}")

    try:
        _run_inner(
            pdf_path=pdf_path,
            output_path=output_path,
            vision_model=vision_model,
            vision_mmproj=vision_mmproj,
            text_model=text_model,
            base_url=base_url,
            dpi=dpi,
            clean_chunk_size=clean_chunk_size,
            clean_overlap=clean_overlap,
            cache_dir=cache_dir,
            skip_clean=skip_clean,
            page_range=page_range,
            save_images_dir=save_images_dir,
            log=log,
            run_stats=run_stats,
        )
    finally:
        run_stats.stop_gpu_monitor()
        run_stats.save()
        log.info(f"Stats: {run_stats.stats_path}")
        log.info(f"GPU:   {run_stats.gpu_path}")
        log.info("=== run finished ===")


def _run_inner(
    pdf_path, output_path, vision_model, vision_mmproj, text_model, base_url,
    dpi, clean_chunk_size, clean_overlap, cache_dir, skip_clean,
    page_range, save_images_dir, log, run_stats,
):
    cache_key = _pdf_hash(pdf_path)
    page_cache = cache_dir / cache_key
    page_cache.mkdir(parents=True, exist_ok=True)

    meta_path = page_cache / "meta.json"
    if not meta_path.exists():
        meta_path.write_text(
            json.dumps({"pdf": str(pdf_path), "model": vision_model, "dpi": dpi}, indent=2),
            encoding="utf-8",
        )

    total = page_count(pdf_path)
    start = (page_range[0] - 1) if page_range else 0
    end = page_range[1] if page_range else total
    n_pages = end - start

    if start < 0 or end > total or start >= end:
        log.error(f"Invalid page range {start+1}-{end} for {total}-page document")
        raise SystemExit(1)

    step = max(1, clean_chunk_size - clean_overlap)
    n_chunks = math.ceil(n_pages / step) if not skip_clean else 0
    n_phases = 2 if skip_clean else 3

    log.info(f"PDF: {pdf_path.name}  ({total} pages total, processing {start+1}-{end})")

    # ── Overall progress bar ──────────────────────────────────────────────────
    outer = tqdm(
        total=n_phases,
        desc="Overall          ",
        unit="phase",
        position=0,
        leave=True,
        dynamic_ncols=True,
        bar_format="{desc} {bar} {n_fmt}/{total_fmt} phases [{elapsed}]{postfix}",
    )

    # ── Phase 0: Render ───────────────────────────────────────────────────────
    outer.set_postfix_str("Render")
    log.info(f"\n[Phase 0] Rendering pages {start+1}-{end} at {dpi} DPI")
    if save_images_dir:
        save_images_dir.mkdir(parents=True, exist_ok=True)

    run_stats.begin_render_phase()
    page_images: dict[int, bytes] = {}

    with _phase_bar("[0] Render", n_pages, "pg") as inner:
        for page_i, png_bytes in render_pages(pdf_path, dpi=dpi):
            if page_i < start:
                continue
            if page_i >= end:
                break
            t0 = time.time()
            page_images[page_i] = png_bytes
            elapsed = time.time() - t0
            if save_images_dir:
                (save_images_dir / f"page_{page_i:04d}.png").write_bytes(png_bytes)
            run_stats.end_render_page(page_i, elapsed)
            inner.update(1)

    run_stats.end_render_phase()
    log.info(f"  Render done in {run_stats.render_total_sec:.1f}s")
    outer.update(1)

    # ── Phase 1: Vision ───────────────────────────────────────────────────────
    outer.set_postfix_str("Vision")
    log.info(f"\n[Phase 1] Vision  ({n_pages} pages)")
    run_stats.begin_vision_phase()
    pages_md: list[str] = []

    with _phase_bar("[1] Vision", n_pages, "pg") as inner:
        for page_i in range(start, end):
            cache_file = page_cache / f"page_{page_i:04d}.md"

            run_stats.begin_vision_page(page_i)

            if cache_file.exists():
                pages_md.append(cache_file.read_text(encoding="utf-8"))
                run_stats.end_vision_page(0, 0)
                inner.set_postfix_str("cached")
                log.debug(f"  [cache] page {page_i + 1}/{end}")
            else:
                result = page_to_md(
                    page_images[page_i],
                    model=vision_model,
                    mmproj=vision_mmproj,
                    base_url=base_url,
                )
                run_stats.end_vision_page(result.tokens_in, result.tokens_out)
                cache_file.write_text(result.text, encoding="utf-8")
                pages_md.append(result.text)

                ps = run_stats.page_stats[-1]
                inner.set_postfix_str(
                    f"{ps.vision_sec:.0f}s | in={result.tokens_in} out={result.tokens_out} tok"
                )
                log.info(
                    f"  [vision] page {page_i+1}/{end}  {ps.vision_sec:.0f}s  "
                    f"in={result.tokens_in} out={result.tokens_out} tok"
                )

            inner.update(1)

    run_stats.end_vision_phase()
    total_vin = sum(p.tokens_in for p in run_stats.page_stats)
    total_vout = sum(p.tokens_out for p in run_stats.page_stats)
    log.info(
        f"  Vision done in {run_stats.vision_total_sec:.1f}s  "
        f"total in={total_vin} out={total_vout} tok"
    )
    outer.update(1)

    # ── Phase 2: Merge (instant, no bar) ─────────────────────────────────────
    merged = PAGE_SEP.join(pages_md)
    log.info(f"\n[Phase 2] Merge  ({len(pages_md)} pages -> {len(merged)} chars)")
    unload_vision_model()
    time.sleep(15)  # give CUDA time to fully release vision VRAM before loading text model

    if skip_clean:
        output_path.write_text(merged, encoding="utf-8")
        log.info(f"Written (no cleaning): {output_path}")
        outer.close()
        return

    # ── Phase 3: Clean ────────────────────────────────────────────────────────
    outer.set_postfix_str("Clean")
    log.info(
        f"\n[Phase 3] Cleaning with {text_model}  "
        f"(chunk={clean_chunk_size} pages, overlap={clean_overlap}, {n_chunks} chunks)"
    )
    run_stats.begin_clean_phase()
    cleaned_parts: list[str] = []
    chunk_idx = 0
    i = 0

    with _phase_bar("[3] Clean", n_chunks, "chunk") as inner:
        while i < len(pages_md):
            chunk_pages = pages_md[i: i + clean_chunk_size]
            chunk_text = PAGE_SEP.join(chunk_pages)
            page_from = start + i + 1
            page_to = start + i + len(chunk_pages)

            run_stats.begin_clean_chunk(chunk_idx, page_from, page_to)
            log.info(f"  [clean] chunk {chunk_idx+1}/{n_chunks}  pages {page_from}-{page_to} ...")

            result = clean_chunk(chunk_text, model=text_model)
            run_stats.end_clean_chunk(result.tokens_in, result.tokens_out)

            if clean_overlap > 0 and i > 0:
                sections = result.text.split(PAGE_SEP)
                if len(sections) > clean_overlap:
                    result = result._replace(text=PAGE_SEP.join(sections[clean_overlap:]))

            cleaned_parts.append(result.text)

            cs = run_stats.chunk_stats[-1]
            inner.set_postfix_str(
                f"{cs.clean_sec:.0f}s | in={result.tokens_in} out={result.tokens_out} tok"
            )
            log.info(
                f"  [clean] chunk {chunk_idx+1}/{n_chunks}  done  {cs.clean_sec:.0f}s  "
                f"in={result.tokens_in} out={result.tokens_out} tok"
            )

            inner.update(1)
            i += step
            chunk_idx += 1

    run_stats.end_clean_phase()
    total_cin = sum(c.tokens_in for c in run_stats.chunk_stats)
    total_cout = sum(c.tokens_out for c in run_stats.chunk_stats)
    log.info(
        f"  Cleaning done in {run_stats.clean_total_sec:.1f}s  "
        f"total in={total_cin} out={total_cout} tok"
    )
    outer.update(1)
    outer.close()

    output_path.write_text(PAGE_SEP.join(cleaned_parts), encoding="utf-8")
    log.info(f"\nWritten: {output_path}")
