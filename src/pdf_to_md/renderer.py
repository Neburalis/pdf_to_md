from pathlib import Path
from typing import Iterator

import fitz  # pymupdf


def render_pages(pdf_path: Path, dpi: int = 200) -> Iterator[tuple[int, bytes]]:
    """Yield (page_index, png_bytes) for each page."""
    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat)
            yield i, pix.tobytes("jpg")
    finally:
        doc.close()


def extract_page_text(pdf_path: Path, page_index: int) -> str:
    doc = fitz.open(str(pdf_path))
    try:
        text = doc[page_index].get_text("text")
    finally:
        doc.close()
    return text.strip()


def page_count(pdf_path: Path) -> int:
    doc = fitz.open(str(pdf_path))
    n = doc.page_count
    doc.close()
    return n
