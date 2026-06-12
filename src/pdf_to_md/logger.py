import logging
import sys
from pathlib import Path


class _TqdmHandler(logging.StreamHandler):
    """Routes console log output through tqdm.write() to avoid bar corruption."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm
            tqdm.write(self.format(record))
        except Exception:
            super().emit(record)


def setup(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("pdf_to_md")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    ))

    console_handler = _TqdmHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    # Silence UnicodeEncodeError on Windows cp1251 consoles
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def get() -> logging.Logger:
    return logging.getLogger("pdf_to_md")
