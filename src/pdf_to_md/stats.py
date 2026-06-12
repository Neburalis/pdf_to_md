import csv
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PageStat:
    page_index: int
    render_sec: float = 0.0
    vision_sec: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class ChunkStat:
    chunk_index: int
    page_from: int
    page_to: int
    clean_sec: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


class RunStats:
    def __init__(self, stats_path: Path, gpu_path: Path, gpu_interval: float = 5.0):
        self.stats_path = stats_path
        self.gpu_path = gpu_path
        self.gpu_interval = gpu_interval

        self.page_stats: list[PageStat] = []
        self.chunk_stats: list[ChunkStat] = []

        self.render_total_sec: float = 0.0
        self.vision_total_sec: float = 0.0
        self.clean_total_sec: float = 0.0

        self._t0: float = 0.0
        self._current_page: Optional[PageStat] = None
        self._current_chunk: Optional[ChunkStat] = None

        self._gpu_stop = threading.Event()
        self._gpu_thread = threading.Thread(target=self._gpu_loop, daemon=True)

    # --- GPU monitoring ---

    def start_gpu_monitor(self) -> None:
        self.gpu_path.parent.mkdir(parents=True, exist_ok=True)
        self._gpu_thread.start()

    def stop_gpu_monitor(self) -> None:
        self._gpu_stop.set()
        self._gpu_thread.join(timeout=15)

    def _gpu_loop(self) -> None:
        with open(self.gpu_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "temp_c", "gpu_pct", "vram_used_mb", "vram_total_mb"])
            f.flush()
            while not self._gpu_stop.wait(self.gpu_interval):
                row = _query_gpu()
                if row:
                    w.writerow(row)
                    f.flush()

    # --- render phase ---

    def begin_render_phase(self) -> None:
        self._t0 = time.time()

    def end_render_page(self, page_index: int, elapsed: float) -> None:
        ps = self._get_or_create_page(page_index)
        ps.render_sec = elapsed

    def end_render_phase(self) -> None:
        self.render_total_sec = time.time() - self._t0

    # --- vision phase ---

    def begin_vision_phase(self) -> None:
        self._t0 = time.time()

    def begin_vision_page(self, page_index: int) -> None:
        ps = self._get_or_create_page(page_index)
        self._current_page = ps
        ps._vision_t0 = time.time()  # type: ignore[attr-defined]

    def end_vision_page(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        if self._current_page:
            t0 = getattr(self._current_page, "_vision_t0", time.time())
            self._current_page.vision_sec = time.time() - t0
            self._current_page.tokens_in = tokens_in
            self._current_page.tokens_out = tokens_out

    def end_vision_phase(self) -> None:
        self.vision_total_sec = time.time() - self._t0

    # --- clean phase ---

    def begin_clean_phase(self) -> None:
        self._t0 = time.time()

    def begin_clean_chunk(self, chunk_index: int, page_from: int, page_to: int) -> None:
        cs = ChunkStat(chunk_index=chunk_index, page_from=page_from, page_to=page_to)
        cs._clean_t0 = time.time()  # type: ignore[attr-defined]
        self._current_chunk = cs
        self.chunk_stats.append(cs)

    def end_clean_chunk(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        if self._current_chunk:
            t0 = getattr(self._current_chunk, "_clean_t0", time.time())
            self._current_chunk.clean_sec = time.time() - t0
            self._current_chunk.tokens_in = tokens_in
            self._current_chunk.tokens_out = tokens_out

    def end_clean_phase(self) -> None:
        self.clean_total_sec = time.time() - self._t0

    # --- save ---

    def save(self) -> None:
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.stats_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "phase", "index", "page_from", "page_to",
                "duration_sec", "tokens_in", "tokens_out",
                "tok_in_per_sec", "tok_out_per_sec",
            ])

            # Render per page
            render_total = self.render_total_sec or sum(p.render_sec for p in self.page_stats)
            for ps in self.page_stats:
                w.writerow(["render_page", ps.page_index, ps.page_index, ps.page_index,
                             f"{ps.render_sec:.2f}", "", "", "", ""])
            w.writerow(["render_total", "", "", "", f"{render_total:.2f}", "", "", "", ""])

            # Vision per page
            for ps in self.page_stats:
                tps_in = _rate(ps.tokens_in, ps.vision_sec)
                tps_out = _rate(ps.tokens_out, ps.vision_sec)
                w.writerow(["vision_page", ps.page_index, ps.page_index, ps.page_index,
                             f"{ps.vision_sec:.2f}", ps.tokens_in, ps.tokens_out,
                             f"{tps_in:.1f}", f"{tps_out:.1f}"])
            total_vin = sum(p.tokens_in for p in self.page_stats)
            total_vout = sum(p.tokens_out for p in self.page_stats)
            w.writerow(["vision_total", "", "", "", f"{self.vision_total_sec:.2f}",
                         total_vin, total_vout,
                         f"{_rate(total_vin, self.vision_total_sec):.1f}",
                         f"{_rate(total_vout, self.vision_total_sec):.1f}"])

            # Clean per chunk
            for cs in self.chunk_stats:
                tps_in = _rate(cs.tokens_in, cs.clean_sec)
                tps_out = _rate(cs.tokens_out, cs.clean_sec)
                w.writerow(["clean_chunk", cs.chunk_index, cs.page_from, cs.page_to,
                             f"{cs.clean_sec:.2f}", cs.tokens_in, cs.tokens_out,
                             f"{tps_in:.1f}", f"{tps_out:.1f}"])
            total_cin = sum(c.tokens_in for c in self.chunk_stats)
            total_cout = sum(c.tokens_out for c in self.chunk_stats)
            w.writerow(["clean_total", "", "", "", f"{self.clean_total_sec:.2f}",
                         total_cin, total_cout,
                         f"{_rate(total_cin, self.clean_total_sec):.1f}",
                         f"{_rate(total_cout, self.clean_total_sec):.1f}"])

            # Cloud cost estimate (GPT-4o pricing as reference)
            total_in = total_vin + total_cin
            total_out = total_vout + total_cout
            cost_in = total_in / 1_000_000 * 2.50   # $2.50 / 1M input tokens
            cost_out = total_out / 1_000_000 * 10.00  # $10.00 / 1M output tokens
            w.writerow([])
            w.writerow(["# Cloud cost estimate (GPT-4o rates)", "", "", "",
                         "", total_in, total_out, "", ""])
            w.writerow(["# cost_usd", "", "", "", f"${cost_in + cost_out:.4f}",
                         f"in=${cost_in:.4f}", f"out=${cost_out:.4f}", "", ""])

    # --- helpers ---

    def _get_or_create_page(self, page_index: int) -> PageStat:
        for ps in self.page_stats:
            if ps.page_index == page_index:
                return ps
        ps = PageStat(page_index=page_index)
        self.page_stats.append(ps)
        return ps


def _query_gpu() -> Optional[list]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
            text=True,
        ).strip()
        temp, util, used, total = [x.strip() for x in out.split(",")]
        return [time.strftime("%H:%M:%S"), temp, util, used, total]
    except Exception:
        return None


def _rate(tokens: int, sec: float) -> float:
    return tokens / sec if sec > 0 else 0.0
