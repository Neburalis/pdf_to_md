# pdf-to-md — архитектура

## Pipeline

```
PDF
 │
 ▼ Phase 0: Render (renderer.py)
PNG per page ──────────────────────────────────── [--save-images DIR]
 │
 ▼ Phase 1: Vision (vision.py + llama-cpp-python)
MD per page ────────────────────────────────────── cache: .pdf_to_md_cache/<hash>/page_NNNN.md
 │
 ▼ Phase 2: Merge (pipeline.py)
merged MD  (pages joined with "\n\n---\n\n")
 │  └── unload_vision_model() ← освобождает VRAM
 ▼ Phase 3: Clean (cleaner.py + llama-cpp-python)   [--no-clean пропускает]
final MD
 │
 ▼ logs/
    <name>_<ts>.log          # полный лог
    <name>_<ts>_stats.csv    # тайминги + токены + оценка стоимости
    <name>_<ts>_gpu.csv      # GPU temp/util/VRAM каждые 5 сек
```

---

## Структура файлов

```
pdf_to_md/
├── src/pdf_to_md/
│   ├── __main__.py    # точка входа: python -m pdf_to_md
│   ├── cli.py         # argparse + валидация путей к GGUF
│   ├── pipeline.py    # оркестратор фаз, tqdm прогресс-бары
│   ├── renderer.py    # PDF → PNG (PyMuPDF), extract_page_text()
│   ├── vision.py      # загрузка Qwen2-VL, page_to_md() → VisionResult
│   ├── cleaner.py     # загрузка text-модели, clean_chunk() → CleanResult
│   ├── stats.py       # RunStats: тайминги, токены, GPU-мониторинг (фоновый поток)
│   └── logger.py      # setup(): file handler + TqdmHandler для консоли
├── pdf/               # входные PDF
├── output/            # выходные MD + logs/
├── .pdf_to_md_cache/  # кэш: sha256(pdf)[:16] / page_NNNN.md
├── pyproject.toml
├── CLAUDE.md
├── PROJECT.md         # описание проекта, команды, нюансы
└── ARCHITECTURE.md    # этот файл
```

---

## Модули

### `pipeline.py` — оркестратор

Создаёт `RunStats` и `Logger`, запускает GPU-монитор, затем последовательно:

```
Phase 0  render_pages(pdf, dpi)
           → page_images dict[int, bytes]
           → опционально записывает PNG на диск

Phase 1  for page_i in range(start, end):
           cache_file = page_cache / f"page_{page_i:04d}.md"
           if cached: load from cache
           else: page_to_md(png, model, mmproj, text_hint) → VisionResult
         stats.end_vision_page(tokens_in, tokens_out)

         unload_vision_model()   ← del + gc.collect()

Phase 3  while i < len(pages_md):
           chunk = pages_md[i : i + chunk_size]
           clean_chunk(chunk_text, model) → CleanResult
           drop overlap prefix from result
         stats.end_clean_chunk(tokens_in, tokens_out)
```

**tqdm:** два бара — `Overall (N/3 phases)` на position=0 и `[Phase] (n/N pages)` на position=1. Лог-сообщения идут через `tqdm.write()` (TqdmHandler).

**Кэш-ключ:** `sha256(pdf_bytes)[:16]`.

---

### `renderer.py` — PyMuPDF

| Функция | Возвращает |
|---------|-----------|
| `render_pages(pdf, dpi)` | `Iterator[(page_index, png_bytes)]` |
| `extract_page_text(pdf, page_i)` | `str` — текст-подсказка для vision |
| `page_count(pdf)` | `int` |

---

### `vision.py` — vision backend

**Единственный бэкенд: llama-cpp-python.**

```python
page_to_md(png_bytes, model, mmproj, base_url, text_hint) -> VisionResult(text, tokens_in, tokens_out)
```

Цепочка вызовов при загрузке:
1. `unload_ollama_models(base_url)` — `POST /api/generate {keep_alive:0}` для каждой загруженной Ollama-модели
2. `_register_cuda_dll_dirs()` — `os.add_dll_directory` + `ctypes.CDLL` для CUDA/ggml DLL (Windows)
3. `Qwen25VLChatHandler(mmproj, verbose=False)` + `Llama(model, n_ctx=4096, n_gpu_layers=-1)`

Всё обёрнуто в `suppress_stdout_stderr()` — шум llama.cpp не виден в консоли.

После каждой страницы: `llm.reset()` — сброс KV-кэша.

`unload_vision_model()`: `del _vision_llama; gc.collect()` — освобождает VRAM.

---

### `cleaner.py` — text backend

**Единственный бэкенд: llama-cpp-python.**

```python
clean_chunk(text, model) -> CleanResult(text, tokens_in, tokens_out)
```

Загрузка: `_register_cuda_dll_dirs()` → `Llama(model, n_ctx=8192, n_gpu_layers=-1)` в `suppress_stdout_stderr()`.

`_strip_code_fences()`: убирает внешние и внутренние ` ```markdown ``` ` блоки (gemma3 иногда их добавляет).

**Chunking:**
```
step = chunk_size - overlap
chunk 0: pages[0 : chunk_size]
chunk 1: pages[step : step + chunk_size]   ← первые overlap страниц отрезаются из результата
...
```

---

### `stats.py` — RunStats

Собирает статистику без блокировки основного потока.

| Метод | Действие |
|-------|---------|
| `start_gpu_monitor()` | Запускает daemon-поток, пишет GPU CSV каждые 5 сек |
| `begin/end_render_phase()` | Общее время рендера |
| `begin/end_vision_page(page_i)` | Время + токены на страницу |
| `begin/end_clean_chunk(idx, from, to)` | Время + токены на chunk |
| `save()` | Пишет stats CSV + оценку стоимости по ценам GPT-4o |

GPU CSV: `time, temp_c, gpu_pct, vram_used_mb, vram_total_mb` — через `nvidia-smi`.

Stats CSV дополнительно включает строки:
```
# Cloud cost estimate (GPT-4o rates)  total_in  total_out
# cost_usd  $X.XXXX  in=$X.XXXX  out=$X.XXXX
```

---

### `logger.py`

`setup(log_path)` → `logging.Logger("pdf_to_md")` с двумя handlers:
- `FileHandler(log_path, encoding="utf-8")` — DEBUG+, формат `HH:MM:SS  LEVEL  message`
- `TqdmHandler` — INFO+, пишет через `tqdm.tqdm.write()` чтобы не ломать прогресс-бары

---

## CLI

```
pdf-to-md INPUT [OUTPUT]
  --vision-model FILE    путь к GGUF vision-модели (обязателен)
  --vision-mmproj FILE   путь к mmproj GGUF (обязателен)
  --text-model FILE      путь к GGUF text-модели (обязателен без --no-clean)
  --ollama-url URL       только для unload_ollama_models (default: http://localhost:11434)
  --dpi N                DPI рендеринга (default: 200, range: 72-600)
  --pages N-M            диапазон страниц (1-indexed)
  --clean-chunk N        страниц в chunk (default: 10)
  --clean-overlap K      overlap страниц (default: 2, < chunk)
  --no-clean             пропустить фазу cleaning
  --force                перезаписать output
  --cache-dir DIR        (default: .pdf_to_md_cache)
  --log-dir DIR          (default: <output-dir>/logs)
  --save-images DIR      сохранить PNG страниц после рендера
```

---

## Зависимости

```toml
pymupdf>=1.24.0               # рендеринг PDF, извлечение текста
llama-cpp-python==0.3.23      # cu124 wheel; включает mtmd.dll (новый multimodal API)
nvidia-cuda-runtime-cu12      # cudart64_12.dll и др. (Windows DLL search path)
nvidia-cublas-cu12
requests>=2.31.0              # unload_ollama_models()
tqdm>=4.66.0                  # прогресс-бары
gguf>=0.19.0                  # утилиты (gguf-dump)
```

---

## Changelog

### 2026-06-11 — n_ctx=4096: DPI=200 без OOM

**Проблема:** pipeline работал только при `--dpi 90` — при DPI=200 vision encoder падал с CUDA OOM из-за фрагментации VRAM (max_contiguous=374 МБ < ~800 МБ compute buffer).

**Решение:** `n_ctx=4096` вместо 8192 в `_load_vision_model()`. KV cache: 940 МБ → 470 МБ → достаточно места для vision encoder при DPI=200.

**Результат:** 3 страницы при DPI=200 — 45.1s, no OOM. Стандартный DPI=200 теперь рабочий.

**Отвергнутые альтернативы:** giant malloc pre-warm, image-first в промпте — не потребовались.

---

### 2026-06-11 — VRAM OOM: 8 фиксов для Qwen2.5-VL-7B на RTX 4070 Ti

**Проблема:** pipeline падал с CUDA OOM при запуске vision inference. Несколько вложенных причин:
1. Warmup mmproj пытался аллоцировать 704 МБ → провал из-за фрагментации после загрузки модели
2. CUDA graph warmup при первом decode фрагментировал VRAM до max_contiguous=374 МБ
3. Vision encoder при DPI=200 требовал ~800 МБ → не помещался
4. Дополнительно: `_log_vram_state` binary search (cudaMalloc до 5120 МБ) фрагментировал VRAM → OOM в matmul

**Решение:** 8 изменений в `vision.py` и `llama_chat_format.py`:
- `warmup=False` в `_init_mtmd_context` (llama_chat_format.py)
- `flash_attn=True`, `n_batch=64` в конструкторе Llama
- `suppress_stdout_stderr(disable=False)` для inference
- Удалён вызов `_log_vram_state` после загрузки модели
- `GGML_CUDA_NO_GRAPHS=1` — нельзя устанавливать (снижает max_contiguous до 4032 МБ при init)
- `_strip_repeated_paragraphs()` — дедупликация: обрезает при первом повторяющемся блоке
- Снижен DPI до 90 (compute buffer ~330 МБ < 374 МБ max_contiguous)

**Результат:** 240 стр. термодинамика (~34 мин), 561 стр. сборник задач (~97 мин). Формулы корректны. Подробнее — в `EXPERIMENTS.md`.

**Отвергнутые альтернативы:**
- `GGML_CUDA_NO_GRAPHS=1` — фрагментирует при DLL init, модель не грузится (нужно 4168 МБ, остаётся 4032)
- `repeat_penalty=1.3` — перестаёт зацикливаться, но ухудшает формулы
- DPI=200 без других изменений — OOM (800 МБ > 374 МБ max_contiguous)

---

### 2026-06-05 — Детектор галлюцинаций: size_deviation_check на основе σ с кэпом

**Проблема:** детектор не ловил страницы, где модель почти ничего не вывела (50–200 chars при типичных 1500), и длинные петли вне threshold повторяемости. Фиксированные пороги (25%/300%) не адаптировались к контенту.

**Решение:** `size_deviation_check(page, mean_len, std_len)` с адаптивным коридором `[mean − K·σ, mean + K·σ]`, K=2.0; σ кэпирован на 500 chars. Порядок проверок: size (O(1)) → heuristic → gemma. Константы `_SIZE_K` и `_SIZE_CAP` вынесены наверх. `main()` принимает пути файлами аргументами; без аргументов — `output/*.md`.

**Результат:** скрипт выводит `mean σ window` в начале каждого файла; ложные срабатывания на обложках ожидаемы и допустимы.

**Отвергнутые альтернативы:** фиксированные 25%/300% — не учитывали естественный разброс; медиана без σ — не даёт адаптивного порога.

---

### 2026-06-03 — Система детекции галлюцинаций (новые скрипты)

**Проблема:** вручную проверить 861 страницу невозможно.

**Решение:** `src/scripts/detect_hallucinations.py` — двухпроходный анализ (эвристики → gemma); `src/scripts/clear_bad_cache.py` — удаляет плохие файлы кэша для точечной перегенерации.

**Результат:** 83/861 страниц выявлено как галлюцинации (9.6%), pipeline перегенерировал их при следующем запуске без полного перезапуска.

---

### 2026-05-30 — VRAM management: последовательная загрузка моделей

**Проблема:** vision (~6.4 ГБ) и cleaner (~9.3 ГБ) не помещаются в 12 ГБ одновременно.

**Решение:** `unload_vision_model()` (del + gc.collect()) в конце Phase 1, затем `time.sleep(15)` — пауза нужна, чтобы CUDA успела освободить VRAM до загрузки cleaner.

**Отвергнутые альтернативы:** одновременная загрузка обеих моделей — невозможно физически; выгрузка через Ollama API — Ollama удалён из проекта.

---

### 2026-05-30 — Первый запуск pipeline

**Результат:** Часть 1 (278 страниц) обработана vision. VRAM overflow при загрузке cleaner — исправлено Q4 KV-кэшем (см. запись Q4 KV-кэш в AI.md).

---

### 2026-05-30 — Ollama backend удалён

**Проблема:** Ollama v0.24.0 использует устаревший llama.cpp, не умеет загружать mmproj нового формата (`failed to seek for tensor v.post_ln.weight`).

**Решение:** весь код через llama-cpp-python напрямую; `Qwen25VLChatHandler` + mtmd.dll. Ollama-код удалён навсегда.
