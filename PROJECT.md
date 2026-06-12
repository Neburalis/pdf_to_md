# pdf-to-md — описание проекта

## Что делает

Конвертирует PDF в Markdown через локальные LLM-модели. Pipeline из 4 фаз:

| # | Фаза | Что происходит |
|---|------|---------------|
| 0 | **Render** | PDF → PNG (PyMuPDF, 200 DPI по умолчанию) |
| 1 | **Vision** | PNG → Markdown на страницу (Qwen2.5-VL-7B) |
| 2 | **Merge** | Страницы склеиваются через `---` |
| 3 | **Clean** | Текстовая модель чистит артефакты, переносы, колонтитулы (gemma-3-12b) |

Фаза 3 опциональна (`--no-clean`). Результаты vision кэшируются — повторный запуск пропускает обработанные страницы.

Подробно о моделях, промптах и результатах тестов — в **[AI.md](AI.md)**.

---

## PDF-файлы для конвертации

Лежат в `pdf/`:
- `lektsii-po-matematicheskomu-analizu-v-3-ch-chast-1-vvedenie-v-matematicheskij-analiz.pdf` — 278 страниц
- `lektsii-po-matematicheskomu-analizu-v-3-ch-chast-2-mnogomernyj-analiz-integraly-i-rjady.pdf` — 273 страницы
- `lektsii-po-matematicheskomu-analizu-v-3-ch-chast-3-kratnye-integraly-garmonicheskij-analiz.pdf` — 310 страниц

---

## Команды запуска

```powershell
# Тест (6 страниц)
.venv\Scripts\python.exe -m pdf_to_md convert `
  pdf\lektsii-po-matematicheskomu-analizu-v-3-ch-chast-1-vvedenie-v-matematicheskij-analiz.pdf `
  output\chast-1.md `
  --vision-model "models\Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" `
  --vision-mmproj "models\mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf" `
  --text-model "models\gemma-3-12b-it-heretic-Q5_K_M.gguf" `
  --pages 1-6

# Полная конвертация одной части (--no-clean, кэш пропускает готовые страницы)
.venv\Scripts\python.exe -m pdf_to_md convert `
  pdf\lektsii-po-matematicheskomu-analizu-v-3-ch-chast-1-vvedenie-v-matematicheskij-analiz.pdf `
  output\chast-1.md `
  --vision-model "models\Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" `
  --vision-mmproj "models\mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf" `
  --text-model "models\gemma-3-12b-it-heretic-Q5_K_M.gguf" `
  --no-clean --force
```

---

## Известные нюансы

- **~60 сек/страницу** на vision inference (RTX 4070 Ti, Qwen2.5-VL Q4_K_M). 278 страниц ≈ 5 часов.
- **Кэш** в `.pdf_to_md_cache/` — при повторном запуске пропускает уже обработанные страницы.
- **UV_NATIVE_TLS=1** — нужен для всех `uv`-команд (антивирус делает SSL-инспекцию).
- **HF_HUB_DISABLE_SSL_VERIFY=1** — нужен для скачивания с HuggingFace по той же причине.
- **suppress_stdout_stderr** — весь шум llama.cpp подавлен, только логи pipeline видны в консоли.
- **Cleaning временно отключён** (`--no-clean` во всех текущих командах) — промпт gemma удаляет `---` разделители страниц, превращая 278 страниц в 14 блоков. Подробнее и план исправления — в AI.md.

---

## Инструменты контроля качества

| Команда / скрипт | Назначение |
|------------------|-----------|
| `pdf-to-md detect [FILES...]` | Сканирует Markdown, флагирует плохие страницы (эвристики + опциональная gemma) |
| `pdf-to-md clear-cache` | Удаляет плохие страницы из кэша — pipeline перегенерирует их при следующем запуске |
| `src/scripts/test_prompts.py` | A/B тест промптов на фиксированном наборе страниц |
| `src/scripts/ab_test_ocr.py` | A/B тест OCR hint (with vs without) на 30 страницах |

---

## Что реализовано

1. ✅ llama-cpp-python v0.3.23 с CUDA (cu124 wheel)
2. ✅ CUDA runtime DLL — предзагрузка через ctypes (Windows DLL search path)
3. ✅ Qwen25VLChatHandler + mtmd.dll (новый multimodal API)
4. ✅ Q4_0 KV-кэш + flash_attn для экономии VRAM
5. ✅ repeat_penalty=1.15, temperature=0.1 против repetition loops
6. ✅ Автовыгрузка vision-модели перед cleaning (освобождение VRAM)
7. ✅ Логирование: `logs/<name>_<ts>.log` (файл UTF-8 + консоль через tqdm.write)
8. ✅ Статистика: `logs/<name>_<ts>_stats.csv` (время, токены, оценка стоимости GPT-4o)
9. ✅ GPU-мониторинг: `logs/<name>_<ts>_gpu.csv` (temp/util/VRAM каждые 5 сек)
10. ✅ tqdm: двойной прогресс-бар (Overall фазы + текущая фаза)
11. ✅ Детекция галлюцинаций: эвристики + gemma верификация
12. ✅ Ollama backend удалён, только llama-cpp-python
