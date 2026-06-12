# scripts

Вспомогательные скрипты. Запускать из корня проекта (`pdf_to_md/`), не из этой папки — все пути относительные.

---

## detect_hallucinations.py

Сканирует `output/chast-*.md` и флагирует страницы с галлюцинациями.

**Два прохода:**
1. Быстрые эвристики (fake imgur URLs, английский текст, Unicode-математика без LaTeX, repetition)
2. Gemma для страниц, прошедших эвристики

**Результат:** `output/hallucination_report.txt` с постраничным списком и причиной.

```powershell
.venv\Scripts\python.exe src\scripts\detect_hallucinations.py
```

---

## clear_bad_cache.py

Применяет те же эвристики **напрямую к файлам кэша** (`.pdf_to_md_cache/.../page_NNNN.md`) и удаляет плохие. Pipeline автоматически перегенерирует их при следующем запуске.

Используется после `detect_hallucinations.py` для точечного исправления без полного перезапуска.

```powershell
.venv\Scripts\python.exe src\scripts\clear_bad_cache.py
```

---

## test_prompts.py

A/B тест нескольких вариантов vision-промпта на фиксированном наборе страниц.

Рендерит страницы из PDF напрямую (минуя кэш), прогоняет каждый промпт, сохраняет `.md` файлы в `output/prompt_test/` для ручного сравнения. Уже обработанные страницы/промпты пропускает (кэш по имени файла).

```powershell
.venv\Scripts\python.exe src\scripts\test_prompts.py
```

---

## ab_test_ocr.py

Сравнивает два варианта промпта на 30 равномерно распределённых страницах chast-2:
- **no_ocr**: промпт без OCR-подсказки
- **with_ocr**: тот же промпт + OCR-текст страницы в конце

Считает метрики (LaTeX usage, Unicode math, флаги галлюцинаций), выводит сводную таблицу.
Результаты: `output/ab_test_ocr/` + `results.json`.

```powershell
.venv\Scripts\python.exe src\scripts\ab_test_ocr.py
```

---

## merge_gguf.py

Склеивает split GGUF шарды (00001-of-00002, 00002-of-00002) в один файл.
Нужен когда модель скачана по частям, а llama-cpp-python требует единый файл.

Пути к шардам и выходному файлу прописаны внизу скрипта — отредактировать перед запуском.

```powershell
.venv\Scripts\python.exe src\scripts\merge_gguf.py
```
