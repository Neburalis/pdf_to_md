# pdf-to-md — AI часть

Всё, что связано с моделями, промптами, параметрами inference и результатами тестов.

---

## Текущие модели

| Роль | Файл | Размер | Источник |
|------|------|--------|---------|
| Vision | `models\Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | 4.4 ГБ | `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF` |
| Vision projector | `models\mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf` | 1.3 ГБ | то же |
| Cleaning | `models\gemma-3-12b-it-heretic-Q5_K_M.gguf` | 7.9 ГБ | HuggingFace |

**GPU:** RTX 4070 Ti, 12 ГБ VRAM.

**VRAM при inference:**
- Vision: ~6.2 ГБ (модель 4.4 + mmproj 1.3 + fp16 KV-кэш ~0.47 при n_ctx=4096)
- Cleaning: ~9.3 ГБ (модель 7.9 + Q4 KV-кэш ~1.4 при n_ctx=16384)
- Загружаются **последовательно** — вместе не помещаются. `unload_vision_model()` между фазами.

### Почему Qwen2.5-VL, а не Qwen2-VL (старая)

Qwen2-VL-7B давала ~22% галлюцинаций (190 из 861 страниц):
- heading-depth repetition: один блок текста дублировался на `#`, `##`, `###`, `####`
- fake imgur URLs вместо рисунков: `![](https://i.imgur.com/3Q5Q5Q5.png)`
- content loops: параграф повторялся 7–20 раз подряд
- ~14 страниц английского текста вместо русской математики

Qwen2.5-VL устранила повторения и fake URLs, улучшила следование инструкциям промпта.
Старые файлы (`models\Qwen2-VL-7B-Instruct-Q4_K_M.gguf`, `models\mmproj-Qwen2-VL-7B-Instruct-f32.gguf`) можно удалить.

### Почему не другие модели

- **Ollama**: v0.24.0 использует старый llama.cpp, не умеет загружать mmproj нового формата (`failed to seek for tensor v.post_ln.weight`). Backend удалён навсегда.
- **Qwen2.5-VL-32B**: не влезает в 12 ГБ даже при агрессивной квантизации.
- **InternVL / Pixtral**: хуже поддержаны в llama.cpp, менее протестированы на математических документах.

---

## Параметры inference (Vision — Qwen2.5-VL)

```python
# vision.py — _load_vision_model()
n_ctx=4096        # было 8192; KV cache: 940 МБ → 470 МБ; нужно для DPI=200 (см. EXPERIMENTS.md)
n_gpu_layers=-1   # полный GPU offload
flash_attn=True   # обязательно — без этого warmup требует ~4872 МБ

# page_to_md() — create_chat_completion()
max_tokens=2048
temperature=0.1       # не 0.0: greedy decoding провоцирует repetition loops
repeat_penalty=1.15   # подавляет повторяющиеся токены
```

**Почему НЕТ type_k/type_v у vision:**
Не нужно — fp16 KV при n_ctx=4096 (~0.47 ГБ) влезает. Qwen2.5-VL-7B (4.4 ГБ) + mmproj (1.3 ГБ) + KV (~0.47 ГБ) ≈ 6.2 ГБ из 12 ГБ.

**Почему n_ctx=4096, а не 8192:**
При n_ctx=8192 KV cache занимает ~940 МБ, что после загрузки модели и CUDA graph warmup оставляло max_contiguous=374 МБ — vision encoder при DPI=200 не помещался (~800 МБ). При n_ctx=4096 KV cache ~470 МБ, max_contiguous увеличивается достаточно для DPI=200. Подробнее — в EXPERIMENTS.md.

**Почему temperature=0.1, а не 0.0:**
При t=0.0 (greedy) модель застревала в петлях повторений (heading-depth repetition, content loops). t=0.1 вносит минимальную вариативность — достаточно для выхода из петли, незаметно для качества.

**Почему repeat_penalty=1.15:**
Дополнительная защита от repetition loops на уровне токенов. Убирает heading-depth repetition без влияния на легитимные повторяющиеся конструкции (нумерованные списки, схожие по структуре формулы).

## Параметры inference (Cleaning — gemma-3-12b)

```python
# cleaner.py — _load_text_model()
n_ctx=16384       # нужно для чанков ~10k токенов (10 страниц × ~1100 tok)
n_gpu_layers=-1   # полный GPU offload
type_k=2          # Q4_0 KV-кэш — 4× меньше fp16 (обязательно!)
type_v=2          # Q4_0
flash_attn=True

# clean_chunk() — create_chat_completion()
max_tokens=8192
temperature=0.0
```

**Почему type_k/type_v ОБЯЗАТЕЛЬНЫ для cleaner:**
Без Q4 KV-кэша gemma не влезает: 7.9 ГБ модель + fp16 KV при n_ctx=16384 (~5.8 ГБ) = 13.7 ГБ > 12 ГБ. С Q4 KV (~1.4 ГБ): 7.9 + 1.4 = 9.3 ГБ — влезает. Это фундаментальное ограничение, не оптимизация.

---

## Текущий промпт (Vision)

```
Convert this PDF page image to Markdown.

Rules:
- Use # ## ### for headings based on visual size and weight
- Preserve lists (- or 1.) exactly as structured in the image
- Preserve tables using Markdown table syntax (| col | col |)
- Preserve bold (**text**) and italic (*text*) formatting
- For mathematical formulas use LaTeX with $ delimiters: inline $formula$, display block $$formula$$ — do NOT use \( \) or \[ \]
- For figures and diagrams write [Рисунок N] using the actual figure number from the text — do NOT generate image URLs
- Do NOT wrap the output in code fences
- Do NOT repeat the same content block multiple times or at different heading levels
- If the page contains only a page number or is blank, output an empty string
- Output ONLY the Markdown content, no explanations
```

### История изменений промпта

| Версия | Ключевое изменение | Результат |
|--------|-------------------|-----------|
| v1 | Исходный, с OCR text hint | 22% галлюцинаций, English текст, heading repetition |
| v2 (russian) | Промпт на русском языке | Нестабилен: code fences, потеря LaTeX, контент с соседней страницы |
| v3 (faithful) | "Transcribe only what you see" | Паритет с v1, без выигрыша |
| v4 (minimal) | Очень короткий промпт | Пустой вывод на всех страницах — модель молчит без явных инструкций |
| **v5** | Запрет imgur URLs, запрет code fences, LaTeX правило | Убраны fake URLs, лучше LaTeX |
| **v6** | + запрет repetition, temperature=0.1, repeat_penalty=1.15 | Убраны heading-depth repetition loops |
| **v10 (текущий)** | + "ONLY (NEVER)" LaTeX, матрицы, нумерованные уравнения, русские списки | Устранён `\(\)` на координатных страницах; матрицы/определители; `\quad (N)` |

### Тестирование v6 vs v7/v8/v9/v10 на Беклемишеве (5 страниц, 2026-06-05)

Книга: Беклемишев Д.В. "Курс аналитической геометрии и линейной алгебры", 304 страницы.

**Метрики суммарно (5 страниц):**

| Промпт | $-sum | `\(\)`-sum | unicode | code fences | Победитель |
|--------|-------|-----------|---------|------------|-----------|
| v6_baseline | 246 | **84** | 28 | **0** | — |
| v7_matrix_aware | 260 | 0 | 25 | 2 | — |
| v8_examples | **372** | 0 | **0** | **5/5** | — |
| v9_math_priority | 294 | 0 | 25 | 5/5 | — |
| **v10_hybrid** | ~280 | **0** | ~5 | **0** | ✅ |

**Ключевые находки:**

- `\(\)` у v6 сконцентрированы на p020 (полярные/цилиндрические координаты) — 84 случая на одной странице; p050–p150 — 0.
- v7 на p020 даёт нулевой LaTeX (только Unicode φ, ≤, π) — хуже v6.
- v8 лучшая математика, но code fence на 5/5 — неприемлемо: ломает merge-шаг pipeline.
- Причина code fences у v7/v8/v9: контекстный абзац ("This is a textbook...") или CAPS-заголовки секций переключают модель в режим "форматированного вывода".
- Правило "skip running header" ненадёжно: работает непредсказуемо, иногда ухудшает (v10 включал заголовок там, где v6 его пропускал).
- v10 (v6-структура + усиленный LaTeX + матрицы/определители/нумерованные уравнения): нет `\(\)`, нет code fences, появляется `\quad(N)` в нумерованных уравнениях.

**Паттерн `\( \)` vs `$ $`:** проблема возникает на страницах с парами полярных координат `(r, φ)` — модель оборачивает их в `\(\)`. Формулировка "ONLY (NEVER \( \) or \[ \])" в v10 устранила это.

**Отвергнутые подходы:**
- Контекстный абзац о типе документа → провоцирует code fences
- ✓/✗ примеры как отдельный раздел (v8) → самая чистая математика, но code fences 100%
- "skip running chapter header" → ненадёжно, не внедрять

---

## Текущий промпт (Cleaning / gemma-3-12b)

```
Clean up the following Markdown section converted from a PDF. Apply these fixes:

1. Remove page number artifacts (e.g. a lone "42" on its own line, "Page 42 of 100")
2. Remove running headers or footers that repeat across pages (book title, chapter name, etc.)
3. Fix words broken by hyphens at line boundaries only when clearly a split word, not a compound word
4. Merge paragraph sentences split across page breaks into continuous text
5. DO NOT remove, summarize, shorten, or rewrite any meaningful content
6. DO NOT change headings, lists, tables, or code blocks
7. Output ONLY the cleaned Markdown, no explanations
8. Do NOT wrap the output in ```markdown or any other code fences

Section:
{text}
```

**⚠️ Cleaning временно отключён (`--no-clean`) и не применяется ни к одной части.**

Причина: правило 4 промпта ("Merge paragraph sentences split across page breaks") интерпретируется как удаление всех `---` разделителей внутри чанков. После cleaning 278 страниц сжались в 14 блоков — вся постраничная структура была уничтожена.

Что нужно исправить перед включением cleaning:
- Убрать из промпта правило "merge across page breaks"
- Или изменить логику chunking: передавать страницы по одной, а не чанками через `PAGE_SEP`

---

## Результаты тестов промптов (Qwen2-VL, chast-2)

Тест на 5 страницах (p010, p025, p050, p080, p120):

| Страница | v1 | v5 | v6 |
|----------|----|----|-----|
| p010 | ✅ LaTeX | ✅ LaTeX | ✅ LaTeX |
| p025 | ✅ | ✅ | ✅ |
| p050 | ✅ | ✅ | ✅ |
| p080 (рисунок) | ❌ imgur URL | ✅ [Рисунок], 2× больше контента | ✅ |
| p120 | ✅ | ✅ | ✅ |

**Критический тест p080:** v5 дала 1888 символов реального контента вместо 903 символов с fake imgur у v1.

---

## Детекция галлюцинаций

Скрипты: `detect_hallucinations.py`, `clear_bad_cache.py`

### Эвристики (порядок применения)

1. **Fake imgur URLs** — `i.imgur.com` в тексте
2. **Template placeholders** — `[Section X Title]`, `[img-X]`, `[Figure X]`
3. **Утечка промпта** — "Convert this PDF page", "Output ONLY the Markdown"
4. **Английский контент** — Cyrillic ratio < 8% при объёме > 300 символов
5. **Unicode-математика вне LaTeX** — ≥8 символов (`∈∀∂→∞αβ…`) при < 4 знаках `$` (ключевая эвристика для формульных страниц без LaTeX)
6. **Высокая повторяемость** — топ-1 строка > 62% от всех строк

**Порог повторяемости 62%, а не ниже:** при 45% было много ложных тревог на легитимных нумерованных списках (27-30 свойств действительных чисел) и формульных страницах с похожей структурой строк.

### Статистика по прогонам

| Дата | Модель | Промпт | Галлюцинаций |
|------|--------|--------|-------------|
| 2026-05-30 | Qwen2-VL | v1 + OCR hint | ~190/861 (22%) — до детектора |
| 2026-06-03 | Qwen2-VL | v5, без OCR | 190/861 (22%) выявлено детектором |
| 2026-06-04 | Qwen2-VL | v6, без OCR | 83/861 (9.6%) после перегенерации |
| 2026-06-04 | Qwen2.5-VL | v6, без OCR | A/B тест в процессе |

---

## A/B тест: OCR hint с Qwen2.5-VL

**Гипотеза:** Qwen2.5-VL умнее Qwen2-VL и корректно приоритизирует изображение над OCR-текстом, не галлюцинируя на его основе.

**Тест:** `src/scripts/ab_test_ocr.py` — 30 страниц из chast-2, равномерно по всему документу.
- Вариант A (no_ocr): промпт v6 без OCR hint
- Вариант B (with_ocr): промпт v6 + OCR текст в конце (после правил, с пометкой "reference only")

**Расположение OCR hint в промпте B:** в конце, после всех правил — в отличие от v1, где hint шёл перед правилами.

**Результаты (ручной анализ + метрики):**

| Метрика | no_ocr | with_ocr |
|---------|--------|----------|
| Побед | **5** | 2 |
| Ничьих | 23 | 23 |
| Flagged | **4/30** | 7/30 |
| Avg LaTeX ($ count) | **25.0** | 15.0 |
| Avg Unicode math | **8.6** | 13.0 |

**Критические находки при просмотре файлов:**

- `p0046` (no_ocr wins): with_ocr скопировал OCR-текст почти дословно — 707 символов vs 2143 у no_ocr с полноценным LaTeX. Модель «доверилась» OCR и перестала читать изображение.
- `p0128`, `p0155`, `p0201`: схожий паттерн — with_ocr даёт меньше контента + Unicode math вместо LaTeX.
- `p0210` (with_ocr wins): no_ocr вставил HTML-теги `<sub>`/`<sup>` и сырой Unicode; with_ocr — нормальный LaTeX. Редкий позитивный случай.

**Вывод: OCR hint отклонён.** Катастрофические падения на 3–5 страницах перевешивают редкие улучшения на 2. Гипотеза не подтвердилась: Qwen2.5-VL при наличии OCR часто предпочитает его изображению.

**Сопутствующая находка:** Qwen2.5-VL иногда пишет `\( \)` и `\[ \]` вместо `$ $` несмотря на явный запрет в промпте (встречается в обоих вариантах, не коррелирует с OCR hint). Требует отдельного решения.

---

## Changelog

### 2026-06-11 — n_ctx=4096: DPI=200 теперь работает без OOM

**Проблема:** при n_ctx=8192 KV cache занимал ~940 МБ → после загрузки модели и CUDA graph warmup max_contiguous падал до 374 МБ → vision encoder при DPI=200 (требует ~800 МБ) не помещался. Pipeline работал только при DPI=90 (~330 МБ compute buffer).

**Решение:** `n_ctx=4096` в `_load_vision_model()`. KV cache: 940 МБ → 470 МБ → max_contiguous увеличивается достаточно.

**Результат:** страницы 17-19 термодинамики обработаны при DPI=200 без OOM за 45.1s (3 страницы). Качество формул не хуже, вывод чуть полнее (6878 vs 6295 chars при DPI=90).

**Отвергнутые альтернативы:** giant malloc pre-warm (Эксперимент 2), image-first порядок (Эксперимент 1) — не потребовались, n_ctx=4096 решил проблему самостоятельно.

---

### 2026-06-05 — Промпт v10: матрицы, нумерованные уравнения, устранение \( \)

**Проблема:** v6 давал `\(\)` вместо `$` на страницах с полярными/цилиндрическими координатами (84 случая на одной странице). Новая книга (Беклемишев) требует поддержки матриц, определителей, нумерованных уравнений.

**Решение:** v10 — хирургический апгрейд v6 при сохранении flat-list структуры: формулировка "LaTeX $ delimiters ONLY (NEVER \\( \\) or \\[ \\])"; добавлены `\begin{pmatrix}`, `\begin{vmatrix}`, `$$formula \quad (N)$$`; русские списки `а), б), в)` в правиле сохранения списков.

**Результат:** p020: 84 `\(\)` → 0; нумерованные уравнения `\quad(12)` работают; code fences 0/5.

**Отвергнутые альтернативы:** v7 (CAPS-секции + контекст документа) — code fences на 2/5, нулевой LaTeX на проблемной странице; v8 (✓/✗ примеры) — лучшая математика (372 $ vs 246), но code fences 5/5; "skip running chapter header" — ненадёжно.

---

### 2026-06-04 — A/B тест OCR hint: отрицательный результат

**Гипотеза:** Qwen2.5-VL достаточно умна, чтобы использовать OCR-подсказку как reference, не копируя её.

**Результат:** отклонено. no_ocr выиграл 5 страниц vs 2 у with_ocr; with_ocr flagged 7/30 vs 4/30; LaTeX usage упал с 25.0 до 15.0 avg dollar signs.

**Паттерн провала:** на страницах с dense math (p0046, p0128, p0155, p0201) модель «доверилась» OCR-тексту и скопировала его почти дословно вместо конвертации изображения. Объём output: 707 символов vs 2143 у no_ocr.

**Решение:** OCR hint не добавляется. Промпт остаётся v6 без изменений.

**Сопутствующая находка:** `\( \)` / `\[ \]` вместо `$ $` встречается в обоих вариантах — это паттерн самой модели, не связан с OCR. Требует решения (усиление правила в промпте или постпроцессинг).

---

### 2026-06-04 — Qwen2-VL → Qwen2.5-VL-7B

**Проблема:** Qwen2-VL давала ~22% галлюцинаций: heading-depth repetition, fake imgur URLs, content loops, ~14 страниц английского текста.

**Решение:** `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` из `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF`.

**Результат:** повторения и fake URLs исчезли, figure references стали корректными.

**Отвергнутые альтернативы:** Qwen2.5-VL-32B не влезает в 12 ГБ; InternVL/Pixtral хуже поддержаны в llama.cpp.

---

### 2026-06-03 — Q4 KV-кэш для cleaner (обязательный фикс)

**Проблема:** gemma-3-12b не помещалась в VRAM: 7.9 ГБ + fp16 KV при n_ctx=16384 (~5.8 ГБ) = 13.7 ГБ > 12 ГБ.

**Решение:** `type_k=2, type_v=2` (Q4_0 KV) + `flash_attn=True` + `n_ctx=16384`.

**Результат:** 9.3 ГБ суммарно, ~25 tok/sec вместо 4 tok/sec при частичном offload.

**Отвергнутые альтернативы:** `n_gpu_layers=24` — работало, но 4 tok/sec (12 часов на одну часть).

---

### 2026-06-03 — Промпт v5→v6: запрет repetition + temperature

**Проблема (v5):** heading-depth repetition — тот же контент на `#`, `##`, `###`, `####`. Greedy decoding (t=0.0) застревал в петле.

**Решение:** добавлено правило "Do NOT repeat the same content block", `temperature=0.1`, `repeat_penalty=1.15`.

**Проблема (v1→v5):** fake imgur URLs; v4 (minimal) давал пустой вывод; v2 (русский промпт) нестабилен — code fences, потеря LaTeX.

**Решение v5:** запрет imgur URLs и code fences, явное правило LaTeX.

---

### 2026-06-01 — Удаление OCR text hint

**Проблема:** `_TEXT_HINT_SECTION` передавала OCR-текст как «подсказку». Симптом: английский текст о нутрициологии вместо математики (~86% страниц chast-1 галлюцинировало).

**Решение:** убран text_hint из промпта и сигнатуры `page_to_md()`.

**Статус:** проводится A/B тест с Qwen2.5-VL — возможно, более умная модель использует hint корректно.

---

### 2026-05-31 — Cleaning уничтожает постраничную структуру

**Проблема:** правило 4 промпта ("Merge paragraph sentences split across page breaks") интерпретировалось как удаление всех `---` внутри чанков. После cleaning 278 страниц → 14 блоков.

**Решение:** `--no-clean` до исправления промпта.

**Что нужно исправить:** убрать правило "merge across page breaks" или передавать страницы по одной вместо чанков.
