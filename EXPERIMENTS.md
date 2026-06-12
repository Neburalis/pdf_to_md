# Эксперименты по устранению фрагментации VRAM

## Контекст

**Железо:** RTX 4070 Ti, 12 282 МБ VRAM  
**Модель:** Qwen2.5-VL-7B (Q4_K_M LLM + f16 mmproj)  
**Текущее состояние:** ✅ DPI=200 работает после Эксперимента 3 (n_ctx=4096)

### Почему DPI=90

При DPI=200 vision encoder падает с CUDA OOM. Причина — фрагментация VRAM:

| Момент | Max contiguous |
|--------|---------------|
| Старт (только DLL) | ~5120 МБ |
| После загрузки модели | ~500–700 МБ |
| После CUDA graph warmup | ~374 МБ |

Vision encoder при DPI=200 нужно **~800 МБ непрерывно** → не помещается в 374 МБ.  
При DPI=90 нужно **~330 МБ** → едва влезает в 374 МБ.

Фрагментация возникает из-за того, что GGML делает много отдельных `cudaMalloc` разного размера при загрузке модели (веса, KV cache, compute буфер, mmproj), а CUDA graph warmup при первом decode ещё добавляет постоянные буферы.

### Что уже сделано (не трогать)

В `src/pdf_to_md/vision.py`:
- `warmup=False` в `llama_chat_format.py` — отключён warmup mmproj (пытался аллоцировать 704 МБ)
- `flash_attn=True` — без этого warmup требовал ~4872 МБ
- `n_batch=64` — дефолтный батч вызывал OOM на numpy (297 МБ)
- `suppress_stdout_stderr(disable=False)` для inference
- `_strip_repeated_paragraphs()` — дедупликация повторяющихся параграфов

Нельзя: `GGML_CUDA_NO_GRAPHS=1` перед инициализацией DLL — снижает max_contiguous до 4032 МБ, модель не загружается.

---

## Эксперимент 1: Поменять порядок [image, text]

**Гипотеза:** сейчас в промпте текст идёт ДО изображения → CUDA graph warmup происходит при декодинге текста → max_contiguous падает до 374 МБ → vision encoder при 200 DPI не помещается.

Если поставить изображение первым, vision encoder аллоцирует compute buffer до CUDA graph warmup (в менее фрагментированном состоянии ~500–700 МБ).

**Изменение в `src/pdf_to_md/vision.py`, функция `page_to_md`:**

```python
# СЕЙЧАС (text первым):
messages=[{
    "role": "user",
    "content": [
        {"type": "text", "text": _PAGE_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
    ],
}]

# НУЖНО (image первым):
messages=[{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        {"type": "text", "text": _PAGE_PROMPT},
    ],
}]
```

**Тест:**
```powershell
# Сначала тест с DPI=200 (2048 мб на странице = возможный OOM, смотреть на ошибку)
.venv\Scripts\python.exe -m pdf_to_md `
  "pdf\lektsii_po_termodinamike_i_molekuljarnoj_fizike_uc_260327_003432.pdf" `
  "output\test_exp1.md" `
  --vision-model "models\Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" `
  --vision-mmproj "models\mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf" `
  --no-clean --pages 17-19 --dpi 200 --force
```

**Успех:** нет CUDA OOM, страницы 17-19 обработаны.  
**Провал:** всё тот же `cudaMalloc failed` или `CUDA error: out of memory`.

**Риск:** Qwen2.5-VL мог обучаться на формате "text, image" → качество может ухудшиться при "image, text". Нужно сравнить output с DPI=90 baseline.

**Если провал:** не возвращать обратно вручную — просто перейти к следующему эксперименту.

---

## Эксперимент 2: Giant malloc перед загрузкой

**Гипотеза:** CUDA's caching allocator кэширует освобождённые блоки. Если до загрузки модели аллоцировать и сразу освободить ~10 ГБ, аллокатор может отдавать суббуферы из одного большого кэшированного блока → меньше фрагментации.

**Изменение в `src/pdf_to_md/vision.py`, функция `_load_vision_model`, ПЕРЕД строкой `from llama_cpp import Llama`:**

```python
# Giant malloc trick: pre-warm CUDA allocator before model load
# to reduce fragmentation of the remaining free VRAM.
# CUDA caching allocator may carve sub-blocks from one cached chunk.
import ctypes as _ct
try:
    _cudart = _ct.CDLL("cudart64_12.dll")
    _ptr = _ct.c_void_p()
    _ret = _cudart.cudaMalloc(_ct.byref(_ptr), 10 * 1024 * 1024 * 1024)
    if _ret == 0:
        _cudart.cudaFree(_ptr)
        log.info("Giant malloc pre-warm: OK")
    else:
        log.info(f"Giant malloc pre-warm: failed (ret={_ret}), continuing")
except Exception:
    pass
```

**Тест:**
```powershell
.venv\Scripts\python.exe -m pdf_to_md `
  "pdf\lektsii_po_termodinamike_i_molekuljarnoj_fizike_uc_260327_003432.pdf" `
  "output\test_exp2.md" `
  --vision-model "models\Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" `
  --vision-mmproj "models\mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf" `
  --no-clean --pages 17-19 --dpi 200 --force
```

**Успех:** нет CUDA OOM.  
**Провал:** тот же OOM → гипотеза неверна, CUDA аллокатор не использует кэш так.

---

## Эксперимент 3: Уменьшить n_ctx

**Гипотеза:** KV cache при n_ctx=8192 занимает ~940 МБ. При n_ctx=2048 — ~235 МБ. Это освобождает ~700 МБ, меняет паттерн аллокаций и потенциально увеличивает max_contiguous после загрузки.

При DPI=90 максимальный контекст: 486 visual + 312 prompt + 2048 output = ~2846 токенов → n_ctx=2048 слишком мало.  
При DPI=90 с n_ctx=4096: 486 + 312 + 2048 = 2846 < 4096 — достаточно.

**Изменение в `src/pdf_to_md/vision.py`, функция `_load_vision_model`:**

```python
# СЕЙЧАС:
_vision_llama = Llama(
    model_path=model_path,
    chat_handler=chat_handler,
    n_ctx=8192,
    ...
)

# НУЖНО:
_vision_llama = Llama(
    model_path=model_path,
    chat_handler=chat_handler,
    n_ctx=4096,  # было 8192; KV cache: 940 МБ → 470 МБ
    ...
)
```

**Тест — два шага:**

Шаг А: проверить что DPI=90 не сломалось:
```powershell
.venv\Scripts\python.exe -m pdf_to_md `
  "pdf\lektsii_po_termodinamike_i_molekuljarnoj_fizike_uc_260327_003432.pdf" `
  "output\test_exp3a.md" `
  --vision-model "models\Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" `
  --vision-mmproj "models\mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf" `
  --no-clean --pages 17-19 --dpi 90 --force
```

Шаг Б: попробовать DPI=200:
```powershell
.venv\Scripts\python.exe -m pdf_to_md `
  "pdf\lektsii_po_termodinamike_i_molekuljarnoj_fizike_uc_260327_003432.pdf" `
  "output\test_exp3b.md" `
  --vision-model "models\Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" `
  --vision-mmproj "models\mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf" `
  --no-clean --pages 17-19 --dpi 200 --force
```

**Успех Б:** нет OOM → можно переходить на 200 DPI и n_ctx=4096.  
**Провал Б:** OOM → попробовать комбинацию с экспериментом 1.

**✅ РЕЗУЛЬТАТ (2026-06-11):** Шаг А и Шаг Б — успех. DPI=200, 3 страницы, 45.1s, no OOM. Эксперименты 2, 1, 4 не потребовались.

---

## Эксперимент 4: Промежуточный DPI (130–150)

**Гипотеза:** max_contiguous ПОСЛЕ загрузки модели но ДО CUDA graph warmup — предположительно ~500–700 МБ. При DPI=130 compute buffer ~450 МБ, при 150 — ~520 МБ. Это может помещаться в pre-warmup пространство, если vision encoder аллоцируется первым (комбинация с Экспериментом 1).

**Это запасной вариант** — лучше качество чем 90 DPI, но хуже 200 DPI.

**Тест (только если Эксперимент 1 хотя бы частично работает):**
```powershell
# 130 DPI
.venv\Scripts\python.exe -m pdf_to_md `
  "pdf\lektsii_po_termodinamike_i_molekuljarnoj_fizike_uc_260327_003432.pdf" `
  "output\test_exp4_130.md" `
  --vision-model "models\Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" `
  --vision-mmproj "models\mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf" `
  --no-clean --pages 17-19 --dpi 130 --force

# 150 DPI
.venv\Scripts\python.exe -m pdf_to_md `
  "pdf\lektsii_po_termodinamike_i_molekuljarnoj_fizike_uc_260327_003432.pdf" `
  "output\test_exp4_150.md" `
  --vision-model "models\Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" `
  --vision-mmproj "models\mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf" `
  --no-clean --pages 17-19 --dpi 150 --force
```

---

## Порядок запуска

Рекомендуемый порядок (от простого к сложному):

1. **Эксперимент 3** (n_ctx=4096) — одна строка, не ломает DPI=90
2. **Эксперимент 2** (giant malloc) — несколько строк, независимо от 3
3. **Эксперимент 1** (image first) — меняет промпт-формат, может влиять на качество
4. **Эксперимент 4** (промежуточный DPI) — запасной, после 1

Каждый тест на **pages 17-19** — там формулы, хорошо видно качество, страницы уже известны.

## Baseline для сравнения качества

Страницы 17-19 при DPI=90 уже в кэше: `.pdf_to_md_cache/585d3a443c0ad42d/page_0016.md` … `page_0018.md`.

Ключевые формулы которые должны быть корректны:

- `$$P = \frac{1}{3} n m \overline{v^2}$$` (стр. 17)  
- `$$\frac{m_1 \overline{v^2}_1}{2} = \frac{m_2 \overline{v^2}_2}{2}$$` (стр. 20)

При DPI=200 формулы должны быть не хуже, а скорее лучше.
