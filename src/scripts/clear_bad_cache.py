"""
Scan cache files directly with hallucination heuristics.
Deletes bad page_NNNN.md files so the pipeline regenerates them on next run.
"""

import json, re, sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_MATH_UNICODE_RE = re.compile(
    r"[∈∉∀∃∂→←↔⇒⇔≤≥≠≈∞∑∏∫∬∭√∪∩⊂⊃⊆⊇∧∨⊕⊗±∓≡≃≅∝∠⊥∥⌊⌋⌈⌉⩽⩾"
    r"αβγδεζηθικλμνξπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ"
    r"¹²³⁰⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉]"
)


def _cyrillic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum(1 for c in letters if "Ѐ" <= c <= "ӿ") / len(letters)


def _repetition_score(text: str) -> float:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 6:
        return 0.0
    return Counter(lines).most_common(1)[0][1] / len(lines)


def _strip_latex(text: str) -> str:
    text = re.sub(r"\$\$[\s\S]*?\$\$", "", text)
    text = re.sub(r"\\\[[\s\S]*?\\\]", "", text)
    text = re.sub(r"\$[^\$\n]{1,300}\$", "", text)
    text = re.sub(r"\\\([^\)]*\\\)", "", text)
    return text


def is_bad(text: str) -> tuple[bool, str]:
    s = text.strip()
    if len(s) < 50:
        return False, ""

    if "i.imgur.com" in s:
        return True, "Fake imgur image URLs"

    if re.search(r"\[Section \d+[^\]]*\]|\[img-\d+\]|\[Figure \d+\]", s):
        return True, "Template placeholders"

    if "Convert this PDF page" in s or "Output ONLY the Markdown" in s:
        return True, "OCR prompt text in output"

    if len(s) > 300 and _cyrillic_ratio(s) < 0.08:
        return True, f"English content (Cyrillic {_cyrillic_ratio(s):.1%})"

    if len(s) > 150 and _cyrillic_ratio(s) >= 0.10:
        no_latex = _strip_latex(s)
        n_math = len(_MATH_UNICODE_RE.findall(no_latex))
        n_dollar = s.count("$")
        if n_math >= 8 and n_dollar < 4:
            return True, f"Unicode math outside LaTeX ({n_math} symbols, {n_dollar} $)"

    rep = _repetition_score(s)
    if rep > 0.62:
        return True, f"Highly repetitive ({rep:.0%} identical lines)"

    return False, ""


def main() -> None:
    cache_root = Path(".pdf_to_md_cache")
    if not cache_root.exists():
        print("Cache directory not found.")
        return

    total_checked = total_deleted = 0

    for hash_dir in sorted(cache_root.iterdir()):
        if not hash_dir.is_dir():
            continue

        meta_path = hash_dir / "meta.json"
        pdf_name = "?"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            pdf_name = Path(meta.get("pdf", "?")).name

        page_files = sorted(hash_dir.glob("page_*.md"))
        print(f"\n{hash_dir.name}  ({pdf_name})  {len(page_files)} cached pages")

        deleted = 0
        for f in page_files:
            text = f.read_text(encoding="utf-8")
            bad, reason = is_bad(text)
            if bad:
                print(f"  DELETE {f.name}  {reason}", flush=True)
                f.unlink()
                deleted += 1

        print(f"  => deleted {deleted}/{len(page_files)}", flush=True)
        total_checked += len(page_files)
        total_deleted += deleted

    print(f"\nDone. Deleted {total_deleted}/{total_checked} cache files.")


if __name__ == "__main__":
    main()
