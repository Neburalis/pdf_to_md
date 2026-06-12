import re
from collections import Counter


_MATH_UNICODE_RE = re.compile(
    r"[∈∉∀∃∂→←↔⇒⇔≤≥≠≈∞∑∏∫∬∭√∪∩⊂⊃⊆⊇∧∨⊕⊗±∓≡≃≅∝∠⊥∥⌊⌋⌈⌉⩽⩾"
    r"αβγδεζηθικλμνξπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ"
    r"¹²³⁰⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉]"
)


def _cyrillic_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 1.0
    return sum(1 for char in letters if "Ѐ" <= char <= "ӿ") / len(letters)


def _repetition_score(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
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
    stripped = text.strip()
    if len(stripped) < 50:
        return False, ""
    if "i.imgur.com" in stripped:
        return True, "Fake imgur image URLs"
    if re.search(
        r"\[Section \d+[^\]]*\]|\[img-\d+\]|\[Figure \d+\]",
        stripped,
    ):
        return True, "Template placeholders"
    if (
        "Convert this PDF page" in stripped
        or "Output ONLY the Markdown" in stripped
    ):
        return True, "OCR prompt text in output"

    cyrillic_ratio = _cyrillic_ratio(stripped)
    if len(stripped) > 300 and cyrillic_ratio < 0.08:
        return True, f"English content (Cyrillic {cyrillic_ratio:.1%})"
    if len(stripped) > 150 and cyrillic_ratio >= 0.10:
        no_latex = _strip_latex(stripped)
        math_symbols = len(_MATH_UNICODE_RE.findall(no_latex))
        dollar_signs = stripped.count("$")
        if math_symbols >= 8 and dollar_signs < 4:
            return True, (
                "Unicode math outside LaTeX "
                f"({math_symbols} symbols, {dollar_signs} $)"
            )

    repetition = _repetition_score(stripped)
    if repetition > 0.62:
        return True, f"Highly repetitive ({repetition:.0%} identical lines)"
    return False, ""
