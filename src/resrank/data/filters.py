"""Language filtering helpers (English-only training data).

Two script checks (this is not language identification):
- CJK characters, rejected strictly (ratio <= max_ratio, 0.0 by default), as before.
- Any other non-Latin script (Cyrillic, Arabic, Thai, Devanagari, Bengali, Telugu, ...): a text is rejected when
  more than NON_LATIN_MAX_RATIO of its letters are in such a script (CJK is left to the first check). The tolerance keeps English passages that quote a
  Greek symbol or a foreign name; a Russian or Arabic text is ~100% non-Latin.
Latin-script languages (Finnish, Indonesian, Swahili, Spanish, ...) pass both checks: drop them by `source`
(s02_build_train --allow-sources / --deny-sources; checks/inspect_e2rank.py lists the sources).
"""
from __future__ import annotations

import re
import unicodedata

from resrank.schemas import RankingSample

NON_LATIN_MAX_RATIO = 0.3   # ASSUMED threshold, see the module docstring

# Unicode ranges for Chinese, Japanese and Korean scripts.
_CJK_RANGES = (
    (0x3400, 0x4DBF),    # CJK Extension A
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x3040, 0x30FF),    # Hiragana and Katakana
    (0xAC00, 0xD7AF),    # Hangul syllables
    (0x20000, 0x2A6DF),  # CJK Extension B
)


def is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def cjk_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are CJK. 0.0 for empty text."""
    if text.isascii():                      # fast path (C speed): most English text is pure ASCII
        return 0.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(is_cjk_char(c) for c in chars) / len(chars)


def contains_cjk(text: str) -> bool:
    return any(is_cjk_char(c) for c in text)


def non_latin_letter_ratio(text: str) -> float:
    """Fraction of letters in a script other than Latin or CJK (CJK has its own check). 0.0 without letters."""
    if text.isascii():
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    other = sum(not unicodedata.name(c, "").startswith("LATIN") and not is_cjk_char(c) for c in letters)
    return other / len(letters)


def _looks_english(text: str, max_cjk_ratio: float) -> bool:
    return cjk_ratio(text) <= max_cjk_ratio and non_latin_letter_ratio(text) <= NON_LATIN_MAX_RATIO


def is_english_sample(sample: RankingSample, max_ratio: float = 0.0) -> bool:
    """True if the query and every passage have a CJK ratio <= max_ratio and are mostly Latin-script letters.

    With max_ratio=0.0 any CJK character rejects the sample. The default is strict because
    a single Chinese passage among the 16 candidates makes the whole list mixed-language.
    """
    return all(_looks_english(t, max_ratio) for t in (sample.query, *sample.passages))


# Very common English function words. In English prose they are roughly a third of all words; in Finnish,
# Indonesian, Swahili, Spanish and other Latin-script languages they are rare. Used per *source*, not per sample.
_ENGLISH_WORDS = frozenset(
    "the of and to a in is that for it on with as was are be by this from or an at which not have has were "
    "their they you can what how".split()
)
_WORD = re.compile(r"[a-z]+")


def english_word_rate(text: str) -> float:
    """Fraction of (lower-cased ASCII) words that are common English function words. 0.0 for no words."""
    words = _WORD.findall(text.lower())
    if not words:
        return 0.0
    return sum(w in _ENGLISH_WORDS for w in words) / len(words)
