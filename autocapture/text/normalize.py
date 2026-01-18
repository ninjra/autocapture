"""Text normalization for indexing-safe representations."""

from __future__ import annotations

import re
import unicodedata

_RE_WHITESPACE = re.compile(r"\s+")

_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(_PUNCT_TRANSLATION)
    normalized = "".join(
        ch for ch in normalized if unicodedata.category(ch) != "Cf"
    )
    normalized = _RE_WHITESPACE.sub(" ", normalized)
    return normalized.strip()
