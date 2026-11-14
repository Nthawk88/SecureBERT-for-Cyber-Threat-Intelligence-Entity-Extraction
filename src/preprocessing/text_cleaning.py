"""Clean scraped clear-web text (Stream B) for the Phase 3 case study.

NOTE: Stream B output is used ONLY for the qualitative end-to-end case study.
It must never enter quantitative model evaluation.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

_WHITESPACE_RE = re.compile(r"\s+")
_BOILERPLATE_PATTERNS = [
    re.compile(r"(?i)\bshare this article\b.*$"),
    re.compile(r"(?i)\bsubscribe to .*newsletter\b.*$"),
    re.compile(r"(?i)\bcookie(s)? (policy|settings)\b.*$"),
    re.compile(r"(?i)\ball rights reserved\b.*$"),
]


def normalise_unicode(text: str) -> str:
    """NFKC-normalise and strip control characters."""
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if ch == "\n" or not unicodedata.category(ch).startswith("C"))


def clean_text(text: str) -> str:
    """Light, non-destructive cleaning for downstream NER inference."""
    if not text:
        return ""
    text = normalise_unicode(text)
    for pat in _BOILERPLATE_PATTERNS:
        text = pat.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Naive sentence splitter (sufficient for NER chunking on clear-web prose)."""
    if not text:
        return []
    # Split on sentence-ending punctuation followed by whitespace + capital/digit.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]
