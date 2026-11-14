"""Entity extraction for the Phase 3 case study (scraped clear-web text).

Wraps a fine-tuned token-classification model and returns aggregated entities
(spans), ready to be fed into the correlation layer. Used on Stream B data ONLY;
quantitative metrics never run through this path.
"""

from __future__ import annotations

from typing import Dict, List, Optional


class EntityExtractor:
    """Load a fine-tuned NER model and extract aggregated entity spans."""

    # "simple" must not be used here. It groups by predicted tag without doing
    # word-level aggregation, so a value the tokenizer splits into many subwords
    # is emitted as many separate entities. That destroys exactly the entity type
    # this layer exists for: on a Securelist article containing 78 IoCs, "simple"
    # recovered 0 complete hashes (it produced 1465 fragments such as 'a', 'ed',
    # '98'), while every word-level strategy recovered all 46 MD5 hashes.
    #
    # Among the word-level strategies, "average" was rejected: scoring a span by
    # the mean over its subwords turns span length into a penalty, so downstream
    # filtering on confidence >= 0.7 discards long multi-subword values. Measured
    # over the 80-article corpus it kept only 14 Malware and 2 Vulnerability
    # entities, against 174 and 13 under "max" - a systematic bias against the
    # very entities (malware family names, CVE identifiers, hashes) the
    # correlation layer needs.
    def __init__(self, model_path: str, device: Optional[int] = None,
                 aggregation_strategy: str = "max") -> None:
        self.model_path = model_path
        self.device = device
        self.aggregation_strategy = aggregation_strategy
        self._pipe = None

    def load(self):
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, add_prefix_space=True)
        model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        device = self.device
        if device is None:
            try:
                import torch

                device = 0 if torch.cuda.is_available() else -1
            except ImportError:  # pragma: no cover
                device = -1
        self._pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy=self.aggregation_strategy,
            device=device,
        )
        return self

    @staticmethod
    def _chunk(text: str, max_chars: int = 1500) -> List[str]:
        """Split text into sentence-aligned chunks below the model token cap.

        Transformer NER models (RoBERTa here) cap at 512 tokens, so long articles
        must be chunked. ~1500 chars stays comfortably under the limit.
        """
        from src.preprocessing.text_cleaning import split_sentences

        sentences = split_sentences(text) or [text]
        chunks: List[str] = []
        buf = ""
        for sent in sentences:
            if len(buf) + len(sent) + 1 > max_chars and buf:
                chunks.append(buf.strip())
                buf = ""
            # hard-split sentences that are themselves too long
            while len(sent) > max_chars:
                chunks.append(sent[:max_chars])
                sent = sent[max_chars:]
            buf = f"{buf} {sent}".strip()
        if buf:
            chunks.append(buf.strip())
        return chunks

    def extract(self, text: str, max_chars: int = 200_000) -> List[Dict]:
        """Return a list of ``{value, type, confidence}`` entities (chunked)."""
        if self._pipe is None:
            self.load()
        if not text:
            return []
        entities: List[Dict] = []
        for chunk in self._chunk(text[:max_chars]):
            for ent in self._pipe(chunk):
                value = (ent.get("word") or "").strip()
                if value:
                    entities.append(
                        {
                            "value": value,
                            "type": ent.get("entity_group") or ent.get("entity"),
                            "confidence": float(ent.get("score", 0.0)),
                        }
                    )
        return entities

    def extract_unique(self, text: str) -> List[Dict]:
        """Like :meth:`extract` but de-duplicates by ``(type, value)``."""
        seen = {}
        for ent in self.extract(text):
            key = (ent["type"], ent["value"])
            if key not in seen or ent["confidence"] > seen[key]["confidence"]:
                seen[key] = ent
        return list(seen.values())
