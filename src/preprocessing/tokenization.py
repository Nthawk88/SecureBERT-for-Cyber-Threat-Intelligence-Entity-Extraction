"""Subword tokenisation + IOB label alignment for token classification.

Only the *first* subword of each word receives the real label; subsequent
subwords and padding/special tokens receive ``-100`` so HuggingFace ignores them
in the loss and metrics.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional


def tokenize_and_align_labels(
    examples: Dict[str, List],
    tokenizer,
    max_length: int = 256,
    label_all_tokens: bool = False,
    ignore_index: int = -100,
) -> Dict[str, List]:
    """Tokenise pre-split words and align IOB labels onto subwords.

    Parameters
    ----------
    examples:
        Batch dict with keys ``tokens`` (List[List[str]]) and ``ner_tags``
        (List[List[int]]).
    tokenizer:
        A fast HuggingFace tokenizer (provides ``word_ids``).
    max_length:
        Maximum sequence length (truncation applied).
    label_all_tokens:
        If True, propagate the word label to *every* subword (with B-> I
        conversion left to the caller). Default False: only the first subword
        is labelled, the rest are ``ignore_index``.
    ignore_index:
        Label id used for ignored positions (default ``-100``).
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    aligned_labels: List[List[int]] = []
    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx: Optional[int] = None
        label_ids: List[int] = []
        for word_idx in word_ids:
            if word_idx is None:
                # special / padding token
                label_ids.append(ignore_index)
            elif word_idx != previous_word_idx:
                # first subword of a word -> real label
                label_ids.append(word_labels[word_idx])
            else:
                # subsequent subword of the same word
                label_ids.append(word_labels[word_idx] if label_all_tokens else ignore_index)
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def make_tokenize_fn(tokenizer, config=None) -> Callable[[Dict[str, List]], Dict[str, List]]:
    """Build a batched ``map`` function bound to a tokenizer + config settings."""
    from src.utils.config import load_config

    cfg = config or load_config()
    max_length = int(cfg.get("tokenizer.max_length", 256))
    label_all_tokens = bool(cfg.get("tokenizer.label_all_tokens", False))
    ignore_index = int(cfg.get("tokenizer.ignore_index", -100))

    def _fn(examples: Dict[str, List]) -> Dict[str, List]:
        return tokenize_and_align_labels(
            examples,
            tokenizer=tokenizer,
            max_length=max_length,
            label_all_tokens=label_all_tokens,
            ignore_index=ignore_index,
        )

    return _fn
