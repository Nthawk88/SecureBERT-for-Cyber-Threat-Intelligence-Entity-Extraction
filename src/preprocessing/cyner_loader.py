"""Load the CyNER dataset (Stream A) from CoNLL/IOB files.

CyNER (Alam et al., 2022): 5,164 annotated records, IOB format, 5 entity
classes (Malware, Indicator, System, Organization, Vulnerability).

This module is the *only* sanctioned entry point for the quantitative data
stream. Scraped clear-web data (Stream B) must never be routed through here.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.utils.labels import LABEL_LIST


def read_conll(
    path: Union[str, Path],
    valid_labels: Optional[List[str]] = None,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Parse a CoNLL/IOB file into parallel token & tag sequences.

    A blank line terminates a sentence. Each non-blank line holds a token and its
    IOB tag separated by whitespace or a tab. ``-DOCSTART-`` markers and comment
    lines (starting with ``#``) are skipped.

    Parameters
    ----------
    path:
        File path to the ``.conll`` / ``.txt`` file.
    valid_labels:
        If given, any tag not in this set is coerced to ``"O"`` (defensive
        normalisation against stray labels).

    Returns
    -------
    (tokens, tags):
        Two lists of equal length; each element is one sentence.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"CyNER file not found: {path}. Place the CoNLL files under data/cyner/ "
            f"(see README 'Obtaining the CyNER dataset')."
        )

    label_set = set(valid_labels) if valid_labels is not None else None
    sentences: List[List[str]] = []
    labels: List[List[str]] = []
    cur_tokens: List[str] = []
    cur_tags: List[str] = []

    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                if cur_tokens:
                    sentences.append(cur_tokens)
                    labels.append(cur_tags)
                    cur_tokens, cur_tags = [], []
                continue
            if stripped.startswith("-DOCSTART-") or stripped.startswith("#"):
                continue
            parts = stripped.split("\t") if "\t" in stripped else stripped.split()
            if len(parts) < 2:
                # Token with no tag -> treat as outside.
                token, tag = parts[0], "O"
            else:
                token, tag = parts[0], parts[-1]
            if label_set is not None and tag not in label_set:
                tag = "O"
            cur_tokens.append(token)
            cur_tags.append(tag)

    if cur_tokens:  # trailing sentence without final blank line
        sentences.append(cur_tokens)
        labels.append(cur_tags)

    return sentences, labels


def load_cyner_datasets(
    config=None,
    label_list: Optional[List[str]] = None,
):
    """Load train/dev/test CyNER splits into a :class:`datasets.DatasetDict`.

    Each example has fields ``tokens`` (List[str]) and ``ner_tags`` (List[int]),
    the latter indexed by ``label_list``.
    """
    from datasets import Dataset, DatasetDict, Features, Sequence, Value, ClassLabel

    from src.utils.config import load_config

    cfg = config or load_config()
    labels = label_list or LABEL_LIST
    label2id = {label: i for i, label in enumerate(labels)}

    split_paths = {
        "train": cfg.path("paths.cyner_train"),
        "validation": cfg.path("paths.cyner_dev"),
        "test": cfg.path("paths.cyner_test"),
    }

    features = Features(
        {
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=labels)),
        }
    )

    splits = {}
    for split, path in split_paths.items():
        tokens, tags = read_conll(path, valid_labels=labels)
        tag_ids = [[label2id[t] for t in seq] for seq in tags]
        splits[split] = Dataset.from_dict(
            {"tokens": tokens, "ner_tags": tag_ids}, features=features
        )

    return DatasetDict(splits)


def class_distribution(
    tags: List[List[str]],
    entity_level: bool = True,
) -> Dict[str, int]:
    """Count entities (or tokens) per class to expose the class imbalance.

    Parameters
    ----------
    tags:
        Sequences of IOB tags (as returned by :func:`read_conll`).
    entity_level:
        If True (default), count *entities* (one per ``B-`` tag). If False, count
        raw token tags (ignoring ``O``).
    """
    counter: Counter = Counter()
    for seq in tags:
        for tag in seq:
            if tag == "O":
                continue
            prefix, _, ent = tag.partition("-")
            if not ent:
                continue
            if entity_level:
                if prefix == "B":
                    counter[ent] += 1
            else:
                counter[ent] += 1
    return dict(counter)
