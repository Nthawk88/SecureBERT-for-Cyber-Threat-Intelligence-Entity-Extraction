"""NER label space utilities (CyNER, IOB, 5 entity classes -> 11 tags)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Canonical 11-tag IOB label list (mirrors config/config.yaml -> labels.label_list).
LABEL_LIST: List[str] = [
    "O",
    "B-Indicator", "I-Indicator",
    "B-Malware", "I-Malware",
    "B-System", "I-System",
    "B-Organization", "I-Organization",
    "B-Vulnerability", "I-Vulnerability",
]


def get_label_list(config=None) -> List[str]:
    """Return the label list, preferring the config if provided."""
    if config is not None and "labels" in config:
        labels = config.get("labels.label_list")
        if labels:
            return list(labels)
    return list(LABEL_LIST)


def build_label_maps(
    label_list: Optional[List[str]] = None,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return ``(label2id, id2label)`` mappings for a label list."""
    labels = label_list or LABEL_LIST
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}
    return label2id, id2label


def entity_types(label_list: Optional[List[str]] = None) -> List[str]:
    """Return the underlying entity classes (without B-/I- prefixes)."""
    labels = label_list or LABEL_LIST
    types: List[str] = []
    for label in labels:
        if label == "O":
            continue
        ent = label.split("-", 1)[1]
        if ent not in types:
            types.append(ent)
    return types
