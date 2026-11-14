"""Ablation: inverse-frequency class weighting.

Computes per-label weights from the TRAIN split only and exposes them as a
tensor that can be passed to :class:`~src.ner.focal_loss.FocalLoss` or
``nn.CrossEntropyLoss``. Rare labels (e.g. ``Vulnerability``) receive larger
weights, counteracting the majority-class gradient dominance.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

import torch


def compute_label_counts(tag_id_sequences: Sequence[Sequence[int]], num_labels: int) -> List[int]:
    """Count occurrences of each label id across (train) sequences."""
    counter: Counter = Counter()
    for seq in tag_id_sequences:
        for label_id in seq:
            if label_id is None or label_id < 0:  # ignore padding / -100
                continue
            counter[int(label_id)] += 1
    return [counter.get(i, 0) for i in range(num_labels)]


def inverse_frequency_weights(
    tag_id_sequences: Sequence[Sequence[int]],
    num_labels: int,
    smoothing: float = 1.0,
    normalise: bool = True,
) -> torch.Tensor:
    """Return inverse-frequency class weights as a float tensor of size ``num_labels``.

    Parameters
    ----------
    tag_id_sequences:
        Label-id sequences from the TRAIN split only (never val/test).
    num_labels:
        Size of the label space (11 for CyNER).
    smoothing:
        Additive smoothing to avoid division by zero for absent labels.
    normalise:
        If True, scale weights so their mean is 1.0 (keeps loss magnitude stable).
    """
    counts = compute_label_counts(tag_id_sequences, num_labels)
    weights = torch.tensor(
        [1.0 / (c + smoothing) for c in counts], dtype=torch.float
    )
    if normalise:
        weights = weights * (num_labels / weights.sum())
    return weights


def weights_summary(
    weights: torch.Tensor, label_list: List[str]
) -> Dict[str, float]:
    """Human-readable label -> weight mapping for logging."""
    return {label_list[i]: round(float(w), 4) for i, w in enumerate(weights)}
