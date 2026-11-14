"""Ablation: random oversampling of minority-class sentences.

CRITICAL: oversampling is applied to the TRAIN split ONLY. The validation and
test splits must retain the natural class distribution. Passing anything other
than the train split here is a data-leakage bug.
"""

from __future__ import annotations

import random
from typing import List, Optional

from src.utils.labels import LABEL_LIST


def _contains_class(tag_ids: List[int], target_label_ids: set) -> bool:
    return any(t in target_label_ids for t in tag_ids)


def random_oversample(
    dataset,
    target_class: str = "Vulnerability",
    oversample_factor: int = 3,
    label_list: Optional[List[str]] = None,
    seed: int = 42,
    split_name: str = "train",
):
    """Duplicate sentences containing ``target_class`` entities in the train split.

    Parameters
    ----------
    dataset:
        A :class:`datasets.Dataset` with a ``ner_tags`` column of label ids.
    target_class:
        Entity class whose sentences are oversampled (default ``Vulnerability``).
    oversample_factor:
        Each qualifying sentence appears this many times in total (>= 1).
    label_list:
        Label list used to resolve ``B-/I-`` ids for ``target_class``.
    seed:
        RNG seed for the post-duplication shuffle (reproducibility).
    split_name:
        Guard rail: must be ``"train"``. Any other value raises ``ValueError``.
    """
    if split_name != "train":
        raise ValueError(
            f"Oversampling must only touch the train split, got '{split_name}'. "
            f"Validation/test must keep the natural class distribution."
        )
    if oversample_factor < 1:
        raise ValueError("oversample_factor must be >= 1")

    labels = label_list or LABEL_LIST
    target_label_ids = {
        i for i, lab in enumerate(labels) if lab in (f"B-{target_class}", f"I-{target_class}")
    }

    extra_indices: List[int] = []
    for idx, tags in enumerate(dataset["ner_tags"]):
        if _contains_class(list(tags), target_label_ids):
            extra_indices.extend([idx] * (oversample_factor - 1))

    if not extra_indices:
        return dataset

    from datasets import concatenate_datasets

    augmented = concatenate_datasets([dataset, dataset.select(extra_indices)])
    return augmented.shuffle(seed=seed)
