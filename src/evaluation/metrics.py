"""Quantitative NER metrics (Phase 4.1).

Entity-level Precision/Recall/F1 via ``seqeval`` (NOT token-level), macro-F1, and
a confusion matrix with special emphasis on the ``Vulnerability`` class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def decode_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_list: List[str],
) -> Tuple[List[List[str]], List[List[str]]]:
    """Convert logits/argmax ids + gold ids into seqeval string sequences.

    Ignored positions (gold == -100) are dropped.
    """
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=2)
    true_pred: List[List[str]] = []
    true_lab: List[List[str]] = []
    for pred, lab in zip(predictions, labels):
        cur_p, cur_l = [], []
        for p, l in zip(pred, lab):
            if l == -100:
                continue
            cur_p.append(label_list[int(p)])
            cur_l.append(label_list[int(l)])
        true_pred.append(cur_p)
        true_lab.append(cur_l)
    return true_pred, true_lab


def per_class_report(
    true_labels: Sequence[Sequence[str]],
    true_predictions: Sequence[Sequence[str]],
) -> Dict[str, Dict[str, float]]:
    """Return seqeval's per-class + averaged P/R/F1 as a nested dict."""
    from seqeval.metrics import classification_report

    return classification_report(
        true_labels, true_predictions, output_dict=True, zero_division=0
    )


def summarise_metrics(
    true_labels: Sequence[Sequence[str]],
    true_predictions: Sequence[Sequence[str]],
    focus_class: str = "Vulnerability",
) -> Dict[str, float]:
    """Flatten the key metrics, surfacing the focus class explicitly."""
    from seqeval.metrics import f1_score, precision_score, recall_score

    report = per_class_report(true_labels, true_predictions)
    out: Dict[str, float] = {
        "overall_precision": precision_score(true_labels, true_predictions, zero_division=0),
        "overall_recall": recall_score(true_labels, true_predictions, zero_division=0),
        "overall_f1": f1_score(true_labels, true_predictions, zero_division=0),
        "macro_f1": report.get("macro avg", {}).get("f1-score", 0.0),
    }
    fc = report.get(focus_class, {})
    out[f"{focus_class}_precision"] = fc.get("precision", 0.0)
    out[f"{focus_class}_recall"] = fc.get("recall", 0.0)
    out[f"{focus_class}_f1"] = fc.get("f1-score", 0.0)
    return out


def token_confusion_matrix(
    true_labels: Sequence[Sequence[str]],
    true_predictions: Sequence[Sequence[str]],
    entity_types: List[str],
) -> np.ndarray:
    """Confusion matrix over entity *types* (B-/I- collapsed, plus ``O``).

    Rows = gold, columns = predicted. Useful to characterise which classes the
    model confuses ``Vulnerability`` with.
    """
    classes = ["O"] + entity_types
    index = {c: i for i, c in enumerate(classes)}

    def base(tag: str) -> str:
        return "O" if tag == "O" else tag.split("-", 1)[1]

    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for gold_seq, pred_seq in zip(true_labels, true_predictions):
        for g, p in zip(gold_seq, pred_seq):
            gi, pi = index.get(base(g)), index.get(base(p))
            if gi is not None and pi is not None:
                cm[gi, pi] += 1
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Entity-type Confusion Matrix",
    out_path: Optional[str] = None,
    normalize: bool = True,
):
    """Render (and optionally save) a confusion-matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = cm.astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(title)
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    return fig
