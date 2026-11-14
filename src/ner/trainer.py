"""Training orchestration for token-classification NER.

Provides:
  * :func:`build_compute_metrics` — seqeval entity-level P/R/F1 + per-class scores.
  * :class:`FocalLossTrainer` — HuggingFace ``Trainer`` with Focal Loss.
  * :class:`NERTrainer` — high-level wrapper that wires model, tokenizer,
    data collator, ``TrainingArguments`` and training/eval for one experiment.

The training configuration is identical for all models (read from config) so
comparisons between ``bert-base-uncased`` and ``SecureBERT`` are fair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from src.utils.config import load_config
from src.utils.labels import LABEL_LIST, build_label_maps
from src.utils.seeding import set_seed


def token_level_scores(true_labels: List[List[str]],
                       pred_labels: List[List[str]]) -> Dict[str, float]:
    """Per-class word-level P/R/F1 with B-/I- merged (keys prefixed ``tok_``).

    Every word counts once, so a 3-word entity found 2/3 correct still earns
    2 true positives, unlike seqeval where the whole span must match. This is
    the "NER category classification" protocol used by TSTEM (Table 6), and is
    reported only for comparison with it.
    """
    def cls(tag: str) -> str:
        return tag.split("-", 1)[1] if "-" in tag else tag

    gold = [cls(t) for seq in true_labels for t in seq]
    pred = [cls(t) for seq in pred_labels for t in seq]
    classes = sorted({c for c in gold if c != "O"})
    out: Dict[str, float] = {}
    f1s = []
    for c in classes:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[f"tok_{c}_precision"], out[f"tok_{c}_recall"], out[f"tok_{c}_f1"] = prec, rec, f1
        out[f"tok_{c}_support"] = float(tp + fn)
        f1s.append(f1)
    out["tok_macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    out["tok_accuracy"] = (sum(1 for g, p in zip(gold, pred) if g == p) / len(gold)) if gold else 0.0
    return out


def build_compute_metrics(label_list: List[str]) -> Callable:
    """Return a ``compute_metrics`` fn using seqeval (entity-level, not token)."""
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions: List[List[str]] = []
        true_labels: List[List[str]] = []
        for pred, lab in zip(predictions, labels):
            cur_pred, cur_lab = [], []
            for p, l in zip(pred, lab):
                if l == -100:  # ignored subword / padding
                    continue
                cur_pred.append(label_list[p])
                cur_lab.append(label_list[l])
            true_predictions.append(cur_pred)
            true_labels.append(cur_lab)

        report = classification_report(
            true_labels, true_predictions, output_dict=True, zero_division=0
        )
        results: Dict[str, float] = {
            "precision": precision_score(true_labels, true_predictions, zero_division=0),
            "recall": recall_score(true_labels, true_predictions, zero_division=0),
            "f1": f1_score(true_labels, true_predictions, zero_division=0),
            "macro_f1": report.get("macro avg", {}).get("f1-score", 0.0),
        }
        # per-class F1 (key emphasis on Vulnerability downstream)
        for cls, scores in report.items():
            if cls in {"micro avg", "macro avg", "weighted avg"}:
                continue
            if isinstance(scores, dict):
                results[f"{cls}_precision"] = scores.get("precision", 0.0)
                results[f"{cls}_recall"] = scores.get("recall", 0.0)
                results[f"{cls}_f1"] = scores.get("f1-score", 0.0)
        results.update(token_level_scores(true_labels, true_predictions))
        return results

    return compute_metrics


def _make_focal_trainer_class():
    """Build FocalLossTrainer lazily so torch/transformers stay optional imports."""
    from transformers import Trainer

    from src.ner.focal_loss import FocalLoss

    class FocalLossTrainer(Trainer):
        """``Trainer`` that replaces cross-entropy with :class:`FocalLoss`."""

        def __init__(self, *args, focal_gamma: float = 2.0, focal_alpha: float = 1.0,
                     class_weights=None, ignore_index: int = -100, **kwargs):
            super().__init__(*args, **kwargs)
            self._focal = FocalLoss(
                gamma=focal_gamma,
                alpha=focal_alpha,
                ignore_index=ignore_index,
                weight=class_weights,
            )

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            self._focal = self._focal.to(logits.device)
            loss = self._focal(logits, labels)
            return (loss, outputs) if return_outputs else loss

    return FocalLossTrainer


class NERTrainer:
    """High-level orchestrator for a single NER experiment / run.

    Parameters
    ----------
    model_key:
        ``"baseline"`` or ``"securebert"`` (keys under ``models`` in config).
    config:
        Optional pre-loaded :class:`~src.utils.config.Config`.
    use_focal_loss:
        Whether to train with Focal Loss (vs standard cross-entropy).
    focal_gamma:
        Focusing parameter; defaults to ``focal_loss.default_gamma``.
    class_weights:
        Optional per-class weight tensor (ablation: inverse-frequency weighting).
    seed:
        Random seed for this run (the multi-seed protocol uses ``project.seeds``).
    """

    def __init__(
        self,
        model_key: str = "securebert",
        config=None,
        use_focal_loss: bool = True,
        focal_gamma: Optional[float] = None,
        class_weights=None,
        seed: int = 42,
        output_dir: Optional[str] = None,
    ) -> None:
        self.cfg = config or load_config()
        self.model_key = model_key
        self.use_focal_loss = use_focal_loss
        self.seed = seed
        self.class_weights = class_weights

        self.label_list = list(self.cfg.get("labels.label_list", LABEL_LIST))
        self.label2id, self.id2label = build_label_maps(self.label_list)

        self.model_name = self.cfg.get(f"models.{model_key}.name")
        if self.model_name is None:
            raise KeyError(f"Unknown model_key '{model_key}' (check config 'models').")
        self.ignore_mismatched = bool(
            self.cfg.get(f"models.{model_key}.ignore_mismatched_sizes", False)
        )
        self.focal_gamma = (
            focal_gamma if focal_gamma is not None
            else float(self.cfg.get("focal_loss.default_gamma", 2.0))
        )
        self.focal_alpha = float(self.cfg.get("focal_loss.alpha", 1.0))
        self.ignore_index = int(self.cfg.get("tokenizer.ignore_index", -100))
        self.output_dir = output_dir or str(
            self.cfg.path("paths.checkpoints_dir") / f"{model_key}_seed{seed}"
        )

        self.tokenizer = None
        self.model = None

    # -- setup ----------------------------------------------------------------
    def load_model_and_tokenizer(self):
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        set_seed(self.seed)
        # SecureBERT is RoBERTa-based; add_prefix_space is required for
        # pre-tokenised (is_split_into_words=True) inputs.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, add_prefix_space=True
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=self.ignore_mismatched,
        )
        return self.model, self.tokenizer

    def build_training_arguments(self):
        import inspect
        from transformers import TrainingArguments

        t = self.cfg.training
        kwargs = dict(
            output_dir=self.output_dir,
            learning_rate=float(t.learning_rate),
            per_device_train_batch_size=int(t.train_batch_size),
            per_device_eval_batch_size=int(t.eval_batch_size),
            num_train_epochs=int(t.num_epochs),
            weight_decay=float(t.weight_decay),
            eval_strategy=str(t.evaluation_strategy),
            evaluation_strategy=str(t.evaluation_strategy),
            save_strategy=str(t.save_strategy),
            load_best_model_at_end=bool(t.load_best_model_at_end),
            metric_for_best_model=str(t.metric_for_best_model),
            greater_is_better=bool(t.greater_is_better),
            logging_steps=int(t.logging_steps),
            warmup_ratio=float(t.get("warmup_ratio", 0.0)),
            seed=self.seed,
            report_to=[],
            save_total_limit=1,
        )
        accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return TrainingArguments(**kwargs)

    def build_trainer(self, train_dataset, eval_dataset):
        from transformers import DataCollatorForTokenClassification, Trainer

        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()

        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        args = self.build_training_arguments()
        compute_metrics = build_compute_metrics(self.label_list)

        common = dict(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processing_class=self.tokenizer,
        )

        if self.use_focal_loss:
            FocalLossTrainer = _make_focal_trainer_class()
            return FocalLossTrainer(
                focal_gamma=self.focal_gamma,
                focal_alpha=self.focal_alpha,
                class_weights=self.class_weights,
                ignore_index=self.ignore_index,
                **common,
            )
        return Trainer(**common)
