"""Phase 4 evaluation: confusion matrix + per-class report + stats roll-up.

Uses the fine-tuned SecureBERT model to predict on the CyNER **test** split
(Stream A — quantitative), then:
  * per-class Precision/Recall/F1 (seqeval, entity-level),
  * entity-type confusion matrix (PNG) characterising Vulnerability errors,
  * paired t-test summary from results/phase2_multiseed.csv,
  * correlation-layer overhead from results/phase3_casestudy.json (if present).

    python scripts/run_phase4.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.labels import entity_types
from src.preprocessing.cyner_loader import load_cyner_datasets
from src.preprocessing.tokenization import make_tokenize_fn
from src.evaluation.metrics import (
    decode_predictions,
    per_class_report,
    summarise_metrics,
    token_confusion_matrix,
    plot_confusion_matrix,
)


def main() -> None:
    cfg = load_config()
    labels = list(cfg.get("labels.label_list"))
    results_dir = cfg.abspath("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    model_dir = str(cfg.path("paths.securebert_dir"))
    print(f"[phase4] loading model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    raw = load_cyner_datasets(cfg)
    tok = raw.map(make_tokenize_fn(tokenizer, cfg), batched=True,
                  remove_columns=raw["test"].column_names)

    args = TrainingArguments(output_dir=str(cfg.path("paths.checkpoints_dir") / "phase4_eval"),
                             per_device_eval_batch_size=16, report_to=[])
    trainer = Trainer(model=model, args=args,
                      data_collator=DataCollatorForTokenClassification(tokenizer),
                      processing_class=tokenizer)

    print("[phase4] predicting on CyNER test split...")
    pred = trainer.predict(tok["test"])
    logits, gold = pred.predictions, pred.label_ids
    true_pred, true_lab = decode_predictions(np.asarray(logits), np.asarray(gold), labels)

    # per-class report
    report = per_class_report(true_lab, true_pred)
    summary = summarise_metrics(true_lab, true_pred, focus_class=cfg.get("labels.focus_class", "Vulnerability"))
    print("\n[phase4] per-class (seqeval):")
    for cls, sc in report.items():
        if isinstance(sc, dict) and "f1-score" in sc:
            print(f"  {cls:16s} P={sc['precision']:.3f} R={sc['recall']:.3f} "
                  f"F1={sc['f1-score']:.3f} (n={int(sc.get('support', 0))})")

    # confusion matrix over entity types
    ent_types = entity_types(labels)
    cm = token_confusion_matrix(true_lab, true_pred, ent_types)
    cm_path = str(results_dir / "confusion_matrix.png")
    plot_confusion_matrix(cm, ["O"] + ent_types, out_path=cm_path,
                          title="SecureBERT — Entity-type Confusion Matrix (CyNER test)")
    print(f"[phase4] confusion matrix -> {cm_path}")

    # roll up Phase 2 stats + Phase 3 overhead
    out = {"per_class": {k: v for k, v in report.items() if isinstance(v, dict)},
           "summary": summary, "confusion_matrix": cm.tolist(),
           "confusion_labels": ["O"] + ent_types}

    p2 = results_dir / "phase2_multiseed.csv"
    if p2.exists():
        import pandas as pd
        from src.evaluation.statistical_tests import paired_ttest
        df = pd.read_csv(p2)
        df = df[df.arm != "gamma_search"]
        out["ttests"] = {}
        for metric in ("vuln_f1", "macro_f1", "overall_f1"):
            base = df[df.arm == "baseline_ce"].sort_values("seed")[metric].tolist()
            cand = df[df.arm == "securebert_focal"].sort_values("seed")[metric].tolist()
            if len(base) == len(cand) >= 2:
                out["ttests"][metric] = paired_ttest(base, cand, metric_name=metric)

    p3 = results_dir / "phase3_casestudy.json"
    if p3.exists():
        out["phase3_overhead"] = json.load(open(p3, encoding="utf-8")).get("overhead")

    def _np(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(results_dir / "phase4_evaluation.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=_np)
    print(f"[phase4] evaluation written to {results_dir / 'phase4_evaluation.json'}")


if __name__ == "__main__":
    main()
