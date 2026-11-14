"""Train and persist the FINAL NER model used by Phase 3 and Phase 4.

The mitigation strategy is not chosen by hand: it is the winner of the ablation
study (``scripts/run_ablation.py`` -> ``results/ablation.csv``), where random
oversampling of the Vulnerability class reached the highest Vulnerability F1
(0.822 +/- 0.021) while also improving macro F1, and reproduced identically
across two independent executions.

The seed is fixed to ``project.seed`` from the config (42) rather than the
best-scoring seed, so the saved model is not cherry-picked. Oversampling touches
the TRAIN split only; validation and test keep their natural distribution.

    python scripts/train_final_model.py                 # SecureBERT + oversampling
    python scripts/train_final_model.py --epochs 10 --lr 5e-5  # tuned (val macro F1)
    python scripts/train_final_model.py --seed 123
    python scripts/train_final_model.py --strategy focal # reproduce the old model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.seeding import set_seed
from src.preprocessing.cyner_loader import load_cyner_datasets
from src.preprocessing.tokenization import make_tokenize_fn
from src.ner.trainer import NERTrainer
from src.ner.oversampling import random_oversample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["oversample", "focal"], default="oversample",
                        help="imbalance mitigation; default is the ablation winner")
    parser.add_argument("--seed", type=int, default=None,
                        help="defaults to project.seed so the model is not cherry-picked")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override training.num_epochs (default: config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="override training.learning_rate (default: config)")
    args = parser.parse_args()

    cfg = load_config()
    if args.epochs is not None:
        cfg.to_dict()["training"]["num_epochs"] = args.epochs
    if args.lr is not None:
        cfg.to_dict()["training"]["learning_rate"] = args.lr
    labels = list(cfg.get("labels.label_list"))
    seed = args.seed if args.seed is not None else int(cfg.get("project.seed", 42))
    use_focal = args.strategy == "focal"

    set_seed(seed)
    runner = NERTrainer(model_key="securebert", config=cfg, use_focal_loss=use_focal,
                        seed=seed)
    model, tokenizer = runner.load_model_and_tokenizer()

    raw = load_cyner_datasets(cfg)
    train_before = len(raw["train"])
    if args.strategy == "oversample":
        raw["train"] = random_oversample(
            raw["train"],
            target_class=cfg.get("ablation.oversampling.target_class", "Vulnerability"),
            oversample_factor=int(cfg.get("ablation.oversampling.oversample_factor", 3)),
            label_list=labels, seed=seed, split_name="train",
        )
    print(f"[final] strategy={args.strategy} seed={seed} "
          f"epochs={cfg.get('training.num_epochs')} lr={cfg.get('training.learning_rate')} "
          f"train {train_before} -> {len(raw['train'])} sentences")

    tok = raw.map(make_tokenize_fn(tokenizer, cfg), batched=True,
                  remove_columns=raw["train"].column_names)
    trainer = runner.build_trainer(tok["train"], tok["validation"])

    t0 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t0
    m = trainer.evaluate(tok["test"], metric_key_prefix="test")

    out = str(cfg.path("paths.securebert_dir"))
    trainer.save_model(out)
    tokenizer.save_pretrained(out)

    classes = ("Malware", "Indicator", "Organization", "System", "Vulnerability")
    summary = {
        "strategy": args.strategy,
        "seed": seed,
        "epochs": int(cfg.get("training.num_epochs")),
        "learning_rate": float(cfg.get("training.learning_rate")),
        "train_sentences": len(raw["train"]),
        "vuln_f1": m.get("test_Vulnerability_f1"),
        "vuln_precision": m.get("test_Vulnerability_precision"),
        "vuln_recall": m.get("test_Vulnerability_recall"),
        "macro_f1": m.get("test_macro_f1"),
        "overall_f1": m.get("test_f1"),
        "per_class": {
            c: {
                "precision": m.get(f"test_{c}_precision"),
                "recall": m.get(f"test_{c}_recall"),
                "f1": m.get(f"test_{c}_f1"),
            } for c in classes
        },
        "token_level": {
            c: {
                "precision": m.get(f"test_tok_{c}_precision"),
                "recall": m.get(f"test_tok_{c}_recall"),
                "f1": m.get(f"test_tok_{c}_f1"),
            } for c in classes
        },
        "tok_macro_f1": m.get("test_tok_macro_f1"),
        "tok_accuracy": m.get("test_tok_accuracy"),
        "train_seconds": round(train_s, 1),
        "saved_to": out,
    }
    results_dir = cfg.abspath("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "final_model.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n[final] test metrics:")
    for k in ("vuln_f1", "vuln_precision", "vuln_recall", "macro_f1", "overall_f1"):
        v = summary[k]
        print(f"  {k:16s} {v:.4f}" if isinstance(v, float) else f"  {k:16s} {v}")
    print(f"[final] model saved -> {out}")
    print(f"[final] summary     -> {results_dir / 'final_model.json'}")


if __name__ == "__main__":
    main()
