"""Fast end-to-end smoke test of the NER training pipeline (CPU-friendly).

Validates the full path WITHOUT a GPU or long runtime:
  CyNER load -> subword label alignment -> FocalLossTrainer -> seqeval metrics.

Uses a tiny model and a small data subset so it finishes in ~1 minute on CPU.
This does NOT produce research results; it only proves the plumbing works on the
installed library versions. Run from the project root:

    python scripts/smoke_test_training.py --model prajjwal1/bert-tiny
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.seeding import set_seed
from src.preprocessing.cyner_loader import load_cyner_datasets
from src.preprocessing.tokenization import make_tokenize_fn
from src.ner.trainer import NERTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="prajjwal1/bert-tiny",
                        help="small HF model id for a quick CPU check")
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--n-eval", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--focal", action="store_true", default=True)
    args = parser.parse_args()

    cfg = load_config()
    # shrink the run for a smoke test (mutates the in-memory config only)
    cfg.to_dict()["training"]["num_epochs"] = args.epochs
    cfg.to_dict()["training"]["train_batch_size"] = 8
    cfg.to_dict()["training"]["eval_batch_size"] = 8
    cfg.to_dict()["training"]["logging_steps"] = 5

    set_seed(cfg.get("project.seed", 42))
    raw = load_cyner_datasets(cfg)
    raw["train"] = raw["train"].select(range(min(args.n_train, len(raw["train"]))))
    raw["validation"] = raw["validation"].select(range(min(args.n_eval, len(raw["validation"]))))
    raw["test"] = raw["test"].select(range(min(args.n_eval, len(raw["test"]))))
    print(f"[smoke] subset sizes: train={len(raw['train'])} "
          f"val={len(raw['validation'])} test={len(raw['test'])}")

    runner = NERTrainer(model_key="baseline", config=cfg, use_focal_loss=args.focal, seed=42)
    runner.model_name = args.model          # override to a tiny model
    runner.ignore_mismatched = True         # tiny head reinit, just in case
    runner.output_dir = str(cfg.path("paths.checkpoints_dir") / "smoke")

    model, tokenizer = runner.load_model_and_tokenizer()
    tok = raw.map(make_tokenize_fn(tokenizer, cfg), batched=True,
                  remove_columns=raw["train"].column_names)

    trainer = runner.build_trainer(tok["train"], tok["validation"])
    print("[smoke] training...")
    trainer.train()
    metrics = trainer.evaluate(tok["test"], metric_key_prefix="test")

    print("\n[smoke] test metrics:")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:28s} {v:.4f}")

    assert "test_f1" in metrics, "seqeval metrics missing"
    print("\n[smoke] PASS — pipeline (load -> align -> focal-loss train -> seqeval) works.")


if __name__ == "__main__":
    main()
