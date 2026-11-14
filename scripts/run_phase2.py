"""Turnkey Phase 2 runner: multi-seed NER training + paired t-test.

Trains the comparison arms under identical config, across every seed in
``project.seeds``, writes a tidy results CSV, and prints a per-arm summary plus
the baseline-vs-SecureBERT paired t-tests.

This script does NOT produce the final model. It writes its best-scoring
``securebert_focal`` run to ``models/checkpoints/securebert_focal_best/`` for
inspection only; ``scripts/train_final_model.py`` owns ``models/securebert_ner/``.

Designed to run unattended on a single GPU. Examples:

    # quick functional check (tiny subset, 1 epoch) -- verify it runs end to end
    python scripts/run_phase2.py --quick

    # full study: all seeds from config (currently 10), gamma 2.0
    python scripts/run_phase2.py

    # also sweep the focusing parameter on SecureBERT (first seed)
    python scripts/run_phase2.py --gamma-search

    # skip arm/seed combinations already present in the CSV
    python scripts/run_phase2.py --resume

    # rerun into a separate CSV (per-class scores for scripts/compare_models.py)
    python scripts/run_phase2.py --out phase2_perclass.csv --drop-checkpoints --resume
"""

from __future__ import annotations

import argparse
import shutil
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


def _subset(raw, n_train, n_eval):
    raw["train"] = raw["train"].select(range(min(n_train, len(raw["train"]))))
    raw["validation"] = raw["validation"].select(range(min(n_eval, len(raw["validation"]))))
    raw["test"] = raw["test"].select(range(min(n_eval, len(raw["test"]))))
    return raw


CLASSES = ("Indicator", "Malware", "Organization", "System", "Vulnerability")


def run_one(cfg, raw, model_key, focal, gamma, seed, arm_name=None):
    set_seed(seed)
    output_dir = (str(cfg.path("paths.checkpoints_dir") / f"phase2_{arm_name}_seed{seed}")
                  if arm_name else None)
    runner = NERTrainer(model_key=model_key, config=cfg, use_focal_loss=focal,
                        focal_gamma=gamma, seed=seed, output_dir=output_dir)
    model, tokenizer = runner.load_model_and_tokenizer()
    tok = raw.map(make_tokenize_fn(tokenizer, cfg), batched=True,
                  remove_columns=raw["train"].column_names)
    trainer = runner.build_trainer(tok["train"], tok["validation"])
    t0 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    m = trainer.evaluate(tok["test"], metric_key_prefix="test")
    infer_s = time.perf_counter() - t1
    return trainer, tokenizer, {
        "vuln_f1": m.get("test_Vulnerability_f1"),
        "vuln_precision": m.get("test_Vulnerability_precision"),
        "vuln_recall": m.get("test_Vulnerability_recall"),
        "macro_f1": m.get("test_macro_f1"),
        "overall_f1": m.get("test_f1"),
        "overall_precision": m.get("test_precision"),
        "overall_recall": m.get("test_recall"),
        **{f"{c}_{k}": m.get(f"test_{c}_{k}")
           for c in CLASSES for k in ("precision", "recall", "f1")},
        "train_seconds": round(train_s, 1),
        "infer_seconds": round(infer_s, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gamma-search", action="store_true",
                        help="sweep focal gamma on SecureBERT (first seed only)")
    parser.add_argument("--quick", action="store_true",
                        help="tiny subset + 1 epoch to verify the pipeline")
    parser.add_argument("--arms", nargs="*", default=None,
                        help="subset of arms: baseline_ce baseline_focal securebert_focal")
    parser.add_argument("--resume", action="store_true",
                        help="skip arm+seed rows already in the results CSV (Colab quota-safe)")
    parser.add_argument("--out", default=None,
                        help="results CSV name inside results/ (default phase2_multiseed.csv); "
                             "with --out, securebert_focal_best is not touched")
    parser.add_argument("--drop-checkpoints", action="store_true",
                        help="delete each run's checkpoint after scoring (~1.4 GB per run)")
    args = parser.parse_args()

    import pandas as pd
    from src.evaluation.statistical_tests import paired_ttest, format_report

    try:
        from scripts.get_cyner import ensure_cyner
        ensure_cyner(verbose=True)
    except Exception as e:  # pragma: no cover
        print(f"[warn] could not auto-fetch CyNER ({e}); assuming files exist.")

    cfg = load_config()
    seeds = args.seeds or cfg.get("project.seeds", [42, 123, 456])
    gamma = args.gamma if args.gamma is not None else float(cfg.get("focal_loss.default_gamma", 2.0))

    if args.quick:
        cfg.to_dict()["training"]["num_epochs"] = 1
        seeds = seeds[:1]
        print("[quick] 1 epoch, single seed, subset")

    raw = load_cyner_datasets(cfg)
    if args.quick:
        raw = _subset(raw, 200, 100)

    results_dir = cfg.abspath("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # -- experiment arms (identical training config) -------------------------
    arms = [
        {"name": "baseline_ce", "model": "baseline", "focal": False, "gamma": None},
        {"name": "baseline_focal", "model": "baseline", "focal": True, "gamma": gamma},
        {"name": "securebert_focal", "model": "securebert", "focal": True, "gamma": gamma},
    ]
    if args.arms:
        wanted = set(args.arms)
        arms = [a for a in arms if a["name"] in wanted]
        if not arms:
            raise SystemExit(f"--arms must be one of baseline_ce, baseline_focal, securebert_focal")

    csv_name = args.out or ("phase2_quick.csv" if args.quick else "phase2_multiseed.csv")
    csv_path = results_dir / csv_name
    rows = []
    done = set()
    if args.resume and csv_path.exists():
        prev = pd.read_csv(csv_path)
        rows = prev.to_dict("records")
        done = set()
        for r in rows:
            g = r.get("gamma")
            g = None if g is None or (isinstance(g, float) and pd.isna(g)) else float(g)
            if r["arm"] == "gamma_search":
                done.add((r["arm"], int(r["seed"]), g))
            else:
                done.add((r["arm"], int(r["seed"])))
        print(f"[resume] loaded {len(rows)} rows from {csv_path}")

    def _flush():
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    best_vuln = -1.0
    sb_rows = [r for r in rows if r.get("arm") == "securebert_focal"]
    if sb_rows:
        best_vuln = max(float(r.get("vuln_f1") or 0) for r in sb_rows)

    for arm in arms:
        for seed in seeds:
            key = (arm["name"], int(seed))
            if key in done:
                print(f"[skip] {arm['name']} seed={seed} already in CSV")
                continue
            print(f"\n=== {arm['name']} | seed={seed} | gamma={arm['gamma']} ===")
            trainer, tokenizer, metrics = run_one(
                cfg, load_cyner_datasets(cfg) if not args.quick else raw,
                arm["model"], arm["focal"], arm["gamma"], seed, arm_name=arm["name"],
            )
            row = {"arm": arm["name"], "model": arm["model"], "focal": arm["focal"],
                   "gamma": arm["gamma"], "seed": seed, **metrics}
            rows.append(row)
            done.add(key)
            _flush()
            print({k: row[k] for k in ("vuln_f1", "macro_f1", "overall_f1")})
            print(f"[checkpoint] wrote {csv_path}")
            # Keep the best-scoring SecureBERT+Focal run for inspection only.
            #
            # This deliberately does NOT write to paths.securebert_dir: selecting
            # the seed with the highest Vulnerability F1 is cherry-picking, and
            # that metric swings from 0.00 to 0.82 across seeds. The model used
            # by Phase 3 and Phase 4 is produced by scripts/train_final_model.py,
            # which fixes the seed to project.seed and applies the mitigation
            # strategy that won the ablation.
            if (args.out is None and arm["name"] == "securebert_focal"
                    and (metrics["vuln_f1"] or 0) > best_vuln):
                best_vuln = metrics["vuln_f1"] or 0
                out = str(cfg.path("paths.checkpoints_dir") / "securebert_focal_best")
                trainer.save_model(out)
                tokenizer.save_pretrained(out)
                print(f"[save] best securebert_focal so far (vuln_f1={best_vuln:.4f}, "
                      f"seed={seed}) -> {out} [not the final model]")
            if args.drop_checkpoints:
                shutil.rmtree(trainer.args.output_dir, ignore_errors=True)

    # optional gamma sweep on SecureBERT (first seed)
    if args.gamma_search:
        for g in cfg.get("focal_loss.gamma_grid", [1.0, 2.0, 3.0]):
            key = ("gamma_search", int(seeds[0]), float(g))
            if key in done:
                print(f"[skip] gamma-search g={g} already in CSV")
                continue
            print(f"\n=== gamma-search | securebert | seed={seeds[0]} | gamma={g} ===")
            _, _, metrics = run_one(cfg, load_cyner_datasets(cfg), "securebert", True, g, seeds[0])
            rows.append({"arm": "gamma_search", "model": "securebert", "focal": True,
                         "gamma": g, "seed": seeds[0], **metrics})
            done.add(key)
            _flush()

    df = pd.DataFrame(rows)
    _flush()
    print(f"\nResults written to {csv_path}")

    # -- summary + t-test ----------------------------------------------------
    print("\n=== Per-arm summary (mean +/- std) ===")
    summ = df[df.arm != "gamma_search"].groupby("arm")[["vuln_f1", "macro_f1", "overall_f1"]]
    print(summ.agg(["mean", "std"]).round(4).to_string())

    alpha = cfg.get("evaluation.alpha", 0.05)
    any_test = False
    for metric in ("vuln_f1", "macro_f1", "overall_f1"):
        base = df[df.arm == "baseline_ce"].sort_values("seed")[metric].tolist()
        cand = df[df.arm == "securebert_focal"].sort_values("seed")[metric].tolist()
        if len(base) == len(cand) >= 2:
            any_test = True
            print(f"\n=== Paired t-test: baseline_ce vs securebert_focal ({metric}) ===")
            print(format_report(paired_ttest(base, cand, metric_name=metric, alpha=alpha)))
    if not any_test:
        print("\n[info] need >=2 shared seeds for a paired t-test (use full run, not --quick).")


if __name__ == "__main__":
    main()
