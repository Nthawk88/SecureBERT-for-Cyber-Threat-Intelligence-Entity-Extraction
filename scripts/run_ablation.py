"""Turnkey ablation: which class-imbalance mitigation helps Vulnerability most?

Holds the model fixed (SecureBERT by default, or BERT-base-uncased via --model
baseline) and the training config identical, varying ONLY the imbalance-mitigation
strategy, so the effect of each technique is isolated:

  * securebert_ce            - control (plain cross-entropy, no mitigation)
  * securebert_focal         - Focal Loss (gamma=2)
  * securebert_classweight   - inverse-frequency class weighting (weighted CE)
  * securebert_oversample    - random oversampling of Vulnerability sentences
  * securebert_augment       - template CVE augmentation of the train split
  * (--combos) focal+oversample, focal+augment

Resampling/weighting touch the TRAIN split ONLY (val/test keep the natural
distribution). Writes results/ablation.csv + per-technique summary and a paired
t-test of each technique vs the CE control on Vulnerability F1.

    python scripts/run_ablation.py                 # 5 arms x first 3 seeds = 15 runs
    python scripts/run_ablation.py --seeds 42 123  # faster
    python scripts/run_ablation.py --combos        # also test Focal+resampling
    python scripts/run_ablation.py --quick         # tiny smoke run
    python scripts/run_ablation.py --model baseline --arms ce oversample
                                                   # same arms on BERT-base-uncased
                                                   # -> results/ablation_bert.csv
    # both models with oversampling on all 10 seeds, into one CSV
    python scripts/run_ablation.py --arms oversample --seeds 42 123 456 789 1011 1213 1415 1617 1819 2021 --out oversample_10seeds.csv --drop-checkpoints --resume
    python scripts/run_ablation.py --model baseline --arms oversample --seeds 42 123 456 789 1011 1213 1415 1617 1819 2021 --out oversample_10seeds.csv --drop-checkpoints --resume
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.seeding import set_seed
from src.preprocessing.cyner_loader import load_cyner_datasets
from src.preprocessing.tokenization import make_tokenize_fn
from src.ner.trainer import NERTrainer
from src.ner.class_weighting import inverse_frequency_weights
from src.ner.oversampling import random_oversample
from src.ner.augmentation import augment_train_dataset


def _apply_transform(raw, transform, labels, cfg, seed):
    """Apply an imbalance transform to the TRAIN split only; return new raw dict."""
    if transform == "oversample":
        factor = int(cfg.get("ablation.oversampling.oversample_factor", 3))
        raw["train"] = random_oversample(
            raw["train"], target_class=cfg.get("ablation.oversampling.target_class", "Vulnerability"),
            oversample_factor=factor, label_list=labels, seed=seed, split_name="train",
        )
    elif transform == "augment":
        n_aug = int(cfg.get("ablation.augmentation.num_augmented", 200))
        raw["train"] = augment_train_dataset(
            raw["train"], num_augmented=n_aug,
            target_class=cfg.get("ablation.augmentation.target_class", "Vulnerability"),
            label_list=labels, seed=seed, split_name="train",
        )
    return raw


def _random_resplit(raw, seed):
    """Pool train/val/test and re-split at random, keeping the official sizes.

    Mirrors TSTEM's "random sub-sampling" protocol. Sentences from one report can
    land on both sides of the split, so test entities are far more often seen in
    training than with the official split; used only to compare with TSTEM.
    """
    from datasets import concatenate_datasets

    sizes = {k: len(raw[k]) for k in ("train", "validation", "test")}
    pool = concatenate_datasets([raw["train"], raw["validation"], raw["test"]]).shuffle(seed=seed)
    n_te, n_va = sizes["test"], sizes["validation"]
    raw["test"] = pool.select(range(0, n_te))
    raw["validation"] = pool.select(range(n_te, n_te + n_va))
    raw["train"] = pool.select(range(n_te + n_va, len(pool)))
    return raw


def _calibrate_vuln_bias(val_out, test_out, labels, grid):
    """Pick a logit bonus for B-/I-Vulnerability on VALIDATION, apply it to test.

    The bonus is chosen by validation token-level Vulnerability F1 (ties go to the
    smaller bonus), so the test set never influences the choice.
    """
    from src.ner.trainer import build_compute_metrics

    score = build_compute_metrics(labels)
    cols = [labels.index(t) for t in ("B-Vulnerability", "I-Vulnerability")]

    def shifted(logits, b):
        out = np.array(logits, dtype=np.float32, copy=True)
        out[..., cols] += b
        return out

    best_b, best_f1 = 0.0, -1.0
    per_bias = {}
    for b in grid:
        f1 = score((shifted(val_out.predictions, b), val_out.label_ids)).get("tok_Vulnerability_f1", 0.0)
        per_bias[f"bias{b:g}_val_tok_vuln_f1"] = f1
        # logged so a single bias can later be fixed from mean validation F1 across seeds
        per_bias[f"bias{b:g}_tok_vuln_f1"] = score(
            (shifted(test_out.predictions, b), test_out.label_ids)).get("tok_Vulnerability_f1", 0.0)
        if f1 > best_f1 + 1e-12:
            best_b, best_f1 = float(b), f1
    v = score((shifted(val_out.predictions, best_b), val_out.label_ids))
    m = score((shifted(test_out.predictions, best_b), test_out.label_ids))
    return (best_b, per_bias, {f"val_{k}": x for k, x in v.items()},
            {f"test_{k}": x for k, x in m.items()})


CVE_RE = re.compile(r"(?i)^CVE-\d{4}-\d+")


def _apply_cve_rule(out, words, labels):
    """Force words shaped like a CVE ID to Vulnerability (B- unless already B/I-).

    In the official CyNER train split every CVE-ID word is labelled Vulnerability
    (13/13), so the rule follows the annotation guideline, not the test set.
    Labels sit on the first subword of each word, so the i-th labelled position
    is word i. Returns (logits, number of words changed).
    """
    logits = np.array(out.predictions, dtype=np.float32, copy=True)
    b_v, i_v = labels.index("B-Vulnerability"), labels.index("I-Vulnerability")
    changed = 0
    for row, lab, ws in zip(logits, out.label_ids, words):
        for w, p in zip(ws, np.where(lab != -100)[0]):
            if CVE_RE.match(w) and int(row[p].argmax()) not in (b_v, i_v):
                row[p, b_v] = row[p].max() + 1.0
                changed += 1
    return logits, changed


def _softmax(x):
    x = np.asarray(x, dtype=np.float32)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _make_snapshot_callback(datasets):
    """Collect softmax outputs on each dataset after every epoch (no extra training)."""
    from transformers import TrainerCallback

    class SnapshotCallback(TrainerCallback):
        def __init__(self):
            self.trainer = None
            self.probs = {k: [] for k in datasets}
            self.label_ids = {}

        def on_epoch_end(self, args, state, control, **kwargs):
            for k, ds in datasets.items():
                out = self.trainer.predict(ds, metric_key_prefix=f"snap_{k}")
                self.probs[k].append(_softmax(out.predictions).astype(np.float16))
                self.label_ids[k] = out.label_ids

    return SnapshotCallback()


def _snapshot_scores(cb, labels, starts):
    """Per-epoch scores and averaged-epoch ("snapshot ensemble") scores."""
    from src.ner.trainer import build_compute_metrics

    score = build_compute_metrics(labels)
    n = len(cb.probs["test"])
    out = {}
    for e in range(n):
        for k, pre in (("validation", "val_"), ("test", "")):
            s = score((cb.probs[k][e].astype(np.float32), cb.label_ids[k]))
            out[f"ep{e + 1}_{pre}tok_vuln_f1"] = s.get("tok_Vulnerability_f1")
    full = {}
    for s0 in starts:
        if s0 > n:
            continue
        for k, pre in (("validation", "val_"), ("test", "test_")):
            avg = np.mean([p.astype(np.float32) for p in cb.probs[k][s0 - 1:]], axis=0)
            s = score((avg, cb.label_ids[k]))
            full[(s0, k)] = {f"{pre}{key}": v for key, v in s.items()}
            short = "val_" if k == "validation" else ""
            out[f"snap{s0}_{short}tok_vuln_f1"] = s.get("tok_Vulnerability_f1")
            out[f"snap{s0}_{short}tok_macro_f1"] = s.get("tok_macro_f1")
    return out, full


def run_one(cfg, labels, arm, seed, subset=None, model_key="securebert", drop_checkpoint=False,
            random_split=False, bias_grid=None, snapshot_starts=None, snapshot_from=None,
            cve_rule=False, init_seed=None, return_probs=False):
    """Train and score one model. ``seed`` fixes the data split; ``init_seed``
    (default: same as ``seed``) fixes weight init and batch order, so ensemble
    members can share a split but differ in training randomness."""
    init_seed = seed if init_seed is None else init_seed
    set_seed(init_seed)
    class_weights = None
    runner = NERTrainer(
        model_key=model_key, config=cfg,
        use_focal_loss=arm["focal"], focal_gamma=arm["gamma"],
        class_weights=None, seed=init_seed,
        output_dir=str(cfg.path("paths.checkpoints_dir")
                       / f"ablation_{arm['name']}_seed{seed}_init{init_seed}"),
    )
    model, tokenizer = runner.load_model_and_tokenizer()

    raw = load_cyner_datasets(cfg)
    if random_split:
        raw = _random_resplit(raw, seed)
    if subset:
        for sp, n in (("train", subset[0]), ("validation", subset[1]), ("test", subset[1])):
            raw[sp] = raw[sp].select(range(min(n, len(raw[sp]))))
    raw = _apply_transform(raw, arm.get("transform"), labels, cfg, seed)

    if arm.get("weight"):
        class_weights = inverse_frequency_weights(raw["train"]["ner_tags"], num_labels=len(labels))
        runner.class_weights = class_weights  # consumed when building focal trainer

    tok = raw.map(make_tokenize_fn(tokenizer, cfg), batched=True,
                  remove_columns=raw["train"].column_names)
    trainer = runner.build_trainer(tok["train"], tok["validation"])
    snap_cb = None
    if snapshot_starts or snapshot_from:
        snap_cb = _make_snapshot_callback({"validation": tok["validation"], "test": tok["test"]})
        snap_cb.trainer = trainer
        trainer.add_callback(snap_cb)
    t0 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t0
    test_out = trainer.predict(tok["test"], metric_key_prefix="test")
    val_out = trainer.predict(tok["validation"], metric_key_prefix="val")
    m, v = test_out.metrics, val_out.metrics
    calib = {}
    if cve_rule:
        from src.ner.trainer import build_compute_metrics

        score = build_compute_metrics(labels)
        calib.update({
            "model_only_val_tok_vuln_f1": v.get("val_tok_Vulnerability_f1"),
            "model_only_val_tok_macro_f1": v.get("val_tok_macro_f1"),
            "model_only_tok_Vulnerability_f1": m.get("test_tok_Vulnerability_f1"),
            "model_only_tok_Vulnerability_precision": m.get("test_tok_Vulnerability_precision"),
            "model_only_tok_Vulnerability_recall": m.get("test_tok_Vulnerability_recall"),
            "model_only_tok_macro_f1": m.get("test_tok_macro_f1"),
            "model_only_vuln_f1": m.get("test_Vulnerability_f1"),
            "model_only_macro_f1": m.get("test_macro_f1"),
        })
        lv, nv = _apply_cve_rule(val_out, raw["validation"]["tokens"], labels)
        lt, nt = _apply_cve_rule(test_out, raw["test"]["tokens"], labels)
        v = {f"val_{k}": x for k, x in score((lv, val_out.label_ids)).items()}
        m = {f"test_{k}": x for k, x in score((lt, test_out.label_ids)).items()}
        calib["cve_rule_changed_val"], calib["cve_rule_changed_test"] = nv, nt
    if snap_cb is not None:
        starts = sorted(set(snapshot_starts or []) | ({snapshot_from} if snapshot_from else set()))
        snap_log, snap_full = _snapshot_scores(snap_cb, labels, starts)
        calib.update(snap_log)
        if snapshot_from and (snapshot_from, "test") in snap_full:
            calib["best_epoch_tok_Vulnerability_f1"] = m.get("test_tok_Vulnerability_f1")
            calib["best_epoch_tok_macro_f1"] = m.get("test_tok_macro_f1")
            calib["best_epoch_vuln_f1"] = m.get("test_Vulnerability_f1")
            m = snap_full[(snapshot_from, "test")]
            v = snap_full[(snapshot_from, "validation")]
    if bias_grid:
        calib = {
            "uncal_val_tok_vuln_f1": v.get("val_tok_Vulnerability_f1"),
            "uncal_tok_Vulnerability_f1": m.get("test_tok_Vulnerability_f1"),
            "uncal_tok_Vulnerability_precision": m.get("test_tok_Vulnerability_precision"),
            "uncal_tok_Vulnerability_recall": m.get("test_tok_Vulnerability_recall"),
            "uncal_tok_macro_f1": m.get("test_tok_macro_f1"),
            "uncal_vuln_f1": m.get("test_Vulnerability_f1"),
            "uncal_macro_f1": m.get("test_macro_f1"),
        }
        bias, per_bias, v, m = _calibrate_vuln_bias(val_out, test_out, labels, bias_grid)
        calib["vuln_bias"] = bias
        calib.update(per_bias)
    metrics = _metric_row(m, v, calib)
    # label-distribution diagnostic (final reported model of this run, test split)
    try:
        preds = np.argmax(test_out.predictions, axis=-1)
        gold = np.asarray(test_out.label_ids)
        keep = gold != -100
        vid = {labels.index("B-Vulnerability"), labels.index("I-Vulnerability")}
        metrics["diag_gold_vuln_tokens_test"] = int(np.isin(gold[keep], list(vid)).sum())
        metrics["diag_pred_vuln_tokens_test"] = int(np.isin(preds[keep], list(vid)).sum())
        metrics["diag_pred_B_vuln_test"] = int((preds[keep] == labels.index("B-Vulnerability")).sum())
        metrics["diag_gold_B_vuln_test"] = int((gold[keep] == labels.index("B-Vulnerability")).sum())
        metrics["diag_pred_O_test"] = int((preds[keep] == labels.index("O")).sum())
        metrics["diag_gold_O_test"] = int((gold[keep] == labels.index("O")).sum())
        fin = test_out.predictions
        metrics["diag_mean_p_vuln_on_gold_vuln"] = float(
            _softmax(fin)[..., list(vid)].sum(-1)[keep & np.isin(gold, list(vid))].mean()
        ) if np.isin(gold[keep], list(vid)).any() else None
    except Exception as e:  # diagnostics must never break a run
        print(f"[warn] diagnostics failed: {e}")
    metrics["train_seconds"] = round(train_s, 1)
    metrics["train_size"] = len(raw["train"])
    if drop_checkpoint:
        shutil.rmtree(runner.output_dir, ignore_errors=True)
    if return_probs:
        probs = {"validation": _softmax(val_out.predictions).astype(np.float16),
                 "test": _softmax(test_out.predictions).astype(np.float16),
                 "validation_labels": val_out.label_ids, "test_labels": test_out.label_ids}
        return metrics, probs
    return metrics


def _metric_row(m, v, extra=None):
    """Flatten test metrics ``m`` (test_*) and validation metrics ``v`` (val_*) into a CSV row."""
    return {
        "val_f1": v.get("val_f1"),
        "val_macro_f1": v.get("val_macro_f1"),
        "val_vuln_f1": v.get("val_Vulnerability_f1"),
        "val_tok_macro_f1": v.get("val_tok_macro_f1"),
        "val_tok_vuln_f1": v.get("val_tok_Vulnerability_f1"),
        **(extra or {}),
        **{k[len("test_"):]: val for k, val in m.items() if k.startswith("test_tok_")},
        "vuln_f1": m.get("test_Vulnerability_f1"),
        "vuln_precision": m.get("test_Vulnerability_precision"),
        "vuln_recall": m.get("test_Vulnerability_recall"),
        "macro_f1": m.get("test_macro_f1"),
        "overall_f1": m.get("test_f1"),
        "overall_precision": m.get("test_precision"),
        "overall_recall": m.get("test_recall"),
        **{f"{c}_{k}": m.get(f"test_{c}_{k}")
           for c in ("Malware", "Indicator", "Organization", "System", "Vulnerability")
           for k in ("precision", "recall", "f1")},
    }


def run_ensemble(cfg, labels, arm, seed, n_members, **kw):
    """Train ``n_members`` models on the same split (different init seeds) and
    score the average of their softmax outputs. Member scores are kept too."""
    from src.ner.trainer import build_compute_metrics

    score = build_compute_metrics(labels)
    members, probs = [], []
    for k in range(n_members):
        mk, pk = run_one(cfg, labels, arm, seed, init_seed=seed + 10000 * k, return_probs=True, **kw)
        members.append(mk)
        probs.append(pk)
    avg = {sp: np.mean([p[sp].astype(np.float32) for p in probs], axis=0)
           for sp in ("validation", "test")}
    v = {f"val_{k}": x for k, x in score((avg["validation"], probs[0]["validation_labels"])).items()}
    m = {f"test_{k}": x for k, x in score((avg["test"], probs[0]["test_labels"])).items()}
    extra = {}
    for k, mk in enumerate(members):
        extra[f"member{k}_val_tok_vuln_f1"] = mk["val_tok_vuln_f1"]
        extra[f"member{k}_tok_vuln_f1"] = mk["tok_Vulnerability_f1"]
        extra[f"member{k}_tok_macro_f1"] = mk["tok_macro_f1"]
    for key in ("val_tok_vuln_f1", "tok_Vulnerability_f1", "tok_macro_f1"):
        extra[f"member_mean_{key}"] = float(np.mean([mk[key] for mk in members]))
    row = _metric_row(m, v, extra)
    row["train_seconds"] = round(sum(mk["train_seconds"] for mk in members), 1)
    row["train_size"] = members[0]["train_size"]
    row["n_members"] = n_members
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--combos", action="store_true",
                        help="also test Focal Loss combined with resampling")
    parser.add_argument("--quick", action="store_true",
                        help="tiny subset + 1 epoch to verify the pipeline")
    parser.add_argument("--resume", action="store_true",
                        help="skip technique+seed rows already in ablation.csv")
    parser.add_argument("--model", choices=["securebert", "baseline", "roberta"], default="securebert",
                        help="'baseline' = BERT-base-uncased; results go to ablation_bert.csv")
    parser.add_argument("--arms", nargs="*", default=None,
                        choices=["ce", "focal", "classweight", "oversample", "augment",
                                 "focal_oversample", "focal_augment"],
                        help="run only these arms (default: all)")
    parser.add_argument("--out", default=None,
                        help="results CSV name inside results/ (overrides ablation[_bert].csv)")
    parser.add_argument("--drop-checkpoints", action="store_true",
                        help="delete each run's checkpoint after scoring (~1.4 GB per run)")
    parser.add_argument("--epochs", type=int, default=None, help="override training.num_epochs")
    parser.add_argument("--lr", type=float, default=None, help="override training.learning_rate")
    parser.add_argument("--random-split", action="store_true",
                        help="TSTEM protocol: pool all CyNER sentences and re-split at random per seed")
    parser.add_argument("--oversample-factor", type=int, default=None,
                        help="override ablation.oversampling.oversample_factor")
    parser.add_argument("--vuln-bias-grid", type=float, nargs="*", default=None,
                        help="logit bonuses for B-/I-Vulnerability; the best one on validation "
                             "is applied to test (e.g. 0 0.5 1 1.5 2 2.5 3 4)")
    parser.add_argument("--snapshot-log", type=int, nargs="*", default=None,
                        help="also log per-epoch scores and the average of epochs S..last "
                             "for each start epoch S given (logging only)")
    parser.add_argument("--snapshot-from", type=int, default=None,
                        help="report the average of epochs S..last as the main test metrics")
    parser.add_argument("--cve-rule", action="store_true",
                        help="mark CVE-ID words as Vulnerability after the model; model-only "
                             "scores are kept in model_only_* columns")
    parser.add_argument("--ensemble", type=int, default=1,
                        help="train N models per split (init seeds seed, seed+10000, ...) and "
                             "score the average of their outputs; member scores are logged")
    args = parser.parse_args()

    import pandas as pd
    from src.evaluation.statistical_tests import paired_ttest, format_report

    try:
        from scripts.get_cyner import ensure_cyner
        ensure_cyner(verbose=True)
    except Exception as e:  # pragma: no cover
        print(f"[warn] could not auto-fetch CyNER ({e}); assuming files exist.")

    cfg = load_config()
    labels = list(cfg.get("labels.label_list"))
    # Deliberately the first 3 of project.seeds, not all 10: the ablation has 5
    # arms, so 10 seeds would mean 50 runs (~2h) to rank strategies whose gap is
    # large and consistent. The 10-seed protocol is reserved for run_phase2.py,
    # where the paired t-tests need the statistical power.
    seeds = args.seeds or list(cfg.get("project.seeds", [42, 123, 456]))[:3]
    gamma = args.gamma if args.gamma is not None else float(cfg.get("focal_loss.default_gamma", 2.0))
    subset = None
    if args.quick:
        cfg.to_dict()["training"]["num_epochs"] = 1
        seeds = seeds[:1]
        subset = (200, 100)
        print("[quick] 1 epoch, single seed, subset")

    prefix = {"securebert": "securebert", "baseline": "bert", "roberta": "roberta"}[args.model]
    arms = [
        {"name": "ce", "focal": False, "gamma": None, "weight": False, "transform": None},
        {"name": "focal", "focal": True, "gamma": gamma, "weight": False, "transform": None},
        {"name": "classweight", "focal": True, "gamma": 0.0, "weight": True, "transform": None},
        {"name": "oversample", "focal": False, "gamma": None, "weight": False, "transform": "oversample"},
        {"name": "augment", "focal": False, "gamma": None, "weight": False, "transform": "augment"},
    ]
    if args.combos:
        arms += [
            {"name": "focal_oversample", "focal": True, "gamma": gamma, "weight": False, "transform": "oversample"},
            {"name": "focal_augment", "focal": True, "gamma": gamma, "weight": False, "transform": "augment"},
        ]
    if args.arms:
        arms = [a for a in arms if a["name"] in args.arms]
    tag = ""
    if args.epochs is not None:
        cfg.to_dict()["training"]["num_epochs"] = args.epochs
        tag += f"_e{args.epochs}"
    if args.lr is not None:
        cfg.to_dict()["training"]["learning_rate"] = args.lr
        tag += f"_lr{args.lr:g}"
    if args.random_split:
        tag += "_rs"
    os_tag = ""
    if args.oversample_factor is not None:
        cfg.to_dict().setdefault("ablation", {}).setdefault("oversampling", {})[
            "oversample_factor"] = args.oversample_factor
        os_tag = f"_os{args.oversample_factor}"
    if args.vuln_bias_grid:
        tag += f"_b{args.vuln_bias_grid[0]:g}" if len(args.vuln_bias_grid) == 1 else "_cal"
    if args.snapshot_from:
        tag += f"_snap{args.snapshot_from}"
    if args.cve_rule:
        tag += "_cve"
    if args.ensemble > 1:
        tag += f"_ens{args.ensemble}"
    for a in arms:
        if a["transform"] == "oversample":
            a["name"] += os_tag
        a["name"] = f"{prefix}_{a['name']}{tag}"
    control = f"{prefix}_ce{tag}"

    results_dir = cfg.abspath("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = {"securebert": "ablation", "baseline": "ablation_bert", "roberta": "ablation_roberta"}[args.model]
    csv_path = results_dir / (args.out or (f"{stem}_quick.csv" if args.quick else f"{stem}.csv"))

    rows = []
    done = set()
    if args.resume and csv_path.exists():
        prev = pd.read_csv(csv_path)
        rows = prev.to_dict("records")
        done = {(r["technique"], int(r["seed"])) for r in rows}
        print(f"[resume] loaded {len(rows)} rows from {csv_path}")

    def _flush():
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    for arm in arms:
        for seed in seeds:
            if (arm["name"], int(seed)) in done:
                print(f"[skip] {arm['name']} seed={seed} already in CSV")
                continue
            print(f"\n=== {arm['name']} | seed={seed} | focal={arm['focal']} "
                  f"gamma={arm['gamma']} weight={arm['weight']} transform={arm['transform']} ===")
            kw = dict(subset=subset, model_key=args.model, drop_checkpoint=args.drop_checkpoints,
                      random_split=args.random_split, bias_grid=args.vuln_bias_grid,
                      snapshot_starts=args.snapshot_log, snapshot_from=args.snapshot_from,
                      cve_rule=args.cve_rule)
            if args.ensemble > 1:
                metrics = run_ensemble(cfg, labels, arm, seed, args.ensemble, **kw)
            else:
                metrics = run_one(cfg, labels, arm, seed, **kw)
            rows.append({"technique": arm["name"], "model": args.model,
                         "split": "random" if args.random_split else "official",
                         "epochs": int(cfg.get("training.num_epochs")),
                         "lr": float(cfg.get("training.learning_rate")),
                         "focal": arm["focal"], "gamma": arm["gamma"],
                         "weighted": arm["weight"], "transform": arm["transform"] or "none",
                         "oversample_factor": (int(cfg.get("ablation.oversampling.oversample_factor", 3))
                                               if arm["transform"] == "oversample" else None),
                         "seed": seed, **metrics})
            done.add((arm["name"], int(seed)))
            _flush()
            print({k: rows[-1][k] for k in ("vuln_f1", "macro_f1", "overall_f1")})
            print(f"[checkpoint] wrote {csv_path}")

    df = pd.DataFrame(rows)
    _flush()
    print(f"\nResults written to {csv_path}")

    print("\n=== Per-technique summary (Vulnerability F1, mean +/- std) ===")
    g = df.groupby("technique")[["vuln_f1", "vuln_recall", "macro_f1", "overall_f1"]].agg(["mean", "std"])
    print(g.round(4).sort_values(("vuln_f1", "mean"), ascending=False).to_string())

    alpha = cfg.get("evaluation.alpha", 0.05)
    ctrl = df[df.technique == control].sort_values("seed")["vuln_f1"].tolist()
    print("\n=== Paired t-test vs CE control (Vulnerability F1) ===")
    for tech in [a["name"] for a in arms if a["name"] != control]:
        cand = df[df.technique == tech].sort_values("seed")["vuln_f1"].tolist()
        if len(ctrl) == len(cand) >= 2:
            r = paired_ttest(ctrl, cand, metric_name=f"{tech}_vuln_f1", alpha=alpha)
            print(f"  {tech:30s} delta={r['mean_improvement']:+.4f} p={r['p_value']:.4f} "
                  f"sig={r['significant']}")
        else:
            print(f"  {tech:30s} (need >=2 shared seeds)")


if __name__ == "__main__":
    main()
