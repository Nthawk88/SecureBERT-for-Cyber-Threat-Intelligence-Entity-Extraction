"""Per-class F1 vs the numbers TSTEM reports, as CSV proof.

Reads ``results/tstem_comparison_runs.csv`` (written by run_ablation.py --out) and
writes:

  results/tstem_proof_per_seed.csv  one row per run: per-class F1 under both
                                    protocols, plus TSTEM's value per class
  results/tstem_proof_summary.csv   one row per arm x protocol x class: TSTEM's
                                    value, our mean/SD/min/max over seeds, how many
                                    seeds beat TSTEM, and a one-sample t-test of
                                    H1: our mean > TSTEM (one-sided)

Two scoring protocols on the CyNER test split:
  * token  - word-level per-class P/R/F1, B-/I- merged ("NER category
             classification", the protocol behind TSTEM's Table 6)
  * entity - seqeval strict span-level F1 (stricter; the rest of the thesis)

TSTEM publishes one score per class, not per-seed scores, so the test treats its
value as a fixed reference (one-sample), not as a paired observation.

    python scripts/tstem_proof.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CLASSES = ["Malware", "Indicator", "Organization", "System", "Vulnerability"]

# Balasubramanian et al., TSTEM, Table 6 (bert-base-uncased BERT-NER on CyNER)
TSTEM = {
    "Malware": {"precision": 0.85, "recall": 0.94, "f1": 0.90},
    "Indicator": {"precision": 0.83, "recall": 0.85, "f1": 0.84},
    "Organization": {"precision": 0.81, "recall": 0.87, "f1": 0.84},
    "System": {"precision": 0.71, "recall": 0.76, "f1": 0.73},
    "Vulnerability": {"precision": 0.45, "recall": 0.50, "f1": 0.48},
}
TSTEM_MACRO_F1 = sum(v["f1"] for v in TSTEM.values()) / len(TSTEM)

ARM_LABELS = {
    "bert_ce": "BERT-base-uncased, cross-entropy (TSTEM setup, reproduced)",
    "bert_oversample": "BERT-base-uncased + oversampling",
    "securebert_oversample": "SecureBERT + oversampling (proposed)",
}


def _col(protocol: str, cls: str, metric: str = "f1") -> str:
    if protocol == "token":
        return f"tok_{cls}_{metric}" if cls != "macro" else "tok_macro_f1"
    if cls == "macro":
        return "macro_f1"
    return f"{cls}_{metric}"


def _arm_key(technique: str) -> str:
    """Strip run-setting suffixes (epochs, lr, random split) but keep method ones."""
    return re.sub(r"_(e\d+|lr[0-9.]+(e-?\d+)?|rs)(?=_|$)", "", technique)


def _describe(arm: str) -> str:
    if arm in ARM_LABELS:
        return ARM_LABELS[arm]
    m = re.fullmatch(r"(securebert|bert)_(focal_)?oversample(?:_os(\d+))?(?:_(cal|b[0-9.]+))?(_cve)?"
                     r"(?:_ens(\d+))?", arm)
    if not m:
        return arm
    model = "SecureBERT" if m.group(1) == "securebert" else "BERT-base-uncased"
    parts = [model]
    if m.group(2):
        parts.append("Focal Loss")
    parts.append(f"oversampling x{m.group(3) or 3}")
    if m.group(4) == "cal":
        parts.append("Vulnerability bonus chosen per run on validation")
    elif m.group(4):
        parts.append(f"fixed Vulnerability bonus {m.group(4)[1:]} (chosen on validation)")
    if m.group(5):
        parts.append("CVE-ID rule")
    if m.group(6):
        parts.append(f"ensemble of {m.group(6)} models")
    return " + ".join(parts)


SPLIT_LABELS = {
    "official": "CyNER official split (reports kept apart)",
    "random": "random re-split per seed (TSTEM protocol)",
}


def main() -> None:
    src = RESULTS / "tstem_comparison_runs.csv"
    if not src.exists():
        sys.exit(f"{src} not found - run the final 10-seed runs first (see RUNBOOK).")
    df = pd.read_csv(src)
    df["arm"] = df["technique"].map(_arm_key)
    if "split" not in df:
        df["split"] = "official"
    df["split"] = df["split"].fillna("official")

    per_seed = []
    for _, r in df.sort_values(["split", "arm", "seed"]).iterrows():
        row = {"split": r["split"], "arm": r["arm"],
               "description": _describe(r["arm"]),
               "technique": r["technique"], "epochs": r.get("epochs"), "lr": r.get("lr"),
               "seed": int(r["seed"])}
        for proto in ("token", "entity"):
            for c in CLASSES + ["macro"]:
                row[f"{proto}_{c}_f1"] = round(float(r[_col(proto, c)]), 4)
        per_seed.append(row)
    per_seed_df = pd.DataFrame(per_seed)
    for c in CLASSES:
        per_seed_df[f"TSTEM_{c}_f1"] = TSTEM[c]["f1"]
    per_seed_df["TSTEM_macro_f1"] = round(TSTEM_MACRO_F1, 4)
    per_seed_path = RESULTS / "tstem_proof_per_seed.csv"
    per_seed_df.to_csv(per_seed_path, index=False)

    summary = []
    for (split, arm), sub in df.groupby(["split", "arm"]):
        for proto in ("token", "entity"):
            for c in CLASSES + ["macro"]:
                x = sub[_col(proto, c)].astype(float)
                ref = TSTEM_MACRO_F1 if c == "macro" else TSTEM[c]["f1"]
                t, p = stats.ttest_1samp(x, ref, alternative="greater")
                row = {
                    "split": split, "arm": arm, "description": _describe(arm),
                    "protocol": proto, "class": c,
                    "tstem_f1": round(ref, 4),
                    "our_mean_f1": round(x.mean(), 4), "our_sd": round(x.std(ddof=1), 4),
                    "our_min": round(x.min(), 4), "our_max": round(x.max(), 4),
                    "n_seeds": int(len(x)),
                    "difference": round(x.mean() - ref, 4),
                    "seeds_above_tstem": int((x > ref).sum()),
                    "t_one_sample": round(float(t), 3),
                    "p_one_sided_greater": float(f"{p:.3g}"),
                    "significantly_higher": bool(p < 0.05 and x.mean() > ref),
                }
                if proto == "token" and c != "macro":
                    for m in ("precision", "recall"):
                        row[f"our_mean_{m}"] = round(sub[_col(proto, c, m)].astype(float).mean(), 4)
                        row[f"tstem_{m}"] = TSTEM[c][m]
                summary.append(row)
    summary_df = pd.DataFrame(summary)
    summary_path = RESULTS / "tstem_proof_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    for split, proto in [(s, p) for s in summary_df.split.unique() for p in ("token", "entity")]:
        print(f"\n=== {SPLIT_LABELS.get(split, split)} | {proto}-level F1 vs TSTEM "
              f"(mean over seeds; seeds above TSTEM; * = significantly higher, p<0.05) ===")
        piv = summary_df[(summary_df.protocol == proto) & (summary_df.split == split)]
        arms = list(dict.fromkeys(piv.arm))
        print(f"  {'class':<14}{'TSTEM':>7}" + "".join(f"{a:>26}" for a in arms))
        for c in CLASSES + ["macro"]:
            line = f"  {c:<14}{piv[piv['class'] == c].tstem_f1.iloc[0]:>7.3f}"
            for a in arms:
                r = piv[(piv.arm == a) & (piv["class"] == c)].iloc[0]
                mark = "*" if r.significantly_higher else " "
                line += f"{r.our_mean_f1:>15.3f} ({r.seeds_above_tstem:>2}/{r.n_seeds}){mark:>4}"
            print(line)
    print(f"\nSaved -> {per_seed_path}\nSaved -> {summary_path}")


if __name__ == "__main__":
    main()
