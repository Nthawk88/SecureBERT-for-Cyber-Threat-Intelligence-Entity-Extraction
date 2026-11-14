"""Bootstrap confidence intervals for the revision (no change to any reported number).

1. Seed-level: percentile bootstrap (B=10,000) of the ten per-seed values -> 95% CI of the
   mean, and of paired differences between encoders (same seeds).
2. Sentence-level: bootstrap over the 748 official test sentences for the saved model
   (seqeval entity-level P/R/F1 per class).

Output: results/revision_bootstrap.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = ROOT / "results"
rng = np.random.default_rng(0)
B = 10_000
CLASSES = ["Malware", "Indicator", "Organization", "System", "Vulnerability"]


def ci_mean(x):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(1)
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def ci_paired(a, b):
    return ci_mean(np.asarray(a, float) - np.asarray(b, float))


def block(df_a, df_b, cols):
    out = {}
    for c in cols:
        a = df_a.sort_values("seed")[c].values
        b = df_b.sort_values("seed")[c].values
        out[c] = {"A_mean": round(float(a.mean()), 3), "A_ci": ci_mean(a),
                  "B_mean": round(float(b.mean()), 3), "B_ci": ci_mean(b),
                  "diff_B_minus_A": round(float((b - a).mean()), 3), "diff_ci": ci_paired(b, a)}
    return out


res = {}
m_cols = ["overall_f1", "macro_f1"] + [f"{c}_f1" for c in CLASSES]

# Table I (default regime, P1 entity-level)
p2 = pd.read_csv(R / "phase2_multiseed.csv")
t1 = {}
for arm in ("baseline_ce", "baseline_focal", "securebert_focal"):
    d = p2[p2.arm == arm].sort_values("seed")
    t1[arm] = {c: {"mean": round(float(d[c].mean()), 3), "ci": ci_mean(d[c])}
               for c in ["vuln_f1", "macro_f1", "overall_f1"]}
t1["securebert_focal_minus_baseline_ce"] = {
    c: {"diff": round(float((p2[p2.arm == "securebert_focal"].sort_values("seed")[c].values
                              - p2[p2.arm == "baseline_ce"].sort_values("seed")[c].values).mean()), 3),
        "ci": ci_paired(p2[p2.arm == "securebert_focal"].sort_values("seed")[c].values, p2[p2.arm == "baseline_ce"].sort_values("seed")[c].values)}
    for c in ["vuln_f1", "macro_f1", "overall_f1"]}
res["table1_default_focal"] = t1

# Table IV (default regime, oversampling for both)
os10 = pd.read_csv(R / "oversample_10seeds.csv")
res["table4_default_os"] = block(os10[os10.technique == "bert_oversample"],
                                 os10[os10.technique == "securebert_oversample"], m_cols)

# Table V (tuned regime, oversampling for both)
tc = pd.read_csv(R / "tstem_comparison_runs.csv")
res["table5_tuned_os"] = block(tc[tc.technique == "bert_oversample_e10_lr5e-05"],
                               tc[tc.technique == "securebert_oversample_e10_lr5e-05"], m_cols)

# P1 token-level (official split) and P2 (random split), vs TSTEM constants
pp = pd.read_csv(R / "tstem_proof_per_seed.csv")
tstem = {c: float(pp[f"TSTEM_{c}_f1"].iloc[0]) for c in CLASSES + ["macro"]}
cols = [f"token_{c}_f1" for c in CLASSES] + ["token_macro_f1"]
for name, split, tech in (("p1_token_securebert_os", "official", "securebert_oversample_e10_lr5e-05"),
                          ("p2_securebert_os", "random", "securebert_oversample_e10_lr5e-05_rs"),
                          ("p2_bert_ce", "random", "bert_ce_rs"),
                          ("p1_token_bert_ce", "official", "bert_ce")):
    d = pp[(pp.technique == tech)].sort_values("seed")
    res[name] = {c.replace("token_", "").replace("_f1", ""): {
        "mean": round(float(d[c].mean()), 3), "ci": ci_mean(d[c]),
        "tstem": tstem[c.replace("token_", "").replace("_f1", "")]} for c in cols}
a = pp[pp.technique == "bert_ce_rs"].sort_values("seed")
b = pp[pp.technique == "securebert_oversample_e10_lr5e-05_rs"].sort_values("seed")
res["p2_paired_securebert_os_minus_bert_ce"] = {
    c.replace("token_", "").replace("_f1", ""): {
        "diff": round(float((b[c].values - a[c].values).mean()), 3), "ci": ci_paired(b[c].values, a[c].values)}
    for c in cols}

# ---- sentence-level bootstrap on the saved model ----
try:
    import torch
    from seqeval.metrics import classification_report
    from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments

    from src.preprocessing.cyner_loader import load_cyner_datasets
    from src.preprocessing.tokenization import make_tokenize_fn
    from src.utils.config import load_config

    cfg = load_config()
    labels = list(cfg.get("labels.label_list"))
    mdir = str(cfg.path("paths.securebert_dir"))
    tok = AutoTokenizer.from_pretrained(mdir, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(mdir)
    raw = load_cyner_datasets(cfg)
    test = raw["test"]
    tk = test.map(make_tokenize_fn(tok, cfg), batched=True, remove_columns=test.column_names)
    tr = Trainer(model=model, args=TrainingArguments(output_dir=str(R / "_tmp_boot"), report_to=[],
                                                     per_device_eval_batch_size=32),
                 data_collator=DataCollatorForTokenClassification(tok))
    out = tr.predict(tk)
    pred = out.predictions.argmax(-1)
    gold_seqs, pred_seqs = [], []
    for p, l in zip(pred, out.label_ids):
        keep = l != -100
        gold_seqs.append([labels[i] for i in l[keep]])
        pred_seqs.append([labels[i] for i in p[keep]])

    def scores(idx):
        g = [gold_seqs[i] for i in idx]
        p = [pred_seqs[i] for i in idx]
        rep = classification_report(g, p, output_dict=True, zero_division=0)
        d = {c: rep.get(c, {}).get("f1-score", 0.0) for c in CLASSES}
        d["micro"] = rep["micro avg"]["f1-score"]
        d["macro"] = rep["macro avg"]["f1-score"]
        return d

    json.dump({"gold": gold_seqs, "pred": pred_seqs}, open(R / "_boot_preds.json", "w"))
    full = scores(range(len(gold_seqs)))
    NB = 2000
    draws = []
    n = len(gold_seqs)
    brng = np.random.default_rng(1)
    for _ in range(NB):
        draws.append(scores(brng.integers(0, n, n)))
    sb = {}
    for k in full:
        v = np.array([d[k] for d in draws])
        sb[k] = {"point": round(float(full[k]), 3),
                 "ci": [round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3)], "resamples_with_support": int(np.sum(~np.isnan(v)))}
    vv = np.array([d["Vulnerability"] for d in draws]); res["vuln_boot_dist"] = {"pcts": {q: round(float(np.nanpercentile(vv, q)), 3) for q in (1, 2.5, 5, 10, 25, 50, 75, 90, 97.5)}, "share_zero": float(np.mean(vv == 0))}
    res["sentence_bootstrap_saved_model"] = {"n_sentences": n, "resamples": NB, "scores": sb}
except Exception as e:  # pragma: no cover
    res["sentence_bootstrap_saved_model"] = {"error": repr(e)}

json.dump(res, open(R / "revision_bootstrap.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
