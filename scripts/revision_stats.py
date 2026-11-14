"""Summaries + paired tests for the reviewer-requested extra runs (reads results/rev_*.csv).

Output: results/revision_stats.json and a printed summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path(__file__).resolve().parents[1] / "results"
rng = np.random.default_rng(0)
CL = ["Malware", "Indicator", "Organization", "System", "Vulnerability"]
COLS = ["overall_f1", "macro_f1"] + [f"{c}_f1" for c in CL]


def ci_mean(x, B=10000):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def load(name, tech):
    d = pd.read_csv(R / name)
    d = d[d.technique == tech].sort_values("seed")
    return d


def summ(d, cols=COLS):
    return {c: {"mean": round(float(d[c].mean()), 3), "sd": round(float(d[c].std(ddof=1)), 3),
                "ci": ci_mean(d[c])} for c in cols if c in d}


def paired(a, b, cols=COLS):
    out = {}
    for c in cols:
        x, y = a[c].values, b[c].values
        if len(x) != len(y) or len(x) < 2:
            continue
        t, p = stats.ttest_rel(y, x)
        out[c] = {"A_mean": round(float(x.mean()), 3), "B_mean": round(float(y.mean()), 3),
                  "diff_B_minus_A": round(float((y - x).mean()), 3), "diff_ci": ci_mean(y - x),
                  "t": round(float(t), 2), "p": float(p), "B_wins": int((y > x).sum()), "n": len(x)}
    return out


res = {}
sb_os = load("tstem_comparison_runs.csv", "securebert_oversample_e10_lr5e-05")
bt_os = load("tstem_comparison_runs.csv", "bert_oversample_e10_lr5e-05")
sb_ce_t = load("rev_securebert_ce_tuned.csv", "securebert_ce_e10_lr5e-05") if (R / "rev_securebert_ce_tuned.csv").exists() else None
rb_os = load("rev_roberta_os_tuned.csv", "roberta_oversample_e10_lr5e-05") if (R / "rev_roberta_os_tuned.csv").exists() else None
bt_ce_t = load("rev_bert_ce_tuned.csv", "bert_ce_e10_lr5e-05") if (R / "rev_bert_ce_tuned.csv").exists() else None

if sb_ce_t is not None and len(sb_ce_t):
    res["securebert_ce_tuned"] = {"n": len(sb_ce_t), "summary": summ(sb_ce_t),
                                  "vuln_f1_per_seed": [round(float(v), 3) for v in sb_ce_t.vuln_f1],
                                  "diag": {c: sb_ce_t[c].tolist() for c in sb_ce_t.columns if c.startswith("diag_")}}
    if len(sb_ce_t) == len(sb_os):
        res["securebert_os_vs_ce_tuned"] = paired(sb_ce_t, sb_os)
d = R / "rev_securebert_ce_default_diag.csv"
if d.exists():
    dd = pd.read_csv(d)
    res["securebert_ce_default_diag"] = {c: dd[c].tolist() for c in dd.columns
                                         if c.startswith("diag_") or c in ("seed", "vuln_f1", "macro_f1", "overall_f1")}
if rb_os is not None and len(rb_os):
    res["roberta_os_tuned"] = {"n": len(rb_os), "summary": summ(rb_os)}
    if len(rb_os) == len(sb_os):
        res["securebert_vs_roberta_os_tuned"] = paired(rb_os, sb_os)
        res["roberta_vs_bert_os_tuned"] = paired(bt_os, rb_os)
if bt_ce_t is not None and len(bt_ce_t):
    res["bert_ce_tuned"] = {"n": len(bt_ce_t), "summary": summ(bt_ce_t),
                            "vuln_f1_per_seed": [round(float(v), 3) for v in bt_ce_t.vuln_f1]}
    if len(bt_ce_t) == len(bt_os):
        res["bert_os_vs_ce_tuned"] = paired(bt_ce_t, bt_os)

# tuning budgets (selection metric: mean development macro F1 over seeds 42/123)
for m in ("baseline", "roberta"):
    f = R / f"rev_tuning_{m}.csv"
    if f.exists():
        d = pd.read_csv(f)
        res[f"tuning_{m}"] = {t: {"val_macro_f1": round(float(g.val_macro_f1.mean()), 3), "n": int(len(g))}
                              for t, g in d.groupby("technique")}
json.dump(res, open(R / "revision_stats.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
