"""BERT-base-uncased vs SecureBERT: every comparison used in the thesis, in one file.

Reads only saved results (no training) and writes ``results/model_comparison.json``
plus a readable report on screen:

  1. same_loss          - baseline_focal vs securebert_focal, 10 seeds
                          (results/phase2_multiseed.csv). Only the model differs.
  2. vs_ce_baseline     - baseline_ce vs securebert_focal, 10 seeds
                          (same file; the test reported in phase4_evaluation.json).
  3. per_class          - per-class P/R/F1 per arm + paired t-tests per class,
                          10 seeds (results/phase2_perclass.csv).
  4. reproducibility    - per_class rerun vs the official phase2_multiseed.csv.
  5. oversampling       - bert_oversample vs securebert_oversample and the effect
                          of oversampling within each model, 3 seeds
                          (results/ablation.csv + results/ablation_bert.csv).
  6. final_model        - per-class scores of models/securebert_ner
                          (results/phase4_evaluation.json, single run).

    python scripts/compare_models.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.evaluation.statistical_tests import paired_ttest

RESULTS = ROOT / "results"
CLASSES = ["Malware", "Indicator", "Organization", "System", "Vulnerability"]
SUMMARY = ["overall_f1", "macro_f1", "vuln_f1"]


def _clean(obj):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def compare(df, key, base_name, cand_name, metric):
    """Paired t-test of ``cand_name`` vs ``base_name`` on ``metric``, matched by seed."""
    a = df[df[key] == base_name].set_index("seed")[metric]
    b = df[df[key] == cand_name].set_index("seed")[metric]
    seeds = sorted(set(a.index) & set(b.index))
    if len(seeds) < 2:
        return None
    base, cand = a.loc[seeds].astype(float), b.loc[seeds].astype(float)
    r = paired_ttest(base.tolist(), cand.tolist(), metric_name=metric)
    r["seeds"] = [int(s) for s in seeds]
    r["candidate_wins"] = int((cand.values > base.values).sum())
    r["ties"] = int((cand.values == base.values).sum())
    return r


def arm_stats(df, key, name, metrics):
    sub = df[df[key] == name]
    return {m: {"mean": float(sub[m].mean()), "std": float(sub[m].std(ddof=1)),
                "n": int(sub[m].notna().sum())} for m in metrics if m in sub}


def main() -> None:
    out: dict = {"sources": {}}
    main_csv = RESULTS / "phase2_multiseed.csv"
    p2 = pd.read_csv(main_csv)
    p2 = p2[p2.arm != "gamma_search"]
    out["sources"]["phase2"] = str(main_csv.relative_to(ROOT))

    out["same_loss"] = {m: compare(p2, "arm", "baseline_focal", "securebert_focal", m)
                        for m in SUMMARY}
    out["vs_ce_baseline"] = {m: compare(p2, "arm", "baseline_ce", "securebert_focal", m)
                             for m in SUMMARY}

    pc_csv = RESULTS / "phase2_perclass.csv"
    if pc_csv.exists():
        pc = pd.read_csv(pc_csv)
        out["sources"]["per_class"] = str(pc_csv.relative_to(ROOT))
        f1_cols = [f"{c}_f1" for c in CLASSES] + ["macro_f1", "overall_f1"]
        prf_cols = [f"{c}_{k}" for c in CLASSES for k in ("precision", "recall", "f1")]
        out["per_class"] = {
            "arms": {arm: arm_stats(pc, "arm", arm, prf_cols + ["macro_f1", "overall_f1"])
                     for arm in sorted(pc.arm.unique())},
            "same_loss_tests": {c: compare(pc, "arm", "baseline_focal", "securebert_focal", c)
                                for c in f1_cols},
            "vs_ce_baseline_tests": {c: compare(pc, "arm", "baseline_ce", "securebert_focal", c)
                                     for c in f1_cols},
        }
        rep = {}
        for arm in sorted(set(pc.arm) & set(p2.arm)):
            a = p2[p2.arm == arm].set_index("seed")
            b = pc[pc.arm == arm].set_index("seed")
            seeds = sorted(set(a.index) & set(b.index))
            rep[arm] = {m: {"official_mean": float(a.loc[seeds, m].mean()),
                            "rerun_mean": float(b.loc[seeds, m].mean()),
                            "max_abs_seed_diff": float((a.loc[seeds, m] - b.loc[seeds, m]).abs().max())}
                        for m in SUMMARY} | {"n_seeds": len(seeds)}
        out["reproducibility"] = rep
    else:
        print(f"[info] {pc_csv.name} not found - run: "
              f"python scripts/run_phase2.py --out phase2_perclass.csv --drop-checkpoints")

    ab_sb, ab_bert = RESULTS / "ablation.csv", RESULTS / "ablation_bert.csv"
    if ab_sb.exists() and ab_bert.exists():
        ab = pd.concat([pd.read_csv(ab_sb), pd.read_csv(ab_bert)], ignore_index=True)
        out["sources"]["oversampling"] = [str(ab_sb.relative_to(ROOT)), str(ab_bert.relative_to(ROOT))]
        arms = ["bert_ce", "bert_oversample", "securebert_ce", "securebert_oversample"]
        out["oversampling"] = {
            "arms": {a: arm_stats(ab, "technique", a,
                                  SUMMARY + ["vuln_precision", "vuln_recall"]) for a in arms},
            "securebert_vs_bert_with_oversampling":
                {m: compare(ab, "technique", "bert_oversample", "securebert_oversample", m) for m in SUMMARY},
            "oversampling_effect_bert":
                {m: compare(ab, "technique", "bert_ce", "bert_oversample", m) for m in SUMMARY},
            "oversampling_effect_securebert":
                {m: compare(ab, "technique", "securebert_ce", "securebert_oversample", m) for m in SUMMARY},
        }

    os_csv = RESULTS / "oversample_10seeds.csv"
    osd = None
    if os_csv.exists():
        osd = pd.read_csv(os_csv)
        out["sources"]["oversampling_10seeds"] = str(os_csv.relative_to(ROOT))
        f1_cols = ["overall_f1", "macro_f1"] + [f"{c}_f1" for c in CLASSES]
        prf_cols = [f"{c}_{k}" for c in CLASSES for k in ("precision", "recall", "f1")]
        out["oversampling_10seeds"] = {
            "arms": {a: arm_stats(osd, "technique", a, prf_cols + ["macro_f1", "overall_f1"])
                     for a in sorted(osd.technique.unique())},
            "securebert_vs_bert": {
                m: compare(osd, "technique", "bert_oversample", "securebert_oversample", m)
                for m in f1_cols + ["Vulnerability_precision", "Vulnerability_recall"]},
        }

    tstem_vs_ours = None
    if osd is not None and pc_csv.exists():
        ce = pc[pc.arm == "baseline_ce"].assign(technique="bert_ce_tstem_setup")
        tstem_vs_ours = pd.concat([ce, osd[osd.technique == "securebert_oversample"]],
                                  ignore_index=True)
        out["tstem_setup_vs_final_approach"] = {
            m: compare(tstem_vs_ours, "technique", "bert_ce_tstem_setup", "securebert_oversample", m)
            for m in ["overall_f1", "macro_f1"] + [f"{c}_f1" for c in CLASSES]}

    tstem = _tstem_table(pc if pc_csv.exists() else None, osd)
    if tstem:
        out["tstem_comparison"] = tstem

    p4 = RESULTS / "phase4_evaluation.json"
    if p4.exists():
        pcls = json.load(open(p4, encoding="utf-8"))["per_class"]
        out["final_model"] = {c: pcls[c] for c in CLASSES + ["macro avg", "micro avg"] if c in pcls}
        out["sources"]["final_model"] = str(p4.relative_to(ROOT))

    path = RESULTS / "model_comparison.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_clean(out), fh, indent=2)

    _print_report(out)
    print(f"\nSaved -> {path}")

    sections = [
        ("1. Without oversampling, same loss (Focal Loss, gamma = 2), 10 seeds",
         "results/phase2_multiseed.csv", p2, "arm", "baseline_focal", "securebert_focal",
         SUMMARY),
    ]
    if pc_csv.exists():
        sections.append(
            ("2. Without oversampling, per class (Focal Loss), 10 seeds",
             "results/phase2_perclass.csv (rerun of the Phase 2 protocol)", pc, "arm",
             "baseline_focal", "securebert_focal", [f"{c}_f1" for c in CLASSES]))
    if osd is not None:
        sections.append(
            ("5. Both with oversampling (cross-entropy + oversampling x3), 10 seeds",
             "results/oversample_10seeds.csv", osd, "technique",
             "bert_oversample", "securebert_oversample",
             ["overall_f1", "macro_f1"] + [f"{c}_f1" for c in CLASSES]
             + ["Vulnerability_precision", "Vulnerability_recall"]))
    if tstem_vs_ours is not None:
        sections.append(
            ("6. TSTEM setup reproduced (BERT, plain cross-entropy) vs final approach "
             "(SecureBERT + oversampling), 10 seeds",
             "results/phase2_perclass.csv (baseline_ce) + results/oversample_10seeds.csv",
             tstem_vs_ours, "technique", "bert_ce_tstem_setup", "securebert_oversample",
             ["overall_f1", "macro_f1"] + [f"{c}_f1" for c in CLASSES]))
    if "oversampling" in out:
        sections.append(
            ("3. Without any mitigation (plain cross-entropy), 3 seeds",
             "results/ablation.csv + results/ablation_bert.csv", ab, "technique",
             "bert_ce", "securebert_ce", SUMMARY))
        sections.append(
            ("4. Both with oversampling (cross-entropy + oversampling x3), 3 seeds",
             "results/ablation.csv + results/ablation_bert.csv", ab, "technique",
             "bert_oversample", "securebert_oversample",
             SUMMARY + ["vuln_precision", "vuln_recall"]))
    sections.sort(key=lambda s: s[0])
    md = RESULTS / "model_comparison_worked.md"
    md.write_text(_tstem_markdown(out.get("tstem_comparison")) + _worked_markdown(sections),
                  encoding="utf-8")
    print(f"Saved -> {md}")


# Per-class F1 of TSTEM's BERT-NER (bert-base-uncased, CyNER) as reported by
# Balasubramanian et al.; their evaluation protocol is not published in enough
# detail to reproduce, so these are reference values, not paired observations.
TSTEM_F1 = {"Malware": 0.90, "Indicator": 0.84, "Organization": 0.84,
            "System": 0.73, "Vulnerability": 0.48}


def _tstem_table(pc, osd):
    """Per-class F1: TSTEM (reported) vs our runs, all on the CyNER test split."""
    cols = {}
    if pc is not None:
        cols["bert_ce (TSTEM setup, ours)"] = pc[pc.arm == "baseline_ce"]
    if osd is not None:
        cols["bert_oversample"] = osd[osd.technique == "bert_oversample"]
        cols["securebert_oversample"] = osd[osd.technique == "securebert_oversample"]
    if not cols:
        return None
    rows = {}
    for c in CLASSES:
        rows[c] = {"tstem_reported": TSTEM_F1[c]}
        for name, sub in cols.items():
            v = sub[f"{c}_f1"].astype(float)
            rows[c][name] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "n": int(len(v)),
                             "minus_tstem": float(v.mean() - TSTEM_F1[c])}
    rows["macro (5 classes)"] = {"tstem_reported": sum(TSTEM_F1.values()) / len(TSTEM_F1)}
    for name, sub in cols.items():
        v = sub["macro_f1"].astype(float)
        rows["macro (5 classes)"][name] = {
            "mean": float(v.mean()), "std": float(v.std(ddof=1)), "n": int(len(v)),
            "minus_tstem": float(v.mean() - rows["macro (5 classes)"]["tstem_reported"])}
    return rows


def _tstem_markdown(t) -> str:
    if not t:
        return ""
    names = [k for k in next(iter(t.values())) if k != "tstem_reported"]
    L = ["# Comparison with TSTEM (per-class F1, CyNER)", "",
         "TSTEM values are as reported by Balasubramanian et al. (bert-base-uncased BERT-NER).",
         "Our values are mean ± SD over 10 seeds on the CyNER test split, entity-level",
         "seqeval F1. `bert_ce` reproduces the TSTEM setup (same model, plain",
         "cross-entropy, no imbalance handling). TSTEM does not publish per-seed scores,",
         "so no t-test against TSTEM is possible; the gap is shown in brackets.", "",
         "| Class | TSTEM (reported) | " + " | ".join(names) + " |",
         "|---|---|" + "---|" * len(names)]
    for cls, r in t.items():
        cells = [f"{r[n]['mean']:.3f} ± {r[n]['std']:.3f} ({r[n]['minus_tstem']:+.3f})" for n in names]
        L.append(f"| {cls} | {r['tstem_reported']:.3f} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n\n---\n\n"


def _worked(df, key, base_name, cand_name, metric):
    """Per-seed table and every intermediate value of the paired t-test."""
    from scipy import stats

    a = df[df[key] == base_name].set_index("seed")[metric].astype(float)
    b = df[df[key] == cand_name].set_index("seed")[metric].astype(float)
    seeds = sorted(set(a.index) & set(b.index))
    x, y = a.loc[seeds].values, b.loc[seeds].values
    d = y - x
    n = len(d)
    mean_d = d.mean()
    sd_d = d.std(ddof=1)
    se = sd_d / math.sqrt(n) if n > 1 else float("nan")
    t = mean_d / se if se and se > 0 else float("nan")
    p = 2 * stats.t.sf(abs(t), n - 1) if math.isfinite(t) else float("nan")
    t_check, p_check = stats.ttest_rel(y, x)
    assert (not math.isfinite(t)) or abs(t - t_check) < 1e-9, "manual t != scipy"
    return {"seeds": seeds, "bert": x, "securebert": y, "diff": d, "n": n,
            "mean_bert": x.mean(), "sd_bert": x.std(ddof=1),
            "mean_sb": y.mean(), "sd_sb": y.std(ddof=1),
            "mean_d": mean_d, "sd_d": sd_d, "se": se, "t": t, "df": n - 1, "p": p,
            "wins": int((d > 0).sum()), "p_scipy": float(p_check)}


def _worked_markdown(sections) -> str:
    L = ["# BERT-base-uncased vs SecureBERT: worked paired t-tests",
         "",
         "Generated by `scripts/compare_models.py` from the saved results files.",
         "",
         "Each seed trains both models with identical data, split and settings, so",
         "the scores are paired by seed. For each metric:",
         "",
         "- d_i = SecureBERT_i - BERT_i (one difference per seed)",
         "- mean_d = sum(d_i) / n",
         "- sd_d = sqrt( sum((d_i - mean_d)^2) / (n - 1) )",
         "- SE = sd_d / sqrt(n)",
         "- t = mean_d / SE, with df = n - 1",
         "- p = two-sided p-value of t under Student's t with df degrees of freedom",
         "  (identical to `scipy.stats.ttest_rel`, checked in the script)",
         "",
         "Significant means p < 0.05.",
         ""]
    for title, source, df, key, base, cand, metrics in sections:
        L += [f"## {title}", "", f"Source: `{source}` (`{base}` vs `{cand}`)", ""]
        summary = ["| Metric | BERT mean ± SD | SecureBERT mean ± SD | mean_d | sd_d | SE | t | df | p | SecureBERT better in | Significant? |",
                   "|---|---|---|---|---|---|---|---|---|---|---|"]
        details = []
        for m in metrics:
            w = _worked(df, key, base, cand, m)
            sig = "yes" if math.isfinite(w["p"]) and w["p"] < 0.05 else "no"
            p_txt = f"{w['p']:.2e}" if math.isfinite(w["p"]) else "n/a (all differences equal)"
            t_txt = f"{w['t']:.3f}" if math.isfinite(w["t"]) else "n/a"
            summary.append(
                f"| {m} | {w['mean_bert']:.4f} ± {w['sd_bert']:.4f} | "
                f"{w['mean_sb']:.4f} ± {w['sd_sb']:.4f} | {w['mean_d']:+.4f} | "
                f"{w['sd_d']:.4f} | {w['se']:.4f} | {t_txt} | {w['df']} | {p_txt} | "
                f"{w['wins']}/{w['n']} | {sig} |")
            details += [f"### {m}", "",
                        "| Seed | BERT | SecureBERT | d = SecureBERT - BERT |",
                        "|---|---|---|---|"]
            for s, xb, ys, di in zip(w["seeds"], w["bert"], w["securebert"], w["diff"]):
                details.append(f"| {s} | {xb:.4f} | {ys:.4f} | {di:+.4f} |")
            details += ["",
                        f"mean_d = {w['diff'].sum():+.4f} / {w['n']} = {w['mean_d']:+.4f}; "
                        f"sd_d = {w['sd_d']:.4f}; SE = {w['sd_d']:.4f} / sqrt({w['n']}) = {w['se']:.4f}; "
                        f"t = {w['mean_d']:+.4f} / {w['se']:.4f} = {t_txt}; df = {w['df']}; p = {p_txt}",
                        ""]
        L += summary + [""] + details
    return "\n".join(L)


def _line(label, r):
    if r is None:
        return f"  {label:<22} (not enough shared seeds)"
    return (f"  {label:<22} BERT {r['baseline_mean']:.3f}±{r['baseline_std']:.3f}  "
            f"SecureBERT {r['candidate_mean']:.3f}±{r['candidate_std']:.3f}  "
            f"Δ {r['mean_improvement']:+.3f}  p={r['p_value']:.2g}  "
            f"wins {r['candidate_wins']}/{r['n_seeds']}")


def _print_report(out):
    print("\n=== 1. Same loss (Focal) - only the model differs, 10 seeds ===")
    for m, r in out["same_loss"].items():
        print(_line(m, r))
    print("\n=== 2. BERT plain cross-entropy vs SecureBERT Focal, 10 seeds ===")
    for m, r in out["vs_ce_baseline"].items():
        print(_line(m, r))
    if "per_class" in out:
        print("\n=== 3. Per-class F1, same loss, 10 seeds (phase2_perclass.csv) ===")
        for c, r in out["per_class"]["same_loss_tests"].items():
            print(_line(c, r))
        print("\n=== 4. Reproducibility: rerun vs official phase2_multiseed.csv ===")
        for arm, d in out["reproducibility"].items():
            print(f"  {arm:<18} " + "  ".join(
                f"{m}: {d[m]['official_mean']:.4f} vs {d[m]['rerun_mean']:.4f} "
                f"(max seed diff {d[m]['max_abs_seed_diff']:.4f})" for m in SUMMARY))
    if "oversampling" in out:
        print("\n=== 5. Both models with oversampling, 3 seeds ===")
        for m, r in out["oversampling"]["securebert_vs_bert_with_oversampling"].items():
            print(_line(m, r))
    if "oversampling_10seeds" in out:
        print("\n=== 6. Both models with oversampling, 10 seeds (oversample_10seeds.csv) ===")
        for m, r in out["oversampling_10seeds"]["securebert_vs_bert"].items():
            print(_line(m, r))
    if "tstem_setup_vs_final_approach" in out:
        print("\n=== 8. TSTEM setup (BERT, CE) vs SecureBERT + oversampling, 10 seeds ===")
        for m, r in out["tstem_setup_vs_final_approach"].items():
            print(_line(m, r))
    if "tstem_comparison" in out:
        t = out["tstem_comparison"]
        names = [k for k in next(iter(t.values())) if k != "tstem_reported"]
        print("\n=== 7. Per-class F1 vs TSTEM (reported) ===")
        print(f"  {'class':<18}{'TSTEM':>7}  " + "  ".join(f"{n:>28}" for n in names))
        for cls, r in t.items():
            print(f"  {cls:<18}{r['tstem_reported']:>7.3f}  " + "  ".join(
                f"{r[n]['mean']:>12.3f} ±{r[n]['std']:.3f} ({r[n]['minus_tstem']:+.3f})" for n in names))


if __name__ == "__main__":
    main()
