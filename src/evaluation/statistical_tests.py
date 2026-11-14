"""Statistical significance testing (Phase 4.2).

Paired t-test (``scipy.stats.ttest_rel``) between baseline and SecureBERT scores
collected across multiple random seeds. Proves an improvement is not merely
random seed variance.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def paired_ttest(
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
    alpha: float = 0.05,
    metric_name: str = "Vulnerability_f1",
) -> Dict[str, object]:
    """Run a paired t-test between two sets of per-seed scores.

    Parameters
    ----------
    baseline_scores, candidate_scores:
        Equal-length sequences of scores (one per shared random seed).
    alpha:
        Significance threshold (default 0.05).
    metric_name:
        Label for reporting.

    Returns
    -------
    dict with means, stds, mean improvement, t-statistic, p-value, df and a
    ``significant`` boolean (``p < alpha``).
    """
    from scipy import stats

    base = np.asarray(baseline_scores, dtype=float)
    cand = np.asarray(candidate_scores, dtype=float)
    if base.shape != cand.shape:
        raise ValueError("baseline and candidate score arrays must be the same length")
    if base.size < 2:
        raise ValueError("need at least 2 paired observations (seeds) for a t-test")

    t_stat, p_value = stats.ttest_rel(cand, base)
    mean_improvement = float(np.mean(cand - base))

    return {
        "metric": metric_name,
        "n_seeds": int(base.size),
        "baseline_mean": float(np.mean(base)),
        "baseline_std": float(np.std(base, ddof=1)),
        "candidate_mean": float(np.mean(cand)),
        "candidate_std": float(np.std(cand, ddof=1)),
        "mean_improvement": mean_improvement,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "df": int(base.size - 1),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
        "direction": "candidate_better" if mean_improvement > 0 else "baseline_better",
    }


def format_report(result: Dict[str, object]) -> str:
    """Pretty multi-line summary for logs / notebooks."""
    return (
        f"Paired t-test on {result['metric']} (n={result['n_seeds']} seeds)\n"
        f"  baseline : {result['baseline_mean']:.4f} +/- {result['baseline_std']:.4f}\n"
        f"  candidate: {result['candidate_mean']:.4f} +/- {result['candidate_std']:.4f}\n"
        f"  mean improvement: {result['mean_improvement']:+.4f}\n"
        f"  t = {result['t_statistic']:.4f}, p = {result['p_value']:.4g}, df = {result['df']}\n"
        f"  significant (p < {result['alpha']}): {result['significant']} "
        f"({result['direction']})"
    )
