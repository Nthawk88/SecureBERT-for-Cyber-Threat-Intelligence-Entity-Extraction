"""Tests for statistical testing and class weighting."""

import pytest

from src.evaluation.statistical_tests import paired_ttest


def test_paired_ttest_detects_consistent_improvement():
    baseline = [0.48, 0.47, 0.49]
    candidate = [0.61, 0.60, 0.62]
    res = paired_ttest(baseline, candidate, alpha=0.05, metric_name="Vulnerability_f1")
    assert res["mean_improvement"] > 0
    assert res["significant"] is True
    assert res["direction"] == "candidate_better"
    assert res["n_seeds"] == 3


def test_paired_ttest_non_significant_for_noise():
    baseline = [0.50, 0.55, 0.45]
    candidate = [0.51, 0.44, 0.56]
    res = paired_ttest(baseline, candidate)
    assert res["significant"] is False


def test_paired_ttest_requires_equal_length():
    with pytest.raises(ValueError):
        paired_ttest([0.1, 0.2], [0.3])


def test_inverse_frequency_weights_favour_rare_classes():
    torch = pytest.importorskip("torch")
    from src.ner.class_weighting import inverse_frequency_weights

    # label 0 (O) very frequent, label 9 (B-Vulnerability) rare
    seqs = [[0, 0, 0, 0, 0, 9], [0, 0, 0, 0]]
    weights = inverse_frequency_weights(seqs, num_labels=11)
    assert weights.shape[0] == 11
    # rare class should get a larger weight than the frequent one
    assert weights[9] > weights[0]
