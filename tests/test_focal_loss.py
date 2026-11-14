"""Focal Loss correctness tests, incl. equivalence to CE when gamma=0."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from src.ner.focal_loss import FocalLoss  # noqa: E402


def test_focal_equals_cross_entropy_when_gamma_zero():
    torch.manual_seed(0)
    logits = torch.randn(8, 11)
    target = torch.randint(0, 11, (8,))

    focal = FocalLoss(gamma=0.0, alpha=1.0)
    fl = focal(logits, target)
    ce = F.cross_entropy(logits, target)
    assert torch.allclose(fl, ce, atol=1e-6)


def test_focal_ignores_minus_100():
    torch.manual_seed(1)
    logits = torch.randn(6, 11)
    target = torch.tensor([0, -100, 3, -100, 9, 7])

    focal = FocalLoss(gamma=0.0, alpha=1.0)
    fl = focal(logits, target)
    # reference: CE with ignore_index over the same data
    ce = F.cross_entropy(logits, target, ignore_index=-100)
    assert torch.allclose(fl, ce, atol=1e-6)


def test_focal_downweights_easy_examples():
    # one very-confident-correct example + one hard example
    logits = torch.tensor([[10.0, -10.0], [0.1, 0.0]])
    target = torch.tensor([0, 1])

    ce = FocalLoss(gamma=0.0)(logits, target)
    fl = FocalLoss(gamma=2.0)(logits, target)
    # focal loss should be smaller (easy example down-weighted)
    assert fl < ce


def test_focal_all_ignored_returns_zero():
    logits = torch.randn(4, 11, requires_grad=True)
    target = torch.full((4,), -100)
    loss = FocalLoss(gamma=2.0)(logits, target)
    assert float(loss) == 0.0
    loss.backward()  # graph stays connected


def test_focal_handles_3d_logits():
    torch.manual_seed(2)
    logits = torch.randn(2, 5, 11)  # (batch, seq, classes)
    target = torch.randint(0, 11, (2, 5))
    loss = FocalLoss(gamma=2.0)(logits, target)
    assert loss.dim() == 0 and float(loss) >= 0.0
