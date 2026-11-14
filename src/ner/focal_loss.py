"""Focal Loss for token classification (Lin et al., 2017).

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

A modulation factor ``(1 - p_t)^gamma`` down-weights well-classified examples so
the model concentrates capacity on hard / minority-class tokens (here, the
under-represented ``Vulnerability`` entities). With ``gamma = 0`` and
``alpha = 1`` this reduces *exactly* to standard cross-entropy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with ``-100`` ignore-index support.

    Parameters
    ----------
    gamma:
        Focusing parameter (>= 0). Higher values push more focus onto hard
        examples. ``gamma = 0`` recovers cross-entropy.
    alpha:
        Scalar weighting factor applied to all classes. For per-class weighting,
        use :mod:`src.ner.class_weighting` together with ``weight``.
    ignore_index:
        Target value to ignore (subword continuations / padding). Default -100.
    weight:
        Optional per-class weight tensor (e.g. inverse-frequency weights).
    reduction:
        ``"mean"`` (default), ``"sum"`` or ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 1.0,
        ignore_index: int = -100,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.ignore_index = int(ignore_index)
        self.reduction = reduction
        self.register_buffer("weight", weight if weight is not None else None, persistent=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss.

        Parameters
        ----------
        logits:
            ``(N, C)`` or ``(B, T, C)`` unnormalised class scores.
        target:
            ``(N,)`` or ``(B, T)`` integer class ids (``ignore_index`` allowed).
        """
        num_classes = logits.size(-1)
        logits = logits.view(-1, num_classes)
        target = target.view(-1)

        # Drop ignored positions BEFORE computing the loss.
        active = target != self.ignore_index
        if active.sum() == 0:
            return logits.sum() * 0.0  # keeps graph connected, zero loss

        logits = logits[active]
        target = target[active]

        weight = self.weight.to(logits.dtype) if self.weight is not None else None
        ce_loss = F.cross_entropy(logits, target, weight=weight, reduction="none")

        # p_t = exp(-ce) is the probability of the true class.
        pt = torch.exp(-ce_loss)
        focal_term = (1.0 - pt) ** self.gamma
        loss = self.alpha * focal_term * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
