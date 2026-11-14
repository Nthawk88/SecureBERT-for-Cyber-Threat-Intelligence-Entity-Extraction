"""Reproducible seeding across ``random``, ``numpy``, ``torch`` and HF."""

from __future__ import annotations

import os
import random


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Seed all relevant RNGs and return the seed (for logging).

    Sets ``random``, ``numpy``, ``torch`` (CPU + CUDA), and the ``PYTHONHASHSEED``
    environment variable. When ``deterministic`` is True, cuDNN is configured for
    deterministic behaviour (slower but reproducible).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass

    try:
        import transformers

        transformers.set_seed(seed)
    except ImportError:  # pragma: no cover
        pass

    return seed
