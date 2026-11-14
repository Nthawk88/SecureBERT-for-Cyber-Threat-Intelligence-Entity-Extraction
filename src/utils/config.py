"""Centralised configuration loading.

All paths and hyperparameters are defined in ``config/config.yaml`` so they are
never hardcoded across the codebase. :class:`Config` provides attribute- and
dict-style access plus convenient absolute-path resolution.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def project_root() -> Path:
    """Return the repository root (the directory that contains ``config/``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "config.yaml").exists():
            return parent
    # Fallback: two levels up from src/utils/
    return here.parents[2]


class Config:
    """Thin wrapper around the parsed YAML config.

    Supports attribute access (``cfg.training.learning_rate``), dict access
    (``cfg["training"]["learning_rate"]``), and ``get`` with a dotted key
    (``cfg.get("training.learning_rate")``).
    """

    def __init__(self, data: Dict[str, Any], root: Optional[Path] = None) -> None:
        self._data = data
        self._root = root or project_root()

    # -- access helpers -------------------------------------------------------
    def __getattr__(self, item: str) -> Any:
        try:
            value = self._data[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(value, self._root) if isinstance(value, dict) else value

    def __getitem__(self, item: str) -> Any:
        value = self._data[item]
        return Config(value, self._root) if isinstance(value, dict) else value

    def __contains__(self, item: str) -> bool:
        return item in self._data

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Fetch a value via a dotted path, e.g. ``"training.learning_rate"``."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    # -- path resolution ------------------------------------------------------
    @property
    def root(self) -> Path:
        return self._root

    def path(self, dotted_key: str) -> Path:
        """Resolve a path defined under ``paths.*`` to an absolute path."""
        rel = self.get(dotted_key)
        if rel is None:
            raise KeyError(f"No path configured for '{dotted_key}'")
        return (self._root / rel).resolve()

    def abspath(self, relative: str) -> Path:
        return (self._root / relative).resolve()


@lru_cache(maxsize=4)
def load_config(config_path: Optional[str] = None) -> Config:
    """Load and cache the project configuration.

    Parameters
    ----------
    config_path:
        Optional explicit path. Defaults to ``<root>/config/config.yaml`` or the
        ``CTI_CONFIG`` environment variable if set.
    """
    root = project_root()
    if config_path is None:
        config_path = os.environ.get("CTI_CONFIG", str(root / "config" / "config.yaml"))
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data, root)
