"""Download the public CyNER dataset and convert it to the configured paths.

Source: https://github.com/aiforsec/CyNER (dataset/mitre/{train,valid,test}.txt),
token<TAB>IOB-tag format, 5 entity classes matching this project's LABEL_LIST.

Idempotent: re-downloads only if files are missing (use --force to refresh).
Useful on a fresh machine / Colab where the .conll files are gitignored.

    python scripts/get_cyner.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.labels import LABEL_LIST
from src.preprocessing.cyner_loader import read_conll, class_distribution

_BASE = "https://raw.githubusercontent.com/aiforsec/CyNER/main/dataset/mitre"
_FILES = {"train.txt": "paths.cyner_train", "valid.txt": "paths.cyner_dev", "test.txt": "paths.cyner_test"}


def ensure_cyner(config=None, force: bool = False, verbose: bool = True) -> bool:
    """Ensure the CyNER .conll files exist; download+convert if needed.

    Returns True if files are present afterwards.
    """
    cfg = config or load_config()
    all_present = all(cfg.path(key).exists() for key in _FILES.values())
    if all_present and not force:
        if verbose:
            print("[cyner] files already present; skipping download (use --force to refresh)")
        return True

    for src, key in _FILES.items():
        dst = cfg.path(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"{_BASE}/{src}"
        if verbose:
            print(f"[cyner] downloading {url} -> {dst}")
        req = urllib.request.Request(url, headers={"User-Agent": "CTI-Research/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            dst.write_bytes(resp.read())
    return True


def report(config=None) -> None:
    cfg = config or load_config()
    agg = {}
    for key in _FILES.values():
        toks, tags = read_conll(cfg.path(key), valid_labels=LABEL_LIST)
        dist = class_distribution(tags, entity_level=True)
        for k, v in dist.items():
            agg[k] = agg.get(k, 0) + v
        print(f"  {cfg.path(key).name:12s} {len(toks):5d} sentences  {dist}")
    minority = min(agg, key=agg.get)
    print(f"  TOTAL entities: {agg}")
    print(f"  minority class: {minority} ({agg[minority]})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    cfg = load_config()
    ensure_cyner(cfg, force=args.force)
    print("[cyner] validation:")
    report(cfg)


if __name__ == "__main__":
    main()
