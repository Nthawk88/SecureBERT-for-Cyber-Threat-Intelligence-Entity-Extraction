"""Export split manifests (release artifact).

One row per CyNER sentence (pool order = official train, validation, test):
  - official_split / official_index : the official CyNER assignment (protocols P1-ent, P1-tok)
  - p2_seed_<s>                     : split assigned by the TSTEM-style random re-split for seed <s> (protocol P2)

Output: results/split_manifest.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import concatenate_datasets  # noqa: E402

from src.preprocessing.cyner_loader import load_cyner_datasets  # noqa: E402
from src.utils.config import load_config  # noqa: E402

cfg = load_config()
raw = load_cyner_datasets(cfg)
seeds = list(cfg.get("project.seeds"))
names = ("train", "validation", "test")
sizes = {k: len(raw[k]) for k in names}
pool = concatenate_datasets([raw[k] for k in names])
n = len(pool)
rows = pd.DataFrame({
    "pool_index": range(n),
    "official_split": [k for k in names for _ in range(sizes[k])],
    "official_index": [i for k in names for i in range(sizes[k])],
})
n_te, n_va = sizes["test"], sizes["validation"]
for s in seeds:
    perm = pool.add_column("pool_index", list(range(n))).shuffle(seed=s)["pool_index"]
    assign = {}
    for pos, idx in enumerate(perm):
        assign[idx] = "test" if pos < n_te else ("validation" if pos < n_te + n_va else "train")
    rows[f"p2_seed_{s}"] = [assign[i] for i in range(n)]
out = ROOT / "results" / "split_manifest.csv"
rows.to_csv(out, index=False)
print("wrote", out, rows.shape, "seeds:", seeds)
