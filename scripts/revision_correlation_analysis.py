"""Revision analyses for the correlation layer and the crawled corpus (CPU only).

Re-uses the exact Phase-3 filtering code and the cached NER output
(results/phase3_entities.json), so no NER inference is re-run and the headline
graph (1,111 nodes / 39,590 edges / modularity 0.785) is reproduced first.

Writes:
  results/revision_correlation.json   (edge-type ablation, window / filter sensitivity, baselines)
  results/revision_corpus.json        (corpus description, CyNER overlap check)
  data/scraped/article_manifest.csv   (per-article metadata for release)
"""
from __future__ import annotations

import csv
import glob
import importlib.util
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("run_phase3", ROOT / "scripts" / "run_phase3.py")
p3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p3)

from src.correlation.graph_builder import CorrelationGraphBuilder  # noqa: E402
from src.utils.config import load_config  # noqa: E402

cfg = load_config()
RESULTS = cfg.abspath("results")
extracted = json.load(open(RESULTS / "phase3_entities.json", encoding="utf-8"))
articles = p3.load_articles(cfg.path("paths.scraped_raw"))
meta = {a.get("url"): a for a in articles}


def make_docs(min_conf=0.7, min_len=3, hub=0.20):
    raw = []
    for doc in extracted:
        a = meta.get(doc.get("doc_id"), {})
        pub = doc.get("published_at") or a.get("published_at")
        ents = [dict(e) for e in doc["entities"]
                if e.get("confidence", 0) >= min_conf and p3._quality_ok(e, min_len, strict_ioc=True)]
        raw.append({**doc, "entities": ents, "timestamp": pub or doc.get("timestamp"),
                    "temporal_eligible": bool(pub)})
    docs, _, _ = p3._apply_hub_filter(raw, hub)
    return docs


def build(docs, window=24.0, doc_edges=True, temporal=True):
    b = CorrelationGraphBuilder(config=cfg, window_hours=window)
    b.add_documents(docs)
    n_doc = b.graph.number_of_edges()
    if not doc_edges:
        b.graph.remove_edges_from(list(b.graph.edges()))
        n_doc = 0
    n_tmp = b.build_temporal_edges() if temporal else 0
    return b, n_doc, n_tmp


def summarize(b, n_doc, n_tmp):
    g = b.graph
    if g.number_of_edges() == 0:
        return {"nodes": g.number_of_nodes(), "edges": 0, "doc_edges": n_doc, "temporal_edges": n_tmp,
                "coverage_pct": 0.0, "communities_gt1": 0, "modularity": None}
    cl = b.get_clusters()
    return {
        "nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
        "doc_edges": n_doc, "temporal_edges": n_tmp,
        "coverage_pct": b.compute_coverage()["coverage_percentage"],
        "communities_gt1": len(cl), "largest_community": len(cl[0]) if cl else 0,
        "modularity": b.modularity(),
        "components": nx.number_connected_components(g),
    }


out = {}
base_docs = make_docs()
b0, nd, nt = build(base_docs)
out["reproduction"] = summarize(b0, nd, nt)
print("reproduction:", out["reproduction"])

# ---- edge-type ablation (document-only / temporal-only / combined) ----
out["edge_types"] = {
    "document_only": summarize(*build(base_docs, temporal=False)),
    "temporal_only": summarize(*build(base_docs, doc_edges=False, temporal=True)),
    "combined": out["reproduction"],
}
print("edge types:", json.dumps(out["edge_types"], indent=1))

# ---- time-window sensitivity ----
out["window_hours"] = {str(w): summarize(*build(base_docs, window=w)) for w in (6, 12, 24, 48, 72, 168)}
print("windows:", json.dumps(out["window_hours"], indent=1))

# ---- filter sensitivity ----
out["hub_docfreq"] = {str(h): summarize(*build(make_docs(hub=h))) for h in (0.10, 0.20, 0.30, 0.50, 1.0)}
out["min_confidence"] = {str(c): summarize(*build(make_docs(min_conf=c))) for c in (0.5, 0.7, 0.9)}
print("hub:", json.dumps(out["hub_docfreq"], indent=1))
print("conf:", json.dumps(out["min_confidence"], indent=1))

# ---- baselines on the headline graph ----
g = b0.graph
node_docs = defaultdict(list)          # node key -> [doc ids]
doc_size = {}
doc_source = {}
for d in base_docs:
    keys = {f"{e['type']}:{e['value']}" for e in d["entities"]}
    doc_size[d["doc_id"]] = len(keys)
    doc_source[d["doc_id"]] = d["source"]
    for k in keys:
        node_docs[k].append(d["doc_id"])


def partition_from(label_of):
    groups = defaultdict(list)
    for n in g.nodes:
        groups[label_of(n)].append(n)
    return list(groups.values())


def mod(part):
    return round(nx.community.modularity(g, part, weight="weight"), 4)


first_doc = lambda n: node_docs[n][0]
largest_doc = lambda n: max(node_docs[n], key=lambda d: (doc_size[d], -list(doc_size).index(d)))
first_source = lambda n: doc_source[node_docs[n][0]]

louvain = b0._partition()
sizes = [len(c) for c in louvain]
rng = random.Random(0)
nodes = list(g.nodes)
rand_mods = []
for _ in range(20):
    shuf = nodes[:]
    rng.shuffle(shuf)
    it, part = iter(shuf), []
    for s in sizes:
        part.append([next(it) for _ in range(s)])
    rand_mods.append(mod(part))

lab_l = {n: i for i, c in enumerate(louvain) for n in c}
art_l = {n: first_doc(n) for n in nodes}


def nmi(a, b):
    from sklearn.metrics import normalized_mutual_info_score as f
    return round(float(f([a[n] for n in nodes], [b[n] for n in nodes])), 4)


multi_doc_nodes = sum(1 for n in nodes if len(node_docs[n]) > 1)
multi_art_comm = sum(1 for c in louvain if len({d for n in c for d in node_docs[n]}) > 1)
multi_src_comm = sum(1 for c in louvain if len({doc_source[d] for n in c for d in node_docs[n]}) > 1)
cross_doc_edges = 0
for u, v, d in g.edges(data=True):
    if d.get("type") == "document":
        cross_doc_edges += 0
out["baselines"] = {
    "louvain": {"groups": len(louvain), "modularity": mod(louvain)},
    "group_by_first_article": {"groups": len(set(art_l.values())), "modularity": mod(partition_from(first_doc)),
                               "nmi_with_louvain": nmi(lab_l, art_l)},
    "group_by_largest_article": {"groups": len({largest_doc(n) for n in nodes}),
                                 "modularity": mod(partition_from(largest_doc)),
                                 "nmi_with_louvain": nmi(lab_l, {n: largest_doc(n) for n in nodes})},
    "group_by_outlet": {"groups": len({first_source(n) for n in nodes}), "modularity": mod(partition_from(first_source))},
    "connected_components": {"groups": nx.number_connected_components(g),
                             "modularity": mod([list(c) for c in nx.connected_components(g)])},
    "random_partition_same_sizes": {"modularity_mean": round(float(np.mean(rand_mods)), 4),
                                    "modularity_max": round(float(np.max(rand_mods)), 4)},
    "entities_in_more_than_one_article": multi_doc_nodes,
    "louvain_communities_spanning_gt1_article": multi_art_comm,
    "louvain_communities_spanning_gt1_outlet": multi_src_comm,
    "louvain_total_communities": len(louvain),
}
# community composition
comp = []
for c in sorted(louvain, key=len, reverse=True)[:10]:
    docs_in = Counter(d for n in c for d in node_docs[n])
    comp.append({"size": len(c), "articles": len(docs_in), "outlets": len({doc_source[d] for d in docs_in}),
                 "top_article_share": round(max(docs_in.values()) / sum(docs_in.values()), 3)})
out["baselines"]["top10_community_composition"] = comp
print("baselines:", json.dumps(out["baselines"], indent=1))
json.dump(out, open(RESULTS / "revision_correlation.json", "w", encoding="utf-8"), indent=2)

# =================== corpus description ===================
def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) >= 3]


def norm_words(t):
    return re.findall(r"[a-z0-9]+", t.lower())


def shingles(words, n=8):
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


from src.preprocessing.cyner_loader import load_cyner_datasets  # noqa: E402

raw = load_cyner_datasets(cfg)
cy_sh, cy_ents = set(), set()
cy_sent = {k: len(raw[k]) for k in raw.keys()}
for sp in raw.keys():
    for toks, tags in zip(raw[sp]["tokens"], raw[sp]["ner_tags"]):
        cy_sh |= shingles(norm_words(" ".join(toks)))
n_cy_ents = 0

man_rows = []
tot_sent = tot_words = 0
overlaps = []
dated_cnt = 0
doc_by_id = {d["doc_id"]: d for d in extracted}
pub_vs_scrape = []
for a in sorted(articles, key=lambda x: (x.get("source", ""), x.get("url", ""))):
    text = a["content"]
    sents = sentences(text)
    words = norm_words(text)
    sh = shingles(words)
    ov = (len(sh & cy_sh) / len(sh)) if sh else 0.0
    overlaps.append(ov)
    d = doc_by_id.get(a["url"], {})
    n_raw = len(d.get("entities", []))
    kept = next((len(x["entities"]) for x in base_docs if x["doc_id"] == a["url"]), 0)
    pub = a.get("published_at")
    dated_cnt += bool(pub)
    tot_sent += len(sents)
    tot_words += len(words)
    gap = None
    if pub and a.get("scraped_at"):
        from datetime import datetime
        gap = (datetime.fromisoformat(a["scraped_at"]) - datetime.fromisoformat(pub)).total_seconds() / 86400
        pub_vs_scrape.append(gap)
    man_rows.append({"url": a["url"], "source": a.get("source"), "title": a.get("title"),
                     "published_at": pub or "", "scraped_at": a.get("scraped_at"),
                     "date_used_in_temporal_edges": "published_at" if pub else "none (undated; excluded)",
                     "n_sentences": len(sents), "n_words": len(words),
                     "ner_entities_raw": n_raw, "entities_after_filters": kept,
                     "max_8gram_overlap_with_cyner": round(ov, 4)})
man_path = ROOT / "data" / "scraped" / "article_manifest.csv"
with open(man_path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(man_rows[0].keys()))
    w.writeheader()
    w.writerows(man_rows)

pub_dates = sorted(a["published_at"] for a in articles if a.get("published_at"))
midnight = sum(1 for p in pub_dates if p.endswith("T00:00:00+00:00"))
corp = {
    "articles": len(articles), "per_source": dict(Counter(a.get("source") for a in articles)),
    "sentences": tot_sent, "words": tot_words,
    "dated_articles": dated_cnt, "undated_articles": len(articles) - dated_cnt,
    "published_min": pub_dates[0], "published_max": pub_dates[-1],
    "published_date_only_midnight": midnight,
    "scraped_min": min(a["scraped_at"] for a in articles), "scraped_max": max(a["scraped_at"] for a in articles),
    "published_to_crawl_gap_days": {"min": round(min(pub_vs_scrape), 2), "median": round(float(np.median(pub_vs_scrape)), 2),
                                    "max": round(max(pub_vs_scrape), 2)},
    "dated_published_within_24h_of_crawl": sum(1 for g_ in pub_vs_scrape if g_ <= 1.0),
    "cyner_sentences": cy_sent,
    "cyner_8gram_overlap_per_article": {"mean": round(float(np.mean(overlaps)), 4), "max": round(float(np.max(overlaps)), 4),
                                         "articles_with_any_overlap": int(sum(1 for o in overlaps if o > 0)),
                                         "articles_with_overlap_gt_5pct": int(sum(1 for o in overlaps if o > 0.05))},
}
print("corpus:", json.dumps(corp, indent=1))
json.dump(corp, open(RESULTS / "revision_corpus.json", "w", encoding="utf-8"), indent=2)
print("manifest ->", man_path)
