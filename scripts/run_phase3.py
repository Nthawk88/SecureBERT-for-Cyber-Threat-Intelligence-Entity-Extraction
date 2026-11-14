"""Turnkey Phase 3 case study: scraping -> NER -> correlation -> outputs.

Runs the end-to-end proof-of-concept on the scraped clear-web corpus (Stream B):
  1. Load + clean scraped articles.
  2. Extract entities with the fine-tuned SecureBERT NER model.
  3. Build the heuristic correlation graph (document + temporal co-occurrence).
  4. Compute coverage, campaign clusters, and correlation-layer overhead.
  5. Export the graph (Neo4j if reachable, else GraphML) + before/after figure.

Stream B is used ONLY for this qualitative case study, never for model metrics.

    python scripts/run_phase3.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.preprocessing.text_cleaning import clean_text
from src.ner.inference import EntityExtractor
from src.correlation.graph_builder import CorrelationGraphBuilder
from src.correlation.neo4j_export import Neo4jExporter, export_graphml
from src.evaluation.overhead import measure_correlation_overhead


import re

_STOPWORDS = {"the", "and", "for", "with", "that", "this", "from", "are", "was",
              "have", "has", "not", "but", "you", "your", "all", "any", "use"}

# Canonical atomic-IoC formats. NER on out-of-domain clear-web prose emits many
# fragment "Indicators" (e.g. "170", "ece"); validating against these patterns is
# standard CTI post-processing and removes that noise.
_IOC_PATTERNS = [
    re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$"),                       # IPv4
    re.compile(r"^https?://", re.I),                                # URL
    re.compile(r"^[a-fA-F0-9]{32}$|^[a-fA-F0-9]{40}$|^[a-fA-F0-9]{64}$"),  # md5/sha1/sha256
    re.compile(r"^CVE-\d{4}-\d{4,}$", re.I),                        # CVE
    re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),                      # email
    re.compile(r"^(?=.{4,253}$)(?:[a-zA-Z0-9-]{1,63}\.)+[a-zA-Z]{2,}$"),  # domain/host
]


def _is_valid_ioc(value: str) -> bool:
    return any(p.search(value) for p in _IOC_PATTERNS)


def _quality_ok(ent: dict, min_len: int, strict_ioc: bool = True) -> bool:
    """Reject NER noise. ``Indicator`` values must match a canonical IoC format."""
    v = (ent.get("value") or "").replace("##", "").strip()
    if len(v) < min_len or not any(ch.isalnum() for ch in v) or v.lower() in _STOPWORDS:
        return False
    if strict_ioc and ent.get("type") == "Indicator" and not _is_valid_ioc(v):
        return False
    ent["value"] = v
    return True


def _apply_hub_filter(raw_docs, hub_docfreq: float):
    """Drop entities present in > hub_docfreq fraction of documents.

    Returns (filtered_docs, num_entities_kept, num_hub_keys_dropped).
    """
    from collections import Counter

    n_docs = max(1, len(raw_docs))
    docfreq: Counter = Counter()
    for doc in raw_docs:
        for key in {(e["type"], e["value"]) for e in doc["entities"]}:
            docfreq[key] += 1

    threshold = hub_docfreq * n_docs
    hub_keys = {k for k, c in docfreq.items() if c > threshold}

    filtered = []
    kept = 0
    for doc in raw_docs:
        ents = [e for e in doc["entities"] if (e["type"], e["value"]) not in hub_keys]
        kept += len(ents)
        filtered.append({**doc, "entities": ents})
    return filtered, kept, len(hub_keys)


def load_articles(raw_dir: Path):
    articles = []
    for f in glob.glob(os.path.join(str(raw_dir), "**", "*.json"), recursive=True):
        with open(f, "r", encoding="utf-8") as fh:
            a = json.load(fh)
        a["content"] = clean_text(a.get("content", ""))
        if a["content"]:
            articles.append(a)
    return articles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.7,
                        help="drop entities below this NER confidence")
    parser.add_argument("--min-token-len", type=int, default=3,
                        help="drop entity values shorter than this")
    parser.add_argument("--hub-docfreq", type=float, default=0.20,
                        help="drop ubiquitous 'hub' entities present in > this "
                             "fraction of documents (generic, non-discriminative). "
                             "Set to 1.0 to disable hub filtering (naive heuristic).")
    parser.add_argument("--refresh", action="store_true",
                        help="re-run NER even if a cached entity file exists")
    parser.add_argument("--no-temporal", action="store_true",
                        help="use document co-occurrence only (skip temporal edges)")
    parser.add_argument("--loose-ioc", action="store_true",
                        help="do NOT validate Indicator values against IoC regex "
                             "(keeps NER noise; shows the naive result)")
    parser.add_argument("--cluster-method", choices=["louvain", "components"],
                        default="louvain",
                        help="how campaign clusters are derived. 'louvain' = "
                             "modularity-based community detection (default); "
                             "'components' = connected components, which "
                             "degenerate to one giant component once document "
                             "co-occurrence percolates.")
    parser.add_argument("--resolution", type=float, default=1.0,
                        help="Louvain resolution; higher yields more, smaller "
                             "communities")
    args = parser.parse_args()

    cfg = load_config()
    results_dir = cfg.abspath("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    corr_dir = cfg.path("paths.correlation_dir")
    corr_dir.mkdir(parents=True, exist_ok=True)

    # 1) load corpus
    articles = load_articles(cfg.path("paths.scraped_raw"))
    if args.max_articles:
        articles = articles[: args.max_articles]
    print(f"[phase3] loaded {len(articles)} cleaned articles")

    # 2) NER extraction (cached: re-run only with --refresh)
    cache_path = results_dir / "phase3_entities.json"
    if cache_path.exists() and not args.refresh:
        with open(cache_path, "r", encoding="utf-8") as fh:
            extracted = json.load(fh)
        print(f"[phase3] loaded cached entities from {cache_path} "
              f"(use --refresh to re-run NER)")
    else:
        model_dir = str(cfg.path("paths.securebert_dir"))
        print(f"[phase3] loading NER model from {model_dir}")
        extractor = EntityExtractor(model_path=model_dir).load()
        extracted = []
        for i, a in enumerate(articles, 1):
            extracted.append({
                "doc_id": a.get("url"),
                "source": a.get("source", ""),
                "published_at": a.get("published_at"),
                "timestamp": a.get("published_at") or a.get("scraped_at"),
                "entities": extractor.extract_unique(a["content"]),
            })
            if i % 10 == 0:
                print(f"  ...{i}/{len(articles)} articles processed")
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(extracted, fh, ensure_ascii=False)
        print(f"[phase3] cached raw entities -> {cache_path}")

    # apply confidence + quality filters; mark temporal eligibility by real date
    articles_meta = {a.get("url"): a.get("published_at") for a in articles}
    raw_docs = []
    n_undated = 0
    for doc in extracted:
        pub = doc.get("published_at") or articles_meta.get(doc.get("doc_id"))
        eligible = bool(pub)
        n_undated += 0 if eligible else 1
        ents = [e for e in doc["entities"]
                if e.get("confidence", 0) >= args.min_confidence
                and _quality_ok(e, args.min_token_len, strict_ioc=not args.loose_ioc)]
        raw_docs.append({**doc, "entities": ents,
                         "timestamp": pub or doc.get("timestamp"),
                         "temporal_eligible": eligible})
    print(f"[phase3] documents without a reliable publish date "
          f"(excluded from temporal edges): {n_undated}/{len(raw_docs)}")

    # Hub filtering: ubiquitous entities (in many documents) are generic and
    # bridge unrelated reports into one giant component. Dropping them is a
    # documented, lightweight heuristic that surfaces distinct campaigns.
    documents, n_kept, n_hubs = _apply_hub_filter(raw_docs, args.hub_docfreq)
    total_ents = sum(len(d["entities"]) for d in documents)
    print(f"[phase3] entities kept={total_ents} (conf>={args.min_confidence}, "
          f"len>={args.min_token_len}); hub entities dropped={n_hubs} "
          f"(docfreq>{args.hub_docfreq:.0%})")

    # 3) build correlation graph
    builder = CorrelationGraphBuilder(config=cfg)
    builder.add_documents(documents)
    before_edges = builder.graph.number_of_edges()
    n_temporal = 0 if args.no_temporal else builder.build_temporal_edges()
    g = builder.graph
    print(f"[phase3] graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges "
          f"(document={before_edges}, temporal={n_temporal})")

    # 4) coverage + clusters + overhead
    coverage = builder.compute_coverage()
    cluster_kwargs = dict(method=args.cluster_method, resolution=args.resolution)
    clusters = builder.get_clusters(**cluster_kwargs)
    modularity = builder.modularity(**cluster_kwargs)
    summaries = [builder.get_cluster_summary(c) for c in clusters[:5]]

    # report the alternative partition too, so the choice of method is auditable
    other_method = "components" if args.cluster_method == "louvain" else "louvain"
    other_clusters = builder.get_clusters(method=other_method, resolution=args.resolution)
    other_modularity = builder.modularity(method=other_method, resolution=args.resolution)

    # dump actual entity members of the top clusters for the qualitative narrative
    campaigns = []
    for rank, c in enumerate(clusters[:5], 1):
        members = {}
        for key in c:
            data = g.nodes[key]
            members.setdefault(data["type"], []).append(data["value"])
        summ = builder.get_cluster_summary(c)
        campaigns.append({
            "rank": rank, "size": summ["size"], "sources": summ["sources"],
            "first_seen": summ["first_seen"], "last_seen": summ["last_seen"],
            "members": {t: sorted(set(v)) for t, v in members.items()},
        })
    with open(results_dir / "phase3_campaigns.json", "w", encoding="utf-8") as fh:
        json.dump(campaigns, fh, ensure_ascii=False, indent=2)
    overhead = measure_correlation_overhead(documents, config=cfg,
                                            runs=cfg.get("evaluation.overhead_runs", 5))
    print(f"[phase3] coverage: {coverage}")
    print(f"[phase3] [{args.cluster_method}] {len(clusters)} multi-node campaign "
          f"clusters; top sizes: {[len(c) for c in clusters[:5]]}; "
          f"modularity={modularity}")
    print(f"[phase3] [{other_method}] for comparison: {len(other_clusters)} clusters; "
          f"top sizes: {[len(c) for c in other_clusters[:5]]}; "
          f"modularity={other_modularity}")

    # 5) exports
    graphml_path = str(corr_dir / "campaign_graph.graphml")
    export_graphml(g, graphml_path)
    neo4j_status = "graphml-only"
    try:
        with Neo4jExporter(config=cfg) as exporter:
            exporter.export(g, clear_first=True)
        neo4j_status = "exported-to-neo4j"
    except Exception as e:  # Neo4j not running -> GraphML fallback already written
        neo4j_status = f"neo4j-unavailable ({type(e).__name__})"
    print(f"[phase3] graph exported: GraphML={graphml_path}; neo4j={neo4j_status}")

    _save_before_after_figure(builder, documents, corr_dir)

    out = {
        "num_articles": len(articles),
        "num_entities": total_ents,
        "min_confidence": args.min_confidence,
        "graph": {
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "document_edges": before_edges,
            "temporal_edges": n_temporal,
        },
        "coverage": coverage,
        "cluster_method": args.cluster_method,
        "resolution": args.resolution,
        "num_clusters": len(clusters),
        "modularity": modularity,
        "comparison_method": {
            "method": other_method,
            "num_clusters": len(other_clusters),
            "top_sizes": [len(c) for c in other_clusters[:5]],
            "modularity": other_modularity,
        },
        "top_clusters": summaries,
        "overhead": {k: v for k, v in overhead.items() if k != "detail"},
        "neo4j": neo4j_status,
    }
    out_path = results_dir / "phase3_casestudy.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[phase3] summary written to {out_path}")


def _save_before_after_figure(builder, documents, corr_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx

        after = builder.graph
        before = after.copy()
        before.remove_edges_from(list(before.edges()))
        type_color = {"Vulnerability": "#c0392b", "Malware": "#8e44ad",
                      "Indicator": "#2980b9", "Organization": "#27ae60", "System": "#f39c12"}

        def colors(graph):
            return [type_color.get(d.get("type"), "#7f8c8d") for _, d in graph.nodes(data=True)]

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        pos_b = nx.spring_layout(before, seed=42)
        nx.draw(before, pos_b, ax=axes[0], node_size=40, node_color=colors(before), with_labels=False)
        axes[0].set_title(f"BEFORE — {before.number_of_nodes()} isolated IoCs")
        pos_a = nx.spring_layout(after, seed=42, k=0.3)
        nx.draw(after, pos_a, ax=axes[1], node_size=40, node_color=colors(after),
                with_labels=False, edge_color="#cccccc", width=0.5)
        axes[1].set_title(f"AFTER — {after.number_of_edges()} correlations (campaign picture)")
        fig.tight_layout()
        path = corr_dir / "before_after.png"
        fig.savefig(path, dpi=140)
        print(f"[phase3] before/after figure -> {path}")
    except Exception as e:  # pragma: no cover
        print(f"[phase3] could not render figure: {e}")


if __name__ == "__main__":
    main()
