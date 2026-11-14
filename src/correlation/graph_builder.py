"""Lightweight, heuristic IoC co-occurrence correlation layer (Phase 3).

This is deliberately rule-based (NOT a GNN / neural relation extractor) so it can
ride along a streaming pipeline with minimal overhead. It turns isolated IoCs
into a relational "campaign picture" via two heuristics:

  1. Document co-occurrence — entities in the same document are linked with a
     high, fixed confidence (default 0.85). Repeated co-occurrence strengthens
     the edge weight and records the contributing documents.
  2. Temporal co-occurrence — entities seen in *different sources* within a time
     window (default 24h) are linked with a confidence that decays with the time
     gap. Pairs already joined by a document edge are skipped.

Nodes carry: ``value``, ``type``, ``confidence``, ``first_seen``, ``last_seen``,
``sources`` (list of URLs).
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import networkx as nx

DateLike = Union[str, datetime]


def _parse_dt(value: DateLike) -> datetime:
    """Parse ISO-8601 strings / datetimes into a timezone-aware UTC datetime."""
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def node_key(value: str, ent_type: str) -> str:
    """Stable node identifier combining entity type and value."""
    return f"{ent_type}:{value}"


class CorrelationGraphBuilder:
    """Build and analyse the IoC correlation graph."""

    def __init__(
        self,
        config=None,
        doc_confidence: Optional[float] = None,
        window_hours: Optional[float] = None,
        temporal_min: Optional[float] = None,
        temporal_max: Optional[float] = None,
    ) -> None:
        cfg = config
        if cfg is None:
            try:
                from src.utils.config import load_config

                cfg = load_config()
            except Exception:  # pragma: no cover - allow standalone use
                cfg = None

        def _cfg(key, default):
            return default if cfg is None else cfg.get(key, default)

        self.doc_confidence = float(
            doc_confidence if doc_confidence is not None
            else _cfg("correlation.document_cooccurrence.confidence", 0.85)
        )
        self.window_hours = float(
            window_hours if window_hours is not None
            else _cfg("correlation.temporal_cooccurrence.window_hours", 24)
        )
        self.temporal_min = float(
            temporal_min if temporal_min is not None
            else _cfg("correlation.temporal_cooccurrence.min_confidence", 0.3)
        )
        self.temporal_max = float(
            temporal_max if temporal_max is not None
            else _cfg("correlation.temporal_cooccurrence.max_confidence", 0.7)
        )
        self.graph = nx.Graph()

    # -- node ingestion -------------------------------------------------------
    def _add_or_update_node(
        self,
        value: str,
        ent_type: str,
        confidence: float,
        ts: datetime,
        source: str,
        temporal_eligible: bool = True,
    ) -> str:
        key = node_key(value, ent_type)
        if self.graph.has_node(key):
            data = self.graph.nodes[key]
            data["first_seen"] = min(data["first_seen"], ts)
            data["last_seen"] = max(data["last_seen"], ts)
            data["confidence"] = max(data["confidence"], float(confidence))
            data["temporal_eligible"] = data.get("temporal_eligible", False) or temporal_eligible
            if source and source not in data["sources"]:
                data["sources"].append(source)
        else:
            self.graph.add_node(
                key,
                value=value,
                type=ent_type,
                confidence=float(confidence),
                first_seen=ts,
                last_seen=ts,
                sources=[source] if source else [],
                temporal_eligible=temporal_eligible,
            )
        return key

    def add_document(
        self,
        doc_id: str,
        source: str,
        timestamp: DateLike,
        entities: Sequence[Dict],
        temporal_eligible: bool = True,
    ) -> List[str]:
        """Ingest one document's entities and add document co-occurrence edges.

        Parameters
        ----------
        doc_id:
            Unique document identifier (e.g. URL or hash).
        source:
            Source URL / outlet for the document.
        timestamp:
            Document timestamp (ISO string or datetime).
        entities:
            Iterable of dicts with keys ``value``, ``type`` and optional
            ``confidence`` (defaults to 1.0). Duplicate entities are de-duplicated
            within the document.
        """
        ts = _parse_dt(timestamp)
        seen: Dict[str, Dict] = {}
        for ent in entities:
            value = ent["value"]
            ent_type = ent["type"]
            conf = float(ent.get("confidence", 1.0))
            key = self._add_or_update_node(
                value, ent_type, conf, ts, source, temporal_eligible=temporal_eligible
            )
            seen[key] = ent

        keys = list(seen.keys())
        for u, v in combinations(keys, 2):
            self._add_document_edge(u, v, doc_id)
        return keys

    def add_documents(self, documents: Iterable[Dict]) -> None:
        """Ingest many documents. Each dict needs ``doc_id``/``id``, ``source``,
        ``timestamp`` and ``entities``."""
        for doc in documents:
            self.add_document(
                doc_id=doc.get("doc_id") or doc.get("id") or doc.get("url"),
                source=doc.get("source", ""),
                timestamp=doc.get("timestamp") or doc.get("scraped_at"),
                entities=doc.get("entities", []),
                temporal_eligible=doc.get("temporal_eligible", True),
            )

    # -- edges ----------------------------------------------------------------
    def _add_document_edge(self, u: str, v: str, doc_id: str) -> None:
        if self.graph.has_edge(u, v):
            data = self.graph.edges[u, v]
            if data.get("type") == "document":
                if doc_id not in data["co_docs"]:
                    data["co_docs"].append(doc_id)
                    data["weight"] += self.doc_confidence
                    data["confidence"] = min(1.0, data["confidence"] + 0.05)
            else:
                # upgrade a temporal edge to a (stronger) document edge
                data["type"] = "document"
                data["confidence"] = self.doc_confidence
                data["weight"] = self.doc_confidence
                data["co_docs"] = [doc_id]
        else:
            self.graph.add_edge(
                u,
                v,
                type="document",
                confidence=self.doc_confidence,
                weight=self.doc_confidence,
                co_docs=[doc_id],
            )

    def temporal_confidence(self, hours_diff: float) -> float:
        """Confidence that decays linearly with the time gap within the window."""
        decayed = self.temporal_max * (1.0 - hours_diff / self.window_hours)
        return max(self.temporal_min, decayed)

    def build_temporal_edges(self) -> int:
        """Link entities seen in *different sources* within the time window.

        Returns the number of temporal edges added. Pairs that already share a
        document edge are skipped (document evidence dominates).
        """
        added = 0
        nodes = list(self.graph.nodes(data=True))
        for (ka, da), (kb, db) in combinations(nodes, 2):
            if self.graph.has_edge(ka, kb):
                continue  # document edge already present
            # skip nodes whose timestamp is unreliable (e.g. only a scrape time,
            # no real publish date) to avoid artificial temporal clustering
            if not da.get("temporal_eligible", True) or not db.get("temporal_eligible", True):
                continue
            sources_a = set(da.get("sources", []))
            sources_b = set(db.get("sources", []))
            # require at least one differing source between the two entities
            if sources_a and sources_b and sources_a == sources_b:
                continue
            hours_diff = abs((da["first_seen"] - db["first_seen"]).total_seconds()) / 3600.0
            if hours_diff > self.window_hours:
                continue
            conf = self.temporal_confidence(hours_diff)
            self.graph.add_edge(
                ka,
                kb,
                type="temporal",
                confidence=round(conf, 4),
                weight=round(conf, 4),
                hours_diff=round(hours_diff, 3),
            )
            added += 1
        return added

    # -- analysis -------------------------------------------------------------
    def _partition(
        self,
        method: str = "louvain",
        seed: int = 42,
        resolution: float = 1.0,
    ) -> List[List[str]]:
        """Full partition of the graph (every node in exactly one group)."""
        if method == "components":
            return [list(c) for c in nx.connected_components(self.graph)]
        if method == "louvain":
            if self.graph.number_of_nodes() == 0:
                return []
            communities = nx.community.louvain_communities(
                self.graph, weight="weight", resolution=resolution, seed=seed
            )
            return [list(c) for c in communities]
        raise ValueError(f"unknown cluster method '{method}' (use 'louvain' or 'components')")

    def get_clusters(
        self,
        include_isolated: bool = False,
        method: str = "louvain",
        seed: int = 42,
        resolution: float = 1.0,
    ) -> List[List[str]]:
        """Campaign clusters, largest first.

        ``method="louvain"`` applies modularity-based community detection, which
        finds densely connected groups *inside* a connected graph. This is the
        default because document co-occurrence percolates: once enough entities
        are extracted, nearly every node becomes reachable from every other one.

        ``method="components"`` returns connected components. On a dense
        co-occurrence graph this degenerates into a single giant component and
        the number of groups then reflects the hub-filter threshold rather than
        any campaign structure; it is kept for comparison only.

        Louvain is stochastic, so ``seed`` is fixed for reproducibility.
        """
        groups = self._partition(method=method, seed=seed, resolution=resolution)
        if not include_isolated:
            groups = [g for g in groups if len(g) > 1]
        groups.sort(key=len, reverse=True)
        return groups

    def modularity(
        self,
        method: str = "louvain",
        seed: int = 42,
        resolution: float = 1.0,
    ) -> Optional[float]:
        """Newman modularity of the partition; higher means better separated.

        Computed over the *complete* partition (singletons included), because
        modularity is only defined for a partition that covers every node.
        Returns ``None`` for an empty or edgeless graph.
        """
        if self.graph.number_of_edges() == 0:
            return None
        partition = self._partition(method=method, seed=seed, resolution=resolution)
        if not partition:
            return None
        return round(
            nx.community.modularity(
                self.graph, partition, weight="weight", resolution=resolution
            ),
            4,
        )

    def get_cluster_summary(self, cluster_nodes: Sequence[str]) -> Dict:
        """Summarise a cluster: per-type IoC breakdown, sources, timespan, edges."""
        sub = self.graph.subgraph(cluster_nodes)
        type_breakdown: Dict[str, int] = {}
        sources: set = set()
        first_seen: Optional[datetime] = None
        last_seen: Optional[datetime] = None
        for _, data in sub.nodes(data=True):
            type_breakdown[data["type"]] = type_breakdown.get(data["type"], 0) + 1
            sources.update(data.get("sources", []))
            fs, ls = data["first_seen"], data["last_seen"]
            first_seen = fs if first_seen is None else min(first_seen, fs)
            last_seen = ls if last_seen is None else max(last_seen, ls)
        return {
            "size": sub.number_of_nodes(),
            "num_edges": sub.number_of_edges(),
            "ioc_by_type": dict(sorted(type_breakdown.items(), key=lambda kv: -kv[1])),
            "sources": sorted(sources),
            "first_seen": first_seen.isoformat() if first_seen else None,
            "last_seen": last_seen.isoformat() if last_seen else None,
        }

    def compute_coverage(self) -> Dict[str, float]:
        """Coverage = non-isolated nodes / total nodes * 100."""
        total = self.graph.number_of_nodes()
        if total == 0:
            return {
                "total_iocs": 0,
                "correlated_iocs": 0,
                "isolated_iocs": 0,
                "coverage_percentage": 0.0,
            }
        isolated = sum(1 for _, d in self.graph.degree() if d == 0)
        correlated = total - isolated
        return {
            "total_iocs": total,
            "correlated_iocs": correlated,
            "isolated_iocs": isolated,
            "coverage_percentage": round(100.0 * correlated / total, 2),
        }
