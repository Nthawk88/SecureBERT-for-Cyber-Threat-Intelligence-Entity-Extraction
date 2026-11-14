"""Export the NetworkX correlation graph to Neo4j Community Edition.

Creates ``(:IOC)`` nodes and ``[:CORRELATED]`` relationships carrying ``type``,
``confidence`` and ``weight``. Existing data is cleared before export so the
"campaign picture" in the Neo4j Browser always reflects the latest run.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import networkx as nx


def _iso(value) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


class Neo4jExporter:
    """Thin wrapper around the official ``neo4j`` Python driver."""

    def __init__(self, config=None, uri=None, user=None, password=None,
                 database=None, node_label=None, rel_type=None) -> None:
        cfg = config
        if cfg is None:
            try:
                from src.utils.config import load_config

                cfg = load_config()
            except Exception:  # pragma: no cover
                cfg = None

        def _cfg(key, default):
            return default if cfg is None else cfg.get(key, default)

        self.uri = uri or _cfg("correlation.neo4j.uri", "bolt://localhost:7687")
        self.user = user or _cfg("correlation.neo4j.user", "neo4j")
        self.password = password or _cfg("correlation.neo4j.password", "ctiresearch")
        self.database = database or _cfg("correlation.neo4j.database", "neo4j")
        self.node_label = node_label or _cfg("correlation.neo4j.node_label", "IOC")
        self.rel_type = rel_type or _cfg("correlation.neo4j.rel_type", "CORRELATED")
        self._driver = None

    def connect(self):
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._driver.verify_connectivity()
        return self

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    def clear(self) -> None:
        """Remove all nodes/relationships (idempotent export)."""
        with self._driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def export(self, graph: nx.Graph, clear_first: bool = True) -> dict:
        """Write nodes and edges from ``graph`` into Neo4j.

        Returns a small dict with the number of nodes/relationships written.
        """
        if self._driver is None:
            self.connect()
        if clear_first:
            self.clear()

        node_label = self.node_label
        rel_type = self.rel_type

        with self._driver.session(database=self.database) as session:
            session.run(
                f"CREATE CONSTRAINT ioc_key IF NOT EXISTS "
                f"FOR (n:{node_label}) REQUIRE n.key IS UNIQUE"
            )
            for key, data in graph.nodes(data=True):
                session.run(
                    f"MERGE (n:{node_label} {{key: $key}}) "
                    f"SET n.value=$value, n.type=$type, n.confidence=$confidence, "
                    f"n.first_seen=$first_seen, n.last_seen=$last_seen, n.sources=$sources",
                    key=key,
                    value=data.get("value"),
                    type=data.get("type"),
                    confidence=float(data.get("confidence", 0.0)),
                    first_seen=_iso(data.get("first_seen")),
                    last_seen=_iso(data.get("last_seen")),
                    sources=list(data.get("sources", [])),
                )
            for u, v, data in graph.edges(data=True):
                session.run(
                    f"MATCH (a:{node_label} {{key:$u}}), (b:{node_label} {{key:$v}}) "
                    f"MERGE (a)-[r:{rel_type}]-(b) "
                    f"SET r.type=$type, r.confidence=$confidence, r.weight=$weight",
                    u=u,
                    v=v,
                    type=data.get("type"),
                    confidence=float(data.get("confidence", 0.0)),
                    weight=float(data.get("weight", 0.0)),
                )

        return {
            "nodes_written": graph.number_of_nodes(),
            "relationships_written": graph.number_of_edges(),
        }


def _sanitize_attrs(data: dict) -> None:
    """Coerce attribute values to GraphML-safe scalars (str/int/float/bool)."""
    for k, val in list(data.items()):
        if isinstance(val, datetime):
            data[k] = val.isoformat()
        elif isinstance(val, (list, tuple, set)):
            data[k] = ",".join(map(str, val))
        elif val is None:
            data[k] = ""
        elif not isinstance(val, (str, int, float, bool)):
            data[k] = str(val)


def export_graphml(graph: nx.Graph, path: str) -> str:
    """Fallback export to GraphML (no Neo4j required). Stringifies attributes."""
    g = graph.copy()
    for _, data in g.nodes(data=True):
        _sanitize_attrs(data)
    for _, _, data in g.edges(data=True):
        _sanitize_attrs(data)
    nx.write_graphml(g, path)
    return path
