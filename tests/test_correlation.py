"""Tests for the heuristic correlation layer: edges, clustering, coverage."""

import pytest

pytest.importorskip("networkx")

from src.correlation.graph_builder import CorrelationGraphBuilder, node_key


def _builder():
    # explicit params -> no dependency on config file during the unit test
    return CorrelationGraphBuilder(
        config=None,
        doc_confidence=0.85,
        window_hours=24,
        temporal_min=0.3,
        temporal_max=0.7,
    )


def test_document_cooccurrence_links_same_doc_entities():
    b = _builder()
    b.add_document(
        doc_id="doc1",
        source="https://a.example/1",
        timestamp="2025-01-01T10:00:00Z",
        entities=[
            {"value": "CVE-2025-0001", "type": "Vulnerability"},
            {"value": "Emotet", "type": "Malware"},
        ],
    )
    u = node_key("CVE-2025-0001", "Vulnerability")
    v = node_key("Emotet", "Malware")
    assert b.graph.has_edge(u, v)
    edge = b.graph.edges[u, v]
    assert edge["type"] == "document"
    assert edge["confidence"] == 0.85
    assert "doc1" in edge["co_docs"]


def test_repeated_document_cooccurrence_strengthens_weight():
    b = _builder()
    ents = [
        {"value": "1.2.3.4", "type": "Indicator"},
        {"value": "Mirai", "type": "Malware"},
    ]
    b.add_document("doc1", "https://a.example/1", "2025-01-01T10:00:00Z", ents)
    b.add_document("doc2", "https://a.example/2", "2025-01-01T11:00:00Z", ents)
    edge = b.graph.edges[node_key("1.2.3.4", "Indicator"), node_key("Mirai", "Malware")]
    assert edge["weight"] > 0.85
    assert set(edge["co_docs"]) == {"doc1", "doc2"}


def test_temporal_edge_between_different_sources_within_window():
    b = _builder()
    b.add_document("d1", "https://a.example", "2025-01-01T00:00:00Z",
                   [{"value": "evilcorp", "type": "Organization"}])
    b.add_document("d2", "https://b.example", "2025-01-01T06:00:00Z",
                   [{"value": "PlugX", "type": "Malware"}])
    added = b.build_temporal_edges()
    assert added == 1
    u = node_key("evilcorp", "Organization")
    v = node_key("PlugX", "Malware")
    edge = b.graph.edges[u, v]
    assert edge["type"] == "temporal"
    # 6h gap -> 0.7 * (1 - 6/24) = 0.525
    assert edge["confidence"] == pytest.approx(0.525, abs=1e-3)


def test_temporal_edge_skipped_outside_window():
    b = _builder()
    b.add_document("d1", "https://a.example", "2025-01-01T00:00:00Z",
                   [{"value": "x", "type": "System"}])
    b.add_document("d2", "https://b.example", "2025-01-03T00:00:00Z",
                   [{"value": "y", "type": "System"}])
    assert b.build_temporal_edges() == 0


def test_temporal_edge_respects_min_confidence_floor():
    b = _builder()
    b.add_document("d1", "https://a.example", "2025-01-01T00:00:00Z",
                   [{"value": "x", "type": "System"}])
    b.add_document("d2", "https://b.example", "2025-01-01T23:30:00Z",
                   [{"value": "y", "type": "System"}])
    b.build_temporal_edges()
    edge = b.graph.edges[node_key("x", "System"), node_key("y", "System")]
    assert edge["confidence"] >= 0.3


def test_coverage_and_clusters():
    b = _builder()
    # connected pair
    b.add_document("d1", "https://a.example", "2025-01-01T00:00:00Z", [
        {"value": "CVE-2025-1", "type": "Vulnerability"},
        {"value": "BadRAT", "type": "Malware"},
    ])
    # isolated single-entity doc, far in time + same-content guard
    b.add_document("d2", "https://a.example", "2025-06-01T00:00:00Z", [
        {"value": "loner", "type": "System"},
    ])
    cov = b.compute_coverage()
    assert cov["total_iocs"] == 3
    assert cov["correlated_iocs"] == 2
    assert cov["isolated_iocs"] == 1
    assert cov["coverage_percentage"] == pytest.approx(66.67, abs=0.1)

    clusters = b.get_clusters()
    assert len(clusters) == 1
    assert len(clusters[0]) == 2
    summary = b.get_cluster_summary(clusters[0])
    assert summary["size"] == 2
    assert summary["ioc_by_type"].get("Vulnerability") == 1
