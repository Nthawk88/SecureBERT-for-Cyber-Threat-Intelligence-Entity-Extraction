"""Computational overhead of the correlation layer (Phase 4.3).

The central claim for the correlation layer is that it is *lightweight*. This
module measures the additional wall-clock latency it introduces so the claim of
streaming-pipeline integration can be substantiated empirically.
"""

from __future__ import annotations

import statistics
import time
from typing import Callable, Dict, List, Sequence


def time_callable(fn: Callable[[], None], runs: int = 5, warmup: int = 1) -> Dict[str, float]:
    """Time a zero-arg callable over ``runs`` repeats (after ``warmup`` runs)."""
    for _ in range(max(0, warmup)):
        fn()
    samples: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return {
        "runs": runs,
        "mean_s": statistics.mean(samples),
        "median_s": statistics.median(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_s": min(samples),
        "max_s": max(samples),
    }


def measure_correlation_overhead(
    documents: Sequence[Dict],
    config=None,
    runs: int = 5,
) -> Dict[str, object]:
    """Measure latency of building the correlation graph for ``documents``.

    Reports document-edge construction time, temporal-edge construction time, the
    combined total, and a per-document amortised cost.
    """
    from src.correlation.graph_builder import CorrelationGraphBuilder

    n_docs = len(documents)
    total_entities = sum(len(d.get("entities", [])) for d in documents)

    def _ingest_only() -> None:
        builder = CorrelationGraphBuilder(config=config)
        builder.add_documents(documents)

    def _full_pipeline() -> None:
        builder = CorrelationGraphBuilder(config=config)
        builder.add_documents(documents)
        builder.build_temporal_edges()

    ingest = time_callable(_ingest_only, runs=runs)
    full = time_callable(_full_pipeline, runs=runs)
    temporal_mean = max(0.0, full["mean_s"] - ingest["mean_s"])

    return {
        "num_documents": n_docs,
        "num_entities": total_entities,
        "document_edges_mean_s": ingest["mean_s"],
        "temporal_edges_mean_s": temporal_mean,
        "total_mean_s": full["mean_s"],
        "per_document_ms": (full["mean_s"] / n_docs * 1000.0) if n_docs else 0.0,
        "per_entity_ms": (full["mean_s"] / total_entities * 1000.0) if total_entities else 0.0,
        "detail": {"ingest": ingest, "full": full},
    }
