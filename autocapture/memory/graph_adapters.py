"""Graph retrieval adapters (GraphRAG/HyperGraphRAG/Hyper-RAG)."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import httpx

from ..config import GraphAdapterConfig, GraphAdaptersConfig
from ..logging_utils import get_logger
from ..resilience import RetryPolicy, retry_sync, is_retryable_exception


@dataclass(frozen=True)
class GraphHit:
    event_id: str
    score: float
    snippet: str | None
    source: str


class GraphAdapterClient:
    def __init__(self, name: str, config: GraphAdapterConfig) -> None:
        self._name = name
        self._config = config
        self._log = get_logger(f"graph.{name}")
        self._retry_policy = RetryPolicy(max_retries=2)

    @property
    def enabled(self) -> bool:
        if os.environ.get("AUTOCAPTURE_TEST_MODE") or os.environ.get("PYTEST_CURRENT_TEST"):
            return False
        return bool(self._config.enabled and self._config.base_url)

    def query(
        self,
        query: str,
        *,
        limit: int,
        time_range: tuple[str, str] | None,
        filters: dict | None,
    ) -> list[GraphHit]:
        if not self.enabled:
            return []
        payload = {
            "query": query,
            "limit": min(int(limit), int(self._config.max_results)),
            "time_range": (
                {
                    "start": time_range[0],
                    "end": time_range[1],
                }
                if time_range
                else None
            ),
            "filters": filters or {},
        }
        url = f"{self._config.base_url.rstrip('/')}/{self._name}/query"

        def _request() -> dict[str, Any]:
            with httpx.Client(timeout=self._config.timeout_s) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json()

        try:
            data = retry_sync(
                _request, policy=self._retry_policy, is_retryable=is_retryable_exception
            )
        except Exception as exc:
            self._log.warning("Graph adapter {} query failed: {}", self._name, exc)
            return []
        hits = data.get("hits") if isinstance(data, dict) else None
        if not isinstance(hits, list):
            return []
        results: list[GraphHit] = []
        for item in hits:
            if not isinstance(item, dict):
                continue
            event_id = str(item.get("event_id") or "")
            if not event_id:
                continue
            results.append(
                GraphHit(
                    event_id=event_id,
                    score=float(item.get("score") or 0.0),
                    snippet=item.get("snippet"),
                    source=self._name,
                )
            )
        return results


class GraphAdapterGroup:
    def __init__(self, config: GraphAdaptersConfig) -> None:
        self._adapters = {
            "graphrag": GraphAdapterClient("graphrag", config.graphrag),
            "hypergraphrag": GraphAdapterClient("hypergraphrag", config.hypergraphrag),
            "hyperrag": GraphAdapterClient("hyperrag", config.hyperrag),
        }

    def enabled(self) -> bool:
        return any(adapter.enabled for adapter in self._adapters.values())

    def query(
        self,
        query: str,
        *,
        limit: int,
        time_range: tuple[str, str] | None,
        filters: dict | None,
    ) -> list[GraphHit]:
        hits: list[GraphHit] = []
        for adapter in self._adapters.values():
            hits.extend(adapter.query(query, limit=limit, time_range=time_range, filters=filters))
        return hits


__all__ = ["GraphAdapterGroup", "GraphHit"]
