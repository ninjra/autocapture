"""PromptOps ingestion and proposal scaffolding."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable

import httpx

from ..config import PromptOpsConfig
from ..logging_utils import get_logger
from ..resilience import (
    CircuitBreaker,
    RetryPolicy,
    is_retryable_exception,
    is_retryable_http_status,
    retry_sync,
)
from ..storage.database import DatabaseManager
from ..storage.models import PromptOpsRunRecord


@dataclass(frozen=True)
class SourceItem:
    url: str
    fetched_at: str
    status: int
    body: str


class PromptOpsRunner:
    def __init__(self, config: PromptOpsConfig, db: DatabaseManager) -> None:
        self._config = config
        self._db = db
        self._log = get_logger("promptops")
        self._retry_policy = RetryPolicy()
        self._breaker = CircuitBreaker()

    def run_once(self, sources: Iterable[str]) -> PromptOpsRunRecord:
        fetched: list[SourceItem] = []
        for url in sources:
            if not self._breaker.allow():
                self._log.warning("PromptOps breaker open; skipping fetch {}", url)
                fetched.append(
                    SourceItem(
                        url=url,
                        fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                        status=0,
                        body="",
                    )
                )
                continue

            def _fetch() -> httpx.Response:
                response = httpx.get(url, timeout=20.0)
                if is_retryable_http_status(response.status_code):
                    raise httpx.HTTPStatusError(
                        "Retryable status",
                        request=response.request,
                        response=response,
                    )
                return response

            try:
                response = retry_sync(
                    _fetch,
                    policy=self._retry_policy,
                    is_retryable=is_retryable_exception,
                )
                self._breaker.record_success()
                fetched.append(
                    SourceItem(
                        url=url,
                        fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                        status=response.status_code,
                        body=response.text[:5000],
                    )
                )
            except Exception as exc:
                self._breaker.record_failure(exc)
                self._log.warning("PromptOps fetch failed for {}: {}", url, exc)
                fetched.append(
                    SourceItem(
                        url=url,
                        fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                        status=0,
                        body="",
                    )
                )
        run = PromptOpsRunRecord(
            sources_fetched=[item.__dict__ for item in fetched],
            proposals={},
            eval_results={},
            status="completed",
        )
        with self._db.session() as session:
            session.add(run)
        return run
