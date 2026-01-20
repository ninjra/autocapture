"""HTTP client for Memory Service."""

from __future__ import annotations

import httpx

from ..config import MemoryServiceConfig
from ..logging_utils import get_logger
from .schemas import MemoryQueryRequest, MemoryQueryResponse, MemoryIngestRequest, MemoryIngestResponse

_LOG = get_logger("memory.client")


class MemoryServiceClient:
    def __init__(self, config: MemoryServiceConfig) -> None:
        self._config = config
        base_url = (config.base_url or "").strip()
        if not base_url:
            base_url = f"http://{config.bind_host}:{config.port}"
        self._base_url = base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        if self._config.require_api_key and self._config.api_key:
            return {"Authorization": f"Bearer {self._config.api_key}"}
        return {}

    def ingest(self, payload: MemoryIngestRequest) -> MemoryIngestResponse:
        url = f"{self._base_url}/v1/memory/ingest"
        response = httpx.post(
            url,
            json=payload.model_dump(mode="json"),
            headers=self._headers(),
            timeout=self._config.request_timeout_s,
        )
        response.raise_for_status()
        return MemoryIngestResponse.model_validate(response.json())

    def query(self, payload: MemoryQueryRequest) -> MemoryQueryResponse:
        url = f"{self._base_url}/v1/memory/query"
        response = httpx.post(
            url,
            json=payload.model_dump(mode="json"),
            headers=self._headers(),
            timeout=self._config.request_timeout_s,
        )
        response.raise_for_status()
        return MemoryQueryResponse.model_validate(response.json())
