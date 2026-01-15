from __future__ import annotations

from autocapture.api.server import RetrieveRequest, _resolve_retrieve_paging
from autocapture.config import APIConfig, AppConfig


def test_retrieve_paging_defaults_to_api_config() -> None:
    config = AppConfig(api=APIConfig(default_page_size=33, max_page_size=50))
    request = RetrieveRequest(query="hello")
    offset, limit = _resolve_retrieve_paging(request, config)
    assert limit == 33
    assert offset == 0


def test_retrieve_paging_uses_k_when_set() -> None:
    config = AppConfig(api=APIConfig(default_page_size=33, max_page_size=50))
    request = RetrieveRequest(query="hello", k=7)
    offset, limit = _resolve_retrieve_paging(request, config)
    assert limit == 7
    assert offset == 0


def test_retrieve_paging_clamps_page_size_and_offsets() -> None:
    config = AppConfig(api=APIConfig(default_page_size=33, max_page_size=10))
    request = RetrieveRequest(query="hello", page=2, page_size=99)
    offset, limit = _resolve_retrieve_paging(request, config)
    assert limit == 10
    assert offset == 20
