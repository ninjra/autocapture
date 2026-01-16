from __future__ import annotations

import datetime as dt
from pathlib import Path

import httpx

from autocapture.config import AppConfig
from autocapture.research import scout
from autocapture.research.scout import run_scout


_ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2509.26507</id>
    <title>Baby Dragon Hatchling Architecture</title>
    <summary>Diffusion transformer for screen understanding.</summary>
    <published>2025-09-30T00:00:00Z</published>
    <updated>2025-10-01T00:00:00Z</updated>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2501.12345</id>
    <title>Screen Understanding with Vision-Language Models</title>
    <summary>Vision-language retrieval and reranking for screen data.</summary>
    <published>2025-01-15T00:00:00Z</published>
    <updated>2025-01-16T00:00:00Z</updated>
  </entry>
</feed>
"""


def _mock_transport(request: httpx.Request) -> httpx.Response:
    if request.url.host == "huggingface.co":
        tag = request.url.params.get("filter")
        if tag == "vision-language":
            return httpx.Response(
                200,
                json=[
                    {
                        "modelId": "hustvl/DiffusionVL-Qwen2.5VL-7B",
                        "downloads": 5000,
                        "likes": 120,
                        "tags": ["vision-language"],
                        "lastModified": "2025-01-01T00:00:00.000Z",
                    }
                ],
            )
        if tag == "ocr":
            return httpx.Response(
                200,
                json=[
                    {
                        "modelId": "deepseek-ai/DeepSeek-OCR",
                        "downloads": 4000,
                        "likes": 200,
                        "tags": ["ocr"],
                        "lastModified": "2025-01-02T00:00:00.000Z",
                    }
                ],
            )
        return httpx.Response(200, json=[])
    if request.url.host == "export.arxiv.org":
        return httpx.Response(200, text=_ARXIV_FEED)
    return httpx.Response(404)


def test_scout_report_uses_cache_offline(tmp_path: Path) -> None:
    config = AppConfig()
    config.capture.data_dir = tmp_path
    config.offline = False
    config.privacy.cloud_enabled = True
    now = dt.datetime(2025, 10, 2, tzinfo=dt.timezone.utc)

    with httpx.Client(transport=httpx.MockTransport(_mock_transport)) as client:
        report = run_scout(config, http_client=client, now=now)
        assert report["sources"]["huggingface"]["status"] == "ok"
        assert report["sources"]["arxiv"]["status"] == "ok"
        assert report["ranked_items"]

    config.offline = True
    config.privacy.cloud_enabled = False
    offline_report = run_scout(config, now=now)
    assert offline_report["sources"]["huggingface"]["status"] == "cached"
    assert offline_report["sources"]["arxiv"]["status"] == "cached"
    assert offline_report["ranked_items"]


def test_scout_report_offline_without_cache(tmp_path: Path) -> None:
    config = AppConfig()
    config.capture.data_dir = tmp_path
    config.offline = True
    config.privacy.cloud_enabled = False
    now = dt.datetime(2025, 10, 2, tzinfo=dt.timezone.utc)

    report = run_scout(config, now=now)
    assert report["sources"]["huggingface"]["status"] == "offline"
    assert report["sources"]["arxiv"]["status"] == "offline"
    assert report["ranked_items"] == []


def test_arxiv_query_expands_phrase_and_tokens() -> None:
    query = scout._build_arxiv_query(["prompt repetition"])

    assert 'all:"prompt repetition"' in query
    assert '(all:"prompt" AND all:"repetition")' in query
    assert 'all:"and"' not in query


def test_arxiv_query_hyphen_variants_without_token_clause() -> None:
    query = scout._build_arxiv_query(["vision-language"])

    assert 'all:"vision-language"' in query
    assert 'all:"vision language"' in query
    assert '(all:"vision" AND all:"language")' not in query


def test_match_keyword_multiword_tokens(monkeypatch) -> None:
    monkeypatch.setattr(scout, "ARXIV_KEYWORDS", ["prompt repetition"])

    matched = scout._match_keyword(
        title="Prompting strategies",
        summary="We study repetition effects on outputs.",
    )

    assert matched == "prompt repetition"


def test_arxiv_request_includes_expanded_query(tmp_path: Path) -> None:
    def _transport(request: httpx.Request) -> httpx.Response:
        if request.url.host == "export.arxiv.org":
            query = request.url.params.get("search_query", "")
            assert 'all:"screen understanding"' in query
            assert '(all:"screen" AND all:"understanding")' in query
            return httpx.Response(200, text=_ARXIV_FEED)
        if request.url.host == "huggingface.co":
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    config = AppConfig()
    config.capture.data_dir = tmp_path
    config.offline = False
    config.privacy.cloud_enabled = True
    now = dt.datetime(2025, 10, 2, tzinfo=dt.timezone.utc)

    with httpx.Client(transport=httpx.MockTransport(_transport)) as client:
        report = run_scout(config, http_client=client, now=now)

    assert report["sources"]["arxiv"]["status"] == "ok"


def test_scout_report_includes_watchlist(tmp_path: Path) -> None:
    config = AppConfig()
    config.capture.data_dir = tmp_path
    config.offline = True
    config.privacy.cloud_enabled = False
    now = dt.datetime(2025, 10, 2, tzinfo=dt.timezone.utc)

    report = run_scout(config, now=now)
    urls = {item.get("url") for item in report.get("watchlist", [])}
    assert "https://arxiv.org/abs/2509.26507" in urls
