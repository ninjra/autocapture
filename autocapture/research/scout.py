"""Research scout for model/paper discovery."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import httpx

from ..config import AppConfig
from ..logging_utils import get_logger

HF_TAGS = ["vision-language", "ocr", "reranker", "embeddings"]
ARXIV_KEYWORDS = [
    "vision-language",
    "document understanding",
    "reranking",
    "screen understanding",
    "diffusion transformer",
    "prompt repetition",
    "prompt duplication",
]
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "when",
    "while",
    "for",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "as",
    "with",
    "without",
    "from",
    "into",
    "over",
    "under",
    "between",
    "among",
    "about",
    "around",
    "through",
    "during",
    "before",
    "after",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "we",
    "our",
    "you",
    "your",
}
WATCHLIST = [
    {
        "source": "watchlist",
        "title": "Baby Dragon Hatchling Architecture",
        "url": "https://arxiv.org/abs/2509.26507",
        "rationale": "Track diffusion transformer work relevant to screen understanding.",
    },
    {
        "source": "watchlist",
        "title": "DiffusionVL-Qwen2.5VL-7B",
        "url": "https://huggingface.co/hustvl/DiffusionVL-Qwen2.5VL-7B",
        "rationale": "Monitor diffusion-based VLM checkpoint adoption and usage.",
    },
    {
        "source": "watchlist",
        "title": "DeepSeek-OCR",
        "url": "https://huggingface.co/deepseek-ai/DeepSeek-OCR",
        "rationale": "Monitor OCR model updates and throughput improvements.",
    },
    {
        "source": "watchlist",
        "title": "DeepSeek-OCR (GitHub)",
        "url": "https://github.com/deepseek-ai/DeepSeek-OCR",
        "rationale": "Track repository releases and implementation notes.",
    },
    {
        "source": "watchlist",
        "title": "NVIDIA RTX open-source AI tooling",
        "url": "https://developer.nvidia.com/blog/open-source-ai-tool-upgrades-speed-up-llm-and-diffusion-models-on-nvidia-rtx-pcs/",
        "rationale": "Monitor RTX-optimized inference updates for local acceleration.",
    },
    {
        "source": "watchlist",
        "title": "Tiny diffusion experiments",
        "url": "https://github.com/nathan-barry/tiny-diffusion",
        "rationale": "Track lightweight diffusion experiments for fast local adapters.",
    },
]
_NORMALIZATION_MAP = {
    "repitition": "repetition",
}
_CACHE_VERSION = 1


def run_scout(
    config: AppConfig,
    *,
    http_client: httpx.Client | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    log = get_logger("research.scout")
    now = now or dt.datetime.now(dt.timezone.utc)
    offline = config.offline and not config.privacy.cloud_enabled
    warnings: list[str] = []
    if offline:
        warnings.append("offline")
    cache_path = _cache_path(config.capture.data_dir)
    cache = _load_cache(cache_path)

    client = http_client
    close_client = False
    if client is None:
        client = httpx.Client(timeout=10.0)
        close_client = True

    try:
        hf_items, hf_status = _resolve_hf_items(client, offline, cache, warnings, log=log)
        arxiv_items, arxiv_status = _resolve_arxiv_items(
            client, offline, cache, warnings, now=now, log=log
        )
    finally:
        if close_client:
            client.close()

    hf_items = _normalize_items(hf_items)
    arxiv_items = _normalize_items(arxiv_items)
    ranked_items = _rank_items(hf_items + arxiv_items)

    report = {
        "generated_at": now.isoformat(),
        "offline": offline,
        "watchlist": WATCHLIST,
        "sources": {
            "huggingface": {
                "status": hf_status,
                "tags": HF_TAGS,
                "items": hf_items,
            },
            "arxiv": {
                "status": arxiv_status,
                "keywords": ARXIV_KEYWORDS,
                "items": arxiv_items,
            },
        },
        "ranked_items": ranked_items,
        "warnings": warnings,
    }

    if not offline:
        _write_cache(cache_path, report)
    return report


def write_report(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True)
    out_path.write_text(payload, encoding="utf-8")


def append_report_log(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = report.get("generated_at", "")
    lines = [f"## {timestamp}", ""]
    ranked = report.get("ranked_items") or []
    if not ranked:
        lines.append("- No items found.")
    else:
        for item in ranked[:10]:
            title = item.get("title") or "untitled"
            url = item.get("url") or ""
            rationale = item.get("rationale") or ""
            lines.append(f"- [{title}]({url}) - {rationale}")
    lines.append("")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _cache_path(data_dir: Path) -> Path:
    return Path(data_dir) / "research" / "scout_cache.json"


def _load_cache(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("cache_version") != _CACHE_VERSION:
        return None
    return data


def _write_cache(cache_path: Path, report: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": _CACHE_VERSION,
        "generated_at": report.get("generated_at"),
        "sources": report.get("sources", {}),
    }
    cache_path.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


def _resolve_hf_items(
    client: httpx.Client,
    offline: bool,
    cache: dict[str, Any] | None,
    warnings: list[str],
    *,
    log,
) -> tuple[list[dict[str, Any]], str]:
    if offline:
        cached = _cached_items(cache, "huggingface")
        if cached:
            warnings.append("offline_using_cached_huggingface")
            return cached, "cached"
        warnings.append("offline_no_huggingface_cache")
        return [], "offline"
    try:
        items = _fetch_hf_items(client)
        return items, "ok"
    except Exception as exc:
        log.warning("Hugging Face scout failed: {}", exc)
        warnings.append("huggingface_fetch_failed")
        cached = _cached_items(cache, "huggingface")
        if cached:
            return cached, "cached"
        return [], "error"


def _resolve_arxiv_items(
    client: httpx.Client,
    offline: bool,
    cache: dict[str, Any] | None,
    warnings: list[str],
    *,
    now: dt.datetime,
    log,
) -> tuple[list[dict[str, Any]], str]:
    if offline:
        cached = _cached_items(cache, "arxiv")
        if cached:
            warnings.append("offline_using_cached_arxiv")
            return cached, "cached"
        warnings.append("offline_no_arxiv_cache")
        return [], "offline"
    try:
        items = _fetch_arxiv_items(client, now=now)
        return items, "ok"
    except Exception as exc:
        log.warning("arXiv scout failed: {}", exc)
        warnings.append("arxiv_fetch_failed")
        cached = _cached_items(cache, "arxiv")
        if cached:
            return cached, "cached"
        return [], "error"


def _fetch_hf_items(client: httpx.Client) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for tag in HF_TAGS:
        response = client.get(
            "https://huggingface.co/api/models",
            params={"filter": tag, "sort": "downloads", "direction": -1, "limit": 12},
        )
        response.raise_for_status()
        payload = response.json()
        for model in payload or []:
            model_id = model.get("modelId") or model.get("id")
            if not model_id:
                continue
            downloads = int(model.get("downloads") or 0)
            likes = int(model.get("likes") or 0)
            url = f"https://huggingface.co/{model_id}"
            item = {
                "source": "huggingface",
                "id": model_id,
                "title": model_id,
                "url": url,
                "rationale": f"{tag} model with {downloads} downloads.",
                "raw_score": downloads,
                "score": 0.0,
                "tags": sorted({tag, *(model.get("tags") or [])}),
                "metadata": {
                    "downloads": downloads,
                    "likes": likes,
                    "tag": tag,
                    "last_modified": model.get("lastModified"),
                },
            }
            items.append(item)
    return _dedupe_items(items, key="id")


def _fetch_arxiv_items(client: httpx.Client, *, now: dt.datetime) -> list[dict[str, Any]]:
    query = _build_arxiv_query(ARXIV_KEYWORDS)
    response = client.get(
        "https://export.arxiv.org/api/query",
        params={
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": 20,
        },
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        paper_id = entry.findtext("atom:id", default="", namespaces=ns)
        title = _clean_text(entry.findtext("atom:title", default="", namespaces=ns))
        summary = _clean_text(entry.findtext("atom:summary", default="", namespaces=ns))
        published = entry.findtext("atom:published", default="", namespaces=ns)
        published_dt = _parse_datetime(published) or now
        days_ago = max((now - published_dt).days, 0)
        matched = _match_keyword(title, summary)
        rationale = f"Recent arXiv paper on {matched or 'screen understanding'}."
        item = {
            "source": "arxiv",
            "id": paper_id,
            "title": title or paper_id,
            "url": paper_id,
            "rationale": rationale,
            "raw_score": max(0.0, 1.0 - (days_ago / 365.0)),
            "score": 0.0,
            "tags": [matched] if matched else [],
            "metadata": {
                "published": published,
                "summary": summary[:500],
            },
        }
        items.append(item)
    return _dedupe_items(items, key="id")


def _match_keyword(title: str, summary: str) -> str | None:
    text = _normalize_phrase(f"{title} {summary}")
    text_tokens = _significant_tokens(text)
    tokens_in_text = set(text_tokens)
    for keyword in ARXIV_KEYWORDS:
        phrases = _keyword_phrase_variants(keyword)
        if any(phrase in text for phrase in phrases):
            return keyword
        if " " in keyword and len(_significant_tokens(keyword)) >= 2:
            tokens = _significant_tokens(keyword)
            if all(
                token in tokens_in_text
                or any(text_token.startswith(token) for text_token in text_tokens)
                for token in tokens
            ):
                return keyword
    return None


def _build_arxiv_query(keywords: list[str]) -> str:
    clauses: list[str] = []
    seen = set()
    for keyword in keywords:
        phrases = _keyword_phrase_variants(keyword)
        if len(phrases) == 1:
            phrase_clause = f'all:"{phrases[0]}"'
        else:
            phrase_clause = "(" + " OR ".join([f'all:"{phrase}"' for phrase in phrases]) + ")"
        tokens = _significant_tokens(keyword)
        if " " in keyword and len(tokens) >= 2:
            token_clause = "(" + " AND ".join([f'all:"{token}"' for token in tokens]) + ")"
            keyword_clause = f"({phrase_clause} OR {token_clause})"
        else:
            keyword_clause = phrase_clause
        if keyword_clause in seen:
            continue
        seen.add(keyword_clause)
        clauses.append(keyword_clause)
    return " OR ".join(clauses)


def _keyword_phrase_variants(keyword: str) -> list[str]:
    base = _normalize_phrase(keyword).strip()
    variants = [base]
    if "-" in base and " " not in base:
        variants.append(" ".join(base.replace("-", " ").split()))
    return _stable_dedupe(variants)


def _significant_tokens(keyword: str) -> list[str]:
    tokens: list[str] = []
    seen = set()
    for token in _extract_tokens(keyword):
        if token in STOPWORDS:
            continue
        if len(token) < 4:
            continue
        if token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _extract_tokens(text: str) -> list[str]:
    normalized = _normalize_phrase(text)
    tokens = []
    for match in re.findall(r"[a-z0-9]+", normalized):
        tokens.append(_NORMALIZATION_MAP.get(match, match))
    return tokens


def _normalize_phrase(value: str) -> str:
    normalized = (value or "").strip().lower()

    def replace_token(match: re.Match[str]) -> str:
        token = match.group(0)
        return _NORMALIZATION_MAP.get(token, token)

    return re.sub(r"[a-z0-9]+", replace_token, normalized)


def _stable_dedupe(items: list[str]) -> list[str]:
    seen = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _clean_text(value: str) -> str:
    return " ".join((value or "").split())


def _parse_datetime(value: str) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def _normalize_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = [float(item.get("raw_score", 0.0)) for item in items]
    if not scores:
        return items
    min_score = min(scores)
    max_score = max(scores)
    for item in items:
        raw = float(item.get("raw_score", 0.0))
        if max_score == min_score:
            item["score"] = 1.0
        else:
            item["score"] = (raw - min_score) / (max_score - min_score)
    return items


def _rank_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        items,
        key=lambda item: (
            -float(item.get("score", 0.0)),
            item.get("source", ""),
            item.get("title", ""),
            item.get("url", ""),
        ),
    )
    output: list[dict[str, Any]] = []
    for idx, item in enumerate(ranked, start=1):
        entry = dict(item)
        entry["rank"] = idx
        output.append(entry)
    return output


def _dedupe_items(items: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    seen = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        value = item.get(key)
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(item)
    return deduped


def _cached_items(cache: dict[str, Any] | None, source: str) -> list[dict[str, Any]]:
    if not cache:
        return []
    sources = cache.get("sources") or {}
    data = sources.get(source) or {}
    items = data.get("items") or []
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items]
