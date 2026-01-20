"""LLM transport helpers for live and replay modes."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import httpx

from ..security.redaction import redact_mapping


class LLMTransport(Protocol):
    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None,
        *,
        timeout_s: float,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RequestSummary:
    url: str
    payload: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {"url": self.url, "payload": self.payload},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )


class HttpxTransport:
    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None,
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()


class ReplayTransport:
    def __init__(self, fixture_dir: Path, *, case_id: str | None = None) -> None:
        self._fixture_dir = Path(fixture_dir)
        self._case_id = case_id

    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None,
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        _ = headers
        _ = timeout_s
        request_hash, _summary = build_request_hash(url, payload)
        candidates = []
        if request_hash:
            candidates.append(f"{request_hash}.json")
        if self._case_id:
            candidates.append(f"{self._case_id}.json")
        for name in candidates:
            path = self._fixture_dir / name
            if path.exists():
                return _load_fixture(path)
        raise FileNotFoundError(
            f"Replay fixture not found in {self._fixture_dir} (case_id={self._case_id})"
        )


class RecordingTransport:
    def __init__(
        self,
        inner: LLMTransport,
        fixture_dir: Path,
        *,
        case_id: str | None = None,
    ) -> None:
        self._inner = inner
        self._fixture_dir = Path(fixture_dir)
        self._case_id = case_id

    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None,
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        response = await self._inner.post_json(url, payload, headers, timeout_s=timeout_s)
        request_hash, summary = build_request_hash(url, payload)
        fixture = {
            "meta": {
                "case_id": self._case_id,
                "request_hash": request_hash,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "request": summary,
            },
            "response": redact_mapping(response) if isinstance(response, dict) else response,
        }
        filename = self._case_id or request_hash or "fixture"
        path = self._fixture_dir / f"{filename}.json"
        _atomic_write_json(path, fixture)
        return response


def build_request_hash(url: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    summary = summarize_request_payload(payload)
    digest = hashlib.sha256(RequestSummary(url=url, payload=summary).to_json().encode("utf-8"))
    return digest.hexdigest(), summary


def summarize_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    summary["model"] = payload.get("model")
    summary["temperature"] = payload.get("temperature")

    messages = payload.get("input") or payload.get("messages")
    if isinstance(messages, list):
        msg_summary, total_chars = _summarize_messages(messages)
        summary["message_count"] = len(msg_summary)
        summary["message_chars"] = total_chars
        summary["messages"] = msg_summary

    for key, value in payload.items():
        if key in {"input", "messages", "model", "temperature"}:
            continue
        if isinstance(value, (int, float, bool)):
            summary[key] = value
        elif isinstance(value, str):
            summary[f"{key}_chars"] = len(value)
        elif isinstance(value, list):
            summary[f"{key}_count"] = len(value)
        elif isinstance(value, dict):
            summary[f"{key}_keys"] = sorted(value.keys())
    return summary


def _summarize_messages(messages: list[Any]) -> tuple[list[dict[str, Any]], int]:
    summarized: list[dict[str, Any]] = []
    total_chars = 0
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or ""
        content = item.get("content")
        content_len, part_count, types = _summarize_content(content)
        total_chars += content_len
        summarized.append(
            {
                "role": str(role),
                "content_len": content_len,
                "parts": part_count,
                "types": types,
            }
        )
    return summarized, total_chars


def _summarize_content(content: Any) -> tuple[int, int, list[str]]:
    if content is None:
        return 0, 0, []
    if isinstance(content, str):
        return len(content), 1, ["text"]
    if isinstance(content, list):
        total = 0
        types: set[str] = set()
        parts = 0
        for part in content:
            parts += 1
            if isinstance(part, dict):
                part_type = str(part.get("type") or "")
                types.add(part_type or "unknown")
                text = part.get("text")
                if isinstance(text, str):
                    total += len(text)
            elif isinstance(part, str):
                total += len(part)
                types.add("text")
            else:
                types.add(type(part).__name__)
        return total, parts, sorted(types)
    return 0, 1, [type(content).__name__]


def _load_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "response" in payload:
        response = payload.get("response")
        if isinstance(response, dict):
            return response
    if not isinstance(payload, dict):
        raise ValueError(f"Fixture {path} must contain a JSON object")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)
