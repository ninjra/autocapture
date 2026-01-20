"""Shared helpers for training runner scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_params(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    return json.loads(raw)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    ensure_output_dir(path)
    manifest_path = path / "train_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_text_samples(path: Path, *, max_samples: int) -> list[str]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl_text(path, max_samples=max_samples)
    if path.suffix.lower() == ".json":
        return _read_json_text(path, max_samples=max_samples)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line][:max_samples]


def read_dpo_samples(path: Path, *, max_samples: int) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        rows = _read_jsonl(path, max_samples=max_samples)
    else:
        rows = _read_json(path)
    samples: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "")
        chosen = str(row.get("chosen") or "")
        rejected = str(row.get("rejected") or "")
        if not (prompt and chosen and rejected):
            continue
        samples.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        if len(samples) >= max_samples:
            break
    return samples


def dataset_stats(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".jsonl":
        count = sum(1 for _ in path.read_text(encoding="utf-8").splitlines() if _.strip())
        return {"format": "jsonl", "rows": count}
    if path.suffix.lower() == ".json":
        rows = _read_json(path)
        return {"format": "json", "rows": len(rows)}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"format": "text", "rows": len(lines)}


def _read_json(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
        return payload["data"]
    return []


def _read_jsonl(path: Path, *, max_samples: int | None = None) -> list[Any]:
    rows: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def _read_json_text(path: Path, *, max_samples: int) -> list[str]:
    rows = _read_json(path)
    return _extract_text(rows, max_samples=max_samples)


def _read_jsonl_text(path: Path, *, max_samples: int) -> list[str]:
    rows = _read_jsonl(path, max_samples=max_samples)
    return _extract_text(rows, max_samples=max_samples)


def _extract_text(rows: list[Any], *, max_samples: int) -> list[str]:
    samples: list[str] = []
    for row in rows:
        if isinstance(row, str):
            text = row
        elif isinstance(row, dict):
            text = row.get("text") or row.get("prompt") or ""
            if row.get("response"):
                text = f"{text}\n{row['response']}".strip()
        else:
            continue
        text = str(text).strip()
        if not text:
            continue
        samples.append(text)
        if len(samples) >= max_samples:
            break
    return samples
