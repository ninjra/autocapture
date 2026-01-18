"""Optional PaddleOCR PP-Structure layout extraction."""

from __future__ import annotations

import importlib.util
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import OCRConfig
from ..gpu_lease import get_global_gpu_lease
from ..logging_utils import get_logger


@dataclass(frozen=True)
class PPStructureLayout:
    blocks: list[dict[str, Any]]
    markdown: str
    tags: dict[str, Any]


class PaddleLayoutExtractor:
    def __init__(self, config: OCRConfig) -> None:
        self._config = config
        self._log = get_logger("ocr.ppstructure")
        self._engine: Any | None = None
        self._lease_key = f"ppstructure:{id(self)}"
        get_global_gpu_lease().register_release_hook(self._lease_key, self._on_release)

    def extract(self, image: np.ndarray) -> PPStructureLayout | None:
        if not self._config.paddle_ppstructure_enabled:
            return PPStructureLayout([], "", {"status": "disabled"})
        if importlib.util.find_spec("paddleocr") is None:
            return PPStructureLayout([], "", {"status": "missing_dependency"})
        model_dir = self._config.paddle_ppstructure_model_dir
        if not model_dir:
            return PPStructureLayout([], "", {"status": "model_dir_required"})
        model_path = Path(model_dir).expanduser()
        if not model_path.exists():
            return PPStructureLayout([], "", {"status": "model_dir_missing"})
        engine = self._engine or self._init_engine(model_path)
        if engine is None:
            return PPStructureLayout([], "", {"status": "init_failed"})
        bgr = image[:, :, ::-1]
        try:
            result = engine(bgr)
        except Exception as exc:  # pragma: no cover - best effort
            self._log.debug("PP-Structure extraction failed: {}", exc)
            return PPStructureLayout([], "", {"status": "extract_failed", "error": str(exc)})
        blocks, markdown = _convert_result(result)
        return PPStructureLayout(
            blocks=blocks,
            markdown=markdown,
            tags={
                "status": "ok",
                "blocks": len(blocks),
                "model_dir": str(model_path),
                "use_gpu": bool(self._config.paddle_ppstructure_use_gpu),
            },
        )

    def _init_engine(self, model_dir: Path) -> Any | None:
        try:
            from paddleocr import PPStructure  # type: ignore
        except Exception as exc:  # pragma: no cover - best effort
            self._log.debug("PP-Structure import failed: {}", exc)
            return None
        os.environ.setdefault("PADDLEOCR_BASEDIR", str(model_dir))
        os.environ.setdefault("PADDLEOCR_HOME", str(model_dir))
        kwargs: dict[str, Any] = {"show_log": False}
        try:
            params = inspect.signature(PPStructure).parameters
        except (TypeError, ValueError):
            params = {}
        if "use_gpu" in params:
            kwargs["use_gpu"] = bool(self._config.paddle_ppstructure_use_gpu)
        if "layout" in params:
            kwargs["layout"] = True
        if "image_orientation" in params:
            kwargs["image_orientation"] = True
        if "model_dir" in params:
            kwargs["model_dir"] = str(model_dir)
        try:
            self._engine = PPStructure(**kwargs)
            return self._engine
        except Exception as exc:  # pragma: no cover - best effort
            self._log.debug("PP-Structure init failed: {}", exc)
            return None

    def _on_release(self, reason: str) -> None:
        _ = reason
        self._engine = None


def _convert_result(result: Any) -> tuple[list[dict[str, Any]], str]:
    blocks: list[dict[str, Any]] = []
    if isinstance(result, list):
        for item in result:
            block = _convert_item(item)
            if block:
                blocks.append(block)
    blocks.sort(key=_block_sort_key)
    md_lines = []
    for block in blocks:
        text = (block.get("text") or "").strip()
        label = block.get("type") or "block"
        if text:
            md_lines.append(f"[{label}] {text}")
        else:
            md_lines.append(f"[{label}]")
    return blocks, "\n".join(md_lines).strip()


def _convert_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    block_type = item.get("type") or item.get("label") or "block"
    bbox = item.get("bbox")
    text = _extract_text(item.get("res"))
    block: dict[str, Any] = {"type": block_type, "bbox": bbox, "text": text}
    if "score" in item:
        try:
            block["confidence"] = float(item["score"])
        except (TypeError, ValueError):
            pass
    return block


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "html", "caption"):
            if isinstance(value.get(key), str):
                return value[key]
        if isinstance(value.get("cells"), list):
            cells = value.get("cells") or []
            parts = []
            for cell in cells:
                if isinstance(cell, dict) and isinstance(cell.get("text"), str):
                    parts.append(cell["text"])
            return " | ".join(parts)
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return ""


def _block_sort_key(block: dict[str, Any]) -> tuple[int, int]:
    bbox = block.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x0 = int(float(bbox[0]))
            y0 = int(float(bbox[1]))
            return (y0, x0)
        except (TypeError, ValueError):
            return (0, 0)
    return (0, 0)
