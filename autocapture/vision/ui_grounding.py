"""UI grounding extraction helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..agents.structured_output import extract_json_payload
from ..config import AppConfig, UIGroundingConfig, is_loopback_host
from ..image_utils import ensure_rgb
from ..logging_utils import get_logger
from ..llm.governor import get_global_governor
from ..llm.prompt_strategy import PromptStrategySettings
from ..runtime_governor import RuntimeGovernor
from .clients import VisionClient
from .tiling import VisionTile, build_tiles


UI_SCHEMA_VERSION = "v1"
_FENCE_RE = re.compile(r"```(?:json|tron)?(.*?)```", re.DOTALL | re.IGNORECASE)


class UIElement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Stable UI element identifier.")
    role: str = Field(..., description="UI element role/type (button, input, menu, etc).")
    label: str = Field("", description="Visible label or accessible name.")
    bbox_norm: list[float] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    click_point_norm: list[float] | None = None
    canonical_label: str | None = None

    @field_validator("bbox_norm")
    @classmethod
    def _validate_bbox(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("bbox_norm must have 4 values")
        vals = [float(val) for val in value]
        for val in vals:
            if val < 0.0 or val > 1.0:
                raise ValueError("bbox_norm values must be within [0,1]")
        return vals

    @field_validator("click_point_norm")
    @classmethod
    def _validate_click_point(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError("click_point_norm must have 2 values")
        vals = [float(val) for val in value]
        for val in vals:
            if val < 0.0 or val > 1.0:
                raise ValueError("click_point_norm values must be within [0,1]")
        return vals


class UIGroundingPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    elements: list[UIElement] = Field(default_factory=list)


@dataclass(frozen=True)
class UIGroundingResult:
    elements: list[dict[str, Any]]
    tags: dict[str, Any]


class UIGroundingRouter:
    def __init__(
        self, config: AppConfig, *, runtime_governor: RuntimeGovernor | None = None
    ) -> None:
        self._config = config
        self._settings = config.vision_extract.ui_grounding
        self._log = get_logger("vision.ui_grounding")
        self._runtime = runtime_governor
        self._vlm: UIGroundingVLMExtractor | None = None

    def extract(self, image: np.ndarray) -> UIGroundingResult:
        if not self._settings.enabled:
            return _disabled_result("disabled")
        if self._runtime and not self._runtime.allow_ui_grounding():
            return _disabled_result("qos_disabled")
        backend = (self._settings.backend or "").lower()
        if backend in {"qwen_vl_ui_prompt", "vlm"}:
            if self._vlm is None:
                self._vlm = UIGroundingVLMExtractor(self._config, self._settings)
            return self._vlm.extract(image)
        if backend == "ui_venus":
            return _disabled_result("ui_venus_unavailable")
        return _disabled_result("unsupported_backend")


class UIGroundingVLMExtractor:
    def __init__(self, config: AppConfig, settings: UIGroundingConfig) -> None:
        self._config = config
        self._settings = settings
        self._log = get_logger("vision.ui_grounding.vlm")
        backend = settings.vlm
        provider = backend.provider
        base_url = backend.base_url
        api_key = backend.api_key
        if provider == "ollama":
            base_url = base_url or config.llm.ollama_url
        if not _cloud_images_allowed(config, base_url, provider, backend.allow_cloud):
            raise RuntimeError("Cloud UI grounding calls not permitted by privacy settings")
        self._client = VisionClient(
            provider=provider,
            model=backend.model,
            base_url=base_url,
            api_key=api_key,
            timeout_s=config.llm.timeout_s,
            retries=config.llm.retries,
            prompt_strategy=PromptStrategySettings.from_llm_config(
                config.llm, data_dir=config.capture.data_dir
            ),
            governor=get_global_governor(config),
            priority="background",
        )

    def extract(self, image: np.ndarray) -> UIGroundingResult:
        rgb = ensure_rgb(image)
        tiles = build_tiles(rgb, tiles_x=2, tiles_y=2, max_tile_px=1024, include_full_frame=True)
        system_prompt, user_prompt = _build_prompt(tiles)
        response = self._client.generate(
            system_prompt, user_prompt, [tile.image_bytes for tile in tiles]
        )
        payload, meta = _parse_payload(response)
        elements = payload.elements if payload else []
        elements = _normalize_elements(elements)
        tags = {
            "schema_version": UI_SCHEMA_VERSION,
            "engine": "qwen_vl_ui_prompt",
            "parse_failed": meta["parse_failed"],
            "parse_format": meta["format"],
            "parse_error": meta.get("error"),
            "raw_output": meta["raw_output"],
            "elements": [element.model_dump() for element in elements],
            "tiles": [
                {
                    "index": tile.index,
                    "bbox_norm": list(tile.bbox_norm),
                    "kind": tile.kind,
                }
                for tile in tiles
            ],
        }
        return UIGroundingResult(
            elements=[element.model_dump() for element in elements],
            tags=tags,
        )


def _build_prompt(tiles: list[VisionTile]) -> tuple[str, str]:
    system_prompt = (
        "You extract UI elements from screenshots. Respond with JSON only. "
        "Do not include commentary or markdown."
    )
    user_prompt = (
        "Return strict JSON with schema:\n"
        "{\n"
        '  "elements": [\n'
        "    {\n"
        '      "id": "stable id like U1",\n'
        '      "role": "button|input|menu|tab|checkbox|radio|link|text|image|panel",\n'
        '      "label": "visible label or accessible name",\n'
        '      "bbox_norm": [x0, y0, x1, y1],\n'
        '      "click_point_norm": [x, y],\n'
        '      "canonical_label": "normalized label",\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Order elements top-left to bottom-right.\n"
        "- bbox_norm/click_point_norm are normalized to the full frame.\n"
        "- Use deterministic ids U1, U2, ... in order.\n"
        "- Leave optional fields empty if unknown.\n\n"
        "Tiles:\n" + _format_tile_list(tiles)
    )
    return system_prompt, user_prompt


def _format_tile_list(tiles: list[VisionTile]) -> str:
    lines: list[str] = []
    for tile in tiles:
        bbox = ",".join(f"{val:.4f}" for val in tile.bbox_norm)
        lines.append(f"{tile.index}. kind={tile.kind} bbox_norm=[{bbox}]")
    return "\n".join(lines)


def _parse_payload(raw_text: str) -> tuple[UIGroundingPayload | None, dict[str, Any]]:
    if not raw_text:
        return None, {"parse_failed": True, "format": "empty", "raw_output": ""}
    candidate = _extract_fenced(raw_text)
    data = None
    fmt = "unknown"
    try:
        payload = extract_json_payload(candidate)
        data = json.loads(payload)
        fmt = "json"
    except Exception:
        data = None
    if data is None:
        return None, {
            "parse_failed": True,
            "format": "plain",
            "raw_output": _truncate(raw_text),
        }
    try:
        parsed = UIGroundingPayload.model_validate(data)
    except Exception as exc:
        return None, {
            "parse_failed": True,
            "format": fmt,
            "raw_output": _truncate(raw_text),
            "error": str(exc),
        }
    return parsed, {"parse_failed": False, "format": fmt, "raw_output": _truncate(raw_text)}


def _normalize_elements(elements: list[UIElement]) -> list[UIElement]:
    def _sort_key(item: tuple[int, UIElement]) -> tuple[float, float, float, float, int]:
        idx, elem = item
        bbox = elem.bbox_norm if elem.bbox_norm else [0.0, 0.0, 0.0, 0.0]
        return (bbox[1], bbox[0], bbox[3], bbox[2], idx)

    ordered = sorted(list(enumerate(elements)), key=_sort_key)
    normalized: list[UIElement] = []
    for out_idx, (_in_idx, element) in enumerate(ordered, start=1):
        data = element.model_dump()
        data["id"] = data.get("id") or f"U{out_idx}"
        if isinstance(data.get("bbox_norm"), list):
            data["bbox_norm"] = [round(float(val), 4) for val in data["bbox_norm"]]
        if isinstance(data.get("click_point_norm"), list):
            data["click_point_norm"] = [round(float(val), 4) for val in data["click_point_norm"]]
        if data.get("confidence") is not None:
            data["confidence"] = round(float(data["confidence"]), 4)
        normalized.append(UIElement.model_validate(data))
    return normalized


def _extract_fenced(raw: str) -> str:
    matches = _FENCE_RE.findall(raw)
    if matches:
        return "\n".join(match.strip() for match in matches if match.strip()).strip()
    return raw.strip()


def _truncate(text: str, max_chars: int = 4000) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars]


def _is_local_endpoint(base_url: str | None) -> bool:
    if not base_url:
        return True
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    return is_loopback_host(host)


def _cloud_images_allowed(
    config: AppConfig, base_url: str | None, provider: str, allow_cloud: bool
) -> bool:
    if _is_local_endpoint(base_url) and provider != "openai":
        return True
    if not allow_cloud:
        return False
    if config.offline:
        return False
    if not config.privacy.cloud_enabled:
        return False
    return bool(config.privacy.allow_cloud_images)


def _disabled_result(reason: str) -> UIGroundingResult:
    return UIGroundingResult(
        elements=[],
        tags={
            "schema_version": UI_SCHEMA_VERSION,
            "engine": "disabled",
            "parse_failed": True,
            "parse_format": "disabled",
            "reason": reason,
            "elements": [],
        },
    )
