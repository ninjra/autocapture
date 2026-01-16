"""Screen extraction providers and routing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import numpy as np

from ..agents.structured_output import extract_json_payload
from ..config import AppConfig, VisionBackendConfig, is_loopback_host
from ..image_utils import ensure_rgb
from ..logging_utils import get_logger
from .clients import VisionClient
from .rapidocr import RapidOCRExtractor
from .tiling import VisionTile, build_tiles
from .types import ExtractionResult, VisionExtractionPayload, VisionRegion, build_ocr_payload

_FENCE_RE = re.compile(r"```(?:json|tron)?(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class _ParseMeta:
    parse_failed: bool
    format: str
    raw_output: str
    error: str | None = None


def _extract_fenced(raw: str) -> str:
    if not raw:
        return ""
    matches = _FENCE_RE.findall(raw)
    if matches:
        return "\n".join(match.strip() for match in matches if match.strip()).strip()
    return raw.strip()


def _truncate(text: str, max_chars: int = 4000) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars]


def _parse_payload(raw_text: str) -> tuple[VisionExtractionPayload | None, _ParseMeta]:
    if not raw_text:
        return None, _ParseMeta(parse_failed=True, format="empty", raw_output="")
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
        try:
            from ..format.tron import decode_tron

            data = decode_tron(candidate)
            fmt = "tron"
        except Exception:
            data = None
    if data is None:
        return None, _ParseMeta(
            parse_failed=True,
            format="plain",
            raw_output=_truncate(raw_text),
        )
    try:
        payload = VisionExtractionPayload.model_validate(data)
    except Exception as exc:
        return None, _ParseMeta(
            parse_failed=True,
            format=fmt,
            raw_output=_truncate(raw_text),
            error=str(exc),
        )
    return payload, _ParseMeta(parse_failed=False, format=fmt, raw_output=_truncate(raw_text))


def _format_tile_list(tiles: list[VisionTile]) -> str:
    lines = []
    for tile in tiles:
        bbox = ",".join(f"{val:.4f}" for val in tile.bbox_norm)
        lines.append(f"{tile.index}. kind={tile.kind} bbox_norm=[{bbox}]")
    return "\n".join(lines)


def _build_prompt(tiles: list[VisionTile]) -> tuple[str, str]:
    system_prompt = (
        "You extract screen text and layout. Respond with JSON only. "
        "Do not include commentary or markdown."
    )
    user_prompt = (
        "You will receive multiple images for a single full-screen capture.\n"
        "The first image (if present) is a downscaled full frame. Remaining images are tiles "
        "in row-major order. Use the tile list to place regions in full-screen coordinates.\n\n"
        "Return strict JSON with this schema:\n"
        "{\n"
        '  "screen_summary": "short summary",\n'
        '  "visible_text": "concatenated text transcript",\n'
        '  "regions": [\n'
        "    {\n"
        '      "bbox_norm": [x0, y0, x1, y1],\n'
        '      "label": "region label or ui type",\n'
        '      "app_hint": "app name hint",\n'
        '      "title_hint": "window title hint",\n'
        '      "text_verbatim": "verbatim text",\n'
        '      "keywords": ["keyword1", "keyword2"],\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ],\n"
        '  "tables_detected": false,\n'
        '  "spreadsheets_detected": false\n'
        "}\n\n"
        "Rules:\n"
        "- Preserve text verbatim; do not invent text.\n"
        "- Order regions top-left to bottom-right.\n"
        "- bbox_norm values are normalized to the full-screen frame.\n"
        "- If unsure, leave fields empty and lower confidence.\n\n"
        "Tiles:\n"
        f"{_format_tile_list(tiles)}"
    )
    return system_prompt, user_prompt


def _rect_bbox(bbox_norm: list[float], width: int, height: int) -> list[int]:
    x0 = int(round(max(0.0, min(1.0, bbox_norm[0])) * width))
    y0 = int(round(max(0.0, min(1.0, bbox_norm[1])) * height))
    x1 = int(round(max(0.0, min(1.0, bbox_norm[2])) * width))
    y1 = int(round(max(0.0, min(1.0, bbox_norm[3])) * height))
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def _build_spans_from_regions(
    regions: list[VisionRegion], width: int, height: int
) -> tuple[str, list[dict[str, Any]]]:
    text_parts: list[str] = []
    spans: list[dict[str, Any]] = []
    offset = 0
    span_index = 1
    for region in regions:
        text = (region.text_verbatim or "").strip()
        if not text:
            continue
        start = offset
        end = start + len(text)
        text_parts.append(text)
        spans.append(
            {
                "span_id": f"S{span_index}",
                "span_key": f"S{span_index}",
                "start": start,
                "end": end,
                "conf": float(region.confidence or 0.0),
                "bbox": _rect_bbox(region.bbox_norm, width, height),
                "text": text,
            }
        )
        offset = end + 1
        span_index += 1
    return "\n".join(text_parts), spans


def _resolve_backend(
    config: AppConfig, backend: VisionBackendConfig
) -> tuple[str, str | None, str | None, str]:
    provider = backend.provider
    base_url = backend.base_url
    api_key = backend.api_key
    if provider == "ollama":
        base_url = base_url or config.llm.ollama_url
    if provider == "openai_compatible":
        base_url = base_url or config.llm.openai_compatible_base_url
        api_key = api_key or config.llm.openai_compatible_api_key
    if provider == "openai":
        api_key = api_key or config.llm.openai_api_key
    return provider, base_url, api_key, backend.model


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


class RapidOCRScreenExtractor:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._ocr = RapidOCRExtractor(config.ocr)

    def extract(self, image: np.ndarray) -> ExtractionResult:
        spans = self._ocr.extract(image)
        text, ocr_spans = build_ocr_payload(spans)
        tags = {
            "vision_extract": {
                "schema_version": "v1",
                "engine": "rapidocr-onnxruntime",
                "screen_summary": "",
                "regions": [],
                "visible_text": text,
                "tables_detected": None,
                "spreadsheets_detected": None,
                "parse_failed": False,
                "parse_format": "rapidocr",
                "tiles": [],
            }
        }
        return ExtractionResult(text=text, spans=ocr_spans, tags=tags)


class VLMExtractor:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._vision = config.vision_extract
        self._log = get_logger("vision.vlm")
        provider, base_url, api_key, model = _resolve_backend(config, self._vision.vlm)
        self._provider = provider
        self._base_url = base_url
        if not _cloud_images_allowed(config, base_url, provider, self._vision.vlm.allow_cloud):
            raise RuntimeError("Cloud vision calls not permitted by privacy settings")
        self._client = VisionClient(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_s=config.llm.timeout_s,
            retries=config.llm.retries,
        )

    def extract(self, image: np.ndarray) -> ExtractionResult:
        rgb = ensure_rgb(image)
        tiles = build_tiles(
            rgb,
            tiles_x=self._vision.tiles_x,
            tiles_y=self._vision.tiles_y,
            max_tile_px=self._vision.max_tile_px,
            include_full_frame=self._vision.include_downscaled_full_frame,
        )
        system_prompt, user_prompt = _build_prompt(tiles)
        response = self._client.generate(
            system_prompt, user_prompt, [tile.image_bytes for tile in tiles]
        )
        payload, meta = _parse_payload(response)
        height, width, _ = rgb.shape
        regions = payload.regions if payload else []
        text, spans = _build_spans_from_regions(regions, width, height) if regions else ("", [])
        if not text:
            text = payload.visible_text if payload else _truncate(response)
        tags = _build_tags(
            engine="vlm",
            tiles=tiles,
            payload=payload,
            meta=meta,
            visible_text=text,
        )
        return ExtractionResult(text=text, spans=spans, tags=tags)


class DeepSeekOCRExtractor:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._vision = config.vision_extract
        self._log = get_logger("vision.deepseek")
        provider, base_url, api_key, model = _resolve_backend(config, self._vision.deepseek_ocr)
        self._provider = provider
        self._base_url = base_url
        if not _cloud_images_allowed(
            config, base_url, provider, self._vision.deepseek_ocr.allow_cloud
        ):
            raise RuntimeError("Cloud vision calls not permitted by privacy settings")
        self._client = VisionClient(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_s=config.llm.timeout_s,
            retries=config.llm.retries,
        )

    def extract(self, image: np.ndarray) -> ExtractionResult:
        rgb = ensure_rgb(image)
        tiles = build_tiles(
            rgb,
            tiles_x=self._vision.tiles_x,
            tiles_y=self._vision.tiles_y,
            max_tile_px=self._vision.max_tile_px,
            include_full_frame=self._vision.include_downscaled_full_frame,
        )
        system_prompt, user_prompt = _build_prompt(tiles)
        response = self._client.generate(
            system_prompt, user_prompt, [tile.image_bytes for tile in tiles]
        )
        payload, meta = _parse_payload(response)
        height, width, _ = rgb.shape
        regions = payload.regions if payload else []
        text, spans = _build_spans_from_regions(regions, width, height) if regions else ("", [])
        if not text:
            text = payload.visible_text if payload else _truncate(response)
        tags = _build_tags(
            engine="deepseek-ocr",
            tiles=tiles,
            payload=payload,
            meta=meta,
            visible_text=text,
        )
        return ExtractionResult(text=text, spans=spans, tags=tags)


def _build_tags(
    *,
    engine: str,
    tiles: list[VisionTile],
    payload: VisionExtractionPayload | None,
    meta: _ParseMeta,
    visible_text: str,
) -> dict[str, Any]:
    regions = [region.model_dump() for region in payload.regions] if payload else []
    return {
        "vision_extract": {
            "schema_version": "v1",
            "engine": engine,
            "screen_summary": payload.screen_summary if payload else "",
            "regions": regions,
            "visible_text": visible_text,
            "tables_detected": payload.tables_detected if payload else None,
            "spreadsheets_detected": payload.spreadsheets_detected if payload else None,
            "parse_failed": meta.parse_failed,
            "parse_format": meta.format,
            "parse_error": meta.error,
            "raw_output": meta.raw_output,
            "tiles": [
                {
                    "index": tile.index,
                    "bbox_norm": list(tile.bbox_norm),
                    "kind": tile.kind,
                }
                for tile in tiles
            ],
        }
    }


class ScreenExtractorRouter:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._vision = config.vision_extract
        self._log = get_logger("vision.router")
        self._vlm: VLMExtractor | None = None
        self._rapid: RapidOCRScreenExtractor | None = None
        self._deepseek: DeepSeekOCRExtractor | None = None

    def extract(self, image: np.ndarray) -> ExtractionResult:
        if self._config.routing.ocr == "disabled":
            return _disabled_result("routing_disabled")
        engine = (self._vision.engine or "").lower()
        if engine in {"disabled", "off"}:
            return _disabled_result("engine_disabled")
        try:
            extractor = self._get_extractor(engine)
        except Exception as exc:
            self._log.warning("Failed to initialize extractor {}: {}", engine, exc)
            extractor = None
        if extractor is None:
            return _disabled_result(f"engine_{engine}_unsupported")
        try:
            return extractor.extract(image)
        except Exception as exc:
            self._log.warning("Primary extractor failed ({}): {}", engine, exc)
            try:
                fallback = self._get_fallback(engine)
            except Exception as fallback_exc:
                self._log.warning("Failed to initialize fallback: {}", fallback_exc)
                fallback = None
            if fallback is None:
                return _failed_result(engine, str(exc))
            try:
                return fallback.extract(image)
            except Exception as fallback_exc:
                self._log.warning("Fallback extractor failed: {}", fallback_exc)
                return _failed_result(engine, str(fallback_exc))

    def _get_extractor(self, engine: str):
        if engine in {"vlm", "qwen-vl"}:
            if self._vlm is None:
                self._vlm = VLMExtractor(self._config)
            return self._vlm
        if engine in {"deepseek-ocr", "deepseek"}:
            if self._deepseek is None:
                self._deepseek = DeepSeekOCRExtractor(self._config)
            return self._deepseek
        if engine in {"rapidocr", "rapidocr-onnxruntime"}:
            if self._rapid is None:
                self._rapid = RapidOCRScreenExtractor(self._config)
            return self._rapid
        return None

    def _get_fallback(self, engine: str):
        fallback_engine = (self._vision.fallback_engine or "").lower()
        if fallback_engine == engine:
            return None
        return self._get_extractor(fallback_engine)


def _disabled_result(reason: str) -> ExtractionResult:
    return ExtractionResult(
        text="",
        spans=[],
        tags={
            "vision_extract": {
                "schema_version": "v1",
                "engine": "disabled",
                "parse_failed": True,
                "parse_format": "disabled",
                "reason": reason,
                "regions": [],
                "visible_text": "",
                "tiles": [],
            }
        },
    )


def _failed_result(engine: str, error: str) -> ExtractionResult:
    return ExtractionResult(
        text="",
        spans=[],
        tags={
            "vision_extract": {
                "schema_version": "v1",
                "engine": engine,
                "parse_failed": True,
                "parse_format": "error",
                "error": error,
                "regions": [],
                "visible_text": "",
                "tiles": [],
            }
        },
    )
