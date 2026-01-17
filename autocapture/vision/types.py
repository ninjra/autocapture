"""Vision extraction schemas and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

VISION_SCHEMA_VERSION = "v2"


@dataclass(frozen=True)
class ExtractionResult:
    text: str
    spans: list[dict[str, Any]]
    tags: dict[str, Any]


class VisionRegion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    bbox_norm: list[float] = Field(default_factory=list)
    label: str | None = None
    app_hint: str | None = None
    title_hint: str | None = None
    url_hint: str | None = None
    role_hint: str | None = None
    text_verbatim: str = ""
    keywords: list[str] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    @field_validator("bbox_norm")
    @classmethod
    def _validate_bbox(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("bbox_norm must have 4 values")
        normalized: list[float] = []
        for item in value:
            normalized.append(max(0.0, min(1.0, float(item))))
        return normalized


class VisionExtractionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    screen_summary: str = ""
    regions: list[VisionRegion] = Field(default_factory=list)
    visible_text: str = ""
    content_flags: list[str] = Field(default_factory=list)
    tables_detected: bool | None = None
    spreadsheets_detected: bool | None = None


def build_ocr_payload(
    spans: list[tuple[str, float, list[int]]],
) -> tuple[str, list[dict[str, Any]]]:
    text_parts: list[str] = []
    ocr_spans: list[dict[str, Any]] = []
    offset = 0
    for idx, (text, conf, bbox) in enumerate(spans, start=1):
        text = text or ""
        start = offset
        end = start + len(text)
        text_parts.append(text)
        ocr_spans.append(
            {
                "span_id": f"S{idx}",
                "span_key": f"S{idx}",
                "start": start,
                "end": end,
                "conf": float(conf),
                "bbox": bbox,
                "text": text,
            }
        )
        offset = end + 1
    return "\n".join(text_parts), ocr_spans
