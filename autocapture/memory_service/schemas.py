"""Pydantic models for the Memory Service API."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProvenancePointer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_version_id: str
    chunk_id: str
    start_offset: int
    end_offset: int
    excerpt_hash: str


class PolicyLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audience: list[str]
    sensitivity: str
    labels: list[str] = Field(default_factory=list)


class EntityRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    name: str


class MemoryProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    memory_type: Literal["fact", "procedure", "decision", "episodic", "glossary"] = Field(
        alias="type"
    )
    content_text: str
    content_json: dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(0.5, ge=0.0, le=1.0)
    trust: float = Field(0.5, ge=0.0, le=1.0)
    valid_from: dt.datetime | None = None
    valid_to: dt.datetime | None = None
    policy: PolicyLabels
    entities: list[EntityRef] = Field(default_factory=list)
    provenance: list[ProvenancePointer]
    source: str | None = None

    @model_validator(mode="after")
    def validate_dates(self) -> "MemoryProposal":
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ValueError("valid_from must be <= valid_to")
        return self


class IngestDebug(BaseModel):
    model_config = ConfigDict(extra="forbid")

    now_override: dt.datetime | None = None


class MemoryIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace: str | None = None
    proposals: list[MemoryProposal]
    request_id: str | None = None
    debug: IngestDebug | None = None


class MemoryRejectDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    reasons: list[str]


class MemoryIngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    accepted: int = 0
    deduped: int = 0
    rejected: int = 0
    rejects: list[MemoryRejectDetail] = Field(default_factory=list)


class MemoryPolicyContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audience: list[str]
    sensitivity_max: str


class EntityHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    name: str


class MemoryQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace: str | None = None
    query: str
    policy: MemoryPolicyContext
    entity_hints: list[EntityHint] = Field(default_factory=list)
    max_cards: int | None = None
    max_tokens: int | None = None
    topk_vector: int | None = None
    topk_keyword: int | None = None
    topk_graph: int | None = None
    query_embedding: list[float] | None = None
    request_id: str | None = None
    now_override: dt.datetime | None = None


class MemoryCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str
    memory_type: str
    text: str
    content_json: dict[str, Any] = Field(default_factory=dict)
    citations: list[ProvenancePointer] = Field(default_factory=list)
    why_retrieved: list[str] = Field(default_factory=list)
    score: float = 0.0


class MemoryQueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    cards: list[MemoryCard] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    truncated: bool = False


class MemoryFeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace: str | None = None
    memory_id: str
    useful: bool
    reason: str | None = None
    request_id: str | None = None


class MemoryFeedbackResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    stored: bool


class MemoryHealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    db_connected: bool
    pgvector: bool
    tables_ready: bool
    warnings: list[str] = Field(default_factory=list)
