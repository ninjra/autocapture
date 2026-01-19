"""Graph service request/response models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GraphTimeRange(BaseModel):
    start: str
    end: str


class GraphFilters(BaseModel):
    apps: list[str] | None = None
    domains: list[str] | None = None


class GraphIndexRequest(BaseModel):
    corpus_id: str = Field("default", min_length=1)
    time_range: GraphTimeRange | None = None
    filters: GraphFilters | None = None
    max_events: int | None = Field(None, ge=1)


class GraphQueryRequest(BaseModel):
    corpus_id: str = Field("default", min_length=1)
    query: str
    limit: int = Field(20, ge=1, le=200)
    time_range: GraphTimeRange | None = None
    filters: GraphFilters | None = None


class GraphHit(BaseModel):
    event_id: str
    score: float
    snippet: str | None = None


class GraphIndexResponse(BaseModel):
    status: str
    corpus_id: str
    events_indexed: int
    segments: int


class GraphQueryResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    hits: list[GraphHit] = Field(default_factory=list)


__all__ = [
    "GraphFilters",
    "GraphHit",
    "GraphIndexRequest",
    "GraphIndexResponse",
    "GraphQueryRequest",
    "GraphQueryResponse",
    "GraphTimeRange",
]
