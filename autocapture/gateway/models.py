"""Gateway request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    stream: bool | None = None
    lora_adapter_id: str | None = None
    tenant_id: str | None = None


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: Any


class GatewayError(BaseModel):
    error: str
    detail: dict[str, Any] | None = None


class GatewayHealth(BaseModel):
    status: str = Field("ok")
    registry_enabled: bool
    providers: list[str] = Field(default_factory=list)

