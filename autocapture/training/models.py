"""Training pipeline request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrainingRunRequest(BaseModel):
    run_id: str | None = None
    dataset_path: str | None = None
    output_dir: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = False


class TrainingRunResult(BaseModel):
    status: str
    message: str | None = None
    artifacts: list[str] = Field(default_factory=list)


__all__ = ["TrainingRunRequest", "TrainingRunResult"]
