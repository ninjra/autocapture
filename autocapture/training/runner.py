"""Training pipeline runner helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .models import TrainingRunRequest, TrainingRunResult


def load_training_request(path: Path | None) -> TrainingRunRequest:
    if path is None:
        return TrainingRunRequest()
    if not path.exists():
        raise FileNotFoundError(str(path))
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ValueError("training request must be a mapping")
    return TrainingRunRequest.model_validate(payload)


def list_training_pipelines(plugins) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for record in plugins.list_extensions("training.pipeline"):
        entries.append(
            {
                "id": record.extension_id,
                "name": record.name,
                "plugin_id": record.plugin_id,
                "source": record.source,
            }
        )
    return entries


def run_training_pipeline(
    plugins,
    pipeline_id: str,
    request: TrainingRunRequest,
) -> TrainingRunResult:
    pipeline = plugins.resolve_extension("training.pipeline", pipeline_id)
    if not hasattr(pipeline, "run"):
        raise RuntimeError("training pipeline missing run()")
    return pipeline.run(request)


__all__ = ["load_training_request", "list_training_pipelines", "run_training_pipeline"]
