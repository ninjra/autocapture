"""Training pipeline scaffolds."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Protocol

from .models import TrainingRunRequest, TrainingRunResult


class TrainingPipeline(Protocol):
    def run(self, request: TrainingRunRequest) -> TrainingRunResult: ...


@dataclass(frozen=True)
class DisabledTrainingPipeline:
    pipeline_id: str
    reason: str | None = None

    def run(self, request: TrainingRunRequest) -> TrainingRunResult:
        _ = request
        message = self.reason or "Training pipeline not configured."
        return TrainingRunResult(status="unavailable", message=message)


@dataclass(frozen=True)
class CommandTrainingPipeline:
    pipeline_id: str
    command: list[str]
    working_dir: str | None = None
    env: dict[str, str] | None = None
    timeout_s: float | None = None
    dry_run_message: str | None = None

    def run(self, request: TrainingRunRequest) -> TrainingRunResult:
        mapping = _format_mapping(request)
        command = [_format_arg(arg, mapping) for arg in self.command]
        env = _format_env(self.env, mapping)
        if request.dry_run:
            return TrainingRunResult(
                status="ok",
                message=self.dry_run_message or "dry_run",
                artifacts=[],
            )
        try:
            result = subprocess.run(
                command,
                check=False,
                cwd=self.working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return TrainingRunResult(status="failed", message="timeout", artifacts=[])
        if result.returncode != 0:
            message = _trim_message(result.stderr or result.stdout, limit=800)
            return TrainingRunResult(status="failed", message=message, artifacts=[])
        return TrainingRunResult(status="ok", message=None, artifacts=[])


def pipeline_from_settings(
    pipeline_id: str,
    settings: dict[str, Any] | None,
) -> TrainingPipeline:
    settings = settings or {}
    if not isinstance(settings, dict):
        return DisabledTrainingPipeline(pipeline_id, reason=None)
    command = settings.get("command")
    if isinstance(command, str):
        command = [command]
    if isinstance(command, list):
        args = settings.get("args")
        if isinstance(args, list):
            command = [*command, *[str(item) for item in args]]
    if command:
        if not isinstance(command, list) or not command:
            raise ValueError("training command must be a non-empty list")
        working_dir = settings.get("working_dir")
        env = settings.get("env") if isinstance(settings.get("env"), dict) else None
        timeout_s = settings.get("timeout_s")
        dry_run_message = settings.get("dry_run_message")
        return CommandTrainingPipeline(
            pipeline_id=pipeline_id,
            command=[str(item) for item in command],
            working_dir=str(working_dir) if working_dir else None,
            env={str(k): str(v) for k, v in env.items()} if env else None,
            timeout_s=float(timeout_s) if timeout_s else None,
            dry_run_message=str(dry_run_message) if dry_run_message else None,
        )
    reason = settings.get("reason") if isinstance(settings.get("reason"), str) else None
    return DisabledTrainingPipeline(pipeline_id, reason=reason)


def _format_mapping(request: TrainingRunRequest) -> dict[str, str]:
    params_json = json.dumps(request.params or {}, ensure_ascii=False)
    return {
        "run_id": str(request.run_id or ""),
        "dataset_path": str(request.dataset_path or ""),
        "output_dir": str(request.output_dir or ""),
        "params_json": params_json,
    }


def _format_arg(arg: str, mapping: dict[str, str]) -> str:
    try:
        return str(arg).format(**mapping)
    except KeyError:
        return str(arg)


def _format_env(
    env: dict[str, str] | None,
    mapping: dict[str, str],
) -> dict[str, str] | None:
    if env is None:
        return None
    merged = dict(os.environ)
    for key, value in env.items():
        merged[str(key)] = _format_arg(str(value), mapping)
    return merged


def _trim_message(text: str | None, *, limit: int) -> str | None:
    if not text:
        return None
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


__all__ = [
    "TrainingPipeline",
    "DisabledTrainingPipeline",
    "CommandTrainingPipeline",
    "pipeline_from_settings",
]
