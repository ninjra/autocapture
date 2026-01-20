from __future__ import annotations

import json
import sys
from pathlib import Path

from autocapture.training.models import TrainingRunRequest
from autocapture.training.pipelines import DisabledTrainingPipeline, pipeline_from_settings
from autocapture.training.runner import load_training_request, run_training_pipeline


class _StubPlugins:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.calls: list[tuple[str, str]] = []

    def resolve_extension(self, kind: str, extension_id: str, **_kwargs):
        self.calls.append((kind, extension_id))
        return self.pipeline


def test_run_training_pipeline_returns_unavailable() -> None:
    pipeline = DisabledTrainingPipeline("lora")
    plugins = _StubPlugins(pipeline)
    result = run_training_pipeline(plugins, "lora", TrainingRunRequest())
    assert result.status == "unavailable"
    assert plugins.calls == [("training.pipeline", "lora")]


def test_load_training_request_from_json(tmp_path: Path) -> None:
    payload = {"run_id": "r1", "dataset_path": "data.json", "params": {"lr": 1e-4}}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    request = load_training_request(path)
    assert request.run_id == "r1"
    assert request.dataset_path == "data.json"
    assert request.params["lr"] == 1e-4


def test_command_training_pipeline_executes(tmp_path: Path) -> None:
    settings = {
        "command": [
            sys.executable,
            "-c",
            "import sys, pathlib; pathlib.Path(sys.argv[1]).write_text('ok')",
            "{output_dir}/out.txt",
        ]
    }
    pipeline = pipeline_from_settings("lora", settings)
    request = TrainingRunRequest(output_dir=str(tmp_path))
    result = pipeline.run(request)
    assert result.status == "ok"
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "ok"


def test_command_training_pipeline_dry_run(tmp_path: Path) -> None:
    settings = {
        "command": [sys.executable, "-c", "import sys; sys.exit(0)"],
        "dry_run_message": "dry",
    }
    pipeline = pipeline_from_settings("lora", settings)
    request = TrainingRunRequest(output_dir=str(tmp_path), dry_run=True)
    result = pipeline.run(request)
    assert result.status == "ok"
    assert result.message == "dry"
