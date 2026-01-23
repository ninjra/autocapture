import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _run_warm_models(args: list[str], config_path: Path) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["AUTOCAPTURE_CONFIG"] = str(config_path)
    result = subprocess.run(
        [sys.executable, str(repo_root / "tools" / "warm_models.py"), *args],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_ollama_models_skip_unused_vlm(tmp_path: Path) -> None:
    config = {
        "agents": {"enabled": True, "vision": {"provider": "ollama", "model": "llava"}},
        "vision_extract": {
            "engine": "rapidocr",
            "fallback_engine": "rapidocr",
            "vlm": {"provider": "ollama", "model": "qwen2.5-vl:7b-instruct"},
        },
        "llm": {"provider": "gateway"},
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    output = _run_warm_models(["--ollama-models"], config_path)
    assert output == "llava"


def test_ollama_model_map_includes_enabled_vlm(tmp_path: Path) -> None:
    config = {
        "agents": {"enabled": False},
        "vision_extract": {
            "engine": "vlm",
            "fallback_engine": "rapidocr",
            "vlm": {"provider": "ollama", "model": "qwen2.5-vl:7b-instruct"},
        },
        "llm": {"provider": "gateway"},
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    output = _run_warm_models(["--ollama-models-json"], config_path)
    entries = json.loads(output)
    assert any(
        entry["path"] == "vision_extract.vlm.model"
        and entry["model"] == "qwen2.5-vl:7b-instruct"
        for entry in entries
    )
