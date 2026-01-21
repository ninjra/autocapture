"""Guided setup helpers for production-like local runs."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .fs_utils import fsync_dir, fsync_file, safe_replace
from .logging_utils import get_logger
from .paths import ensure_config_path


@dataclass(frozen=True)
class SetupChange:
    path: str
    before: Any
    after: Any


@dataclass(frozen=True)
class SetupPlan:
    profile: str
    config_path: Path
    changes: tuple[SetupChange, ...]
    warnings: tuple[str, ...]
    notes: tuple[str, ...]


def run_setup(
    config_path: Path,
    *,
    profile: str = "full",
    apply: bool = False,
    json_output: bool = False,
) -> int:
    """Plan or apply a setup profile to the YAML config."""

    log = get_logger("setup")
    ensure_config_path(config_path)
    plan = plan_setup(config_path, profile=profile)

    if json_output:
        payload = {
            "profile": plan.profile,
            "config_path": str(plan.config_path),
            "changes": [
                {"path": change.path, "before": change.before, "after": change.after}
                for change in plan.changes
            ],
            "warnings": list(plan.warnings),
            "notes": list(plan.notes),
            "applied": apply,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_plan(plan)

    if not apply:
        return 0

    if not plan.changes:
        log.info("No config changes required.")
        return 0

    data = _load_yaml(config_path)
    _apply_changes(data, plan.changes)
    backup_path = _backup_config(config_path)
    _write_yaml(config_path, data)
    log.info("Updated config written to %s (backup: %s)", config_path, backup_path)
    return 0


def plan_setup(config_path: Path, *, profile: str = "full") -> SetupPlan:
    data = _load_yaml(config_path)
    changes: list[SetupChange] = []
    warnings: list[str] = []
    notes: list[str] = []

    if profile != "full":
        raise ValueError(f"Unsupported setup profile: {profile}")

    _set_value(data, ["offline"], False, changes)
    _set_value(data, ["database", "encryption_enabled"], True, changes)
    _set_value(data, ["database", "secure_mode_required"], True, changes)
    _set_value(data, ["database", "allow_insecure_dev"], False, changes)
    _set_value(data, ["tracking", "encryption_enabled"], True, changes)
    _set_value(data, ["ocr", "device"], "cuda", changes)
    _set_value(data, ["reranker", "device"], "cuda", changes)
    _set_value(data, ["routing", "embedding"], "local", changes)
    _set_value(data, ["presets", "active_preset"], "high_fidelity", changes)

    db_url = _get_value(data, ["database", "url"])
    if isinstance(db_url, str) and db_url and not db_url.startswith("sqlite"):
        warnings.append("database.url is not sqlite; SQLCipher encryption applies to sqlite only.")

    if not _has_module("pysqlcipher3"):
        detail = "SQLCipher module missing; run: poetry install --extras sqlcipher"
        if os.name == "nt":
            detail += " (Windows uses rotki-pysqlcipher3 wheels)"
        warnings.append(detail)

    if not _has_module("torch"):
        warnings.append("PyTorch missing; install a CUDA-enabled build.")

    onnx_ok = _probe_onnx_cuda()
    if onnx_ok is False:
        warnings.append("onnxruntime CUDA provider missing; run: poetry install --extras ocr-gpu")

    if not _has_module("fastembed") and not _has_module("sentence_transformers"):
        warnings.append("Embedding backend missing; run: poetry install --extras embed-fast")

    if os.name == "nt" and not _has_module("win32crypt"):
        warnings.append(
            "DPAPI helpers missing (win32crypt); install pywin32 for strongest key protection."
        )

    notes.append("Set AUTOCAPTURE_GPU_MODE=on for a strict GPU run.")
    notes.append("If a plaintext DB exists, run: autocapture db encrypt")
    notes.append("If tracking DB already exists, delete or migrate before enabling encryption.")

    return SetupPlan(
        profile=profile,
        config_path=config_path,
        changes=tuple(changes),
        warnings=tuple(warnings),
        notes=tuple(notes),
    )


def _has_module(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _probe_onnx_cuda() -> bool | None:
    if not _has_module("onnxruntime"):
        return None
    try:
        import onnxruntime as ort  # type: ignore

        providers = ort.get_available_providers() or []
        return "CUDAExecutionProvider" in providers
    except Exception:
        return False


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    payload = yaml.safe_dump(data, sort_keys=False)
    temp_path = path.with_name(f".tmp-{path.name}")
    temp_path.write_text(payload, encoding="utf-8")
    fsync_file(temp_path)
    fsync_dir(temp_path.parent)
    safe_replace(temp_path, path)
    fsync_dir(path.parent)


def _backup_config(path: Path) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak.{stamp}")
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup_path


def _set_value(
    data: dict[str, Any], keys: Iterable[str], value: Any, changes: list[SetupChange]
) -> None:
    keys = list(keys)
    cursor: dict[str, Any] = data
    for key in keys[:-1]:
        next_val = cursor.get(key)
        if not isinstance(next_val, dict):
            next_val = {}
            cursor[key] = next_val
        cursor = next_val
    leaf = keys[-1]
    before = cursor.get(leaf)
    if before != value:
        cursor[leaf] = value
        changes.append(SetupChange(".".join(keys), before, value))


def _get_value(data: dict[str, Any], keys: Iterable[str]) -> Any:
    cursor: Any = data
    for key in keys:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


def _apply_changes(data: dict[str, Any], changes: Iterable[SetupChange]) -> None:
    for change in changes:
        _set_value(data, change.path.split("."), change.after, [])


def _print_plan(plan: SetupPlan) -> None:
    print("SETUP PROFILE:", plan.profile)
    print("CONFIG PATH:", plan.config_path)
    if plan.changes:
        print("CHANGES:")
        for change in plan.changes:
            print(f"  - {change.path}: {change.before!r} -> {change.after!r}")
    else:
        print("CHANGES: none")
    if plan.warnings:
        print("WARNINGS:")
        for warning in plan.warnings:
            print(f"  - {warning}")
    if plan.notes:
        print("NOTES:")
        for note in plan.notes:
            print(f"  - {note}")
