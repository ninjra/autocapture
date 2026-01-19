"""Benchmark result schema utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchResult:
    run_id: str
    git_sha: str
    ts_ms: int
    env: dict[str, Any]
    config: dict[str, Any]
    metrics: dict[str, Any]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "ts_ms": self.ts_ms,
            "env": self.env,
            "config": self.config,
            "metrics": self.metrics,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchResult":
        cls.validate_dict(payload)
        return cls(
            run_id=payload["run_id"],
            git_sha=payload["git_sha"],
            ts_ms=payload["ts_ms"],
            env=payload["env"],
            config=payload["config"],
            metrics=payload["metrics"],
            notes=payload.get("notes") or "",
        )

    @classmethod
    def validate_dict(cls, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("BenchResult payload must be a dict")
        required = {"run_id", "git_sha", "ts_ms", "env", "config", "metrics", "notes"}
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"BenchResult missing keys: {sorted(missing)}")
        if not isinstance(payload["run_id"], str) or not payload["run_id"]:
            raise ValueError("BenchResult.run_id must be a non-empty string")
        if not isinstance(payload["git_sha"], str) or not payload["git_sha"]:
            raise ValueError("BenchResult.git_sha must be a non-empty string")
        if not isinstance(payload["ts_ms"], int):
            raise ValueError("BenchResult.ts_ms must be an int")
        _validate_env(payload["env"])
        _validate_config(payload["config"])
        _validate_metrics(payload["metrics"])
        if not isinstance(payload.get("notes"), str):
            raise ValueError("BenchResult.notes must be a string")


def _validate_env(env: dict[str, Any]) -> None:
    if not isinstance(env, dict):
        raise ValueError("BenchResult.env must be a dict")
    required = {"os", "python", "cpu", "ram_gb", "gpu_name"}
    missing = required - env.keys()
    if missing:
        raise ValueError(f"BenchResult.env missing keys: {sorted(missing)}")
    if not isinstance(env["os"], str):
        raise ValueError("BenchResult.env.os must be a string")
    if not isinstance(env["python"], str):
        raise ValueError("BenchResult.env.python must be a string")
    if env["cpu"] is not None and not isinstance(env["cpu"], str):
        raise ValueError("BenchResult.env.cpu must be a string or None")
    if env["ram_gb"] is not None and not isinstance(env["ram_gb"], (int, float)):
        raise ValueError("BenchResult.env.ram_gb must be a number or None")
    if env["gpu_name"] is not None and not isinstance(env["gpu_name"], str):
        raise ValueError("BenchResult.env.gpu_name must be a string or None")


def _validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("BenchResult.config must be a dict")
    required = {"gpu_mode", "profile", "tuning"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"BenchResult.config missing keys: {sorted(missing)}")
    if not isinstance(config["gpu_mode"], str):
        raise ValueError("BenchResult.config.gpu_mode must be a string")
    if not isinstance(config["profile"], str):
        raise ValueError("BenchResult.config.profile must be a string")
    tuning = config["tuning"]
    if not isinstance(tuning, dict):
        raise ValueError("BenchResult.config.tuning must be a dict")
    tuning_required = {
        "max_workers",
        "batch_size",
        "poll_interval_ms",
        "max_queue_depth",
        "max_cpu_pct_hint",
    }
    missing = tuning_required - tuning.keys()
    if missing:
        raise ValueError(f"BenchResult.config.tuning missing keys: {sorted(missing)}")
    for key in tuning_required:
        if not isinstance(tuning[key], int):
            raise ValueError(f"BenchResult.config.tuning.{key} must be an int")


def _validate_metrics(metrics: dict[str, Any]) -> None:
    if not isinstance(metrics, dict):
        raise ValueError("BenchResult.metrics must be a dict")
    required = {
        "latency_ms_p50",
        "latency_ms_p95",
        "throughput_items_per_s",
        "processed",
        "errors",
        "skipped",
    }
    missing = required - metrics.keys()
    if missing:
        raise ValueError(f"BenchResult.metrics missing keys: {sorted(missing)}")
    for key in ("latency_ms_p50", "latency_ms_p95", "throughput_items_per_s"):
        if not isinstance(metrics[key], (int, float)):
            raise ValueError(f"BenchResult.metrics.{key} must be a number")
    for key in ("processed", "errors", "skipped"):
        if not isinstance(metrics[key], int):
            raise ValueError(f"BenchResult.metrics.{key} must be an int")
