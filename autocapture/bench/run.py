"""Deterministic CPU/GPU benchmark harness."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..config import AppConfig, load_config
from ..image_utils import hash_rgb_image
from ..paths import default_config_path
from ..runtime_context import build_runtime_context
from ..runtime_device import DeviceKind, DeviceManager, cuda_available
from ..runtime_env import (
    RuntimeEnvConfig,
    configure_cuda_visible_devices,
    load_runtime_env,
    runtime_env_snapshot,
)
from ..runtime_pause import PauseController


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autocapture benchmark harness")
    parser.add_argument("--cpu", action="store_true", help="Run CPU benchmark")
    parser.add_argument("--gpu", action="store_true", help="Run GPU benchmark")
    parser.add_argument("--both", action="store_true", help="Run both CPU and GPU benchmarks")
    parser.add_argument("--out", default=None, help="Output path override")
    parser.add_argument(
        "--json-name", default=None, help="Override JSON output filename (within bench dir)"
    )
    parser.add_argument("--config", default=None, help="Config path override")
    parser.add_argument(
        "--ignore-pause",
        action="store_true",
        help="Run even if pause latch is active",
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args(argv)
    if not (args.cpu or args.gpu or args.both):
        args.cpu = True
    if args.both:
        args.cpu = True
        args.gpu = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    runtime_env = load_runtime_env()
    configure_cuda_visible_devices(runtime_env)
    config = _load_config(args.config)
    pause, device_manager = _build_runtime(runtime_env, config)
    if args.ignore_pause:
        device_manager = DeviceManager(runtime_env, pause_controller=None)

    if pause.is_paused() and not args.ignore_pause:
        print("Pause latch active; waiting for resume...", file=sys.stderr)
        pause.wait_until_resumed()

    results: list[tuple[str, dict[str, Any]]] = []
    exit_code = 0

    if args.cpu:
        results.append(
            (
                "cpu",
                run_cpu_bench(
                    runtime_env,
                    config,
                    pause,
                    args.iterations,
                    args.warmup,
                    ignore_pause=args.ignore_pause,
                ),
            )
        )

    if args.gpu:
        try:
            results.append(
                (
                    "gpu",
                    run_gpu_bench(
                        runtime_env,
                        config,
                        pause,
                        device_manager,
                        args.iterations,
                        args.warmup,
                        ignore_pause=args.ignore_pause,
                    ),
                )
            )
        except RuntimeError as exc:
            exit_code = 2
            results.append(("gpu", _error_result(runtime_env, config, pause, str(exc))))

    for mode, payload in results:
        out_path = _resolve_output_path(runtime_env, mode, args.out, args.json_name)
        _atomic_write_json(out_path, payload)
        print(f"Wrote {mode} bench results to {out_path}")

    return exit_code


def _load_config(path_override: str | None) -> AppConfig | None:
    path = path_override or os.environ.get("AUTOCAPTURE_CONFIG")
    if path is None:
        default_path = default_config_path()
        if default_path.exists():
            path = str(default_path)
    if path is None:
        return None
    try:
        return load_config(Path(path))
    except FileNotFoundError:
        return None


def _build_runtime(
    runtime_env: RuntimeEnvConfig, config: AppConfig | None
) -> tuple[PauseController, DeviceManager]:
    if config is not None:
        context = build_runtime_context(config, runtime_env)
        return context.pause, context.device
    pause = PauseController(
        runtime_env.pause_latch_path,
        runtime_env.pause_reason_path,
        poll_interval_s=0.5,
        redact_window_titles=runtime_env.redact_window_titles,
    )
    return pause, DeviceManager(runtime_env, pause)


def run_cpu_bench(
    runtime_env: RuntimeEnvConfig,
    config: AppConfig | None,
    pause: PauseController,
    iterations: int,
    warmup: int,
    *,
    ignore_pause: bool = False,
) -> dict[str, Any]:
    fixture = Path(__file__).resolve().parent / "fixtures" / "sample_text.txt"
    seed = _seed_from_fixture(fixture)
    rng = np.random.default_rng(seed)
    inputs = [rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8) for _ in range(iterations)]

    for _ in range(max(0, warmup)):
        if not ignore_pause:
            pause.wait_until_resumed(timeout=None)
        hash_rgb_image(inputs[0])

    latencies_ms: list[float] = []
    start = time.perf_counter()
    for item in inputs:
        if not ignore_pause:
            pause.wait_until_resumed(timeout=None)
        t0 = time.perf_counter()
        _ = hash_rgb_image(item)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    total = time.perf_counter() - start

    return _build_result(
        runtime_env,
        config,
        pause,
        "cpu",
        latencies_ms,
        total,
        backend_info={"workload": "hash_rgb_image"},
    )


def run_gpu_bench(
    runtime_env: RuntimeEnvConfig,
    config: AppConfig | None,
    pause: PauseController,
    device_manager: DeviceManager,
    iterations: int,
    warmup: int,
    *,
    ignore_pause: bool = False,
) -> dict[str, Any]:
    selection = device_manager.select_device()
    if selection.device_kind != DeviceKind.CUDA:
        return _skipped_result(runtime_env, config, pause, "cuda_unavailable")

    if not _torch_available():
        return _skipped_result(runtime_env, config, pause, "torch_missing")

    import torch  # type: ignore

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device(selection.torch_device or "cuda:0")
    size = 512
    a = torch.randn((size, size), device=device)
    b = torch.randn((size, size), device=device)

    for _ in range(max(0, warmup)):
        if not ignore_pause:
            pause.wait_until_resumed(timeout=None)
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()

    latencies_ms: list[float] = []
    start = time.perf_counter()
    for _ in range(iterations):
        if not ignore_pause:
            pause.wait_until_resumed(timeout=None)
        t0 = time.perf_counter()
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    total = time.perf_counter() - start

    device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else None
    return _build_result(
        runtime_env,
        config,
        pause,
        "gpu",
        latencies_ms,
        total,
        backend_info={
            "workload": "torch.matmul",
            "device": str(device),
            "device_name": device_name,
        },
    )


def _build_result(
    runtime_env: RuntimeEnvConfig,
    config: AppConfig | None,
    pause: PauseController,
    mode: str,
    latencies_ms: list[float],
    total_seconds: float,
    *,
    backend_info: dict[str, object],
) -> dict[str, Any]:
    processed = len(latencies_ms)
    throughput = processed / total_seconds if total_seconds > 0 else 0.0
    return {
        "schema_version": 1,
        "status": "ok",
        "mode": mode,
        "metrics": {
            "p50_ms": _percentile(latencies_ms, 0.5),
            "p95_ms": _percentile(latencies_ms, 0.95),
            "throughput_per_s": throughput,
            "iterations": processed,
        },
        "counters": {"processed": processed, "skipped": 0, "errors": 0},
        "env": _environment_snapshot(),
        "config": _config_snapshot(runtime_env, config, pause),
        "backend": backend_info,
    }


def _skipped_result(
    runtime_env: RuntimeEnvConfig,
    config: AppConfig | None,
    pause: PauseController,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "skipped",
        "mode": "gpu",
        "reason": reason,
        "metrics": None,
        "counters": {"processed": 0, "skipped": 1, "errors": 0},
        "env": _environment_snapshot(),
        "config": _config_snapshot(runtime_env, config, pause),
        "backend": {"cuda_available": cuda_available()},
    }


def _error_result(
    runtime_env: RuntimeEnvConfig,
    config: AppConfig | None,
    pause: PauseController,
    message: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "error",
        "mode": "gpu",
        "reason": message,
        "metrics": None,
        "counters": {"processed": 0, "skipped": 0, "errors": 1},
        "env": _environment_snapshot(),
        "config": _config_snapshot(runtime_env, config, pause),
        "backend": {"cuda_available": cuda_available()},
    }


def _config_snapshot(
    runtime_env: RuntimeEnvConfig,
    config: AppConfig | None,
    pause: PauseController,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "runtime_env": runtime_env_snapshot(runtime_env),
        "pause_state": asdict(pause.get_state()),
    }
    if config is None:
        snapshot["status"] = "config_missing"
        return snapshot
    snapshot.update(
        {
            "workers": {
                "ocr": config.worker.ocr_workers,
                "embed": config.worker.embed_workers,
                "agents": config.worker.agent_workers,
            },
            "batch_sizes": {
                "ocr": config.ocr.batch_size,
                "embed": config.embed.text_batch_size,
                "reranker_active": config.reranker.batch_size_active,
                "reranker_idle": config.reranker.batch_size_idle,
            },
            "qos_profiles": {
                "active": {
                    "ocr_workers": config.runtime.qos.profile_active.ocr_workers,
                    "embed_workers": config.runtime.qos.profile_active.embed_workers,
                    "agent_workers": config.runtime.qos.profile_active.agent_workers,
                },
                "idle": {
                    "ocr_workers": config.runtime.qos.profile_idle.ocr_workers,
                    "embed_workers": config.runtime.qos.profile_idle.embed_workers,
                    "agent_workers": config.runtime.qos.profile_idle.agent_workers,
                },
            },
        }
    )
    return snapshot


def _environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "total_ram_bytes": _total_ram_bytes(),
        "git_sha": _git_sha(),
    }


def _total_ram_bytes() -> int | None:
    try:
        if hasattr(os, "sysconf"):
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        return None
    return None


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _seed_from_fixture(path: Path) -> int:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        data = b"autocapture-bench-default"
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:4], "big")


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * quantile
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = idx - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def _resolve_output_path(
    runtime_env: RuntimeEnvConfig,
    mode: str,
    out_override: str | None,
    json_name: str | None,
) -> Path:
    if out_override:
        return Path(out_override)
    filename = json_name or f"bench_{mode}.json"
    return runtime_env.bench_output_dir / filename


def _torch_available() -> bool:
    return bool(importlib.util.find_spec("torch"))


if __name__ == "__main__":
    raise SystemExit(main())
