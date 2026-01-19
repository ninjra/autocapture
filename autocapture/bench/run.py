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
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .schema import BenchResult
from ..config import AppConfig, load_config
from ..image_utils import hash_rgb_image
from ..paths import default_config_path
from ..runtime_context import build_runtime_context
from ..runtime_device import DeviceManager, GpuRequiredError
from ..runtime_env import RuntimeEnvConfig, load_runtime_env
from ..runtime_pause import PauseController


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autocapture benchmark harness")
    parser.add_argument(
        "--mode",
        choices=("cpu", "gpu", "both"),
        default=None,
        help="Benchmark mode (cpu, gpu, both).",
    )
    parser.add_argument("--cpu", action="store_true", help="Run CPU benchmark (legacy)")
    parser.add_argument("--gpu", action="store_true", help="Run GPU benchmark (legacy)")
    parser.add_argument("--both", action="store_true", help="Run both benchmarks (legacy)")
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
    if args.mode is None:
        if args.both:
            args.mode = "both"
        elif args.gpu:
            args.mode = "gpu"
        elif args.cpu:
            args.mode = "cpu"
        else:
            args.mode = "cpu"
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    runtime_env = load_runtime_env()
    config = _load_config(args.config)
    pause, device_manager = _build_runtime(runtime_env, config)
    if args.ignore_pause:
        device_manager = DeviceManager(runtime_env, pause_controller=None)

    if pause.is_paused() and not args.ignore_pause:
        print("Pause latch active; waiting for resume...", file=sys.stderr)
        pause.wait_until_resumed()

    results: list[tuple[str, BenchResult]] = []
    exit_code = 0

    modes = ["cpu", "gpu"] if args.mode == "both" else [args.mode]
    for mode in modes:
        if mode == "cpu":
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
            continue
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
        except GpuRequiredError as exc:
            exit_code = 2
            results.append(("gpu", _error_result(runtime_env, str(exc), exc.provider_info)))
        except Exception as exc:
            exit_code = 2
            results.append(("gpu", _error_result(runtime_env, str(exc), {})))

    for mode, payload in results:
        out_path = _resolve_output_path(runtime_env, mode, payload.run_id, args.out, args.json_name)
        _atomic_write_json(out_path, payload.to_dict())
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
) -> BenchResult:
    _ = config
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

    return _build_result(runtime_env, latencies_ms, total, notes="")


def run_gpu_bench(
    runtime_env: RuntimeEnvConfig,
    config: AppConfig | None,
    pause: PauseController,
    device_manager: DeviceManager,
    iterations: int,
    warmup: int,
    *,
    ignore_pause: bool = False,
) -> BenchResult:
    _ = config
    compute_device, provider_info = device_manager.resolve_compute_device()
    if compute_device != "cuda":
        reason = provider_info.get("reason") if isinstance(provider_info, dict) else None
        note = f"gpu requested but {reason}" if reason else "gpu requested but CUDA unavailable"
        return _skipped_result(runtime_env, note, provider_info, iterations)

    if not _torch_available():
        return _skipped_result(
            runtime_env, "gpu requested but torch missing", provider_info, iterations
        )

    import torch  # type: ignore

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    selection = device_manager.select_device()
    device_str = selection.torch_device or selection.compute_device
    device = torch.device(device_str)
    size = 512
    with device_manager.with_gpu_env():
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
    provider_info = dict(provider_info)
    if device_name:
        provider_info.setdefault("gpu_name", device_name)
    return _build_result(runtime_env, latencies_ms, total, notes="", provider_info=provider_info)


def _build_result(
    runtime_env: RuntimeEnvConfig,
    latencies_ms: list[float],
    total_seconds: float,
    *,
    notes: str,
    provider_info: dict[str, object] | None = None,
    errors: int = 0,
    skipped: int = 0,
) -> BenchResult:
    processed = len(latencies_ms)
    throughput = processed / total_seconds if total_seconds > 0 else 0.0
    metrics = {
        "latency_ms_p50": _percentile(latencies_ms, 0.5),
        "latency_ms_p95": _percentile(latencies_ms, 0.95),
        "throughput_items_per_s": throughput,
        "processed": processed,
        "errors": errors,
        "skipped": skipped,
    }
    return BenchResult(
        run_id=_run_id(),
        git_sha=_git_sha() or "unknown",
        ts_ms=_now_ms(),
        env=_environment_snapshot(provider_info),
        config=_config_snapshot(runtime_env),
        metrics=metrics,
        notes=notes,
    )


def _skipped_result(
    runtime_env: RuntimeEnvConfig,
    reason: str,
    provider_info: dict[str, object] | None,
    skipped: int,
) -> BenchResult:
    return _build_result(
        runtime_env,
        [],
        0.0,
        notes=reason,
        provider_info=provider_info,
        skipped=skipped,
    )


def _error_result(
    runtime_env: RuntimeEnvConfig,
    message: str,
    provider_info: dict[str, object] | None,
) -> BenchResult:
    return _build_result(
        runtime_env,
        [],
        0.0,
        notes=message,
        provider_info=provider_info,
        errors=1,
    )


def _config_snapshot(
    runtime_env: RuntimeEnvConfig,
) -> dict[str, Any]:
    tuning = (
        runtime_env.idle_tuning
        if runtime_env.profile.value == "idle"
        else runtime_env.foreground_tuning
    )
    return {
        "gpu_mode": runtime_env.gpu_mode.value,
        "profile": runtime_env.profile.value,
        "tuning": tuning.as_dict(),
    }


def _environment_snapshot(provider_info: dict[str, object] | None) -> dict[str, Any]:
    gpu_name = None
    if provider_info:
        name = provider_info.get("gpu_name")
        if isinstance(name, str):
            gpu_name = name
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": _cpu_name(),
        "ram_gb": _ram_gb(),
        "gpu_name": gpu_name,
    }


def _now_ms() -> int:
    return int(time.time() * 1000)


def _run_id() -> str:
    return uuid.uuid4().hex


def _cpu_name() -> str | None:
    name = platform.processor() or ""
    if name:
        return name
    try:
        if Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if "model name" in line:
                    return line.split(":", 1)[-1].strip() or None
    except Exception:
        return None
    return None


def _ram_gb() -> float | None:
    try:
        import psutil  # type: ignore

        total = getattr(psutil.virtual_memory(), "total", None)
        if total:
            return round(total / (1024**3), 2)
    except Exception:
        pass
    try:
        if Path("/proc/meminfo").exists():
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        kb = int(parts[1])
                        return round(kb / (1024**2), 2)
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


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
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
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _resolve_output_path(
    runtime_env: RuntimeEnvConfig,
    mode: str,
    run_id: str,
    out_override: str | None,
    json_name: str | None,
) -> Path:
    if out_override:
        out_path = Path(out_override)
        if out_path.is_dir() or out_override.endswith(("/", "\\")):
            filename = json_name or f"bench_{mode}_{run_id}.json"
            return out_path / filename
        return out_path
    filename = json_name or f"bench_{mode}_{run_id}.json"
    return runtime_env.bench_output_dir / filename


def _torch_available() -> bool:
    return bool(importlib.util.find_spec("torch"))


if __name__ == "__main__":
    raise SystemExit(main())
