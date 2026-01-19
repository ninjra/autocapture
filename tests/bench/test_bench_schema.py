from __future__ import annotations

import json

from autocapture.bench import run as bench_run
from autocapture.bench.schema import BenchResult


def test_bench_schema(tmp_path, monkeypatch) -> None:
    bench_dir = tmp_path / "bench"
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setenv("AUTOCAPTURE_BENCH_DIR", str(bench_dir))
    monkeypatch.setenv("AUTOCAPTURE_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("AUTOCAPTURE_GPU_MODE", "off")
    monkeypatch.setenv("AUTOCAPTURE_PROFILE", "foreground")

    exit_code = bench_run.main(["--mode", "cpu", "--iterations", "5", "--warmup", "1"])
    assert exit_code == 0

    files = list(bench_dir.glob("bench_cpu_*.json"))
    assert files
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    BenchResult.validate_dict(payload)
