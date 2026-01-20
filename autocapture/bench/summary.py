"""Benchmark summary utilities."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def parse_timings(path: Path) -> list[float]:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if "\t" in stripped:
            parts = stripped.split("\t")
        elif "," in stripped:
            parts = stripped.split(",")
        else:
            parts = stripped.split()
        if not parts:
            continue
        try:
            value = float(parts[-1])
        except ValueError:
            continue
        values.append(value)
    return values


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * quantile
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    fraction = idx - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def build_summary(case_id: str, mode: str, values: list[float]) -> dict:
    stats = {
        "min_ms": min(values) if values else 0.0,
        "median_ms": percentile(values, 0.5),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values) if values else 0.0,
    }
    meta = {
        "tool": "autocapture.bench",
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "arch": platform.machine(),
        "python": platform.python_version(),
    }
    return {
        "case_id": case_id,
        "mode": mode,
        "unit": "ms",
        "n": len(values),
        "stats": stats,
        "meta": meta,
    }


def write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize bench timing TSV")
    parser.add_argument("--input", required=True, help="Input timing TSV/CSV path")
    parser.add_argument("--output", required=True, help="Output summary JSON path")
    parser.add_argument("--case-id", required=True, help="Case id")
    parser.add_argument("--mode", default="offline", help="Mode (offline|live)")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    values = parse_timings(Path(args.input))
    summary = build_summary(args.case_id, args.mode, values)
    write_summary(summary, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
