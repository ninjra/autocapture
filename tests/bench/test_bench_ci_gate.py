from __future__ import annotations

import json
from pathlib import Path

from autocapture.bench import ci_gate


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_ci_gate_passes_on_improvement(tmp_path: Path) -> None:
    baseline = {
        "cases": [
            {
                "case_id": "case1",
                "mode": "offline",
                "stats": {"median_ms": 100.0, "p95_ms": 200.0},
            }
        ]
    }
    current = {
        "case_id": "case1",
        "mode": "offline",
        "stats": {"median_ms": 90.0, "p95_ms": 180.0},
    }
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    _write_json(baseline_path, baseline)
    _write_json(current_path, current)

    exit_code = ci_gate.main(
        [
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--case-id",
            "case1",
            "--mode",
            "offline",
        ]
    )
    assert exit_code == 0


def test_ci_gate_fails_on_regression(tmp_path: Path) -> None:
    baseline = {
        "cases": [
            {
                "case_id": "case1",
                "mode": "offline",
                "stats": {"median_ms": 100.0, "p95_ms": 200.0},
            }
        ],
        "thresholds": {
            "median_pct": 0.1,
            "median_abs_ms": 10.0,
            "p95_pct": 0.1,
            "p95_abs_ms": 10.0,
        },
    }
    current = {
        "case_id": "case1",
        "mode": "offline",
        "stats": {"median_ms": 150.0, "p95_ms": 260.0},
    }
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    _write_json(baseline_path, baseline)
    _write_json(current_path, current)

    exit_code = ci_gate.main(
        [
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--case-id",
            "case1",
            "--mode",
            "offline",
        ]
    )
    assert exit_code == 2
