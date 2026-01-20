"""CI regression gate for bench summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DEFAULT_THRESHOLDS = {
    "median_pct": 0.10,
    "median_abs_ms": 100.0,
    "p95_pct": 0.15,
    "p95_abs_ms": 150.0,
}


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _extract_case(payload: dict, case_id: str, mode: str) -> dict | None:
    if "cases" in payload and isinstance(payload["cases"], list):
        for case in payload["cases"]:
            if not isinstance(case, dict):
                continue
            if case.get("case_id") == case_id and case.get("mode") == mode:
                return case
        return None
    if payload.get("case_id") == case_id and payload.get("mode") == mode:
        return payload
    return None


def _regression(current: float, baseline: float, pct: float, abs_ms: float) -> bool:
    delta = current - baseline
    if delta <= abs_ms:
        return False
    if baseline <= 0:
        return True
    return (delta / baseline) > pct


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bench regression gate")
    parser.add_argument("--baseline", required=True, help="Baseline JSON path")
    parser.add_argument("--current", required=True, help="Current summary JSON path")
    parser.add_argument("--case-id", default="case1", help="Case id")
    parser.add_argument("--mode", default="offline", help="Mode")
    parser.add_argument("--median-pct", type=float, default=None, help="Median pct threshold")
    parser.add_argument("--median-abs-ms", type=float, default=None, help="Median abs threshold")
    parser.add_argument("--p95-pct", type=float, default=None, help="P95 pct threshold")
    parser.add_argument("--p95-abs-ms", type=float, default=None, help="P95 abs threshold")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    baseline_payload = _load_json(Path(args.baseline))
    current_payload = _load_json(Path(args.current))

    thresholds = dict(_DEFAULT_THRESHOLDS)
    thresholds.update(baseline_payload.get("thresholds", {}))
    if args.median_pct is not None:
        thresholds["median_pct"] = args.median_pct
    if args.median_abs_ms is not None:
        thresholds["median_abs_ms"] = args.median_abs_ms
    if args.p95_pct is not None:
        thresholds["p95_pct"] = args.p95_pct
    if args.p95_abs_ms is not None:
        thresholds["p95_abs_ms"] = args.p95_abs_ms

    baseline_case = _extract_case(baseline_payload, args.case_id, args.mode)
    current_case = _extract_case(current_payload, args.case_id, args.mode)

    if baseline_case is None:
        raise SystemExit(f"Baseline missing case {args.case_id}/{args.mode}")
    if current_case is None:
        raise SystemExit(f"Current summary missing case {args.case_id}/{args.mode}")

    baseline_stats = baseline_case.get("stats", {})
    current_stats = current_case.get("stats", {})
    baseline_median = float(baseline_stats.get("median_ms", 0.0))
    baseline_p95 = float(baseline_stats.get("p95_ms", 0.0))
    current_median = float(current_stats.get("median_ms", 0.0))
    current_p95 = float(current_stats.get("p95_ms", 0.0))

    median_regression = _regression(
        current_median,
        baseline_median,
        thresholds["median_pct"],
        thresholds["median_abs_ms"],
    )
    p95_regression = _regression(
        current_p95,
        baseline_p95,
        thresholds["p95_pct"],
        thresholds["p95_abs_ms"],
    )

    if median_regression or p95_regression:
        print("Benchmark regression detected.")
        print(f"Median: baseline={baseline_median:.2f}ms current={current_median:.2f}ms")
        print(f"P95: baseline={baseline_p95:.2f}ms current={current_p95:.2f}ms")
        return 2

    print("Benchmark gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
