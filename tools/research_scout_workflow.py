"""Workflow helper for scheduled research scout runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from autocapture.config import AppConfig
from autocapture.research.scout import append_report_log, run_scout, write_report


def _load_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _top_ids(report: dict[str, Any], top_n: int) -> list[str]:
    items = report.get("ranked_items") or []
    if not isinstance(items, list):
        return []
    ids: list[str] = []
    for item in items[:top_n]:
        if not isinstance(item, dict):
            continue
        ident = item.get("id") or item.get("url")
        if isinstance(ident, str) and ident:
            ids.append(ident)
    return ids


def compute_diff(
    old_report: dict[str, Any] | None, new_report: dict[str, Any], top_n: int
) -> dict[str, Any]:
    new_ids = _top_ids(new_report, top_n)
    if not old_report:
        return {"changed": len(new_ids), "ratio": 1.0, "top_n": top_n}
    old_ids = _top_ids(old_report, top_n)
    if not new_ids:
        return {"changed": 0, "ratio": 0.0, "top_n": top_n}
    old_set = set(old_ids)
    changed = sum(1 for ident in new_ids if ident not in old_set)
    ratio = changed / max(1, top_n)
    return {"changed": changed, "ratio": ratio, "top_n": top_n}


def _report_is_online(report: dict[str, Any]) -> bool:
    if report.get("offline"):
        return False
    sources = report.get("sources") or {}
    for payload in sources.values():
        status = payload.get("status") if isinstance(payload, dict) else None
        if status in {"error", "offline"}:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run research scout with diff threshold.")
    parser.add_argument("--out", default="docs/research/scout_report.json")
    parser.add_argument("--log", default="docs/research/scout_log.md")
    parser.add_argument("--diff-threshold", type=float, default=0.3)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--meta", default=None)
    args = parser.parse_args()

    config = AppConfig()
    config.offline = False
    config.privacy.cloud_enabled = True
    runner_temp = os.environ.get("AUTOCAPTURE_SCOUT_DATA_DIR") or os.environ.get("RUNNER_TEMP")
    if runner_temp:
        data_root = Path(runner_temp) / "autocapture_scout"
        data_root.mkdir(parents=True, exist_ok=True)
        config.capture.data_dir = data_root
        config.capture.staging_dir = data_root / "staging"

    out_path = Path(args.out)
    old_report = _load_report(out_path)
    new_report = run_scout(config)
    diff = compute_diff(old_report, new_report, args.top_n)
    threshold_exceeded = (
        diff["ratio"] >= args.diff_threshold if _report_is_online(new_report) else False
    )

    write_report(new_report, out_path)
    if threshold_exceeded and args.log:
        append_report_log(new_report, Path(args.log))

    if args.meta:
        meta_path = Path(args.meta)
        meta_payload = {
            "threshold_exceeded": threshold_exceeded,
            "diff": diff,
            "online": _report_is_online(new_report),
        }
        meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
