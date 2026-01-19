"""No-evidence response helpers."""

from __future__ import annotations

from typing import Any


def build_no_evidence_payload(query: str, *, has_time_range: bool) -> dict[str, Any]:
    message = (
        "No evidence found in your captured data for the query."
        if query
        else "No evidence found in your captured data."
    )
    if has_time_range:
        message = "No evidence found in your captured data for the selected time range."
    hints = []
    actions = [
        {"type": "time_range", "label": "Try last 24 hours", "value": "24h"},
        {"type": "time_range", "label": "Try last 7 days", "value": "7d"},
        {"type": "time_range", "label": "Try last 30 days", "value": "30d"},
        {"type": "refine_query", "label": "Add app or keyword filters", "value": query},
    ]
    return {"message": message, "hints": hints, "actions": actions}
