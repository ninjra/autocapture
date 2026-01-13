"""Pure privacy filtering helpers for capture pipelines."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from ..logging_utils import get_logger


_LOG = get_logger("capture.privacy_filter")
_WARNED_REGEX_PATTERNS: set[str] = set()


def normalize_process_name(name: str | None) -> str | None:
    if name is None:
        return None
    stripped = name.strip()
    if not stripped:
        return None
    return stripped.casefold()


def should_skip_capture(
    *,
    paused: bool,
    monitor_id: str | None,
    process_name: str | None,
    window_title: str | None,
    exclude_monitors: list[str],
    exclude_processes: list[str],
    exclude_window_title_regex: list[str],
) -> bool:
    if paused:
        return True
    if monitor_id and monitor_id in exclude_monitors:
        return True
    normalized_process = normalize_process_name(process_name)
    if normalized_process and normalized_process in _normalize_list(exclude_processes):
        return True
    if window_title:
        for pattern in exclude_window_title_regex:
            if not isinstance(pattern, str) or not pattern:
                continue
            try:
                if re.search(pattern, window_title):
                    return True
            except re.error as exc:
                _warn_invalid_regex(pattern, exc)
    return False


def apply_exclude_region_masks(
    roi: np.ndarray,
    *,
    monitor_id: str,
    roi_origin_x: int,
    roi_origin_y: int,
    exclude_regions: list[dict],
) -> np.ndarray:
    """Mask excluded rectangles in the ROI in-place.

    Returns the ROI array (which may have been mutated in-place).
    """

    if not exclude_regions:
        return roi
    roi_height, roi_width = roi.shape[:2]
    roi_left = roi_origin_x
    roi_top = roi_origin_y
    roi_right = roi_left + roi_width
    roi_bottom = roi_top + roi_height
    for entry in exclude_regions:
        if not isinstance(entry, dict):
            _LOG.debug("Ignoring malformed exclude region (not dict): %r", entry)
            continue
        entry_monitor = entry.get("monitor_id")
        if entry_monitor != monitor_id:
            continue
        try:
            region_x = int(entry["x"])
            region_y = int(entry["y"])
            region_w = int(entry["width"])
            region_h = int(entry["height"])
        except (KeyError, TypeError, ValueError) as exc:
            _LOG.warning("Ignoring malformed exclude region %r: %s", entry, exc)
            continue
        if region_w <= 0 or region_h <= 0:
            _LOG.debug("Ignoring empty exclude region: %r", entry)
            continue
        region_right = region_x + region_w
        region_bottom = region_y + region_h
        intersect_left = max(roi_left, region_x)
        intersect_top = max(roi_top, region_y)
        intersect_right = min(roi_right, region_right)
        intersect_bottom = min(roi_bottom, region_bottom)
        if intersect_right <= intersect_left or intersect_bottom <= intersect_top:
            continue
        local_x0 = max(0, intersect_left - roi_left)
        local_y0 = max(0, intersect_top - roi_top)
        local_x1 = min(roi_width, intersect_right - roi_left)
        local_y1 = min(roi_height, intersect_bottom - roi_top)
        roi[local_y0:local_y1, local_x0:local_x1] = 0
    return roi


def _normalize_list(values: Iterable[str]) -> set[str]:
    normalized = set()
    for value in values:
        normalized_value = normalize_process_name(value)
        if normalized_value:
            normalized.add(normalized_value)
    return normalized


def _warn_invalid_regex(pattern: str, exc: re.error) -> None:
    if pattern in _WARNED_REGEX_PATTERNS:
        return
    _WARNED_REGEX_PATTERNS.add(pattern)
    _LOG.warning("Invalid exclude regex %s: %s", pattern, exc)
