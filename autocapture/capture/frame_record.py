"""Frame record builders shared across capture entrypoints."""

from __future__ import annotations

import datetime as dt
import time
from typing import Optional

from ..config import PrivacyConfig
from ..contracts import FrameRecordV1, PrivacyFlags


def build_privacy_flags(
    privacy: PrivacyConfig,
    *,
    excluded: bool,
    masked_regions_applied: bool,
    capture_paused: bool,
    offline: bool,
) -> PrivacyFlags:
    cloud_allowed = bool(privacy.cloud_enabled and privacy.allow_cloud_images and not offline)
    return PrivacyFlags(
        excluded=excluded,
        masked_regions_applied=masked_regions_applied,
        cloud_allowed=cloud_allowed,
        capture_paused=capture_paused,
    )


def build_frame_record_v1(
    *,
    frame_id: str,
    captured_at: dt.datetime,
    monotonic_ts: Optional[float],
    monitor_id: Optional[str],
    monitor_bounds: Optional[tuple[int, int, int, int]],
    app_name: Optional[str],
    window_title: Optional[str],
    image_path: Optional[str],
    privacy_flags: PrivacyFlags,
    frame_hash: Optional[str],
    event_id: Optional[str] = None,
) -> FrameRecordV1:
    if monotonic_ts is None:
        monotonic_ts = time.monotonic()
    bounds_list = list(monitor_bounds) if monitor_bounds else None
    return FrameRecordV1(
        frame_id=frame_id,
        event_id=event_id,
        created_at_utc=captured_at,
        monotonic_ts=float(monotonic_ts),
        monitor_id=monitor_id,
        monitor_bounds=bounds_list,
        app_name=app_name,
        window_title=window_title,
        active_window=window_title,
        image_path=image_path,
        blob_ref=None,
        privacy_flags=privacy_flags,
        frame_hash=frame_hash,
        phash=None,
    )


def capture_record_kwargs(
    *,
    frame: FrameRecordV1,
    captured_at: dt.datetime,
    image_path: Optional[str],
    focus_path: Optional[str],
    foreground_process: Optional[str],
    foreground_window: Optional[str],
    url: Optional[str] = None,
    domain: Optional[str] = None,
    monitor_id: str,
    is_fullscreen: bool,
    ocr_status: str,
    ocr_last_error: Optional[str] = None,
) -> dict:
    return {
        "id": frame.frame_id,
        "event_id": frame.event_id,
        "captured_at": captured_at,
        "created_at_utc": frame.created_at_utc,
        "monotonic_ts": frame.monotonic_ts,
        "image_path": image_path,
        "focus_path": focus_path,
        "foreground_process": foreground_process or "unknown",
        "foreground_window": foreground_window or "unknown",
        "url": url,
        "domain": domain,
        "monitor_id": monitor_id,
        "monitor_bounds": frame.monitor_bounds,
        "is_fullscreen": is_fullscreen,
        "privacy_flags": frame.privacy_flags.model_dump(),
        "frame_hash": frame.frame_hash,
        "schema_version": frame.schema_version,
        "ocr_status": ocr_status,
        "ocr_last_error": ocr_last_error,
    }
