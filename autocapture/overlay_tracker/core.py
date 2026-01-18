"""Pure overlay tracker domain logic."""

from __future__ import annotations

import datetime as dt
import unicodedata
from urllib.parse import urlparse

from ..capture.privacy_filter import normalize_process_name
from ..config import OverlayTrackerConfig
from .schemas import OverlayIdentity


def hotness(last_activity_utc: dt.datetime, now_utc: dt.datetime, half_life_s: float) -> float:
    if half_life_s <= 0:
        return 0.0
    age = (now_utc - last_activity_utc).total_seconds()
    if age <= 0:
        return 1.0
    return 2 ** (-age / half_life_s)


def is_stale(last_activity_utc: dt.datetime, now_utc: dt.datetime, stale_after_s: float) -> bool:
    if stale_after_s <= 0:
        return False
    age = (now_utc - last_activity_utc).total_seconds()
    return age > stale_after_s


def normalize_title(raw_title: str | None, *, max_len: int) -> str:
    if not raw_title:
        return ""
    value = unicodedata.normalize("NFKC", raw_title)
    value = " ".join(value.split())
    value = value.strip()
    if max_len > 0 and len(value) > max_len:
        value = value[:max_len]
    return value


def should_deny_process(process_name: str | None, deny_list: list[str]) -> bool:
    normalized = normalize_process_name(process_name)
    if not normalized:
        return False
    return normalized in {normalize_process_name(name) for name in deny_list}


def resolve_identity(
    config: OverlayTrackerConfig,
    *,
    process_name: str,
    window_title: str | None,
    browser_url: str | None,
) -> OverlayIdentity:
    normalized_title = normalize_title(
        window_title,
        max_len=config.policy.max_window_title_length,
    )
    title_identity = _title_identity(process_name, normalized_title)
    if not config.url_plugin.enabled:
        return title_identity
    try:
        if not _allow_browser_process(process_name, config.url_plugin.allow_browsers):
            return title_identity
        if not browser_url:
            return title_identity
        parsed = _parse_url(browser_url)
        if not parsed:
            return title_identity
        domain, segments = parsed
        allowed_domains = {d.lower() for d in config.url_plugin.allow_domains}
        if allowed_domains and domain not in allowed_domains:
            return title_identity
        token = _token_from_rules(domain, segments, config.url_plugin.token_rules)
        if not token:
            token = _token_default(domain, segments)
        if not token:
            return title_identity
        return OverlayIdentity(identity_type="url", identity_key=f"{domain}/{token}")
    except Exception:
        return title_identity


def _title_identity(process_name: str, normalized_title: str) -> OverlayIdentity:
    normalized_process = normalize_process_name(process_name) or process_name.strip().casefold()
    if normalized_title:
        return OverlayIdentity(
            identity_type="title",
            identity_key=f"{normalized_process}:{normalized_title}",
        )
    return OverlayIdentity(identity_type="title", identity_key=normalized_process)


def _allow_browser_process(process_name: str, allow_browsers: list[str]) -> bool:
    if not allow_browsers:
        return False
    normalized = normalize_process_name(process_name) or process_name.strip().casefold()
    allow = {normalize_process_name(name) for name in allow_browsers}
    return normalized in allow


def _parse_url(url: str) -> tuple[str, list[str]] | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    domain = (parsed.hostname or "").lower()
    if not domain:
        return None
    segments = [seg for seg in (parsed.path or "").split("/") if seg]
    return domain, segments


def _token_from_rules(domain: str, segments: list[str], rules: list[dict]) -> str | None:
    if not rules:
        return None
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if (rule.get("domain") or "").lower() != domain:
            continue
        count = rule.get("path_segments")
        if isinstance(count, int) and count > 0:
            if len(segments) >= count:
                return "/".join(segments[:count])
            return None
    return None


def _token_default(domain: str, segments: list[str]) -> str | None:
    if domain == "github.com" and len(segments) >= 2:
        return "/".join(segments[:2])
    if domain == "chatgpt.com" and len(segments) >= 2 and segments[0] in {"c", "share"}:
        return "/".join(segments[:2])
    if segments:
        return segments[0]
    return None
