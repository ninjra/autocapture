"""Deterministic time intent parsing for memory queries."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

from dateutil import tz


@dataclass(frozen=True)
class TimeIntent:
    time_expression: str | None
    time_range: tuple[dt.datetime, dt.datetime] | None


def resolve_timezone(timezone: str | None) -> dt.tzinfo:
    if timezone:
        resolved = tz.gettz(timezone)
        if resolved is not None:
            return resolved
    return tz.tzlocal()


def parse_time_expression(
    expression: str, *, now: dt.datetime, tzinfo: dt.tzinfo
) -> tuple[dt.datetime, dt.datetime] | None:
    if not expression:
        return None
    text = " ".join(expression.strip().lower().split())
    if not text:
        return None

    if re.search(r"\b(last|past)\s+hour\b", text) or re.search(r"\ban?\s+hour\s+ago\b", text):
        start = now - dt.timedelta(hours=1)
        return _to_utc_range(start, now)

    patterns = [
        r"\byesterday\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
        r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s+yesterday\b",
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s+yesterday\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = match.group(3)
        if meridiem:
            if meridiem == "pm" and hour < 12:
                hour += 12
            if meridiem == "am" and hour == 12:
                hour = 0
        if hour > 23 or minute > 59:
            continue
        local_now = now.astimezone(tzinfo)
        day = local_now.date() - dt.timedelta(days=1)
        start_local = dt.datetime.combine(day, dt.time(hour, minute), tzinfo=tzinfo)
        end_local = start_local + dt.timedelta(hours=1)
        return _to_utc_range(start_local, end_local)

    return None


def parse_time_range_payload(
    payload: dict[str, str] | None, *, tzinfo: dt.tzinfo
) -> tuple[dt.datetime, dt.datetime] | None:
    if not payload:
        return None
    start_iso = payload.get("start_iso")
    end_iso = payload.get("end_iso")
    if not start_iso or not end_iso:
        return None
    start = _parse_iso(start_iso, tzinfo)
    end = _parse_iso(end_iso, tzinfo)
    if not start or not end:
        return None
    return _to_utc_range(start, end)


def resolve_time_intent(
    *,
    time_expression: str | None,
    time_range_payload: dict[str, str] | None,
    now: dt.datetime,
    tzinfo: dt.tzinfo,
) -> TimeIntent:
    time_range = parse_time_range_payload(time_range_payload, tzinfo=tzinfo)
    if time_range is None and time_expression:
        time_range = parse_time_expression(time_expression, now=now, tzinfo=tzinfo)
    return TimeIntent(time_expression=time_expression, time_range=time_range)


def resolve_time_range_for_query(
    *,
    query: str,
    time_range: tuple[dt.datetime, dt.datetime] | None,
    now: dt.datetime,
    tzinfo: dt.tzinfo,
) -> tuple[dt.datetime, dt.datetime] | None:
    normalized = normalize_time_range(time_range, tzinfo=tzinfo)
    if normalized is not None:
        return normalized
    if query:
        return parse_time_expression(query, now=now, tzinfo=tzinfo)
    return None


def normalize_time_range(
    time_range: tuple[dt.datetime, dt.datetime] | None, *, tzinfo: dt.tzinfo
) -> tuple[dt.datetime, dt.datetime] | None:
    if not time_range:
        return None
    start, end = time_range
    if start.tzinfo is None:
        start = start.replace(tzinfo=tzinfo)
    if end.tzinfo is None:
        end = end.replace(tzinfo=tzinfo)
    return _to_utc_range(start, end)


def _parse_iso(value: str, tzinfo: dt.tzinfo) -> dt.datetime | None:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=tzinfo)
    return parsed


def _to_utc_range(
    start: dt.datetime, end: dt.datetime
) -> tuple[dt.datetime, dt.datetime]:
    return (
        start.astimezone(dt.timezone.utc),
        end.astimezone(dt.timezone.utc),
    )
