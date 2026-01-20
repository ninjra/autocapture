"""Hotness plugin for deterministic memory item ranking."""

from __future__ import annotations

import datetime as dt
import math
import re
import sqlite3

from ..config import MemoryHotnessConfig, MemoryHotnessQuotasConfig
from ..logging_utils import get_logger
from .models import (
    HotnessGcResult,
    HotnessPinResult,
    HotnessRankEntry,
    HotnessRankResult,
    HotnessStateResult,
    HotnessTouchResult,
    HotnessUnpinResult,
)
from .store import MemoryStore

_TS_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_PIN_LEVELS = {"hard", "soft"}
_BIN_ORDER = ("hot", "recent", "warm", "cool")


class HotnessPlugin:
    def __init__(self, store: MemoryStore, config: MemoryHotnessConfig) -> None:
        self._store = store
        self._config = config
        self._log = get_logger("memory.hotness")

    def touch(
        self,
        *,
        scope: str,
        item_id: str,
        ts_utc: str,
        signal: str,
        weight: float,
        source: str,
    ) -> HotnessTouchResult:
        scope = self._normalize_scope(scope)
        ts = _parse_ts(ts_utc)
        signal = signal.strip()
        source = source.strip()
        self._validate_signal_source(signal, source)
        if not item_id:
            raise ValueError("item_id is required")
        if not self._item_exists(item_id):
            raise ValueError("memory item not found")

        with self._store._connect() as conn:
            if self._rate_limited(conn, scope, item_id, source, ts):
                return HotnessTouchResult(
                    item_id=item_id,
                    scope=scope,
                    ts_utc=ts_utc,
                    signal=signal,
                    weight=float(weight),
                    source=source,
                    applied=False,
                    rate_limited=True,
                    reason="rate_limited",
                )

            row = conn.execute(
                """
                SELECT last_ts_utc, h_fast, h_mid, h_warm, h_cool
                FROM memory_hotness_state
                WHERE scope = ? AND item_id = ?
                """,
                (scope, item_id),
            ).fetchone()
            last_ts = _parse_ts(row["last_ts_utc"]) if row else None
            if last_ts and ts < last_ts:
                return HotnessTouchResult(
                    item_id=item_id,
                    scope=scope,
                    ts_utc=ts_utc,
                    signal=signal,
                    weight=float(weight),
                    source=source,
                    applied=False,
                    rate_limited=False,
                    reason="out_of_order",
                )

            if row:
                dt_seconds = max(0.0, (ts - last_ts).total_seconds()) if last_ts else 0.0
                h_fast = _decay(
                    float(row["h_fast"]), dt_seconds, self._config.half_lives.fast_seconds
                )
                h_mid = _decay(float(row["h_mid"]), dt_seconds, self._config.half_lives.mid_seconds)
                h_warm = _decay(
                    float(row["h_warm"]), dt_seconds, self._config.half_lives.warm_seconds
                )
                h_cool = _decay(
                    float(row["h_cool"]), dt_seconds, self._config.half_lives.cool_seconds
                )
            else:
                h_fast = h_mid = h_warm = h_cool = 0.0

            h_fast += float(weight)
            h_mid += float(weight)
            h_warm += float(weight)
            h_cool += float(weight)

            conn.execute(
                """
                INSERT INTO memory_hotness_events(
                    scope, item_id, ts_utc, signal, weight, source
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (scope, item_id, _format_ts(ts), signal, float(weight), source),
            )
            conn.execute(
                """
                INSERT INTO memory_hotness_state(
                    scope, item_id, last_ts_utc, h_fast, h_mid, h_warm, h_cool
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope, item_id) DO UPDATE SET
                    last_ts_utc = excluded.last_ts_utc,
                    h_fast = excluded.h_fast,
                    h_mid = excluded.h_mid,
                    h_warm = excluded.h_warm,
                    h_cool = excluded.h_cool
                """,
                (scope, item_id, _format_ts(ts), h_fast, h_mid, h_warm, h_cool),
            )

        return HotnessTouchResult(
            item_id=item_id,
            scope=scope,
            ts_utc=ts_utc,
            signal=signal,
            weight=float(weight),
            source=source,
            applied=True,
        )

    def pin(
        self,
        *,
        scope: str,
        item_id: str,
        level: str,
        rank: int,
        ts_utc: str,
        source: str,
    ) -> HotnessPinResult:
        scope = self._normalize_scope(scope)
        ts = _parse_ts(ts_utc)
        level = level.strip().lower()
        source = source.strip()
        if level not in _PIN_LEVELS:
            raise ValueError("pin level must be hard or soft")
        if not item_id:
            raise ValueError("item_id is required")
        if not self._item_exists(item_id):
            raise ValueError("memory item not found")
        self._validate_signal_source("pin_set", source)

        with self._store._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_hotness_pins(
                    scope, item_id, pin_level, pin_rank, set_ts_utc
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(scope, item_id) DO UPDATE SET
                    pin_level = excluded.pin_level,
                    pin_rank = excluded.pin_rank,
                    set_ts_utc = excluded.set_ts_utc
                """,
                (scope, item_id, level, int(rank), _format_ts(ts)),
            )
            conn.execute(
                """
                INSERT INTO memory_hotness_events(
                    scope, item_id, ts_utc, signal, weight, source
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (scope, item_id, _format_ts(ts), "pin_set", 0.0, source),
            )

        return HotnessPinResult(
            item_id=item_id,
            scope=scope,
            ts_utc=ts_utc,
            pin_level=level,
            pin_rank=int(rank),
            source=source,
            applied=True,
        )

    def unpin(
        self,
        *,
        scope: str,
        item_id: str,
        ts_utc: str,
        source: str,
    ) -> HotnessUnpinResult:
        scope = self._normalize_scope(scope)
        ts = _parse_ts(ts_utc)
        source = source.strip()
        if not item_id:
            raise ValueError("item_id is required")
        self._validate_signal_source("pin_unset", source)

        with self._store._connect() as conn:
            cur = conn.execute(
                "DELETE FROM memory_hotness_pins WHERE scope = ? AND item_id = ?",
                (scope, item_id),
            )
            conn.execute(
                """
                INSERT INTO memory_hotness_events(
                    scope, item_id, ts_utc, signal, weight, source
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (scope, item_id, _format_ts(ts), "pin_unset", 0.0, source),
            )

        return HotnessUnpinResult(
            item_id=item_id,
            scope=scope,
            ts_utc=ts_utc,
            source=source,
            removed=cur.rowcount > 0,
        )

    def rank(
        self,
        *,
        scope: str,
        as_of_utc: str,
        limit: int,
        quotas: MemoryHotnessQuotasConfig | None = None,
    ) -> HotnessRankResult:
        scope = self._normalize_scope(scope)
        as_of = _parse_ts(as_of_utc)
        limit = max(0, int(limit))
        quotas = quotas or self._config.quotas
        selected: list[HotnessRankEntry] = []
        selected_ids: set[str] = set()
        counts: dict[str, int] = {}
        pinned_over_budget_hard = 0
        pinned_over_budget_soft = 0

        with self._store._connect() as conn:
            hard_pins = _fetch_pins(conn, scope, "hard")
            hard_selected, pinned_over_budget_hard = _select_pins(hard_pins, limit, selected_ids)
            for item in hard_selected:
                selected.append(
                    HotnessRankEntry(
                        item_id=item["item_id"],
                        bin="pinned",
                        score=None,
                        pin_level="hard",
                    )
                )
            counts["hard_pins"] = len(hard_selected)
            if len(selected) >= limit:
                return HotnessRankResult(
                    scope=scope,
                    as_of_utc=as_of_utc,
                    limit=limit,
                    selected=selected[:limit],
                    pinned_over_budget_hard=pinned_over_budget_hard,
                    pinned_over_budget_soft=0,
                    counts=counts,
                )

            soft_pins = _fetch_pins(conn, scope, "soft")
            soft_selected, pinned_over_budget_soft = _select_pins(
                soft_pins, limit - len(selected), selected_ids
            )
            for item in soft_selected:
                selected.append(
                    HotnessRankEntry(
                        item_id=item["item_id"],
                        bin="pinned",
                        score=None,
                        pin_level="soft",
                    )
                )
            counts["soft_pins"] = len(soft_selected)

            if len(selected) < limit:
                candidates = _fetch_candidates(conn, scope, selected_ids)
                hotness_selected, bin_counts = _select_hotness_candidates(
                    candidates,
                    as_of,
                    limit - len(selected),
                    self._config,
                    quotas,
                )
                counts.update(bin_counts)
                hotness_selected_sorted = sorted(
                    hotness_selected,
                    key=lambda item: (-item["score"], item["key"], item["item_id"]),
                )
                for item in hotness_selected_sorted:
                    selected.append(
                        HotnessRankEntry(
                            item_id=item["item_id"],
                            bin=item["bin"],
                            score=round(float(item["score"]), 6),
                            pin_level=None,
                        )
                    )
                selected_ids.update({item["item_id"] for item in hotness_selected_sorted})

            if len(selected) < limit:
                recency = _fetch_recency_fallback(conn, limit - len(selected), selected_ids)
                counts["recency"] = len(recency)
                for item in recency:
                    selected.append(
                        HotnessRankEntry(
                            item_id=item["item_id"],
                            bin="recency",
                            score=None,
                            pin_level=None,
                        )
                    )

        return HotnessRankResult(
            scope=scope,
            as_of_utc=as_of_utc,
            limit=limit,
            selected=selected[:limit],
            pinned_over_budget_hard=pinned_over_budget_hard,
            pinned_over_budget_soft=pinned_over_budget_soft,
            counts=counts,
        )

    def gc(
        self,
        *,
        scope: str,
        as_of_utc: str,
        max_age_days: int | None = None,
        max_events: int | None = None,
    ) -> HotnessGcResult:
        scope = self._normalize_scope(scope)
        as_of = _parse_ts(as_of_utc)
        max_age_days = (
            int(max_age_days)
            if max_age_days is not None
            else int(self._config.retention.event_max_age_days)
        )
        max_events = (
            int(max_events)
            if max_events is not None
            else int(self._config.retention.event_max_count)
        )
        max_events = max(0, max_events)
        deleted = 0

        with self._store._connect() as conn:
            cutoff = as_of - dt.timedelta(days=max_age_days)
            cur = conn.execute(
                "DELETE FROM memory_hotness_events WHERE scope = ? AND ts_utc < ?",
                (scope, _format_ts(cutoff)),
            )
            deleted += cur.rowcount

            if max_events >= 0:
                cur = conn.execute(
                    """
                    DELETE FROM memory_hotness_events
                    WHERE event_id IN (
                        SELECT event_id
                        FROM memory_hotness_events
                        WHERE scope = ?
                        ORDER BY ts_utc DESC, event_id DESC
                        LIMIT -1 OFFSET ?
                    )
                    """,
                    (scope, max_events),
                )
                deleted += cur.rowcount

            row = conn.execute(
                "SELECT COUNT(*) AS count FROM memory_hotness_events WHERE scope = ?",
                (scope,),
            ).fetchone()
            remaining = int(row["count"] if row else 0)

        return HotnessGcResult(events_deleted=deleted, events_remaining=remaining)

    def state(self, *, scope: str, item_id: str) -> HotnessStateResult | None:
        scope = self._normalize_scope(scope)
        if not item_id:
            raise ValueError("item_id is required")
        with self._store._connect() as conn:
            row = conn.execute(
                """
                SELECT last_ts_utc, h_fast, h_mid, h_warm, h_cool
                FROM memory_hotness_state
                WHERE scope = ? AND item_id = ?
                """,
                (scope, item_id),
            ).fetchone()
        if not row:
            return None
        return HotnessStateResult(
            item_id=item_id,
            scope=scope,
            last_ts_utc=row["last_ts_utc"],
            h_fast=float(row["h_fast"]),
            h_mid=float(row["h_mid"]),
            h_warm=float(row["h_warm"]),
            h_cool=float(row["h_cool"]),
        )

    def _validate_signal_source(self, signal: str, source: str) -> None:
        allowed = self._config.allowed_signals
        sources = allowed.get(signal)
        if not sources:
            raise ValueError(f"signal not allowed: {signal}")
        if source not in sources:
            raise ValueError(f"source not allowed for signal {signal}: {source}")

    def _normalize_scope(self, scope: str) -> str:
        scope = (scope or "").strip()
        if scope:
            return scope
        return self._config.scope_default

    def _rate_limited(
        self,
        conn: sqlite3.Connection,
        scope: str,
        item_id: str,
        source: str,
        ts: dt.datetime,
    ) -> bool:
        if not self._config.rate_limit.enabled:
            return False
        min_interval_ms = max(0, int(self._config.rate_limit.min_interval_ms))
        if min_interval_ms <= 0:
            return False
        row = conn.execute(
            """
            SELECT ts_utc
            FROM memory_hotness_events
            WHERE scope = ? AND item_id = ? AND source = ?
            ORDER BY ts_utc DESC, event_id DESC
            LIMIT 1
            """,
            (scope, item_id, source),
        ).fetchone()
        if not row:
            return False
        last_ts = _parse_ts(row["ts_utc"])
        delta_ms = max(0.0, (ts - last_ts).total_seconds() * 1000.0)
        return delta_ms < min_interval_ms

    def _item_exists(self, item_id: str) -> bool:
        with self._store._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM memory_items WHERE item_id = ?",
                (item_id,),
            ).fetchone()
        return row is not None


def _parse_ts(value: str) -> dt.datetime:
    if not value or not _TS_PATTERN.match(value):
        raise ValueError("timestamp must be UTC ISO-8601 with 'Z' and second precision")
    parsed = dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    return parsed.replace(tzinfo=dt.timezone.utc)


def _format_ts(value: dt.datetime) -> str:
    ts = value.astimezone(dt.timezone.utc)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _decay(value: float, delta_seconds: float, half_life_seconds: int) -> float:
    if value <= 0.0:
        return 0.0
    half_life = max(1, int(half_life_seconds))
    if delta_seconds <= 0.0:
        return float(value)
    return float(value) * math.pow(2.0, -(delta_seconds / half_life))


def _fetch_pins(conn: sqlite3.Connection, scope: str, level: str) -> list[dict[str, str]]:
    rows = conn.execute(
        """
        SELECT p.item_id, p.pin_rank, m.key
        FROM memory_hotness_pins p
        JOIN memory_items m ON m.item_id = p.item_id
        WHERE p.scope = ? AND p.pin_level = ? AND m.status = 'active'
        ORDER BY p.pin_rank ASC, m.key ASC, m.item_id ASC
        """,
        (scope, level),
    ).fetchall()
    return [
        {"item_id": row["item_id"], "pin_rank": int(row["pin_rank"]), "key": row["key"]}
        for row in rows
    ]


def _select_pins(
    pins: list[dict[str, str]],
    limit: int,
    selected_ids: set[str],
) -> tuple[list[dict[str, str]], int]:
    remaining = max(0, int(limit))
    eligible = [item for item in pins if item["item_id"] not in selected_ids]
    selected = eligible[:remaining]
    selected_ids.update({item["item_id"] for item in selected})
    over_budget = max(0, len(eligible) - remaining)
    return selected, over_budget


def _fetch_candidates(
    conn: sqlite3.Connection,
    scope: str,
    selected_ids: set[str],
) -> list[dict[str, str]]:
    rows = conn.execute(
        """
        SELECT s.item_id, s.last_ts_utc, s.h_fast, s.h_mid, s.h_warm, s.h_cool,
               m.key, m.updated_at
        FROM memory_hotness_state s
        JOIN memory_items m ON m.item_id = s.item_id
        WHERE s.scope = ? AND m.status = 'active'
        """,
        (scope,),
    ).fetchall()
    candidates: list[dict[str, str]] = []
    for row in rows:
        if row["item_id"] in selected_ids:
            continue
        candidates.append(
            {
                "item_id": row["item_id"],
                "last_ts_utc": row["last_ts_utc"],
                "h_fast": float(row["h_fast"]),
                "h_mid": float(row["h_mid"]),
                "h_warm": float(row["h_warm"]),
                "h_cool": float(row["h_cool"]),
                "key": row["key"],
                "updated_at": row["updated_at"],
            }
        )
    return candidates


def _select_hotness_candidates(
    candidates: list[dict[str, str]],
    as_of: dt.datetime,
    limit: int,
    config: MemoryHotnessConfig,
    quotas: MemoryHotnessQuotasConfig,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    bins: dict[str, list[dict[str, str]]] = {bin_name: [] for bin_name in _BIN_ORDER}
    bins["cold"] = []
    for item in candidates:
        last_ts = _parse_ts(item["last_ts_utc"])
        delta = max(0.0, (as_of - last_ts).total_seconds())
        h_fast = _decay(item["h_fast"], delta, config.half_lives.fast_seconds)
        h_mid = _decay(item["h_mid"], delta, config.half_lives.mid_seconds)
        h_warm = _decay(item["h_warm"], delta, config.half_lives.warm_seconds)
        h_cool = _decay(item["h_cool"], delta, config.half_lives.cool_seconds)
        score = (
            h_fast * config.weights.fast
            + h_mid * config.weights.mid
            + h_warm * config.weights.warm
            + h_cool * config.weights.cool
        )
        bin_name = _score_to_bin(score, config)
        item["score"] = float(score)
        item["bin"] = bin_name
        bins[bin_name].append(item)

    for bin_name in bins:
        bins[bin_name] = sorted(
            bins[bin_name],
            key=lambda item: (-item.get("score", 0.0), item["key"], item["item_id"]),
        )

    selected: list[dict[str, str]] = []
    remaining = max(0, int(limit))
    targets = {
        "hot": int(math.floor(remaining * quotas.hot)),
        "recent": int(math.floor(remaining * quotas.recent)),
        "warm": int(math.floor(remaining * quotas.warm)),
        "cool": int(math.floor(remaining * quotas.cool)),
    }
    assigned = sum(targets.values())
    leftover = max(0, remaining - assigned)
    for bin_name in _BIN_ORDER:
        if leftover <= 0:
            break
        targets[bin_name] += 1
        leftover -= 1

    for bin_name in _BIN_ORDER:
        if remaining <= 0:
            break
        available = bins[bin_name]
        if not available:
            continue
        take = min(targets.get(bin_name, 0), remaining, len(available))
        if take <= 0:
            continue
        selected.extend(available[:take])
        bins[bin_name] = available[take:]
        remaining -= take

    if remaining > 0:
        spillover: list[dict[str, str]] = []
        for bin_name in _BIN_ORDER:
            spillover.extend(bins[bin_name])
        spillover = sorted(
            spillover,
            key=lambda item: (-item["score"], item["key"], item["item_id"]),
        )
        selected.extend(spillover[:remaining])
        remaining = 0

    counts = {
        bin_name: len([item for item in selected if item["bin"] == bin_name])
        for bin_name in _BIN_ORDER
    }
    counts["cold"] = len(bins.get("cold", []))
    return selected, counts


def _fetch_recency_fallback(
    conn: sqlite3.Connection,
    limit: int,
    selected_ids: set[str],
) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    placeholders = ",".join(["?"] * len(selected_ids)) if selected_ids else ""
    params: list[object] = []
    clause = ""
    if selected_ids:
        clause = f"AND item_id NOT IN ({placeholders})"
        params.extend(sorted(selected_ids))
    params.append(int(limit))
    rows = conn.execute(
        f"""
        SELECT item_id, updated_at, key
        FROM memory_items
        WHERE status = 'active' {clause}
        ORDER BY updated_at DESC, key ASC, item_id ASC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    return [
        {"item_id": row["item_id"], "key": row["key"], "updated_at": row["updated_at"]}
        for row in rows
    ]


def _score_to_bin(score: float, config: MemoryHotnessConfig) -> str:
    thresholds = config.thresholds
    if score >= thresholds.hot:
        return "hot"
    if score >= thresholds.recent:
        return "recent"
    if score >= thresholds.warm:
        return "warm"
    if score >= thresholds.cool:
        return "cool"
    return "cold"
