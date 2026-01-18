"""Persistence adapter for overlay tracker data."""

from __future__ import annotations

import datetime as dt
from typing import Sequence

from sqlalchemy import func, select

from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from ..storage.models import (
    OverlayEventRecord,
    OverlayItemIdentityRecord,
    OverlayItemRecord,
    OverlayKvRecord,
    OverlayProjectRecord,
)
from .clock import Clock
from .schemas import (
    OverlayEventEvidence,
    OverlayItemSummary,
    OverlayPersistEvent,
    OverlayProjectSummary,
)

DEFAULT_PROJECT_NAME = "Inbox"


class OverlayTrackerStore:
    def __init__(self, db: DatabaseManager, clock: Clock) -> None:
        self._db = db
        self._clock = clock
        self._log = get_logger("overlay_tracker.store")

    def record_events(self, events: Sequence[OverlayPersistEvent]) -> list[int]:
        if not events:
            return []

        def _tx(session) -> list[int]:
            project = self._ensure_default_project(session, self._clock.now())
            identity_cache: dict[tuple[str, str], OverlayItemRecord] = {}
            item_ids: list[int] = []
            for event in events:
                if not event.identity_type or not event.identity_key:
                    continue
                cache_key = (event.identity_type, event.identity_key)
                item = identity_cache.get(cache_key)
                if item is None:
                    item = self._resolve_item(session, project, event)
                    identity_cache[cache_key] = item
                self._update_item_from_event(item, event)
                self._append_event(session, item, event)
                item_ids.append(item.id)
            return item_ids

        return self._db.transaction(_tx)

    def query_projects(self) -> list[OverlayProjectSummary]:
        def _load(session) -> list[OverlayProjectSummary]:
            self._ensure_default_project(session, self._clock.now())
            projects = session.execute(
                select(OverlayProjectRecord).order_by(OverlayProjectRecord.name.asc())
            ).scalars()
            return [
                OverlayProjectSummary(project_id=project.id, name=project.name)
                for project in projects
            ]

        return self._db.transaction(_load)

    def query_items(
        self,
        now_utc: dt.datetime,
        *,
        stale_after_s: float,
        include_snoozed: bool = False,
    ) -> tuple[list[OverlayItemSummary], list[OverlayItemSummary]]:
        def _load(session) -> tuple[list[OverlayItemSummary], list[OverlayItemSummary]]:
            stmt = select(OverlayItemRecord).order_by(OverlayItemRecord.last_activity_at_utc.desc())
            if not include_snoozed:
                stmt = stmt.where(
                    (OverlayItemRecord.snooze_until_utc.is_(None))
                    | (OverlayItemRecord.snooze_until_utc <= now_utc)
                )
            items = session.execute(stmt).scalars().all()
            active: list[OverlayItemSummary] = []
            stale: list[OverlayItemSummary] = []
            cutoff = now_utc - dt.timedelta(seconds=stale_after_s)
            for item in items:
                last_activity = _ensure_aware(item.last_activity_at_utc)
                snooze_until = _ensure_aware(item.snooze_until_utc)
                summary = OverlayItemSummary(
                    item_id=item.id,
                    project_id=item.project_id,
                    display_name=item.display_name,
                    process_name=item.last_process_name,
                    window_title=item.last_window_title_raw,
                    browser_url=item.last_browser_url_raw,
                    last_activity_at_utc=last_activity,
                    state=item.state,
                    snooze_until_utc=snooze_until,
                )
                if last_activity >= cutoff:
                    active.append(summary)
                else:
                    stale.append(summary)
            return active, stale

        return self._db.transaction(_load)

    def query_evidence(self, item_id: int, *, limit: int = 50) -> list[OverlayEventEvidence]:
        def _load(session) -> list[OverlayEventEvidence]:
            stmt = (
                select(OverlayEventRecord)
                .where(OverlayEventRecord.item_id == item_id)
                .order_by(OverlayEventRecord.ts_utc.desc())
                .limit(limit)
            )
            records = session.execute(stmt).scalars().all()
            return [
                OverlayEventEvidence(
                    event_id=record.id,
                    ts_utc=_ensure_aware(record.ts_utc),
                    event_type=record.event_type,
                    process_name=record.process_name,
                    raw_window_title=record.raw_window_title,
                    raw_browser_url=record.raw_browser_url,
                    collector=record.collector,
                    schema_version=record.schema_version,
                    app_version=record.app_version,
                    payload=record.payload_json,
                )
                for record in records
            ]

        return self._db.transaction(_load)

    def get_kv(self, key: str) -> dict | None:
        def _load(session) -> dict | None:
            record = session.get(OverlayKvRecord, key)
            return record.value_json if record else None

        return self._db.transaction(_load)

    def update_item_state(self, item_id: int, state: str) -> None:
        def _tx(session) -> None:
            item = session.get(OverlayItemRecord, item_id)
            if not item:
                return
            item.state = state
            item.updated_at = self._clock.now()

        self._db.transaction(_tx)

    def toggle_running(self, item_id: int) -> None:
        def _tx(session) -> None:
            item = session.get(OverlayItemRecord, item_id)
            if not item:
                return
            item.state = "idle" if item.state == "running" else "running"
            item.updated_at = self._clock.now()

        self._db.transaction(_tx)

    def rename_item(self, item_id: int, name: str) -> None:
        def _tx(session) -> None:
            item = session.get(OverlayItemRecord, item_id)
            if not item:
                return
            item.display_name = name.strip() if name else None
            item.updated_at = self._clock.now()

        self._db.transaction(_tx)

    def snooze_item(self, item_id: int, until_utc: dt.datetime | None) -> None:
        def _tx(session) -> None:
            item = session.get(OverlayItemRecord, item_id)
            if not item:
                return
            item.snooze_until_utc = until_utc
            item.updated_at = self._clock.now()

        self._db.transaction(_tx)

    def cycle_project(self, item_id: int) -> None:
        def _tx(session) -> None:
            item = session.get(OverlayItemRecord, item_id)
            if not item:
                return
            projects = (
                session.execute(
                    select(OverlayProjectRecord).order_by(OverlayProjectRecord.name.asc())
                )
                .scalars()
                .all()
            )
            if len(projects) < 2:
                return
            project_ids = [project.id for project in projects]
            try:
                idx = project_ids.index(item.project_id)
            except ValueError:
                idx = 0
            next_id = project_ids[(idx + 1) % len(project_ids)]
            item.project_id = next_id
            item.updated_at = self._clock.now()

        self._db.transaction(_tx)

    def append_action_event(
        self,
        item_id: int,
        *,
        event_type: str,
        payload: dict,
        ts_utc: dt.datetime,
        app_version: str | None,
    ) -> None:
        def _tx(session) -> None:
            item = session.get(OverlayItemRecord, item_id)
            if not item:
                return
            record = OverlayEventRecord(
                item_id=item.id,
                project_id=item.project_id,
                event_type=event_type,
                ts_utc=ts_utc,
                process_name=item.last_process_name,
                raw_window_title=item.last_window_title_raw,
                raw_browser_url=item.last_browser_url_raw,
                identity_type=item.identity_type,
                identity_key=item.identity_key,
                collector="hotkey",
                schema_version="v1",
                app_version=app_version,
                payload_json=payload,
            )
            session.add(record)

        self._db.transaction(_tx)

    def retention_cleanup(self, *, event_days: int, event_cap: int, now_utc: dt.datetime) -> None:
        def _tx(session) -> None:
            if event_days > 0:
                cutoff = now_utc - dt.timedelta(days=event_days)
                session.query(OverlayEventRecord).filter(OverlayEventRecord.ts_utc < cutoff).delete(
                    synchronize_session=False
                )
            if event_cap > 0:
                total = session.execute(select(func.count(OverlayEventRecord.id))).scalar_one()
                if total > event_cap:
                    over = total - event_cap
                    ids = (
                        session.execute(
                            select(OverlayEventRecord.id)
                            .order_by(OverlayEventRecord.ts_utc.asc())
                            .limit(over)
                        )
                        .scalars()
                        .all()
                    )
                    if ids:
                        session.query(OverlayEventRecord).filter(
                            OverlayEventRecord.id.in_(ids)
                        ).delete(synchronize_session=False)
            self._set_kv(session, "retention_last_run", {"ts": now_utc.isoformat()}, now_utc)

        self._db.transaction(_tx)

    def _ensure_default_project(self, session, now_utc: dt.datetime) -> OverlayProjectRecord:
        project = session.execute(
            select(OverlayProjectRecord).where(OverlayProjectRecord.name == DEFAULT_PROJECT_NAME)
        ).scalar_one_or_none()
        if project:
            return project
        project = OverlayProjectRecord(name=DEFAULT_PROJECT_NAME)
        project.created_at = now_utc
        project.updated_at = now_utc
        session.add(project)
        session.flush()
        self._log.info("Created default overlay project: {}", DEFAULT_PROJECT_NAME)
        return project

    def _resolve_item(
        self,
        session,
        default_project: OverlayProjectRecord,
        event: OverlayPersistEvent,
    ) -> OverlayItemRecord:
        mapping = session.execute(
            select(OverlayItemIdentityRecord).where(
                OverlayItemIdentityRecord.identity_type == event.identity_type,
                OverlayItemIdentityRecord.identity_key == event.identity_key,
            )
        ).scalar_one_or_none()
        if mapping:
            item = session.get(OverlayItemRecord, mapping.item_id)
            if item:
                return item
        now_utc = event.ts_utc
        item = OverlayItemRecord(
            project_id=default_project.id,
            display_name=None,
            last_process_name=event.process_name,
            last_window_title_raw=event.raw_window_title,
            last_browser_url_raw=event.raw_browser_url,
            identity_type=event.identity_type,
            identity_key=event.identity_key,
            state="idle",
            last_activity_at_utc=event.ts_utc,
            snooze_until_utc=None,
            created_at=now_utc,
            updated_at=now_utc,
        )
        session.add(item)
        session.flush()
        identity = OverlayItemIdentityRecord(
            item_id=item.id,
            identity_type=event.identity_type or "title",
            identity_key=event.identity_key or "",
            created_at=now_utc,
        )
        session.add(identity)
        return item

    def _update_item_from_event(self, item: OverlayItemRecord, event: OverlayPersistEvent) -> None:
        item.last_process_name = event.process_name
        item.last_window_title_raw = event.raw_window_title
        item.last_browser_url_raw = event.raw_browser_url
        item.identity_type = event.identity_type
        item.identity_key = event.identity_key
        if event.event_type in {"foreground", "input_activity"}:
            item.last_activity_at_utc = event.ts_utc
            item.snooze_until_utc = None
        item.updated_at = self._clock.now()

    def _append_event(
        self,
        session,
        item: OverlayItemRecord,
        event: OverlayPersistEvent,
    ) -> None:
        record = OverlayEventRecord(
            item_id=item.id,
            project_id=item.project_id,
            event_type=event.event_type,
            ts_utc=event.ts_utc,
            process_name=event.process_name,
            raw_window_title=event.raw_window_title,
            raw_browser_url=event.raw_browser_url,
            identity_type=event.identity_type,
            identity_key=event.identity_key,
            collector=event.collector,
            schema_version=event.schema_version,
            app_version=event.app_version,
            payload_json=event.payload,
        )
        session.add(record)

    @staticmethod
    def _set_kv(session, key: str, payload: dict, now_utc: dt.datetime) -> None:
        record = session.get(OverlayKvRecord, key)
        if record is None:
            record = OverlayKvRecord(key=key, value_json=payload, updated_at=now_utc)
            session.add(record)
            return
        record.value_json = payload
        record.updated_at = now_utc


def _ensure_aware(value: dt.datetime | None) -> dt.datetime:
    if value is None:
        return dt.datetime.now(dt.timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value
