"""Delete preview/apply service for UX."""

from __future__ import annotations

import datetime as dt
from pathlib import Path


from sqlalchemy import func, select

from .models import DeleteApplyRequest, DeleteApplyResponse, DeletePreviewRequest, DeletePreviewResponse, DeleteSample
from .preview import PreviewTokenManager, hash_payload
from .redaction import normalize_path
from ..config import AppConfig
from ..storage.database import DatabaseManager
from ..storage.deletion import delete_range as delete_range_records
from ..storage.models import CaptureRecord, EventRecord, SegmentRecord


class DeleteService:
    def __init__(
        self,
        config: AppConfig,
        db: DatabaseManager,
        *,
        index_pruner: object | None = None,
    ) -> None:
        self._config = config
        self._db = db
        self._index_pruner = index_pruner
        self._preview = PreviewTokenManager(Path(config.capture.data_dir))

    def preview(self, request: DeletePreviewRequest) -> DeletePreviewResponse:
        criteria = request.criteria
        start, end = self._resolve_range(criteria)
        process = criteria.process
        window_title = criteria.window_title
        counts, samples = self._collect_counts(start, end, process, window_title, criteria.sample_limit)
        preview_id = self._preview.issue(
            kind="delete",
            version=hash_payload(criteria.model_dump(mode="json")),
            payload_hash=hash_payload(criteria.model_dump(mode="json")),
        )
        warnings: list[str] = []
        impacts: list[str] = []
        if counts.get("captures", 0) > 10_000:
            warnings.append("Large delete operation; consider narrowing the range.")
        if criteria.kind == "all":
            impacts.append("All captured data will be deleted.")
        return DeletePreviewResponse(
            preview_id=preview_id,
            counts=counts,
            sample=samples,
            warnings=warnings,
            impacts=impacts,
        )

    def apply(self, request: DeleteApplyRequest) -> DeleteApplyResponse:
        criteria = request.criteria
        self._preview.validate(
            request.preview_id,
            kind="delete",
            version=hash_payload(criteria.model_dump(mode="json")),
            payload_hash=hash_payload(criteria.model_dump(mode="json")),
        )
        if not request.confirm:
            raise ValueError("confirm=true required")
        phrase = (request.confirm_phrase or "").strip().upper()
        if phrase not in {"DELETE", "I UNDERSTAND"}:
            raise ValueError("confirm_phrase must be DELETE or I UNDERSTAND")
        start, end = self._resolve_range(criteria)
        counts = delete_range_records(
            self._db,
            Path(self._config.capture.data_dir),
            start_utc=start,
            end_utc=end,
            process=criteria.process,
            window_title=criteria.window_title,
            index_pruner=self._index_pruner,
        )
        response_counts = {
            "captures": counts.deleted_captures,
            "events": counts.deleted_events,
            "segments": counts.deleted_segments,
            "files": counts.deleted_files,
        }
        if request.expected_counts:
            for key, expected in request.expected_counts.items():
                actual = response_counts.get(key)
                if actual is not None and expected is not None and actual != expected:
                    raise ValueError("delete counts did not match preview")
        return DeleteApplyResponse(
            counts=response_counts,
            applied_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        )

    def _collect_counts(
        self,
        start: dt.datetime,
        end: dt.datetime,
        process: str | None,
        window_title: str | None,
        sample_limit: int,
    ) -> tuple[dict[str, int], list[DeleteSample]]:
        with self._db.session() as session:
            captures_stmt = select(func.count()).select_from(CaptureRecord).where(
                CaptureRecord.captured_at.between(start, end)
            )
            if process:
                captures_stmt = captures_stmt.where(
                    func.lower(CaptureRecord.foreground_process) == process.lower()
                )
            if window_title:
                captures_stmt = captures_stmt.where(CaptureRecord.foreground_window == window_title)
            captures = session.execute(captures_stmt).scalar_one()

            events_stmt = select(func.count()).select_from(EventRecord).where(
                EventRecord.ts_start.between(start, end)
            )
            if process:
                events_stmt = events_stmt.where(func.lower(EventRecord.app_name) == process.lower())
            if window_title:
                events_stmt = events_stmt.where(EventRecord.window_title == window_title)
            events = session.execute(events_stmt).scalar_one()

            segments_stmt = select(func.count()).select_from(SegmentRecord).where(
                SegmentRecord.started_at.between(start, end)
            )
            segments = session.execute(segments_stmt).scalar_one()

            sample_stmt = select(EventRecord).where(EventRecord.ts_start.between(start, end))
            if process:
                sample_stmt = sample_stmt.where(func.lower(EventRecord.app_name) == process.lower())
            if window_title:
                sample_stmt = sample_stmt.where(EventRecord.window_title == window_title)
            sample_stmt = sample_stmt.order_by(EventRecord.ts_start.desc()).limit(sample_limit)
            rows = session.execute(sample_stmt).scalars().all()

        sample: list[DeleteSample] = []
        data_root = Path(self._config.capture.data_dir)
        for row in rows:
            detail = None
            if row.screenshot_path:
                detail = normalize_path(row.screenshot_path, root=data_root)
            sample.append(
                DeleteSample(
                    kind="event",
                    identifier=row.event_id,
                    ts_utc=row.ts_start.isoformat() if row.ts_start else None,
                    detail=detail,
                )
            )
        return (
            {
                "captures": int(captures or 0),
                "events": int(events or 0),
                "segments": int(segments or 0),
                "files": 0,
            },
            sample,
        )

    def _resolve_range(self, criteria) -> tuple[dt.datetime, dt.datetime]:
        now = dt.datetime.now(dt.timezone.utc)
        if criteria.kind not in {"range", "all"}:
            raise ValueError("unsupported delete kind")
        if criteria.kind == "all":
            return dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc), now
        start = self._parse_datetime(criteria.start_utc)
        end = self._parse_datetime(criteria.end_utc)
        if not start or not end:
            raise ValueError("start_utc and end_utc required for range deletes")
        if end < start:
            start, end = end, start
        return start, end

    @staticmethod
    def _parse_datetime(value: str | None) -> dt.datetime | None:
        if not value:
            return None
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = dt.datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed
