"""State aggregation for UX surfaces."""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import os
import shutil
import time
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from .models import (
    AppInfo,
    ComponentHeartbeat,
    HealthIssue,
    HealthSummary,
    LockStatus,
    PluginSummary,
    PrivacySummary,
    QueueStatus,
    RoutingSummary,
    StateDiagnostics,
    StateSnapshot,
    StorageStatus,
)
from .state_store import compute_staleness, read_heartbeat
from .redaction import redact_payload
from ..config import AppConfig, is_loopback_host
from ..observability.metrics import get_gpu_snapshot
from ..storage.database import DatabaseManager
from ..storage.models import CaptureRecord, EmbeddingRecord, EventRecord
from ..storage.retention import RetentionManager


class StateService:
    def __init__(
        self,
        config: AppConfig,
        db: DatabaseManager,
        *,
        plugins: object | None = None,
        worker_supervisor: object | None = None,
    ) -> None:
        self._config = config
        self._db = db
        self._plugins = plugins
        self._worker_supervisor = worker_supervisor
        self._snapshot_cache: dict[str, Any] | None = None
        self._snapshot_ts = 0.0
        self._snapshot_ttl_s = 0.5
        self._storage_cache = {"ts": 0.0, "bytes": 0, "free": None}
        self._storage_cache_ttl_s = 30.0

    def snapshot(self, *, unlocked: bool | None = None) -> StateSnapshot:
        now = time.monotonic()
        if self._snapshot_cache and now - self._snapshot_ts < self._snapshot_ttl_s:
            cached = self._snapshot_cache["snapshot"]
            diagnostics = cached.diagnostics.model_copy()
            diagnostics.cache_hit = True
            diagnostics.cache_age_ms = (now - self._snapshot_ts) * 1000.0
            return cached.model_copy(update={"diagnostics": diagnostics})

        start = time.monotonic()
        db_queries = 0
        issues: list[HealthIssue] = []
        state_dir = Path(self._config.capture.data_dir) / "state"
        components = self._load_components(state_dir)
        queues, queue_queries = self._load_queue_status()
        db_queries += queue_queries

        storage_status, storage_age = self._storage_status()
        privacy = PrivacySummary(
            paused=self._config.privacy.paused,
            snooze_until_utc=_to_iso(self._config.privacy.snooze_until_utc),
            sanitize_default=self._config.privacy.sanitize_default,
            extractive_only_default=self._config.privacy.extractive_only_default,
            cloud_enabled=self._config.privacy.cloud_enabled,
            allow_cloud_images=self._config.privacy.allow_cloud_images,
            allow_token_vault_decrypt=self._config.privacy.allow_token_vault_decrypt,
        )
        routing = RoutingSummary(**_model_dump(self._config.routing))
        plugins = self._plugins_summary()
        lock = self._lock_status(unlocked)

        overall = "ok"
        if not self._db_ok():
            issues.append(
                HealthIssue(
                    issue_id="db_unreachable",
                    title="Database unavailable",
                    detail="Database health check failed.",
                    severity="critical",
                    remediation="Verify database path and encryption key.",
                )
            )
            overall = "blocked"
        db_queries += 1
        min_free_mb = self._config.capture.data_min_free_mb
        if storage_status.free_bytes is not None and min_free_mb is not None:
            if storage_status.free_bytes < int(min_free_mb) * 1024 * 1024:
                issues.append(
                    HealthIssue(
                        issue_id="low_disk",
                        title="Low disk space",
                        detail="Free space is below configured minimum.",
                        severity="critical",
                        remediation="Free disk space or lower capture retention.",
                    )
                )
                overall = "blocked"

        stale_components = [comp for comp in components if comp.stale]
        if stale_components:
            issues.append(
                HealthIssue(
                    issue_id="stale_components",
                    title="Stale components",
                    detail=f"Stale: {', '.join(comp.component for comp in stale_components)}",
                    severity="warning",
                    remediation="Verify runtime/worker services are running.",
                )
            )
            if overall == "ok":
                overall = "degraded"
        if privacy.paused:
            issues.append(
                HealthIssue(
                    issue_id="capture_paused",
                    title="Capture paused",
                    detail="Capture is paused.",
                    severity="warning",
                    remediation="Resume capture when ready.",
                )
            )
            if overall == "ok":
                overall = "degraded"

        health = HealthSummary(overall=overall, issues=issues)
        app_info = AppInfo(
            name="autocapture",
            version=_resolve_version(),
            git_sha=_resolve_git_sha(),
            mode=self._config.mode.mode,
            bind_host=self._config.api.bind_host,
            port=self._config.api.port,
            offline=self._config.offline,
        )
        diagnostics = StateDiagnostics(
            cache_hit=False,
            cache_age_ms=0.0,
            assembled_ms=(time.monotonic() - start) * 1000.0,
            db_queries=db_queries,
            disk_usage_age_s=storage_age,
        )
        snapshot = StateSnapshot(
            schema_version=1,
            time_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            app=app_info,
            health=health,
            components=components,
            queues=queues,
            storage=storage_status,
            privacy=privacy,
            routing=routing,
            plugins=plugins,
            lock=lock,
            diagnostics=diagnostics,
        )
        self._snapshot_cache = {"snapshot": snapshot}
        self._snapshot_ts = now
        return snapshot

    def _load_queue_status(self) -> tuple[QueueStatus, int]:
        with self._db.session() as session:
            ocr_pending = session.execute(
                select(func.count()).select_from(CaptureRecord).where(
                    CaptureRecord.ocr_status == "pending"
                )
            ).scalar_one()
            ocr_processing = session.execute(
                select(func.count()).select_from(CaptureRecord).where(
                    CaptureRecord.ocr_status == "processing"
                )
            ).scalar_one()
            span_embed_pending = session.execute(
                select(func.count()).select_from(EmbeddingRecord).where(
                    EmbeddingRecord.status == "pending"
                )
            ).scalar_one()
            event_embed_pending = session.execute(
                select(func.count()).select_from(EventRecord).where(
                    EventRecord.embedding_status == "pending"
                )
            ).scalar_one()
        return (
            QueueStatus(
                ocr_pending=int(ocr_pending),
                ocr_processing=int(ocr_processing),
                span_embed_pending=int(span_embed_pending),
                event_embed_pending=int(event_embed_pending),
            ),
            4,
        )

    def _storage_status(self) -> tuple[StorageStatus, float]:
        now = time.monotonic()
        if now - self._storage_cache["ts"] > self._storage_cache_ttl_s:
            media_root = Path(self._config.capture.data_dir) / "media"
            self._storage_cache["bytes"] = RetentionManager._folder_size_bytes(media_root)
            try:
                usage = shutil.disk_usage(self._config.capture.data_dir)
                self._storage_cache["free"] = int(usage.free)
            except Exception:
                self._storage_cache["free"] = None
            self._storage_cache["ts"] = now
        storage = StorageStatus(
            data_dir=str(Path(self._config.capture.data_dir)),
            media_path=str(Path(self._config.capture.data_dir) / "media"),
            media_usage_bytes=int(self._storage_cache["bytes"] or 0),
            screenshot_ttl_days=int(self._config.retention.screenshot_ttl_days),
            free_bytes=self._storage_cache.get("free"),
            min_free_mb=int(self._config.capture.data_min_free_mb),
        )
        return storage, now - self._storage_cache["ts"]

    def _load_components(self, state_dir: Path) -> list[ComponentHeartbeat]:
        now_dt = dt.datetime.now(dt.timezone.utc)
        components: list[ComponentHeartbeat] = []
        paths = {
            "api": state_dir / "api.json",
            "runtime": state_dir / "runtime.json",
            "workers": state_dir / "workers.json",
        }
        for name, path in paths.items():
            raw = read_heartbeat(path)
            if not raw:
                components.append(
                    ComponentHeartbeat(
                        component=name,
                        status="unknown",
                        time_utc=None,
                        interval_s=0.0,
                        stale=True,
                        signals={},
                        errors=[],
                    )
                )
                continue
            time_utc = raw.get("time_utc")
            interval = float(raw.get("interval_s") or 0.0)
            parsed = _parse_datetime(time_utc)
            stale = compute_staleness(now_dt, parsed, interval or 0.0)
            components.append(
                ComponentHeartbeat(
                    component=raw.get("component", name),
                    status=raw.get("status", "unknown"),
                    time_utc=time_utc,
                    interval_s=interval,
                    stale=stale,
                    signals=redact_payload(raw.get("signals") or {}),
                    errors=list(raw.get("errors") or []),
                )
            )
        gpu = get_gpu_snapshot()
        if gpu:
            components.append(
                ComponentHeartbeat(
                    component="gpu",
                    status="ok",
                    time_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    interval_s=0.0,
                    stale=False,
                    signals=redact_payload(gpu),
                    errors=[],
                )
            )
        if self._worker_supervisor is not None:
            try:
                health = self._worker_supervisor.health_snapshot()
                components.append(
                    ComponentHeartbeat(
                        component="worker_supervisor",
                        status="ok" if all(health.values()) else "degraded",
                        time_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                        interval_s=0.0,
                        stale=False,
                        signals=redact_payload(health),
                        errors=[],
                    )
                )
            except Exception:
                pass
        return components

    def _plugins_summary(self) -> PluginSummary:
        if not self._plugins:
            return PluginSummary(safe_mode=False, enabled_count=0, blocked_count=0)
        try:
            statuses = getattr(self._plugins, "catalog")()
            enabled = 0
            blocked = 0
            for status in statuses:
                if status.plugin.plugin_id == "__discovery__":
                    continue
                if status.blocked:
                    blocked += 1
                if status.enabled and not status.blocked:
                    enabled += 1
            safe_mode = bool(getattr(self._plugins, "safe_mode", False))
            return PluginSummary(safe_mode=safe_mode, enabled_count=enabled, blocked_count=blocked)
        except Exception:
            return PluginSummary(safe_mode=False, enabled_count=0, blocked_count=0)

    def _lock_status(self, unlocked: bool | None) -> LockStatus:
        required = bool(
            self._config.security.local_unlock_enabled
            and self._config.mode.mode != "remote"
            and is_loopback_host(self._config.api.bind_host)
            and self._config.security.provider != "disabled"
        )
        return LockStatus(
            required=required,
            unlocked=unlocked,
            provider=self._config.security.provider,
            expires_at_utc=None,
        )

    def _db_ok(self) -> bool:
        try:
            with self._db.session() as session:
                session.execute(select(1))
            return True
        except Exception:
            return False


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


def _resolve_version() -> str | None:
    try:
        return importlib.metadata.version("autocapture")
    except Exception:
        return None


def _resolve_git_sha() -> str | None:
    head = Path(".git/HEAD")
    if not head.exists():
        return os.environ.get("AUTOCAPTURE_GIT_SHA")
    try:
        content = head.read_text(encoding="utf-8").strip()
        if content.startswith("ref:"):
            ref_path = content.split(" ", 1)[1].strip()
            ref_file = Path(".git") / ref_path
            if ref_file.exists():
                return ref_file.read_text(encoding="utf-8").strip()[:12]
        return content[:12]
    except Exception:
        return os.environ.get("AUTOCAPTURE_GIT_SHA")


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


def _to_iso(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.isoformat()
