"""UX-focused API routes (state/settings/doctor/delete/audit)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ...ux.audit_service import AuditService
from ...ux.delete_service import DeleteService
from ...ux.doctor_service import DoctorService
from ...ux.models import (
    DeleteApplyRequest,
    DeleteApplyResponse,
    DeleteCriteria,
    DeletePreviewRequest,
    DeletePreviewResponse,
    SettingsApplyRequest,
    SettingsApplyResponse,
    SettingsEffectiveResponse,
    SettingsPreviewRequest,
    SettingsPreviewResponse,
    SettingsSchema,
    StateSnapshot,
)
from ...ux.settings_service import SettingsService
from ...ux.state_service import StateService
from ...ux.perf_log import read_perf_log
from ...runtime_env import ProfileName
from ...runtime_profile_override import write_profile_override
from ..container import AppContainer


class DeleteCriteriaBody(BaseModel):
    start_utc: Optional[str] = None
    end_utc: Optional[str] = None
    process: Optional[str] = None
    window_title: Optional[str] = None
    sample_limit: int = 20


class DeleteApplyBody(DeleteCriteriaBody):
    preview_id: str
    confirm: bool = False
    confirm_phrase: Optional[str] = None
    expected_counts: Optional[dict[str, int]] = None


class RuntimeProfileBody(BaseModel):
    profile: Optional[str] = None


def build_ux_router(container: AppContainer) -> APIRouter:
    router = APIRouter()
    config = container.config
    db = container.db
    plugins = container.plugins
    state_service = StateService(
        config,
        db,
        plugins=plugins,
        worker_supervisor=container.worker_supervisor,
    )
    settings_service = SettingsService(config, plugins=plugins)
    doctor_service = DoctorService(config)
    delete_service = DeleteService(config, db, index_pruner=container.index_pruner)
    audit_service = AuditService(db)

    @router.get("/api/state", response_model=StateSnapshot)
    def state_snapshot() -> StateSnapshot:
        return state_service.snapshot(unlocked=True)

    @router.get("/api/settings/schema", response_model=SettingsSchema)
    def settings_schema() -> SettingsSchema:
        return settings_service.schema()

    @router.get("/api/settings/effective", response_model=SettingsEffectiveResponse)
    def settings_effective() -> SettingsEffectiveResponse:
        return settings_service.effective()

    @router.post("/api/settings/preview", response_model=SettingsPreviewResponse)
    def settings_preview(request: SettingsPreviewRequest) -> SettingsPreviewResponse:
        try:
            return settings_service.preview(request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/settings/apply", response_model=SettingsApplyResponse)
    def settings_apply(request: SettingsApplyRequest) -> SettingsApplyResponse:
        try:
            return settings_service.apply(request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/doctor")
    def doctor(verbose: bool = False):
        return doctor_service.run(verbose=verbose)

    @router.post("/api/delete/{kind}/preview", response_model=DeletePreviewResponse)
    def delete_preview(kind: str, body: DeleteCriteriaBody) -> DeletePreviewResponse:
        criteria = DeleteCriteria(
            kind=kind,
            start_utc=body.start_utc,
            end_utc=body.end_utc,
            process=body.process,
            window_title=body.window_title,
            sample_limit=body.sample_limit,
        )
        try:
            return delete_service.preview(DeletePreviewRequest(criteria=criteria))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/delete/{kind}/apply", response_model=DeleteApplyResponse)
    def delete_apply(kind: str, body: DeleteApplyBody) -> DeleteApplyResponse:
        criteria = DeleteCriteria(
            kind=kind,
            start_utc=body.start_utc,
            end_utc=body.end_utc,
            process=body.process,
            window_title=body.window_title,
            sample_limit=body.sample_limit,
        )
        request = DeleteApplyRequest(
            criteria=criteria,
            preview_id=body.preview_id,
            confirm=body.confirm,
            confirm_phrase=body.confirm_phrase,
            expected_counts=body.expected_counts,
        )
        try:
            return delete_service.apply(request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/audit/requests")
    def audit_requests(limit: int = Query(20, ge=1, le=200)):
        return audit_service.list_requests(limit=limit)

    @router.get("/api/audit/answers/{answer_id}")
    def audit_answer(answer_id: str, verbose: bool = False):
        try:
            return audit_service.answer_detail(answer_id, verbose=verbose)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/api/perf/log")
    def perf_log(
        component: str = Query("runtime"),
        limit: int = Query(200, ge=1, le=2000),
    ):
        try:
            return read_perf_log(Path(config.capture.data_dir), component=component, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/runtime/profile")
    def runtime_profile(body: RuntimeProfileBody):
        raw = (body.profile or "").strip().lower()
        if raw in {"", "auto", "balanced", "default"}:
            profile = None
        elif raw in {"max", "max_capture", "foreground"}:
            profile = ProfileName.FOREGROUND
        elif raw in {"low", "low_impact", "idle"}:
            profile = ProfileName.IDLE
        else:
            raise HTTPException(status_code=400, detail="Unknown profile value")
        override_path = Path(config.capture.data_dir) / "state" / "profile_override.json"
        write_profile_override(override_path, profile, source="api")
        return {"status": "ok", "profile": profile.value if profile else None}

    return router
