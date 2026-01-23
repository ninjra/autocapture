"""Shared UX models for API + CLI surfaces."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_id: str
    title: str
    detail: Optional[str] = None
    severity: str = Field("info", description="info|warning|critical")
    remediation: Optional[str] = None


class ComponentHeartbeat(BaseModel):
    model_config = ConfigDict(extra="allow")

    component: str
    status: str = Field("unknown", description="ok|degraded|blocked|unknown")
    time_utc: Optional[str] = None
    interval_s: float = 0.0
    stale: bool = False
    signals: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class QueueStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ocr_pending: int = 0
    ocr_processing: int = 0
    span_embed_pending: int = 0
    event_embed_pending: int = 0


class StorageStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: str
    media_path: str
    media_usage_bytes: int = 0
    screenshot_ttl_days: int = 0
    free_bytes: Optional[int] = None
    min_free_mb: Optional[int] = None


class StorageStatsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: str
    media_dir: str
    staging_dir: str
    db_path: Optional[str] = None
    media_bytes: int = 0
    staging_bytes: int = 0
    db_bytes: int = 0
    total_bytes: int = 0
    free_bytes: Optional[int] = None
    collected_at_utc: str
    cache_hit: bool = False
    cache_age_s: float = 0.0


class PrivacySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paused: bool = False
    snooze_until_utc: Optional[str] = None
    sanitize_default: bool = True
    extractive_only_default: bool = True
    cloud_enabled: bool = False
    allow_cloud_images: bool = False
    allow_token_vault_decrypt: bool = False


class RoutingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capture: str = "local"
    ocr: str = "local"
    embedding: str = "local"
    retrieval: str = "local"
    vector_backend: str = "local"
    spans_v2_backend: str = "local"
    table_extractor: str = "local"
    reranker: str = "enabled"
    compressor: str = "extractive"
    verifier: str = "rules"
    llm: str = "gateway"


class PluginSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    safe_mode: bool = False
    enabled_count: int = 0
    blocked_count: int = 0


class LockStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: bool = False
    unlocked: Optional[bool] = None
    provider: Optional[str] = None
    expires_at_utc: Optional[str] = None


class AppInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "autocapture"
    version: Optional[str] = None
    git_sha: Optional[str] = None
    mode: Optional[str] = None
    bind_host: Optional[str] = None
    port: Optional[int] = None
    offline: bool = False


class HealthSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall: str = Field("ok", description="ok|degraded|blocked")
    issues: list[HealthIssue] = Field(default_factory=list)


class StateDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_hit: bool = False
    cache_age_ms: float = 0.0
    assembled_ms: float = 0.0
    db_queries: int = 0
    disk_usage_age_s: float = 0.0


class StateSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    time_utc: str
    app: AppInfo
    health: HealthSummary
    components: list[ComponentHeartbeat] = Field(default_factory=list)
    queues: QueueStatus
    storage: StorageStatus
    privacy: PrivacySummary
    routing: RoutingSummary
    plugins: PluginSummary
    lock: LockStatus
    diagnostics: StateDiagnostics


class EvidenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int = 0
    citable: int = 0
    redacted: int = 0
    injection_risk_max: float = 0.0
    time_range: Optional[tuple[str, str]] = None


class BannerAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    type: str
    value: Optional[str] = None


class AnswerBanner(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = Field("none", description="none|no_evidence|degraded|locked")
    title: str = ""
    message: str = ""
    reasons: list[str] = Field(default_factory=list)
    actions: list[BannerAction] = Field(default_factory=list)


class DiffEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    before: Any
    after: Any
    kind: str = Field("change", description="add|remove|change")


class SettingsTier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier_id: str
    label: str
    description: Optional[str] = None
    rank: int = 0
    requires_confirm: bool = False
    confirm_phrase: Optional[str] = None


class SettingsOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    value: str


class SettingsField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    label: str
    description: Optional[str] = None
    kind: str = Field("string", description="string|bool|int|float|select|list|json")
    tier: str
    default: Any = None
    options: list[SettingsOption] = Field(default_factory=list)
    options_source: Optional[str] = None
    placeholder: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    multiline: bool = False
    requires_restart: bool = False
    sensitive: bool = False
    danger_level: str = Field("info", description="info|warn|danger")


class SettingsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str
    label: str
    description: Optional[str] = None
    fields: list[SettingsField] = Field(default_factory=list)


class SettingsSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    generated_at_utc: str
    tiers: list[SettingsTier]
    sections: list[SettingsSection]


class SettingsEffectiveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    settings: dict[str, Any]
    effective: dict[str, Any]
    redacted: bool = True


class SettingsPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate: dict[str, Any]
    tier: Optional[str] = None


class SettingsPreviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    preview_id: str
    diff: list[DiffEntry]
    effective_diff: list[DiffEntry]
    impacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    effective_preview: dict[str, Any]


class SettingsApplyRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate: dict[str, Any]
    preview_id: str
    confirm: bool = False
    confirm_phrase: Optional[str] = None
    tier: Optional[str] = None


class SettingsApplyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    applied_at_utc: str
    effective: dict[str, Any]


class DoctorCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    ok: bool
    detail: str
    severity: str = Field("info", description="info|warning|critical")


class DoctorReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    generated_at_utc: str
    results: list[DoctorCheck]


class DeleteCriteria(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: str
    start_utc: Optional[str] = None
    end_utc: Optional[str] = None
    process: Optional[str] = None
    window_title: Optional[str] = None
    sample_limit: int = 20


class DeletePreviewRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    criteria: DeleteCriteria


class DeleteSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    identifier: str
    ts_utc: Optional[str] = None
    detail: Optional[str] = None


class DeletePreviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preview_id: str
    counts: dict[str, int]
    sample: list[DeleteSample] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    impacts: list[str] = Field(default_factory=list)


class DeleteApplyRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    criteria: DeleteCriteria
    preview_id: str
    confirm: bool = False
    confirm_phrase: Optional[str] = None
    expected_counts: Optional[dict[str, int]] = None


class DeleteApplyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    counts: dict[str, int]
    applied_at_utc: str


class AuditRequestSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    query_id: Optional[str]
    query_text: str
    status: str
    started_at_utc: Optional[str] = None
    completed_at_utc: Optional[str] = None
    warnings: dict[str, Any] = Field(default_factory=dict)
    evidence_count: int = 0
    provider_calls: int = 0
    answer_id: Optional[str] = None
    answer_mode: Optional[str] = None
    citations_count: int = 0


class AuditClaimCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    span_id: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    confidence: Optional[float] = None


class AuditClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_index: int
    text: str
    entailment_verdict: Optional[str] = None
    entailment_rationale: Optional[str] = None
    citations: list[AuditClaimCitation] = Field(default_factory=list)


class AuditAnswerDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_id: str
    query_id: Optional[str]
    mode: str
    created_at_utc: Optional[str] = None
    coverage: dict[str, Any] = Field(default_factory=dict)
    confidence: dict[str, Any] = Field(default_factory=dict)
    budgets: dict[str, Any] = Field(default_factory=dict)
    answer_text: Optional[str] = None
    claims: list[AuditClaim] = Field(default_factory=list)
    citations_count: int = 0


class AuditSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests: list[AuditRequestSummary]
    generated_at_utc: str


class AuditAnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: AuditAnswerDetail
    generated_at_utc: str
