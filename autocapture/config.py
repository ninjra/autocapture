"""Configuration loading and validation using Pydantic models."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from .paths import default_data_dir, default_memory_dir, default_staging_dir
from .presets import apply_preset
from .settings_store import read_settings


def _default_database_url() -> str:
    path = (default_data_dir() / "autocapture.db").resolve()
    return f"sqlite:///{path.as_posix()}"


def _default_security_provider() -> str:
    if os.environ.get("AUTOCAPTURE_TEST_MODE") or os.environ.get("PYTEST_CURRENT_TEST"):
        return "test"
    if sys.platform == "win32":
        return "windows_hello"
    return "disabled"


def is_dev_mode(env: dict[str, str] | None = None) -> bool:
    env = env or os.environ
    value = (env.get("APP_ENV") or env.get("AUTOCAPTURE_ENV") or "").strip().lower()
    return value in {"dev", "development"}


def _default_allow_insecure_dev() -> bool:
    if os.environ.get("AUTOCAPTURE_TEST_MODE") or os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    return is_dev_mode()


class HIDConfig(BaseModel):
    min_interval_ms: int = Field(
        500,
        ge=100,
        description="Minimum interval between captures while input is active.",
    )
    idle_grace_ms: int = Field(
        1500,
        ge=200,
        description="Delay before stopping capture after input stops.",
    )
    duplicate_threshold: float = Field(
        0.02,
        ge=0.0,
        le=1.0,
        description="Normalized perceptual hash distance to treat frames as duplicates.",
    )
    duplicate_window_s: float = Field(
        10.0,
        gt=0.0,
        description="Sliding window (seconds) for duplicate detection comparisons.",
    )
    duplicate_max_items: int = Field(
        16,
        ge=2,
        description="Max recent frames to compare for duplicate detection.",
    )
    duplicate_pixel_threshold: float = Field(
        2.5,
        gt=0.0,
        description="Mean absolute pixel diff threshold for duplicate detection.",
    )
    fps_soft_cap: float = Field(
        4.0,
        gt=0,
        description="Maximum capture rate enforced during intense bursts.",
    )
    block_fullscreen: bool = Field(
        True,
        description="Skip captures when a full screen exclusive application is focused.",
    )


class CaptureConfig(BaseModel):
    hid: HIDConfig = HIDConfig()
    fps_min: float = Field(0.5, gt=0.0)
    fps_max: float = Field(2.0, gt=0.0)
    tile_size: int = Field(512, ge=256, le=2048)
    fullscreen_primary: bool = Field(
        True,
        description="Store full-screen captures as the primary artifact (default).",
    )
    fullscreen_width: int = Field(
        3840,
        ge=0,
        description="Stored full-screen width; 0 keeps native width.",
    )
    focus_crop_enabled: bool = Field(
        True,
        description="Store focus/ROI crop as a supplemental artifact.",
    )
    focus_crop_size: int = Field(
        512,
        ge=128,
        le=4096,
        description="Square focus crop size (pixels); defaults to tile_size if unset.",
    )
    focus_crop_reference: str = Field(
        "event",
        description="Where to store focus crop reference (event|tags).",
    )
    diff_epsilon: float = Field(0.04, ge=0.0, le=1.0)
    downscale_width: int = Field(256, ge=64)
    always_store_fullres: bool = Field(True)
    thumbnail_width: int = Field(640, ge=64)
    staging_dir: Path = Field(
        default_factory=default_staging_dir,
        description="Local NVMe-backed directory for temporary assets.",
    )
    data_dir: Path = Field(
        default_factory=default_data_dir,
        description="Base directory for media and recorder assets.",
    )
    encoder: str = Field(
        "nvenc_webp",
        description="Encoder preset (nvenc_webp, nvenc_avif, cpu_webp).",
    )
    record_video: bool = Field(
        True,
        description="Enable FFmpeg video recording for activity segments.",
    )
    multi_monitor_enabled: bool = Field(
        True, description="Enable multi-monitor capture metadata and normalization."
    )
    hdr_enabled: bool = Field(
        False, description="Enable HDR detection/tone mapping hooks for capture frames."
    )
    layout_mode: str = Field(
        "virtual_desktop",
        description="Frame layout mode (virtual_desktop or per_monitor).",
    )
    video_bitrate: str = Field(
        "8M",
        description="Target bitrate for video segments (e.g. 8M).",
    )
    video_preset: str = Field(
        "p4",
        description="Encoder preset for NVENC codecs.",
    )
    max_pending: int = Field(
        5000,
        ge=100,
        description="Backpressure limit for outstanding capture tasks.",
    )
    staging_min_free_mb: int = Field(
        512,
        ge=0,
        description=("Minimum free space (MB) required in staging_dir. Set to 0 to disable."),
    )
    data_min_free_mb: int = Field(
        1024,
        ge=0,
        description="Minimum free space (MB) required in data_dir. Set to 0 to disable.",
    )
    vision_sample_rate: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Sample rate for vision-caption jobs (0 disables).",
    )

    @field_validator("focus_crop_reference")
    @classmethod
    def _validate_focus_reference(cls, value: str) -> str:
        allowed = {"event", "tags"}
        if value not in allowed:
            raise ValueError(f"focus_crop_reference must be one of {sorted(allowed)}")
        return value


class TrackingConfig(BaseModel):
    enabled: bool = True
    db_path: Path = Field(
        Path("./host_events.sqlite"),
        description="SQLite file path for host events (relative to capture.data_dir).",
    )
    queue_maxsize: int = Field(20000, ge=1000)
    flush_interval_ms: int = Field(1000, ge=100)
    foreground_poll_ms: int = Field(250, ge=50)
    clipboard_poll_ms: int = Field(250, ge=50)
    track_mouse_movement: bool = True
    mouse_move_sample_ms: int = Field(50, ge=10)
    enable_clipboard: bool = False
    retention_days: int | None = None
    encryption_enabled: bool = Field(
        False, description="Enable SQLCipher encryption for host events DB."
    )
    encryption_key_provider: str = Field(
        "dpapi_file",
        description="dpapi_file|file|env (for host events DB).",
    )
    encryption_key_path: Path = Field(
        Path("./secrets/host_events.key"),
        description="Path to store encryption key for host events DB.",
    )
    encryption_env_var: str = Field("AUTOCAPTURE_HOST_EVENTS_KEY")


class OCRConfig(BaseModel):
    queue_maxsize: int = Field(2000, ge=100)
    batch_size: int = Field(32, ge=1)
    max_latency_s: int = Field(900, ge=10)
    engine: str = Field("rapidocr-onnxruntime")
    device: str = Field("cuda", description="cuda|cpu; cuda preferred when available")
    onnx_providers: list[str] = Field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        description="Preferred ONNX Runtime execution providers (ordered).",
    )
    languages: list[str] = Field(default_factory=lambda: ["en"])
    output_format: str = Field("json")
    layout_enabled: bool = Field(
        True, description="Enable deterministic OCR layout reconstruction."
    )
    paddle_ppstructure_enabled: bool = Field(
        False, description="Enable PaddleOCR PP-Structure layout backend."
    )
    paddle_ppstructure_model_dir: Path | None = Field(
        None,
        description=(
            "Local model directory for PaddleOCR PP-Structure (required to avoid downloads)."
        ),
    )
    paddle_ppstructure_use_gpu: bool = Field(
        False, description="Allow GPU execution for PaddleOCR PP-Structure."
    )


class VisionBackendConfig(BaseModel):
    provider: str = Field("ollama", description="ollama|openai_compatible|openai")
    model: str = Field("qwen2.5-vl:7b-instruct")
    base_url: Optional[str] = Field(None)
    api_key: Optional[str] = None
    allow_cloud: bool = Field(False, description="Allow cloud vision calls for this backend.")


class UIGroundingConfig(BaseModel):
    enabled: bool = Field(False, description="Enable UI grounding extraction.")
    backend: str = Field(
        "qwen_vl_ui_prompt",
        description="UI grounding backend (qwen_vl_ui_prompt|ui_venus).",
    )
    vlm: VisionBackendConfig = VisionBackendConfig()

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        allowed = {"qwen_vl_ui_prompt", "ui_venus"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"ui_grounding.backend must be one of {sorted(allowed)}")
        return normalized


class VisionExtractConfig(BaseModel):
    engine: str = Field("vlm", description="vlm|rapidocr|deepseek-ocr|disabled")
    fallback_engine: str = Field("rapidocr-onnxruntime")
    tiles_x: int = Field(3, ge=1)
    tiles_y: int = Field(2, ge=1)
    max_tile_px: int = Field(1280, ge=256)
    include_downscaled_full_frame: bool = Field(True)
    vlm: VisionBackendConfig = VisionBackendConfig()
    deepseek_ocr: VisionBackendConfig = Field(
        default_factory=lambda: VisionBackendConfig(
            provider="openai_compatible",
            model="deepseek-ocr",
            base_url=None,
            api_key=None,
        )
    )
    ui_grounding: UIGroundingConfig = UIGroundingConfig()


class EmbedConfig(BaseModel):
    text_model: str = Field("BAAI/bge-base-en-v1.5")
    image_model: str = Field("google/siglip2-so400m-patch14-384")
    text_batch_size: int = Field(256, ge=1)
    image_batch_size: int = Field(32, ge=1)
    use_half_precision: bool = Field(
        True, description="Use float16 embeddings to shrink storage bandwidth."
    )
    schedule_cron: str = Field("0 2 * * *", description="Cron string for nightly batches.")
    model: Optional[str] = Field(None, description="Legacy alias for text_model (deprecated).")
    batch_size: Optional[int] = Field(
        None, description="Legacy alias for text_batch_size (deprecated)."
    )


class RerankerConfig(BaseModel):
    enabled: bool = True
    model: str = Field("BAAI/bge-reranker-v2-m3")
    device: str = Field("auto", description="auto|cuda|cpu; auto prefers CUDA when available")
    top_k: int = Field(100, ge=1)
    batch_size_active: int = Field(8, ge=1)
    batch_size_idle: int = Field(32, ge=1)
    disable_in_active: bool = Field(False)
    disable_in_fullscreen: bool = Field(True)
    force_cpu_in_active: bool = Field(True)


class WorkerConfig(BaseModel):
    data_dir: Path = Field(
        default_factory=default_data_dir,
        description="Local directory for worker databases, indexes, and media.",
    )
    lease_ms: int = Field(60_000, ge=1000)
    ocr_lease_ms: int = Field(60_000, ge=1000)
    embedding_lease_ms: int = Field(60_000, ge=1000)
    poll_interval_s: float = Field(1.0, ge=0.1)
    ocr_backlog_soft_limit: int = Field(
        5000, ge=100, description="Soft limit for OCR backlog throttling."
    )
    ocr_max_attempts: int = Field(5, ge=1)
    ocr_workers: int = Field(
        max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
        ge=1,
        description="Number of OCR ingest workers.",
    )
    embed_workers: int = Field(
        1,
        ge=1,
        description="Number of embedding/indexing workers.",
    )
    agent_workers: int = Field(
        1,
        ge=0,
        description="Number of agent job workers.",
    )
    embedding_max_attempts: int = Field(5, ge=1)
    watchdog_interval_s: float = Field(
        5.0, gt=0.0, description="Worker watchdog polling interval (seconds)."
    )
    max_task_runtime_s: float = Field(
        900.0, gt=0.0, description="Max heartbeat duration before lease reclaim."
    )


class RuntimeAutoPauseConfig(BaseModel):
    enabled: bool = Field(
        True, description="Enable fullscreen auto-pause handling (alias: on_fullscreen)."
    )
    on_fullscreen: bool = Field(True, description="Pause pipeline when fullscreen detected.")
    mode: str = Field(
        "hard",
        description="Pause mode: hard (pause all workers + capture) or soft (capture only).",
    )
    fullscreen_hard_pause_enabled: bool = Field(
        True, description="Allow FULLSCREEN_HARD_PAUSE mode when fullscreen detected."
    )
    release_gpu: bool = Field(
        True,
        description="Release GPU allocations when entering fullscreen hard pause.",
    )
    poll_hz: float = Field(2.0, ge=0.1, description="Fullscreen monitor polling rate (Hz).")

    @model_validator(mode="before")
    @classmethod
    def reconcile_aliases(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if "enabled" not in data and "on_fullscreen" in data:
            data["enabled"] = data.get("on_fullscreen")
        if "on_fullscreen" not in data and "enabled" in data:
            data["on_fullscreen"] = data.get("enabled")
        if "fullscreen_hard_pause_enabled" not in data and "mode" in data:
            mode = str(data.get("mode") or "").strip().lower()
            if mode in {"hard", "soft"}:
                data["fullscreen_hard_pause_enabled"] = mode == "hard"
        if "mode" not in data and "fullscreen_hard_pause_enabled" in data:
            data["mode"] = "hard" if data.get("fullscreen_hard_pause_enabled") else "soft"
        return data

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        allowed = {"hard", "soft"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"runtime.auto_pause.mode must be one of {sorted(allowed)}")
        return normalized


class RuntimeQosProfile(BaseModel):
    ocr_workers: int = Field(1, ge=0)
    embed_workers: int = Field(0, ge=0)
    agent_workers: int = Field(0, ge=0)
    vision_extract: bool = Field(False)
    ui_grounding: bool = Field(False)
    cpu_priority: str = Field("below_normal", description="below_normal|normal")
    ocr_batch_size: int | None = Field(None, ge=1, description="Override OCR batch size.")
    embed_batch_size: int | None = Field(None, ge=1, description="Override embedding batch size.")
    reranker_batch_size: int | None = Field(None, ge=1, description="Override reranker batch size.")
    sleep_ms: int | None = Field(
        None, ge=0, description="Optional sleep budget for paused/idle loops."
    )
    max_batch: int | None = Field(None, ge=0, description="Optional max batch size hint.")
    max_concurrency: int | None = Field(None, ge=0, description="Optional concurrency hint.")
    gpu_policy: str = Field(
        "allow_gpu", description="allow_gpu|prefer_cpu|disallow_gpu|release_on_pause"
    )

    @field_validator("cpu_priority")
    @classmethod
    def validate_cpu_priority(cls, value: str) -> str:
        allowed = {"below_normal", "normal"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"runtime.qos.cpu_priority must be one of {sorted(allowed)}")
        return normalized

    @field_validator("gpu_policy")
    @classmethod
    def validate_gpu_policy(cls, value: str) -> str:
        allowed = {"allow_gpu", "prefer_cpu", "disallow_gpu", "release_on_pause"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"runtime.qos.gpu_policy must be one of {sorted(allowed)}")
        return normalized


class RuntimeQosConfig(BaseModel):
    enabled: bool = Field(True, description="Enable runtime QoS profiles.")
    idle_grace_ms: int = Field(2000, ge=0)
    profile_active: RuntimeQosProfile = Field(
        default_factory=lambda: RuntimeQosProfile(
            ocr_workers=1,
            embed_workers=0,
            agent_workers=0,
            vision_extract=False,
            ui_grounding=False,
            cpu_priority="below_normal",
        )
    )
    profile_idle: RuntimeQosProfile = Field(
        default_factory=lambda: RuntimeQosProfile(
            ocr_workers=4,
            embed_workers=2,
            agent_workers=1,
            vision_extract=True,
            ui_grounding=True,
            cpu_priority="normal",
        )
    )


class RuntimeConfig(BaseModel):
    auto_pause: RuntimeAutoPauseConfig = RuntimeAutoPauseConfig()
    qos: RuntimeQosConfig = RuntimeQosConfig()


class RetentionPolicyConfig(BaseModel):
    video_days: int = Field(3, ge=1)
    roi_days: int = Field(14, ge=1)
    max_media_gb: int = Field(200, ge=1)
    screenshot_ttl_days: int = Field(
        60, ge=1, description="Days to keep raw screenshots before pruning."
    )
    protect_recent_minutes: int = Field(
        60, ge=1, description="Protect media newer than this window from pruning."
    )


class StorageQuotaConfig(BaseModel):
    image_quota_gb: int = Field(2500, ge=10)
    prune_grace_days: int = Field(90, ge=1)
    prune_batch: int = Field(2000, ge=10)


class DatabaseConfig(BaseModel):
    url: str = Field(
        default_factory=lambda: _default_database_url(),
        description="SQLAlchemy URL for local metadata storage (SQLite by default).",
    )
    echo: bool = False
    pool_size: int = Field(10, ge=1)
    max_overflow: int = Field(10, ge=0)
    sqlite_busy_timeout_ms: int = Field(5000, ge=0)
    sqlite_wal: bool = True
    sqlite_synchronous: str = Field("NORMAL")
    encryption_enabled: bool = Field(
        False, description="Enable SQLCipher encryption for SQLite databases."
    )
    encryption_provider: str = Field(
        "dpapi_file",
        description="dpapi_file|file|env (key storage for SQLCipher).",
    )
    encryption_key_name: str = Field(
        "autocapture/sqlcipher-key",
        description="Key name for DPAPI-backed storage.",
    )
    encryption_key_path: Path = Field(
        Path("./secrets/sqlcipher.key"),
        description="Path for SQLCipher key (file provider).",
    )
    encryption_env_var: str = Field(
        "AUTOCAPTURE_SQLCIPHER_KEY",
        description="Environment variable for SQLCipher key hex.",
    )
    secure_mode_required: bool = Field(
        True,
        description="Refuse to start if SQLite is not encrypted and secure mode is required.",
    )
    allow_insecure_dev: bool = Field(
        default_factory=_default_allow_insecure_dev,
        description="Allow insecure (unencrypted) SQLite for dev/test only.",
    )
    require_tls_for_remote: bool = Field(
        True,
        description="Require TLS for remote Postgres connections.",
    )

    @field_validator("sqlite_synchronous")
    @classmethod
    def validate_sqlite_synchronous(cls, value: str) -> str:
        allowed = {"NORMAL", "FULL", "OFF"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"sqlite_synchronous must be one of {sorted(allowed)}; got {value!r}")
        return upper


class QdrantConfig(BaseModel):
    enabled: bool = True
    url: str = Field("http://127.0.0.1:6333")
    binary_path: Optional[Path] = Field(
        None, description="Optional path to qdrant.exe for sidecar management."
    )
    text_collection: str = Field("text_spans")
    image_collection: str = Field("image_tiles")
    spans_v2_collection: str = Field("spans_v2")
    text_vector_size: int = Field(768, ge=64)
    image_vector_size: int = Field(768, ge=64)
    late_vector_size: int = Field(128, ge=32)
    distance: str = Field("Cosine")
    hnsw_ef_construct: int = Field(128, ge=1)
    hnsw_m: int = Field(16, ge=1)
    search_ef: int = Field(64, ge=1)
    require_tls_for_remote: bool = Field(
        True,
        description="Require HTTPS for remote Qdrant connections.",
    )

    collection_name: Optional[str] = Field(
        None, description="Legacy alias for text_collection (deprecated)."
    )
    vector_size: Optional[int] = Field(
        None, description="Legacy alias for text_vector_size (deprecated)."
    )


class FFmpegConfig(BaseModel):
    enabled: bool = Field(True)
    require_bundled: bool = Field(True)
    explicit_path: Optional[Path] = Field(None, description="Explicit ffmpeg binary path.")
    allow_system: bool = Field(True, description="Allow falling back to system PATH ffmpeg.")
    allow_disable: bool = Field(
        True,
        description="Allow disabling video capture when ffmpeg is missing.",
    )
    relative_path_candidates: list[str] = Field(
        default_factory=lambda: ["ffmpeg/bin/ffmpeg.exe", "ffmpeg/ffmpeg.exe"]
    )


class EncryptionConfig(BaseModel):
    enabled: bool = Field(True)
    key_provider: str = Field(
        default_factory=lambda: (
            "windows-credential-manager"
            if sys.platform == "win32"
            else "file:./data/autocapture.key"
        ),
        description="Strategy to fetch AES key (file, env, kms).",
    )
    key_name: str = Field("autocapture/nas-aes-key")
    chunk_size: int = Field(
        4 * 1024 * 1024,
        ge=64 * 1024,
        description="Chunk size for streaming encryption uploads.",
    )


class TelemetryConfig(BaseModel):
    capture_payloads: str = Field(
        "none", description="Payload capture mode (none|redacted|full)."
    )
    exporter: str = Field("none", description="Exporter (none|otlp).")
    otlp_endpoint: Optional[str] = None
    otlp_protocol: str = Field("http/protobuf", description="OTLP protocol (http/protobuf).")
    allow_cloud_export: bool = Field(False)
    max_attr_len: int = Field(128, ge=32, le=1024)

    @field_validator("capture_payloads")
    @classmethod
    def validate_capture_payloads(cls, value: str) -> str:
        allowed = {"none", "redacted", "full"}
        if value not in allowed:
            raise ValueError(f"telemetry.capture_payloads must be one of {sorted(allowed)}")
        return value

    @field_validator("exporter")
    @classmethod
    def validate_exporter(cls, value: str) -> str:
        allowed = {"none", "otlp"}
        if value not in allowed:
            raise ValueError(f"telemetry.exporter must be one of {sorted(allowed)}")
        return value

    @field_validator("otlp_protocol")
    @classmethod
    def validate_otlp_protocol(cls, value: str) -> str:
        allowed = {"http/protobuf"}
        if value not in allowed:
            raise ValueError(f"telemetry.otlp_protocol must be one of {sorted(allowed)}")
        return value


class ObservabilityConfig(BaseModel):
    prometheus_bind_host: str = Field("127.0.0.1")
    prometheus_port: int = Field(9005, ge=1024, le=65535)
    prometheus_port_fallbacks: int = Field(10, ge=0, le=100)
    prometheus_fail_fast: bool = Field(False)
    grafana_url: Optional[str] = None
    enable_gpu_stats: bool = Field(True)
    telemetry: TelemetryConfig = TelemetryConfig()


class FeatureFlagsConfig(BaseModel):
    enable_frame_record_v1: bool = Field(
        True, description="Enable FrameRecord v1 fields for new captures."
    )
    enable_frame_hash: bool = Field(
        True, description="Compute frame_hash for new captures (post-masking)."
    )
    enable_normalized_indexing: bool = Field(
        True, description="Index normalized OCR text alongside raw text."
    )
    enable_thresholding: bool = Field(
        True, description="Apply retrieval thresholds and no-evidence responses."
    )
    enable_retention_prune: bool = Field(
        False, description="Enable retention-aware index pruning and scans."
    )
    enable_otel: bool = Field(True, description="Enable OpenTelemetry tracing/metrics.")
    enable_memory_service_write_hook: bool = Field(
        False, description="Enable Memory Service write hook in capture pipeline."
    )
    enable_memory_service_read_hook: bool = Field(
        False, description="Enable Memory Service read hook in context building."
    )


class Next10Config(BaseModel):
    enabled: bool = Field(True, description="Enable SPEC-260117 Next-10 enforcement.")
    policy_defaults_path: Path = Field(
        Path("config/defaults/policy.json"),
        description="Path to default privacy policy JSON.",
    )
    budgets_defaults_path: Path = Field(
        Path("config/defaults/budgets.json"),
        description="Path to default stage budgets JSON.",
    )
    tiers_defaults_path: Path = Field(
        Path("config/defaults/tiers.json"),
        description="Path to default retrieval tiers JSON.",
    )
    tier_stats_window: int = Field(200, ge=50, description="Window size for tier stats.")
    tier_help_rate_min: float = Field(0.05, ge=0.0, le=1.0)
    index_versions: dict[str, str] = Field(
        default_factory=lambda: {"event_fts": "v1", "span_fts": "v1", "vector": "v1"},
        description="Pinned index/engine versions for provenance records.",
    )


class APIConfig(BaseModel):
    bind_host: str = Field("127.0.0.1", description="Bind host for the local API server.")
    port: int = Field(8008, ge=1024, le=65535)
    require_api_key: bool = Field(False)
    api_key: Optional[str] = None
    bridge_token: Optional[str] = Field(
        None, description="Optional bridge token for ingest-only authorization."
    )
    rate_limit_rps: float = Field(2.0, gt=0.0)
    rate_limit_burst: int = Field(5, ge=1)
    max_page_size: int = Field(200, ge=1)
    default_page_size: int = Field(50, ge=1)
    max_query_chars: int = Field(2000, ge=1)
    max_context_k: int = Field(50, ge=1)

    @model_validator(mode="after")
    def validate_paging(self) -> "APIConfig":
        if self.default_page_size > self.max_page_size:
            raise ValueError("api.default_page_size must be <= api.max_page_size")
        return self


class ModeConfig(BaseModel):
    mode: str = Field("local", description="local or remote")
    overlay_interface: Optional[str] = Field(
        None, description="Overlay interface name (tailscale0, wg0)."
    )
    https_enabled: bool = Field(False)
    tls_cert_path: Optional[Path] = Field(None)
    tls_key_path: Optional[Path] = Field(None)
    google_oauth_client_id: Optional[str] = Field(None)
    google_oauth_client_secret: Optional[str] = Field(None)
    google_allowed_emails: list[str] = Field(default_factory=list)


class PluginsConfig(BaseModel):
    directory: Optional[Path] = Field(
        None,
        description="Optional plugin directory override (defaults to data_dir/plugins).",
    )
    safe_mode: bool = Field(
        False,
        description="Load built-in plugins only (ignore external plugins).",
    )


class ProviderRoutingConfig(BaseModel):
    capture: str = Field("local")
    ocr: str = Field("local")
    embedding: str = Field("local")
    retrieval: str = Field("local")
    vector_backend: str = Field("local", description="Vector backend plugin id.")
    spans_v2_backend: str = Field("local", description="Spans v2 backend plugin id.")
    table_extractor: str = Field("disabled", description="Table extractor plugin id.")
    reranker: str = Field("disabled")
    compressor: str = Field("extractive")
    verifier: str = Field("rules")
    llm: str = Field("gateway")


class PrivacyConfig(BaseModel):
    cloud_enabled: bool = Field(False)
    sanitize_default: bool = Field(True)
    extractive_only_default: bool = Field(True)
    token_vault_enabled: bool = Field(
        False, description="Store reversible tokens in the encrypted token vault."
    )
    allow_token_vault_decrypt: bool = Field(
        False, description="Allow decrypting token vault values via API."
    )
    allow_cloud_images: bool = Field(
        False,
        description="Allow sending images to cloud vision endpoints.",
    )
    paused: bool = Field(False)
    snooze_until_utc: dt.datetime | None = None
    exclude_monitors: list[str] = Field(default_factory=list)
    exclude_processes: list[str] = Field(default_factory=list)
    exclude_window_title_regex: list[str] = Field(default_factory=list)
    exclude_regions: list[dict] = Field(default_factory=list)
    mask_regions: list[dict] = Field(default_factory=list)


class PolicyConfig(BaseModel):
    enforce_prompt_injection: bool = Field(
        False, description="Enable prompt-injection enforcement in PolicyEnvelope."
    )
    prompt_injection_warn_threshold: float = Field(0.7, ge=0.0, le=1.0)
    prompt_injection_block_threshold: float = Field(0.9, ge=0.0, le=1.0)
    structured_output_mode: str = Field(
        "none", description="Structured output mode (none|json_schema|regex|choice|grammar)."
    )
    max_context_chars: Optional[int] = Field(
        None, description="Hard cap on context pack chars (None disables)."
    )
    max_evidence_items: Optional[int] = Field(
        None, description="Hard cap on evidence items (None disables)."
    )

    @field_validator("structured_output_mode")
    @classmethod
    def validate_structured_output_mode(cls, value: str) -> str:
        allowed = {"none", "json_schema", "regex", "choice", "grammar"}
        if value not in allowed:
            raise ValueError(f"policy.structured_output_mode must be one of {sorted(allowed)}")
        return value


class OutputConfig(BaseModel):
    format: str = Field("text", description="text|json|tron")
    context_pack_format: str = Field("json", description="json|tron")
    allow_tron_compression: bool = Field(
        False,
        description="Allow TRON context packs for cloud LLM calls when enabled.",
    )

    @field_validator("format")
    @classmethod
    def validate_format(cls, value: str) -> str:
        allowed = {"text", "json", "tron"}
        if value not in allowed:
            raise ValueError(f"output.format must be one of {sorted(allowed)}")
        return value

    @field_validator("context_pack_format")
    @classmethod
    def validate_context_pack_format(cls, value: str) -> str:
        allowed = {"json", "tron"}
        if value not in allowed:
            raise ValueError(f"output.context_pack_format must be one of {sorted(allowed)}")
        return value


class CacheConfig(BaseModel):
    enabled: bool = Field(False)
    path: Optional[Path] = Field(None, description="Cache file path (sqlite).")
    max_entries: int = Field(10_000, ge=0)
    ttl_s: int = Field(86_400, ge=0)
    prune_interval_s: int = Field(3_600, ge=0)
    redact_on_cloud: bool = Field(True)


class TimeConfig(BaseModel):
    timezone: Optional[str] = Field(
        None, description="IANA timezone override (e.g., America/Denver)."
    )


class SecurityConfig(BaseModel):
    local_unlock_enabled: bool = Field(
        True, description="Require local unlock session for sensitive endpoints."
    )
    session_ttl_seconds: int = Field(300, ge=30, description="Unlock session TTL in seconds.")
    secure_mode: bool = Field(
        True,
        description="Fail closed on checksum mismatches or unknown native extensions.",
    )
    provider: str = Field(
        default_factory=_default_security_provider,
        description="windows_hello|cred_ui|test|disabled",
    )


class RetrievalConfig(BaseModel):
    v2_enabled: bool = Field(True, description="Enable RetrievalService v2 enhancements.")
    use_spans_v2: bool = Field(True, description="Use spans_v2 Qdrant collection.")
    sparse_enabled: bool = Field(True, description="Enable learned sparse retrieval.")
    late_enabled: bool = Field(True, description="Enable late-interaction reranking.")
    fusion_enabled: bool = Field(True, description="Enable multi-query fusion (RRF).")
    multi_query_enabled: bool = Field(True, description="Enable multi-query query rewrites.")
    rrf_enabled: bool = Field(True, description="Enable RRF fusion for multi-query retrieval.")
    lexical_min_score: float = Field(0.15, ge=0.0, le=1.0)
    dense_min_score: float = Field(0.15, ge=0.0, le=1.0)
    rerank_min_score: float = Field(0.15, ge=0.0, le=1.0)
    sparse_min_score: float = Field(0.1, ge=0.0, le=1.0)
    late_min_score: float = Field(0.1, ge=0.0, le=1.0)
    fusion_rewrites: int = Field(4, ge=1, le=8)
    fusion_rrf_k: int = Field(60, ge=1)
    fusion_confidence_min: float = Field(0.65, ge=0.0, le=1.0)
    fusion_rank_gap_min: float = Field(0.1, ge=0.0, le=1.0)
    sparse_model: str = Field("hash-splade")
    late_max_days: int = Field(30, ge=1)
    late_max_spans_per_event: int = Field(128, ge=1)
    late_text_max_chars: int = Field(200, ge=1)
    late_candidate_k: int = Field(100, ge=1)
    late_rerank_k: int = Field(50, ge=1)
    late_stage1_enabled: bool = Field(
        True, description="Enable late-interaction stage-1 retrieval for narrow windows."
    )
    late_stage1_max_days: int = Field(
        7, ge=1, description="Max time window (days) for late stage-1 retrieval."
    )
    late_stage1_k: int = Field(50, ge=1, description="Top-K for late stage-1 retrieval.")
    rewrite_max_chars: int = Field(200, ge=10)
    speculative_enabled: bool = Field(True, description="Enable speculative draft/verify.")
    speculative_draft_k: int = Field(6, ge=1)
    speculative_final_k: int = Field(12, ge=1)
    traces_enabled: bool = Field(True, description="Persist retrieval traces.")
    graph_adapters: "GraphAdaptersConfig" = Field(
        default_factory=lambda: GraphAdaptersConfig(),
        description="Optional graph retrieval adapters.",
    )


class GraphAdapterConfig(BaseModel):
    enabled: bool = Field(True)
    base_url: Optional[str] = Field("http://127.0.0.1:8020")
    timeout_s: float = Field(10.0, gt=0.0)
    max_results: int = Field(20, ge=1)


class GraphAdaptersConfig(BaseModel):
    graphrag: GraphAdapterConfig = GraphAdapterConfig()
    hypergraphrag: GraphAdapterConfig = GraphAdapterConfig()
    hyperrag: GraphAdapterConfig = GraphAdapterConfig()


class MemoryStorageConfig(BaseModel):
    root_dir: Path = Field(
        default_factory=default_memory_dir,
        description="Root directory for the memory store (env: AUTOCAPTURE_MEMORY_DIR).",
    )
    db_filename: str = Field("memory.sqlite3")
    artifacts_dir: str = Field("artifacts")
    snapshots_dir: str = Field("snapshots")
    require_fts: bool = Field(
        True, description="Fail memory retrieval if SQLite FTS5 is unavailable."
    )
    snapshot_retention_days: int = Field(90, ge=1)


class MemoryPolicyConfig(BaseModel):
    blocked_labels: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    redact_patterns: list[str] = Field(default_factory=list)
    redact_token: str = Field("[REDACTED]", min_length=1)


class MemorySpanConfig(BaseModel):
    max_chars: int = Field(800, ge=100)
    min_chars: int = Field(200, ge=0)


class MemoryRetrievalConfig(BaseModel):
    enabled: bool = Field(True)
    default_k: int = Field(8, ge=1)
    max_k: int = Field(24, ge=1)
    recency_half_life_days: int = Field(30, ge=1)


class MemoryCompilerConfig(BaseModel):
    max_total_chars: int = Field(6000, ge=500)
    max_chars_per_span: int = Field(1200, ge=100)
    max_spans: int = Field(12, ge=1)
    max_memory_items: int = Field(32, ge=1)


class MemoryHotnessHalfLivesConfig(BaseModel):
    fast_seconds: int = Field(3600, ge=1)
    mid_seconds: int = Field(6 * 3600, ge=1)
    warm_seconds: int = Field(24 * 3600, ge=1)
    cool_seconds: int = Field(7 * 24 * 3600, ge=1)


class MemoryHotnessWeightsConfig(BaseModel):
    fast: float = Field(0.4, ge=0.0)
    mid: float = Field(0.3, ge=0.0)
    warm: float = Field(0.2, ge=0.0)
    cool: float = Field(0.1, ge=0.0)


class MemoryHotnessThresholdsConfig(BaseModel):
    hot: float = Field(0.75, ge=0.0)
    recent: float = Field(0.5, ge=0.0)
    warm: float = Field(0.25, ge=0.0)
    cool: float = Field(0.1, ge=0.0)


class MemoryHotnessQuotasConfig(BaseModel):
    hot: float = Field(0.4, ge=0.0)
    recent: float = Field(0.3, ge=0.0)
    warm: float = Field(0.2, ge=0.0)
    cool: float = Field(0.1, ge=0.0)


class MemoryHotnessRateLimitConfig(BaseModel):
    enabled: bool = Field(True)
    min_interval_ms: int = Field(60_000, ge=0)


class MemoryHotnessRetentionConfig(BaseModel):
    event_max_age_days: int = Field(30, ge=1)
    event_max_count: int = Field(50_000, ge=0)


class MemoryHotnessConfig(BaseModel):
    enabled: bool = Field(True)
    mode_default: str = Field("off", description="off|as_of|dynamic")
    scope_default: str = Field("default")
    half_lives: MemoryHotnessHalfLivesConfig = MemoryHotnessHalfLivesConfig()
    weights: MemoryHotnessWeightsConfig = MemoryHotnessWeightsConfig()
    thresholds: MemoryHotnessThresholdsConfig = MemoryHotnessThresholdsConfig()
    quotas: MemoryHotnessQuotasConfig = MemoryHotnessQuotasConfig()
    rate_limit: MemoryHotnessRateLimitConfig = MemoryHotnessRateLimitConfig()
    retention: MemoryHotnessRetentionConfig = MemoryHotnessRetentionConfig()
    allowed_signals: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "manual_touch": ["cli", "api"],
            "pin_set": ["cli", "api"],
            "pin_unset": ["cli", "api"],
        }
    )

    @field_validator("mode_default")
    @classmethod
    def validate_mode_default(cls, value: str) -> str:
        allowed = {"off", "as_of", "dynamic"}
        if value not in allowed:
            raise ValueError(f"memory.hotness.mode_default must be one of {sorted(allowed)}")
        return value

    @field_validator("allowed_signals")
    @classmethod
    def validate_allowed_signals(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        if not value:
            raise ValueError("memory.hotness.allowed_signals must not be empty")
        for signal, sources in value.items():
            if not signal.strip():
                raise ValueError("memory.hotness.allowed_signals keys must be non-empty")
            if not sources:
                raise ValueError(f"memory.hotness.allowed_signals[{signal!r}] must list sources")
        return value

    @model_validator(mode="after")
    def validate_thresholds(self) -> "MemoryHotnessConfig":
        thresholds = self.thresholds
        if not (thresholds.hot >= thresholds.recent >= thresholds.warm >= thresholds.cool):
            raise ValueError("memory.hotness.thresholds must be descending hot>=recent>=warm>=cool")
        return self


class MemoryConfig(BaseModel):
    enabled: bool = Field(True)
    api_context_pack_enabled: bool = Field(
        True, description="Include memory snapshots in /api/context-pack responses."
    )
    storage: MemoryStorageConfig = MemoryStorageConfig()
    policy: MemoryPolicyConfig = MemoryPolicyConfig()
    spans: MemorySpanConfig = MemorySpanConfig()
    retrieval: MemoryRetrievalConfig = MemoryRetrievalConfig()
    compiler: MemoryCompilerConfig = MemoryCompilerConfig()
    hotness: MemoryHotnessConfig = MemoryHotnessConfig()


class PresetConfig(BaseModel):
    active_preset: str = Field(
        "privacy_first", description="privacy_first or high_fidelity preset name."
    )


class PromptOpsConfig(BaseModel):
    enabled: bool = Field(False)
    schedule_cron: str = Field("0 6 * * 1")
    sources: list[str] = Field(default_factory=list)
    github_token: Optional[str] = Field(None)
    github_repo: Optional[str] = Field(None)
    acceptance_tolerance: float = Field(0.02, ge=0.0, le=1.0)
    max_attempts: int = Field(3, ge=1)
    early_stop_on_first_accept: bool = Field(False)
    max_iterations: int = Field(3, ge=1)
    max_llm_attempts_per_prompt: int = Field(3, ge=1)
    eval_repeats: int = Field(3, ge=1)
    eval_aggregation: str = Field("worst_case")
    require_improvement: bool = Field(True)
    min_improve_verifier_pass_rate: float = Field(0.01)
    min_improve_citation_coverage: float = Field(0.01)
    min_improve_refusal_rate: float = Field(0.01)
    min_improve_mean_latency_ms: float = Field(100.0)
    min_delta_verifier_pass_rate: float = Field(0.02)
    min_delta_citation_coverage: float = Field(0.02)
    min_delta_refusal_rate: float = Field(0.02)
    min_delta_latency_ms: float = Field(200.0)
    tolerance_citation_coverage: float = Field(0.02)
    tolerance_refusal_rate: float = Field(0.02)
    tolerance_mean_latency_ms: float | None = Field(250.0)
    tolerance_latency_ms: float = Field(250.0)
    min_verifier_pass_rate: float = Field(0.60)
    min_citation_coverage: float = Field(0.60)
    max_refusal_rate: float = Field(0.30)
    max_mean_latency_ms: float = Field(15000.0)
    max_prompt_chars: int = Field(12000, ge=1)
    max_mean_latency_ms: float | None = Field(15000.0)
    pr_cooldown_hours: float = Field(0.0, ge=0.0)
    max_source_bytes: int = Field(1_048_576, ge=1)
    max_source_excerpt_chars: int = Field(2000, ge=1)
    max_sources: int = Field(32, ge=1)

    @model_validator(mode="before")
    @classmethod
    def apply_promptops_compat(cls, values: dict) -> dict:
        if not isinstance(values, dict):
            return values
        if "max_attempts" not in values and "max_iterations" in values:
            values["max_attempts"] = values["max_iterations"]
        if "tolerance_mean_latency_ms" not in values and "tolerance_latency_ms" in values:
            values["tolerance_mean_latency_ms"] = values["tolerance_latency_ms"]
        if (
            "min_improve_verifier_pass_rate" not in values
            and "min_delta_verifier_pass_rate" in values
        ):
            values["min_improve_verifier_pass_rate"] = values["min_delta_verifier_pass_rate"]
        if (
            "min_improve_citation_coverage" not in values
            and "min_delta_citation_coverage" in values
        ):
            values["min_improve_citation_coverage"] = values["min_delta_citation_coverage"]
        if "min_improve_refusal_rate" not in values and "min_delta_refusal_rate" in values:
            values["min_improve_refusal_rate"] = values["min_delta_refusal_rate"]
        if "min_improve_mean_latency_ms" not in values and "min_delta_latency_ms" in values:
            values["min_improve_mean_latency_ms"] = values["min_delta_latency_ms"]
        return values

    @field_validator("eval_aggregation")
    @classmethod
    def validate_eval_aggregation(cls, value: str) -> str:
        if value not in {"worst_case", "mean"}:
            raise ValueError("promptops.eval_aggregation must be 'worst_case' or 'mean'")
        return value


class TemplateHardeningConfig(BaseModel):
    enabled: bool = Field(True, description="Enable template hardening checks.")
    log_provenance: bool = Field(True, description="Log template provenance hashes on load.")


class UIConfig(BaseModel):
    overlay_citations_enabled: bool = Field(
        False, description="Enable citation overlay rendering in the UI/API."
    )


class OverlayHotkeySpec(BaseModel):
    modifiers: list[str] = Field(default_factory=list)
    key: str = Field("F24", description="Virtual key name (e.g., F24, SPACE, O).")


class OverlayTrackerHotkeysConfig(BaseModel):
    toggle_overlay: OverlayHotkeySpec = Field(
        default_factory=lambda: OverlayHotkeySpec(modifiers=["ctrl", "shift"], key="O")
    )
    interactive_mode: OverlayHotkeySpec = Field(
        default_factory=lambda: OverlayHotkeySpec(modifiers=["ctrl", "shift"], key="I")
    )
    project_cycle: OverlayHotkeySpec = Field(
        default_factory=lambda: OverlayHotkeySpec(modifiers=["ctrl", "shift"], key="P")
    )
    toggle_running: OverlayHotkeySpec = Field(
        default_factory=lambda: OverlayHotkeySpec(modifiers=["ctrl", "shift"], key="R")
    )
    rename: OverlayHotkeySpec = Field(
        default_factory=lambda: OverlayHotkeySpec(modifiers=["ctrl", "shift"], key="N")
    )
    snooze: OverlayHotkeySpec = Field(
        default_factory=lambda: OverlayHotkeySpec(modifiers=["ctrl", "shift"], key="S")
    )
    snooze_minutes: list[int] = Field(
        default_factory=lambda: [15, 60, 240],
        description="Preset durations for snooze hotkey.",
    )

    @field_validator("snooze_minutes")
    @classmethod
    def _validate_snooze_minutes(cls, value: list[int]) -> list[int]:
        if any(minutes <= 0 for minutes in value):
            raise ValueError("overlay_tracker.hotkeys.snooze_minutes must be positive")
        return value


class OverlayTrackerCollectorConfig(BaseModel):
    foreground_enabled: bool = True
    input_enabled: bool = True
    fallback_foreground_poll_ms: int = Field(1000, ge=50, le=2000)
    input_poll_ms: int = Field(250, ge=50, le=2000)
    input_debounce_ms: int = Field(500, ge=50, le=5000)

    @model_validator(mode="after")
    def _validate_debounce(self) -> "OverlayTrackerCollectorConfig":
        if self.input_debounce_ms < self.input_poll_ms:
            raise ValueError(
                "overlay_tracker.collectors.input_debounce_ms must be >= input_poll_ms"
            )
        return self


class OverlayTrackerUiConfig(BaseModel):
    enabled: bool = True
    dock: str = Field("right", description="right|left")
    width_px: int = Field(320, ge=200, le=480)
    click_through_default: bool = True
    interactive_timeout_seconds: int = Field(10, ge=1, le=120)
    refresh_ms: int = Field(1000, ge=100, le=5000)
    auto_hide_fullscreen: bool = True

    @field_validator("dock")
    @classmethod
    def _validate_dock(cls, value: str) -> str:
        if value not in {"left", "right"}:
            raise ValueError("overlay_tracker.ui.dock must be 'left' or 'right'")
        return value


class OverlayTrackerPolicyConfig(BaseModel):
    deny_processes: list[str] = Field(default_factory=list)
    max_window_title_length: int = Field(512, ge=64, le=2048)


class OverlayTrackerRetentionConfig(BaseModel):
    event_days: int = Field(14, ge=1, le=365)
    event_cap: int = Field(200000, ge=1000, le=1000000)


class OverlayTrackerUrlPluginConfig(BaseModel):
    enabled: bool = False
    allow_browsers: list[str] = Field(default_factory=lambda: ["chrome.exe", "msedge.exe"])
    allow_domains: list[str] = Field(default_factory=list)
    token_rules: list[dict] = Field(default_factory=list)


class OverlayTrackerConfig(BaseModel):
    enabled: bool = False
    platforms: list[str] = Field(default_factory=lambda: ["windows"])
    stale_after_hours: float = Field(48.0, gt=0.0)
    hotness_half_life_minutes: float = Field(30.0, gt=0.0)
    collectors: OverlayTrackerCollectorConfig = OverlayTrackerCollectorConfig()
    ui: OverlayTrackerUiConfig = OverlayTrackerUiConfig()
    hotkeys: OverlayTrackerHotkeysConfig = OverlayTrackerHotkeysConfig()
    policy: OverlayTrackerPolicyConfig = OverlayTrackerPolicyConfig()
    retention: OverlayTrackerRetentionConfig = OverlayTrackerRetentionConfig()
    url_plugin: OverlayTrackerUrlPluginConfig = OverlayTrackerUrlPluginConfig()


class LLMConfig(BaseModel):
    provider: str = Field("ollama", description="ollama|openai|openai_compatible")
    ollama_url: str = Field("http://127.0.0.1:11434")
    ollama_model: str = Field("llama3")
    ollama_keep_alive_s: float | None = Field(
        None, ge=0.0, description="Optional Ollama keep_alive duration (seconds)."
    )
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    openai_model: str = Field("gpt-4.1-mini")
    openai_compatible_base_url: Optional[str] = Field(
        None, description="OpenAI-compatible base URL (llama.cpp/Open WebUI/etc)."
    )
    openai_compatible_api_key: Optional[str] = Field(
        None, description="Optional API key for OpenAI-compatible server."
    )
    openai_compatible_model: str = Field("llama3")
    timeout_s: float = Field(60.0, gt=0.0)
    retries: int = Field(3, ge=0, le=10)
    prompt_strategy_default: str = Field(
        "repeat_2x",
        description=(
            "Prompt strategy default: baseline|repeat_2x|repeat_3x|step_by_step|"
            "step_by_step_plus_repeat_2x."
        ),
    )
    prompt_repeat_factor: int = Field(
        2, ge=2, le=3, description="Repeat factor for prompt repetition strategies."
    )
    enable_step_by_step: bool = Field(False, description="Enable optional step-by-step prompting.")
    step_by_step_phrase: str = Field(
        "Let's think step by step.",
        description="Zero-shot-CoT trigger phrase for step-by-step mode.",
    )
    step_by_step_two_stage: bool = Field(
        False, description="Enable two-stage prompting for step-by-step mode."
    )
    max_prompt_chars_for_repetition: int = Field(
        12000, ge=1, description="Max prompt chars allowed before repetition is disabled."
    )
    max_tokens_headroom: int = Field(
        512, ge=0, description="Reserved response tokens when checking context limits."
    )
    max_context_tokens: int | None = Field(
        8192, ge=256, description="Approximate model context window for safety checks."
    )
    force_no_reasoning: bool = Field(
        False, description="Force non-reasoning pathway (disables step-by-step)."
    )
    strategy_auto_mode: bool = Field(
        True,
        description=(
            "Automatically choose repeat strategy when reasoning is disabled, "
            "baseline otherwise."
        ),
    )
    prompt_repetition_delimiter: str = Field(
        "\n\n---\n\n", description="Delimiter inserted between repeated prompts."
    )
    store_prompt_transforms: bool = Field(
        False, description="Persist transformed prompts to disk for local debugging."
    )
    prompt_store_redaction: bool = Field(
        True, description="Redact stored prompts when store_prompt_transforms is enabled."
    )
    prompt_repetition: bool = Field(
        False,
        description=(
            "Legacy: repeat non-system prompt content once; use prompt_strategy_default " "instead."
        ),
    )

    @field_validator("prompt_strategy_default")
    @classmethod
    def validate_prompt_strategy_default(cls, value: str) -> str:
        allowed = {
            "baseline",
            "repeat_2x",
            "repeat_3x",
            "step_by_step",
            "step_by_step_plus_repeat_2x",
        }
        if value not in allowed:
            raise ValueError(f"llm.prompt_strategy_default must be one of {sorted(allowed)}")
        return value

    @field_validator("prompt_repeat_factor")
    @classmethod
    def validate_prompt_repeat_factor(cls, value: int) -> int:
        if value not in {2, 3}:
            raise ValueError("llm.prompt_repeat_factor must be 2 or 3")
        return value

    @model_validator(mode="after")
    def apply_prompt_strategy_compat(self) -> "LLMConfig":
        if self.prompt_repetition and self.prompt_strategy_default == "repeat_2x":
            return self
        if self.prompt_repetition and self.prompt_strategy_default == "baseline":
            self.prompt_strategy_default = "repeat_2x"
        return self


class LLMGovernorConfig(BaseModel):
    enabled: bool = Field(True, description="Enable adaptive LLM concurrency governor.")
    min_in_flight: int = Field(1, ge=1, description="Minimum concurrent LLM requests.")
    max_in_flight: int = Field(4, ge=1, description="Maximum concurrent LLM requests.")
    low_pressure_threshold: float = Field(
        0.45, ge=0.0, le=1.0, description="Pressure below this increases concurrency."
    )
    high_pressure_threshold: float = Field(
        0.85, ge=0.0, le=1.0, description="Pressure above this decreases concurrency."
    )
    adjust_interval_s: float = Field(
        5.0, ge=0.1, description="Minimum seconds between concurrency adjustments."
    )


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = Field(5, ge=1)
    reset_timeout_s: float = Field(30.0, gt=0.0)


class ProviderSpec(BaseModel):
    id: str
    type: str = Field(
        "openai_compatible", description="Provider type (openai|openai_compatible|ollama|gateway)."
    )
    base_url: Optional[str] = None
    api_key_env: Optional[str] = Field(
        None, description="Env var name holding the API key (preferred over config secrets)."
    )
    api_key: Optional[str] = Field(
        None,
        description="Optional inline API key. Avoid using this in committed configs.",
    )
    timeout_s: float = Field(60.0, gt=0.0)
    retries: int = Field(3, ge=0, le=10)
    headers: dict[str, str] = Field(default_factory=dict)
    allow_cloud: bool = Field(False, description="Allow cloud usage for this provider.")
    circuit_breaker: CircuitBreakerConfig = CircuitBreakerConfig()
    max_concurrency: int = Field(4, ge=1, description="Per-provider concurrency cap.")


class QuantizationConfig(BaseModel):
    method: str = Field("none", description="none|awq|gptq|int8|int4|fp16")
    bits: Optional[int] = Field(None, ge=1, le=16)
    group_size: Optional[int] = Field(None, ge=1)

    @field_validator("method")
    @classmethod
    def validate_method(cls, value: str) -> str:
        allowed = {"none", "awq", "gptq", "int8", "int4", "fp16"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"quantization.method must be one of {sorted(allowed)}")
        return normalized


class LoRAConfig(BaseModel):
    enabled: bool = False
    allowed_adapters: list[str] = Field(default_factory=list)
    load_policy: str = Field("lazy", description="lazy|preload|pinned")
    enforce_allowlist: bool = True


class ModelRuntimeConfig(BaseModel):
    device: str = Field("auto", description="auto|cuda|cpu")
    max_memory_utilization: Optional[float] = Field(None, ge=0.0, le=1.0)


class ModelSpec(BaseModel):
    id: str
    provider_id: str
    upstream_model_name: str
    context_tokens: int = Field(8192, ge=256)
    supports_json: bool = False
    supports_tools: bool = False
    supports_vision: bool = False
    quantization: Optional[QuantizationConfig] = None
    lora: Optional[LoRAConfig] = None
    runtime: ModelRuntimeConfig = ModelRuntimeConfig()
    lmcache_enabled: bool = False


class StageSamplingConfig(BaseModel):
    temperature: float = Field(0.2, ge=0.0, le=1.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, ge=1)
    seed: Optional[int] = Field(None, ge=0)


class StageRequirementsConfig(BaseModel):
    require_json: bool = False
    require_citations: bool = False
    claims_schema: Optional[str] = Field(
        None, description="claims_json_v1 for claim-level outputs."
    )


class DecodeConfig(BaseModel):
    strategy: str = Field("standard", description="standard|swift|lookahead|medusa")
    backend_provider_id: Optional[str] = None
    max_concurrency: int = Field(2, ge=1)


class StagePolicy(BaseModel):
    id: str
    primary_model_id: str
    fallback_model_ids: list[str] = Field(default_factory=list)
    sampling: StageSamplingConfig = StageSamplingConfig()
    requirements: StageRequirementsConfig = StageRequirementsConfig()
    decode: DecodeConfig = DecodeConfig()
    max_attempts: int = Field(2, ge=1)
    allow_cloud: bool = Field(False)
    repair_on_failure: bool = True


class ModelRegistryConfig(BaseModel):
    enabled: bool = True
    providers: list[ProviderSpec] = Field(default_factory=list)
    models: list[ModelSpec] = Field(default_factory=list)
    stages: list[StagePolicy] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_registry(self) -> "ModelRegistryConfig":
        provider_ids = {provider.id for provider in self.providers}
        model_ids = {model.id for model in self.models}
        stage_ids = {stage.id for stage in self.stages}
        if len(provider_ids) != len(self.providers):
            raise ValueError("model_registry.providers must have unique ids")
        if len(model_ids) != len(self.models):
            raise ValueError("model_registry.models must have unique ids")
        if len(stage_ids) != len(self.stages):
            raise ValueError("model_registry.stages must have unique ids")
        for model in self.models:
            if model.provider_id not in provider_ids:
                raise ValueError(
                    f"model_registry.models references unknown provider '{model.provider_id}'"
                )
        for stage in self.stages:
            if stage.primary_model_id not in model_ids:
                raise ValueError(
                    f"model_registry.stages references unknown model '{stage.primary_model_id}'"
                )
            if len(set(stage.fallback_model_ids)) != len(stage.fallback_model_ids):
                raise ValueError(
                    f"stage '{stage.id}' has duplicate fallback_model_ids; order must be unique"
                )
            for fallback in stage.fallback_model_ids:
                if fallback not in model_ids:
                    raise ValueError(
                        f"model_registry.stages references unknown fallback model '{fallback}'"
                    )
            if stage.decode.strategy != "standard" and not stage.decode.backend_provider_id:
                raise ValueError(
                    f"stage '{stage.id}' requires decode.backend_provider_id for "
                    f"decode.strategy={stage.decode.strategy!r}"
                )
            if (
                stage.decode.backend_provider_id
                and stage.decode.backend_provider_id not in provider_ids
            ):
                raise ValueError(
                    f"stage '{stage.id}' references unknown decode backend "
                    f"'{stage.decode.backend_provider_id}'"
                )
        return self


class GatewayConfig(BaseModel):
    enabled: bool = True
    bind_host: str = Field("127.0.0.1")
    port: int = Field(8010, ge=1024, le=65535)
    require_api_key: bool = False
    api_key: Optional[str] = None
    internal_token: Optional[str] = Field(
        None, description="Token for internal stage calls (X-Internal-Token)."
    )
    gpu_max_concurrency: int = Field(2, ge=1, description="Global GPU concurrency cap.")
    max_body_bytes: int = Field(2_000_000, ge=1024)
    request_timeout_s: float = Field(60.0, gt=0.0)
    upstream_probe_timeout_s: float = Field(5.0, gt=0.0)
    startup_probe: bool = True


class GraphServiceConfig(BaseModel):
    enabled: bool = True
    bind_host: str = Field("127.0.0.1")
    port: int = Field(8020, ge=1024, le=65535)
    workspace_root: Path = Field(default_factory=lambda: default_data_dir() / "graphs")
    max_events: int = Field(50_000, ge=100)
    max_results: int = Field(200, ge=1)
    require_workers: bool = Field(
        True, description="Require external graph worker CLIs for all adapters."
    )
    graphrag_cli: str = Field("scripts/graphrag_worker.sh")
    hypergraphrag_cli: str = Field("scripts/hypergraphrag_worker.sh")
    hyperrag_cli: str = Field("scripts/hyperrag_worker.sh")
    worker_timeout_s: float = Field(60.0, gt=0.0)


class MemoryServiceEmbedderConfig(BaseModel):
    provider: str = Field("stub", description="stub|local")
    dim: int = Field(256, ge=8, description="Embedding dimension for stub/local providers.")
    model_id: str = Field("hash-v1")


class MemoryServiceRerankerConfig(BaseModel):
    provider: str = Field("disabled", description="disabled|stub")
    max_window: int = Field(50, ge=1)


class MemoryServiceRetrievalConfig(BaseModel):
    topk_vector: int = Field(40, ge=1)
    topk_keyword: int = Field(40, ge=1)
    topk_graph: int = Field(40, ge=0)
    graph_depth: int = Field(2, ge=0, le=3)
    graph_max_nodes: int = Field(200, ge=1)
    rerank_window: int = Field(50, ge=1)
    max_cards: int = Field(8, ge=1)
    max_tokens: int = Field(1200, ge=200)
    max_per_type: int = Field(2, ge=1)
    type_priority: list[str] = Field(
        default_factory=lambda: ["decision", "procedure", "fact", "episodic", "glossary"]
    )


class MemoryServiceRankingConfig(BaseModel):
    weight_semantic: float = Field(0.45, ge=0.0)
    weight_keyword: float = Field(0.25, ge=0.0)
    weight_graph: float = Field(0.1, ge=0.0)
    weight_recency: float = Field(0.1, ge=0.0)
    weight_importance: float = Field(0.05, ge=0.0)
    weight_trust: float = Field(0.05, ge=0.0)
    weight_rerank: float = Field(0.2, ge=0.0)
    recency_half_life_days: int = Field(90, ge=1)


class MemoryServicePolicyConfig(BaseModel):
    allowed_audiences: list[str] = Field(default_factory=lambda: ["internal"])
    sensitivity_order: list[str] = Field(default_factory=lambda: ["low", "medium", "high"])
    reject_person_entities: bool = Field(True)
    reject_person_text: bool = Field(False)
    person_text_patterns: list[str] = Field(default_factory=list)
    reject_preferences: bool = Field(True)
    pii_patterns: list[str] = Field(default_factory=list)
    secret_patterns: list[str] = Field(default_factory=list)


class MemoryServiceConfig(BaseModel):
    enabled: bool = Field(False)
    bind_host: str = Field("127.0.0.1")
    port: int = Field(8030, ge=1024, le=65535)
    require_api_key: bool = Field(False)
    api_key: Optional[str] = None
    base_url: Optional[str] = Field(
        None, description="Override base URL for Memory Service clients."
    )
    database_url: Optional[str] = Field(
        None, description="Override database URL for Memory Service."
    )
    default_namespace: str = Field("default")
    max_body_bytes: int = Field(1_000_000, ge=1024)
    request_timeout_s: float = Field(10.0, gt=0.0)
    enable_ingest: bool = Field(True)
    enable_query: bool = Field(True)
    enable_feedback: bool = Field(True)
    enable_rerank: bool = Field(False)
    enable_query_embedding: bool = Field(True)
    enable_rls: bool = Field(False)
    embedder: MemoryServiceEmbedderConfig = MemoryServiceEmbedderConfig()
    reranker: MemoryServiceRerankerConfig = MemoryServiceRerankerConfig()
    retrieval: MemoryServiceRetrievalConfig = MemoryServiceRetrievalConfig()
    ranking: MemoryServiceRankingConfig = MemoryServiceRankingConfig()
    policy: MemoryServicePolicyConfig = MemoryServicePolicyConfig()


class CitationValidatorConfig(BaseModel):
    max_claims: int = Field(20, ge=1)
    max_citations_per_claim: int = Field(8, ge=1)
    allow_empty: bool = False
    allow_legacy_evidence_ids: bool = Field(
        False, description="Allow legacy evidence_ids-only claims without line ranges."
    )
    max_line_span: int = Field(8, ge=1, description="Max lines allowed per citation span.")


class EntailmentConfig(BaseModel):
    enabled: bool = True
    judge_stage: str = Field("entailment_judge")
    on_contradiction: str = Field("block", description="block|regenerate")
    on_nei: str = Field("abstain", description="abstain|expand_retrieval")
    max_attempts: int = Field(2, ge=1)


class VerificationConfig(BaseModel):
    claims_enabled: bool = True
    citation_validator: CitationValidatorConfig = CitationValidatorConfig()
    entailment: EntailmentConfig = EntailmentConfig()


class ModelStageConfig(BaseModel):
    provider: Optional[str] = Field(
        None, description="ollama|openai_compatible|openai (defaults to llm.provider)"
    )
    model: Optional[str] = Field(None, description="Stage-specific model override.")
    base_url: Optional[str] = Field(None, description="Stage-specific base URL override.")
    api_key: Optional[str] = Field(None, description="Stage-specific API key override.")
    allow_cloud: bool = Field(False, description="Allow cloud usage for this stage.")
    enabled: bool = Field(True)
    temperature: float = Field(0.2, ge=0.0, le=1.0)


class ModelStagesConfig(BaseModel):
    query_refine: ModelStageConfig = ModelStageConfig()
    draft_generate: ModelStageConfig = ModelStageConfig()
    final_answer: ModelStageConfig = ModelStageConfig()
    tool_transform: ModelStageConfig = Field(
        default_factory=lambda: ModelStageConfig(enabled=False)
    )
    entailment_judge: ModelStageConfig = ModelStageConfig()


class AgentAnswerConfig(BaseModel):
    enabled: bool = Field(True)


class AgentVisionConfig(BaseModel):
    provider: str = Field("ollama", description="ollama|openai_compatible")
    model: str = Field("llava")
    base_url: Optional[str] = Field(None)
    api_key: Optional[str] = Field(None)
    max_jobs_per_hour: int = Field(20, ge=0)
    run_only_when_idle: bool = Field(True)
    idle_hours_start: int = Field(22, ge=0, le=23)
    idle_hours_end: int = Field(6, ge=0, le=23)


class AgentConfig(BaseModel):
    enabled: bool = Field(True)
    max_pending_jobs: int = Field(5000, ge=10)
    nightly_cron: str = Field("0 3 * * *")
    answer_agent: AgentAnswerConfig = AgentAnswerConfig()
    vision: AgentVisionConfig = AgentVisionConfig()


class EnrichmentSchedulerConfig(BaseModel):
    enabled: bool = Field(True)
    scan_interval_s: float = Field(900.0, ge=5.0)
    max_events_per_scan: int = Field(2000, ge=10)
    window_days: int | None = Field(
        None,
        description="Override enrichment window in days (defaults to retention screenshot TTL).",
    )
    at_risk_hours: int = Field(24, ge=1, description="Window before expiry for at-risk metric.")


class ThreadingConfig(BaseModel):
    enabled: bool = Field(True)
    max_gap_minutes: float = Field(15.0, ge=1.0)
    app_similarity_threshold: float = Field(0.5, ge=0.0, le=1.0)
    title_similarity_threshold: float = Field(0.3, ge=0.0, le=1.0)
    max_events_per_thread: int = Field(100, ge=10)


class TableExtractorConfig(BaseModel):
    enabled: bool = Field(False, description="Enable table extraction pipeline.")
    allow_cloud: bool = Field(
        False, description="Allow cloud-backed table extraction when enabled."
    )


class AppConfig(BaseModel):
    offline: bool = Field(
        True,
        description="Hard offline mode: blocks all network egress unless a cloud profile is active.",
    )
    capture: CaptureConfig = CaptureConfig()
    tracking: TrackingConfig = TrackingConfig()
    ocr: OCRConfig = OCRConfig()
    vision_extract: VisionExtractConfig = VisionExtractConfig()
    embed: EmbedConfig = EmbedConfig()
    reranker: RerankerConfig = RerankerConfig()
    worker: WorkerConfig = WorkerConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    retention: RetentionPolicyConfig = RetentionPolicyConfig()
    storage: StorageQuotaConfig = StorageQuotaConfig()
    database: DatabaseConfig = DatabaseConfig()
    qdrant: QdrantConfig = QdrantConfig()
    ffmpeg: FFmpegConfig = FFmpegConfig()
    encryption: EncryptionConfig = EncryptionConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    features: FeatureFlagsConfig = FeatureFlagsConfig()
    next10: Next10Config = Next10Config()
    api: APIConfig = APIConfig()
    llm: LLMConfig = LLMConfig()
    llm_governor: LLMGovernorConfig = LLMGovernorConfig()
    model_registry: ModelRegistryConfig = ModelRegistryConfig()
    gateway: GatewayConfig = GatewayConfig()
    graph_service: GraphServiceConfig = GraphServiceConfig()
    memory_service: MemoryServiceConfig = MemoryServiceConfig()
    model_stages: ModelStagesConfig = ModelStagesConfig()
    mode: ModeConfig = ModeConfig()
    plugins: PluginsConfig = PluginsConfig()
    routing: ProviderRoutingConfig = ProviderRoutingConfig()
    privacy: PrivacyConfig = PrivacyConfig()
    policy: PolicyConfig = PolicyConfig()
    output: OutputConfig = OutputConfig()
    cache: CacheConfig = CacheConfig()
    verification: VerificationConfig = VerificationConfig()
    time: TimeConfig = TimeConfig()
    security: SecurityConfig = SecurityConfig()
    templates: TemplateHardeningConfig = TemplateHardeningConfig()
    ui: UIConfig = UIConfig()
    overlay_tracker: OverlayTrackerConfig = OverlayTrackerConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    memory: MemoryConfig = MemoryConfig()
    presets: PresetConfig = PresetConfig()
    promptops: PromptOpsConfig = PromptOpsConfig()
    agents: AgentConfig = AgentConfig()
    enrichment: EnrichmentSchedulerConfig = EnrichmentSchedulerConfig()
    table_extractor: TableExtractorConfig = TableExtractorConfig()
    threads: ThreadingConfig = ThreadingConfig()

    @field_validator("capture")
    @classmethod
    def validate_staging_dir(cls, value: CaptureConfig) -> CaptureConfig:
        value.staging_dir.mkdir(parents=True, exist_ok=True)
        value.data_dir.mkdir(parents=True, exist_ok=True)
        return value

    @field_validator("worker")
    @classmethod
    def validate_data_dir(cls, value: WorkerConfig) -> WorkerConfig:
        value.data_dir.mkdir(parents=True, exist_ok=True)
        return value

    @field_validator("tracking")
    @classmethod
    def validate_tracking_dir(cls, value: TrackingConfig, info: ValidationInfo) -> TrackingConfig:
        capture = info.data.get("capture")
        if capture:
            capture.data_dir.mkdir(parents=True, exist_ok=True)
        return value

    @model_validator(mode="after")
    def validate_api_security(self) -> "AppConfig":
        """Fail closed when binding to a non-loopback host in local mode.

        Remote mode is authenticated via Google OIDC (see api.server), so we do not
        require an API key there.
        """

        # If a user explicitly enables API-key auth, ensure a key is actually set.
        if self.api.require_api_key and not self.api.api_key:
            raise ValueError("api.api_key is required when api.require_api_key=true")

        default_db = _default_database_url()
        if self.database.url == default_db:
            data_dir = Path(self.capture.data_dir).resolve()
            self.database.url = f"sqlite:///{(data_dir / 'autocapture.db').as_posix()}"

        if self.mode.mode != "remote" and not is_loopback_host(self.api.bind_host):
            if not self.api.require_api_key:
                raise ValueError(
                    "api.require_api_key must be true when binding to non-loopback host"
                )
            if not self.api.api_key:
                raise ValueError("api.api_key is required when binding to non-loopback host")
            if not self.mode.https_enabled:
                raise ValueError(
                    "mode.https_enabled must be true when binding to non-loopback host"
                )
        if self.gateway.enabled and not is_loopback_host(self.gateway.bind_host):
            if not self.gateway.require_api_key:
                raise ValueError(
                    "gateway.require_api_key must be true when binding to non-loopback host"
                )
            if not self.gateway.api_key:
                raise ValueError("gateway.api_key is required when binding to non-loopback host")
        _validate_database_tls(self.database)
        _validate_qdrant_tls(self.qdrant)
        return self


def is_loopback_host(host: str) -> bool:
    """Return True if *host* is loopback.

    Accepts common hostnames (localhost) and any 127.0.0.0/8 or ::1 style IPs.
    """

    if host.lower() == "localhost":
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(host).is_loopback
    except Exception:
        return False


def _validate_database_tls(config: DatabaseConfig) -> None:
    from urllib.parse import parse_qs, urlparse

    if not config.url.startswith("postgres"):
        return
    if not config.require_tls_for_remote:
        return
    parsed = urlparse(config.url)
    host = parsed.hostname
    if not host or is_loopback_host(host):
        return
    sslmode = (parse_qs(parsed.query).get("sslmode") or [""])[0]
    if sslmode not in {"require", "verify-ca", "verify-full"}:
        raise ValueError(
            "database.require_tls_for_remote=true but postgres sslmode is not set "
            "to require/verify-ca/verify-full. Add ?sslmode=require to database.url."
        )


def _validate_qdrant_tls(config: QdrantConfig) -> None:
    from urllib.parse import urlparse

    if not config.require_tls_for_remote:
        return
    parsed = urlparse(config.url)
    if not parsed.hostname or is_loopback_host(parsed.hostname):
        return
    if parsed.scheme != "https":
        raise ValueError(
            "qdrant.require_tls_for_remote=true but qdrant.url is not https for remote host."
        )


def overlay_interface_ips(interface: str) -> list[str]:
    """Return a list of candidate IPs bound to a network interface.

    We prefer globally reachable addresses on overlay adapters (e.g. tailscale0, wg0).
    """

    try:
        import socket

        import psutil
    except Exception:
        return []

    addrs = psutil.net_if_addrs().get(interface) or []
    results: list[str] = []
    seen: set[str] = set()

    def _add(ip: str) -> None:
        ip = ip.split("%", 1)[0]
        if not ip or ip in seen:
            return
        if ip == "0.0.0.0":
            return
        if is_loopback_host(ip):
            return
        # Skip IPv6 link-local.
        if ip.lower().startswith("fe80:"):
            return
        seen.add(ip)
        results.append(ip)

    for addr in addrs:
        if addr.family == socket.AF_INET:
            _add(addr.address)
    for addr in addrs:
        if addr.family == socket.AF_INET6:
            _add(addr.address)
    return results


def resolve_overlay_bind_host(interface: str) -> str | None:
    """Return the best bind host for an overlay interface."""

    ips = overlay_interface_ips(interface)
    if not ips:
        return None
    # Prefer IPv4 when available for maximum client compatibility.
    for ip in ips:
        if ":" not in ip:
            return ip
    return ips[0]


def load_config(path: Path | str) -> AppConfig:
    """Load YAML configuration from disk."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    logger = logging.getLogger(__name__)

    if "offline" not in data:
        data["offline"] = True

    # Remote mode: derive api.bind_host from the configured overlay interface.
    mode = data.get("mode")
    if isinstance(mode, dict) and mode.get("mode") == "remote":
        overlay_if = mode.get("overlay_interface")
        if isinstance(overlay_if, str) and overlay_if:
            api = data.setdefault("api", {})
            if isinstance(api, dict):
                current = str(api.get("bind_host") or "127.0.0.1")
                if is_loopback_host(current):
                    resolved = resolve_overlay_bind_host(overlay_if)
                    if resolved:
                        logger.info(
                            "Remote mode: binding API to %s (from overlay_interface=%s)",
                            resolved,
                            overlay_if,
                        )
                        api["bind_host"] = resolved
                    else:
                        logger.warning(
                            "Remote mode enabled but overlay_interface=%s has no usable IP; "
                            "api.bind_host remains %s",
                            overlay_if,
                            current,
                        )

    if "embeddings" in data and "embed" not in data:
        data["embed"] = data.pop("embeddings")

    embed = data.get("embed")
    if isinstance(embed, dict):
        if "text_model" not in embed and "model" in embed:
            embed["text_model"] = embed.get("model")
        if "text_batch_size" not in embed and "batch_size" in embed:
            embed["text_batch_size"] = embed.get("batch_size")

    ocr = data.get("ocr")
    legacy_ocr_engine = "paddle" + "ocr-cuda"
    if isinstance(ocr, dict) and ocr.get("engine") == legacy_ocr_engine:
        logger.debug("Replacing legacy OCR engine with rapidocr-onnxruntime")
        ocr["engine"] = "rapidocr-onnxruntime"

    qdrant = data.get("qdrant")
    if isinstance(qdrant, dict):
        if "text_collection" not in qdrant and qdrant.get("collection_name"):
            logger.debug("Mapping legacy qdrant.collection_name to text_collection")
            qdrant["text_collection"] = qdrant.get("collection_name")
        if "text_vector_size" not in qdrant and qdrant.get("vector_size"):
            qdrant["text_vector_size"] = qdrant.get("vector_size")
    # Pydantic v2 compatibility (model_validate) with v1 fallback (parse_obj).
    if hasattr(AppConfig, "model_validate"):
        config = AppConfig.model_validate(data)
    else:
        config = AppConfig.parse_obj(data)
    return apply_settings_overrides(config)


def apply_settings_overrides(config: AppConfig, raw: dict | None = None) -> AppConfig:
    if raw is None:
        settings_path = Path(config.capture.data_dir) / "settings.json"
        raw = read_settings(settings_path)
    if not raw:
        apply_preset(config, config.presets.active_preset)
        return apply_dev_overrides(config)
    routing = raw.get("routing")
    if isinstance(routing, dict):
        if hasattr(config.routing, "model_dump"):
            merged = config.routing.model_dump()
        else:
            merged = config.routing.dict()
        for key, value in routing.items():
            if value and key in merged:
                merged[key] = value
        config.routing = ProviderRoutingConfig(**merged)
    llm = raw.get("llm")
    if isinstance(llm, dict):
        if hasattr(config.llm, "model_dump"):
            merged = config.llm.model_dump()
        else:
            merged = config.llm.dict()
        for key, value in llm.items():
            if key in merged and value is not None:
                merged[key] = value
        config.llm = LLMConfig(**merged)
    privacy = raw.get("privacy")
    if isinstance(privacy, dict):
        cloud_enabled = privacy.get("cloud_enabled")
        if isinstance(cloud_enabled, bool):
            config.privacy.cloud_enabled = cloud_enabled
        sanitize_default = privacy.get("sanitize_default")
        if isinstance(sanitize_default, bool):
            config.privacy.sanitize_default = sanitize_default
        extractive_only_default = privacy.get("extractive_only_default")
        if isinstance(extractive_only_default, bool):
            config.privacy.extractive_only_default = extractive_only_default
        allow_cloud_images = privacy.get("allow_cloud_images")
        if isinstance(allow_cloud_images, bool):
            config.privacy.allow_cloud_images = allow_cloud_images
        allow_token_vault_decrypt = privacy.get("allow_token_vault_decrypt")
        if isinstance(allow_token_vault_decrypt, bool):
            config.privacy.allow_token_vault_decrypt = allow_token_vault_decrypt
        paused = privacy.get("paused")
        if isinstance(paused, bool):
            config.privacy.paused = paused
        snooze_until = privacy.get("snooze_until_utc")
        parsed = _parse_datetime(snooze_until)
        if snooze_until is None:
            config.privacy.snooze_until_utc = None
        elif parsed is not None:
            config.privacy.snooze_until_utc = parsed
        exclude_monitors = privacy.get("exclude_monitors")
        if isinstance(exclude_monitors, list):
            config.privacy.exclude_monitors = list(exclude_monitors)
        exclude_processes = privacy.get("exclude_processes")
        if isinstance(exclude_processes, list):
            config.privacy.exclude_processes = list(exclude_processes)
        exclude_titles = privacy.get("exclude_window_title_regex")
        if isinstance(exclude_titles, list):
            config.privacy.exclude_window_title_regex = list(exclude_titles)
        exclude_regions = privacy.get("exclude_regions")
        if isinstance(exclude_regions, list):
            config.privacy.exclude_regions = list(exclude_regions)
        mask_regions = privacy.get("mask_regions")
        if isinstance(mask_regions, list):
            config.privacy.mask_regions = list(mask_regions)
    tracking = raw.get("tracking")
    if isinstance(tracking, dict):
        enabled = tracking.get("enabled")
        if isinstance(enabled, bool):
            config.tracking.enabled = enabled
        track_mouse = tracking.get("track_mouse_movement")
        if isinstance(track_mouse, bool):
            config.tracking.track_mouse_movement = track_mouse
        enable_clipboard = tracking.get("enable_clipboard")
        if isinstance(enable_clipboard, bool):
            config.tracking.enable_clipboard = enable_clipboard
        retention_days = tracking.get("retention_days")
        if isinstance(retention_days, int) or retention_days is None:
            config.tracking.retention_days = retention_days
    active_preset = raw.get("active_preset")
    if isinstance(active_preset, str) and active_preset:
        config.presets.active_preset = active_preset
    apply_preset(config, config.presets.active_preset)
    apply_policy_defaults(config)
    return apply_dev_overrides(config)


def apply_policy_defaults(config: AppConfig) -> AppConfig:
    path = Path(config.next10.policy_defaults_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return config
    if not isinstance(payload, dict):
        return config

    def _merge_list(current: list, defaults: list) -> list:
        merged = list(current)
        for item in defaults:
            if item not in merged:
                merged.append(item)
        return merged

    if isinstance(payload.get("exclude_processes"), list):
        config.privacy.exclude_processes = _merge_list(
            config.privacy.exclude_processes,
            payload.get("exclude_processes", []),
        )
    if isinstance(payload.get("exclude_window_title_regex"), list):
        config.privacy.exclude_window_title_regex = _merge_list(
            config.privacy.exclude_window_title_regex,
            payload.get("exclude_window_title_regex", []),
        )
    exclude_regions = (
        payload.get("exclude_regions") if isinstance(payload.get("exclude_regions"), list) else []
    )
    mask_regions = (
        payload.get("mask_regions") if isinstance(payload.get("mask_regions"), list) else []
    )
    if exclude_regions or mask_regions:
        combined = _merge_list(config.privacy.exclude_regions, exclude_regions)
        combined = _merge_list(combined, mask_regions)
        config.privacy.exclude_regions = combined
    if isinstance(payload.get("mask_regions"), list):
        config.privacy.mask_regions = _merge_list(config.privacy.mask_regions, mask_regions)
    return config


def apply_dev_overrides(config: AppConfig) -> AppConfig:
    if not is_dev_mode():
        return config
    logger = logging.getLogger(__name__)
    if (config.routing.vector_backend or "").strip().lower() == "qdrant":
        logger.info("Dev mode: routing vector backend to sqlite.")
        config.routing.vector_backend = "local"
    if (config.routing.spans_v2_backend or "").strip().lower() == "qdrant":
        logger.info("Dev mode: routing spans v2 backend to sqlite.")
        config.routing.spans_v2_backend = "local"
    if config.ocr.device.lower() != "cpu":
        logger.info("Dev mode: forcing OCR device to cpu.")
        config.ocr.device = "cpu"
    if config.ocr.engine != "disabled":
        import importlib.util

        if importlib.util.find_spec("rapidocr_onnxruntime") is None:
            logger.info("Dev mode: disabling OCR (rapidocr_onnxruntime not installed).")
            config.ocr.engine = "disabled"
            if config.vision_extract.engine in {"rapidocr", "rapidocr-onnxruntime"}:
                config.routing.ocr = "disabled"
    if (
        config.encryption.enabled
        and config.encryption.key_provider == "windows-credential-manager"
        and sys.platform != "win32"
    ):
        key_path = Path(config.capture.data_dir) / "autocapture.key"
        logger.info("Dev mode: using file-based encryption key at %s.", key_path)
        config.encryption.key_provider = f"file:{key_path.as_posix()}"
    return config


def _parse_datetime(value: object) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = dt.datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=dt.timezone.utc)
        return parsed
    return None
