"""Configuration loading and validation using Pydantic models."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, validator


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
    staging_dir: Path = Field(
        Path("./staging"),
        description="Local NVMe-backed directory for temporary assets.",
    )
    data_dir: Path = Field(
        Path("./data"),
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
        ge=64,
        description="Minimum free space (MB) required in staging_dir.",
    )
    data_min_free_mb: int = Field(
        1024,
        ge=64,
        description="Minimum free space (MB) required in data_dir.",
    )


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
    enable_clipboard: bool = True
    retention_days: int | None = None


class OCRConfig(BaseModel):
    queue_maxsize: int = Field(2000, ge=100)
    batch_size: int = Field(32, ge=1)
    max_latency_s: int = Field(900, ge=10)
    engine: str = Field("paddleocr-cuda")
    languages: list[str] = Field(default_factory=lambda: ["en"])
    output_format: str = Field("json")


class EmbeddingConfig(BaseModel):
    model: str = Field("sentence-transformers/all-MiniLM-L6-v2")
    batch_size: int = Field(256, ge=1)
    schedule_cron: str = Field(
        "0 2 * * *", description="Cron string for nightly batches."
    )
    use_half_precision: bool = Field(
        True, description="Use float16 embeddings to shrink storage bandwidth."
    )


class WorkerConfig(BaseModel):
    data_dir: Path = Field(
        Path("./data"),
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
    embedding_max_attempts: int = Field(5, ge=1)


class RetentionPolicyConfig(BaseModel):
    video_days: int = Field(3, ge=1)
    roi_days: int = Field(14, ge=1)
    max_media_gb: int = Field(200, ge=1)
    screenshot_ttl_days: int = Field(
        90, ge=1, description="Days to keep raw screenshots before pruning."
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
        "sqlite:///./data/autocapture.db",
        description="SQLAlchemy URL for local metadata storage (SQLite by default).",
    )
    echo: bool = False
    pool_size: int = Field(10, ge=1)
    max_overflow: int = Field(10, ge=0)
    sqlite_busy_timeout_ms: int = Field(5000, ge=0)
    sqlite_wal: bool = True
    sqlite_synchronous: str = Field("NORMAL")

    @validator("sqlite_synchronous")
    def validate_sqlite_synchronous(cls, value: str) -> str:  # type: ignore[name-defined]
        allowed = {"NORMAL", "FULL", "OFF"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(
                f"sqlite_synchronous must be one of {sorted(allowed)}; got {value!r}"
            )
        return upper


class QdrantConfig(BaseModel):
    url: str = Field("http://localhost:6333")
    collection_name: str = Field("autocapture_spans")
    vector_size: int = Field(384, ge=64)
    distance: str = Field("Cosine")


class EncryptionConfig(BaseModel):
    enabled: bool = Field(True)
    key_provider: str = Field(
        default_factory=lambda: "windows-credential-manager"
        if sys.platform == "win32"
        else "file:./data/autocapture.key",
        description="Strategy to fetch AES key (file, env, kms).",
    )
    key_name: str = Field("autocapture/nas-aes-key")
    chunk_size: int = Field(
        4 * 1024 * 1024,
        ge=64 * 1024,
        description="Chunk size for streaming encryption uploads.",
    )


class ObservabilityConfig(BaseModel):
    prometheus_port: int = Field(9005, ge=1024, le=65535)
    grafana_url: Optional[str] = None
    enable_gpu_stats: bool = Field(True)


class APIConfig(BaseModel):
    port: int = Field(5273, ge=1024, le=65535)
    bind_host: str = Field(
        "127.0.0.1", description="Bind host for the local API server."
    )


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


class ProviderRoutingConfig(BaseModel):
    capture: str = Field("local")
    ocr: str = Field("local")
    embedding: str = Field("local")
    retrieval: str = Field("local")
    reranker: str = Field("disabled")
    compressor: str = Field("extractive")
    verifier: str = Field("rules")
    llm: str = Field("ollama")


class PrivacyConfig(BaseModel):
    cloud_enabled: bool = Field(False)
    sanitize_default: bool = Field(True)
    extractive_only_default: bool = Field(True)


class PromptOpsConfig(BaseModel):
    enabled: bool = Field(False)
    schedule_cron: str = Field("0 6 * * 1")
    sources: list[str] = Field(default_factory=list)
    github_token: Optional[str] = Field(None)
    github_repo: Optional[str] = Field(None)


class LLMConfig(BaseModel):
    provider: str = Field("ollama", description="ollama or openai")
    ollama_url: str = Field("http://127.0.0.1:11434")
    ollama_model: str = Field("llama3")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    openai_model: str = Field("gpt-4.1-mini")


class AppConfig(BaseModel):
    capture: CaptureConfig = CaptureConfig()
    tracking: TrackingConfig = TrackingConfig()
    ocr: OCRConfig = OCRConfig()
    embeddings: EmbeddingConfig = EmbeddingConfig()
    worker: WorkerConfig = WorkerConfig()
    retention: RetentionPolicyConfig = RetentionPolicyConfig()
    storage: StorageQuotaConfig = StorageQuotaConfig()
    database: DatabaseConfig = DatabaseConfig()
    qdrant: QdrantConfig = QdrantConfig()
    encryption: EncryptionConfig = EncryptionConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    api: APIConfig = APIConfig()
    llm: LLMConfig = LLMConfig()
    mode: ModeConfig = ModeConfig()
    routing: ProviderRoutingConfig = ProviderRoutingConfig()
    privacy: PrivacyConfig = PrivacyConfig()
    promptops: PromptOpsConfig = PromptOpsConfig()

    @validator("capture")
    def validate_staging_dir(cls, value: CaptureConfig) -> CaptureConfig:  # type: ignore[name-defined]
        value.staging_dir.mkdir(parents=True, exist_ok=True)
        value.data_dir.mkdir(parents=True, exist_ok=True)
        return value

    @validator("worker")
    def validate_data_dir(cls, value: WorkerConfig) -> WorkerConfig:  # type: ignore[name-defined]
        value.data_dir.mkdir(parents=True, exist_ok=True)
        return value

    @validator("tracking")
    def validate_tracking_dir(
        cls, value: TrackingConfig, values: dict
    ) -> TrackingConfig:  # type: ignore[name-defined]
        capture = values.get("capture")
        if capture:
            capture.data_dir.mkdir(parents=True, exist_ok=True)
        return value


def load_config(path: Path | str) -> AppConfig:
    """Load YAML configuration from disk."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    # Pydantic v2 compatibility (model_validate) with v1 fallback (parse_obj).
    if hasattr(AppConfig, "model_validate"):
        return AppConfig.model_validate(data)
    return AppConfig.parse_obj(data)
