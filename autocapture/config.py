"""Configuration loading and validation using Pydantic models."""

from __future__ import annotations

import datetime as dt
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from .paths import default_data_dir, default_staging_dir
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
    model: str = Field("cross-encoder/ms-marco-MiniLM-L-6-v2")
    device: str = Field("auto", description="auto|cuda|cpu; auto prefers CUDA when available")
    top_k: int = Field(100, ge=1)


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
    text_vector_size: int = Field(768, ge=64)
    image_vector_size: int = Field(768, ge=64)
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


class ObservabilityConfig(BaseModel):
    prometheus_bind_host: str = Field("127.0.0.1")
    prometheus_port: int = Field(9005, ge=1024, le=65535)
    prometheus_port_fallbacks: int = Field(10, ge=0, le=100)
    prometheus_fail_fast: bool = Field(False)
    grafana_url: Optional[str] = None
    enable_gpu_stats: bool = Field(True)


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


class SecurityConfig(BaseModel):
    local_unlock_enabled: bool = Field(
        True, description="Require local unlock session for sensitive endpoints."
    )
    session_ttl_seconds: int = Field(300, ge=30, description="Unlock session TTL in seconds.")
    provider: str = Field(
        default_factory=_default_security_provider,
        description="windows_hello|cred_ui|test|disabled",
    )


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
    max_iterations: int = Field(3, ge=1)
    max_llm_attempts_per_prompt: int = Field(3, ge=1)
    eval_repeats: int = Field(3, ge=1)
    eval_aggregation: str = Field("worst_case")
    require_improvement: bool = Field(True)
    min_delta_verifier_pass_rate: float = Field(0.02)
    min_delta_citation_coverage: float = Field(0.02)
    min_delta_refusal_rate: float = Field(0.02)
    min_delta_latency_ms: float = Field(200.0)
    tolerance_citation_coverage: float = Field(0.02)
    tolerance_refusal_rate: float = Field(0.02)
    tolerance_latency_ms: float = Field(250.0)
    min_verifier_pass_rate: float = Field(0.60)
    min_citation_coverage: float = Field(0.60)
    max_refusal_rate: float = Field(0.30)
    max_mean_latency_ms: float = Field(15000.0)
    max_prompt_chars: int = Field(12000, ge=1)
    max_source_bytes: int = Field(1_048_576, ge=1)
    max_source_excerpt_chars: int = Field(2000, ge=1)
    max_sources: int = Field(32, ge=1)

    @field_validator("eval_aggregation")
    @classmethod
    def validate_eval_aggregation(cls, value: str) -> str:
        if value not in {"worst_case", "mean"}:
            raise ValueError("promptops.eval_aggregation must be 'worst_case' or 'mean'")
        return value


class LLMConfig(BaseModel):
    provider: str = Field("ollama", description="ollama|openai|openai_compatible")
    ollama_url: str = Field("http://127.0.0.1:11434")
    ollama_model: str = Field("llama3")
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


class AppConfig(BaseModel):
    offline: bool = Field(
        True,
        description="Hard offline mode: blocks all network egress unless a cloud profile is active.",
    )
    capture: CaptureConfig = CaptureConfig()
    tracking: TrackingConfig = TrackingConfig()
    ocr: OCRConfig = OCRConfig()
    embed: EmbedConfig = EmbedConfig()
    reranker: RerankerConfig = RerankerConfig()
    worker: WorkerConfig = WorkerConfig()
    retention: RetentionPolicyConfig = RetentionPolicyConfig()
    storage: StorageQuotaConfig = StorageQuotaConfig()
    database: DatabaseConfig = DatabaseConfig()
    qdrant: QdrantConfig = QdrantConfig()
    ffmpeg: FFmpegConfig = FFmpegConfig()
    encryption: EncryptionConfig = EncryptionConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    api: APIConfig = APIConfig()
    llm: LLMConfig = LLMConfig()
    mode: ModeConfig = ModeConfig()
    routing: ProviderRoutingConfig = ProviderRoutingConfig()
    privacy: PrivacyConfig = PrivacyConfig()
    security: SecurityConfig = SecurityConfig()
    presets: PresetConfig = PresetConfig()
    promptops: PromptOpsConfig = PromptOpsConfig()
    agents: AgentConfig = AgentConfig()

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


def apply_settings_overrides(config: AppConfig) -> AppConfig:
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
    privacy = raw.get("privacy")
    if isinstance(privacy, dict):
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
    active_preset = raw.get("active_preset")
    if isinstance(active_preset, str) and active_preset:
        config.presets.active_preset = active_preset
    apply_preset(config, config.presets.active_preset)
    return apply_dev_overrides(config)


def apply_dev_overrides(config: AppConfig) -> AppConfig:
    if not is_dev_mode():
        return config
    logger = logging.getLogger(__name__)
    if config.qdrant.enabled:
        logger.info("Dev mode: disabling qdrant backend.")
        config.qdrant.enabled = False
    if config.ocr.device.lower() != "cpu":
        logger.info("Dev mode: forcing OCR device to cpu.")
        config.ocr.device = "cpu"
    if config.ocr.engine != "disabled":
        import importlib.util

        if importlib.util.find_spec("rapidocr_onnxruntime") is None:
            logger.info("Dev mode: disabling OCR (rapidocr_onnxruntime not installed).")
            config.ocr.engine = "disabled"
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
