"""Configuration loading and validation using Pydantic models."""

from __future__ import annotations

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
    encoder: str = Field(
        "nvenc_webp",
        description="Encoder preset (nvenc_webp, nvenc_avif, cpu_webp).",
    )
    max_pending: int = Field(
        5000,
        ge=100,
        description="Backpressure limit for outstanding capture tasks.",
    )


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


class StorageQuotaConfig(BaseModel):
    image_quota_gb: int = Field(2500, ge=10)
    prune_grace_days: int = Field(90, ge=1)
    prune_batch: int = Field(2000, ge=10)


class DatabaseConfig(BaseModel):
    url: str = Field(
        "postgresql+psycopg://autocapture:autocapture@nas/autocapture",
        description="SQLAlchemy URL for Postgres instance.",
    )
    echo: bool = False
    pool_size: int = Field(10, ge=1)
    max_overflow: int = Field(10, ge=0)


class QdrantConfig(BaseModel):
    url: str = Field("http://nas:6333")
    collection_name: str = Field("autocapture_spans")
    vector_size: int = Field(384, ge=64)
    distance: str = Field("Cosine")


class EncryptionConfig(BaseModel):
    enabled: bool = Field(True)
    key_provider: str = Field(
        "windows-credential-manager",
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


class LLMConfig(BaseModel):
    provider: str = Field("ollama", description="ollama or openai")
    ollama_url: str = Field("http://127.0.0.1:11434")
    ollama_model: str = Field("llama3")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    openai_model: str = Field("gpt-4.1-mini")


class AppConfig(BaseModel):
    capture: CaptureConfig = CaptureConfig()
    ocr: OCRConfig = OCRConfig()
    embeddings: EmbeddingConfig = EmbeddingConfig()
    storage: StorageQuotaConfig = StorageQuotaConfig()
    database: DatabaseConfig = DatabaseConfig()
    qdrant: QdrantConfig = QdrantConfig()
    encryption: EncryptionConfig = EncryptionConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    api: APIConfig = APIConfig()
    llm: LLMConfig = LLMConfig()

    @validator("capture")
    def validate_staging_dir(cls, value: CaptureConfig) -> CaptureConfig:  # type: ignore[name-defined]
        value.staging_dir.mkdir(parents=True, exist_ok=True)
        return value


def load_config(path: Path | str) -> AppConfig:
    """Load YAML configuration from disk."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return AppConfig.parse_obj(data)
