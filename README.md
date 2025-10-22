# Autocapture

A high-performance, privacy-first desktop recall pipeline for Windows 11 that captures GPU-accelerated screenshots on HID activity, performs delayed OCR + embedding, and exposes search-ready metadata entirely on local infrastructure.

## Features

- **Event-driven capture** using Windows raw HID hooks with continuous screenshots while input is active.
- **GPU-first processing** leveraging Windows Graphics Capture, NVENC/AVIF encoding, PaddleOCR (CUDA), and nightly embedding batches.
- **Intelligent deduplication** with perceptual hashes and window-context metadata to discard redundant frames.
- **Configurable retention** with automatic pruning, NAS-aware quotas, and encryption at rest.
- **Observability** via Prometheus metrics and Grafana dashboards to track capture throughput, GPU utilization, OCR backlog, and storage trends.
- **Modular services** for capture, OCR, embeddings, storage, and retention orchestrated via asyncio-friendly workers.

## Repository Layout

```text
autocapture/
  capture/            # Input hooks, screen capture, duplicate detection
  embeddings/         # Batch embedding generation and Qdrant client helpers
  observability/      # Prometheus metrics integration
  ocr/                # OCR queue workers and result normalization
  storage/            # Database models, retention policies, encryption utilities
  config.py           # YAML configuration loader and validation models
  encryption.py       # AES-GCM helpers for NAS-bound artifacts
  logging_utils.py    # Structured logging configuration
  main.py             # Bootstrap entry point tying services together
config/
  example.yml         # Reference configuration with detailed tuning knobs
docs/
  architecture.md     # Component diagrams and data flow description
  operations.md       # NAS encryption, observability, and deployment guidance
pyproject.toml        # Project dependencies and tooling configuration
```

## Getting Started

1. Install dependencies (preferably with [Poetry](https://python-poetry.org/)):
   ```bash
   poetry install --with dev
   ```
2. Copy the example configuration and adjust values:
   ```bash
   cp config/example.yml autocapture.yml
   ```
3. Run linting and formatting:
   ```bash
   poetry run ruff check .
   poetry run black --check .
   ```
4. Launch the orchestrator (requires Windows 11 with GPU drivers):
   ```bash
   poetry run python -m autocapture.main --config autocapture.yml
   ```

> **Note:** Hardware-specific pieces (raw input hooks, DirectX capture, NVENC) are implemented behind interfaces so they can be replaced with stubs on non-Windows environments for development/testing.

## License

MIT
