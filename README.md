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

1. Create a virtual environment (works with the built-in `venv`, so no Poetry required):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # PowerShell (use activate.bat for cmd.exe)
   python -m pip install --upgrade pip
   ```
2. Install project and development dependencies (include the Windows extras to
   pull in the GPU-friendly capture backend):
   ```powershell
   python -m pip install -e .[windows]
   python -m pip install ruff black
   ```
   The `[windows]` extra installs the high-performance capture dependencies
   (`mss` for DXGI capture and `psutil` for foreground process metadata). If you
   install packages manually, ensure both are available in your environment.
3. Copy the example configuration and adjust values:
   ```powershell
   Copy-Item config/example.yml autocapture.yml
   ```
4. Run linting and formatting:
   ```powershell
   ruff check .
   black --check .
   ```
5. Launch the orchestrator (requires Windows 11 with GPU drivers):
   ```powershell
   python -m autocapture.main --config autocapture.yml
   ```

> **Note:** Hardware-specific pieces (raw input hooks, DirectX capture, NVENC) are implemented behind interfaces so they can be replaced with stubs on non-Windows environments for development/testing.

For an end-to-end production rollout—including NAS-hosted services, AES
encryption for artifacts, Prometheus/Grafana monitoring, OCR workers, and
long-term maintenance—follow the detailed checklist in
[`docs/step_by_step.md`](docs/step_by_step.md).

## License

MIT
