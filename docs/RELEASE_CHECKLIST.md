# Release Checklist

Use this checklist before cutting a release. The `tools/release_gate.py` script
runs the same commands.

## Required gates

- `poetry run ruff check .`
- `poetry run black --check .`
- `poetry run python tools/repo_hygiene_check.py`
- `python .tools/memory_guard.py --check`
- `poetry run pytest -q`
- `poetry run python -m autocapture.main --config autocapture.yml doctor`
  - FFmpeg present (or explicitly disabled)
  - Qdrant manageable (sidecar binary found or remote URL configured)
  - OCR providers reported and CUDA usage explained when `ocr.device=cuda`
- Windows build artifacts:
  - `poetry run pyinstaller pyinstaller.spec`
  - Inno Setup installer build via `iscc installer/autocapture.iss`

## Notes

- Windows release builds should bundle vendor binaries by running
  `python tools/vendor_windows_binaries.py` before PyInstaller.
- Docker is optional and only required for advanced remote/NAS deployments.
