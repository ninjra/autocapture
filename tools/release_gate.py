"""Release checklist gate runner."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _poetry_run(args: list[str]) -> None:
    _run(["poetry", "run", *args])


def _read_version(pyproject_path: Path) -> str:
    try:
        import tomllib  # Python 3.11+
    except Exception:  # pragma: no cover - fallback
        import tomli as tomllib  # type: ignore

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    version = data.get("tool", {}).get("poetry", {}).get("version")
    if not version:
        raise RuntimeError("Could not resolve version from pyproject.toml")
    return str(version)


def _write_doctor_config(*, windows: bool) -> Path:
    content = """offline: true
capture:
  record_video: false
  data_dir: "./data"
tracking:
  enabled: false
ocr:
  engine: "disabled"
  device: "cpu"
embed:
  text_model: "disabled"
routing:
  ocr: "disabled"
  embedding: "disabled"
qdrant:
  enabled: false
encryption:
  enabled: false
database:
  url: "sqlite:///./data/autocapture.db"
  allow_insecure_dev: true
"""
    if windows:
        content = """offline: true
capture:
  record_video: true
  data_dir: "./data"
tracking:
  enabled: true
ocr:
  engine: "rapidocr-onnxruntime"
  device: "cuda"
embed:
  text_model: "disabled"
routing:
  embedding: "disabled"
qdrant:
  enabled: true
encryption:
  enabled: false
database:
  url: "sqlite:///./data/autocapture.db"
  allow_insecure_dev: true
"""
    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix="-doctor.yml")
    tmp.write(content)
    tmp.flush()
    return Path(tmp.name)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    _poetry_run(["ruff", "check", "."])
    _poetry_run(["black", "--check", "."])
    _poetry_run(["python", "tools/repo_hygiene_check.py"])
    _run([sys.executable, ".tools/memory_guard.py", "--check"])
    _poetry_run(["pytest", "-q"])
    doctor_config = Path("autocapture.yml")
    if sys.platform != "win32":
        doctor_config = _write_doctor_config(windows=False)
    else:
        _poetry_run(["python", "tools/vendor_windows_binaries.py"])
        doctor_config = _write_doctor_config(windows=True)
    _poetry_run(
        [
            "python",
            "-m",
            "autocapture.main",
            "--config",
            str(doctor_config),
            "doctor",
        ]
    )

    if sys.platform == "win32":
        _poetry_run(["python", "tools/vendor_windows_binaries.py"])
        skip_packaging = _env_flag("AUTOCAPTURE_PREFLIGHT_SKIP_PACKAGING")
        run_bundle = _env_flag("AUTOCAPTURE_PREFLIGHT_BUNDLE")
        run_installer = _env_flag("AUTOCAPTURE_PREFLIGHT_INSTALLER")
        if skip_packaging:
            run_bundle = False
            run_installer = False
        if run_bundle:
            _poetry_run(["pyinstaller", "pyinstaller.spec"])
        else:
            print("== PyInstaller bundle skipped (AUTOCAPTURE_PREFLIGHT_BUNDLE=1 to enable) ==")
        if run_installer:
            if not run_bundle:
                raise RuntimeError("Installer requested but bundle disabled.")
            if shutil.which("iscc") is None:
                raise RuntimeError("Inno Setup (iscc) not found on PATH")
            version = _read_version(repo_root / "pyproject.toml")
            _run(
                [
                    "iscc",
                    "installer/autocapture.iss",
                    f"/DMyAppVersion={version}",
                ]
            )
        else:
            print(
                "== Inno Setup installer skipped "
                "(AUTOCAPTURE_PREFLIGHT_INSTALLER=1 to enable) =="
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
