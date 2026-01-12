"""Release checklist gate runner."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


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


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    _poetry_run(["ruff", "check", "."])
    _poetry_run(["black", "--check", "."])
    _poetry_run(["pytest", "-q"])
    _poetry_run([
        "python",
        "-m",
        "autocapture.main",
        "--config",
        "autocapture.yml",
        "doctor",
    ])

    if sys.platform == "win32":
        _poetry_run(["python", "tools/vendor_windows_binaries.py"])
        _poetry_run(["pyinstaller", "pyinstaller.spec"])
        if shutil.which("iscc") is None:
            raise RuntimeError("Inno Setup (iscc) not found on PATH")
        version = _read_version(repo_root / "pyproject.toml")
        _run([
            "iscc",
            "installer/autocapture.iss",
            f"/DMyAppVersion={version}",
        ])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
