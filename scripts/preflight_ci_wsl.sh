#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

ensure_python() {
  if command -v python3 >/dev/null 2>&1; then
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
      if ! sudo -n true 2>/dev/null; then
        echo "== sudo required for Python install; you may be prompted =="
      fi
      sudo apt-get update -y
      sudo apt-get install -y python3 python3-venv python3-pip
    elif [ "$(id -u)" = "0" ]; then
      apt-get update -y
      apt-get install -y python3 python3-venv python3-pip
    fi
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Python3 not found. Install Python 3.12+ and re-run." >&2
    exit 1
  fi
}

ensure_poetry() {
  if command -v poetry >/dev/null 2>&1; then
    return 0
  fi
  ensure_python
  export POETRY_HOME="${POETRY_HOME:-$HOME/.local/share/pypoetry}"
  local installer="/tmp/poetry_install_$$.py"
  if command -v curl >/dev/null 2>&1; then
    curl -sSL https://install.python-poetry.org -o "$installer"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$installer" https://install.python-poetry.org
  else
    echo "curl/wget not found. Install one of them and re-run." >&2
    exit 1
  fi
  python3 "$installer" -y
  rm -f "$installer"
  export PATH="$HOME/.local/bin:$POETRY_HOME/bin:$PATH"
  if ! command -v poetry >/dev/null 2>&1; then
    echo "Poetry install completed but poetry is not on PATH. Reopen the shell and retry." >&2
    exit 1
  fi
}

ensure_sqlcipher_deps() {
  if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists sqlcipher; then
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
      if ! sudo -n true 2>/dev/null; then
        echo "== sudo required for SQLCipher headers; you may be prompted =="
      fi
      sudo apt-get update -y
      sudo apt-get install -y build-essential pkg-config libsqlcipher-dev python3-dev
    elif [ "$(id -u)" = "0" ]; then
      apt-get update -y
      apt-get install -y build-essential pkg-config libsqlcipher-dev python3-dev
    else
      echo "Missing sqlcipher headers and no sudo/root available."
      echo "Install: sudo apt-get install -y build-essential pkg-config libsqlcipher-dev python3-dev"
      return 1
    fi
  else
    echo "apt-get not found. Install SQLCipher headers for your distro and retry."
    return 1
  fi
}

ensure_poetry
if ! ensure_sqlcipher_deps; then
  exit 1
fi

export AUTOCAPTURE_TEST_MODE=1
export AUTOCAPTURE_GPU_MODE=off

echo "== Autocapture CI preflight (WSL/Linux) =="
if ! poetry install --with dev --extras "sqlcipher"; then
  echo "== poetry install failed; retrying with legacy build settings =="
  if command -v python3 >/dev/null 2>&1; then
    poetry env use python3 >/dev/null 2>&1 || true
  fi
  PIP_USE_PEP517=0 PIP_NO_BUILD_ISOLATION=1 poetry install --with dev --extras "sqlcipher"
fi
poetry run black --check .
poetry run ruff check .
poetry run pytest -q -m "not gpu"
poetry run python .tools/memory_guard.py --check
poetry run python tools/verify_checksums.py
poetry run python tools/license_guardrails.py
poetry run python tools/release_gate.py
poetry run python -m autocapture.promptops.validate
poetry run python tools/pillar_gate.py
poetry run python tools/privacy_scanner.py
poetry run python tools/provenance_gate.py
poetry run python tools/coverage_gate.py
poetry run python tools/latency_gate.py
poetry run python tools/retrieval_sensitivity.py
poetry run python tools/no_evidence_gate.py
poetry run python tools/conflict_gate.py
poetry run python tools/integrity_gate.py
