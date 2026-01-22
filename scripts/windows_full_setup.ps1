$ErrorActionPreference = "Stop"

$repoRoot = ""
try {
    $repoRoot = (git rev-parse --show-toplevel).Trim()
} catch {
    $repoRoot = (Get-Location).Path
}

Set-Location $repoRoot

if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Error "Poetry not found. Install Poetry and re-run this script."
    exit 1
}

$configPath = $env:AUTOCAPTURE_CONFIG
if (-not $configPath) {
    $configPath = Join-Path $env:LOCALAPPDATA "Autocapture\\autocapture.yml"
}
$env:AUTOCAPTURE_CONFIG = $configPath

if (-not (Test-Path $configPath)) {
    $template = Join-Path $repoRoot "autocapture.yml"
    if (-not (Test-Path $template)) {
        Write-Error "Config template missing in repo. Aborting."
        exit 1
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $configPath) | Out-Null
    Copy-Item $template $configPath
}

$env:AUTOCAPTURE_GPU_MODE = "on"

poetry install --with dev --extras "ui windows ocr ocr-gpu embed-fast sqlcipher"
poetry run python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"
if ($LASTEXITCODE -ne 0) {
    poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
}

function Select-BaseDir {
    try {
        $shell = New-Object -ComObject Shell.Application
        $folder = $shell.BrowseForFolder(0, "Choose a folder for Autocapture data", 0, 0)
        if ($folder) {
            return $folder.Self.Path
        }
    } catch {
        return ""
    }
    return ""
}

function Get-BaseDirFromConfig {
    $code = @'
import os
from pathlib import Path
import yaml

config_path = Path(os.environ["AUTOCAPTURE_CONFIG"])
data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
paths = data.get("paths") or {}
base_dir = paths.get("base_dir") or ""
print(base_dir)
'@
    $tempPath = Join-Path ([System.IO.Path]::GetTempPath()) ("autocapture_base_dir_{0}.py" -f ([System.Guid]::NewGuid().ToString("N")))
    Set-Content -Path $tempPath -Value $code -Encoding UTF8
    $output = & poetry run python $tempPath 2>$null
    Remove-Item $tempPath -ErrorAction SilentlyContinue
    if ($LASTEXITCODE -ne 0) {
        return ""
    }
    return $output.Trim()
}

function Set-BaseDirInConfig {
    param([string]$BaseDir)
    $env:AUTOCAPTURE_BASE_DIR = $BaseDir
    $code = @'
import os
from pathlib import Path
import yaml

config_path = Path(os.environ["AUTOCAPTURE_CONFIG"])
base_dir = os.environ.get("AUTOCAPTURE_BASE_DIR") or ""
data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
paths = data.setdefault("paths", {})
paths["base_dir"] = base_dir
payload = yaml.safe_dump(data, sort_keys=False)
config_path.write_text(payload, encoding="utf-8")
'@
    $tempPath = Join-Path ([System.IO.Path]::GetTempPath()) ("autocapture_set_base_dir_{0}.py" -f ([System.Guid]::NewGuid().ToString("N")))
    Set-Content -Path $tempPath -Value $code -Encoding UTF8
    & poetry run python $tempPath 2>$null
    Remove-Item $tempPath -ErrorAction SilentlyContinue
}

$baseDir = $env:AUTOCAPTURE_BASE_DIR
if (-not $baseDir) {
    $baseDir = Get-BaseDirFromConfig
}
if (-not $baseDir) {
    $picked = Select-BaseDir
    if ($picked) {
        $baseDir = $picked
    } else {
        $baseDir = Join-Path $env:LOCALAPPDATA "Autocapture"
    }
}
if ($baseDir) {
    Write-Host "Using base directory: $baseDir"
    Set-BaseDirInConfig $baseDir
}

poetry run autocapture setup --profile full --apply
poetry run python tools/vendor_windows_binaries.py
$doctorOutput = & poetry run autocapture doctor --verbose 2>&1
$doctorOutput | ForEach-Object { Write-Host $_ }

function Confirm-ResetDb {
    try {
        $wshell = New-Object -ComObject WScript.Shell
        $result = $wshell.Popup(
            "Encrypted DB appears invalid. Reset now? This deletes the local DB and key.",
            0,
            "Autocapture",
            0x4 + 0x30
        )
        return $result -eq 6
    } catch {
        return $false
    }
}

if ($doctorOutput -match "file is not a database|hmac check failed") {
    if (Confirm-ResetDb) {
        & (Join-Path $repoRoot "scripts\\windows_reset_db.ps1")
        $doctorOutput = & poetry run autocapture doctor --verbose 2>&1
        $doctorOutput | ForEach-Object { Write-Host $_ }
    }
}
