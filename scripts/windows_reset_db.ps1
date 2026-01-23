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
        Write-Error "Config not found at $configPath and repo template missing. Aborting."
        exit 1
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $configPath) | Out-Null
    Copy-Item $template $configPath
}

$code = @'
import json
import os
import sys
from pathlib import Path

def _emit(payload, code=0):
    print(json.dumps(payload))
    raise SystemExit(code)

def _default_data_dir():
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or ""
        if base:
            return Path(base) / "Autocapture" / "data"
        return Path.home() / "AppData" / "Local" / "Autocapture" / "data"
    return Path("./data")

config_path = Path(os.environ.get("AUTOCAPTURE_CONFIG") or "")
if not config_path:
    _emit({"error": "config_path_missing"}, code=2)
if not config_path.exists():
    _emit({"error": f"config_missing:{config_path}"}, code=2)

db_url = ""
data_dir = ""
key_path = ""
try:
    from autocapture.config import load_config
    from autocapture.paths import ensure_config_path

    config_path = ensure_config_path(config_path)
    cfg = load_config(config_path)
    db_url = cfg.database.url or ""
    data_dir = str(cfg.capture.data_dir)
    key_path = str(cfg.database.encryption_key_path)
    if not key_path and cfg.database.encryption_key_name:
        try:
            from autocapture.security.sqlcipher import load_sqlcipher_key

            _ = load_sqlcipher_key(cfg.database, Path(data_dir))
            key_path = str(cfg.database.encryption_key_path)
        except Exception:
            pass
except Exception:
    try:
        import yaml
    except Exception as exc:
        _emit({"error": f"yaml_import_failed:{exc.__class__.__name__}"}, code=2)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    db_url = str((data.get("database") or {}).get("url") or "")
    data_dir = str((data.get("capture") or {}).get("data_dir") or "")
    key_path = str((data.get("database") or {}).get("encryption_key_path") or "")

if not data_dir:
    data_dir = str(_default_data_dir())

db_path = ""
if db_url.startswith("sqlite:///"):
    db_path = str(Path(db_url.replace("sqlite:///", "")))
elif db_url.startswith("sqlite:////"):
    db_path = str(Path(db_url.replace("sqlite:////", "/")))
elif data_dir:
    db_path = str(Path(data_dir) / "autocapture.db")

key_path = str(Path(key_path)) if key_path else ""
print(json.dumps({
    "config_path": str(config_path),
    "db_path": db_path,
    "data_dir": data_dir,
    "key_path": key_path,
}))
'@

function Get-JsonFromText {
    param([string]$Text)
    if (-not $Text) {
        return ""
    }
    $start = $Text.IndexOf("{")
    $end = $Text.LastIndexOf("}")
    if ($start -lt 0 -or $end -le $start) {
        return ""
    }
    return $Text.Substring($start, ($end - $start + 1))
}

$tempPath = Join-Path ([System.IO.Path]::GetTempPath()) ("autocapture_config_probe_{0}.py" -f ([System.Guid]::NewGuid().ToString("N")))
Set-Content -Path $tempPath -Value $code -Encoding UTF8
$infoText = & poetry run python $tempPath 2>$null
$infoExit = $LASTEXITCODE
Remove-Item $tempPath -ErrorAction SilentlyContinue
$infoJson = Get-JsonFromText $infoText
if ($infoExit -ne 0 -or -not $infoJson) {
    Write-Error "Failed to load config via Poetry. Run .\\scripts\\windows_full_setup.bat first."
    exit 1
}

try {
    $info = $infoJson | ConvertFrom-Json
} catch {
    Write-Error "Failed to parse config output. Aborting reset."
    exit 1
}
if ($info.error) {
    Write-Error "Config load error: $($info.error). Run .\\scripts\\windows_full_setup.bat first."
    exit 1
}

if (-not $info.db_path) {
    Write-Error "Non-SQLite database configured. Reset script supports sqlite only."
    exit 1
}

$dbPath = $info.db_path
$dataDir = $info.data_dir
if (-not $dataDir) {
    $dataDir = Split-Path $dbPath -Parent
}
$keyPath = $info.key_path
if (-not $keyPath) {
    $keyPath = Join-Path $dataDir "secrets\\sqlcipher.key"
} elseif (-not [System.IO.Path]::IsPathRooted($keyPath)) {
    $keyPath = Join-Path $dataDir $keyPath
}

$tmpPath = "$dbPath.enc.tmp"
$backupPath = "$dbPath.bak"

if ($info.config_path) {
    Write-Host "Config path:"
    Write-Host "  $($info.config_path)"
}
if ($info.data_dir) {
    Write-Host "Data dir:"
    Write-Host "  $($info.data_dir)"
}
Write-Host "This will reset the encrypted DB used at:"
Write-Host "  $dbPath"
Write-Host "It will also rotate the SQLCipher key at:"
Write-Host "  $keyPath"

$stamp = Get-Date -Format "yyyyMMddHHmmss"
if (Test-Path $dbPath) {
    Copy-Item $dbPath "$dbPath.bak.$stamp"
    Remove-Item $dbPath
}
if (Test-Path $tmpPath) {
    Remove-Item $tmpPath
}
if (Test-Path $backupPath) {
    Copy-Item $backupPath "$backupPath.$stamp"
}
if (Test-Path $keyPath) {
    Copy-Item $keyPath "$keyPath.bak.$stamp"
    Remove-Item $keyPath
}

Write-Host "Reset complete. Re-run:"
Write-Host "  .\\scripts\\windows_full_setup.bat"
