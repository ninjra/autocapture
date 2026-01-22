$ErrorActionPreference = "Stop"

$repoRoot = ""
try {
    $repoRoot = (git rev-parse --show-toplevel).Trim()
} catch {
    $repoRoot = (Get-Location).Path
}

if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Error "Poetry not found. Install Poetry and re-run this script."
    exit 1
}

$configPath = $env:AUTOCAPTURE_CONFIG
if (-not $configPath) {
    $configPath = Join-Path $env:LOCALAPPDATA "Autocapture\\autocapture.yml"
}
$env:AUTOCAPTURE_CONFIG = $configPath

$fallbackDataDir = Join-Path $env:LOCALAPPDATA "Autocapture\\data"
$dbPath = Join-Path $fallbackDataDir "autocapture.db"
$keyPath = Join-Path $fallbackDataDir "secrets\\sqlcipher.key"

$code = @'
import json
import os
from pathlib import Path

from autocapture.config import load_config
from autocapture.paths import default_config_path, ensure_config_path

config_path = Path(os.environ.get("AUTOCAPTURE_CONFIG") or default_config_path())
config_path = ensure_config_path(config_path)
cfg = load_config(config_path)
db_url = cfg.database.url or ""
db_path = ""
data_dir = ""
if db_url.startswith("sqlite:///"):
    db_path = str(Path(db_url.replace("sqlite:///", "")))
    data_dir = str(Path(db_path).parent)
else:
    data_dir = str(cfg.capture.data_dir)
key_path = Path(cfg.database.encryption_key_path)
if not key_path.is_absolute():
    key_path = Path(data_dir) / key_path
print(json.dumps({
    "config_path": str(config_path),
    "db_path": db_path,
    "data_dir": data_dir,
    "key_path": str(key_path),
}))
'@

$infoJson = poetry run python -c $code 2>$null
if ($LASTEXITCODE -ne 0 -or -not $infoJson) {
    Write-Error "Failed to load config via Poetry. Aborting reset to avoid deleting the wrong DB."
    exit 1
}

try {
    $info = $infoJson | ConvertFrom-Json
} catch {
    Write-Error "Failed to parse config output. Aborting reset."
    exit 1
}

if (-not $info.db_path) {
    Write-Error "Non-SQLite database configured. Reset script supports sqlite only."
    exit 1
}

$dbPath = $info.db_path
$keyPath = $info.key_path

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
