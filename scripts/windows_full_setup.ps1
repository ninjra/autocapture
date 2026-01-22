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

$env:AUTOCAPTURE_GPU_MODE = "on"

poetry install --with dev --extras "ui windows ocr ocr-gpu embed-fast sqlcipher"
poetry run autocapture setup --profile full --apply
poetry run python tools/vendor_windows_binaries.py
poetry run autocapture doctor --verbose
