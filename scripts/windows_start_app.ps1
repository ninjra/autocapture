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

if ($env:AUTOCAPTURE_CONFIG) {
    Write-Host "Using AUTOCAPTURE_CONFIG=$env:AUTOCAPTURE_CONFIG"
}

poetry run autocapture app
