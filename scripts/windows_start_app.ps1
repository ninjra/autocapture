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

if (-not $env:AUTOCAPTURE_GPU_MODE) {
    $env:AUTOCAPTURE_GPU_MODE = "auto"
} elseif ($env:AUTOCAPTURE_GPU_MODE.Trim().ToLower() -eq "on") {
    $cudaOk = $false
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi -L > $null 2>&1
        if ($LASTEXITCODE -eq 0) {
            $cudaOk = $true
        }
    }
    if (-not $cudaOk) {
        Write-Host "CUDA unavailable; overriding AUTOCAPTURE_GPU_MODE=on -> auto"
        $env:AUTOCAPTURE_GPU_MODE = "auto"
    }
}

if ($env:AUTOCAPTURE_CONFIG) {
    Write-Host "Using AUTOCAPTURE_CONFIG=$env:AUTOCAPTURE_CONFIG"
}

poetry run autocapture app
