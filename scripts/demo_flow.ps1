$ErrorActionPreference = "Stop"

$repoRoot = ""
try {
    $repoRoot = (git rev-parse --show-toplevel).Trim()
} catch {
    $repoRoot = (Get-Location).Path
}

Set-Location $repoRoot

Write-Host "== Demo flow: setup + research + promptops + UI =="

if (-not $env:AUTOCAPTURE_GPU_MODE) {
    $env:AUTOCAPTURE_GPU_MODE = "auto"
}

& (Join-Path $repoRoot "scripts\\windows_llm_setup.ps1")

poetry run autocapture setup --profile full --apply
poetry run python tools/vendor_windows_binaries.py

$baseConfig = $env:AUTOCAPTURE_CONFIG
if (-not $baseConfig) {
    $baseConfig = & poetry run python -c "from autocapture.paths import default_config_path; import os; print(os.environ.get('AUTOCAPTURE_CONFIG') or default_config_path())"
}
$baseConfig = $baseConfig.Trim()
$sourceConfig = $baseConfig
if (-not (Test-Path $sourceConfig)) {
    $sourceConfig = Join-Path $repoRoot "autocapture.yml"
}
if (-not (Test-Path $sourceConfig)) {
    Write-Error "No base config found. Set AUTOCAPTURE_CONFIG or ensure autocapture.yml exists."
    exit 1
}

$demoConfig = Join-Path $env:TEMP ("autocapture_demo_{0}.yml" -f ([System.Guid]::NewGuid().ToString("N")))
Copy-Item $sourceConfig $demoConfig -Force

$env:AUTOCAPTURE_CONFIG = $demoConfig
$env:AUTOCAPTURE_REPO_ROOT = $repoRoot

$patchScript = @'
import os
from pathlib import Path
import yaml

cfg_path = Path(os.environ["AUTOCAPTURE_CONFIG"])
repo_root = Path(os.environ["AUTOCAPTURE_REPO_ROOT"])
data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

promptops = data.get("promptops") or {}
promptops.update(
    {
        "enabled": True,
        "sources": [str((repo_root / "docs" / "research" / "scout_log.md").resolve())],
        "max_attempts": 1,
        "eval_repeats": 1,
        "eval_aggregation": "worst_case",
        "require_improvement": False,
    }
)
data["promptops"] = promptops

cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
print(cfg_path)
'@

$tempPatch = Join-Path ([System.IO.Path]::GetTempPath()) ("autocapture_demo_patch_{0}.py" -f ([System.Guid]::NewGuid().ToString("N")))
Set-Content -Path $tempPatch -Value $patchScript -Encoding UTF8
& poetry run python $tempPatch
Remove-Item $tempPatch -ErrorAction SilentlyContinue

Write-Host "Running research scout..."
poetry run autocapture research scout --out "docs/research/scout_report.json" --append "docs/research/scout_log.md"

Write-Host "Running PromptOps..."
poetry run autocapture promptops run
poetry run autocapture promptops status

Write-Host "Running doctor (demo config)..."
poetry run autocapture doctor --verbose

Write-Host "Starting app + UI..."
Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "set AUTOCAPTURE_CONFIG=$demoConfig && poetry run autocapture app" -WorkingDirectory $repoRoot

$deadline = [DateTime]::UtcNow.AddSeconds(30)
while ([DateTime]::UtcNow -lt $deadline) {
    try {
        $resp = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8008/health" -TimeoutSec 2
        if ($resp.StatusCode -eq 200) {
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

Start-Process "http://127.0.0.1:8008"
Write-Host "Demo running. Config: $demoConfig"
