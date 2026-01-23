$ErrorActionPreference = "Stop"

$repoRoot = ""
try {
    $repoRoot = (git rev-parse --show-toplevel).Trim()
} catch {
    $repoRoot = (Get-Location).Path
}

Set-Location $repoRoot

$cacheRoot = Join-Path $env:LOCALAPPDATA "Autocapture\\setup-cache"
New-Item -ItemType Directory -Force -Path $cacheRoot | Out-Null

$env:POETRY_VIRTUALENVS_IN_PROJECT = "1"
if (-not $env:HF_HOME) {
    $env:HF_HOME = Join-Path $env:LOCALAPPDATA "Autocapture\\hf"
}
if (-not $env:HUGGINGFACE_HUB_CACHE) {
    $env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME "hub"
}
if (-not $env:TRANSFORMERS_CACHE) {
    $env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME "transformers"
}
if (-not $env:FASTEMBED_CACHE_PATH) {
    $env:FASTEMBED_CACHE_PATH = Join-Path $env:LOCALAPPDATA "Autocapture\\fastembed"
}
if (-not $env:HF_HUB_DISABLE_SYMLINKS_WARNING) {
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
}

function Resolve-PythonCommand {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "-3")
    }
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Python not found; attempting install via winget..."
        & winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements
        if (Get-Command python -ErrorAction SilentlyContinue) {
            return @("python")
        }
        if (Get-Command py -ErrorAction SilentlyContinue) {
            return @("py", "-3")
        }
    }
    return $null
}

function Ensure-Poetry {
    if (Get-Command poetry -ErrorAction SilentlyContinue) {
        return
    }

    $poetryHome = $env:POETRY_HOME
    if (-not $poetryHome) {
        $poetryHome = Join-Path $env:LOCALAPPDATA "pypoetry"
        $env:POETRY_HOME = $poetryHome
    }
    $poetryBin = Join-Path $poetryHome "bin"
    $poetryExe = Join-Path $poetryBin "poetry.exe"
    if (Test-Path $poetryExe) {
        if (-not (($env:Path -split ";") -contains $poetryBin)) {
            $env:Path = "$poetryBin;$env:Path"
        }
        return
    }

    $pythonCmd = Resolve-PythonCommand
    if (-not $pythonCmd) {
        Write-Error "Python not found. Install Python 3.12+ and re-run this script."
        exit 1
    }

    Write-Host "Poetry not found; installing to $poetryHome..."
    $installerPath = Join-Path ([System.IO.Path]::GetTempPath()) ("install_poetry_{0}.py" -f ([System.Guid]::NewGuid().ToString("N")))
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    } catch {
        # ignore on newer PowerShell versions
    }
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "https://install.python-poetry.org" -OutFile $installerPath
    } catch {
        Write-Error "Failed to download Poetry installer."
        exit 1
    }

    $pythonExe = $pythonCmd[0]
    $pythonExtra = @()
    if ($pythonCmd.Count -gt 1) {
        $pythonExtra = $pythonCmd[1..($pythonCmd.Count - 1)]
    }
    & $pythonExe @pythonExtra $installerPath -y
    $installExit = $LASTEXITCODE
    Remove-Item $installerPath -ErrorAction SilentlyContinue
    if ($installExit -ne 0) {
        Write-Error "Poetry installer failed."
        exit 1
    }

    if (-not (($env:Path -split ";") -contains $poetryBin)) {
        $env:Path = "$poetryBin;$env:Path"
    }
    if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
        Write-Error "Poetry install completed but poetry not found on PATH. Reopen the shell and retry."
        exit 1
    }
}

function Test-CudaAvailable {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi -L > $null 2>&1
        return $LASTEXITCODE -eq 0
    }
    return $false
}

function Normalize-GpuMode {
    param([bool]$CudaAvailable)
    $mode = $env:AUTOCAPTURE_GPU_MODE
    if (-not $mode) {
        $env:AUTOCAPTURE_GPU_MODE = "auto"
        return
    }
    $normalized = $mode.Trim().ToLower()
    if ($normalized -eq "on" -and -not $CudaAvailable) {
        Write-Host "CUDA unavailable; overriding AUTOCAPTURE_GPU_MODE=on -> auto"
        $env:AUTOCAPTURE_GPU_MODE = "auto"
    }
}

Ensure-Poetry
$cudaAvailable = Test-CudaAvailable
Normalize-GpuMode $cudaAvailable

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

if (-not $env:AUTOCAPTURE_GPU_MODE) {
    $env:AUTOCAPTURE_GPU_MODE = "auto"
}

$extras = $env:AUTOCAPTURE_POETRY_EXTRAS
if (-not $extras) {
    $extrasList = @("ui", "windows", "ocr", "embed-fast", "embed-st", "sqlcipher")
    if ($cudaAvailable) {
        $extrasList += "ocr-gpu"
    }
    $extras = ($extrasList -join " ")
}
$groups = $env:AUTOCAPTURE_POETRY_GROUPS
if (-not $groups) {
    $groups = "dev"
}
function Get-PoetryInstallSignature {
    param(
        [string]$Extras,
        [string]$Groups
    )
    $lockPath = Join-Path $repoRoot "poetry.lock"
    if (-not (Test-Path $lockPath)) {
        return ""
    }
    $lockHash = (Get-FileHash $lockPath -Algorithm SHA256).Hash
    $pythonVersion = & poetry run python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    $pythonVersion = $pythonVersion.Trim()
    return "$lockHash|$Extras|$Groups|$pythonVersion"
}

$sig = Get-PoetryInstallSignature -Extras $extras -Groups $groups
$sigPath = Join-Path $cacheRoot "poetry_install.sig"
$prevSig = ""
if (Test-Path $sigPath) {
    $prevSig = (Get-Content $sigPath -Raw).Trim()
}
$venvPath = ""
try {
    $venvPath = (& poetry env info -p 2>$null).Trim()
} catch {
    $venvPath = ""
}
$venvOk = $false
if ($venvPath -and (Test-Path $venvPath)) {
    $venvOk = $true
} elseif (Test-Path (Join-Path $repoRoot ".venv")) {
    $venvOk = $true
}
$forceInstall = $env:AUTOCAPTURE_FORCE_POETRY_INSTALL
if ($forceInstall -and $forceInstall.Trim().ToLower() -in @("1","true","yes","on")) {
    $prevSig = ""
}
if ($sig -and $sig -eq $prevSig -and $venvOk) {
    Write-Host "Poetry deps already installed; skipping."
} else {
    poetry install --with $groups --extras $extras
    if ($sig) {
        Set-Content -Path $sigPath -Value $sig -Encoding UTF8
    }
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
    $baseDir = Join-Path $env:LOCALAPPDATA "Autocapture"
}
if ($baseDir) {
    Write-Host "Using base directory: $baseDir"
    Set-BaseDirInConfig $baseDir
}

poetry run autocapture setup --profile full --apply
poetry run python tools/vendor_windows_binaries.py

$prevSkipInstall = $env:AUTOCAPTURE_SKIP_POETRY_INSTALL
$prevSkipDoctor = $env:AUTOCAPTURE_SKIP_DOCTOR
$env:AUTOCAPTURE_SKIP_POETRY_INSTALL = "1"
$env:AUTOCAPTURE_SKIP_DOCTOR = "1"
& (Join-Path $repoRoot "scripts\\windows_llm_setup.ps1")
if ($null -ne $prevSkipInstall) {
    $env:AUTOCAPTURE_SKIP_POETRY_INSTALL = $prevSkipInstall
} else {
    Remove-Item Env:AUTOCAPTURE_SKIP_POETRY_INSTALL -ErrorAction SilentlyContinue
}
if ($null -ne $prevSkipDoctor) {
    $env:AUTOCAPTURE_SKIP_DOCTOR = $prevSkipDoctor
} else {
    Remove-Item Env:AUTOCAPTURE_SKIP_DOCTOR -ErrorAction SilentlyContinue
}

function Invoke-Doctor {
    $prevPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $prevNative = $null
    if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue) {
        $prevNative = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }
    $output = & poetry run autocapture doctor --verbose 2>&1
    $exitCode = $LASTEXITCODE
    if ($null -ne $prevNative) {
        $PSNativeCommandUseErrorActionPreference = $prevNative
    }
    $ErrorActionPreference = $prevPreference
    return @($output, $exitCode)
}

$doctorResult = Invoke-Doctor
$doctorOutput = $doctorResult[0]
$doctorExit = $doctorResult[1]
$doctorOutput | ForEach-Object { Write-Host $_ }

$autoReset = $env:AUTOCAPTURE_RESET_DB_ON_FAIL
if (-not $autoReset) {
    $env:AUTOCAPTURE_RESET_DB_ON_FAIL = "1"
    $autoReset = "1"
}

if ($doctorOutput -match "file is not a database|hmac check failed|sqlcipher") {
    if ($autoReset -and $autoReset.Trim().ToLower() -in @("1","true","yes","on")) {
        Write-Host "Resetting encrypted DB (AUTOCAPTURE_RESET_DB_ON_FAIL=1)."
        & (Join-Path $repoRoot "scripts\\windows_reset_db.ps1")
        $doctorResult = Invoke-Doctor
        $doctorOutput = $doctorResult[0]
        $doctorExit = $doctorResult[1]
        $doctorOutput | ForEach-Object { Write-Host $_ }
    } else {
        Write-Host "Encrypted DB appears invalid. Set AUTOCAPTURE_RESET_DB_ON_FAIL=1 to auto-reset."
    }
}

$missingMatches = [regex]::Matches(($doctorOutput -join "`n"), 'model_missing=([^\s]+)')
if ($missingMatches.Count -gt 0) {
    $missingModels = $missingMatches | ForEach-Object { $_.Groups[1].Value.Trim() } | Sort-Object -Unique
    $missingCsv = ($missingModels -join ",")
    Write-Host "Pulling missing Ollama models: $missingCsv"
    $prevModels = $env:AUTOCAPTURE_OLLAMA_MODELS
    $prevSkipInstall = $env:AUTOCAPTURE_SKIP_POETRY_INSTALL
    $prevSkipDoctor = $env:AUTOCAPTURE_SKIP_DOCTOR
    $env:AUTOCAPTURE_OLLAMA_MODELS = $missingCsv
    $env:AUTOCAPTURE_SKIP_POETRY_INSTALL = "1"
    $env:AUTOCAPTURE_SKIP_DOCTOR = "1"
    & (Join-Path $repoRoot "scripts\\windows_llm_setup.ps1")
    if ($null -ne $prevModels) {
        $env:AUTOCAPTURE_OLLAMA_MODELS = $prevModels
    } else {
        Remove-Item Env:AUTOCAPTURE_OLLAMA_MODELS -ErrorAction SilentlyContinue
    }
    if ($null -ne $prevSkipInstall) {
        $env:AUTOCAPTURE_SKIP_POETRY_INSTALL = $prevSkipInstall
    } else {
        Remove-Item Env:AUTOCAPTURE_SKIP_POETRY_INSTALL -ErrorAction SilentlyContinue
    }
    if ($null -ne $prevSkipDoctor) {
        $env:AUTOCAPTURE_SKIP_DOCTOR = $prevSkipDoctor
    } else {
        Remove-Item Env:AUTOCAPTURE_SKIP_DOCTOR -ErrorAction SilentlyContinue
    }
    $doctorResult = Invoke-Doctor
    $doctorOutput = $doctorResult[0]
    $doctorExit = $doctorResult[1]
    $doctorOutput | ForEach-Object { Write-Host $_ }
}

$skipAutoStart = $env:AUTOCAPTURE_SKIP_AUTO_START
if ($doctorOutput -notmatch "\\bFAIL\\b" -and $doctorExit -eq 0) {
    if ($skipAutoStart -and $skipAutoStart.Trim().ToLower() -in @("1","true","yes","on")) {
        Write-Host "Auto-start skipped (AUTOCAPTURE_SKIP_AUTO_START=1)."
    } else {
        function Get-ConfigProbe {
            $code = @'
import json
import os
from pathlib import Path
from autocapture.config import load_config
from autocapture.paths import default_config_path

config_path = Path(os.environ.get("AUTOCAPTURE_CONFIG") or default_config_path())
config = load_config(config_path)
payload = {
    "host": config.api.bind_host,
    "port": config.api.port,
    "log_dir": str(Path(config.capture.data_dir) / "logs"),
}
print(json.dumps(payload))
'@
            $tempPath = Join-Path ([System.IO.Path]::GetTempPath()) ("autocapture_probe_{0}.py" -f ([System.Guid]::NewGuid().ToString("N")))
            Set-Content -Path $tempPath -Value $code -Encoding UTF8
            $output = & poetry run python $tempPath 2>$null
            Remove-Item $tempPath -ErrorAction SilentlyContinue
            if ($LASTEXITCODE -ne 0 -or -not $output) {
                $fallback = @{
                    host = "127.0.0.1"
                    port = 8008
                    log_dir = (Join-Path $env:LOCALAPPDATA "Autocapture\\logs")
                }
                return $fallback
            }
            try {
                return $output | ConvertFrom-Json
            } catch {
                $fallback = @{
                    host = "127.0.0.1"
                    port = 8008
                    log_dir = (Join-Path $env:LOCALAPPDATA "Autocapture\\logs")
                }
                return $fallback
            }
        }

        function Wait-ApiHealthy {
            param(
                [string]$ApiHost,
                [int]$ApiPort
            )
            $deadline = [DateTime]::UtcNow.AddSeconds(45)
            $url = "http://{0}:{1}/health" -f $ApiHost, $ApiPort
            while ([DateTime]::UtcNow -lt $deadline) {
                try {
                    $resp = Invoke-WebRequest -UseBasicParsing $url -TimeoutSec 2
                    if ($resp.StatusCode -eq 200) {
                        return $true
                    }
                } catch {
                    Start-Sleep -Seconds 1
                }
            }
            return $false
        }

        function Show-LogTail {
            param([string]$Path, [string]$Label)
            if (-not (Test-Path $Path)) {
                Write-Host "$Label log not found: $Path"
                return
            }
            Write-Host ("==== {0} (tail) ====" -f $Label)
            Get-Content -Path $Path -Tail 120 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
        }

        function Stop-AutocaptureProcesses {
            $candidates = Get-CimInstance Win32_Process | Where-Object {
                $_.CommandLine -and (
                    $_.CommandLine -match "autocapture\\s+app" -or
                    $_.CommandLine -match "autocapture\\.exe" -or
                    $_.CommandLine -match "poetry\\.exe.*autocapture\\s+app"
                )
            }
            foreach ($proc in $candidates) {
                try {
                    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
                } catch {
                    # ignore
                }
            }
            if ($candidates.Count -gt 0) {
                Start-Sleep -Seconds 2
            }
        }

        Write-Host "Starting Autocapture..."
        Stop-AutocaptureProcesses
        $probe = Get-ConfigProbe
        $apiHost = $probe.host
        $apiPort = [int]$probe.port
        $logDir = $probe.log_dir
        $apiLog = Join-Path $logDir "api.log"
        $appLog = Join-Path $logDir "autocapture.log"

        Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "set AUTOCAPTURE_CONFIG=$env:AUTOCAPTURE_CONFIG && poetry run autocapture app" -WorkingDirectory $repoRoot
        if (Wait-ApiHealthy -Host $apiHost -Port $apiPort) {
            Write-Host "Autocapture is up: http://$apiHost`:$apiPort"
        } else {
            Write-Host "Autocapture did not become healthy on http://$apiHost`:$apiPort"
            Show-LogTail -Path $apiLog -Label "api.log"
            Show-LogTail -Path $appLog -Label "autocapture.log"
            Write-Host "If logs are empty, the process may have failed before logging started."
        }
    }
} else {
    Write-Host "Doctor reported failures; fix them before starting capture."
}
