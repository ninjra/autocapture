$ErrorActionPreference = "Stop"

$repoRoot = ""
try {
    $repoRoot = (git rev-parse --show-toplevel).Trim()
} catch {
    $repoRoot = (Get-Location).Path
}

Set-Location $repoRoot

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

function Ensure-Ollama {
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        return
    }
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Error "Ollama not found and winget is unavailable. Install Ollama manually."
        exit 1
    }
    Write-Host "Installing Ollama via winget..."
    & winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Write-Error "Ollama install completed but ollama is not on PATH. Reopen the shell and retry."
        exit 1
    }
}

function Start-Ollama {
    if (Get-Process ollama -ErrorAction SilentlyContinue) {
        return
    }
    Write-Host "Starting Ollama service..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 1
}

function Wait-Ollama {
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:11434/api/tags" -TimeoutSec 2
            if ($resp.StatusCode -eq 200) {
                return
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    Write-Error "Ollama did not start on http://127.0.0.1:11434 within 30s."
    exit 1
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

function Ensure-Torch {
    param([bool]$CudaAvailable)
    $torchStatus = & poetry run python -c "import importlib.util, sys; spec = importlib.util.find_spec('torch'); sys.exit(2 if spec is None else (0 if __import__('torch').cuda.is_available() else 1))"
    if ($LASTEXITCODE -eq 0) {
        return
    }
    if ($CudaAvailable) {
        Write-Host "Installing PyTorch (CUDA)..."
        poetry run pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    } else {
        Write-Host "Installing PyTorch (CPU)..."
        poetry run pip install --upgrade torch torchvision torchaudio
    }
}

function Get-DesiredModels {
    $override = $env:AUTOCAPTURE_OLLAMA_MODELS
    if ($override) {
        return $override.Split(",") | ForEach-Object {
            $name = $_.Trim()
            if ($name) { [PSCustomObject]@{ model = $name } }
        } | Where-Object { $_ }
    }
    $output = & poetry run python tools/warm_models.py --ollama-models-json 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to read config for Ollama models."
        exit 1
    }
    if (-not $output) {
        return @()
    }
    try {
        $parsed = $output | ConvertFrom-Json
        if ($parsed -is [System.Array]) {
            return $parsed
        }
        return @($parsed)
    } catch {
        Write-Error "Failed to parse Ollama model map."
        exit 1
    }
}

function Get-OllamaFallback {
    param([string]$Kind)
    if ($Kind -eq "vision") {
        if ($env:AUTOCAPTURE_OLLAMA_FALLBACK_VISION) {
            return $env:AUTOCAPTURE_OLLAMA_FALLBACK_VISION
        }
        return "llava"
    }
    if ($Kind -eq "text") {
        if ($env:AUTOCAPTURE_OLLAMA_FALLBACK_TEXT) {
            return $env:AUTOCAPTURE_OLLAMA_FALLBACK_TEXT
        }
        return "llama3"
    }
    return $null
}

function Patch-ConfigValue {
    param(
        [string]$Path,
        [string]$Value
    )
    if (-not $Path) {
        return
    }
    $env:AUTOCAPTURE_PATCH_PATH = $Path
    $env:AUTOCAPTURE_PATCH_VALUE = $Value
    $code = @'
import os
from pathlib import Path
import yaml

config_path = Path(os.environ["AUTOCAPTURE_CONFIG"])
data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
path = os.environ["AUTOCAPTURE_PATCH_PATH"].split(".")
value = os.environ["AUTOCAPTURE_PATCH_VALUE"]
current = data
for key in path[:-1]:
    if not isinstance(current.get(key), dict):
        current[key] = {}
    current = current[key]
current[path[-1]] = value
config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
'@
    $tempPath = Join-Path ([System.IO.Path]::GetTempPath()) ("autocapture_patch_{0}.py" -f ([System.Guid]::NewGuid().ToString("N")))
    Set-Content -Path $tempPath -Value $code -Encoding UTF8
    & poetry run python $tempPath 2>$null
    Remove-Item $tempPath -ErrorAction SilentlyContinue
    Remove-Item Env:AUTOCAPTURE_PATCH_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:AUTOCAPTURE_PATCH_VALUE -ErrorAction SilentlyContinue
}

function Pull-Models {
    param([object[]]$ModelMap)
    $pullCache = @{}
    foreach ($entry in $ModelMap) {
        $model = $entry.model
        if (-not $model) {
            continue
        }
        $model = ($model -split "\\s+")[0].Trim()
        if (-not $model) {
            continue
        }
        if (-not $pullCache.ContainsKey($model)) {
            Write-Host "Ensuring Ollama model: $model"
            & ollama pull $model
            $pullCache[$model] = ($LASTEXITCODE -eq 0)
        }
        if ($pullCache[$model]) {
            continue
        }
        $fallback = Get-OllamaFallback $entry.kind
        if (-not $fallback) {
            Write-Error "Ollama model '$model' failed to pull and no fallback is configured."
            exit 1
        }
        Write-Host "Model '$model' unavailable; switching to '$fallback'."
        if ($entry.path) {
            Patch-ConfigValue $entry.path $fallback
        }
        if (-not $pullCache.ContainsKey($fallback)) {
            Write-Host "Ensuring Ollama model: $fallback"
            & ollama pull $fallback
            $pullCache[$fallback] = ($LASTEXITCODE -eq 0)
        }
        if (-not $pullCache[$fallback]) {
            Write-Error "Failed to pull fallback Ollama model '$fallback'."
            exit 1
        }
    }
}

$extras = $env:AUTOCAPTURE_POETRY_EXTRAS
if (-not $extras) {
    $cudaAvailable = Test-CudaAvailable
    Normalize-GpuMode $cudaAvailable
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
Ensure-Poetry
$skipInstall = $env:AUTOCAPTURE_SKIP_POETRY_INSTALL
if ($skipInstall -and $skipInstall.Trim().ToLower() -in @("1","true","yes","on")) {
    Write-Host "Skipping Poetry install (AUTOCAPTURE_SKIP_POETRY_INSTALL=1)."
} else {
    Write-Host "Installing Python deps (extras: $extras)..."
    poetry install --with $groups --extras "$extras"
}
if (-not $cudaAvailable) {
    $cudaAvailable = Test-CudaAvailable
    Normalize-GpuMode $cudaAvailable
}
Ensure-Torch $cudaAvailable

$models = Get-DesiredModels
if ($models.Count -gt 0) {
    Ensure-Ollama
    Start-Ollama
    Wait-Ollama
    Pull-Models $models
} else {
    Write-Host "Ollama not required by config; skipping install/pull."
}

Write-Host "Warming local models..."
$warmVlm = $env:AUTOCAPTURE_WARM_VLM
if ($warmVlm -and $warmVlm.Trim().ToLower() -in @("0","false","no","off")) {
    poetry run python tools/warm_models.py --skip-vlm
} else {
    poetry run python tools/warm_models.py
}

$skipDoctor = $env:AUTOCAPTURE_SKIP_DOCTOR
if ($skipDoctor -and $skipDoctor.Trim().ToLower() -in @("1","true","yes","on")) {
    Write-Host "Doctor skipped (AUTOCAPTURE_SKIP_DOCTOR=1)."
} else {
    Write-Host "Running doctor checks..."
    poetry run autocapture doctor --verbose
}

Write-Host "Ollama + local models ready."
