$ErrorActionPreference = "Stop"

$repoRoot = ""
try {
    $repoRoot = (git rev-parse --show-toplevel).Trim()
} catch {
    $repoRoot = (Get-Location).Path
}

Set-Location $repoRoot

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

Ensure-Poetry

$env:AUTOCAPTURE_TEST_MODE = "1"
$env:AUTOCAPTURE_GPU_MODE = "off"

Write-Host "== Autocapture CI preflight (Windows) =="
poetry install --with dev --extras "ui windows ocr ocr-gpu embed-fast sqlcipher"
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
poetry run pytest -q tests/test_overlay_tracker_windows.py

function Ensure-InnoSetup {
    if (Get-Command iscc -ErrorAction SilentlyContinue) {
        return $true
    }
    Write-Host "Inno Setup not found; attempting installation..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        & winget install --id JRSoftware.InnoSetup -e --accept-source-agreements --accept-package-agreements
    }
    if (-not (Get-Command iscc -ErrorAction SilentlyContinue) -and (Get-Command choco -ErrorAction SilentlyContinue)) {
        & choco install innosetup -y
    }
    return (Get-Command iscc -ErrorAction SilentlyContinue) -ne $null
}

$runBundle = $env:AUTOCAPTURE_PREFLIGHT_BUNDLE
if (-not $runBundle) {
    $runBundle = "0"
}

$runInstaller = $env:AUTOCAPTURE_PREFLIGHT_INSTALLER
if (-not $runInstaller) {
    $runInstaller = "0"
}

if ($runBundle -eq "1") {
    poetry run pyinstaller pyinstaller.spec
} else {
    Write-Host "== PyInstaller bundle skipped (AUTOCAPTURE_PREFLIGHT_BUNDLE=1 to enable) =="
}

if ($runInstaller -eq "1") {
    if ($runBundle -ne "1") {
        Write-Error "Installer requested but bundle disabled. Set AUTOCAPTURE_PREFLIGHT_BUNDLE=1."
        exit 1
    }
    if (-not (Ensure-InnoSetup)) {
        Write-Error "Inno Setup (iscc) not found and auto-install failed. Run this script in an elevated PowerShell."
        exit 1
    }
    iscc installer\\autocapture.iss
} else {
    Write-Host "== Inno Setup installer skipped (AUTOCAPTURE_PREFLIGHT_INSTALLER=1 to enable) =="
}

$skipWsl = $env:AUTOCAPTURE_PREFLIGHT_WSL
if (-not $skipWsl) {
    $skipWsl = "1"
}

if ($skipWsl -eq "0") {
    Write-Host "== WSL preflight skipped (AUTOCAPTURE_PREFLIGHT_WSL=0) =="
    exit 0
}

if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
    Write-Host "== WSL not available; skipping Linux preflight =="
    exit 0
}

try {
    $wslPath = & wsl.exe wslpath -u "$repoRoot" 2>$null
    $wslPath = $wslPath.Trim()
} catch {
    $wslPath = ""
}

if (-not $wslPath) {
    Write-Host "== WSL path conversion failed; skipping Linux preflight =="
    exit 0
}

Write-Host "== Autocapture CI preflight (WSL/Linux) =="
$wslCmd = "cd `"$wslPath`" && bash scripts/preflight_ci_wsl.sh"
wsl.exe -e bash -lc "$wslCmd"
