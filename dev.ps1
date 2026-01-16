param(
    [Parameter(Position = 0)]
    [string]$Command = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Poetry {
    if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
        throw "Poetry is required. Install it first (https://python-poetry.org/docs/#installation)."
    }
}

function Ensure-DevEnv {
    if (-not $env:APP_ENV) {
        $env:APP_ENV = "dev"
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )
    & $Command @Args
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Invoke-FormatCheck {
    Invoke-Checked poetry run black --check .
}

function Show-Usage {
    Write-Host "Usage: .\\dev.ps1 <check|run|test|build|format|format:check|smoke>" -ForegroundColor Yellow
}



switch ($Command) {
    "check" {
        Ensure-Poetry
        Invoke-FormatCheck
        Invoke-Checked poetry run ruff check .
        Invoke-Checked poetry run pytest -q
        Ensure-DevEnv
        Invoke-Checked poetry run autocapture smoke
    }
    "run" {
        Ensure-Poetry
        Ensure-DevEnv
        Invoke-Checked poetry run autocapture
    }
    "test" {
        Ensure-Poetry
        Invoke-Checked poetry run pytest -q
    }
    "build" {
        Ensure-Poetry
        Invoke-Checked poetry build
    }
    "format" {
        Ensure-Poetry
        Invoke-Checked poetry run ruff check . --fix
        Invoke-Checked poetry run black .
    }
    "format:check" {
        Ensure-Poetry
        Invoke-FormatCheck
    }
    "smoke" {
        Ensure-Poetry
        Ensure-DevEnv
        Invoke-Checked poetry run autocapture smoke
    }
    Default {
        Show-Usage
        exit 1
    }
}
