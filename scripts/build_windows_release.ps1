$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "Resolving version from pyproject.toml"
$version = python -c "import tomllib, pathlib; data = tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8')); print(data['tool']['poetry']['version'])"
if (-not $version) {
  throw "Unable to resolve version from pyproject.toml"
}

Write-Host "Downloading vendor binaries"
python tools/vendor_windows_binaries.py

Write-Host "Building PyInstaller bundle"
poetry run pyinstaller pyinstaller.spec

Write-Host "Building Inno Setup installer"
if (-not (Get-Command iscc -ErrorAction SilentlyContinue)) {
  throw "Inno Setup compiler (iscc) not found. Install Inno Setup."
}
& iscc installer/autocapture.iss /DMyAppVersion=$version

Write-Host "Windows release build complete."
