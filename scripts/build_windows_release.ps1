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

if ($env:AUTOCAPTURE_SIGNING_CERT_PATH) {
  Write-Host "Signing installer with provided certificate"
  if (-not (Get-Command signtool -ErrorAction SilentlyContinue)) {
    throw "signtool not found. Install Windows SDK to enable signing."
  }
  $certPath = $env:AUTOCAPTURE_SIGNING_CERT_PATH
  $installerPath = Join-Path $repoRoot "dist-installer/Autocapture-$version-Setup.exe"
  $timestampUrl = $env:AUTOCAPTURE_SIGNING_TIMESTAMP_URL
  if (-not $timestampUrl) {
    $timestampUrl = "http://timestamp.digicert.com"
  }
  $passwordArg = ""
  if ($env:AUTOCAPTURE_SIGNING_CERT_PASSWORD) {
    $passwordArg = "/p $env:AUTOCAPTURE_SIGNING_CERT_PASSWORD"
  }
  & signtool sign /f $certPath $passwordArg /tr $timestampUrl /td sha256 /fd sha256 $installerPath
}

Write-Host "Windows release build complete."
