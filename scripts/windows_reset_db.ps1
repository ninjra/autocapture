$ErrorActionPreference = "Stop"

$repoRoot = ""
try {
    $repoRoot = (git rev-parse --show-toplevel).Trim()
} catch {
    $repoRoot = (Get-Location).Path
}

$configPath = $env:AUTOCAPTURE_CONFIG
if (-not $configPath) {
    $configPath = Join-Path $env:LOCALAPPDATA "Autocapture\\autocapture.yml"
}

$dataDir = Join-Path $env:LOCALAPPDATA "Autocapture\\data"
$dbPath = Join-Path $dataDir "autocapture.db"
$keyPath = Join-Path $dataDir "secrets\\sqlcipher.key"
$tmpPath = "$dbPath.enc.tmp"
$backupPath = "$dbPath.bak"

Write-Host "This will reset the encrypted DB used at:"
Write-Host "  $dbPath"
Write-Host "It will also rotate the SQLCipher key at:"
Write-Host "  $keyPath"

$stamp = Get-Date -Format "yyyyMMddHHmmss"
if (Test-Path $dbPath) {
    Copy-Item $dbPath "$dbPath.bak.$stamp"
    Remove-Item $dbPath
}
if (Test-Path $tmpPath) {
    Remove-Item $tmpPath
}
if (Test-Path $backupPath) {
    Copy-Item $backupPath "$backupPath.$stamp"
}
if (Test-Path $keyPath) {
    Copy-Item $keyPath "$keyPath.bak.$stamp"
    Remove-Item $keyPath
}

Write-Host "Reset complete. Re-run:"
Write-Host "  .\\scripts\\windows_full_setup.bat"
