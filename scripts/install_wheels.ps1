param(
  [string]$VenvPath = ".venv"
)
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$wheelDir = Join-Path $root "..\dev_bundle\wheels"
$venvPython = Join-Path $root "$VenvPath\Scripts\python.exe"
if (-Not (Test-Path $venvPython)) {
  Write-Error "Venv python not found at $venvPython. Run scripts/create_venv.ps1 first."
  exit 1
}
if (-Not (Test-Path $wheelDir)) {
  Write-Host "No local wheels directory at $wheelDir; nothing to install."
  exit 0
}
$wheels = Get-ChildItem -Path $wheelDir -Filter "*.whl" -File
if (-Not $wheels) {
  Write-Host "No .whl files found in $wheelDir"
  exit 0
}
foreach ($w in $wheels) {
  Write-Host "Installing wheel: $($w.Name)"
  & $venvPython -m pip install --no-deps --no-index --find-links $wheelDir $w.FullName
}
Write-Host "Local wheel installation complete."