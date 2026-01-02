# Create a repo virtualenv at .venv and install requirements + local wheels
param(
  [string]$VenvPath = ".venv"
)
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvPython = Join-Path $root $VenvPath\Scripts\python.exe
if (-Not (Test-Path $venvPython)) {
  Write-Host "Creating venv at $VenvPath..."
  python -m venv $VenvPath
} else {
  Write-Host "Venv already exists at $VenvPath"
}
if (-Not (Test-Path $venvPython)) {
  Write-Error "Python venv not found. Ensure 'python' on PATH can create a venv."
  exit 2
}
# Upgrade pip and install requirements if present
& $venvPython -m pip install --upgrade pip
$req = Join-Path $root "requirements.txt"
if (Test-Path $req) {
  Write-Host "Installing requirements from requirements.txt..."
  & $venvPython -m pip install -r $req
}
# Install any local wheels
$installScript = Join-Path $root "scripts\install_wheels.ps1"
if (Test-Path $installScript) {
  & powershell -ExecutionPolicy Bypass -File $installScript -VenvPath $VenvPath
} else {
  Write-Host "No install_wheels.ps1 found; skipping local wheel install."
}
Write-Host "Venv setup complete. To activate: .\$VenvPath\Scripts\Activate.ps1"