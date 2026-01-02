param(
  [string]$VenvPath = ".venv",
  [string]$BuildDir = "build",
  [string]$Config = "Release",
  [string]$WheelDir = "dev_bundle/wheels",
  [string]$CMakeArgs = ""
)

# Determine repository root (one level up from the scripts directory)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$root = (Resolve-Path (Join-Path $scriptDir ".." )).Path
Set-StrictMode -Version Latest

# Resolve full paths
$venvFull = Join-Path $root $VenvPath
$venvPython = Join-Path $venvFull "Scripts\python.exe"
$wheelDirFull = Join-Path $root $WheelDir

function Ensure-Venv {
  if (-Not (Test-Path $venvPython)) {
    Write-Host "Venv not found at $VenvPath; creating..."
    & python -m venv $venvFull
    if ($LASTEXITCODE -ne 0) {
      Write-Error "Failed to create venv using 'python -m venv'. Ensure a system 'python' is available on PATH."
      exit $LASTEXITCODE
    }
  } else {
    Write-Host "Using existing venv at $venvFull"
  }
  if (-Not (Test-Path $venvPython)) {
    Write-Error "Venv python not found at $venvPython after creation."
    exit 2
  }
  & $venvPython -m pip install --upgrade pip setuptools wheel | Out-Null
}

function Run-CMake-Build {
  Push-Location $root
  Write-Host "Configuring CMake (dir: $BuildDir)"
  $configureArgs = @("-S", ".", "-B", $BuildDir)
  if ($CMakeArgs -ne "") { $configureArgs += $CMakeArgs }
  & cmake @configureArgs
  if ($LASTEXITCODE -ne 0) { Write-Error "CMake configure failed"; Pop-Location; exit $LASTEXITCODE }

  Write-Host "Building project"
  & cmake --build $BuildDir --config $Config
  if ($LASTEXITCODE -ne 0) { Write-Error "CMake build failed"; Pop-Location; exit $LASTEXITCODE }
  Pop-Location
}

function Find-Wheels {
  $found = @()
  if (Test-Path $wheelDirFull) {
    $found += Get-ChildItem -Path $wheelDirFull -Filter "*.whl" -File -Recurse -ErrorAction SilentlyContinue
  }
  # Also search the build tree for wheels (some projects output to build/dist or similar)
  $buildFull = Join-Path $root $BuildDir
  if (Test-Path $buildFull) {
    $found += Get-ChildItem -Path $buildFull -Filter "*.whl" -File -Recurse -ErrorAction SilentlyContinue
  }
  # Deduplicate
  $found = $found | Sort-Object -Property FullName -Unique
  return $found
}

function Install-Wheels($wheels) {
  foreach ($w in $wheels) {
    Write-Host "Installing wheel: $($w.FullName)"
    & $venvPython -m pip install --no-deps --no-index --find-links $wheelDirFull $w.FullName
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to install $($w.Name)"; exit $LASTEXITCODE }
  }
}

# Main flow
Ensure-Venv
Run-CMake-Build
$wheels = Find-Wheels
if (-Not $wheels -or $wheels.Count -eq 0) {
  Write-Host "No .whl files found in $WheelDir or $BuildDir. Nothing to install."
  exit 0
}
Install-Wheels $wheels
Write-Host "Build-and-install complete."