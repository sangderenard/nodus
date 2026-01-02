[CmdletBinding(SupportsShouldProcess=$true, ConfirmImpact='High')]
param(
  # Build directory that contains the PyTorch FetchContent build (default matches repo convention)
  [Parameter(Position=0)]
  [string]$BuildDir = "build-torch-avx2",

  # Remove the FetchContent deps build tree (largest chunk). Recommended when changing Torch/XPU/Kineto flags.
  [switch]$WipeDeps,

  # Remove the entire build directory (most aggressive).
  [switch]$WipeAll,

  # Skip confirmation prompts.
  [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-RepoPath {
  param([string]$Relative)
  $root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
  return (Join-Path $root $Relative)
}

$buildPath = Resolve-RepoPath $BuildDir

if (-not (Test-Path -LiteralPath $buildPath)) {
  Write-Host "Nothing to wipe: '$buildPath' does not exist." -ForegroundColor Yellow
  exit 0
}

if ($WipeAll) {
  $items = @($buildPath)
} else {
  $items = @(
    (Join-Path $buildPath 'CMakeCache.txt'),
    (Join-Path $buildPath 'CMakeFiles')
  )

  if ($WipeDeps) {
    $items += @(
      (Join-Path $buildPath '_deps'),
      (Join-Path $buildPath 'pytorch_source-build')
    )
  }
}

$items = $items | Where-Object { Test-Path -LiteralPath $_ }

if ($items.Count -eq 0) {
  Write-Host "Nothing to wipe in '$buildPath' (already clean)." -ForegroundColor Yellow
  exit 0
}

$what = if ($WipeAll) { "entire build directory" } elseif ($WipeDeps) { "cache + _deps" } else { "cache only" }
Write-Host "Wiping Torch build artifacts ($what) under: $buildPath" -ForegroundColor Cyan

foreach ($p in $items) {
  if ($PSCmdlet.ShouldProcess($p, 'Remove-Item')) {
    if ($Force) {
      Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction Stop
    } else {
      Remove-Item -LiteralPath $p -Recurse -Confirm -ErrorAction Stop
    }
  }
}

Write-Host "Done." -ForegroundColor Green

<##
Examples:
  # Fast: just reset configure cache
  ./scripts/wipe_torch_build.ps1

  # Common: reset cache + fetched deps build tree
  ./scripts/wipe_torch_build.ps1 -WipeDeps -Force

  # Nuclear: remove the whole build folder
  ./scripts/wipe_torch_build.ps1 -WipeAll -Force

Then reconfigure:
  cmake -S . -B build-torch-avx2 -DNODUS_ENABLE_TORCH=ON -DUSE_XPU=OFF -DLIBKINETO_NOXPUPTI=ON
  cmake --build build-torch-avx2 --config Release
##>
