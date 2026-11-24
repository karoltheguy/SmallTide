<#
.SYNOPSIS
    Builds a self-contained Windows GitHub Release zip for SmallTide (no Rust or ROCm SDK needed by end users).

.EXAMPLE
    .\scripts\package-release.ps1
    .\scripts\package-release.ps1 -SkipBuild
#>
[CmdletBinding()]
param(
    [switch]$SkipBuild,
    [switch]$NoProbe,
    # Kernel builds to embed, comma-separated; the first one names the package.
    [string]$Arch = "gfx1201,gfx1200",
    [string]$TargetDir = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

try {
    $version = (Get-Content (Join-Path $repoRoot "VERSION") -Raw).Trim()
    if ([string]::IsNullOrWhiteSpace($version)) {
        throw "VERSION file is empty."
    }

    $cargoArgs = @("build", "--release", "-p", "smalltide-cli")
    if (-not [string]::IsNullOrWhiteSpace($TargetDir)) {
        $cargoArgs += @("--target-dir", $TargetDir)
    }

    if (-not $SkipBuild) {
        Write-Host "[package] Building release binary for $Arch ..."
        $env:SMALLTIDE_HIP_ARCH = $Arch
        & cargo @cargoArgs
        if ($LASTEXITCODE -ne 0) {
            throw "cargo build --release failed (exit $LASTEXITCODE). If smalltide-cli.exe is currently running, stop the server first, pass -TargetDir target\pkg-build, or pass -SkipBuild."
        }
    }

    $baseTarget = if ([string]::IsNullOrWhiteSpace($TargetDir)) { Join-Path $repoRoot "target" } elseif ([System.IO.Path]::IsPathRooted($TargetDir)) { $TargetDir } else { Join-Path $repoRoot $TargetDir }
    $exePath = Join-Path $baseTarget "release\smalltide-cli.exe"
    if (-not (Test-Path $exePath)) {
        throw "Release binary not found at $exePath"
    }

    if (-not $NoProbe) {
        Write-Host "[package] Verifying embedded .hsaco kernels via probe..."
        $probeOut = & $exePath probe 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "smalltide-cli.exe probe failed:`n$($probeOut -join "`n")"
        }
        $hsacoLine = $probeOut | Where-Object { $_ -match "Embedded HSACO\s*:\s*(\d+)\s*bytes" }
        if (-not $hsacoLine -or ([int]$Matches[1] -le 0)) {
            throw "Embedded HSACO is 0 bytes! Rebuild with AMD ROCm HIP SDK (hipcc) installed before releasing."
        }
        Write-Host "[package] $hsacoLine"
    }

    $pkgArch = ($Arch -split ",")[0].Trim()
    $pkgName = "smalltide-v$version-windows-x64-$pkgArch"
    $distDir = Join-Path $repoRoot "dist"
    $stageDir = Join-Path $distDir $pkgName
    $zipPath = Join-Path $distDir "$pkgName.zip"
    $shaPath = "$zipPath.sha256"

    if (Test-Path $stageDir) { Remove-Item -Recurse -Force $stageDir }
    if (Test-Path $zipPath)  { Remove-Item -Force $zipPath }
    if (Test-Path $shaPath)  { Remove-Item -Force $shaPath }

    New-Item -ItemType Directory -Force -Path $stageDir | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $stageDir "scripts") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $stageDir "profiles") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $stageDir "bench") | Out-Null

    # Core binary (self-contained, with the embedded .hsaco kernels of every architecture in $Arch)
    Copy-Item $exePath (Join-Path $stageDir "smalltide-cli.exe")

    # Launcher, config template, docs & license
    foreach ($f in @("smalltide.cmd", "smalltide.ps1", ".env.headless.example", ".env.monitor.example", "README.md", "LICENSE", "LIMITATIONS.md", "CREDITS.md", "VERSION")) {
        $src = Join-Path $repoRoot $f
        if (-not (Test-Path $src)) { throw "Missing required release file: $f" }
        Copy-Item $src (Join-Path $stageDir $f)
    }

    # Resumable model downloader (zero external dependencies, uses Windows curl.exe)
    Copy-Item (Join-Path $repoRoot "scripts\download_inkling2.ps1") (Join-Path $stageDir "scripts\download_inkling2.ps1")

    # Fresh-start aggregate routing priors (floats only, no user data)
    Copy-Item (Join-Path $repoRoot "profiles\inkling2-mini-priors.json") (Join-Path $stageDir "profiles\inkling2-mini-priors.json")

    # Benchmark harness
    Copy-Item (Join-Path $repoRoot "bench\bench.py") (Join-Path $stageDir "bench\bench.py")
    Copy-Item (Join-Path $repoRoot "bench\edit_source.py.txt") (Join-Path $stageDir "bench\edit_source.py.txt")

    Write-Host "[package] Creating $zipPath ..."
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $stageDir,
        $zipPath,
        [System.IO.Compression.CompressionLevel]::Optimal,
        $true
    )

    $hash = (Get-FileHash -Algorithm SHA256 $zipPath).Hash.ToLowerInvariant()
    "$hash  $pkgName.zip" | Set-Content -Encoding Ascii $shaPath

    $sizeMb = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
    Write-Host ""
    Write-Host "[package] Release bundle ready!"
    Write-Host "  Archive : $zipPath ($sizeMb MB)"
    Write-Host "  SHA256  : $hash"
    Write-Host "  Staging : $stageDir"
} finally {
    Pop-Location
}
