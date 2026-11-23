<#
.SYNOPSIS
    Build SmallTide (smalltide-cli.exe) cleanly on Windows.

.DESCRIPTION
    Locates cargo.exe (including %USERPROFILE%\.cargo\bin), optionally stops any
    running smalltide-cli.exe that would lock the output binary, runs tests if
    requested, builds smalltide-cli.exe, and can relaunch the server right away.

.EXAMPLE
    .\scripts\build.ps1
    .\scripts\build.ps1 -Test
    .\scripts\build.ps1 -Stop -Serve
    .\scripts\build.ps1 -Offline -Test
#>
param(
    [switch]$Test,
    [switch]$Offline,
    [switch]$DebugBuild,
    [switch]$Stop,
    [switch]$Serve,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CargoArgs = @()
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

function Find-Cargo {
    $cmd = Get-Command cargo -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $fallback = Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"
    if (Test-Path $fallback) { return $fallback }
    throw "cargo.exe not found on PATH or at $fallback. Install Rust from https://rustup.rs"
}

$cargo = Find-Cargo
$profileArgs = if ($DebugBuild) { @() } else { @("--release") }
$offlineArgs = if ($Offline) { @("--offline") } else { @() }
$profileDir = if ($DebugBuild) { "debug" } else { "release" }
$exePath = Join-Path $Root "target\$profileDir\smalltide-cli.exe"

Push-Location $Root
try {
    if ($Test) {
        Write-Host "[build] Running workspace tests..." -ForegroundColor Cyan
        & $cargo test @offlineArgs @profileArgs --workspace
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    # Check if a running smalltide-cli.exe locks the target binary
    $running = @(Get-Process -Name "smalltide-cli" -ErrorAction SilentlyContinue)
    if ($running.Count -gt 0) {
        if ($Stop -or $Serve) {
            Write-Host "[build] Stopping $($running.Count) running smalltide-cli process(es)..." -ForegroundColor Yellow
            $running | Stop-Process -Force
            Start-Sleep -Milliseconds 400
        } else {
            Write-Warning "smalltide-cli.exe is currently running (PID $($running.Id -join ', ')) and may lock $exePath. Re-run with -Stop (or -Serve) to stop it automatically."
        }
    }

    $t0 = Get-Date
    Write-Host "[build] Building smalltide-cli ($profileDir)..." -ForegroundColor Cyan
    & $cargo build @offlineArgs @profileArgs -p smalltide-cli @CargoArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $elapsed = ((Get-Date) - $t0).TotalSeconds
    $sizeMb = if (Test-Path $exePath) { [math]::Round((Get-Item $exePath).Length / 1MB, 2) } else { 0 }
    Write-Host ("[build] Ready: {0} ({1} MB) in {2:F2}s" -f $exePath, $sizeMb, $elapsed) -ForegroundColor Green

    if ($Serve) {
        Write-Host "[build] Starting smalltide-cli serve..." -ForegroundColor Cyan
        & $exePath serve
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}
