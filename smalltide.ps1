<#
.SYNOPSIS
    SmallTide launcher for Windows (works with pre-built release binaries or a source checkout).

.DESCRIPTION
    Commands:
      setup       Copy .env.example -> .env, verify AMD HIP GPU, download Inkling-2-Mini, pack experts.stide
      serve       Start the OpenAI-compatible HTTP server and /expert-map dashboard
      run ...     Run one-shot generation (forwards all flags to `smalltide-cli run`)
      probe       Verify AMD HIP DLL, GPU device, embedded kernels, and PCIe DMA bandwidth
      vram-probe  Measure on-card resident VRAM vs Windows system-memory spillover
      bench-gpu   Verify the GPU kernels and run GPU microbenchmarks
      bench ...   Run bench/bench.py against the running server (http://127.0.0.1:8000)
      pack        Pack model-00001-of-00001.safetensors into 4KB-aligned experts.stide
      build       Build smalltide-cli.exe from source (requires Rust + AMD ROCm HIP SDK)
      help        Show this help text

.EXAMPLE
    .\smalltide.ps1 setup
    .\smalltide.ps1 serve
    .\smalltide.ps1 run -p "Explain speculative decoding in 3 bullet points." -n 256
    .\smalltide.ps1 bench --only prose code --runs 2
#>
param(
    [Parameter(Position = 0)]
    [string]$Command = "help",

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

function Read-DotEnvValue {
    param([string]$EnvPath, [string]$Key, [string]$Default)
    if (-not (Test-Path $EnvPath)) { return $Default }
    foreach ($line in Get-Content $EnvPath) {
        $t = $line.Trim()
        if ($t -eq "" -or $t.StartsWith("#")) { continue }
        if ($t -match "^(?:export\s+)?$([regex]::Escape($Key))\s*=\s*(.+)$") {
            $v = $Matches[1].Trim()
            if (($v.StartsWith('"') -and $v.EndsWith('"')) -or ($v.StartsWith("'") -and $v.EndsWith("'"))) {
                $v = $v.Substring(1, $v.Length - 2)
            } elseif ($v -match "^(.*?)\s+#") {
                $v = $Matches[1].Trim()
            }
            if ($v -ne "") { return $v }
        }
    }
    return $Default
}

function Resolve-SmallTideExe {
    # 1. Pre-built release layout: smalltide-cli.exe right next to smalltide.ps1
    $prebuilt = Join-Path $Root "smalltide-cli.exe"
    if (Test-Path $prebuilt) { return $prebuilt }

    # 2. Source checkout release build
    $cargoRelease = Join-Path $Root "target\release\smalltide-cli.exe"
    if (Test-Path $cargoRelease) { return $cargoRelease }

    # 3. Offer to build from source if Cargo.toml is present
    if (Test-Path (Join-Path $Root "Cargo.toml")) {
        $cargo = Get-Command cargo -ErrorAction SilentlyContinue
        if ($cargo) {
            Write-Host "[smalltide] No binary found; building smalltide-cli.exe from source..."
            Push-Location $Root
            try {
                & cargo build --release -p smalltide-cli
                if ($LASTEXITCODE -ne 0) { throw "cargo build failed with exit code $LASTEXITCODE" }
            } finally {
                Pop-Location
            }
            return $cargoRelease
        }
    }

    throw "smalltide-cli.exe not found. Download the pre-built Windows release zip from GitHub Releases, or install Rust + AMD ROCm HIP SDK and run '.\smalltide.ps1 build'."
}

function Find-Python {
    foreach ($candidate in @("py", "python", "python3")) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source -notlike "*WindowsApps*") {
            return $cmd.Source
        }
    }
    $localPy = Join-Path $env:LOCALAPPDATA "Python\bin\python.exe"
    if (Test-Path $localPy) { return $localPy }
    throw "Python 3 not found on PATH (needed only for bench/bench.py)."
}

function Show-Help {
    Write-Host "SmallTide - Adaptive MoE inference for Inkling-2-Mini on AMD RDNA 4 (Windows)"
    Write-Host ""
    Write-Host "Usage: .\smalltide.cmd <command> [args...]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  setup       Create .env, verify AMD HIP GPU, download Inkling-2-Mini (~21 GB), pack experts.stide"
    Write-Host "  serve       Start OpenAI-compatible server (http://127.0.0.1:8000) + /expert-map"
    Write-Host "  run ...     Generate from the CLI (e.g. .\smalltide.cmd run -p `"Hello`" -n 256)"
    Write-Host "  probe       Verify AMD HIP runtime DLL, GPU, embedded kernels (gfx1201 / gfx1200), and DMA bandwidth"
    Write-Host "  vram-probe  Measure on-card resident VRAM vs Windows system-memory spillover"
    Write-Host "  bench-gpu   Run GPU kernel verification and GPU microbenchmarks"
    Write-Host "  bench ...   Run bench/bench.py against the running server"
    Write-Host "  pack        Build 4KB-aligned experts.stide from model-00001-of-00001.safetensors"
    Write-Host "  build       Compile smalltide-cli.exe from source (requires Rust + AMD ROCm HIP SDK)"
    Write-Host ""
}

Push-Location $Root
try {
    switch ($Command.ToLowerInvariant()) {
        "help" { Show-Help }
        "-h"   { Show-Help }
        "--help" { Show-Help }

        "build" {
            if (-not (Test-Path (Join-Path $Root "Cargo.toml"))) {
                throw "Cargo.toml not found here. This looks like a pre-built release package (no build needed!)."
            }
            $buildScript = Join-Path $Root "scripts\build.ps1"
            if (Test-Path $buildScript) {
                & $buildScript @Rest
            } else {
                & cargo build --release -p smalltide-cli @Rest
            }
            exit $LASTEXITCODE
        }

        "setup" {
            $envPath = Join-Path $Root ".env"
            $envExample = Join-Path $Root ".env.example"
            if (-not (Test-Path $envPath)) {
                if (Test-Path $envExample) {
                    Copy-Item $envExample $envPath
                    Write-Host "[setup] Created .env from .env.example"
                } else {
                    Write-Warning ".env.example not found; continuing with built-in defaults."
                }
            } else {
                Write-Host "[setup] Keeping existing .env"
            }

            $exe = Resolve-SmallTideExe
            Write-Host ""
            Write-Host "[setup] 1/3 Probing AMD HIP GPU..."
            & $exe probe
            if ($LASTEXITCODE -ne 0) {
                throw "GPU probe failed (see the message above). SmallTide needs an AMD Radeon RDNA 4 card (RX 9070 or RX 9060 family) and the Adrenalin 26.10.44+ driver."
            }

            $modelDirRel = if ($env:SMALLTIDE_MODEL_DIR) { $env:SMALLTIDE_MODEL_DIR } else { Read-DotEnvValue -EnvPath $envPath -Key "SMALLTIDE_MODEL_DIR" -Default "models/Inkling-2-Mini" }
            $modelDir = if ([System.IO.Path]::IsPathRooted($modelDirRel)) { $modelDirRel } else { Join-Path $Root $modelDirRel }

            Write-Host ""
            Write-Host "[setup] 2/3 Downloading Inkling-2-Mini weights into $modelDir ..."
            $dlScript = Join-Path $Root "scripts\download_inkling2.ps1"
            & $dlScript -OutDir $modelDir
            if ($LASTEXITCODE -ne 0) { throw "Download failed with exit code $LASTEXITCODE" }

            $stPath = Join-Path $modelDir "model-00001-of-00001.safetensors"
            $packPath = Join-Path $modelDir "experts.stide"
            Write-Host ""
            if (Test-Path $packPath) {
                $packGb = [math]::Round((Get-Item $packPath).Length / 1GB, 2)
                Write-Host "[setup] 3/3 Packed expert store already exists: $packPath ($packGb GB)"
            } else {
                Write-Host "[setup] 3/3 Packing 3,072 routed experts into 4KB-aligned $packPath (~15s)..."
                & $exe pack $stPath $packPath
                if ($LASTEXITCODE -ne 0) { throw "Packing experts.stide failed with exit code $LASTEXITCODE" }
            }

            Write-Host ""
            Write-Host "[setup] Done! Start SmallTide with:"
            Write-Host "  .\smalltide.cmd serve"
            Write-Host "Then open http://127.0.0.1:8000/expert-map or connect any OpenAI client to http://127.0.0.1:8000/v1"
        }

        "pack" {
            $exe = Resolve-SmallTideExe
            if ($Rest.Count -ge 2) {
                & $exe pack @Rest
            } else {
                $envPath = Join-Path $Root ".env"
                $modelDirRel = if ($env:SMALLTIDE_MODEL_DIR) { $env:SMALLTIDE_MODEL_DIR } else { Read-DotEnvValue -EnvPath $envPath -Key "SMALLTIDE_MODEL_DIR" -Default "models/Inkling-2-Mini" }
                $modelDir = if ([System.IO.Path]::IsPathRooted($modelDirRel)) { $modelDirRel } else { Join-Path $Root $modelDirRel }
                & $exe pack (Join-Path $modelDir "model-00001-of-00001.safetensors") (Join-Path $modelDir "experts.stide")
            }
            exit $LASTEXITCODE
        }

        "serve" {
            $exe = Resolve-SmallTideExe
            & $exe serve @Rest
            exit $LASTEXITCODE
        }

        "run" {
            $exe = Resolve-SmallTideExe
            & $exe run @Rest
            exit $LASTEXITCODE
        }

        "probe" {
            $exe = Resolve-SmallTideExe
            & $exe probe @Rest
            exit $LASTEXITCODE
        }

        "vram-probe" {
            $exe = Resolve-SmallTideExe
            & $exe vram-probe @Rest
            exit $LASTEXITCODE
        }

        "bench-gpu" {
            $exe = Resolve-SmallTideExe
            & $exe bench-gpu @Rest
            exit $LASTEXITCODE
        }

        "bench" {
            $py = Find-Python
            $benchPy = Join-Path $Root "bench\bench.py"
            if (-not (Test-Path $benchPy)) { throw "bench/bench.py not found." }
            & $py $benchPy @Rest
            exit $LASTEXITCODE
        }

        default {
            # Pass unknown commands directly to smalltide-cli.exe if valid
            $exe = Resolve-SmallTideExe
            & $exe $Command @Rest
            exit $LASTEXITCODE
        }
    }
} finally {
    Pop-Location
}
