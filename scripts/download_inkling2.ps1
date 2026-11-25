[CmdletBinding()]
param(
    [string]$OutDir = "",
    [switch]$SkipDFlash2
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    if ($env:SMALLTIDE_MODEL_DIR) {
        $OutDir = $env:SMALLTIDE_MODEL_DIR
        if (-not [System.IO.Path]::IsPathRooted($OutDir)) {
            $OutDir = Join-Path $repoRoot $OutDir
        }
    } else {
        $OutDir = Join-Path $repoRoot "models\Inkling-2-Mini"
    }
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$base = "https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main"
$files = @(
    "config.json",
    "generation_config.json",
    "hf_quant_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "mtp.safetensors",
    "model-00001-of-00001.safetensors"
)

$curlCmd = Get-Command curl.exe -ErrorAction SilentlyContinue
if (-not $curlCmd) {
    throw "curl.exe was not found on PATH (built into Windows 10/11)."
}

$headers = @()
if ($env:HF_TOKEN) {
    $headers = @("-H", "Authorization: Bearer $($env:HF_TOKEN)")
}

Write-Host "[Download] Target directory: $OutDir"
Write-Host "[Download] Source: https://huggingface.co/thinkingmachines/Inkling-2-Mini (~21 GB total, resumable)"

foreach ($f in $files) {
    $out = Join-Path $OutDir $f
    Write-Host "[Download] $f -> $out"
    & $curlCmd.Source -fL -C - --retry 3 --retry-delay 2 @headers "$base/$f" -o $out
    if ($LASTEXITCODE -ne 0) {
        throw "curl.exe failed for $f with exit code $LASTEXITCODE"
    }
}

Write-Host "[Download] All Inkling-2-Mini files completed!"

if (-not $SkipDFlash2) {
    $dflash2Dir = if ($env:SMALLTIDE_DFLASH2_DIR) {
        if ([System.IO.Path]::IsPathRooted($env:SMALLTIDE_DFLASH2_DIR)) { $env:SMALLTIDE_DFLASH2_DIR } else { Join-Path $repoRoot $env:SMALLTIDE_DFLASH2_DIR }
    } else {
        Join-Path (Split-Path -Parent $OutDir) "Inkling-2-Mini-DFlash2"
    }
    New-Item -ItemType Directory -Force -Path $dflash2Dir | Out-Null
    $dflash2Base = "https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/resolve/main"
    Write-Host ""
    Write-Host "[Download] DFlash 2 drafter: https://huggingface.co/incoai/Inkling-2-Mini-DFlash2 (~944 MB, resumable) -> $dflash2Dir"
    foreach ($f in @("config.json", "model.safetensors")) {
        $out = Join-Path $dflash2Dir $f
        Write-Host "[Download] $f -> $out"
        & $curlCmd.Source -fL -C - --retry 3 --retry-delay 2 @headers "$dflash2Base/$f" -o $out
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "[Download] Could not download DFlash 2 ($f, exit $LASTEXITCODE); SmallTide will fall back to built-in DSpark (mtp.safetensors)."
            break
        }
    }
}
