$ErrorActionPreference = "Stop"
$dir = "c:\Users\raspb\git\smalltide\models\Inkling-2-Mini"
New-Item -ItemType Directory -Force -Path $dir | Out-Null

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

$headers = @()
if ($env:HF_TOKEN) {
    $headers = @("-H", "Authorization: Bearer $($env:HF_TOKEN)")
}

foreach ($f in $files) {
    $out = Join-Path $dir $f
    Write-Host "[Download] $f -> $out"
    & curl.exe -fL -C - @headers "$base/$f" -o $out
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed for $f with exit code $LASTEXITCODE"
    }
}
Write-Host "[Download] All Inkling-2-Mini files completed!"
