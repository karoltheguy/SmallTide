#!/usr/bin/env bash
# SmallTide container entrypoint.
#
#   serve [env-file]  (default) First start: create /data/.env, check the GPU, download and pack
#                     the model. Every start: run the OpenAI-compatible server on port 8000.
#   setup             Run the first-start steps again (resumes a partial download).
#   bench ...         Run bench/bench.py against the server in this container.
#   shell             Open a bash shell.
#   anything else     Passed to smalltide-cli (probe, run -p "...", vram-probe, bench-gpu, ...).
#
# Environment:
#   SMALLTIDE_CONFIG      headless (default) or monitor: template for a new /data/.env
#   SMALLTIDE_AUTO_SETUP  1 (default) downloads the model on first start; 0 fails instead
#   HF_TOKEN              optional, for higher Hugging Face rate limits

set -euo pipefail

APP=/opt/smalltide
CLI="$APP/smalltide-cli"
DATA="${SMALLTIDE_DATA:-/data}"

log() { echo "[smalltide] $*"; }
die() { echo "[smalltide] error: $*" >&2; exit 1; }

cd "$DATA" || die "$DATA is not available; mount a volume there (-v smalltide-data:/data)"

ensure_env() {
    [[ -f .env ]] && return
    local config="${SMALLTIDE_CONFIG:-headless}"
    [[ "$config" == headless || "$config" == monitor ]] \
        || die "SMALLTIDE_CONFIG must be headless or monitor (got '$config')"
    cp "$APP/.env.$config.example" .env
    log "Created $DATA/.env from the $config template; edit it to tune, then restart."
}

check_devices() {
    [[ -e /dev/kfd ]] || die "/dev/kfd is missing: start the container with --device /dev/kfd --device /dev/dri"
    [[ -d /dev/dri ]] || die "/dev/dri is missing: start the container with --device /dev/kfd --device /dev/dri"
}

# Absolute model directory: $SMALLTIDE_MODEL_DIR, else the last SMALLTIDE_MODEL_DIR in .env.
model_dir() {
    local dir="${SMALLTIDE_MODEL_DIR:-}"
    if [[ -z "$dir" ]]; then
        dir="$(sed -n 's/^[[:space:]]*\(export[[:space:]]\+\)\{0,1\}SMALLTIDE_MODEL_DIR[[:space:]]*=[[:space:]]*//p' .env \
            | sed 's/[[:space:]]\+#.*$//; s/^["'\'']//; s/["'\'']$//' | tail -n 1)"
    fi
    dir="${dir:-models/Inkling-2-Mini}"
    if [[ "$dir" == /* ]]; then printf '%s\n' "$dir"; else printf '%s\n' "$DATA/$dir"; fi
}

setup() {
    check_devices
    log "1/3 Probing the GPU..."
    "$CLI" probe || die "GPU probe failed (see above). SmallTide needs an AMD RDNA 4 GPU and the amdgpu driver on the host."

    local dir
    dir="$(model_dir)"
    log "2/3 Downloading Inkling-2-Mini (~21 GB) into $dir ..."
    "$APP/scripts/download_inkling2.sh" --out-dir "$dir" || die "Download failed; restart the container to resume."

    if [[ -f "$dir/experts.stide" ]]; then
        log "3/3 Packed expert store already exists: $dir/experts.stide"
    else
        log "3/3 Packing routed experts into $dir/experts.stide ..."
        "$CLI" pack "$dir/model-00001-of-00001.safetensors" "$dir/experts.stide" || die "Packing experts.stide failed"
    fi
}

command="${1:-serve}"
[[ $# -gt 0 ]] && shift

case "$command" in
    serve)
        ensure_env
        if [[ ! -f "$(model_dir)/experts.stide" ]]; then
            [[ "${SMALLTIDE_AUTO_SETUP:-1}" == 1 ]] \
                || die "No model in $(model_dir) and SMALLTIDE_AUTO_SETUP=0; run the 'setup' command first."
            setup
        fi
        check_devices
        [[ $# -gt 0 ]] || set -- .env
        log "Starting the server: http://<host>:8000/v1 (expert map at /expert-map)"
        exec "$CLI" serve "$@"
        ;;
    setup)
        ensure_env
        setup
        ;;
    bench)
        exec python3 "$APP/bench/bench.py" "$@"
        ;;
    shell | bash | sh)
        exec bash "$@"
        ;;
    *)
        ensure_env
        exec "$CLI" "$command" "$@"
        ;;
esac
