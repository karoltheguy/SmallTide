# SmallTide container

A turnkey image for Linux hosts with an AMD RDNA 4 GPU (RX 9070 / RX 9060 family, Radeon AI PRO
R9700). It ships the ROCm HIP runtime (10.2 by default), so the host only needs:

- the `amdgpu` kernel driver (any recent distribution kernel has it), and
- Docker or Podman.

No ROCm install on the host, no Rust, no build.

## Run

Pick one of the three ways below. In every case, the first start creates `/data/.env`, checks the
GPU, downloads Inkling-2-Mini (about 21 GB) and packs `experts.stide`; later starts go straight to
the server. Then open `http://localhost:8000/expert-map`, or point any OpenAI client at
`http://localhost:8000/v1`:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "Explain speculative decoding in 3 bullet points."}],
  "max_tokens": 1000
}'
```

The model thinks before it answers, so keep `max_tokens` generous: a small limit can be spent on
thinking (`reasoning_content`) and leave `content` empty.

### docker run

```bash
docker run -d --name smalltide --restart unless-stopped \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined \
  -p 8000:8000 \
  -v smalltide-data:/data \
  ghcr.io/karoltheguy/smalltide:latest
docker logs -f smalltide
```

With rootless Podman, replace `--group-add video` with `--group-add keep-groups` so the container
keeps your access to the GPU devices. On SELinux hosts such as Fedora, also add
`--security-opt label=disable`.

### Docker Compose

[`compose.yaml`](compose.yaml) works with `docker compose` and `podman-compose`:

```bash
docker compose -f docker/compose.yaml up -d          # pull the published image
docker compose -f docker/compose.yaml up -d --build  # or build it from this checkout
docker compose -f docker/compose.yaml logs -f
docker compose -f docker/compose.yaml down           # stop (the data volume is kept)
```

Settings come from the shell or from `docker/.env`:

| Variable | Default | |
|---|---|---|
| `SMALLTIDE_CONFIG` | `headless` | See [Options](#options) |
| `SMALLTIDE_DATA` | `smalltide-data` volume | An absolute host directory for `.env` and the model |
| `SMALLTIDE_GROUP` | `video` | `keep-groups` for rootless Podman |
| `SMALLTIDE_PORT` | `8000` | Host port |
| `SMALLTIDE_IMAGE` | `ghcr.io/karoltheguy/smalltide:latest` | Image to run |
| `HF_TOKEN` | | Optional, for higher Hugging Face rate limits |

For example, rootless Podman with the monitor on the Radeon card:

```bash
SMALLTIDE_GROUP=keep-groups SMALLTIDE_CONFIG=monitor podman-compose -f docker/compose.yaml up -d
```

### Podman Quadlet (systemd service)

[`smalltide.container`](smalltide.container) runs SmallTide as a systemd service that starts at
boot and restarts on failure:

```bash
mkdir -p ~/.config/containers/systemd
cp docker/smalltide.container ~/.config/containers/systemd/
systemctl --user daemon-reload
systemctl --user start smalltide
journalctl --user -u smalltide -f
loginctl enable-linger "$USER"   # keep it running when you are logged out
```

For a system-wide service, copy it to `/etc/containers/systemd/` and drop `--user`. Edit the file
to change the config (`Environment=SMALLTIDE_CONFIG=...`), the port, or to mount a model you
already have. It is marked for `podman auto-update`: with `podman-auto-update.timer` enabled, new
images are pulled and the service restarted automatically.

## Options

| Variable | Default | |
|---|---|---|
| `SMALLTIDE_CONFIG` | `headless` | Template for a new `/data/.env`: `headless` (monitor on the motherboard or another GPU) or `monitor` (monitor plugged into the Radeon card) |
| `SMALLTIDE_AUTO_SETUP` | `1` | `0` stops instead of downloading the model on first start |
| `HF_TOKEN` | | Optional, for higher Hugging Face rate limits |

Pass them with `-e`, for example `-e SMALLTIDE_CONFIG=monitor`. To tune the engine, edit `.env`
in the volume and restart the container.

To keep the model on a specific disk, bind-mount a directory instead of the named volume:
`-v /path/to/smalltide:/data`. Experts are read with O_DIRECT, so on btrfs the downloader turns off
compression for the model directory.

## Other commands

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined -v smalltide-data:/data ghcr.io/karoltheguy/smalltide probe

docker exec smalltide smalltide-entrypoint run -p "Hello" -n 64
docker exec smalltide smalltide-entrypoint bench --only prose --runs 1
docker exec smalltide cat /opt/smalltide/BUILD_INFO
```

`setup` reruns the first-start steps (it resumes a partial download), `shell` opens bash, and any
other command goes to `smalltide-cli` (`probe`, `run`, `vram-probe`, `bench-gpu`, ...).

## Build

```bash
docker build -f docker/Dockerfile -t smalltide .
# other ROCm release or kernel targets:
docker build -f docker/Dockerfile --build-arg ROCM=10.1 --build-arg ARCH=gfx1201 -t smalltide .
```

The GitHub Actions workflow **Container** (manual) builds the image and pushes it to
`ghcr.io/<owner>/smalltide` (the repository owner, so `karoltheguy` for this fork) with the tags `latest`, `<version>` and `<version>-rocm<rocm>`.
