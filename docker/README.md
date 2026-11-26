# SmallTide container

A turnkey image for Linux hosts with an AMD RDNA 4 GPU (RX 9070 / RX 9060 family, Radeon AI PRO
R9700). It ships the ROCm HIP runtime (10.2 by default), so the host only needs:

- the `amdgpu` kernel driver (any recent distribution kernel has it), and
- Docker or Podman.

No ROCm install on the host, no Rust, no build.

## Run

```bash
docker run -d --name smalltide \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined \
  -p 8000:8000 \
  -v smalltide-data:/data \
  ghcr.io/<owner>/smalltide:latest
docker logs -f smalltide
```

The first start creates `/data/.env`, checks the GPU, downloads Inkling-2-Mini (about 21 GB) and
packs `experts.stide`. Later starts go straight to the server. Then open
`http://localhost:8000/expert-map`, or point any OpenAI client at `http://localhost:8000/v1`.

With rootless Podman, replace `--group-add video` with `--group-add keep-groups` so the container
keeps your access to the GPU devices. On SELinux hosts such as Fedora, also add
`--security-opt label=disable`.

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
  --security-opt seccomp=unconfined -v smalltide-data:/data ghcr.io/<owner>/smalltide probe

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
`ghcr.io/<owner>/smalltide` with the tags `latest`, `<version>` and `<version>-rocm<rocm>`.
